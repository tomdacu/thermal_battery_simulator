"""The adaptive mesh against the structured one: the same physics on a tree.

What this module pins, in the order the claims matter:

* **the protocol** - :class:`~src.core.mesh_api.MeshAPI` is satisfied by
  :class:`~src.core.adaptive_mesh.AdaptiveMesh`, and the operator the mesh assembles is
  the octree's own (:meth:`Octree.diffusion_matrix`), so the reuse claimed in the module
  docstring is a fact and not an intention;
* **equivalence** - on a uniform tree and the matching uniform ``Mesh3D`` the two agree
  to the tolerance of the linear solve on the three analytic cases: a two-layer wall
  whose interface sits on a leaf face (the harmonic mean is then exact), a slab with a
  volumetric source and an outer film on one side, and a solid slab whose upper half is
  an excluded *air* region that exchanges with the outside through ``h_out``;
* **the graded treatment** - the same cases on a refined tree, where the film has to use
  each leaf's *own* edge and the faces have hanging nodes: the wall stays exact, the film
  keeps the half-cell truncation of the structured solver and nothing else, and the
  balance closes;
* **the balance** - ``p_source + p_sink == q_fixed + p_film`` at the round-off of the
  per-leaf closure on every case, graded or not;
* **what the adaptivity buys** - a refined tree answers a hot spot at least three times
  cheaper in cells than the graded Cartesian mesh that resolves each axis as well as the
  tree does, with the error measured against the analytic solution.

The box is 1 m and the finest tree cell is 1/16 m, so every level is a power-of-two
subdivision of the box and a uniform tree has exactly the cells of a uniform ``Mesh3D``.
Where a tree is refined, the comparison is against the *analytic* solution (the graded
Cartesian mesh cannot share the tree's staircase cells) and the balance, and the
structured solver is compared on the uniform case with the same physics.
"""
from __future__ import annotations

import time

import numpy as np
import pytest

from src.analysis.fluxes import environment_flux
from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand
from src.core.mesh import Mesh3D
from src.core.mesh_api import MEMBERS, MeshAPI, missing_members
from src.core.refinement import Band, GridSpec
from src.solver.linear import LinearConfig
from src.solver.octree_solver import OctreeSteadySolver
from src.solver.steady import SolverConfig, SteadyStateSolver

BOX = 1.0
FINEST = 16                       # finest cells per side
H = BOX / FINEST                  # edge of one finest cell [m]
ALL_WALLS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
#: the per-cell fields a structured partner has to be given to carry the same physics
SHARED_FIELDS = ("k", "rho", "cp", "Q_source", "Q_sink", "material_id", "boundary_type",
                 "bc_h", "bc_T_inf", "excluded")


# ------------------------------------------------------------------- helpers
def uniform_edge(mesh: AdaptiveMesh) -> float:
    """Edge of the leaves of a uniform tree, in metres."""
    levels = {leaf.level for leaf in mesh.tree.leaves}
    assert len(levels) == 1, "a cell-by-cell partner needs a uniform tree"
    return float(mesh.physical_size * (1 << levels.pop()))


def cells_of(mesh: AdaptiveMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(i, j, k)`` of the ``Mesh3D`` cell every leaf centre falls in.

    A uniform tree and a uniform ``Mesh3D`` of the same edge have the same cells, so the
    map is one to one; the arrays are used to read the structured field back and to write
    the structured properties without going through the Fortran-order ravel (which copies).
    """
    structured = Mesh3D(BOX, BOX, BOX, spacing=uniform_edge(mesh))
    index = [structured.find_cell(*centre) for centre in mesh.centres()]
    return (np.array([i for i, _j, _k in index]), np.array([j for _i, j, _k in index]),
            np.array([k for _i, _j, k in index]))


def structured_partner(mesh: AdaptiveMesh) -> tuple[Mesh3D, tuple]:
    """A ``Mesh3D`` with the leaves' cells, their fields and the mesh's face conditions.

    The properties are *copied*, not re-painted, so the test measures the solver and not
    the geometry module: the two meshes start from the same per-cell state.
    """
    index = cells_of(mesh)
    structured = Mesh3D(BOX, BOX, BOX, spacing=uniform_edge(mesh))
    for name in SHARED_FIELDS:
        getattr(structured, name)[index] = getattr(mesh, name)
    structured.h_out = mesh.h_out
    structured.t_ambient = mesh.t_ambient
    structured.h_contact = mesh.h_contact
    structured.face_bc = dict(mesh.face_bc)
    structured.source_mask[index] = mesh.source_mask
    return structured, index


def solve_pair(mesh: AdaptiveMesh, method: str = "direct", tolerance: float = 1e-12):
    """The adaptive solution, the structured one on the same cells, and their difference."""
    adaptive = mesh.solve_steady(LinearConfig(method=method, tolerance=tolerance))
    structured, index = structured_partner(mesh)
    config = SolverConfig(method=method, tolerance=tolerance)
    reference = SteadyStateSolver(structured, config).solve()
    side = structured.Nx
    field = reference.T.reshape((side, side, side), order="F")
    difference = float(np.abs(adaptive.T - field[index]).max())
    return adaptive, reference, difference


def interface_drop(mesh: AdaptiveMesh, plane: int) -> tuple[float, float]:
    """Temperature drop [K] across the interface and the edge [m] of its leaves.

    ``plane`` is in finest-cell units.  The leaves of the column at ``(x, y) = (0, 0)``
    are picked, and the two of them must be of the same size - the case in which the
    harmonic mean of the conductivities *is* the exact series resistance of the two half
    leaves, which is the claim under test.
    """
    column = [(position, leaf) for position, leaf in enumerate(mesh.tree.leaves)
              if leaf.x == 0 and leaf.y == 0]
    below = next(leaf for _position, leaf in column if leaf.z + leaf.size == plane)
    above = next(leaf for _position, leaf in column if leaf.z == plane)
    assert below.size == above.size, "the interface must join two equal halves"
    return (float(mesh.T[mesh.tree.leaves.index(below)]
                  - mesh.T[mesh.tree.leaves.index(above)]),
            float(below.size) * mesh.physical_size)


def wall_centres(mesh: AdaptiveMesh, face: str) -> np.ndarray:
    """Z coordinates of the leaves on a box face (they must sit at one plane)."""
    positions = mesh.wall_indices(face)
    z = mesh.centres()[positions, 2]
    assert float(z.max() - z.min()) < 1e-12, (
        "the trees of these tests keep the wall leaves of one size, so a 1-D reference is "
        "the exact solution of the cell-pinned problem")
    return z


def two_layer_reference(mesh: AdaptiveMesh, interface: float, k_low: float, k_high: float,
                        t_low: float, t_high: float) -> tuple[np.ndarray, float]:
    """The exact solution of the cell-pinned two-layer wall, sampled at the leaf centres.

    The Dirichlet leaves are pinned at the *wall* temperature and sit half a leaf away
    from it, so the reference is the piecewise linear profile through the pinned centres
    with the kink on the interface: the flux is the series resistance of the two layers
    between those two planes.
    """
    z = mesh.centres()[:, 2]
    pin_low = float(wall_centres(mesh, "z_min")[0])
    pin_high = float(wall_centres(mesh, "z_max")[0])
    resistance = (interface - pin_low) / k_low + (pin_high - interface) / k_high
    flux = (t_low - t_high) / resistance
    below = t_low - flux * (z - pin_low) / k_low
    above = t_low - flux * ((interface - pin_low) / k_low + (z - interface) / k_high)
    return np.where(z <= interface, below, above), flux


# ------------------------------------------------------------------ the protocol
def test_the_adaptive_mesh_satisfies_the_protocol_it_defines():
    """The interface of :mod:`src.core.mesh_api` is what the ported consumers may use."""
    mesh = AdaptiveMesh.uniform(FINEST, H, 2)
    assert isinstance(mesh, MeshAPI)
    assert missing_members(mesh) == []
    # the data member list and the protocol itself must not drift apart: a member added
    # to one has to be added to the other, or `missing_members` would under-report
    declared = set(getattr(MeshAPI, "__annotations__", {})) | {
        name for name in vars(MeshAPI) if not name.startswith("_")}
    assert set(MEMBERS) == declared
    assert mesh.n_cells == mesh.tree.n_cells == 64
    # the flat arrays the protocol promises, one value per cell, and the face list that
    # indexes them
    for name in MEMBERS:
        if name in ("n_cells", "faces", "h_out", "t_ambient", "h_contact", "face_bc"):
            continue
        assert getattr(mesh, name).shape == (mesh.n_cells,), name
    assert isinstance(mesh.h_out, float) and isinstance(mesh.t_ambient, float)
    assert set(mesh.face_bc) == set(ALL_WALLS)
    faces = mesh.faces()
    assert len(faces) == 144 and len(faces[0]) == 5
    assert max(max(row[0], row[1]) for row in faces) < mesh.n_cells


def test_the_conduction_operator_is_the_octree_operator():
    """``matrix`` reuses the octree assembly, and the flux report is that operator's.

    With no contact resistance and no excluded leaf the correction is empty, so the
    operator is *bit for bit* ``Octree.diffusion_matrix`` and the heat rates are the ones
    :class:`OctreeSteadySolver` reports for the same field - which is what makes the
    adaptive flux report trustworthy rather than a second implementation of the same sum.
    """
    mesh = AdaptiveMesh.uniform(FINEST, H, 2)
    mesh.k[:] = np.linspace(1.0, 2.0, mesh.n_cells)
    without_contact = mesh.matrix()
    delta = without_contact - mesh.tree.diffusion_matrix(mesh.k, H)
    assert float(np.abs(delta.data).max(initial=0.0)) == 0.0

    temperature = np.linspace(400.0, 300.0, mesh.n_cells)
    solver = OctreeSteadySolver(mesh.tree, conductivity=mesh.k, physical_size=H)
    assert np.allclose(mesh.face_fluxes(temperature), solver.face_fluxes(temperature,
                                                                        mesh.tree))
    assert np.allclose(mesh.divergence(temperature), solver.divergence(temperature,
                                                                      mesh.tree))
    # one conductance per face: the divergence of the flux is the assembled operator
    assert np.allclose(without_contact @ temperature * mesh.V, mesh.divergence(temperature))

    # a contact resistance changes only the interfaces between two materials
    mesh.material_id[:] = np.where(mesh.centres()[:, 2] < 0.5, 1, 2)
    mesh.h_contact = 50.0
    with_contact = mesh.matrix()
    assert (with_contact != without_contact).nnz > 0
    entry = (with_contact - without_contact).tocoo()
    pairs = {(min(int(i), int(j)), max(int(i), int(j)))
             for i, j in zip(entry.row, entry.col, strict=True) if i != j}
    assert pairs
    for left, right in pairs:
        assert mesh.material_id[left] != mesh.material_id[right]


# ------------------------------------------------------------- the analytic cases
def test_a_two_layer_wall_matches_the_structured_solver():
    """Interface on a leaf face: the harmonic mean is the exact series resistance.

    1 m wall, ``k = 1`` below ``z = 0.5`` and ``k = 4`` above, the ends held at 400 K and
    300 K, the sides adiabatic.  The discrete solution is the exact piecewise linear
    profile through the pinned centres - on the uniform tree (compared cell by cell with
    ``Mesh3D``) and on a tree refined around the interface, where the two leaves that
    share it are of equal size and the harmonic mean is therefore exact.
    """
    k_low, k_high, t_low, t_high = 1.0, 4.0, 400.0, 300.0

    uniform = AdaptiveMesh.uniform(FINEST, H, 2)
    uniform.k[:] = np.where(uniform.centres()[:, 2] < 0.5, k_low, k_high)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        uniform.set_adiabatic(face)
    uniform.set_fixed_temperature_bc("z_min", t_low)
    uniform.set_fixed_temperature_bc("z_max", t_high)
    adaptive, _reference, difference = solve_pair(uniform)
    reference, flux = two_layer_reference(uniform, 0.5, k_low, k_high, t_low, t_high)
    assert difference < 1e-9, "the adaptive mesh and Mesh3D must assemble the same problem"
    assert float(np.abs(adaptive.T - reference).max()) < 1e-9
    assert adaptive.balance.closure < 1e-12
    # the harmonic mean is the exact series resistance of the two half leaves that share
    # the interface: the drop across the pair is flux * (d/2) * (1/k_low + 1/k_high)
    drop, edge = interface_drop(uniform, 8)
    assert drop == pytest.approx(flux * 0.5 * edge * (1.0 / k_low + 1.0 / k_high),
                                 rel=1e-9)
    # ... and the interface resistance is what the wall pays for: the drop across the
    # pair is a real part of the 100 K the two ends hold
    assert 0.05 < drop < 0.5 * (t_low - t_high)

    graded = AdaptiveMesh.from_bands(
        FINEST, H, [RefinementBand((0.0, 0.0, 0.375), (1.0, 1.0, 0.625), 2 * H)],
        base_level=3)
    assert len(graded.level_histogram()) >= 2, "the test needs hanging nodes"
    # the interface plane (z = 0.5 = 8 finest cells) must be shared by two leaves of
    # equal size, which is when the harmonic mean is the exact series resistance
    sizes = {leaf.size for leaf in graded.tree.leaves if leaf.z in (7, 8)}
    assert sizes == {2}
    graded.k[:] = np.where(graded.centres()[:, 2] < 0.5, k_low, k_high)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        graded.set_adiabatic(face)
    graded.set_fixed_temperature_bc("z_min", t_low)
    graded.set_fixed_temperature_bc("z_max", t_high)
    result = graded.solve_steady()
    reference, graded_flux = two_layer_reference(graded, 0.5, k_low, k_high, t_low, t_high)
    assert float(np.abs(result.T - reference).max()) < 1e-9, "hanging nodes must not touch it"
    assert graded_flux == pytest.approx(flux)
    # a finer interface pair is a smaller resistance, and the harmonic mean follows it
    graded_drop, graded_edge = interface_drop(graded, 8)
    assert graded_edge == pytest.approx(2 * H)
    assert graded_drop == pytest.approx(
        graded_flux * 0.5 * graded_edge * (1.0 / k_low + 1.0 / k_high), rel=1e-9)
    assert result.balance.closure < 1e-10


def test_a_slab_with_a_source_and_a_film_matches_the_structured_solver():
    """A uniform source with an outer film: the two meshes agree, and so does the physics.

    The film sits on the only non-adiabatic face, so the whole power leaves through it.
    The half-cell film ``h_eff = 2kh/(2k + hd)`` is exact for a *constant* flux across the
    half cell, which the source breaks: the discrete solution is the analytic quadratic
    shifted by exactly ``q d^2 / (8 k)`` with ``d`` the edge of the leaves on the film.
    That constant is the structured solver's own truncation, and the adaptive mesh has to
    reproduce it - which is what makes the two solutions equal.
    """
    conductivity, density, film, t_inf = 1.0, 1000.0, 10.0, 293.15

    slab = AdaptiveMesh.uniform(FINEST, H, 2)
    slab.k[:] = conductivity
    slab.Q_source[:] = density
    for face in ALL_WALLS[:-1]:
        slab.set_adiabatic(face)
    slab.set_convection_bc("z_max", film, t_inf)
    adaptive, _reference, difference = solve_pair(slab)
    assert difference < 1e-9

    z = slab.centres()[:, 2]
    surface = t_inf + density * BOX / film
    analytic = surface + density * (BOX ** 2 - z ** 2) / (2 * conductivity)
    error = adaptive.T - analytic
    edge = float(slab.sizes.max())
    assert float(np.abs(error).max()) == pytest.approx(density * edge ** 2
                                                       / (8 * conductivity), rel=1e-9)
    assert float(error.max() - error.min()) < 1e-9, "the offset is the only difference"
    assert adaptive.balance.p_source == pytest.approx(density * BOX ** 2 * BOX)
    assert adaptive.balance.p_film == pytest.approx(adaptive.balance.p_source)
    assert adaptive.balance.closure < 1e-12


def test_the_graded_film_uses_the_local_leaf_size():
    """Refining towards the film keeps the structured film law, leaf by leaf.

    Same slab with the top quarter refined by a factor of four: the leaves on the film are
    then four times smaller, so the shift of the discrete solution drops by sixteen while
    the flux stays equal to the source power.  The energy balance closes to the round-off
    of the per-leaf closure on the graded tree.
    """
    conductivity, density, film, t_inf = 1.0, 1000.0, 10.0, 293.15

    graded = AdaptiveMesh.from_bands(
        FINEST, H, [RefinementBand((0.0, 0.0, 0.75), (1.0, 1.0, 1.0), H)],
        base_level=2)
    assert len(graded.level_histogram()) > 1
    graded.k[:] = conductivity
    graded.Q_source[:] = density
    for face in ALL_WALLS[:-1]:
        graded.set_adiabatic(face)
    graded.set_convection_bc("z_max", film, t_inf)
    result = graded.solve_steady()

    z = graded.centres()[:, 2]
    surface = t_inf + density * BOX / film
    analytic = surface + density * (BOX ** 2 - z ** 2) / (2 * conductivity)
    top = graded.wall_indices("z_max")
    top_edge = float(graded.sizes[top].max())
    assert top_edge == pytest.approx(H), "the film sits on the finest leaves of the band"
    expected = density * top_edge ** 2 / (8 * conductivity)
    # the film's own truncation, at the film's own leaves: had the film used the coarse
    # edge of the rest of the slab, the shift would be sixteen times larger
    assert float(np.abs((result.T - analytic)[top]).max()) == pytest.approx(expected,
                                                                           rel=1e-9)
    assert result.balance.p_film == pytest.approx(result.balance.p_source, rel=1e-9)
    assert result.balance.closure < 1e-10
    assert result.balance.residual < 1e-8


def test_excluded_leaves_and_the_environment_film_match_the_structured_solver():
    """The air is dropped, not meshed: its interface carries the film and its leaves the ambient.

    The upper half of the box is an excluded region held at ``t_ambient`` that conducts
    nothing; every active leaf facing it loses ``h_out A (T - t_ambient)``.  The adaptive
    mesh must build the same film as ``src/solver/matrix.py`` builds for a structured
    mesh - which is also what ``fluxes.environment_flux`` reports back.
    """
    density, film, t_ambient, t_ground = 5000.0, 10.0, 293.15, 400.0

    mesh = AdaptiveMesh.uniform(FINEST, H, 2)
    mesh.excluded = mesh.centres()[:, 2] > 0.5 + 1e-9
    assert 0 < int(mesh.excluded.sum()) < mesh.n_cells
    mesh.k[:] = 1.0
    mesh.Q_source[:] = np.where(mesh.excluded, 0.0, density)
    mesh.h_out = film
    mesh.t_ambient = t_ambient
    mesh.set_fixed_temperature_bc("z_min", t_ground)
    for face in ("x_max", "y_min", "y_max", "z_max"):
        mesh.set_adiabatic(face)
    adaptive, reference, difference = solve_pair(mesh, tolerance=1e-12)

    assert difference < 1e-9
    assert reference.converged
    assert np.allclose(adaptive.T[mesh.excluded], t_ambient)
    # the film read back from the structured mesh, on the same field: the two must report
    # the same exchange with the environment, not merely close temperatures
    structured, index = structured_partner(mesh)
    structured.T[index] = adaptive.T
    assert adaptive.balance.q_environment == pytest.approx(environment_flux(structured),
                                                           rel=1e-9)
    # nothing conducts into the excluded leaves: the film is the only loss on that side
    assert adaptive.balance.p_film == pytest.approx(adaptive.balance.q_environment)
    assert adaptive.balance.p_source == pytest.approx(
        adaptive.balance.q_fixed + adaptive.balance.p_film, rel=1e-9)
    assert adaptive.balance.closure < 1e-12


def test_the_contact_resistance_matches_the_structured_solver():
    """A finite interface conductance is a series resistance, and only between materials."""
    mesh = AdaptiveMesh.uniform(FINEST, H, 2)
    mesh.material_id[:] = np.where(mesh.centres()[:, 2] < 0.5, 1, 2)
    mesh.k[:] = 1.0
    mesh.h_contact = 40.0
    mesh.set_fixed_temperature_bc("z_min", 400.0)
    mesh.set_fixed_temperature_bc("z_max", 300.0)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    adaptive, _reference, difference = solve_pair(mesh)

    assert difference < 1e-9
    # the contact resistance really is in the solution: it doubles the resistance of the
    # wall and the flux of the perfect interface is far from it
    flux = float(np.abs(adaptive.face_flux).max())
    perfect = (400.0 - 300.0) / (1.0 / 1.0)
    assert flux < 0.6 * perfect
    assert adaptive.balance.closure < 1e-12


def test_the_internal_film_matches_the_structured_solver():
    """Tube-side convection: ``bc_h / V^(1/3)`` on the cells that are not on a box face."""
    mesh = AdaptiveMesh.uniform(FINEST, H, 2)
    mesh.k[:] = 1.0
    tube = np.zeros(mesh.n_cells, dtype=bool)
    inside = np.all((mesh.centres() > 0.2) & (mesh.centres() < 0.8), axis=1)
    tube[inside] = True
    assert 0 < int(tube.sum()) < mesh.n_cells
    assert not mesh.on_box_face[tube].any(), "a tube cell on a box face carries a face BC"
    mesh.set_internal_convection(tube, 300.0, 320.0)
    mesh.Q_source[:] = 2000.0
    mesh.set_fixed_temperature_bc("z_min", 400.0)
    for face in ("x_max", "y_min", "y_max", "z_max"):
        mesh.set_adiabatic(face)
    adaptive, _reference, difference = solve_pair(mesh)

    assert difference < 1e-9
    assert adaptive.balance.q_tubes > 0.0
    assert adaptive.balance.p_source == pytest.approx(
        adaptive.balance.q_fixed + adaptive.balance.p_film, rel=1e-9)
    assert adaptive.balance.closure < 1e-12


# ---------------------------------------------------------------- the refinement
def test_refinement_carries_every_per_leaf_field():
    """A round of refinement keeps the physics: a split leaf hands its state to its children.

    Cost, boundary types and the environment survive; the leaf that was a tube stays a
    tube and the leaf that was excluded stays excluded - which is what a refinement round
    driven by the solution needs, since the alternative is repainting the model after
    every round.
    """
    mesh = AdaptiveMesh.uniform(FINEST, H, 3)
    mesh.T[:] = np.linspace(300.0, 400.0, mesh.n_cells)
    mesh.k[:] = 2.0
    mesh.excluded[:] = False
    mesh.excluded[0] = True
    mesh.Q_source[:] = 100.0
    before = mesh.T.copy()
    excluded_leaf = mesh.tree.leaves[0]
    mesh.refine(np.ones(mesh.n_cells), threshold=0.5, levels=1)

    assert mesh.n_cells == 8 * 8, "the eight leaves of the box split once"
    assert mesh.T.shape == (mesh.n_cells,)
    assert np.all(mesh.k == 2.0) and np.all(mesh.Q_source == 100.0)
    children = [position for position, leaf in enumerate(mesh.tree.leaves)
                if leaf.level == excluded_leaf.level - 1 and leaf.x < excluded_leaf.size
                and leaf.y < excluded_leaf.size and leaf.z < excluded_leaf.size]
    assert len(children) == 8 and all(mesh.excluded[position] for position in children)
    assert int(mesh.excluded.sum()) == 8
    assert np.allclose(mesh.T[children], before[0])
    assert mesh.sizes.shape == (mesh.n_cells,) and mesh.V.shape == (mesh.n_cells,)
    with pytest.raises(ValueError, match="one value per leaf"):
        mesh.refine(np.ones(3), threshold=0.5)


def test_the_adapted_tree_beats_a_graded_mesh_on_cells():
    """The hot spot: a refined tree spends cells where the error is, a graded mesh on slabs.

    The analytic solution is a Gaussian hot spot of amplitude 30 K in a 1 m box whose
    walls are held at 300 K, driven by the source ``-k grad^2 T`` that makes it exact.  The
    tree starts at 8^3 leaves and is refined three times on its own flux-jump estimator;
    the graded ``Mesh3D`` resolves each axis as finely as the tree does *anywhere* (its
    bands are the tree's own resolution profile), which is the fair tensor-product
    partner.  Measured: the tree answers with 5.9x fewer cells and an error 1.1x larger
    (2360 leaves / 1.31 K against 13 824 cells / 1.19 K), so the bounds below are the claim
    (at least three times fewer cells, equal accuracy) and not the measurement.
    """
    amplitude, sigma, conductivity, rounds = 30.0, 0.08, 1.0, 3
    n_finest = 64
    edge = BOX / n_finest

    def exact(points: np.ndarray) -> np.ndarray:
        r2 = np.sum((points - 0.5 * BOX) ** 2, axis=1)
        return 300.0 + amplitude * np.exp(-r2 / sigma ** 2)

    def source(points: np.ndarray) -> np.ndarray:
        r2 = np.sum((points - 0.5 * BOX) ** 2, axis=1)
        return -conductivity * (4.0 * r2 / sigma ** 4 - 6.0 / sigma ** 2) * \
            amplitude * np.exp(-r2 / sigma ** 2)

    tree = AdaptiveMesh.uniform(n_finest, edge, 3)
    tree.k[:] = conductivity
    for face in ALL_WALLS:
        tree.set_fixed_temperature_bc(face, 300.0)
    started = time.perf_counter()
    for round_number in range(rounds):
        points = tree.centres()
        density = source(points)
        tree.Q_source[:] = np.maximum(density, 0.0)
        tree.Q_sink[:] = np.minimum(density, 0.0)
        result = tree.solve_steady(LinearConfig(method="cg", tolerance=1e-13))
        tree_error = float(np.abs(result.T - exact(points)).max())
        assert result.balance.closure < 1e-10
        if round_number < rounds - 1:
            jump = np.abs(tree.indicator(result.T))
            tree.refine(jump, threshold=0.02 * float(jump.max()))
            assert len(tree.level_histogram()) > 1, "the tree must really be adapted"
    adapted_seconds = time.perf_counter() - started
    assert result.T.shape == (tree.n_cells,)
    assert tree.n_cells > 512, "the refinement has to buy something over the initial tree"

    bands = tuple(axis_bands(tree, axis) for axis in range(3))
    structured = Mesh3D(BOX, BOX, BOX,
                        grid=GridSpec(x=bands[0], y=bands[1], z=bands[2], growth=1.3))
    points = np.column_stack([structured.X.ravel(order="F"), structured.Y.ravel(order="F"),
                              structured.Z.ravel(order="F")])
    density = source(points)
    structured.k[:] = conductivity
    structured.Q_source = np.maximum(density, 0.0).reshape(structured.T.shape)
    structured.Q_sink = np.minimum(density, 0.0).reshape(structured.T.shape)
    for face in ALL_WALLS:
        structured.set_fixed_temperature_bc(face, 300.0)
    started = time.perf_counter()
    graded = SteadyStateSolver(structured, SolverConfig(method="cg",
                                                        tolerance=1e-11)).solve()
    graded_seconds = time.perf_counter() - started
    assert graded.converged
    graded_error = float(np.abs(graded.T.ravel(order="F") - exact(points)).max())

    assert structured.N_total >= 3 * tree.n_cells, (
        f"the graded mesh must cost at least three times the cells: {tree.n_cells} leaves "
        f"against {structured.N_total} cells")
    assert tree_error <= 1.25 * graded_error, (
        f"the tree must stay as accurate: {tree_error:.4f} K against {graded_error:.4f} K")
    assert adapted_seconds < 60.0 and graded_seconds < 60.0


# -------------------------------------------------------------------- refusals
def test_the_mesh_rejects_what_the_structured_one_rejects():
    """The mistakes this API makes easy: a bad box, a bad face, a bad state, a bad rule."""
    with pytest.raises(ValueError, match="physical_size"):
        AdaptiveMesh.uniform(FINEST, 0.0, 2)

    mesh = AdaptiveMesh.uniform(FINEST, H, 2)
    with pytest.raises(ValueError, match="invalid face"):
        mesh.set_adiabatic("top")
    with pytest.raises(ValueError, match="Kelvin"):
        mesh.set_fixed_temperature_bc("z_min", 20.0)
    with pytest.raises(ValueError, match="Kelvin"):
        mesh.set_convection_bc("z_max", 5.0, 20.0)
    with pytest.raises(ValueError, match="one entry per leaf"):
        mesh.set_internal_convection(np.zeros(3, dtype=bool), 10.0, 300.0)
    with pytest.raises(ValueError, match="band size"):
        AdaptiveMesh.from_bands(FINEST, H, [RefinementBand((0, 0, 0), (1, 1, 1), 0.0)])
    with pytest.raises(ValueError, match="empty band"):
        AdaptiveMesh.from_bands(FINEST, H, [RefinementBand((0, 0, 0), (1, 1, 0), 0.1)])
    with pytest.raises(ValueError, match="at least one refinement band"):
        AdaptiveMesh.from_bands(FINEST, H, [])


# ------------------------------------------------------------------- utilities
def axis_bands(mesh: AdaptiveMesh, axis: int):
    """The tree's resolution profile along one axis, as ``GridSpec`` bands.

    Every finest-cell slab of the axis is given the edge of the finest leaf that reaches
    it, and consecutive slabs of equal resolution are merged into one band: the tensor
    product of the three profiles resolves the axis at least as finely as the tree does,
    everywhere, which is what makes the cell count of the graded mesh a fair comparison.
    """
    n = mesh.tree.n
    profile = np.full(n, n)
    for leaf in mesh.tree.leaves:
        low = (leaf.x, leaf.y, leaf.z)[axis]
        profile[low:low + leaf.size] = np.minimum(profile[low:low + leaf.size], leaf.size)
    assert profile.max() < n, "the profile must be covered by the leaves"
    bands, start = [], 0
    for slab in range(1, n + 1):
        if slab == n or profile[slab] != profile[start]:
            bands.append(Band(start * mesh.physical_size, slab * mesh.physical_size,
                              float(profile[start]) * mesh.physical_size))
            start = slab
    return tuple(bands)
