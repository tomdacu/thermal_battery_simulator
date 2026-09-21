"""The steady solver on both meshes: one physics, two assemblies.

What this module pins, in the order the claims matter:

* **the same face coefficient** - the conductance an octree face carries and the one the
  structured neighbour tables carry are the *same number* on the same interface, because
  :func:`src.solver.matrix.face_conductance` is one rule with two adapters
  (:func:`~src.solver.matrix.face_coefficients` and
  :meth:`~src.core.adaptive_mesh.AdaptiveMesh._face_table`).  The operators the two
  assemblies build then agree on the free rows to the round-off of the arithmetic;
* **the battery model on both meshes** - ``create_small_test_geometry()`` painted on a
  uniform tree and on the matching ``Mesh3D`` (the ``paint_pair`` fixture), solved by the
  same :class:`~src.solver.steady.SteadyStateSolver`: the same field leaf by leaf and the
  same balance, with and without radiation.  The radiative share of the outer film is the
  point of the second half: it rides on the environment film of both roads, which
  re-evaluates it at the surface temperature the film itself drives;
* **a refined tree** - hanging nodes against the graded grid that refines the same way:
  the exact profile of a sourceless two-layer wall (the scheme reproduces it on any mesh,
  which is why the kink and the flux are comparable at all), and the environment film on a
  refined interface, where the two meshes do share their cells.

The box is 8 m with 0.5 m cells for the battery (16 leaves a side, so a uniform tree and a
uniform ``Mesh3D`` have exactly the same cells), and 1 m with a 1/16 m finest edge for the
refined cases.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.balance import compute_balance
from src.analysis.fluxes import environment_flux
from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand
from src.core.grid import GridIndex
from src.core.mesh import Mesh3D
from src.core.refinement import Band, GridSpec
from src.solver.matrix import build_steady_matrix, face_coefficients
from src.solver.steady import SolverConfig, SteadyStateSolver

BOX = 1.0
FINEST = 16                       # finest cells per side
H = BOX / FINEST                  # edge of one finest cell [m]
KINK = 0.25                       # where the two conductivities meet [m]
#: the film-side cell of the refined cases: the whole active slab is at the finest edge
FILM = 0.5


# ------------------------------------------------------------------- helpers
def cell_of(structured: Mesh3D, tree: AdaptiveMesh) -> np.ndarray:
    """Flat index of the ``Mesh3D`` cell every leaf centre falls in."""
    return np.array([structured.ijk_to_linear(*structured.find_cell(*tree.centre(position)))
                     for position in range(tree.n_cells)])


def wall_column(tree: AdaptiveMesh) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                            np.ndarray]:
    """``(positions, z, size, k)`` of the leaves along the ``x = y = 0`` column.

    The two refined cases are one-dimensional, so one column carries the whole solution;
    the leaves of the column are read in ``tree.leaves`` order, which is the order the
    field is in, bottom to top.
    """
    column = sorted(((position, leaf) for position, leaf in enumerate(tree.tree.leaves)
                     if leaf.x == 0 and leaf.y == 0), key=lambda item: item[1].z)
    positions = np.array([position for position, _leaf in column])
    z = np.array([tree.centre(position)[2] for position in positions])
    sizes = np.array([leaf.size * tree.physical_size for _position, leaf in column])
    return positions, z, sizes, tree.k[positions]


def grid_column(grid: Mesh3D) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(z, size, k)`` of the cells along the ``x = y = 0`` column, bottom to top."""
    return grid.z, grid.dz, grid.k[0, 0, :]


def exact_wall(z: np.ndarray, sizes: np.ndarray, conductivity: np.ndarray,
               t_low: float = 400.0, t_high: float = 300.0) -> tuple[np.ndarray, float]:
    """The exact profile of a sourceless two-layer wall, and the flux it carries.

    An independent reference, written from the series chain of the half cells between
    consecutive centres - the same cross-check ``tests/test_solver.py`` makes for the
    structured matrix.  The scheme reproduces this profile on *any* mesh (the second
    difference of a piecewise linear field is zero and the face conductance is the exact
    series resistance), which is why a refined tree and a graded grid can be compared on
    it at all.
    """
    resistance = np.zeros(sizes.size)
    resistance[1:] = np.cumsum(0.5 * sizes[:-1] / conductivity[:-1]
                               + 0.5 * sizes[1:] / conductivity[1:])
    flux = (t_low - t_high) / resistance[-1]
    return t_low - flux * resistance, float(flux)


def kink_drop(z: np.ndarray, field: np.ndarray) -> float:
    """The temperature drop [K] across the kink: the two centres that straddle it."""
    below = np.flatnonzero(z < KINK)[-1]
    above = below + 1
    assert z[below] < KINK < z[above], "the kink must sit between two centres"
    return float(field[below] - field[above])


def wall_tree() -> AdaptiveMesh:
    """A two-layer wall refined in its lower half: 1/16 m leaves below, coarser above.

    The kink of the two conductivities sits at ``z = 0.25``, where the leaves above and
    below it are *equal*: the face conductance is then the exact series resistance of the
    two half leaves.  The leaves across ``z = 0.5`` are *not* equal, so that is where the
    hanging nodes are.
    """
    tree = AdaptiveMesh.from_bands(
        FINEST, H, [RefinementBand((0.0, 0.0, 0.0), (BOX, BOX, FILM), H)], base_level=2)
    tree.k[:] = np.where(tree.centres()[:, 2] < KINK, 1.0, 0.5)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        tree.set_adiabatic(face)
    tree.set_fixed_temperature_bc("z_min", 400.0)
    tree.set_fixed_temperature_bc("z_max", 300.0)
    return tree


def wall_grid() -> Mesh3D:
    """The graded grid of the same request: 1/16 m to ``z = 0.5``, a ramp above."""
    grid = GridSpec(x=(Band(0.0, BOX, KINK),), y=(Band(0.0, BOX, KINK),),
                    z=(Band(0.0, FILM, H), Band(0.0, BOX, KINK)))
    mesh = Mesh3D(BOX, BOX, BOX, grid=grid)
    mesh.k[:] = np.where(mesh.Z < KINK, 1.0, 0.5)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    mesh.set_fixed_temperature_bc("z_min", 400.0)
    mesh.set_fixed_temperature_bc("z_max", 300.0)
    return mesh


# --------------------------------------------------- the shared face coefficient
def test_the_face_coefficient_is_the_same_number_on_both_roads(paint_pair):
    """One interface, two assemblies: the same conductance, so the same operator.

    The comparison is made on the faces between two *active* leaves: the faces into an
    excluded one carry no conductance at all, and the structured adapter zeroes only the
    destination side of them, because the row of an excluded node is pinned by the
    Dirichlet elimination.
    """
    tree, structured = paint_pair()
    index = cell_of(structured, tree)
    volumes = structured.V.ravel(order="F")
    coefficients = face_coefficients(structured, GridIndex.from_mesh(structured))

    faces = np.asarray(tree.faces(), dtype=float).reshape(-1, 5).astype(int)
    first, second, axis = faces[:, 0], faces[:, 1], faces[:, 2]
    centres = tree.centres()
    # the structured coefficient of the direction from the first leaf of the pair to the
    # second one: +axis when the first sits below, -axis when it sits above
    direction = 2 * axis + (centres[first, axis] < centres[second, axis])
    structured_g = coefficients[direction, index[first]] * volumes[index[first]]

    shared = ~tree.excluded[first] & ~tree.excluded[second]
    assert int(shared.sum()) > 1000, "the painted model must have a shared interface"
    difference = np.abs(structured_g[shared] - tree.face_conductances()[shared])
    assert difference.max() == 0.0, "the two roads must assemble the same conductance"

    # and the operators themselves, on the rows that survive the Dirichlet elimination:
    # the tree's field is in leaf order, the grid's in Fortran order, so the rows are
    # picked in each mesh's own order
    tree_matrix, _ = tree.assemble()
    structured_matrix, _ = build_steady_matrix(structured)
    leaves = np.flatnonzero(~tree.excluded)
    tree_free = tree_matrix.tocsr()[leaves][:, leaves]
    structured_free = structured_matrix.tocsr()[index[leaves]][:, index[leaves]]
    difference = (tree_free - structured_free).tocsr()
    assert np.abs(difference.data).max() < 1e-12 * np.abs(tree_free.data).max()


# ------------------------------------------------------- the battery model, both roads
def test_the_battery_model_matches_on_both_meshes(paint_pair):
    """The same geometry painted twice, solved twice: the same field and the same balance."""
    tree, structured = paint_pair()
    config = SolverConfig(method="direct")
    adaptive = SteadyStateSolver(tree, config).solve()
    reference = SteadyStateSolver(structured, config).solve()

    assert adaptive.converged and reference.converged
    assert adaptive.iterations == reference.iterations == 1     # no radiation: one solve
    index = cell_of(structured, tree)
    field = np.asarray(reference.T, dtype=float).ravel(order="F")
    # every leaf, pinned ones included: the two assemblies hold the excluded leaves at
    # the same placeholder temperature (the ambient wins over the ground face on both)
    assert np.abs(adaptive.T - field[index]).max() < 1e-9

    balance = tree.balance()
    structured_balance = compute_balance(structured)
    assert balance.q_environment == pytest.approx(environment_flux(structured), rel=1e-9)
    assert balance.p_source == pytest.approx(structured_balance.p_input, rel=1e-9)
    assert balance.q_fixed == pytest.approx(structured_balance.q_domain
                                            - balance.q_environment, abs=1e-6)
    assert balance.closure < 1e-12
    assert abs(structured_balance.imbalance) < 1e-6 * structured_balance.p_input
    assert adaptive.stats["T_max"] == pytest.approx(reference.stats["T_max"], rel=1e-9)


def test_the_radiant_battery_model_matches_on_both_meshes(paint_pair):
    """With radiation on, the outer film carries the radiative share on both roads.

    The film depends on the surface temperature it drives, so the assembly re-evaluates it
    at the field it assembles for: the two meshes start from the same painted state, run
    the same Picard sweeps and must end on the *same* film, field and loss - the loss
    being the larger one, since the shell now radiates as well as convects.
    """
    plain_tree, _plain_structured = paint_pair()
    SteadyStateSolver(plain_tree, SolverConfig(method="direct")).solve()
    plain_loss = plain_tree.balance().q_environment

    tree, structured = paint_pair()
    config = SolverConfig(method="direct", radiation=True, max_picard=8,
                          picard_tolerance=1e-9)
    adaptive = SteadyStateSolver(tree, config).solve()
    reference = SteadyStateSolver(structured, config).solve()

    assert adaptive.converged and reference.converged
    assert adaptive.iterations == reference.iterations == config.max_picard
    index = cell_of(structured, tree)
    field = np.asarray(reference.T, dtype=float).ravel(order="F")
    assert np.abs(adaptive.T - field[index]).max() < 1e-9

    # the same film, written back by both assemblies, and the same exchange through it
    assert tree.h_out == pytest.approx(structured.h_out, rel=1e-12)
    assert tree.balance().q_environment == pytest.approx(environment_flux(structured),
                                                         rel=1e-9)
    assert tree.balance().q_environment > plain_loss
    assert tree.balance().closure < 1e-12
    structured_balance = compute_balance(structured, radiation=True)
    assert abs(structured_balance.imbalance) < 1e-6 * structured_balance.p_input


# ------------------------------------------------------------------ a refined tree
def test_a_refined_tree_matches_the_equivalent_graded_grid():
    """Hanging nodes against the graded grid: the same wall, the same law, the same flux.

    A tree refines *cubes*, so its staircase is not the tensor product of three 1-D
    refinements and no graded grid shares its cells: the comparison is made on what the
    discretisation must not touch - the exact profile of a sourceless two-layer wall
    (reproduced on any mesh, hanging nodes included), the series resistance of the kink,
    and the flux the two grids carry, which differ by the resolution of the coarse part
    alone.
    """
    tree, grid = wall_tree(), wall_grid()
    assert len(tree.level_histogram()) > 1, "the tree must be refined"
    assert not grid.uniform

    adaptive = SteadyStateSolver(tree, SolverConfig(method="direct")).solve()
    reference = SteadyStateSolver(grid, SolverConfig(method="direct")).solve()
    assert adaptive.converged and reference.converged

    tree_positions, tree_z, tree_sizes, tree_k = wall_column(tree)
    grid_z, grid_sizes, grid_k = grid_column(grid)
    tree_exact, tree_flux = exact_wall(tree_z, tree_sizes, tree_k)
    grid_exact, grid_flux = exact_wall(grid_z, grid_sizes, grid_k)
    tree_field = adaptive.T[tree_positions]
    grid_field = np.asarray(reference.T, dtype=float).reshape(
        (grid.Nx, grid.Ny, grid.Nz), order="F")[0, 0, :]
    assert np.abs(tree_field - tree_exact).max() < 1e-9
    assert np.abs(grid_field - grid_exact).max() < 1e-9

    # the kink carries the same series resistance on both meshes: the leaves it joins are
    # the finest ones on either side, so the law is one number, not two
    tree_drop = kink_drop(tree_z, tree_field)
    grid_drop = kink_drop(grid_z, grid_field)
    kink_resistance = 0.5 * H / 1.0 + 0.5 * H / 0.5
    assert tree_drop == pytest.approx(tree_flux * kink_resistance, rel=1e-9)
    assert grid_drop == pytest.approx(grid_flux * kink_resistance, rel=1e-9)

    # the two discretisations carry the same flux up to the resolution of the coarse
    # half: both walls are the same wall, meshed two ways
    assert tree_flux == pytest.approx(grid_flux, rel=0.1)
    # a sourceless wall accumulates nothing: what enters at one end leaves at the other
    assert tree.balance().closure < 1e-12
    assert abs(tree.balance().q_fixed) < 1e-9 * tree_flux
    assert abs(compute_balance(grid).imbalance) < 1e-9


def test_the_environment_film_on_a_refined_tree_matches_the_graded_grid():
    """The film on a refined interface: the same surface, the same coefficient, one road.

    The active slab is at the finest edge on both meshes, so the cells the film acts on
    are the same and the comparison is leaf by leaf; the air above it is coarser on the
    tree (its staircase), which is what makes the interface a hanging-node boundary and
    the film the only thing crossing it.
    """
    density, film, t_ambient, t_ground, emissivity = 5000.0, 10.0, 293.15, 400.0, 0.8
    tree = AdaptiveMesh.from_bands(
        FINEST, H, [RefinementBand((0.0, 0.0, 0.0), (BOX, BOX, FILM), H)], base_level=2)
    tree.excluded = tree.centres()[:, 2] > FILM + 1e-9
    tree.k[:] = 1.0
    tree.Q_source[:] = np.where(tree.excluded, 0.0, density)
    tree.h_out = film
    tree.h_out_conv = film
    tree.environment_emissivity = emissivity
    tree.t_ambient = t_ambient
    tree.set_fixed_temperature_bc("z_min", t_ground)
    for face in ("x_max", "y_min", "y_max", "z_max"):
        tree.set_adiabatic(face)

    grid = GridSpec(x=(Band(0.0, BOX, H),), y=(Band(0.0, BOX, H),),
                    z=(Band(0.0, FILM, H), Band(0.0, BOX, KINK)))
    structured = Mesh3D(BOX, BOX, BOX, grid=grid)
    structured.excluded = structured.Z > FILM + 1e-9
    structured.k[:] = 1.0
    structured.Q_source[:] = np.where(structured.excluded, 0.0, density)
    structured.h_out = film
    structured.h_out_conv = film
    structured.environment_emissivity = emissivity
    structured.t_ambient = t_ambient
    structured.set_fixed_temperature_bc("z_min", t_ground)
    for face in ("x_max", "y_min", "y_max", "z_max"):
        structured.set_adiabatic(face)

    config = SolverConfig(method="direct", radiation=True, max_picard=8,
                          picard_tolerance=1e-9)
    adaptive = SteadyStateSolver(tree, config).solve()
    reference = SteadyStateSolver(structured, config).solve()
    assert adaptive.converged and reference.converged

    # the active slab has the same cells on both meshes: the same field, leaf by leaf
    active = ~tree.excluded
    index = cell_of(structured, tree)
    field = np.asarray(reference.T, dtype=float).ravel(order="F")
    assert np.abs(adaptive.T[active] - field[index[active]]).max() < 1e-9
    assert np.allclose(adaptive.T[tree.excluded], t_ambient)

    # the film acts on the same faces, with the same coefficient
    assert tree.h_out == pytest.approx(structured.h_out, rel=1e-12)
    assert tree.h_out > film, "the radiative share must be in the film"
    assert tree.balance().q_environment == pytest.approx(environment_flux(structured),
                                                         rel=1e-9)
    assert tree.balance().closure < 1e-12
    assert abs(compute_balance(structured, radiation=True).imbalance) < 1e-6 * 1e4
