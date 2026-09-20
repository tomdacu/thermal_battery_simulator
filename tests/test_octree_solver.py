"""The octree wired to the physics: equivalence, conservation, and the refinement cycle.

What this module pins, in the order the claims matter:

* **equivalence** - on a uniform tree and the matching uniform :class:`Mesh3D` the two
  solvers assemble the same operator, so a two-layer wall (1-D) and a slab with a
  source in half the box (2-D) agree to the tolerance of the linear solve;
* **conservation** - every face carries one conductance shared by the two cells, so the
  flux field is conservative to machine precision on a *graded* tree, and the heat
  leaving through the fixed leaves equals the power the sources deposit;
* **the a posteriori estimator** - the flux-jump indicator marks the hot spot and the
  shell where the flux turns, and leaves the far field at zero;
* **the refinement cycle** - starting from 64 leaves and stopping when the objective
  stops moving, it answers the storage mean temperature of a hot spot about 8x closer to
  the fine reference than a uniform tree of 512 leaves (three times as many cells), and
  it leaves a field with no curvature alone.

The box is 1 m and the finest tree cell is 1/16 m, so every uniform tree is a
power-of-two subdivision of the box and the structured partner has exactly its cells.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.convergence import ConvergenceTarget
from src.core.mesh import Mesh3D
from src.core.octree import Octree, uniform_tree
from src.solver.octree_solver import (
    OctreeSteadySolver,
    dirichlet_walls,
    mean_temperature,
    refine_on_objective,
    storage_mean_objective,
    wall_cells,
)
from src.solver.steady import SolverConfig, SteadyStateSolver

BOX = 1.0
FINEST = 16                      # finest cells per side of the test box
H = BOX / FINEST                 # edge of one finest cell [m]
ALL_WALLS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


# ------------------------------------------------------------------- helpers
def cell_edge(level: int) -> float:
    """Edge of a leaf at ``level`` [m]."""
    return H * (1 << level)


def leaf_rule(rule):
    """Per-leaf field from a rule of the leaf centre in physical coordinates."""
    return lambda leaf: float(rule(*np.array(leaf.centre) * H))


def structured_partner(tree: Octree, walls: dict[str, float], conductivity,
                       source=None) -> Mesh3D:
    """A uniform :class:`Mesh3D` with the same cells as a uniform tree.

    The tree walls carry no face at all, so the mesh gets the same physics by marking
    every other face adiabatic and pinning the leaves of the walls asked for.
    """
    assert len({leaf.level for leaf in tree.leaves}) == 1, "a uniform tree is required"
    mesh = Mesh3D(BOX, BOX, BOX, spacing=cell_edge(tree.leaves[0].level))
    for face in ALL_WALLS:
        mesh.set_adiabatic(face)
    for face, value in walls.items():
        mesh.set_fixed_temperature_bc(face, value)
    mesh.k[:] = conductivity(mesh.X, mesh.Y, mesh.Z) if callable(conductivity) \
        else conductivity
    if source is not None:
        mesh.Q_source[:] = source(mesh.X, mesh.Y, mesh.Z) if callable(source) else source
    return mesh


def solved_pair(tree: Octree, walls: dict[str, float], conductivity, source=None):
    """The octree solution and the structured one, with the mesh fields aligned."""
    solver = OctreeSteadySolver(
        tree, conductivity=conductivity if not callable(conductivity) else leaf_rule(
            conductivity),
        source=None if source is None else (source if not callable(source)
                                            else leaf_rule(source)),
        fixed=lambda t: dirichlet_walls(t, walls), physical_size=H)
    result = solver.solve()
    mesh = structured_partner(tree, walls, conductivity, source)
    structured = SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
    index = [mesh.ijk_to_linear(*mesh.find_cell(*(c * H for c in leaf.centre)))
             for leaf in tree.leaves]
    difference = float(np.abs(result.T - structured.T[index]).max())
    return solver, result, structured, difference


def plane_flux(tree: Octree, solver: OctreeSteadySolver, result, axis: int,
               plane: float) -> float:
    """Heat rate [W] crossing a plane of the box, from the low side to the high side.

    The plane must fall on a leaf face: the leaves on either side of it carry the flow
    and nothing else does.
    """
    carrier = int(round(plane / H))
    flux = solver.face_fluxes(result.T, tree)
    total = 0.0
    for (i, j, face_axis, _area, _distance), rate in zip(tree.faces(), flux, strict=True):
        if face_axis != axis:
            continue
        low, high = (i, j) if _corner(tree, i, axis) < _corner(tree, j, axis) else (j, i)
        if _corner(tree, low, axis) + tree.leaves[low].size != carrier:
            continue
        total += rate if low == i else -rate
    return float(total)


def _corner(tree: Octree, index: int, axis: int) -> int:
    leaf = tree.leaves[index]
    return (leaf.x, leaf.y, leaf.z)[axis]


# ------------------------------------------------------- the physics entry points
def test_the_wall_helpers_pin_the_leaves_of_a_box_face():
    """A wall condition drives the leaves sitting on that wall, and only those."""
    tree = uniform_tree(FINEST, 2)                      # 4x4x4 leaves
    assert len(wall_cells(tree, "z_min")) == 16
    assert all(leaf.z == 0 for leaf in wall_cells(tree, "z_min"))
    assert all(leaf.z + leaf.size == tree.n for leaf in wall_cells(tree, "z_max"))
    assert len(wall_cells(tree, "x_min")) == 16
    with pytest.raises(ValueError):
        wall_cells(tree, "top")

    fixed = dirichlet_walls(tree, {"z_min": 400.0, "z_max": 300.0})
    assert len(fixed) == 32
    assert {value for leaf, value in fixed.items() if leaf.z == 0} == {400.0}
    assert {value for leaf, value in fixed.items()
            if leaf.z + leaf.size == tree.n} == {300.0}


def test_the_mean_temperature_weights_the_leaves_by_volume():
    """A cell count must not move the mean: small leaves cannot outvote a big one.

    The tree of the test has a quarter of the box in sixteen small leaves held at 300 K
    and three quarters held at 400 K: the volume-weighted mean is exactly 375 K, while
    the mean over the *leaves* is 355.6 K, so the two answers tell each other apart.
    """
    tree = Octree(FINEST)
    tree.refine(lambda leaf: 1.0 if leaf.z == 0 else 0.0, threshold=0.5, levels=2)
    assert len(tree.level_histogram()) > 1 and tree.n_cells == 36
    temperature = np.array([300.0 if leaf.z == 0 else 400.0 for leaf in tree.leaves])
    volumes = tree.cell_sizes() ** 3
    assert volumes[[leaf.z == 0 for leaf in tree.leaves]].sum() == 0.25 * FINEST ** 3

    assert mean_temperature(tree, temperature) == pytest.approx(375.0)
    assert float(temperature.mean()) == pytest.approx(355.5556, rel=1e-6)
    assert mean_temperature(tree, temperature, lambda leaf: leaf.z == 0) == \
        pytest.approx(300.0)
    with pytest.raises(ValueError):
        mean_temperature(tree, temperature, lambda leaf: False)


def test_the_physics_entry_points_reject_what_has_no_solution():
    """The mistakes this API makes easy: a floating problem, a stale map, a bad k."""
    tree = uniform_tree(FINEST, 2)
    walls = dirichlet_walls(tree, {"z_min": 300.0})
    solver = OctreeSteadySolver(tree, conductivity=1.0, fixed=walls, physical_size=H)
    assert solver.solve().cells == tree.n_cells

    with pytest.raises(ValueError, match="fixed"):
        OctreeSteadySolver(tree, conductivity=1.0, physical_size=H).solve()
    with pytest.raises(ValueError, match="not in the tree"):
        # the map was built for the coarse tree: a refinement leaves it naming leaves
        # that no longer exist, which is what the rule form of ``fixed`` is for
        tree.refine(lambda _leaf: 1.0, threshold=0.5, levels=1)
        OctreeSteadySolver(tree, conductivity=1.0, fixed=walls, physical_size=H).solve()
    with pytest.raises(ValueError, match="conductivity"):
        OctreeSteadySolver(tree, conductivity=0.0, fixed=walls, physical_size=H).solve()
    with pytest.raises(ValueError, match="physical_size"):
        OctreeSteadySolver(tree, physical_size=0.0, fixed=walls)
    with pytest.raises(ValueError, match="one value per leaf"):
        OctreeSteadySolver(tree, conductivity=np.ones(3), fixed=walls,
                           physical_size=H).solve()


def test_a_fixed_rule_follows_the_refinement_where_a_map_cannot():
    """The rule form is what keeps a wall a wall: its leaves are re-derived per tree."""
    rule = lambda t: dirichlet_walls(t, {"z_min": 400.0, "z_max": 300.0})  # noqa: E731
    solver = OctreeSteadySolver(tree := uniform_tree(FINEST, 2), conductivity=1.0,
                                fixed=rule, physical_size=H)
    assert len(solver.fixed_cells()) == 32
    tree.refine(lambda _leaf: 1.0, threshold=0.5, levels=1)
    assert tree.n_cells == 8 * 64
    fixed = solver.fixed_cells()
    assert len(fixed) == 2 * 64, "the refined wall leaves are the new boundary"
    assert {value for position, value in fixed.items()
            if tree.leaves[position].z == 0} == {400.0}
    assert solver.solve().cells == tree.n_cells


# ------------------------------------------------------------------ conservation
def test_the_flux_report_is_conservative_to_machine_precision():
    """One conductance per face: the flux closes on a graded tree, cell by cell.

    A pair of leaves shares a single number, so summing the per-cell divergence over the
    whole mesh cancels face by face - machine precision, and it holds for *any*
    temperature - while the equality of the boundary flux with the source power is the
    residual of the linear solve and is only as good as its tolerance.
    """
    tree = Octree(FINEST)
    tree.refine(lambda leaf: 1.0 if leaf.z < 8 else 0.0, threshold=0.5, levels=3)
    assert len(tree.level_histogram()) > 1, "the test needs hanging nodes"
    conductivity = lambda x, y, z: np.where(z < 0.5, 1.0, 5.0)        # noqa: E731
    source = lambda x, y, z: np.where((z >= 0.25) & (z < 0.75), 2.0e4, 0.0)  # noqa: E731
    solver = OctreeSteadySolver(tree, conductivity=leaf_rule(conductivity),
                                source=leaf_rule(source),
                                fixed=lambda t: dirichlet_walls(
                                    t, {"z_min": 300.0, "z_max": 350.0}),
                                physical_size=H)
    result = solver.solve()
    assert result.cells == tree.n_cells

    scale = float(np.abs(result.face_flux).max())
    assert abs(result.divergence.sum()) <= 1e-12 * scale, "the mesh creates heat"
    assert result.total_flux == pytest.approx(result.source_power, rel=1e-9)
    assert result.balance_error < 1e-9
    assert result.residual < 1e-9 * max(abs(result.source_power), 1.0)
    assert np.allclose(solver.divergence(result.T), result.divergence, rtol=1e-12,
                       atol=1e-12 * scale)

    # the assembled operator sees the same conductance on both sides of every face:
    # with one area per cell this is false, which is what makes the octree conservative
    volumes = (tree.cell_sizes() * H) ** 3
    matrix = tree.diffusion_matrix(solver.conductivities(), H)
    for i, j, _axis, _area, _distance in tree.faces():
        left = matrix[i, j] * volumes[i]
        right = matrix[j, i] * volumes[j]
        assert abs(left - right) <= 1e-14 * abs(left)


def test_a_two_layer_wall_matches_the_structured_solver_and_the_series_resistance():
    """1-D wall, two materials: same field as ``SteadyStateSolver``, same flux.

    Each layer is one cell thick, so the discrete answer has a closed form: the
    resistance between the two pinned cell centres is the length of each material over
    its conductivity.  The harmonic mean of the face between them is exactly the
    continuity of the flux that formula assumes.
    """
    tree = uniform_tree(FINEST, 2)                       # 4 leaves of 0.25 m
    conductivity = lambda x, y, z: np.where(z < 0.5, 20.0, 0.5)      # noqa: E731
    walls = {"z_min": 500.0, "z_max": 290.0}
    solver, result, structured, difference = solved_pair(tree, walls, conductivity)
    assert difference < 1e-8, f"the two solvers disagree by {difference} K"
    assert structured.residual < 1e-12, "the structured answer is the reference"

    first, last = 0.125, 0.875            # centres of the two pinned leaves
    resistance = (0.5 - first) / 20.0 + (last - 0.5) / 0.5
    analytic = (500.0 - 290.0) / resistance               # W/m^2 over one square metre
    assert plane_flux(tree, solver, result, axis=2, plane=0.5) == pytest.approx(
        analytic, rel=1e-9)
    assert plane_flux(tree, solver, result, axis=2, plane=0.75) == pytest.approx(
        analytic, rel=1e-9), "the flux is the same through every plane of the wall"
    # a two-sided wall takes as much heat as it delivers: the net boundary flux vanishes
    assert abs(result.total_flux) < 1e-6 * analytic


def test_a_source_slab_matches_the_structured_solver_in_two_dimensions():
    """2-D slab: a source in half the box, two pinned walls, four adiabatic.

    Heat flows toward both walls and spreads sideways where the source stops, so the
    field varies in x and z; the octree and the structured mesh must still agree, and the
    field must stay symmetric in y, which neither of them is told.
    """
    tree = uniform_tree(FINEST, 1)                       # 8x8x8 leaves of 0.125 m
    density = 4.0e2
    source = lambda x, y, z: np.where((x < 0.5) & (z >= 0.25) & (z < 0.5),
                                      density, 0.0)                # noqa: E731
    walls = {"z_min": 300.0, "z_max": 350.0}
    _, result, _, difference = solved_pair(tree, walls, 1.0, source)
    assert difference < 1e-8, f"the two solvers disagree by {difference} K"

    power = density * 0.5 * 1.0 * 0.25                   # W/m^3 over half the box
    assert result.source_power == pytest.approx(power)
    assert result.total_flux == pytest.approx(power, rel=1e-9)
    assert result.balance_error < 1e-9
    field = result.T.reshape((8, 8, 8), order="C")       # leaves are z-fastest
    assert abs(field - field[:, ::-1, :]).max() < 1e-9, "the problem is symmetric in y"


# ------------------------------------------------------------------- refinement
def test_the_indicator_marks_the_hot_spot_and_leaves_the_far_field_alone():
    """The estimator is the flux that changes from one face to the next.

    A leaf that generates heat shows the whole source power - the two sides of its
    balance differ by exactly that - and the shell around it shows the flux turning
    sideways; two cells away the field is a constant flux and the estimator is at zero.
    """
    tree = uniform_tree(FINEST, 2)
    density = 6.4e3                                      # 100 W in a 0.25 m cube
    inside = lambda leaf: all(0.25 <= q < 0.5 for q in np.array(leaf.centre) * H)  # noqa: E731
    solver = OctreeSteadySolver(
        tree, conductivity=1.0,
        source=lambda leaf: density if inside(leaf) else 0.0,
        fixed=lambda t: dirichlet_walls(t, {face: 300.0 for face in ALL_WALLS}),
        physical_size=H)
    result = solver.solve()

    heater = [i for i, leaf in enumerate(tree.leaves) if inside(leaf)]
    assert len(heater) == 1, "the coarse tree holds the source in one leaf"
    jumps = result.jump
    assert jumps[heater[0]] == pytest.approx(density * cell_edge(2) ** 3, rel=1e-9)
    assert jumps.max() == jumps[heater[0]]
    assert int(np.count_nonzero(jumps > 1e-6 * jumps.max())) == 7, \
        "the hot spot and its six face neighbours"
    far = [i for i, leaf in enumerate(tree.leaves)
           if leaf.x in (0, 12) and leaf.y in (0, 12) and leaf.z in (0, 12)]
    assert len(far) == 8 and np.abs(jumps[far]).max() == 0.0


def test_the_refinement_cycle_reaches_the_uniform_answer_with_fewer_cells():
    """The claim the module exists for, measured on a hot spot.

    The objective is the storage mean - the volume-weighted mean temperature of the leaves
    of the source region - and the reference is the finest tree the box allows (4096
    leaves of 6.25 mm).  The cycle stops when the objective stops moving by more than the
    tolerance asked for, and the tree it hands back must beat a uniform tree of three times
    its cells by at least a factor of five.
    """
    density = 6.4e3                                      # 100 W in a 0.25 m cube
    inside = lambda leaf: all(0.25 <= q < 0.5 for q in np.array(leaf.centre) * H)  # noqa: E731
    walls = {face: 300.0 for face in ALL_WALLS}

    def solve_on(tree: Octree) -> OctreeSteadySolver:
        return OctreeSteadySolver(
            tree, conductivity=1.0,
            source=lambda leaf: density if inside(leaf) else 0.0,
            fixed=lambda t: dirichlet_walls(t, walls), physical_size=H)

    reference_tree = uniform_tree(FINEST, 0)
    reference = solve_on(reference_tree).solve()
    reference_mean = mean_temperature(reference_tree, reference.T, inside)

    uniform_tree_512 = uniform_tree(FINEST, 1)           # 512 leaves of 125 mm
    uniform = solve_on(uniform_tree_512).solve()
    uniform_error = abs(mean_temperature(uniform_tree_512, uniform.T, inside)
                        - reference_mean)

    tree = uniform_tree(FINEST, 2)                       # 64 leaves of 250 mm
    coarse = solve_on(tree)
    coarse_error = abs(mean_temperature(tree, coarse.solve().T, inside)
                       - reference_mean)
    report = refine_on_objective(
        tree, coarse, storage_mean_objective(inside),
        ConvergenceTarget(delta_power=2e-2, delta_temperature=1e-6, max_levels=6,
                          max_cells=100_000),
        refine_fraction=0.25, floor_ratio=0.05, name="storage mean")

    assert report.converged, report.message
    assert report.tree is tree and report.rounds == 2 and report.levels[-1].cells == 176
    assert report.tree.n_cells < uniform_tree_512.n_cells, "fewer cells, not more"
    adaptive_error = abs(report.chosen.power - reference_mean)
    assert 5.0 * adaptive_error <= uniform_error, (
        f"adaptive {adaptive_error:.4f} K on {report.tree.n_cells} leaves is not 5x "
        f"better than {uniform_error:.4f} K on {uniform_tree_512.n_cells} leaves")
    assert adaptive_error < coarse_error / 5.0, "the cycle must beat its own start"
    assert report.chosen.cells == report.tree.n_cells
    assert report.chosen.min_size == pytest.approx(H)
    assert report.chosen.max_size == pytest.approx(4 * H)


def test_the_refinement_leaves_a_flat_field_alone():
    """A linear profile leaves the estimator at its round-off level: nothing to refine."""
    tree = uniform_tree(FINEST, 2)
    solver = OctreeSteadySolver(tree, conductivity=1.0,
                                fixed=lambda t: dirichlet_walls(
                                    t, {"z_min": 400.0, "z_max": 300.0}),
                                physical_size=H)
    result = solver.solve()
    assert float(result.T.max() - result.T.min()) == pytest.approx(100.0)
    assert np.abs(result.jump).max() < 1e-12 * float(np.abs(result.face_flux).max())

    report = refine_on_objective(tree, solver, lambda t, r: mean_temperature(t, r.T),
                                 ConvergenceTarget(max_levels=4, max_cells=10_000))
    assert not report.converged and report.rounds == 0
    assert report.tree.n_cells == 64, "nothing was marked, nothing was split"
    assert "nothing to refine" in report.message
    assert report.levels[0].d_power != report.levels[0].d_power         # nan, no change


def test_the_estimator_needs_a_two_sided_stencil():
    """A tree of two cells per side has one face per axis: there is nothing to compare.

    The field is not flat - a uniform source drives heat out of the bottom wall - and the
    estimator still reads zero on every leaf, because the second difference of a field
    does not exist without the two neighbours it compares.  The cycle says so instead of
    refining a mesh it cannot read.
    """
    tree = uniform_tree(FINEST, 3)                       # 2x2x2 leaves, all on a wall
    solver = OctreeSteadySolver(tree, conductivity=1.0, source=1.0e3,
                                fixed=lambda t: dirichlet_walls(t, {"z_min": 300.0}),
                                physical_size=H)
    result = solver.solve()
    assert float(result.T.max() - result.T.min()) > 1.0
    assert (result.jump == 0.0).all()

    report = refine_on_objective(tree, solver, lambda t, r: float(r.T.max()),
                                 ConvergenceTarget(max_levels=3, max_cells=10_000))
    assert not report.converged and report.rounds == 0 and report.tree.n_cells == 8
    assert "nothing to refine" in report.message


def test_the_cycle_stops_at_the_cell_budget():
    """The budget is checked *before* a round, so the tree never runs away."""
    tree = uniform_tree(FINEST, 2)
    inside = lambda leaf: all(0.25 <= q < 0.5 for q in np.array(leaf.centre) * H)  # noqa: E731
    solver = OctreeSteadySolver(
        tree, conductivity=1.0,
        source=lambda leaf: 6.4e3 if inside(leaf) else 0.0,
        fixed=lambda t: dirichlet_walls(t, {face: 300.0 for face in ALL_WALLS}),
        physical_size=H)
    report = refine_on_objective(tree, solver, storage_mean_objective(inside),
                                 ConvergenceTarget(delta_power=1e-9, delta_temperature=1e-9,
                                                   max_levels=6, max_cells=100),
                                 refine_fraction=0.25, floor_ratio=0.05)
    assert not report.converged and "budget" in report.message
    assert report.tree.n_cells <= 100
