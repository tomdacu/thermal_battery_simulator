"""The automatic mesh search on an adaptive mesh: the same search, an octree of cells.

The search asks a level for a *scale*, and the two roads realise it differently: a graded
grid scales every band target, a tree scales the bands of the a priori plan and adds one
round of refinement on the flux-jump indicator of the field it has already solved.  The
tests here run the same :func:`~src.analysis.convergence.find_mesh`, with the same
:class:`ConvergenceTarget`, on the same physical model over the two meshes, and compare
the meshes the two searches settle on.

The model is a 1 m cube made one-dimensional on purpose: the sides are adiabatic, the
bottom is held at 300 K and the top loses heat through a film of 8 W/(m2 K), under a
uniform source of 2000 W/m3.  The answer then depends on the *vertical* resolution alone,
which is what makes the two roads comparable number by number: on the same leaf height
they must agree, and they do (see ``test_the_two_roads_choose_the_same_answer``).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.balance import compute_balance
from src.analysis.convergence import AdaptivePlan, ConvergenceTarget, find_mesh
from src.analysis.mesh_plan import refinement_bands, tree_resolution
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.mesh import Mesh3D
from src.core.refinement import Band, GridSpec
from src.solver.steady import SolverConfig, SteadyStateSolver

#: the biggest extent of the domain: an octree spans a cube
BOX = 0.4
#: the finest cell the a priori plan asks for; every band here is a slice of z
TARGET_SIZE = 0.05
#: the search's tolerances: a level of a tree makes a bigger step than a level of a graded
#: grid (a leaf edge is a power of two of the finest cell, so a round halves it or does
#: nothing), and an octree pays for the refinement in three directions where a graded grid
#: pays per axis.  The comparison below is held to these numbers, and the two roads agree
#: far inside them.
TARGET = ConvergenceTarget(delta_temperature=4.0, delta_power=0.08, max_levels=5,
                           refine=0.5, max_cells=40_000)


def spec() -> GridSpec:
    """The a priori request: one band in z, a single cell across the adiabatic sides."""
    return GridSpec(x=(Band(0.0, BOX, BOX),), y=(Band(0.0, BOX, BOX),),
                    z=(Band(0.0, BOX, TARGET_SIZE),))


def plan(budget: int = TARGET.max_cells) -> AdaptivePlan:
    """The same request as boxes of an octree: one box per band of the spec."""
    n_finest, physical_size = tree_resolution((BOX, BOX, BOX), budget)
    return AdaptivePlan(n_finest=n_finest, physical_size=physical_size,
                        bands=refinement_bands(spec(), (BOX, BOX, BOX)))


def paint(mesh: Mesh3D | AdaptiveMesh) -> Mesh3D | AdaptiveMesh:
    """The 1-D model of the module docstring, on whichever mesh is handed over."""
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    mesh.Q_source[:] = 2000.0
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    mesh.set_fixed_temperature_bc("z_min", 300.0)
    mesh.set_convection_bc("z_max", 8.0, 293.15)
    return mesh


def observables(mesh: Mesh3D | AdaptiveMesh) -> dict:
    """Objective of the search: the volume mean temperature and the heat leaving [W]."""
    SteadyStateSolver(mesh, SolverConfig(method="cg", tolerance=1e-9)).solve()
    balance = compute_balance(mesh)
    return {"t_mean_storage": float(np.average(mesh.T, weights=mesh.V)),
            "t_max": float(mesh.T.max()), "power": float(balance.q_domain)}


def build_graded(spec_level: GridSpec) -> Mesh3D:
    return paint(Mesh3D(BOX, BOX, BOX, grid=spec_level))


def build_tree(plan_level: AdaptivePlan) -> AdaptiveMesh:
    return paint(AdaptiveMesh.from_bands(plan_level.n_finest, plan_level.physical_size,
                                         plan_level.bands, plan_level.base_level))


@pytest.fixture(scope="module")
def searched() -> tuple:
    """The two searches on the same model: ``(graded report, tree report)``."""
    graded = find_mesh(build_graded, observables, spec(), TARGET)
    tree = find_mesh(build_tree, observables, plan(), TARGET)
    return graded, tree


# ------------------------------------------------------------------ the plan
def test_the_bands_map_one_to_one_to_refinement_boxes():
    """A band is a slice of one axis, its box spans the others - and keeps its target.

    This is what lets the same a priori estimate drive both meshes: the panel's bands
    (``thickness / N``, ``2 k / h``, the tube pitch) become octree boxes unchanged, one
    for one, and a leaf is refined to the size of the box that covers it.
    """
    request = GridSpec(x=(Band(0.0, BOX, BOX),), y=(Band(0.0, BOX, BOX),),
                       z=(Band(0.0, BOX, 0.1), Band(0.2, BOX, 0.05)))
    boxes = refinement_bands(request, (BOX, BOX, BOX))
    axes = (request.x, request.y, request.z)
    declared = [(axis, band) for axis, bands in enumerate(axes) for band in bands]
    assert len(boxes) == len(declared)
    for box, (axis, band) in zip(boxes, declared, strict=True):
        assert box.size == band.target
        assert box.low[axis] == band.start and box.high[axis] == band.end
        for other in range(3):
            if other != axis:
                assert box.low[other] == 0.0 and box.high[other] == BOX
    # and the tree built from them meets the finest target, on the finest leaf height
    n_finest, physical_size = tree_resolution((BOX, BOX, BOX), 4_000)
    mesh = AdaptiveMesh.from_bands(n_finest, physical_size, boxes)
    assert mesh.sizes.min() <= 0.05
    assert mesh.sizes.min() >= physical_size                 # the resolution floor holds
    assert mesh.sizes.max() <= 0.1                           # the coarsest band


def test_the_tree_box_comes_from_the_cell_budget():
    """The box is a cube of a power-of-two number of finest cells, and it is a request."""
    n_finest, physical_size = tree_resolution((BOX, BOX, 0.3), 10_000)
    assert n_finest & (n_finest - 1) == 0                # a power of two
    assert n_finest * physical_size == pytest.approx(BOX)   # the largest extent
    coarse = tree_resolution((BOX, BOX, BOX), 1_000)[1]
    fine = tree_resolution((BOX, BOX, BOX), 100_000)[1]
    assert coarse > fine, "a bigger budget buys a finer floor"


# -------------------------------------------------------------- the search
def test_the_tree_search_converges_and_matches_the_graded_one(searched):
    """The acceptance: both roads stop, and they stop on the same answer."""
    graded, tree = searched
    assert graded.converged, graded.message
    assert tree.converged, tree.message
    assert tree.chosen.cells > graded.chosen.cells       # an octree refines in 3-D
    assert tree.chosen.t_mean_storage == pytest.approx(graded.chosen.t_mean_storage,
                                                       abs=TARGET.delta_temperature)
    assert tree.chosen.power == pytest.approx(graded.chosen.power, rel=TARGET.delta_power)
    assert "converged" in tree.summary()
    assert "leaves" in tree.summary(), "a tree counts leaves, not cells"


def test_the_two_roads_choose_the_same_answer(searched):
    """Numbers of the two chosen meshes, next to each other.

    The two searches were asked the same tolerances and each one reports the discretisation
    uncertainty of its own mesh; what the numbers below add is that the two meshes are the
    same *discretisation*: at the same leaf height the fields agree to well inside the
    tolerance, which is the equivalence step 2 of the migration measured on a uniform pair.
    """
    graded, tree = searched
    chosen = tree.chosen
    assert chosen.min_size <= graded.chosen.min_size      # at least as fine where it counts
    assert chosen.worst_ratio <= 2.0, "the octree keeps its 2:1 balance"
    # the level the tree stopped on moved by less than the tolerances it was given
    assert chosen.d_temperature <= TARGET.delta_temperature
    assert chosen.d_power <= TARGET.delta_power
    assert chosen.t_mean_storage == pytest.approx(graded.chosen.t_mean_storage,
                                                  abs=TARGET.delta_temperature)
    assert chosen.power == pytest.approx(graded.chosen.power, rel=TARGET.delta_power)
    # the first level of both roads is the a priori plan: same cells, same answer
    assert tree.levels[0].cells != graded.levels[0].cells  # 512 leaves vs 8 cells
    assert tree.levels[0].t_mean_storage == pytest.approx(
        graded.levels[0].t_mean_storage, rel=1e-9)
    assert tree.levels[0].power == pytest.approx(graded.levels[0].power, rel=1e-9)


def test_the_tree_the_search_hands_back_is_the_chosen_one(searched):
    """The report carries the tree itself: a level is a round on a solved field, not a spec."""
    _, tree = searched
    assert isinstance(tree.spec, AdaptivePlan), "a tree search reports the plan it started from"
    mesh = tree.mesh
    assert isinstance(mesh, AdaptiveMesh)
    assert mesh.n_cells == tree.chosen.cells
    assert mesh.sizes.max() <= max(band.size for band in tree.spec.bands) + 1e-12
    assert mesh.tree.max_level == int(np.log2(tree.spec.n_finest))
    # the rounds are what the plan alone cannot produce: the chosen mesh is not uniform
    assert len(mesh.level_histogram()) > 1


def test_the_cell_budget_stops_the_tree_and_says_so():
    """A budget that cannot deliver must say so instead of returning a bad mesh."""
    budget = 4_000
    target = ConvergenceTarget(delta_temperature=0.05, delta_power=0.001, max_levels=5,
                               refine=0.5, max_cells=budget)
    report = find_mesh(build_tree, observables, plan(budget), target)
    assert not report.converged, report.summary()
    assert "budget" in report.message
    cells = [level.cells for level in report.levels]
    assert cells == sorted(cells), "a level that is not finer is not evidence"
    assert cells[-1] <= budget, "the tree never spends more than the budget allows"
    assert report.mesh is not None and report.mesh.n_cells == cells[-1]


def test_the_search_is_cancelled_where_it_is_asked_to_stop():
    """``should_stop`` is what the GUI's cancel button pulls."""
    report = find_mesh(build_tree, observables, plan(), TARGET, should_stop=lambda: True)
    assert not report.converged
    assert "cancelled" in report.message
    assert not report.levels, "the stop is checked before the first solve"
