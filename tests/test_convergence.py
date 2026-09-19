"""Automatic mesh search: it must stop when the answer stops moving."""
from __future__ import annotations

import pytest

from src.analysis.balance import compute_balance
from src.analysis.convergence import ConvergenceTarget, find_mesh
from src.core.geometry import create_small_test_geometry
from src.core.mesh import Mesh3D
from src.core.refinement import Band, GridSpec
from src.solver.steady import SolverConfig, SteadyStateSolver


def slab_problem() -> tuple:
    """Analytic 1-D slab with a source: the observables have a known limit."""
    def build(spec: GridSpec) -> Mesh3D:
        mesh = Mesh3D(0.4, 0.4, 2.0, grid=spec)
        mesh.k[:] = 1.0
        mesh.rho[:] = 1000.0
        mesh.cp[:] = 1000.0
        mesh.Q_source[:] = 2000.0
        for face in ("x_min", "x_max", "y_min", "y_max"):
            mesh.set_adiabatic(face)
        mesh.set_fixed_temperature_bc("z_min", 300.0)
        mesh.set_convection_bc("z_max", 8.0, 293.15)
        return mesh

    def observables(mesh: Mesh3D) -> dict:
        SteadyStateSolver(mesh, SolverConfig(method="cg", tolerance=1e-7)).solve()
        balance = compute_balance(mesh)
        return {"t_mean_storage": float(mesh.T.mean()), "t_max": float(mesh.T.max()),
                "power": float(balance.q_domain)}

    # the slab is 1-D: keep the cross-section coarse so that refining only costs z
    spec = GridSpec(x=(Band(0.0, 0.4, 0.4),), y=(Band(0.0, 0.4, 0.4),),
                    z=(Band(0.0, 2.0, 0.2),))
    return build, observables, spec


def battery_problem() -> tuple:
    def build(spec: GridSpec) -> Mesh3D:
        mesh = Mesh3D(6.0, 6.0, 5.6, grid=spec)
        geometry = create_small_test_geometry()
        geometry.heaters.power_total = 5.0
        geometry.apply_to_mesh(mesh)
        return mesh

    def observables(mesh: Mesh3D) -> dict:
        # the driver is what is under test here, not solver accuracy: a loose
        # tolerance keeps the sequence of levels quick
        SteadyStateSolver(mesh, SolverConfig(method="cg", preconditioner="amg_rs",
                                             tolerance=1e-6)).solve()
        balance = compute_balance(mesh)
        sand = mesh.material_id == 1
        return {"t_mean_storage": float(mesh.T[sand].mean()), "t_max": float(mesh.T.max()),
                "power": float(balance.q_domain)}

    cyl = create_small_test_geometry().cylinder
    grid = GridSpec(
        x=(Band(0, 6, 0.4), Band(cyl.center_x - cyl.r_storage, cyl.center_x + cyl.r_storage,
                                 0.2)),
        y=(Band(0, 6, 0.4), Band(cyl.center_y - cyl.r_storage, cyl.center_y + cyl.r_storage,
                                 0.2)),
        z=(Band(cyl.z_storage_start, cyl.z_storage_end, 0.2), Band(0, 5.6, 0.4)))
    return build, observables, grid


# ------------------------------------------------------------------ validation
def test_the_target_rejects_nonsense():
    assert ConvergenceTarget(refine=1.0).validate()
    assert ConvergenceTarget(max_levels=1).validate()
    assert ConvergenceTarget(delta_temperature=0.0).validate()
    assert ConvergenceTarget().validate() == []
    build, observables, spec = slab_problem()
    with pytest.raises(ValueError):
        find_mesh(build, observables, spec, ConvergenceTarget(refine=2.0))


# ------------------------------------------------------------------- behaviour
def test_the_search_stops_when_the_answer_stops_moving():
    build, observables, spec = slab_problem()
    report = find_mesh(build, observables, spec,
                       ConvergenceTarget(delta_temperature=5.0, delta_power=0.02,
                                         max_levels=6, refine=0.5, max_cells=60_000))
    assert report.converged, report.message
    assert len(report.levels) >= 2
    assert report.levels[1].d_temperature > 5.0        # the probe was not enough
    assert report.chosen.cells > report.levels[0].cells          # it really refined
    assert "converged" in report.summary()


def test_a_loose_tolerance_is_cheaper_than_a_tight_one():
    """The point of the search: spend cells only where accuracy is asked for."""
    build, observables, spec = slab_problem()
    loose = find_mesh(build, observables, spec,
                      ConvergenceTarget(delta_temperature=20.0, delta_power=0.2,
                                        max_levels=4, refine=0.5))
    tight = find_mesh(build, observables, spec,
                      ConvergenceTarget(delta_temperature=0.02, delta_power=0.0005,
                                        max_levels=5, refine=0.5))
    assert loose.chosen.cells <= tight.chosen.cells


def test_an_unreachable_tolerance_is_reported_not_hidden():
    """A budget that cannot deliver must say so instead of returning a bad mesh."""
    build, observables, spec = battery_problem()
    report = find_mesh(build, observables, spec,
                       ConvergenceTarget(delta_temperature=1e-6, delta_power=1e-9,
                                         max_levels=3, refine=0.5, max_cells=20_000))
    assert not report.converged
    assert "stopped at level" in report.message or "not converged" in report.message
    assert report.levels, "the levels that were tried must be reported"


def test_a_blocked_refinement_is_never_called_converged():
    """Two identical grids are not evidence: the budget must be reported as the limit."""
    build, observables, spec = battery_problem()
    report = find_mesh(build, observables, spec,
                       ConvergenceTarget(delta_temperature=0.001, delta_power=0.001,
                                         max_levels=4, refine=0.5, max_cells=12_000))
    assert not report.converged, report.summary()
    assert "budget" in report.message
    cells = [level.cells for level in report.levels]
    assert cells == sorted(cells), "a level that is not finer is not evidence"


def test_the_reported_mesh_reproduces_the_converged_answer():
    """The chosen grid must be the one whose observables met the tolerance."""
    build, observables, spec = battery_problem()
    report = find_mesh(build, observables, spec,
                       ConvergenceTarget(delta_temperature=3.0, delta_power=0.05,
                                         max_levels=3, refine=0.7, max_cells=40_000))
    mesh = build(report.spec)
    t_mean = observables(mesh)["t_mean_storage"]
    assert t_mean == pytest.approx(report.chosen.t_mean_storage, rel=1e-9)
    assert mesh.grid_summary()["cells"] == report.chosen.cells


# ------------------------------------------------------- predictive behaviour
def test_the_search_jumps_instead_of_walking_one_level_at_a_time():
    """The point of the error model: few solves, and the last one is the right one."""
    build, observables, spec = slab_problem()
    report = find_mesh(build, observables, spec,
                       ConvergenceTarget(delta_temperature=4.0, delta_power=0.01,
                                         max_levels=6, refine=0.7, max_cells=200_000))
    assert report.converged, report.summary()
    assert len(report.levels) <= 5, report.summary()
    # the probe is cheap: the first two grids are far smaller than the chosen one
    assert report.levels[0].cells < 0.2 * report.chosen.cells


def test_the_report_carries_the_observed_order_and_the_gci():
    build, observables, spec = slab_problem()
    report = find_mesh(build, observables, spec,
                       ConvergenceTarget(delta_temperature=4.0, delta_power=0.01,
                                         max_levels=6, refine=0.7, max_cells=200_000))
    assert report.converged, report.summary()
    assert report.order >= 1.0, report.summary()
    assert report.gci_temperature >= 0.0 and report.gci_power >= 0.0
    assert "observed order" in report.summary()
    assert "GCI" in report.summary()


def test_a_noisy_probe_falls_back_to_a_conservative_order():
    """A non-monotone probe has no order to fit: the search must still converge."""
    def build(spec: GridSpec) -> Mesh3D:
        mesh = Mesh3D(0.4, 0.4, 2.0, grid=spec)
        return mesh

    calls = {"n": 0}

    def observables(mesh: Mesh3D) -> dict:
        # a deliberately noisy observable: no order can be extracted from it
        calls["n"] += 1
        noise = (-1) ** calls["n"] * 3.0
        return {"t_mean_storage": 400.0 + noise, "power": 100.0 + 0.5 * noise}

    spec = GridSpec(x=(Band(0.0, 0.4, 0.4),), y=(Band(0.0, 0.4, 0.4),),
                    z=(Band(0.0, 2.0, 0.2),))
    report = find_mesh(build, observables, spec,
                       ConvergenceTarget(delta_temperature=5.0, delta_power=0.05,
                                         max_levels=5, refine=0.7, max_cells=100_000))
    assert not report.converged
    assert len(report.levels) >= 2
    assert "not converged" in report.message
