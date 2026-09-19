"""GUI smoke test: the window must build a mesh and survive a full run cycle.

Runs head-less; skipped when PyQt6 is not installed.  The 3D view is disabled
unless a display is available, so the test exercises the panels, the controller
and the result wiring without needing OpenGL.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("THERMAL_DISABLE_3D", "1")


@pytest.fixture(scope="module", autouse=True)
def no_dialogs():
    """Modal dialogs would block a head-less run: replace them with recorders."""
    from PyQt6.QtWidgets import QMessageBox

    recorded = []
    original = {name: getattr(QMessageBox, name) for name in
                ("critical", "warning", "information", "question")}
    for name in original:
        setattr(QMessageBox, name,
                staticmethod(lambda *a, _n=name, **k: recorded.append((_n, a[2] if len(a) > 2 else ""))))
    yield recorded
    for name, func in original.items():
        setattr(QMessageBox, name, func)


@pytest.fixture(scope="module")
def window():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    from gui.main_window import ThermalBatteryGUI

    win = ThermalBatteryGUI()
    win.geometry_panel.refined.setChecked(False)   # small uniform mesh
    # the automatic mesh search is the GUI default but it is a background
    # solve: the tests build the grid they configure, not the one it finds
    win.geometry_panel.auto_first.setChecked(False)
    win.geometry_panel.spacing.setValue(0.5)
    yield win
    win.close()
    del app


def test_window_builds_the_mesh_with_the_panel_defaults(window):
    window.build_mesh()
    mesh = window.mesh
    assert mesh is not None
    assert mesh.source_mask.any()
    assert mesh.T.min() > 100.0                      # Kelvin, not Celsius
    assert "cells" in window.geometry_panel.mesh_info.text()


def test_run_config_is_read_from_the_widgets(window):
    window.build_mesh()
    config = window._run_config()
    assert config.analysis in ("steady", "losses", "transient")
    assert config.method in ("cg", "bicgstab", "gmres", "direct")
    assert 0 < config.tolerance <= 1e-4
    assert config.battery.t_ambient > 200.0          # the degC value is converted
    assert config.battery.heaters.power_w == pytest.approx(
        window.geometry_panel.power.value() * 1000.0)


def test_steady_run_fills_every_result_panel(window):
    from src.solver.steady import SteadyStateSolver

    window.build_mesh()
    result = SteadyStateSolver(window.mesh, window._run_config().solver_config()).solve()
    window._on_finished("steady", result)
    stats = window.results.stats_text.toPlainText()
    energy = window.results.energy_text.toPlainText()
    assert "Temperature" in stats and "degC" in stats
    assert "losses (envelope)" in energy and "imbalance" in energy
    assert "storage" in window.results.materials_text.toPlainText()
    assert result.converged


def test_transient_run_produces_a_time_series(window):
    from src.solver.transient import TransientConfig, TransientSolver

    window.build_mesh()
    window.analysis_panel.duration.setValue(4.0)
    window.analysis_panel.duration_unit.setCurrentIndex(2)      # hours
    window.analysis_panel.dt.setValue(600.0)
    window.analysis_panel.save_interval.setValue(1800.0)
    window.analysis_panel.power_constant.setChecked(True)
    window.analysis_panel.power_value.setValue(100.0)
    config = window._run_config()
    assert config.transient["t_final"] == pytest.approx(4 * 3600.0)

    transient = TransientConfig(initial_condition=config.initial_condition,
                                power_profile=config.power_profile,
                                extraction_profile=config.extraction_profile,
                                t_ambient=config.battery.t_ambient,
                                **config.transient)
    results = TransientSolver(window.mesh, transient, config.solver_config()).run()
    window._on_finished("transient", results)
    assert len(results) >= 4
    assert results.T_mean_storage[-1] > results.T_mean_storage[0]
    assert "t [s]" in window.results.transient_text.toPlainText()
    assert results.E_in_cumulative[-1] > 0


def test_invalid_geometry_is_refused_without_clipping(window):
    """A domain that cannot host the roof must raise, not silently cut it."""
    from src.core.mesh import Mesh3D

    window.geometry_panel.domain_lz.setValue(2.0)
    battery = window._battery_from_panels()
    assert battery is not None
    problems = battery.validate(Mesh3D(Lx=6.0, Ly=6.0, Lz=2.0, spacing=0.5))
    assert any("Lz" in problem for problem in problems), (
        f"apex={battery.cylinder.z_cone_apex:.2f} problems={problems}")
    window.geometry_panel.domain_lz.setValue(5.6)
    battery = window._battery_from_panels()
    assert battery.validate(Mesh3D(Lx=6.0, Ly=6.0, Lz=5.6, spacing=0.5)) == []
