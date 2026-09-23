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
    # the smallest budget the Mesh tab offers: the coarsest tree the GUI can build
    win.geometry_panel.max_cells.setValue(10_000)
    yield win
    win.close()
    del app


def test_window_builds_the_mesh_with_the_panel_defaults(window):
    window.build_mesh()
    mesh = window.mesh
    assert mesh is not None
    assert mesh.source_mask.any()
    assert mesh.T.min() > 100.0                      # Kelvin, not Celsius
    assert "leaves" in window.geometry_panel.mesh_info.text()
    # the build paints the plant's network: the gas loop is the run's heat path
    assert window.geometry_panel.pipe_network() is not None


def test_run_config_is_read_from_the_widgets(window):
    window.build_mesh()
    config = window._run_config()
    assert config.analysis in ("standby", "transient")
    assert (config.method, config.preconditioner) == ("cg", "amg_rs")
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
    window.geometry_panel.power.setValue(100.0)
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
    """A domain that cannot host the roof must raise, not silently cut it; the panel's
    own domain always holds the vessel (it is derived from it)."""
    from src.core.mesh import Mesh3D

    battery = window._battery_from_panels()
    assert battery is not None
    problems = battery.validate(Mesh3D(Lx=6.5, Ly=6.5, Lz=2.0, spacing=0.5))
    assert any("Lz" in problem for problem in problems), (
        f"apex={battery.cylinder.z_cone_apex:.2f} problems={problems}")
    lx, ly, lz = window.geometry_panel.domain()
    assert battery.validate(Mesh3D(Lx=lx, Ly=ly, Lz=lz, spacing=0.5)) == []
    cyl = battery.cylinder
    assert cyl.center_x == cyl.center_y == 0.5 * lx      # the vessel is centred


def test_the_window_solves_the_standby_case_on_a_tree(window):
    """The Mesh tab's adaptive mode, end to end through the controller.

    The whole GUI road on a tree, because none of it may assume a grid of cells per axis:
    the factory of the Mesh tab builds the octree of the same bands, the window paints the
    battery on it, the controller's worker solves it in the background, and the results
    panel reports it.  The budget is the smallest the panel offers, so the tree is the
    coarsest one the GUI can build.
    """
    from PyQt6.QtWidgets import QApplication

    from src.core.adaptive_mesh import AdaptiveMesh

    panel = window.geometry_panel
    try:
        window.build_mesh()
        mesh = window.mesh
        assert isinstance(mesh, AdaptiveMesh)
        assert mesh.source_mask.any()
        assert mesh.T.min() > 100.0                      # Kelvin, not Celsius
        assert "leaves" in panel.mesh_info.text()

        config = window._run_config()
        config.analysis = "standby"
        window.controller.start(config, mesh)
        assert window.controller.wait(600_000), "the standby run did not finish"
        QApplication.processEvents()                     # deliver the queued finish

        stats = window.results.stats_text.toPlainText()
        energy = window.results.energy_text.toPlainText()
        assert "Temperature" in stats and "degC" in stats
        assert "leaves" in stats and "domain" in stats
        assert "losses (envelope)" in energy and "imbalance" in energy
        # the standby state is the plant's: the circuit carries the holding power into
        # the bed, and the storage sits at the temperature it was asked to hold
        assert "Gas loop" in energy and "Steady standby" in energy
        loop = window.controller.last_loop
        assert loop is not None and loop.power > 0.0
        assert "holding power" in energy
    finally:
        panel.max_cells.setValue(10_000)


def test_the_automatic_search_refines_a_tree(window):
    """The search on the adaptive mode builds and refines trees, and adopts the plan.

    The Mesh tab hands the search an :class:`AdaptivePlan` when it builds a tree (the same
    bands as boxes of an octree), and the controller's factory has to build the mesh the
    request describes - a tree, refined one round per level - instead of a ``Mesh3D`` from
    a grid spec.  The plan here is deliberately coarse and well inside the budget, so the
    search refines it in the time a test can afford; it is handed over as the mesh request
    of the run, which is where the window puts the plan when the tab is adaptive.
    """
    from PyQt6.QtWidgets import QApplication

    from src.analysis.convergence import AdaptivePlan
    from src.analysis.mesh_plan import region_bands
    from src.core.adaptive_mesh import AdaptiveMesh

    panel = window.geometry_panel
    # the manual targets alone: the a priori plan of a built mesh asks for millimetres at
    # the shell, which a 375 mm floor could never refine towards
    panel.set_plan_targets({})
    panel.cells_sheath.setValue(2)          # the plan this test was sized on: 25 mm pipes
    plan = AdaptivePlan(n_finest=16, physical_size=0.4375,   # a 7 m box: the default vessel
                        bands=region_bands(panel.mesh_regions())).scaled(20.0)
    try:
        assert isinstance(window._run_config().mesh_spec, AdaptivePlan)
        window.battery = window._battery_from_panels()
        config = window._run_config()
        config.analysis = "automesh"
        config.mesh_spec = plan
        config.convergence = {"delta_temperature": 1e6, "delta_power": 1e3,
                              "max_levels": 4, "refine": 0.5, "probe_levels": 2,
                              "max_cells": 10_000}
        window.controller.start(config, None)
        assert window.controller.wait(600_000), "the search did not finish"
        QApplication.processEvents()

        assert "converged" in panel.auto_result.text()
        assert isinstance(panel.auto_spec(), AdaptivePlan)
        # the budget caps the build now: give the adopted plan room to be realised
        panel.max_cells.blockSignals(True)
        panel.max_cells.setValue(200_000)
        panel.max_cells.blockSignals(False)
        window.build_mesh()
        assert isinstance(window.mesh, AdaptiveMesh)
        assert window.mesh.n_cells > 512, "the rounds must have refined the plan"
    finally:
        panel.set_auto_spec(None)
        panel.max_cells.setValue(10_000)
        panel.cells_sheath.setValue(1)


def test_the_window_fits_the_screen_and_explanations_take_no_room(window):
    """Opened on the screen it fits, every page scrolls, every explanation is a tooltip.

    The owner's review: the window was taller than the desktop and paragraphs of help cut
    the forms.  The window is sized on the screen's free area, every left page sits in a
    scroll area, a labelled control with an explanation says so with the info mark, and
    no wrapped paragraph of help is left in a form.
    """
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication, QFormLayout, QLabel, QScrollArea

    from gui.widgets import INFO

    area = (window.screen() or QApplication.primaryScreen()).availableGeometry()
    assert window.height() <= area.height()
    tabs = window.left_tabs
    assert [tabs.tabText(i) for i in range(tabs.count())] == [
        "Vessel", "Plant", "Site", "Mesh", "Analysis", "Solver"]
    for index in range(tabs.count()):
        if tabs.tabText(index) != "Analysis":
            assert isinstance(tabs.widget(index), QScrollArea)
    for form in window.findChildren(QFormLayout):
        for row in range(form.rowCount()):
            label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
            field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
            if label is None or field is None or label.widget() is None:
                continue
            caption, control = label.widget(), field.widget()
            if isinstance(caption, QLabel) and control is not None and control.toolTip():
                assert INFO in caption.text(), caption.text()
    # a read-out (selectable: the network summary, the mesh counts) may be long; a
    # paragraph of explanation may not
    readout = Qt.TextInteractionFlag.TextSelectableByMouse
    help_paragraphs = [label for label in window.findChildren(QLabel)
                       if label.wordWrap() and len(label.text()) > 200
                       and not label.textInteractionFlags() & readout]
    assert help_paragraphs == [], [label.text()[:120] for label in help_paragraphs]
