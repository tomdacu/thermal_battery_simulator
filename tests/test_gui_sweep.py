"""Systematic GUI sweep: every control and every action, head-less.

Purpose: turn "clicking around and hoping" into a repeatable check.  Modal
dialogs are neutralised, the 3D view is exercised with a head-less PyVista
plotter, and every widget is driven through its whole range while its slots are
connected - which is where a PyQt6 application normally dies, because an
exception inside a slot aborts the process.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PyQt6")
pv = pytest.importorskip("pyvista")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("THERMAL_DISABLE_3D", "1")

from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QMessageBox,
    QRadioButton,
    QSpinBox,
)

from src.core.mesh import Mesh3D  # noqa: E402

FAILURES: list = []


@pytest.fixture(scope="module")
def app():
    application = QApplication.instance() or QApplication([])
    return application


@pytest.fixture(scope="module", autouse=True)
def capture_qt_messages():
    """Record every Qt message so a framework warning cannot go unnoticed."""
    from PyQt6.QtCore import qInstallMessageHandler

    captured: list = []
    qInstallMessageHandler(lambda mode, context, message: captured.append((mode.name, message)))
    yield captured
    qInstallMessageHandler(None)


@pytest.fixture(scope="module", autouse=True)
def capture_dialogs():
    """Replace the modal dialogs with recorders: they would block a head-less run."""
    recorded, original = [], {}
    for name in ("critical", "warning", "information", "question"):
        original[name] = getattr(QMessageBox, name)
        setattr(QMessageBox, name,
                staticmethod(lambda *a, _n=name, **k: recorded.append((_n, a[2] if len(a) > 2 else ""))))
    yield recorded
    for name, func in original.items():
        setattr(QMessageBox, name, func)


@pytest.fixture(scope="module")
def window(app):
    from gui.main_window import ThermalBatteryGUI

    win = ThermalBatteryGUI()
    win.geometry_panel.refined.setChecked(False)   # small uniform mesh
    # the automatic mesh search is the GUI default but it is a background
    # solve: the tests build the grid they configure, not the one it finds
    win.geometry_panel.auto_first.setChecked(False)
    win.geometry_panel.spacing.setValue(0.5)
    win.build_mesh()
    yield win
    win.close()


def widgets(panel):
    for name in sorted(vars(panel)):
        if name == "auto_first":
            continue        # sweeping it would launch the background mesh search
        widget = getattr(panel, name)
        if isinstance(widget, (QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox, QRadioButton)):
            yield name, widget


def exercise(widget) -> int:
    """Drive one widget through its whole range; returns the number of actions."""
    actions = 0
    if isinstance(widget, QComboBox):
        for index in range(widget.count()):
            widget.setCurrentIndex(index)
            actions += 1
    elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        for value in (widget.minimum(), widget.maximum(), widget.value()):
            widget.setValue(value)
            actions += 1
    elif isinstance(widget, QCheckBox):
        for state in (True, False):
            widget.setChecked(state)
            actions += 1
    elif isinstance(widget, QRadioButton):
        widget.setChecked(True)
        actions += 1
    return actions


def _sweep_all_widgets(window) -> int:
    """Drive every control of every panel through its whole range."""
    actions = 0
    for panel in (window.geometry_panel, window.materials_panel,
                  window.analysis_panel, window.solver_panel):
        for _, widget in widgets(panel):
            actions += exercise(widget)
    return actions


def test_every_widget_can_be_driven_without_aborting(window):
    """Every combo index, spin endpoint and checkbox state, with slots connected."""
    assert _sweep_all_widgets(window) > 100


def test_every_panel_getter_survives_extreme_widget_values(window):
    """Getters answer after the sweep; an incomplete profile explains itself."""
    _sweep_all_widgets(window)
    # the sweep drives every spin box to its maximum: keep the mesh affordable so the
    # run kinds below stay a smoke test
    window.geometry_panel.max_cells.setValue(200_000)
    window.geometry_panel.cells_storage.setValue(6)
    window.geometry_panel.cells_sheath.setValue(1)
    window.geometry_panel.refined.setChecked(False)
    for panel in (window.geometry_panel, window.materials_panel,
                  window.analysis_panel, window.solver_panel):
        try:
            if panel is window.geometry_panel:
                panel.domain(), panel.cylinder(), panel.heaters()
            elif panel is window.materials_panel:
                (panel.storage_key(), panel.insulation_key(), panel.shell_key(),
                 panel.packing_fraction(), panel.conditions(), panel.radiation_enabled())
            elif panel is window.analysis_panel:
                panel.analysis_type(), panel.initial_condition()
                panel.power_profile(), panel.extraction_profile()
                panel.transient_settings(), panel.losses_target_kelvin()
            else:
                panel.settings(), panel.losses_settings(), panel.threads()
        except (ValueError, FileNotFoundError) as exc:
            assert panel is window.analysis_panel, f"{panel} raised {exc}"
            assert any(word in str(exc).lower()
                       for word in ("schedule", "csv", "t_", "kelvin")), exc

    battery = window._battery_from_panels()
    problems = battery.validate(Mesh3D(Lx=6.0, Ly=6.0, Lz=5.6, spacing=0.5))
    assert isinstance(problems, list)          # a list, never an exception
    restore_sane_configuration(window)
    window._run_config()                       # a valid set must assemble cleanly


def restore_sane_configuration(window) -> None:
    """Put the panels back into a configuration that must assemble without errors."""
    geometry, analysis, materials = (window.geometry_panel, window.analysis_panel,
                                     window.materials_panel)
    geometry.refined.setChecked(False)
    geometry.spacing.setValue(0.5)
    geometry.domain_lx.setValue(6.0)
    geometry.domain_ly.setValue(6.0)
    geometry.domain_lz.setValue(5.6)
    geometry.radius.setValue(2.0)
    geometry.height.setValue(4.0)
    geometry.base_z.setValue(0.3)
    geometry.insulation_thickness.setValue(0.3)
    geometry.power.setValue(50.0)
    analysis.power_constant.setChecked(True)
    analysis.power_value.setValue(10000.0)
    analysis.power_unit.setCurrentIndex(1)
    analysis.radio_uniform.setChecked(True)
    analysis.extract_off.setChecked(True)
    analysis.duration.setValue(1.0)
    analysis.duration_unit.setCurrentIndex(2)
    analysis.dt.setValue(600.0)
    analysis.save_interval.setValue(600.0)
    materials.reading_default = True          # no state change, keeps flake happy


def test_every_analysis_type_keeps_the_panels_consistent(window):
    restore_sane_configuration(window)
    for radio in (window.analysis_panel.radio_steady, window.analysis_panel.radio_losses,
                  window.analysis_panel.radio_transient):
        radio.setChecked(True)
        assert window.analysis_panel.analysis_type() in ("steady", "losses", "transient")
        assert "Run" in window.run_btn.text()
        window._run_config()
    window.analysis_panel.radio_steady.setChecked(True)


def test_every_profile_mode_builds_or_explains_itself(window):
    panel = window.analysis_panel
    restore_sane_configuration(window)
    for radio in (panel.power_off, panel.power_constant, panel.power_schedule, panel.power_csv):
        radio.setChecked(True)
        try:
            profile = panel.power_profile()
            assert profile.validate() == [] or profile.mode == "schedule"
        except (ValueError, FileNotFoundError) as exc:
            # an incomplete mode must say *what* is missing, never fail silently
            message = str(exc).lower()
            assert "schedule" in message or "csv" in message
    for radio in (panel.extract_off, panel.extract_power, panel.extract_flow):
        radio.setChecked(True)
        profile = panel.extraction_profile()
        assert profile.validate() == [] or radio is panel.extract_flow
    for radio in (panel.radio_uniform, panel.radio_by_material, panel.radio_from_file,
                  panel.radio_from_steady):
        radio.setChecked(True)
        if radio is panel.radio_from_file:
            with pytest.raises((ValueError, FileNotFoundError)):
                panel.initial_condition().apply_to_mesh(window.mesh)
        else:
            panel.initial_condition()
    panel.radio_uniform.setChecked(True)
    panel.power_constant.setChecked(True)
    panel.extract_off.setChecked(True)


def test_three_d_view_renders_every_field_with_a_headless_plotter(window):
    """The view is wired to src.viz.scene; inject an off-screen plotter to run it."""
    view = window.viz
    plotter = pv.Plotter(off_screen=True, window_size=(320, 240))
    view.plotter = plotter
    try:
        for field in ("Temperature", "Material", "Sources", "Conductivity"):
            view.field_combo.setCurrentIndex(view.field_combo.findData(field))
            for axis in ("x", "y", "z"):
                view.axis_combo.setCurrentIndex(view.axis_combo.findData(axis))
                for fraction in (1, 50, 99):
                    view.slice_slider.setValue(fraction)
                    view.opacity_slider.setValue(60)
        view.show_mesh(window.mesh)
        view.show_geometry(window._battery_from_panels(), window.mesh)
        # the geometry preview must react to the cut and opacity controls
        view.axis_combo.setCurrentIndex(0)
        view.slice_slider.setValue(40)
        view.opacity_slider.setValue(30)
        assert view._mode == "geometry"
        view.show_mesh(window.mesh)
        assert view._mode == "field"
    finally:
        view.plotter = None
        plotter.close()


def test_action_buttons_do_not_raise(window):
    window.geometry_panel.auto_first.setChecked(False)
    window.build_mesh()
    window.preview_geometry()
    window._update_mesh_info()
    window.results.update_statistics(window.mesh)
    window.results.update_materials(window.mesh, "steatite", 0.63, "rock_wool")
    window.results.update_energy(window.mesh)
    window.results.log("sweep")


def test_the_refined_mesh_mode_builds_a_graded_grid(window):
    """The panel targets become the grid: fine storage, a coarse cap outside."""
    panel = window.geometry_panel
    panel.refined.setChecked(True)
    panel.cells_storage.setValue(6)
    panel.cells_insulation.setValue(2)
    panel.cells_sheath.setValue(1)
    panel.max_cells.setValue(60_000)
    mesh = panel.build_mesh()
    assert not mesh.uniform
    assert mesh.dx.min() < mesh.dx.max()
    # the summary describes the grid that was built, whatever it says
    assert str(mesh.N_total)[:2] in panel.mesh_info.text().replace(",", "")
    panel.refined.setChecked(False)
    assert panel.build_mesh().uniform


def test_every_run_kind_executes_through_the_controller(window):
    """Run each analysis through the controller's own job factory, synchronously."""
    restore_sane_configuration(window)
    panel = window.analysis_panel
    for kind, radio in (("steady", panel.radio_steady), ("losses", panel.radio_losses),
                        ("transient", panel.radio_transient)):
        radio.setChecked(True)
        if kind == "losses":
            window.solver_panel.losses_max_iterations.setValue(2)
        if kind == "transient":
            panel.duration.setValue(0.5)
            panel.duration_unit.setCurrentIndex(2)
            panel.dt.setValue(600.0)
            panel.save_interval.setValue(600.0)
        config = window._run_config()
        assert config.analysis == kind
        work = window.controller._make_work(config, window.mesh)
        result = work(lambda *_: None, lambda: False)
        window._on_finished(kind, result)
    panel.radio_steady.setChecked(True)


def test_cancellation_stops_a_transient_run(window):
    restore_sane_configuration(window)
    panel = window.analysis_panel
    panel.radio_transient.setChecked(True)
    panel.duration.setValue(5.0)
    panel.duration_unit.setCurrentIndex(2)
    panel.dt.setValue(60.0)
    config = window._run_config()
    work = window.controller._make_work(config, window.mesh)
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    result = work(lambda *_: None, stop)
    assert len(result) < 50
    panel.radio_steady.setChecked(True)


def test_state_save_and_load_round_trip_through_the_window(window, tmp_path):
    from src.io.state import StateManager

    window.state_manager = StateManager(directory=str(tmp_path))
    window.save_state("sweep_state", "written by the sweep")
    window.analysis_panel.file_path.setText(str(tmp_path / "sweep_state.h5"))
    window.load_state()
    assert "loaded" in window.analysis_panel.state_info.text()


def test_zz_no_qt_warning_came_from_the_application(capture_qt_messages):
    """The sweep must not make Qt complain: warnings there mean real defects.

    The only tolerated message is Qt's own font-directory notice of a head-less
    session, which has nothing to do with this code.
    """
    noisy = [(kind, text) for kind, text in capture_qt_messages
             if kind in ("QtWarningMsg", "QtCriticalMsg", "QtFatalMsg")
             and "QFontDatabase" not in text]
    assert noisy == []
