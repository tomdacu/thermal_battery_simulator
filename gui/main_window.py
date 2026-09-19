"""Main window: assembles the panels, routes the runs, displays the results."""
from __future__ import annotations

import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.analysis.mesh_plan import plan_regions
from src.core.geometry import BatteryGeometry, HeaterPattern
from src.core.materials import MaterialManager
from src.core.mesh import Mesh3D
from src.io.state import StateError, StateManager
from src.viz.scene import export_csv, export_vtk

from .assets import window_icon_path
from .controller import RunConfig, SimulationController
from .safe import safe_slot
from .views.analysis_panel import AnalysisPanel
from .views.geometry_panel import GeometryPanel
from .views.materials_panel import MaterialsPanel
from .views.results_panel import ResultsPanel
from .views.solver_panel import SolverPanel
from .views.viz_view import VizView

HELP_TEXT = """
<b>Workflow</b>
<ol>
<li><b>Geometry</b>: domain, cylinder, insulation, heaters, tubes, cell size.</li>
<li><b>Materials</b>: storage medium, insulation, ambient and ground conditions.</li>
<li><b>Analysis</b>: steady state, losses analysis or transient profiles.</li>
<li><b>Tools &gt; Solver</b>: linear method, preconditioner, threading, radiation.</li>
<li><b>Build mesh</b>, then <b>Run</b>.</li>
</ol>
<b>Units</b>: the interface is in degC, the model works in Kelvin; conversion
happens only at this boundary. Powers are W, lengths m, time s.
<b>Results</b>: statistics, energy balance (envelope losses), materials, time series.
"""


class ThermalBatteryGUI(QMainWindow):
    """Application window: widgets, wiring and result display only."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Thermal Battery Simulator")
        self.resize(1500, 950)
        icon = window_icon_path()
        if icon is not None:
            self.setWindowIcon(QIcon(str(icon)))
        self.mesh: Mesh3D = None
        self._auto_tried = False          # the search runs once per configuration
        self._build_after_search = False
        self.battery = BatteryGeometry()
        self.state_manager = StateManager()
        self.controller = SimulationController(self)
        self._last_transient = None

        self.geometry_panel = GeometryPanel()
        self.materials_panel = MaterialsPanel()
        self.analysis_panel = AnalysisPanel()
        self.solver_panel = SolverPanel()
        self.viz = VizView()
        self.results = ResultsPanel()

        self._build_layout()
        self._connect()
        self._update_mesh_info()
        self.log("Ready. Configure the geometry and press 'Build mesh'.")

    # ------------------------------------------------------------- layout
    def _build_layout(self) -> None:
        left = QWidget()
        column = QVBoxLayout(left)
        self.left_tabs = QTabWidget()
        self.left_tabs.addTab(self.geometry_panel, "1. Geometry")
        self.left_tabs.addTab(self.materials_panel, "2. Materials")
        self.left_tabs.addTab(self.analysis_panel, "3. Analysis")
        tools = QTabWidget()
        tools.addTab(self.solver_panel, "Solver")
        help_view = QWidget()
        help_layout = QVBoxLayout(help_view)
        from PyQt6.QtWidgets import QTextEdit

        text = QTextEdit(HELP_TEXT)
        text.setReadOnly(True)
        help_layout.addWidget(text)
        tools.addTab(help_view, "Help")
        self.left_tabs.addTab(tools, "4. Tools")
        column.addWidget(self.left_tabs)

        buttons = QHBoxLayout()
        self.build_btn = QPushButton("Build mesh")
        self.preview_btn = QPushButton("Preview geometry")
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        for btn in (self.build_btn, self.preview_btn, self.run_btn, self.cancel_btn):
            buttons.addWidget(btn)
        column.addLayout(buttons)
        self.progress = QProgressBar()
        column.addWidget(self.progress)
        left.setMaximumWidth(560)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.viz)
        splitter.addWidget(self.results)
        splitter.setSizes([520, 560, 420])
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")

    def _connect(self) -> None:
        self.build_btn.clicked.connect(self.build_mesh)
        self.geometry_panel.auto_mesh_requested.connect(self.find_auto_mesh)
        self.preview_btn.clicked.connect(self.preview_geometry)
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn.clicked.connect(self.controller.cancel)
        self.geometry_panel.mesh_changed.connect(self._update_mesh_info)
        self.geometry_panel.preview_requested.connect(self.preview_elements)
        self.materials_panel.storage_material.currentIndexChanged.connect(
            self.materials_panel.refresh_info)
        self.materials_panel.insulation_material.currentIndexChanged.connect(
            self.materials_panel.refresh_info)
        self.analysis_panel.save_requested.connect(self.save_state)
        self.analysis_panel.load_requested.connect(self.load_state)
        self.analysis_panel.analysis_changed.connect(self._on_analysis_changed)
        self.controller.progressed.connect(self._on_progress)
        self.controller.log.connect(self.log)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.controller.running_changed.connect(self._on_running)

    # -------------------------------------------------------------- actions
    @safe_slot
    def _battery_from_panels(self) -> BatteryGeometry:
        battery = BatteryGeometry()
        self.geometry_panel.apply_geometry(battery)
        battery.storage_material = self.materials_panel.storage_key()
        battery.insulation_material = self.materials_panel.insulation_key()
        battery.shell_material = self.materials_panel.shell_key()
        battery.packing_fraction = self.materials_panel.packing_fraction()
        for key, value in self.materials_panel.conditions().items():
            setattr(battery, key, value)
        return battery

    def _radiation(self) -> bool:
        """The radiation switch of the solver panel (the balance must match it)."""
        return bool(self.solver_panel.settings().get("radiation", False))

    def _refresh_plan(self) -> None:
        """Recompute the a priori mesh plan from the current geometry and materials."""
        try:
            battery = self._battery_from_panels()
        except ValueError:
            return
        plans = plan_regions(battery, MaterialManager())
        self.geometry_panel.set_plan_targets(
            {plan.name: plan.target for plan in plans},
            " | ".join(f"{plan.name} {plan.target * 1000:.0f} mm" for plan in plans))

    @safe_slot
    def build_mesh(self) -> None:
        self._refresh_plan()
        if (self.geometry_panel.wants_auto_search()
                and self.geometry_panel.auto_spec() is None and not self._auto_tried):
            self.log("[mesh] automatic mesh: searching before the build")
            self.find_auto_mesh(build_after=True)
            return
        try:
            battery = self._battery_from_panels()
            mesh = self.geometry_panel.build_mesh()
            report = battery.apply_to_mesh(mesh)
        except ValueError as exc:
            QMessageBox.critical(self, "Geometry error", str(exc))
            return
        self.battery, self.mesh = battery, mesh
        self.log(f"[mesh] {mesh.Nx}x{mesh.Ny}x{mesh.Nz} = {mesh.N_total:,} cells "
                 f"({mesh.size_label()})")
        for note in report.notes:
            self.log(f"[mesh] {note}")
        self.log(f"[mesh] {report.n_source_cells} source cells, "
                 f"{report.n_tube_cells} tube cells")
        self._update_mesh_info()
        self.viz.show_mesh(mesh)
        self.statusBar().showMessage("Mesh ready")

    @safe_slot
    def preview_geometry(self) -> None:
        try:
            self.battery = self._battery_from_panels()
        except ValueError as exc:
            QMessageBox.critical(self, "Geometry error", str(exc))
            return
        self.viz.show_geometry(self.battery, self.mesh)

    @safe_slot
    def preview_elements(self) -> None:
        try:
            battery = self._battery_from_panels()
        except ValueError as exc:
            QMessageBox.critical(self, "Geometry error", str(exc))
            return
        cyl = battery.cylinder
        panel = self.geometry_panel
        if battery.heaters.pattern == HeaterPattern.UNIFORM_ZONE:
            panel.heater_positions.clear()
            panel.heater_positions.addItem("uniform zone: the power is spread over the "
                                           "whole storage volume (no discrete element)")
        else:
            bank = battery.heaters.bank(cyl.z_storage_start, cyl.z_storage_end)
            heaters = bank.generate_elements(cyl.center_x, cyl.center_y,
                                             cyl.r_storage * 0.9)
            panel.heater_positions.clear()
            for index, element in enumerate(heaters):
                panel.heater_positions.addItem(
                    f"{index}: ({element.center_x:.2f}, {element.center_y:.2f}) "
                    f"legs {element.leg_spacing * 1000:.0f} mm  "
                    f"{element.rated_power / 1000:.2f} kW  "
                    f"{element.surface_power_w_cm2:.1f} W/cm2")
            panel.heater_positions.addItem(
                f"total {len(heaters)} hairpin elements, "
                f"{bank.total_power_w / 1000:.1f} kW, "
                f"{bank.surface_power_w_cm2():.1f} W/cm2")
        tubes = battery.tubes.generate_positions(cyl.center_x, cyl.center_y,
                                                 cyl.r_storage * 0.9,
                                                 cyl.z_storage_start, cyl.z_storage_end)
        panel.tube_positions.clear()
        for element in tubes:
            panel.tube_positions.addItem(
                f"({element.x:.2f}, {element.y:.2f}) r {element.radius:.3f} m  "
                f"h {element.h_fluid:.0f} W/(m²·K)")
        panel.tube_positions.addItem(f"total {len(tubes)} tubes")

    @safe_slot
    def find_auto_mesh(self, build_after: bool = False) -> None:
        """Search the mesh that makes the steady answer converge (background thread)."""
        self._refresh_plan()
        if self.controller.running:
            self.log("[automesh] a simulation is already running")
            return
        try:
            self.battery = self._battery_from_panels()
        except ValueError as exc:
            QMessageBox.critical(self, "Geometry error", str(exc))
            return
        if not self.geometry_panel.refined.isChecked():
            self.log("[automesh] the search needs the refined mesh mode")
            return
        self.mesh = None
        self._auto_tried = True
        self._build_after_search = bool(build_after)
        self.geometry_panel.set_auto_spec(None, "searching...")
        config = self._run_config()
        config.analysis = "automesh"
        self.controller.start(config, None)

    def _run_config(self) -> RunConfig:
        return RunConfig(
            analysis=self.analysis_panel.analysis_type(),
            battery=self.battery,
            domain=self.geometry_panel.domain(),
            mesh_spec=self.geometry_panel.grid_spec()
            if self.geometry_panel.refined.isChecked() else None,
            convergence=self.geometry_panel.auto_mesh_settings(),
            n_threads=self.solver_panel.threads(),
            losses={"t_target": self.analysis_panel.losses_target_kelvin(),
                    "t_ambient": self.battery.t_ambient,
                    "t_ground": self.battery.t_ground,
                    **self.solver_panel.losses_settings()},
            transient=self.analysis_panel.transient_settings(),
            initial_condition=self.analysis_panel.initial_condition(),
            power_profile=self.analysis_panel.power_profile(),
            extraction_profile=self.analysis_panel.extraction_profile(),
            start_from_steady=self.analysis_panel.wants_steady_initial_condition(),
            **self.solver_panel.settings(),
        )

    @safe_slot
    def run(self) -> None:
        if self.mesh is None:
            QMessageBox.warning(self, "No mesh", "Build the mesh first.")
            return
        try:
            # the materials, the packing and the ambient may have changed after the
            # build: the run must use what the panels say now, not what was frozen
            self.battery = self._battery_from_panels()
            config = self._run_config()
        except (ValueError, FileNotFoundError) as exc:
            QMessageBox.critical(self, "Invalid configuration", str(exc))
            return
        self.results.log_text.clear()
        self.controller.start(config, self.mesh)

    # ---------------------------------------------------------------- state
    @safe_slot
    def _geometry_params(self) -> dict:
        """Parameters that must match between a saved state and the current model."""
        cyl = self.battery.cylinder
        return {"r_storage": cyl.r_storage, "height": cyl.height, "base_z": cyl.base_z,
                "insulation_thickness": cyl.insulation_thickness,
                "roof_angle_deg": cyl.roof_angle_deg,
                "packing_fraction": self.battery.packing_fraction,
                "storage_material": self.materials_panel.storage_key()}

    def save_state(self, name: str, description: str) -> None:
        if self.mesh is None:
            QMessageBox.warning(self, "No mesh", "Build the mesh first.")
            return
        path = self.state_manager.save_state(
            self.mesh, name=name, geometry_params=self._geometry_params(),
            notes=[description] if description else None)
        self.analysis_panel.state_info.setText(f"saved: {path}")
        self.log(f"[state] saved {path}")

    @safe_slot
    def load_state(self) -> None:
        if self.controller.running:
            QMessageBox.warning(self, "Simulation running",
                                "Stop the run before loading a state: the solver is "
                                "reading the same fields.")
            return
        if self.mesh is None:
            QMessageBox.warning(self, "No mesh", "Build the mesh first.")
            return
        path = self.analysis_panel.file_path.text()
        try:
            state = self.state_manager.load_state(path)
            notes = self.state_manager.apply(self.mesh, state,
                                             geometry_params=self._geometry_params())
        except StateError as exc:
            QMessageBox.critical(self, "Cannot load state", str(exc))
            return
        self.analysis_panel.state_info.setText(f"loaded: {path}")
        for note in notes:
            self.log(f"[state] {note}")
        self._refresh_results(recompute_balance=True)

    # --------------------------------------------------------------- signals
    @safe_slot
    def _on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(max(0, min(value, 100)))
        self.statusBar().showMessage(message)

    @safe_slot
    def _on_running(self, running: bool) -> None:
        for btn in (self.build_btn, self.run_btn, self.preview_btn):
            btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if not running:
            self.progress.setValue(100 if self.progress.value() > 5 else 0)
        self.statusBar().showMessage("running..." if running else "Ready")

    @safe_slot
    def _on_failed(self, message: str) -> None:
        self.statusBar().showMessage("failed")
        QMessageBox.critical(self, "Simulation failed", message)

    @safe_slot
    def _on_finished(self, analysis: str, result) -> None:
        if analysis == "automesh":
            state = "converged" if result.converged else "stopped"
            self.geometry_panel.set_auto_spec(
                result.spec if result.converged else None,
                f"{state}: {result.chosen.cells:,} cells, "
                f"dT {result.chosen.d_temperature:.2f} K, "
                f"dP {100 * result.chosen.d_power:.2f}%")
            if result.converged:
                self.log("[automesh] mesh adopted")
            else:
                QMessageBox.warning(self, "Automatic mesh", result.message)
            self.statusBar().showMessage(f"automesh {state}")
            if self._build_after_search:
                self._build_after_search = False
                self.build_mesh()
            return
        if analysis == "losses":
            self.log(f"[losses] converged={result.converged} iterations={result.iterations} "
                     f"power={result.power / 1000:.2f} kW")
            self.results.update_energy(self.mesh, losses=result,
                                       ambient=self.battery.t_ambient,
                                       radiation=self._radiation())
        elif analysis == "transient":
            self._last_transient = result
            summary = result.summary()
            self.log(f"[transient] {len(result)} samples, "
                     f"T_mean_end={summary.get('T_mean_storage_end', float('nan')) - 273.15:.1f} °C")
            self.results.update_transient(result)
            self.results.update_energy(self.mesh, transient=result,
                                       ambient=self.battery.t_ambient,
                                       radiation=self._radiation())
        else:
            self.log(f"[steady] converged={result.converged} residual={result.residual:.2e}")
            self.results.update_energy(self.mesh, ambient=self.battery.t_ambient,
                                       radiation=self._radiation())
        self._refresh_results()
        self.statusBar().showMessage(f"{analysis} finished")

    @safe_slot
    def _refresh_results(self, recompute_balance: bool = False) -> None:
        if self.mesh is None:
            return
        self.results.update_statistics(self.mesh)
        self.results.update_materials(self.mesh, self.materials_panel.storage_key(),
                                      self.materials_panel.packing_fraction(),
                                      self.materials_panel.insulation_key())
        if recompute_balance:
            self.results.update_energy(self.mesh, ambient=self.battery.t_ambient,
                                       radiation=self._radiation())
        self.viz.show_mesh(self.mesh)
        _ = recompute_balance

    @safe_slot
    def _on_analysis_changed(self, analysis: str) -> None:
        self.run_btn.setText({"steady": "Run steady state",
                              "losses": "Run losses analysis",
                              "transient": "Run transient"}[analysis])

    @safe_slot
    def _update_mesh_info(self) -> None:
        try:
            self.geometry_panel._update_mesh_summary()
        except ValueError as exc:
            self.geometry_panel.set_mesh_info(f"invalid: {exc}", "-")

    @safe_slot
    def log(self, message: str) -> None:
        self.results.log(message)

    # --------------------------------------------------------------- exports
    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self.controller.running:
            self.controller.cancel()
            # a QThread destroyed while it runs aborts the process: give the worker
            # the time to notice the cancel flag and unwind
            self.controller.wait(5000)
        super().closeEvent(event)

    @safe_slot
    def export_vtk(self) -> None:
        if self.mesh is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export VTK", "results/field.vti",
                                              "VTK (*.vti)")
        if path:
            export_vtk(self.mesh, path)
            self.log(f"[export] {path}")

    @safe_slot
    def export_csv_field(self) -> None:
        if self.mesh is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export CSV", "results/field.csv",
                                              "CSV (*.csv)")
        if path:
            export_csv(path, None, mesh=self.mesh)
            self.log(f"[export] {path}")


def main() -> int:
    app = QApplication(sys.argv)
    window = ThermalBatteryGUI()
    window.show()
    return app.exec()
