"""Main window: assembles the panels, routes the runs, displays the results."""
from __future__ import annotations

import sys
from typing import TYPE_CHECKING


from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QFileDialog,
    QGridLayout,
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
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import BatteryGeometry
from src.core.materials import MaterialManager
from src.core.pipe_network import build_pipe_network
from src.io.state import StateError, StateManager
from src.viz.scene import export_vtk, grid_lines

from .assets import window_icon_path
from .controller import RunConfig, SimulationController
from .safe import safe_slot
from .views.analysis_panel import AnalysisPanel
from .views.geometry_panel import GeometryPanel, nearest_leaf
from .views.materials_panel import MaterialsPanel
from .views.results_panel import ResultsPanel
from .views.solver_panel import SolverPanel
from .views.viz_view import VizView
from .widgets import scrollable

if TYPE_CHECKING:
    from src.analysis.convergence import AdaptivePlan

HELP_TEXT = """
<b>The plant</b>: electric resistors heat a gas in a closed circuit; the gas runs
through pipes buried in the sand and hands the heat to the bed across the pipe
walls.  On discharge an exchanger on the same circuit takes the heat back out.
<ol>
<li><b>Vessel</b>: the storage bed, its insulation and shell, their materials.</li>
<li><b>Plant</b>: the gas circuit (rated power, gas, flow, pressure, blower) and the
buried pipes.  The preview follows every edit.</li>
<li><b>Site</b>: ambient air, ground, wind.</li>
<li><b>Mesh</b>: the octree; <b>Build mesh</b> builds it and paints the pipes.</li>
<li><b>Analysis</b>: steady standby (hold the bed at T, get the losses) or a
transient (Charge = the resistors, Discharge = the exchanger); then <b>Run</b>.</li>
</ol>
Every control with an &#9432; explains itself on hover.  The interface is in degC,
the model in Kelvin; powers are W or kW, lengths m, times s.
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
        self.mesh: AdaptiveMesh | None = None
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
        """Six pages, one per thing: the vessel, the plant, the site, the mesh, the run
        and the solver - each scrolls instead of being cut by a short screen."""
        left = QWidget()
        column = QVBoxLayout(left)
        column.setContentsMargins(4, 4, 4, 4)
        self.left_tabs = QTabWidget()
        self.left_tabs.setUsesScrollButtons(True)
        vessel = QWidget()
        vessel_layout = QVBoxLayout(vessel)
        vessel_layout.setContentsMargins(0, 0, 0, 0)
        vessel_layout.addWidget(self.geometry_panel.vessel_page)
        vessel_layout.addWidget(self.materials_panel.materials_page)
        vessel_layout.addStretch(1)
        for page, title, tip in (
                (vessel, "Vessel", "The storage bed, its insulation and shell, their "
                                   "materials"),
                (self.geometry_panel.plant_page, "Plant", "The gas circuit and the buried "
                                                          "pipes"),
                (self.materials_panel.site_page, "Site", "Ambient air, ground and wind"),
                (self.geometry_panel.mesh_page, "Mesh", "The octree and the automatic "
                                                        "search"),
                (self.solver_panel.page, "Solver", "Accuracy, threads, radiation")):
            index = self.left_tabs.addTab(scrollable(page), title)
            self.left_tabs.setTabToolTip(index, tip)
        self.left_tabs.insertTab(4, self.analysis_panel, "Analysis")
        self.left_tabs.setTabToolTip(4, "Standby or transient, charge, discharge, states")
        column.addWidget(self.left_tabs, 1)

        buttons = QGridLayout()
        self.build_btn = QPushButton("Build mesh")
        self.preview_btn = QPushButton("Show geometry")
        self.run_btn = QPushButton("Run")
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        for index, btn in enumerate((self.build_btn, self.preview_btn, self.run_btn,
                                     self.cancel_btn)):
            buttons.addWidget(btn, index // 2, index % 2)
        column.addLayout(buttons)
        self.progress = QProgressBar()
        column.addWidget(self.progress)
        left.setMinimumWidth(360)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.viz)
        splitter.addWidget(self.results)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Ready")
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Workflow", self._show_help)
        self._fit_to_screen(splitter)

    def _fit_to_screen(self, splitter: QSplitter) -> None:
        """Open inside the screen's free area, whatever its size, and share it out."""
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1400, 850)
            splitter.setSizes([440, 560, 400])
            return
        area = screen.availableGeometry()
        width = int(min(1500, 0.95 * area.width()))
        height = int(min(950, 0.90 * area.height()))
        self.resize(width, height)
        self.move(area.x() + (area.width() - width) // 2,
                  area.y() + (area.height() - height) // 2)
        splitter.setSizes([int(0.32 * width), int(0.42 * width), int(0.26 * width)])

    def _show_help(self) -> None:
        QMessageBox.information(self, "Workflow", HELP_TEXT)

    def _connect(self) -> None:
        self.build_btn.clicked.connect(self.build_mesh)
        self.geometry_panel.auto_mesh_requested.connect(self.find_auto_mesh)
        self.geometry_panel.pipe_network_requested.connect(self.build_pipe_network)
        self.preview_btn.clicked.connect(self.preview_geometry)
        self.run_btn.clicked.connect(self.run)
        self.cancel_btn.clicked.connect(self.controller.cancel)
        self.geometry_panel.mesh_changed.connect(self._update_mesh_info)
        # the vessel, the circuit and the pipes redraw the preview as they change: a short
        # timer folds a burst of edits (a spin box held down) into one redraw
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(250)
        self._preview_timer.timeout.connect(self._geometry_edited)
        for page in (self.geometry_panel.vessel_page, self.geometry_panel.plant_page):
            for widget in page.findChildren((QDoubleSpinBox, QSpinBox)):
                widget.valueChanged.connect(self._preview_timer.start)
            for widget in page.findChildren(QComboBox):
                widget.currentIndexChanged.connect(self._preview_timer.start)
            for widget in page.findChildren(QCheckBox):
                widget.toggled.connect(self._preview_timer.start)
        self.analysis_panel.save_requested.connect(self.save_state)
        self.analysis_panel.load_requested.connect(self.load_state)
        self.analysis_panel.analysis_changed.connect(self._on_analysis_changed)
        self.results.export_requested.connect(self.export_vtk)
        self.controller.progressed.connect(self._on_progress)
        self.controller.log.connect(self.log)
        self.controller.finished.connect(self._on_finished)
        self.controller.failed.connect(self._on_failed)
        self.controller.running_changed.connect(self._on_running)

    # -------------------------------------------------------------- actions
    def _battery_from_panels(self) -> BatteryGeometry:
        battery = BatteryGeometry()
        self.geometry_panel.apply_geometry(battery)
        battery.storage_material = self.materials_panel.storage_key()
        battery.insulation_material = self.materials_panel.insulation_key()
        battery.shell_material = self.materials_panel.shell_key()
        battery.packing_fraction = self.materials_panel.packing_fraction()
        battery.particle_diameter = self.materials_panel.particle_diameter()
        for key, value in self.materials_panel.conditions().items():
            setattr(battery, key, value)
        return battery

    def _radiation(self) -> bool:
        """The radiation switch of the solver panel (the balance must match it)."""
        return bool(self.solver_panel.settings().get("radiation", False))

    def _refresh_plan(self) -> None:
        """Recompute the a priori mesh plan from the current geometry and materials.

        The plan is the *active* model's: the sand, the insulation and the shell.  The
        region the pipes need is added from the network's own diameter, because that is
        the surface the power crosses and the mesh has to represent it.
        """
        try:
            battery = self._battery_from_panels()
        except ValueError:
            return
        # the counts of the Mesh tab are the layer rule; the plan adds what a count
        # cannot know - a surface film finer than the count (one cell per layer here,
        # so only a film can bind)
        plans = plan_regions(battery, MaterialManager(), cells_per_layer=1.0)
        panel = self.geometry_panel
        panel.set_plan_targets({plan.name: plan.target for plan in plans})
        # the leaf each region gets, not the size it asked for: leaves are powers of two
        plan = panel.adaptive_plan()
        dz = plan.dz if plan.anisotropic else plan.physical_size
        leaves: dict[str, tuple[float, float]] = {}
        for region in panel.mesh_regions():
            leaves[region.name] = (nearest_leaf(region.target, plan.physical_size),
                                   nearest_leaf(region.height, dz))
        names = {"sand": "sand", "slab_bottom": "bottom slab", "slab_top": "top slab",
                 "insulation_radial": "insulation", "shell": "shell", "roof": "roof",
                 "foundation": "foundation", "ground": "soil"}
        panel.plan_info.setText(", ".join(
            f"{names.get(name, name)} {a * 1000:.0f}x{b * 1000:.0f} mm"
            for name, (a, b) in leaves.items()))

    @safe_slot
    def build_mesh(self) -> None:
        if self.controller.running:
            self.log("[mesh] a simulation is running: the mesh is being read")
            return
        self._refresh_plan()
        try:
            battery = self._battery_from_panels()
            mesh = self.geometry_panel.build_mesh()
            report = battery.apply_to_mesh(mesh)
        except ValueError as exc:
            QMessageBox.critical(self, "Geometry error", str(exc))
            return
        self.battery, self.mesh = battery, mesh
        self.log("[mesh] " + "; ".join(line.strip() for line in grid_lines(mesh)))
        for note in report.notes:
            self.log(f"[mesh] {note}")
        film = getattr(battery, "film", None)
        if film:
            self.log(f"[mesh] outer film {film['total']:.2f} W/(m2 K) "
                     f"(natural {film['natural']:.2f} + wind {film['wind']:.2f}) "
                     f"at {battery.t_ambient - 273.15:.1f} degC ambient")
        # the network is the heat-transfer surface of the plant, so a fresh mesh gets
        # one: the pipes are painted on the very cells the run will solve
        self.paint_pipe_network("painted with the mesh")
        self._update_mesh_info()
        self.viz.show_mesh(mesh)
        self.statusBar().showMessage("Mesh ready")

    def paint_pipe_network(self, reason: str = "") -> object:
        """Build the network of the Pipes tab on this window's mesh and paint it.

        The panel owns the widgets but not the mesh: the cells the paint marks have to
        be the cells the run will solve, so the build happens here, on ``self.mesh``.
        ``build_mesh`` calls it, which is what makes the buried network the *default*
        path of a simulation - the gas loop is what carries the heat - and the Pipes
        tab's button calls it again after a change of the plumbing.  Returns the
        network, or ``None`` when it could not be built.
        """
        panel = self.geometry_panel
        if self.controller.running:
            self.log("[pipes] a simulation is running: the mesh is being read")
            return panel.pipe_network()
        if self.mesh is None:
            panel.set_pipe_network(None, "no mesh: build the mesh first")
            return None
        try:
            cyl = panel.cylinder()
            network = build_pipe_network(self.mesh, panel.pipe_network_config(),
                                         center=(cyl.center_x, cyl.center_y))
            report = network.paint(self.mesh)
        except ValueError as exc:
            panel.set_pipe_network(None, f"invalid: {exc}")
            self.log(f"[pipes] refused: {exc}")
            return None
        panel.set_pipe_network(network, report, reason)
        for note in network.notes:
            self.log(f"[pipes] {note}")
        self.log(f"[pipes] {network.n_risers} risers, {network.specific_area:.2f} m2/m3, "
                 f"painted {report.cells:,} cells ({report.area:.1f} m2)")
        self.log(f"[pipes] {panel.circuit_line()}")
        return network

    @safe_slot
    def build_pipe_network(self) -> None:
        """The Pipes tab's button: rebuild the network on the current mesh."""
        if self.paint_pipe_network() is None:
            return
        self._refresh_results()
        self.statusBar().showMessage("Pipe network painted")

    def preview_network(self):
        """The network the Pipes tab describes *now*, for drawing only (None if invalid).

        It is laid out on a throw-away tree of eight leaves - the centrelines do not
        depend on the mesh, and rasterising them on eight cells costs milliseconds - so
        the preview follows every edit without a mesh, and without touching the network
        painted on the run's mesh.
        """
        panel = self.geometry_panel
        cyl = panel.cylinder()
        box = max(panel.domain())
        try:
            sketch = AdaptiveMesh.uniform(2, 0.5 * box, level=0)
            return build_pipe_network(sketch, panel.pipe_network_config(),
                                      center=(cyl.center_x, cyl.center_y))
        except ValueError as exc:
            self.statusBar().showMessage(f"pipe network: {exc}")
            return None

    @safe_slot
    def preview_geometry(self, quiet: bool = False) -> None:
        """Draw the vessel and the network as the panels describe them now."""
        try:
            self.battery = self._battery_from_panels()
        except ValueError as exc:
            if quiet:
                self.statusBar().showMessage(f"Geometry error: {exc}")
                return
            QMessageBox.critical(self, "Geometry error", str(exc))
            return
        self.viz.show_geometry(self.battery, None, network=self.preview_network(),
                               reset_camera=not quiet)

    @safe_slot
    def _geometry_edited(self) -> None:
        """A vessel, circuit or pipe setting changed: redraw, and say the mesh is old."""
        if self.mesh is not None:
            self.statusBar().showMessage("the geometry changed: press Build mesh to "
                                         "simulate it")
        self.preview_geometry(quiet=True)

    @safe_slot
    def find_auto_mesh(self) -> None:
        """Search the mesh that makes the steady answer converge (background thread)."""
        if self.controller.running:
            self.log("[automesh] a simulation is already running")
            return
        self._refresh_plan()
        try:
            self.battery = self._battery_from_panels()
        except ValueError as exc:
            QMessageBox.critical(self, "Geometry error", str(exc))
            return
        self.geometry_panel.set_auto_spec(None, "searching...")
        config = self._run_config()
        config.analysis = "automesh"
        self.controller.start(config, None)

    def _mesh_request(self) -> AdaptivePlan:
        """The tree the automatic search starts from: the Mesh tab's a priori plan."""
        return self.geometry_panel.adaptive_plan()

    def _run_config(self) -> RunConfig:
        return RunConfig(
            analysis=self.analysis_panel.analysis_type(),
            battery=self.battery,
            mesh_spec=self._mesh_request(),
            convergence=self.geometry_panel.auto_mesh_settings(),
            n_threads=self.solver_panel.threads(),
            losses={"t_target": self.analysis_panel.losses_target_kelvin(),
                    "t_ambient": self.battery.t_ambient,
                    "t_ground": self.battery.t_ground,
                    **self.solver_panel.losses_settings()},
            transient=self.analysis_panel.transient_settings(),
            initial_condition=self.analysis_panel.initial_condition(),
            power_profile=self.analysis_panel.power_profile(
                self.battery.heaters.power_w),
            extraction_profile=self.analysis_panel.extraction_profile(),
            start_from_steady=self.analysis_panel.wants_steady_initial_condition(),
            pipe_network=self.geometry_panel.pipe_network(),
            pipe_config=self.geometry_panel.pipe_network_config(),
            pipe_flow=self.geometry_panel.pipe_mass_flow(),
            pipe_fluid=self.geometry_panel.circuit_fluid(),
            pipe_pressure=self.geometry_panel.circuit_pressure_pa(),
            pipe_fan_efficiency=self.geometry_panel.fan_efficiency_fraction(),
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
                self.log("[automesh] mesh adopted: press Build mesh to use it")
            else:
                QMessageBox.warning(self, "Automatic mesh", result.message)
            self.statusBar().showMessage(f"automesh {state}")
            return
        loop = self.controller.last_loop
        if analysis == "standby":
            self.results.update_energy(self.mesh, losses=result,
                                       ambient=self.battery.t_ambient,
                                       radiation=self._radiation(), loop=loop)
        elif analysis == "transient":
            self._last_transient = result
            summary = result.summary()
            self.log(f"[transient] {len(result)} samples, "
                     f"T_mean_end={summary.get('T_mean_storage_end', float('nan')) - 273.15:.1f} °C")
            self.results.update_transient(result)
            self.results.update_energy(self.mesh, transient=result,
                                       ambient=self.battery.t_ambient,
                                       radiation=self._radiation(), loop=loop)
        else:
            self.results.update_energy(self.mesh, ambient=self.battery.t_ambient,
                                       radiation=self._radiation(), loop=loop)
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

    @safe_slot
    def _on_analysis_changed(self, analysis: str) -> None:
        self.run_btn.setText({"standby": "Run standby",
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
        """Write the current field (every leaf, every field) for ParaView."""
        if self.mesh is None:
            QMessageBox.warning(self, "No mesh", "Build the mesh first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export the field (VTK)",
                                              "results/field.vtu", "VTK (*.vtu)")
        if path:
            written = export_vtk(self.mesh, path)
            self.log(f"[export] {written}")


def main() -> int:
    app = QApplication(sys.argv)
    window = ThermalBatteryGUI()
    window.show()
    return app.exec()
