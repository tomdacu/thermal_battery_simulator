"""Geometry panel: cylinder, insulation, heaters, tubes and mesh sub-tabs."""
from __future__ import annotations


from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (QLabel, QListWidget, QTabWidget, QVBoxLayout,
                             QWidget)

import numpy as np

from src.analysis.convergence import AdaptivePlan
from src.analysis.mesh_plan import refinement_bands, tree_resolution
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.mesh import Mesh3D
from src.core.refinement import Band, GridSpec
from src.core.heaters import (SURFACE_POWER_LIMIT_W_CM2,
                              SURFACE_POWER_MIN_W_CM2)
from src.core.pipe_network import (COLLECTION_CENTRAL, COLLECTION_DIRECT,
                                   COLLECTION_REVERSE, COLLECTION_TWO_LEVEL,
                                   LAYOUT_GRID, LAYOUT_RADIAL, LAYOUT_RINGS,
                                   LAYOUT_SPIRAL, LAYOUT_STAGGERED, PIPE_CARBON,
                                   PIPE_STAINLESS, SPLIT_EQUAL, SPLIT_PATH,
                                   SPLIT_RING, SPLIT_SECTOR, WARNING,
                                   PipeNetworkConfig)
from src.core.geometry import (
    CylinderGeometry,
    HeaterConfig,
    HeaterPattern,
    TubeConfig,
    TubePattern,
)

from ..widgets import FormPanel, button, check, combo, double_spin, hint, int_spin


class GeometryPanel(QWidget):
    """All geometry controls; exposes ``build_*`` accessors returning src objects."""

    mesh_changed = pyqtSignal()
    auto_mesh_requested = pyqtSignal()
    preview_requested = pyqtSignal()
    #: the Pipes tab asks the window - which owns the mesh - to build and paint the
    #: network; the window answers through :meth:`set_pipe_network`
    pipe_network_requested = pyqtSignal()

    HEATER_PATTERNS = (
        ("Uniform zone (volumetric)", HeaterPattern.UNIFORM_ZONE),
        ("Vertical grid", HeaterPattern.GRID_VERTICAL),
        ("Checkerboard", HeaterPattern.CHESS_PATTERN),
        ("Radial array", HeaterPattern.RADIAL_ARRAY),
        ("Spiral", HeaterPattern.SPIRAL),
        ("Concentric rings", HeaterPattern.CONCENTRIC_RINGS),
    )
    TUBE_PATTERNS = (
        ("Central cluster", TubePattern.CENTRAL_CLUSTER),
        ("Radial array", TubePattern.RADIAL_ARRAY),
        ("Grid", TubePattern.GRID),
        ("Hexagonal (dense)", TubePattern.HEXAGONAL),
        ("Single central", TubePattern.SINGLE_CENTRAL),
        ("Custom positions", TubePattern.CUSTOM),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self._auto_spec: GridSpec | None = None
        self._plan_targets: dict[str, float] = {}
        self._pipe_network = None
        #: the tree the last build produced, for the Mesh tab's summary
        self._adaptive_mesh: AdaptiveMesh | None = None
        #: the junction-refinement spin of the Pipes tab; None until that tab is built
        self.pipe_junction = None
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self._build_cylinder_tab()
        self._build_insulation_tab()
        self._build_heaters_tab()
        self._build_tubes_tab()
        self._build_mesh_tab()
        self._build_pipes_tab()

    # ------------------------------------------------------------------ tabs
    def _build_cylinder_tab(self) -> None:
        panel = FormPanel()
        self.domain_lx = panel.add("Lx [m]", double_spin(6.0, 1.0, 50.0, 0.5, 1,
                                                         on_change=self.mesh_changed.emit))
        self.domain_ly = panel.add("Ly [m]", double_spin(6.0, 1.0, 50.0, 0.5, 1,
                                                         on_change=self.mesh_changed.emit))
        self.domain_lz = panel.add("Lz [m]", double_spin(5.6, 1.0, 50.0, 0.5, 1,
                                                         on_change=self.mesh_changed.emit))
        self.center_x = panel.add("Centre X [m]", double_spin(3.0, 0.1, 49.0, 0.1, 2))
        self.center_y = panel.add("Centre Y [m]", double_spin(3.0, 0.1, 49.0, 0.1, 2))
        self.base_z = panel.add("Base elevation [m]", double_spin(0.3, 0.0, 5.0, 0.1, 2,
                                                                  tooltip="Floor of the structure"))
        self.radius = panel.add("Storage radius [m]", double_spin(2.0, 0.2, 20.0, 0.1, 2))
        self.height = panel.add("Storage height [m]", double_spin(4.0, 0.5, 30.0, 0.5, 1))
        self.phase_offset = panel.add("Tubes/heaters phase [deg]",
                                      double_spin(15.0, 0.0, 180.0, 5.0, 1))
        self.enable_roof = panel.add("Conical roof",
                                     check("enable", True, "Cone-shaped steel roof"))
        self.roof_angle = panel.add("Roof angle [deg]", double_spin(15.0, 0.0, 45.0, 1.0, 1))
        self.steel_slab = panel.add("Steel slab [m]", double_spin(0.005, 0.0, 0.2, 0.005, 3))
        self.fill_cone = panel.add("Cone fill", check("fill with sand", False))
        panel.add_hint("The apex of the roof must stay below Lz: the build aborts "
                       "instead of silently cutting the roof.")
        self.tabs.addTab(panel, "Cylinder")

    def _build_insulation_tab(self) -> None:
        panel = FormPanel()
        self.insulation_thickness = panel.add("Radial insulation [m]",
                                              double_spin(0.3, 0.02, 1.0, 0.05, 2))
        self.shell_thickness = panel.add("Steel shell [m]",
                                         double_spin(0.02, 0.0, 0.2, 0.005, 3))
        self.slab_bottom = panel.add("Bottom slab [m]", double_spin(0.2, 0.0, 1.0, 0.05, 2))
        self.slab_top = panel.add("Top slab [m]", double_spin(0.2, 0.0, 1.0, 0.05, 2))
        self.foundation_margin = panel.add("Foundation margin [m]",
                                           double_spin(0.5, 0.0, 3.0, 0.1, 2,
                                                       tooltip="Concrete extends this far "
                                                               "beyond the shell"))
        self.tabs.addTab(panel, "Insulation")

    def _build_heaters_tab(self) -> None:
        panel = FormPanel()
        self._heater_panel = panel
        self.power = panel.add("Total power [kW]", double_spin(5.0, 0.0, 10000.0, 1.0, 1,
                                                               on_change=self._update_power,
                                                               tooltip="The storage temperature "
                                                                       "rise follows the "
                                                                       "geometry: check it "
                                                                       "with a steady run"))
        self.heater_pattern = panel.add("Pattern", combo(self.HEATER_PATTERNS, 0,
                                                         self._update_power))
        self.n_heaters = panel.add("Elements", int_spin(12, 1, 500, 1,
                                                        on_change=self._update_power))
        self.offset_bottom = panel.add("Offset from bottom [m]",
                                       double_spin(0.0, 0.0, 2.0, 0.1, 2))
        self.offset_top = panel.add("Offset from top [m]", double_spin(0.0, 0.0, 2.0, 0.1, 2))
        self.heater_grid_rows = panel.add("Grid rows", int_spin(4, 1, 20))
        self.heater_grid_cols = panel.add("Grid columns", int_spin(4, 1, 20))
        self.sheath_diameter = panel.add("Sheath diameter [m]", double_spin(
            0.012, 0.006, 0.05, 0.002, 3, on_change=self._update_power,
            tooltip="Outer diameter of the tubular element (U-shaped, hairpin)"))
        self.leg_spacing = panel.add("Leg spacing [m]", double_spin(
            0.08, 0.02, 1.0, 0.01, 3, tooltip="Centre-to-centre distance of the two legs"))
        self.active_length = panel.add("Active length [m]", double_spin(
            0.0, 0.0, 30.0, 0.1, 2, tooltip="Heated length inside the sand (0 = whole band)"))
        self.cold_shank = panel.add("Cold shank [m]", double_spin(
            0.15, 0.0, 2.0, 0.05, 2, tooltip="Length through the insulation and the air"))
        self.support_offset = panel.add("Support plate [m]", double_spin(
            0.05, 0.0, 1.0, 0.01, 2, tooltip="Support plate above the storage floor"))
        self.flange_offset = panel.add("Flange above roof [m]", double_spin(
            0.03, 0.0, 1.0, 0.01, 2))
        self.heater_rings = panel.add("Rings", int_spin(2, 1, 10))
        self.power_per_heater = panel.add("Power per element [kW]", hint("-"))
        self.surface_power = panel.add("Surface power [W/cm²]", hint("-"))
        panel.add_row(button("Calculate positions", self.preview_requested.emit))
        self.heater_positions = QListWidget()
        self.heater_positions.setMaximumHeight(120)
        panel.add_row(self.heater_positions)
        panel.add_hint("Uniform zone: the power is spread over the whole storage volume. "
                       "Discrete patterns mark individual cells as heat sources.")
        self.tabs.addTab(panel, "Heaters")

    def _build_tubes_tab(self) -> None:
        panel = FormPanel()
        self.tubes_active = panel.add("Heat exchanger",
                                      check("tubes active (discharge)", False))
        self.tube_t_fluid = panel.add("Fluid inlet [°C]", double_spin(60.0, -20.0, 400.0, 5.0, 1))
        self.tube_h_fluid = panel.add("Fluid h [W/(m²·K)]", double_spin(500.0, 10.0, 20000.0, 50.0, 0))
        self.tube_pattern = panel.add("Pattern", combo(self.TUBE_PATTERNS, 1))
        self.n_tubes = panel.add("Tubes", int_spin(8, 1, 200))
        self.tube_diameter = panel.add("Diameter [m]", double_spin(0.05, 0.01, 0.5, 0.005, 3))
        self.tube_grid_rows = panel.add("Grid rows", int_spin(3, 1, 20))
        self.tube_grid_cols = panel.add("Grid columns", int_spin(3, 1, 20))
        self.tube_grid_spacing = panel.add("Grid spacing [m]", double_spin(0.2, 0.05, 2.0, 0.05, 2))
        self.tube_rings = panel.add("Rings", int_spin(2, 1, 10))
        panel.add_row(button("Calculate positions", self.preview_requested.emit))
        self.tube_positions = QListWidget()
        self.tube_positions.setMaximumHeight(120)
        panel.add_row(self.tube_positions)
        self.tabs.addTab(panel, "Tubes")

    def _build_pipes_tab(self) -> None:
        """Buried pipe network: the layout, the tube, the circuit and the paint."""
        panel = FormPanel()
        self.pipe_layout = panel.add("Layout", combo(
            (("Staggered bundle", LAYOUT_STAGGERED), ("Square grid", LAYOUT_GRID),
             ("Concentric rings", LAYOUT_RINGS), ("Radial files", LAYOUT_RADIAL),
             ("Horizontal spiral", LAYOUT_SPIRAL)), 2))
        self.pipe_collection = panel.add("Collection", combo(
            (("Distributor + collector", COLLECTION_DIRECT),
             ("Reverse return (balanced)", COLLECTION_REVERSE),
             ("Central header", COLLECTION_CENTRAL),
             ("Two level rings", COLLECTION_TWO_LEVEL)), 1))
        self.pipe_rings = panel.add("Rings", int_spin(
            3, 1, 12, 1, tooltip="For the ring layouts"))
        self.pipe_files = panel.add("Radial files", int_spin(
            12, 3, 72, 1, tooltip="For the radial layout"))
        self.pipe_diameter = panel.add("Pipe outer d [m]",
                                       double_spin(0.05, 0.01, 0.3, 0.005, 3,
                                                   tooltip="The pitch follows this "
                                                           "diameter"))
        self.pipe_wall = panel.add("Wall thickness [mm]", double_spin(
            2.0, 0.0, 20.0, 0.5, 2,
            tooltip="The gas flows in the bore d - 2 t: the hydraulics, the film "
                    "coefficient and the velocity are the bore's, the pitches and the "
                    "wetted area are the outer diameter's"))
        self.pipe_material = panel.add("Tube material", combo(
            (("Stainless steel (drawn)", PIPE_STAINLESS),
             ("Carbon steel (commercial)", PIPE_CARBON)), 0))
        self.pipe_material.setToolTip(
            "The label of the tube and the roughness of its wall: 15 um drawn, "
            "46 um commercial")
        self.pipe_roughness = panel.add("Roughness [um]", double_spin(
            0.0, 0.0, 2000.0, 5.0, 1, special="from the material",
            tooltip="Absolute wall roughness: leave it at the minimum to take the "
                    "material's own value"))
        self.pipe_duct = panel.add("Duct d [m]", double_spin(0.15, 0.05, 0.6, 0.05, 3))
        self.pipe_insulated = panel.add("Insulated headers", check(
            "lag the distributor and the collector", False,
            tooltip="A lagged header carries the gas and exchanges nothing with the "
                    "bed, so the heat transfer surface is the risers alone"))
        self.pipe_junction = panel.add("Junction refinement [m]", double_spin(
            0.0, 0.0, 0.5, 0.005, 3, special="off",
            tooltip="Cell size the mesh band around the two header elevations asks "
                    "for: the tube-header junction is where the gas turns"))
        self.pipe_azimuth_in = panel.add("Inlet azimuth [deg]",
                                         double_spin(180.0, 0.0, 360.0, 15.0, 0))
        self.pipe_azimuth_out = panel.add("Outlet azimuth [deg]",
                                          double_spin(0.0, 0.0, 360.0, 15.0, 0))
        self.pipe_split = panel.add("Flow split", combo(
            (("Equal per branch", SPLIT_EQUAL), ("From path length", SPLIT_PATH),
             ("Equal per ring main", SPLIT_RING),
             ("Equal per sector", SPLIT_SECTOR)), 0))
        self.pipe_sectors = panel.add("Sectors", int_spin(
            4, 1, 16, 1, tooltip="For the sector distribution: the flow is divided "
                                 "between the sectors about the inlet azimuth"))
        self.pipe_flow = panel.add("Circuit flow [kg/s]", double_spin(
            0.5, 0.0, 200.0, 0.05, 3,
            tooltip="Total mass flow of the gas loop the transient marches: the "
                    "branches split it by the rule above"))
        self.pipe_h = panel.add("Gas h [W/(m²·K)]", double_spin(
            500.0, 10.0, 20000.0, 50.0, 0,
            tooltip="Film the painted pipe cells are given; a loop run replaces it "
                    "with its own march"))
        self.pipe_gas_t = panel.add("Gas T [°C]", double_spin(
            60.0, -20.0, 400.0, 5.0, 1,
            tooltip="Gas temperature of that film: it is what a steady or losses run "
                    "sees, while the loop computes its own"))
        panel.add_row(button("Build network and paint it on the mesh",
                             self.pipe_network_requested.emit,
                             "Voxelise the network on the mesh, mark its cells as pipes "
                             "and write their convective link"))
        self.pipe_info = panel.add("Network", hint("build the mesh, then the network"))
        panel.add_hint("The risers are buried in the sand and the gas goes in from the "
                       "side at the bottom and out from the side at the top: a vessel "
                       "is not axisymmetric and nothing leaves through the roof. "
                       "Painting the network switches the lumped tube bank of the "
                       "Tubes tab off: the pipes are where the heat now crosses.")
        self.tabs.addTab(panel, "Pipes")

    def pipe_network_config(self) -> PipeNetworkConfig:
        """The buried-pipe network the panel describes, in the current vessel.

        The active band is the *storage* band of the cylinder: the risers span the
        sand (and not the insulation slabs under and over it), which is what makes the
        riser length and the bed volume of the module the physical ones.
        """
        cyl = self.cylinder()
        roughness = self.pipe_roughness.value()
        return PipeNetworkConfig(
            # the wall of the vessel runs from the floor to the cone base, and the
            # sand from the bottom slab to the top one: the risers span the sand
            radius=cyl.r_storage, height=cyl.z_cone_base - cyl.base_z, base_z=cyl.base_z,
            band_bottom=max(cyl.z_storage_start - cyl.base_z, 0.05),
            band_top=cyl.z_storage_end - cyl.base_z, diameter=self.pipe_diameter.value(),
            wall_thickness=self.pipe_wall.value() / 1000.0,
            material=self.pipe_material.currentData(),
            roughness=None if roughness <= 0.0 else roughness * 1e-6,
            layout=self.pipe_layout.currentData(),
            collection=self.pipe_collection.currentData(),
            n_rings=int(self.pipe_rings.value()), n_files=int(self.pipe_files.value()),
            duct_diameter=self.pipe_duct.value(),
            insulated_headers=self.pipe_insulated.isChecked(),
            junction_refinement=self.pipe_junction.value() or None,
            azimuth_in=self.pipe_azimuth_in.value(),
            azimuth_out=self.pipe_azimuth_out.value(),
            split_mode=self.pipe_split.currentData(),
            n_sectors=int(self.pipe_sectors.value()))

    def pipe_paint_settings(self) -> dict:
        """Film coefficient and gas temperature the paint writes on the pipe cells."""
        from src.units import c_to_k

        return {"h_fluid": self.pipe_h.value(), "t_fluid": c_to_k(self.pipe_gas_t.value())}

    def pipe_mass_flow(self) -> float:
        """Total mass flow of the gas circuit the transient marches [kg/s]."""
        return float(self.pipe_flow.value())

    def pipe_network(self):
        """The network the window built and painted (None until then)."""
        return self._pipe_network

    def set_pipe_network(self, network, report=None, message: str = "") -> None:
        """Adopt the network the window built on the mesh it owns and report it."""
        self._pipe_network = network
        if network is None:
            self.pipe_info.setText(message or "build the mesh, then the network")
            return
        problems = [problem for problem in network.config.validate()
                    if problem.startswith(WARNING)]
        problems.extend(network.validate())
        lines = [network.summary()]
        if report is not None:
            lines.append(report.summary())
        if problems:
            lines.append("check: " + "; ".join(problems))
        if message:
            lines.append(message)
        self.pipe_info.setText("\n".join(lines))

    def disable_lumped_tubes(self) -> bool:
        """Turn the lumped tube bank off: the network is the heat exchanger now."""
        if not self.tubes_active.isChecked():
            return False
        self.tubes_active.setChecked(False)
        return True

    def _build_mesh_tab(self) -> None:
        panel = FormPanel()
        self.refined = panel.add("Refined mesh", check(
            "cells placed where the gradients are", True, on_toggle=self._mesh_mode))
        self.adaptive = panel.add("Adaptive mesh", check(
            "an octree of leaves instead of a graded grid", False, on_toggle=self._mesh_mode,
            tooltip="The migration target: the same physical targets, refined as boxes of "
                    "an octree instead of as bands of three axes.  The summary then "
                    "reports leaves and levels rather than cells per axis."))
        self.spacing = panel.add("Cell size (uniform) [m]",
                                 double_spin(0.2, 0.02, 1.0, 0.05, 3,
                                             on_change=self.mesh_changed.emit))
        self.cells_storage = panel.add("Cells across storage", int_spin(
            10, 2, 200, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the storage radius at the wall"))
        self.cells_insulation = panel.add("Cells across insulation", int_spin(
            3, 1, 50, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the radial insulation thickness"))
        self.cells_sheath = panel.add("Cells across sheath", int_spin(
            2, 1, 20, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the heater sheath and tube diameter"))
        self.far_field = panel.add("Far field [m]", double_spin(
            0.4, 0.05, 2.0, 0.05, 2, on_change=self._mesh_mode,
            tooltip="Largest cell allowed in the air around the battery"))
        self.growth = panel.add("Growth ratio", double_spin(
            1.3, 1.02, 2.0, 0.05, 2, on_change=self._mesh_mode,
            tooltip="Largest size change between neighbouring cells"))
        self.min_cell = panel.add("Smallest cell [m]", double_spin(
            0.0, 0.0, 1.0, 0.005, 3, on_change=self._mesh_mode, special="auto",
            tooltip="Floor on the realised cell size: with a value here no cell is "
                    "smaller than this, so a deep refinement cannot explode the cell "
                    "count.  'auto' = only the targets decide"))
        self.max_cell = panel.add("Largest cell [m]", double_spin(
            0.0, 0.0, 5.0, 0.05, 2, on_change=self._mesh_mode, special="auto",
            tooltip="Ceiling on the realised cell size: with a value here no cell is "
                    "larger than this, so the far field gets subdivided.  "
                    "'auto' = only the targets decide"))
        self.max_cells = panel.add("Cell budget", int_spin(
            400_000, 10_000, 20_000_000, 50_000, tooltip="Every target is scaled up "
            "by a common factor to fit this budget", on_change=self._mesh_mode))
        self.plan_info = panel.add("Physics plan", hint(
            "the materials and the film coefficients decide the targets"))
        self.mesh_info = panel.add("Grid", hint("build the mesh to see the grid"))
        self.memory_info = panel.add("Memory", hint("-"))
        panel.add_hint("Refined (graded) mesh: the targets above are physical - they do "
                       "not depend on the domain size.  Uniform mesh: one cell size "
                       "everywhere, box snapped to a whole number of cells.")
        separator = QLabel("Automatic mesh: refine until the answer stops moving")
        separator.setToolTip("Runs the same solve on a sequence of finer grids and stops "
                             "when the storage mean temperature and the heat leaving the "
                             "battery change by less than the tolerances")
        panel.add_row(separator)
        self.auto_dt = panel.add("Temperature tolerance [K]",
                                 double_spin(2.0, 0.05, 100.0, 0.5, 2,
                                             tooltip="Largest change of the storage mean "
                                                     "temperature between two grids"))
        self.auto_dp = panel.add("Power tolerance [%]",
                                 double_spin(2.0, 0.05, 50.0, 0.25, 2,
                                             tooltip="Largest relative change of the heat "
                                                     "leaving the battery"))
        self.auto_levels = panel.add("Levels", int_spin(
            4, 2, 8, 1, tooltip="Grids tried, each one finer by the refine factor"))
        self.auto_refine = panel.add("Refine factor", double_spin(
            0.6, 0.2, 0.9, 0.05, 2, tooltip="Target scale from one level to the next"))
        self.auto_btn = panel.add_row(button("Find the mesh", self.auto_mesh_requested.emit,
                                            "Run the search on a background thread"))
        self.auto_first = panel.add("Automatic mesh", check(
            "search before building (recommended)", True,
            tooltip="Build mesh runs the search first: the physics plan sizes the targets "
                    "and the convergence search moves them until the steady answer stops "
                    "moving.  Uncheck to build exactly the grid configured above."))
        self.auto_result = panel.add("Search", hint("not run yet"))
        panel.add_hint("The search solves the *steady* case: it picks the mesh, then all "
                       "analyses use it.  If the budget or the minimum cell size stops "
                       "the refinement, it says so instead of pretending.")
        panel.add_hint("Adaptive mode: the search picks the same cell sizes - the bands "
                       "of the physics plan - and the mesh is then built as an octree of "
                       "those boxes, so the automatic mesh is a tree.")
        self._mesh_mode()
        self.tabs.addTab(panel, "Mesh")

    def _mesh_mode(self, *_args) -> None:
        """Enable the controls of the selected mesh mode."""
        graded = self.refined.isChecked()
        adaptive = self.mesh_kind() == "adaptive"
        for widget in (self.cells_storage, self.cells_insulation, self.cells_sheath,
                       self.far_field, self.max_cells):
            widget.setEnabled(graded)
        for widget in (self.growth, self.min_cell, self.max_cell):
            # a tree has no growth ratio to keep (the octree holds its own 2:1 balance) and
            # no min/max size rails to sit on (a leaf edge is a power of two of the finest
            # cell), so the controls that would do nothing are switched off rather than
            # accepted and ignored
            widget.setEnabled(graded and not adaptive)
        self.adaptive.setEnabled(graded)
        self.spacing.setEnabled(not graded)
        self._update_mesh_summary()

    def mesh_kind(self) -> str:
        """``"adaptive"`` when the panel builds a tree, ``"structured"`` otherwise."""
        return "adaptive" if self.refined.isChecked() and self.adaptive.isChecked() \
            else "structured"

    def _update_mesh_summary(self) -> None:
        """Live summary of the mesh the current settings would build.

        Only the grid *edges* are computed here (a few thousand numbers): building
        the mesh allocates every field and is reserved for the build button.  A tree is
        summarised from the tree the last build produced - its leaf count is the
        refinement itself, and an estimate of it would be a guess dressed as a
        measurement - while the *recipe* is always shown, because that is what the
        settings decide.
        """
        lx, ly, lz, spacing = self.domain()
        if self.mesh_kind() == "adaptive":
            self._update_tree_summary(lx, ly, lz)
            return
        try:
            if self.refined.isChecked():
                spec = self.grid_spec()
                # the live summary must stay live: an analytic count (band length /
                # target, a lower bound of the real one) tells us when computing the
                # edges would freeze the panel for minutes
                estimate = 1.0
                for bands, length in ((spec.x, lx), (spec.y, ly), (spec.z, lz)):
                    declared = bands or (Band(0.0, length, 1.0),)
                    estimate *= max(sum(max(b.length, 0.0) / max(b.target, 1e-9)
                                        for b in declared), 1.0)
                if estimate > 2_000_000:
                    self.mesh_info.setText(
                        f"refined: about {estimate:,.0f} cells or more - too large to "
                        f"preview, build it with the button")
                    self.memory_info.setText("-")
                    return
                summary = spec.describe(spec.edges(lx, ly, lz))
                mode = "refined"
            else:
                nx = max(3, int(round(lx / spacing)))
                step = lx / nx
                ny, nz = max(3, int(round(ly / step))), max(3, int(round(lz / step)))
                summary = {"cells_axis": (nx, ny, nz), "cells": nx * ny * nz,
                           "min_size": step, "max_size": step, "worst_ratio": 1.0}
                mode = "uniform"
        except (ValueError, RuntimeError) as exc:
            self.mesh_info.setText(f"invalid: {exc}")
            self.memory_info.setText("-")
            return
        nx, ny, nz = summary["cells_axis"]
        size = (f"{summary['min_size']:.3f} m" if mode == "uniform" else
                f"{summary['min_size']:.3f}-{summary['max_size']:.3f} m")
        self.mesh_info.setText(
            f"{mode}: {nx} x {ny} x {nz} = {summary['cells']:,} cells\n"
            f"cell size {size}\n"
            f"box {lx:.3f} x {ly:.3f} x {lz:.3f} m (y/z snapped to the grid)")
        note = ""
        if self.refined.isChecked():
            spec = self.grid_spec()
            wanted = min(spec.targets())
            if self.min_cell.value():
                wanted = max(wanted, self.min_cell.value())
            if summary["min_size"] > wanted * 1.05:
                note = "  (budget raised the targets)"
            note += self._limits_note(spec, summary)
        self.memory_info.setText(
            f"~{summary['cells'] * 98 / 1e6:.1f} MB of fields "
            f"(max ratio {summary['worst_ratio']:.2f}){note}")

    def _update_tree_summary(self, lx: float, ly: float, lz: float) -> None:
        """The Mesh tab's summary of an adaptive mesh: leaves, levels and leaf edges.

        A tree has no cells per axis and no growth ratio (the octree keeps its own 2:1
        balance after every refinement), and the panel does not invent them: it states
        what the recipe fixes - the box, the resolution floor and the refinement boxes -
        and, once a tree has been built, what the tree actually is.
        """
        try:
            plan = self.adaptive_plan()
        except (ValueError, RuntimeError) as exc:
            self.mesh_info.setText(f"invalid: {exc}")
            self.memory_info.setText("-")
            return
        box = plan.n_finest * plan.physical_size
        sizes = [band.size for band in plan.bands]
        lines = [
            f"adaptive: box {box:.3f} m, finest leaf {plan.physical_size * 1000:.1f} mm",
            f"{len(plan.bands)} refinement boxes, target "
            f"{min(sizes) * 1000:.0f}-{max(sizes) * 1000:.0f} mm "
            f"(domain {lx:.3f} x {ly:.3f} x {lz:.3f} m)"]
        if min(sizes) < plan.physical_size:
            lines.append(f"the budget's floor of {plan.physical_size * 1000:.1f} mm is "
                         f"above the finest target: no leaf goes below it")
        mesh = self._adaptive_mesh
        if mesh is None:
            lines.append("build the mesh to count the leaves")
            self.memory_info.setText("-")
        else:
            levels = ", ".join(f"L{level}: {count:,}"
                               for level, count in mesh.level_histogram().items())
            lines.append(f"{mesh.n_cells:,} leaves ({levels})")
            lines.append(f"leaf edge {mesh.sizes.min() * 1000:.1f}-"
                         f"{mesh.sizes.max() * 1000:.1f} mm")
            self.memory_info.setText(f"~{mesh.n_cells * 98 / 1e6:.1f} MB of fields")
        self.mesh_info.setText("\n".join(lines))

    def adaptive_plan(self) -> AdaptivePlan:
        """The tree the current targets ask for: the same bands, as boxes of an octree.

        The structured bands and the tree boxes come from the same list through
        :func:`src.analysis.mesh_plan.refinement_bands`, so the a priori estimate - the
        storage cells, ``thickness / N``, ``2 k / h``, the tube pitch - is decided once
        and the two roads start from it.  The box and its resolution come from the cell
        budget (:func:`src.analysis.mesh_plan.tree_resolution`), which is also the floor
        under every leaf the search then refines.
        """
        lx, ly, lz = (self.domain_lx.value(), self.domain_ly.value(),
                      self.domain_lz.value())
        spec = self.grid_spec()
        n_finest, physical_size = tree_resolution((lx, ly, lz), int(self.max_cells.value()))
        return AdaptivePlan(n_finest=n_finest, physical_size=physical_size,
                            bands=refinement_bands(spec, (lx, ly, lz)))

    def _limits_note(self, spec, summary: dict) -> str:
        """Say whether the min/max cell rails change anything (usually they do not).

        A rail that does not bite is not a bug, but a field that silently does nothing
        is: the summary states the realised range and which rail is actually working.
        """
        lo, hi = self.min_cell.value(), self.max_cell.value()
        if not lo and not hi:
            return ""
        finest, coarsest = spec.targets()
        parts = []
        if lo:
            parts.append(f"min {lo:g} " + ("active" if lo > finest * 1.001 else "idle"))
        if hi:
            parts.append(f"max {hi:g} " + ("active" if hi < coarsest * 0.999 else "idle"))
        return "  (limits: " + ", ".join(parts) + ")"

    # -------------------------------------------------------------- accessors
    def _update_power(self) -> None:
        count = max(self.n_heaters.value(), 1)
        per_element = self.power.value() / count
        self.power_per_heater.setText(f"{per_element:.2f} kW")
        # the rating that matters for a sheathed element: power per heated surface
        length = self.active_length.value()
        if length <= 0:
            length = max(self.height.value() - self.offset_bottom.value()
                         - self.offset_top.value(), 0.0)
        diameter = self.sheath_diameter.value()
        bend = 0.25 * np.pi * self.leg_spacing.value()
        area = np.pi * diameter * (length + bend)
        if area > 0:
            value = per_element * 1000.0 / area / 1e4
            state = ("in range" if SURFACE_POWER_MIN_W_CM2 <= value <= SURFACE_POWER_LIMIT_W_CM2
                     else "out of the 3-8 W/cm2 range")
            self.surface_power.setText(f"{value:.1f} ({state})")
        else:
            self.surface_power.setText("-")

    def domain(self) -> tuple[float, float, float, float]:
        """Box and (uniform) cell size; the run config keeps them as provenance."""
        return (self.domain_lx.value(), self.domain_ly.value(), self.domain_lz.value(),
                self.spacing.value())

    def build_mesh(self, adaptive: bool | None = None) -> Mesh3D | AdaptiveMesh:
        """Mesh of the current settings: graded, uniform, or the octree of the same targets.

        ``adaptive`` overrides the checkbox (``None`` follows it).  A tree is built with
        :meth:`AdaptiveMesh.from_bands` from :meth:`adaptive_plan`, i.e. from the same
        bands the graded road would use, so switching the mode changes the mesh and not
        the request.  The tree is kept for the Mesh tab's summary: its leaf count, its
        levels and its leaf edges are measured, never guessed.
        """
        lx, ly, lz, spacing = self.domain()
        if (self.mesh_kind() if adaptive is None else adaptive) == "adaptive":
            plan = self.adaptive_plan()
            mesh = AdaptiveMesh.from_bands(plan.n_finest, plan.physical_size, plan.bands,
                                           plan.base_level)
            self._adaptive_mesh = mesh
            self._update_tree_summary(lx, ly, lz)
            return mesh
        self._adaptive_mesh = None
        if not self.refined.isChecked():
            return Mesh3D(Lx=lx, Ly=ly, Lz=lz, spacing=spacing)
        return Mesh3D(Lx=lx, Ly=ly, Lz=lz, grid=self.grid_spec())

    def wants_auto_search(self) -> bool:
        """True when the search is the primary path (see the Mesh tab)."""
        return bool(self.auto_first.isChecked() and self.refined.isChecked())

    def auto_mesh_settings(self) -> dict:
        """Tolerances of the automatic mesh search (see analysis/convergence.py)."""
        return {"delta_temperature": self.auto_dt.value(),
                "delta_power": self.auto_dp.value() / 100.0,
                "max_levels": int(self.auto_levels.value()),
                "refine": self.auto_refine.value(),
                "max_cells": int(self.max_cells.value())}

    def set_plan_targets(self, targets: dict[str, float], note: str = "") -> None:
        """Adopt the a priori plan (analysis/mesh_plan.py): the bands never get coarser.

        A band whose own conduction length, or whose convective sub-layer 2k/h, is
        finer than the manual target takes the plan value: that is what makes the
        surface flux - and therefore the losses - converge.
        """
        self._plan_targets = dict(targets)
        self.plan_info.setText(note or "no plan (build the mesh to compute it)")
        self._update_mesh_summary()

    def planned(self, name: str, manual: float) -> float:
        """The tighter of the manual target and the plan target [m]."""
        target = self._plan_targets.get(name)
        return manual if target is None else min(manual, target)

    def auto_spec(self) -> GridSpec | None:
        """The spec adopted by the search (None until it converges)."""
        return self._auto_spec

    def set_auto_spec(self, spec: GridSpec | None, message: str = "") -> None:
        """Adopt the mesh chosen by the search (None clears it)."""
        self._auto_spec = spec
        if spec is not None:
            self.refined.setChecked(True)
        self.auto_result.setText(message or ("adopted" if spec else "not run yet"))
        self._update_mesh_summary()

    def pipe_junction_bands(self) -> list[tuple[float, float, float]]:
        """Bands the Pipes tab asks the mesh to refine, as ``(low, high, target)``.

        The mesh summary runs before the Pipes tab exists, so the widgets are read
        defensively: no table yet, no band.
        """
        panel = self.pipe_junction
        if panel is None or panel.value() <= 0.0:
            return []
        return self.pipe_network_config().junction_bands()

    def grid_spec(self) -> GridSpec:
        """Physical refinement targets from the panel and the battery.

        Bands: the storage core, the insulation/shell ring, the heater bank (only
        when the heaters are discrete) and the far field.  The heater band is what
        makes a 12 mm sheath representable, so it is tied to the sheath diameter and
        to the element spacing, not to the whole shell ring.
        """
        if self._auto_spec is not None:
            return self._auto_spec
        cyl = self.cylinder()
        lx, ly, lz = self.domain_lx.value(), self.domain_ly.value(), self.domain_lz.value()
        fine = self.planned("storage",
                            cyl.r_storage / max(self.cells_storage.value(), 1))
        insulation = self.planned(
            "insulation_radial",
            cyl.insulation_thickness / max(self.cells_insulation.value(), 1))
        far = self.far_field.value()
        heaters = self.heaters()
        sheath_target = max(
            min(heaters.sheath_diameter, self.tube_diameter.value())
            / max(self.cells_sheath.value(), 1), 1e-3)
        discrete = heaters.pattern != HeaterPattern.UNIFORM_ZONE

        # vertical bands: storage + the two insulation slabs, then the far field
        z = (Band(cyl.z_slab_bottom_start, cyl.z_storage_start,
                  self.planned("slab_bottom", insulation)),
             Band(cyl.z_storage_start, cyl.z_storage_end, fine),
             Band(cyl.z_slab_top_start, cyl.z_slab_top_end,
                  self.planned("slab_top", insulation)),
             Band(0.0, lz, far))
        # the tube-header junctions: the gas turns there and the surface is singular,
        # so the two header elevations carry the refinement the Pipes tab asks for
        junction = tuple(Band(low, high, target)
                         for low, high, target in self.pipe_junction_bands())
        if junction:
            z = z + junction
        # radial bands: heater bank (when discrete), storage core, shell ring, far field
        reach = 0.0
        if discrete:
            span = max(heaters.grid_cols - 1, 0) * heaters.leg_spacing * 2.0
            reach = min(0.5 * span + 2.0 * heaters.leg_spacing, cyl.r_storage)
        radial = []
        for center, extent in ((cyl.center_x, lx), (cyl.center_y, ly)):
            bands = []
            if reach > 0:
                bands.append(Band(max(center - reach, 0.0), min(center + reach, extent),
                                  sheath_target))
            bands.append(Band(max(center - cyl.r_storage, 0.0),
                              min(center + cyl.r_storage, extent), fine))
            # the insulation ring is an *annulus*: a single band across the whole
            # diameter would ask for insulation cells inside the storage as well
            for low, high in ((center - cyl.r_shell, center - cyl.r_storage),
                              (center + cyl.r_storage, center + cyl.r_shell)):
                if high - low > 0:
                    bands.append(Band(max(low, 0.0), min(high, extent), insulation))
            bands.append(Band(0.0, extent, far))
            radial.append(tuple(bands))
        return GridSpec(x=radial[0], y=radial[1], z=z, growth=self.growth.value(),
                        max_cells=int(self.max_cells.value()),
                        min_size=self.min_cell.value() or None,
                        max_size=self.max_cell.value() or None)

    def cylinder(self) -> CylinderGeometry:
        return CylinderGeometry(
            center_x=self.center_x.value(), center_y=self.center_y.value(),
            base_z=self.base_z.value(), height=self.height.value(),
            r_storage=self.radius.value(),
            insulation_thickness=self.insulation_thickness.value(),
            shell_thickness=self.shell_thickness.value(),
            insulation_slab_bottom=self.slab_bottom.value(),
            insulation_slab_top=self.slab_top.value(),
            roof_angle_deg=self.roof_angle.value(),
            steel_slab_top=self.steel_slab.value(),
            fill_cone_with_sand=self.fill_cone.isChecked(),
            enable_cone_roof=self.enable_roof.isChecked(),
            phase_offset_deg=self.phase_offset.value(),
            foundation_margin=self.foundation_margin.value(),
        )

    def heaters(self) -> HeaterConfig:
        return HeaterConfig(
            power_total=self.power.value(), n_heaters=self.n_heaters.value(),
            pattern=self.heater_pattern.currentData(),
            offset_bottom=self.offset_bottom.value(), offset_top=self.offset_top.value(),
            n_rings=self.heater_rings.value(),
            grid_rows=self.heater_grid_rows.value(), grid_cols=self.heater_grid_cols.value(),
            sheath_diameter=self.sheath_diameter.value(),
            leg_spacing=self.leg_spacing.value(),
            active_length=self.active_length.value() if self.active_length.value() > 0
            else None,
            cold_shank=self.cold_shank.value(),
            support_plate_offset=self.support_offset.value(),
            flange_offset=self.flange_offset.value(),
        )

    def tubes(self) -> TubeConfig:
        from src.units import c_to_k

        return TubeConfig(
            n_tubes=self.n_tubes.value(), diameter=self.tube_diameter.value(),
            h_fluid=self.tube_h_fluid.value(), t_fluid=c_to_k(self.tube_t_fluid.value()),
            active=self.tubes_active.isChecked(), pattern=self.tube_pattern.currentData(),
            n_rings=self.tube_rings.value(), grid_rows=self.tube_grid_rows.value(),
            grid_cols=self.tube_grid_cols.value(), grid_spacing=self.tube_grid_spacing.value(),
        )

    def apply_geometry(self, battery) -> None:
        """Attach the built config trees to a BatteryGeometry."""
        battery.cylinder = self.cylinder()
        battery.heaters = self.heaters()
        battery.tubes = self.tubes()

    def set_mesh_info(self, text: str, memory: str) -> None:
        self.mesh_info.setText(text)
        self.memory_info.setText(memory)
