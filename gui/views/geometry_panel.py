"""Geometry panel: the vessel, the plant (gas circuit and pipes) and the mesh.

The panel owns the controls and turns them into ``src`` objects; the window lays its
pages out - **Vessel** (the shape; the window adds the layer materials to it), **Plant**
(the gas circuit the resistors heat and the buried network the gas runs through) and
**Mesh**.  The circuit and the network are the same loop seen from its two ends: the
resistors put power into the gas, the gas carries it to the pipe walls and the sand takes
it up; on discharge the exchanger on the same circuit takes it back out.

There is no domain to set: the air around the vessel is excluded from the problem, so the
box is the smallest one that holds the vessel with a margin of air
(:meth:`GeometryPanel.domain`), centred on it.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QWidget

from src.analysis.convergence import AdaptivePlan
from src.analysis.mesh_plan import (MAX_TREE_LEVEL, MIN_TREE_LEVEL, active_regions,
                                    region_bands)
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import CylinderGeometry, HeaterConfig
from src.core.pipe_network import (COLLECTION_CENTRAL, COLLECTION_DIRECT,
                                   COLLECTION_MANIFOLD, COLLECTION_REVERSE,
                                   COLLECTION_TWO_LEVEL, LAYOUT_GRID, LAYOUT_RADIAL,
                                   LAYOUT_RINGS, LAYOUT_SPIRAL, LAYOUT_STAGGERED,
                                   PIPE_CARBON, PIPE_STAINLESS, SPLIT_EQUAL,
                                   SPLIT_HYDRAULIC, SPLIT_PATH, SPLIT_RING,
                                   SPLIT_SECTOR, WARNING, HeaderDesign,
                                   PipeNetworkConfig, design_headers)
from src.core.pipes import pipe_surface_power_w_cm2
from src.solver.fluid import Fluid

from ..widgets import (INFO, FormPanel, button, check, combo, double_spin, hint, int_spin,
                       rich_lines)

#: air kept around the vessel [m]: the outer film sits on the first excluded leaves
AIR_MARGIN = 0.3


#: a leaf may be this much larger than the target it serves: leaves are powers of two,
#: and the next one down would double (or treble) the leaves of a whole region
LEAF_TOLERANCE = 1.6


def nearest_leaf(target: float, finest: float) -> float:
    """The largest leaf edge within :data:`LEAF_TOLERANCE` of ``target`` [m].

    A 67 mm slab target on 52/104 mm leaves takes 104 mm (two leaves across a 200 mm slab
    of a near-linear conduction profile) instead of 52 mm, which cost 100 000 leaves on
    the default vessel; 200 mm takes 207, 50 mm takes 52.
    """
    ratio = LEAF_TOLERANCE * max(target, finest) / finest
    level = max(int(np.floor(np.log2(ratio) + 1e-9)), 0)
    return finest * 2.0 ** level


class GeometryPanel(QWidget):
    """All geometry and plant controls; exposes accessors returning src objects."""

    mesh_changed = Signal()
    auto_mesh_requested = Signal()
    #: the Pipes section asks the window - which owns the mesh - to build and paint the
    #: network; the window answers through :meth:`set_pipe_network`
    pipe_network_requested = Signal()

    #: the gases the circuit can be filled with: the loop reads their cp, rho, mu and k
    #: (``src/solver/fluid.py``), and the pressure is a separate control because it is
    #: what raises the density and the Reynolds number without raising the velocity
    FLUIDS = (
        ("Air", Fluid()),
        ("Nitrogen", Fluid(name="nitrogen", cp=1040.0, rho=1.12, mu=1.76e-5,
                           k=0.0255, pr=0.72, molar_mass=0.02801)),
        ("Steam", Fluid(name="steam", cp=2030.0, rho=0.60, mu=1.3e-5, k=0.025,
                        pr=0.95, molar_mass=0.01802)),
    )

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        #: the tree plan the automatic search adopted (None: the a priori one)
        self._auto_plan: AdaptivePlan | None = None
        self._plan_targets: dict[str, float] = {}
        self._pipe_network = None
        #: the tree the last build produced, for the Mesh page's summary
        self._adaptive_mesh: AdaptiveMesh | None = None
        #: True when the cell budget stopped the last build short of its targets
        self._budget_short = False
        #: the header engine's last design and the inputs it was made for
        self._header_design: HeaderDesign | None = None
        self._header_key: tuple | None = None
        self.vessel_page = FormPanel()
        self.plant_page = FormPanel()
        self.mesh_page = FormPanel()
        self._build_vessel_page()
        self._build_plant_page()
        self._build_mesh_page()

    # ----------------------------------------------------------------- pages
    def _build_vessel_page(self) -> None:
        shape = self.vessel_page.section(
            "Storage bed",
            "The cylinder of sand.  The box around it is not a setting: the air outside "
            "the vessel is excluded from the problem (the outer surface carries the "
            "ambient film), so the domain is the vessel plus a margin of air.")
        changed = self.mesh_changed.emit
        self.radius = shape.add("Radius [m]", double_spin(2.0, 0.2, 20.0, 0.1, 2,
                                                          on_change=changed))
        self.height = shape.add("Height [m]", double_spin(5.0, 0.5, 30.0, 0.5, 1,
                                                          on_change=changed))
        self.base_z = shape.add("Foundation depth [m]", double_spin(
            0.3, 0.0, 5.0, 0.1, 2, on_change=changed,
            tooltip="Concrete pad under the floor of the vessel, on the ground"))
        self.soil = shape.add("Soil under it [m]", double_spin(
            3.0, 0.0, 20.0, 0.5, 1, on_change=changed,
            tooltip="Ground modelled under the pad (moist sand, 1.5 W/(m K)); the Site "
                    "ground temperature is held at its bottom.  0 holds it right under "
                    "the pad, which overstates the losses to the ground"))
        layers = self.vessel_page.section("Insulation and shell")
        self.insulation_thickness = layers.add("Radial insulation [m]", double_spin(
            0.3, 0.02, 1.0, 0.05, 2, on_change=changed))
        self.slab_bottom = layers.add("Bottom slab [m]", double_spin(
            0.2, 0.0, 1.0, 0.05, 2, on_change=changed))
        self.slab_top = layers.add("Top slab [m]", double_spin(
            0.2, 0.0, 1.0, 0.05, 2, on_change=changed))
        self.shell_thickness = layers.add("Steel shell [m]", double_spin(
            0.02, 0.0, 0.2, 0.005, 3, on_change=changed,
            tooltip="A shell thinner than the local cell is painted one cell thick: its "
                    "conduction is negligible either way, its heat capacity is not"))
        self.foundation_margin = layers.add("Foundation margin [m]", double_spin(
            0.5, 0.0, 3.0, 0.1, 2, on_change=changed,
            tooltip="Concrete extends this far beyond the shell"))
        roof = self.vessel_page.section("Roof")
        self.enable_roof = roof.add("Conical roof", check(
            "enable", True, "Cone-shaped steel roof", on_toggle=changed))
        self.roof_angle = roof.add("Roof angle [deg]", double_spin(
            15.0, 0.0, 45.0, 1.0, 1, on_change=changed))
        self.steel_slab = roof.add("Steel plate [m]", double_spin(
            0.005, 0.0, 0.2, 0.005, 3, on_change=changed,
            tooltip="Plate over the top insulation slab, under the cone"))
        self.fill_cone = roof.add("Cone fill", check("fill the cone with sand", False,
                                                     on_toggle=changed))

    def _build_plant_page(self) -> None:
        circuit = self.plant_page.section(
            "Gas circuit",
            "The electric resistors heat the gas in a tank on the circuit; the hot gas "
            "enters the distributor, rises through the buried pipes, gives its heat to "
            "the sand and leaves from the collector.  On discharge the exchanger on the "
            "same circuit takes the heat out of the gas (Analysis > Discharge).  The bed "
            "is heated through the tube walls, never by an element inside it.  The film "
            "of every pipe cell is computed by the loop from the flow, the gas and the "
            "bore.")
        self.power = circuit.add("Rated power [kW]", double_spin(
            200.0, 0.0, 100000.0, 10.0, 1, on_change=self._update_power,
            tooltip="Charging power of the resistors: the constant power of a "
                    "transient.  The reference pilot (4 m across, 7 m tall, ~100 t of sand, 8 MWh) "
                    "charges at 200 kW and discharges at 100 kW"))
        self.fluid = circuit.add("Gas", combo(self.FLUIDS, 0, self._update_power))
        self.flow = circuit.add("Mass flow [kg/s]", double_spin(
            1.0, 0.0, 200.0, 0.05, 3, on_change=self._update_power,
            tooltip="Total mass flow of the loop, split between the branches by the rule "
                    "below.  A discharge needs at least P / (cp dT): 100 kW from a bed at "
                    "330 degC to gas returning at 60 degC needs ~0.4 kg/s of air"))
        self.circuit_pressure = circuit.add("Pressure [bar]", double_spin(
            1.01325, 0.1, 100.0, 0.5, 5, special="1 bar",
            tooltip="Absolute pressure of the loop: at a fixed mass flow the density "
                    "rises with it, so the film coefficient grows and the pressure drop "
                    "falls - the cheap design lever of a gas loop"))
        self.fan_efficiency = circuit.add("Fan efficiency [%]", double_spin(
            70.0, 10.0, 100.0, 5.0, 0,
            tooltip="The electric power of the fan is the shaft power over it"))
        self.circuit_info = circuit.add("Circuit", hint("build the mesh to size it"))

        pipes = self.plant_page.section(
            "Buried pipes",
            "The risers are buried in the sand; the gas goes in from the side at the "
            "bottom and out from the side at the top.  The same network charges and "
            "discharges the bed.  Build mesh paints it on the mesh; the preview shows it "
            "as soon as you change it.")
        self.pipe_layout = pipes.add("Layout", combo(
            (("Staggered bundle", LAYOUT_STAGGERED), ("Square grid", LAYOUT_GRID),
             ("Concentric rings", LAYOUT_RINGS), ("Radial files", LAYOUT_RADIAL),
             ("Horizontal spiral", LAYOUT_SPIRAL)), 2, self._pipe_mode))
        self.pipe_rings = pipes.add("Rings", int_spin(
            6, 1, 20, 1, tooltip="The rings are spread over the whole radius and the "
                                 "risers on a ring are spaced like the rings, so every "
                                 "riser serves a similar area of sand"))
        self.pipe_files = pipes.add("Radial files", int_spin(12, 3, 72, 1))
        self.pipe_collection = pipes.add("Collection", combo(
            (("Distributor + collector", COLLECTION_DIRECT),
             ("Reverse return (balanced)", COLLECTION_REVERSE),
             ("Central header", COLLECTION_CENTRAL),
             ("Two level rings", COLLECTION_TWO_LEVEL),
             ("Radial manifold (rings in parallel)", COLLECTION_MANIFOLD)), 4,
            self._pipe_mode,
            tooltip="How the headers connect the risers.  The rings of a chain are in "
                    "series and every ring's flow crosses the rings before it; the radial "
                    "manifold feeds them in parallel from a trunk along the inlet "
                    "diameter.  The header engine tries the chosen one and the manifold "
                    "and keeps the better"))
        self.pipe_diameter = pipes.add("Pipe outer d [m]", double_spin(
            0.05, 0.01, 0.3, 0.005, 3, tooltip="The pitch of the dense layouts follows it"))
        self.pipe_wall = pipes.add("Wall thickness [mm]", double_spin(
            2.0, 0.0, 20.0, 0.5, 2,
            tooltip="The gas flows in the bore d - 2 t: hydraulics, film coefficient and "
                    "velocity are the bore's, the wetted area the outer diameter's"))
        self.pipe_material = pipes.add("Tube material", combo(
            (("Stainless steel (drawn)", PIPE_STAINLESS),
             ("Carbon steel (commercial)", PIPE_CARBON)), 0,
            tooltip="The default roughness of the wall: 15 um drawn, 46 um commercial"))
        self.circuit_roughness = pipes.add("Wall roughness [um]", double_spin(
            0.0, 0.0, 2000.0, 5.0, 1, special="from the material",
            tooltip="Absolute roughness of the tubes, headers and ducts; at the minimum "
                    "the tube material's own value"))
        self.pipe_insulated = pipes.add("Insulated headers", check(
            "lagged", False,
            tooltip="A lagged header carries the gas and exchanges nothing with the bed"))
        self.pipe_azimuth_in = pipes.add("Inlet azimuth [deg]",
                                         double_spin(180.0, 0.0, 360.0, 15.0, 0))
        self.pipe_azimuth_out = pipes.add("Outlet azimuth [deg]",
                                          double_spin(0.0, 0.0, 360.0, 15.0, 0))
        self.pipe_split = pipes.add("Flow split", combo(
            (("From the network hydraulics", SPLIT_HYDRAULIC),
             ("Imposed: equal per branch", SPLIT_EQUAL),
             ("Imposed: from path length", SPLIT_PATH),
             ("Imposed: equal per ring main", SPLIT_RING),
             ("Imposed: equal per sector", SPLIT_SECTOR)), 0, self._pipe_mode,
            tooltip="How the flow divides between the risers: what the pressures of the "
                    "network give (the plant), or an imposed rule for a comparison"))
        self.pipe_sectors = pipes.add("Sectors", int_spin(
            4, 1, 16, 1, tooltip="The flow is divided between the sectors about the "
                                 "inlet azimuth"))
        engine = self.plant_page.section(
            "Header engine",
            "The headers are not set by hand: the engine solves the hydraulics of the "
            "whole network (Darcy-Weisbach, the tees' losses, the draught of the hot "
            "gas), asks every riser for the share of the flow equal to the share of the "
            "bed it serves, and picks the header and duct diameters from the nominal pipe "
            "sizes - as small as the velocity limit allows and as large as pays for "
            "itself in pressure drop - adding a calibrated orifice at a riser's inlet "
            "where the headers alone leave it off target.  It lifts the headers into the "
            "sand by one radius and a cover, and tries the chosen collection and the "
            "radial manifold.  Build mesh runs it when the plant changed.")
        self.header_tolerance = engine.add("Flow uniformity [%]", double_spin(
            5.0, 0.5, 50.0, 0.5, 1, on_change=self._engine_stale,
            tooltip="Largest deviation of a riser's flow from its target the headers may "
                    "leave before orifices balance it"))
        self.header_velocity = engine.add("Max header velocity [m/s]", double_spin(
            20.0, 3.0, 60.0, 1.0, 1, on_change=self._engine_stale,
            tooltip="Hot-air headers and ducts are designed at 15-25 m/s: faster costs "
                    "pressure drop and noise"))
        self.header_temperature = engine.add("Design gas temperature [°C]", double_spin(
            500.0, 20.0, 1000.0, 25.0, 0, on_change=self._engine_stale,
            tooltip="The gas density the headers are sized at; the runs re-solve the "
                    "hydraulics at the gas's real temperatures"))
        engine.add_row(button("Size the headers", self.run_header_engine,
                              "Run the engine now (a few seconds to half a minute)"))
        self.engine_info = engine.add("Design", hint("not sized yet"))
        self._engine_caption = engine.form.labelForField(self.engine_info)

        pipes.add_row(button("Rebuild the network on the mesh",
                             self.pipe_network_requested.emit,
                             "Voxelise the network on the current mesh (Build mesh "
                             "does it too)"))
        self.pipe_info = pipes.add("Network", hint("build the mesh: it paints the network"))
        self._pipe_caption = pipes.form.labelForField(self.pipe_info)
        self._pipe_mode()

    def _pipe_mode(self, *_args) -> None:
        """Enable the pipe controls the selected layout and split actually read."""
        layout = self.pipe_layout.currentData()
        self.pipe_rings.setEnabled(layout == LAYOUT_RINGS)
        self.pipe_files.setEnabled(layout == LAYOUT_RADIAL)
        self.pipe_sectors.setEnabled(self.pipe_split.currentData() == SPLIT_SECTOR)

    def _build_mesh_page(self) -> None:
        resolution = self.mesh_page.section(
            "Resolution",
            "The mesh is a tree of boxes refined on the active model only: every leaf "
            "has its own width in plan and its own height, so the bed and the insulation "
            "are tall where nothing changes vertically and thin where the slabs, the "
            "roof and the foundation are.  The pipes need no refinement: a tube is a "
            "line inside a cell, coupled by the well model, and the cell around it must "
            "be larger than the tube.  The air around the vessel is excluded, so its "
            "leaves stay coarse.  Every size snaps to the nearest leaf (powers of two of "
            "the finest one).")
        self.cells_storage = resolution.add("Cells across storage", int_spin(
            10, 2, 200, 1, on_change=self._mesh_mode,
            tooltip="Cells across the storage radius: the bed's own size"))
        self.cells_insulation = resolution.add("Cells across insulation", int_spin(
            3, 1, 50, 1, on_change=self._mesh_mode,
            tooltip="Cells across the radial insulation (and the slabs, and the shell)"))
        self.bed_layers = resolution.add("Layers in the bed height", int_spin(
            20, 2, 400, 1, on_change=self._mesh_mode,
            tooltip="Leaves across the height of the storage: the bed and the radial "
                    "insulation take this height, the slabs, the roof and the "
                    "foundation their own thickness over the insulation count"))
        self.max_cells = resolution.add("Cell budget", int_spin(
            400_000, 10_000, 20_000_000, 50_000, on_change=self._mesh_mode,
            tooltip="The most leaves the mesh may have: the refinement goes coarse to "
                    "fine and stops before a round that would exceed it (the Mesh "
                    "summary says when it did)"))
        self.plan_info = resolution.add("Regions", hint("press Build mesh to plan them"))
        self.mesh_info = resolution.add("Mesh", hint("build the mesh to see it"))
        self.memory_info = resolution.add("Memory", hint("-"))

        search = self.mesh_page.section(
            "Automatic mesh",
            "Refines the tree until the standby answer stops moving: the storage held at "
            "the set temperature on a sequence of finer trees, stopping when the mean "
            "temperature and the holding power change by less than the tolerances.  "
            "Build mesh then uses the adopted plan; if the budget stops the refinement, "
            "the search says so.")
        self.auto_dt = search.add("Temperature tolerance [K]", double_spin(
            2.0, 0.05, 100.0, 0.5, 2,
            tooltip="Largest estimated error of the storage temperature"))
        self.auto_dp = search.add("Power tolerance [%]", double_spin(
            2.0, 0.05, 50.0, 0.25, 2,
            tooltip="Largest estimated error of the holding power (the losses)"))
        self.auto_levels = search.add("Levels", int_spin(
            4, 2, 8, 1, tooltip="Meshes tried, each one finer by the refine factor"))
        self.auto_refine = search.add("Refine factor", double_spin(
            0.6, 0.2, 0.9, 0.05, 2, tooltip="Target scale from one level to the next"))
        self.auto_btn = search.add_row(button("Find the mesh", self.auto_mesh_requested.emit,
                                             "Run the search on a background thread"))
        self.auto_result = search.add("Search", hint("not run yet"))
        self._mesh_mode()

    def _mesh_mode(self, *_args) -> None:
        """A mesh setting changed: a plan the search adopted no longer describes it."""
        if getattr(self, "_auto_plan", None) is not None:
            self._auto_plan = None
            self.auto_result.setText("the settings changed: search again")
        self._update_mesh_summary()

    # ----------------------------------------------------------- the pipes
    def pipe_network_config(self) -> PipeNetworkConfig:
        """The network to build: the engine's design when it is current, else the base.

        See :meth:`base_network_config` for the geometry; the design adds the header
        sizes, the lift, the orifices and possibly another collection.
        """
        base = self.base_network_config()
        if self._header_design is not None and self._header_key == self._engine_key(base):
            return self._header_design.config
        return base

    def _engine_key(self, base: PipeNetworkConfig) -> tuple:
        return (repr(base), round(self.flow.value(), 9), self.fluid.currentIndex(),
                round(self.circuit_pressure.value(), 9), self.header_tolerance.value(),
                self.header_velocity.value(), self.header_temperature.value())

    def header_design(self) -> HeaderDesign | None:
        """The engine's design if it is current (None: stale or never run)."""
        base = self.base_network_config()
        if self._header_design is not None and self._header_key == self._engine_key(base):
            return self._header_design
        return None

    def _engine_stale(self, *_args) -> None:
        if self._header_design is not None and self.header_design() is None:
            self.engine_info.setText("the plant changed: Size the headers, or Build mesh")

    def run_header_engine(self) -> HeaderDesign | None:
        """Size the headers for the current plant (see the Header engine section)."""
        from PySide6.QtWidgets import QApplication

        base = self.base_network_config()
        if base.split_mode != SPLIT_HYDRAULIC:
            self.engine_info.setText("the flow split is imposed: the headers keep the "
                                     "base size (choose 'From the network hydraulics')")
            return None
        cyl = self.cylinder()
        self.engine_info.setText("sizing the headers...")
        QApplication.processEvents()
        try:
            design = design_headers(
                base, (cyl.center_x, cyl.center_y), self.pipe_mass_flow(),
                self.circuit_fluid(), tolerance=self.header_tolerance.value() / 100.0,
                max_velocity=self.header_velocity.value(),
                temperature=self.header_temperature.value() + 273.15,
                pressure=self.circuit_pressure_pa())
        except ValueError as exc:
            self.engine_info.setText(f"cannot size: {exc}")
            return None
        self._header_design = design
        self._header_key = self._engine_key(base)
        chosen = design.config.collection
        if chosen != base.collection:
            # the engine found the other collection better: say it where it is set
            index = self.pipe_collection.findData(chosen)
            self.pipe_collection.blockSignals(True)
            self.pipe_collection.setCurrentIndex(index)
            self.pipe_collection.blockSignals(False)
            self._header_key = self._engine_key(self.base_network_config())
        sizing = design.sizing
        brief = [f"{chosen}: headers {min(sizing.sizes.values()) * 1000:.0f}-"
                 f"{max(sizing.sizes.values()) * 1000:.0f} mm, {design.lift * 1000:.0f} mm "
                 f"into the sand",
                 f"{sizing.delta_p:.0f} Pa, header velocity up to "
                 f"{sizing.max_velocity:.1f} m/s, flow on target within "
                 f"{100 * sizing.maldistribution:.1f} %"
                 + (" with orifices" if sizing.orifices_needed else ""),
                 f"{INFO} details" + ("" if sizing.feasible else " - not feasible")]
        self.engine_info.setText("\n".join(brief))
        tooltip = rich_lines(design.details())
        self.engine_info.setToolTip(tooltip)
        self._engine_caption.setToolTip(tooltip)
        self._engine_caption.setText(f"Design {INFO}")
        return design

    def base_network_config(self) -> PipeNetworkConfig:
        """The buried-pipe network the panel describes, in the current vessel.

        The active band is the *storage* band of the cylinder: the risers span the
        sand (and not the insulation slabs under and over it), which is what makes the
        riser length and the bed volume of the module the physical ones.  The wall
        roughness is the circuit's, falling back to the tube material's.
        """
        cyl = self.cylinder()
        roughness = self.circuit_roughness.value()
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
            duct_diameter=0.15,
            insulated_headers=self.pipe_insulated.isChecked(),
            azimuth_in=self.pipe_azimuth_in.value(),
            azimuth_out=self.pipe_azimuth_out.value(),
            split_mode=self.pipe_split.currentData(),
            n_sectors=int(self.pipe_sectors.value()),
            design_flow=max(float(self.flow.value()), 1e-6),
            design_temperature=self.header_temperature.value() + 273.15)

    def pipe_mass_flow(self) -> float:
        """Total mass flow of the gas circuit [kg/s]."""
        return float(self.flow.value())

    def circuit_fluid(self) -> Fluid:
        """The gas the circuit is filled with.

        The pressure is *not* applied here: the loop is handed the gas and the absolute
        pressure and applies it itself (:meth:`~src.solver.fluid.Fluid.at_pressure`),
        so there is one place that decides what the density is.
        """
        return self.fluid.currentData()

    def circuit_pressure_pa(self) -> float:
        """Absolute pressure of the loop [Pa]."""
        return float(self.circuit_pressure.value() * 1e5)

    def fan_efficiency_fraction(self) -> float:
        """Blower efficiency as a fraction [-]."""
        return max(self.fan_efficiency.value() / 100.0, 1e-3)

    def pipe_network(self):
        """The network the window built and painted (None until then)."""
        return self._pipe_network

    def pipe_surface_power(self) -> float:
        """Power the network's own wetted surface carries at the design power [W/cm²].

        The surface is the geometric ``pi d L`` of the network's runs - the risers *and*
        the headers, exactly the surface :meth:`PipeNetwork.paint` marks - because that is
        the surface the gas actually hands the power to the sand across.
        """
        network = self._pipe_network
        if network is None:
            return float("nan")
        return pipe_surface_power_w_cm2(self.power.value() * 1000.0, network.total_area)

    def set_pipe_network(self, network, report=None, message: str = "") -> None:
        """Adopt the network the window built on the mesh it owns and report it."""
        self._pipe_network = network
        if network is None:
            self.pipe_info.setText(message or "build the mesh: it paints the network")
            self._set_pipe_details("")
            self.circuit_info.setText(message or "build the network to size it")
            return
        problems = [problem for problem in network.config.validate()
                    if problem.startswith(WARNING)]
        problems.extend(network.validate())
        lines = network.summary().splitlines()
        if report is not None:
            lines.extend(report.summary().splitlines())
        lines.append(self.circuit_line())
        lines.extend("check: " + problem for problem in problems)
        if message:
            lines.append(message)
        # the network and the paint report both carry the design notes: each once
        details = list(dict.fromkeys(line.strip() for line in lines if line.strip()))
        notes = sum(line.startswith(("note:", "check:")) for line in details)
        # two lines in the form, the whole report behind the info mark
        brief = [f"{network.n_risers} risers, {network.total_area:.0f} m2 of tube "
                 f"({network.riser_area:.0f} m2 on the risers), flow spread "
                 f"{network.path_spread():.3f}x"]
        if report is not None:
            brief.append(f"{report.cells:,} tube cells, {report.area:.0f} m2 painted")
        brief.append(f"{INFO} details" + (f" - {notes} notes" if notes else ""))
        self.pipe_info.setText("\n".join(brief))
        self._set_pipe_details(rich_lines(details))
        self.circuit_info.setText(self._circuit_summary())

    def _set_pipe_details(self, tooltip: str) -> None:
        """The full network report behind the info mark of the Network row."""
        self.pipe_info.setToolTip(tooltip)
        self._pipe_caption.setToolTip(tooltip)
        self._pipe_caption.setText(f"Network {INFO}" if tooltip else "Network")

    def circuit_line(self) -> str:
        """The mean heat flux through the tube wall at the rated power.

        A gas-heated tube is limited by the gas film, not by a sheath rating: the 3-8
        W/cm2 window of an immersion element does not apply to it, so the flux is
        reported, not judged.
        """
        value = self.pipe_surface_power()
        if value != value:                        # NaN: no network yet
            return "wall heat flux: no network"
        return (f"wall heat flux {10.0 * value:.2f} kW/m2 at {self.power.value():.0f} kW "
                f"rated")

    def _circuit_summary(self) -> str:
        """The circuit as a whole: the surface power and the power per riser."""
        network = self._pipe_network
        if network is None:
            return "build the network to size it"
        value = self.pipe_surface_power()
        per_riser = self.power.value() / max(network.n_risers, 1)
        return (f"{self.power.value():.0f} kW over {network.total_area:.1f} m2 of tube "
                f"wall = {10.0 * value:.2f} kW/m2\n"
                f"{per_riser:.2f} kW per riser ({network.n_risers} risers), "
                f"{self.flow.value():.2f} kg/s of {self.fluid.currentText()} at "
                f"{self.circuit_pressure.value():.2f} bar")

    def _update_power(self) -> None:
        """Re-size the circuit summary on the network's surface."""
        self.circuit_info.setText(self._circuit_summary())

    # ------------------------------------------------------------ the mesh
    def _update_mesh_summary(self) -> None:
        """The Mesh tab's summary: the recipe, and the tree once one is built.

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
        lx, ly, lz = self.domain()
        plan_sizes = [band.size for band in plan.bands]
        heights = [band.height for band in plan.bands]
        dz = plan.dz if plan.anisotropic else plan.physical_size
        lines = [f"box {lx:.2f} x {ly:.2f} x {lz:.2f} m, finest leaf "
                 f"{plan.physical_size * 1000:.0f} x {dz * 1000:.0f} mm (plan x height), "
                 f"{len(plan.bands)} regions"]
        well = self.well_model_check(min(plan_sizes))
        if well:
            lines.append(well)
        mesh = self._adaptive_mesh
        if mesh is None:
            lines.append("press Build mesh to count the leaves")
            self.memory_info.setText("-")
        else:
            histogram = sorted(mesh.size_histogram().items(), key=lambda item: -item[1])
            shown = ", ".join(f"{a * 1000:.0f}x{b * 1000:.0f} mm: {count:,}"
                              for (a, b), count in histogram[:4])
            more = f" (+{len(histogram) - 4} more sizes)" if len(histogram) > 4 else ""
            lines.append(f"{mesh.n_cells:,} leaves - {shown}{more}")
            if self._budget_short:
                lines.append(f"the cell budget stopped the refinement before the "
                             f"{min(plan_sizes) * 1000:.0f} x {min(heights) * 1000:.0f} mm "
                             f"the regions ask for: raise it")
            self.memory_info.setText(f"~{mesh.n_cells * 98 / 1e6:.1f} MB of fields")
        self.mesh_info.setText("\n".join(lines))

    def well_model_check(self, finest_plan: float) -> str:
        """A warning when the leaf around a riser is too small for the well model.

        The tube is a line in its cell and the cell's temperature is the bed's at the
        equivalent radius ``0.198 h`` (Peaceman): that radius has to be outside the tube,
        so the plan edge of the cell must exceed ``d / (2 * 0.198) = 2.53 d``.
        """
        diameter = self.pipe_diameter.value()
        smallest = 0.5 * diameter / 0.198
        sand = self.cylinder().r_storage / max(self.cells_storage.value(), 1)
        if nearest_leaf(sand, finest_plan) + 1e-9 < smallest:
            return (f"the bed's leaves ({nearest_leaf(sand, finest_plan) * 1000:.0f} mm) are "
                    f"smaller than the {smallest * 1000:.0f} mm the well model of a "
                    f"{diameter * 1000:.0f} mm tube needs: the tube-to-bed resistance is "
                    f"then overstated - use fewer cells across the storage")
        return ""

    def adaptive_plan(self) -> AdaptivePlan:
        """The tree the current targets ask for: the *regions* as boxes of an octree.

        The boxes come from :func:`src.analysis.mesh_plan.active_regions` - the sand,
        the insulation ring and the two slabs, the shell, the casing the ambient film
        sits on and the box of the buried pipes - so the tree refines the active model
        and nothing else.  The finest leaf is the finest region's target (snapped to a
        power of two of the box), and the cell budget caps the leaves the build makes.

        When the search converged, its own plan is returned unchanged: the boxes the
        search measured are the ones whose answer converged.
        """
        if self._auto_plan is not None:
            return self._auto_plan
        raw = region_bands(self.mesh_regions())
        side, _side, tall = self.domain()

        def finest(extent: float, target: float) -> tuple[int, float]:
            """The finest cell along one direction: the finest region's target."""
            level = int(np.clip(np.round(np.log2(extent / target)), MIN_TREE_LEVEL,
                                MAX_TREE_LEVEL))
            return 1 << level, extent / (1 << level)

        # the finest leaf is the one the finest region asks for, separately in plan and
        # in height; the budget caps the leaves the rounds actually make (build_mesh)
        n_xy, dx = finest(side, min(band.size for band in raw))
        n_z, dz = finest(tall, min(band.height for band in raw))
        # a leaf is a power of two of the finest cell: every target is snapped to the
        # largest such size within LEAF_TOLERANCE of it, not to the next one below -
        # otherwise a 200 mm request on 101.6/203 mm leaves refines the whole bed to
        # 101.6 mm, twice finer than asked and four times the leaves of a layer
        bands = tuple(replace(band, size=nearest_leaf(band.size, dx),
                              size_z=nearest_leaf(band.height, dz)) for band in raw)
        return AdaptivePlan(n_finest=n_xy, physical_size=dx, bands=bands, n_z=n_z, dz=dz)

    def mesh_regions(self):
        """The active regions of the model, each with the cell size the panel asks for.

        The size per region is the finer of the manual count and the a priori plan of
        ``src/analysis/mesh_plan.py`` (see :meth:`planned`).  The pipes are the one
        region the *network* describes - its bundle, its active band and its own
        diameter - so the box is derived from :meth:`pipe_network_config`.
        """
        cyl = self.cylinder()
        insulation_cells = max(self.cells_insulation.value(), 1)
        sand = self.planned("storage", cyl.r_storage / max(self.cells_storage.value(), 1))
        layer = cyl.height / max(self.bed_layers.value(), 1)
        across = self.planned("insulation_radial",
                              cyl.insulation_thickness / insulation_cells)
        # plan edges: the bed's in the bed and under the slabs, the insulation's in the
        # ring (a thin steel shell needs no cells across it: the painter keeps it one
        # cell thick whatever the mesh)
        targets = {"sand": sand, "insulation_radial": across, "shell": across,
                   "slab_bottom": sand, "slab_top": sand, "casing": sand,
                   "ground": 2.0 * sand}
        # heights: the bed's layer along the bed and the ring, the thickness of each
        # horizontal layer over the insulation count where one is
        heights = {
            "sand": layer, "insulation_radial": layer, "shell": layer,
            "slab_bottom": self.planned("slab_bottom",
                                        cyl.insulation_slab_bottom / insulation_cells),
            "slab_top": self.planned("slab_top", cyl.insulation_slab_top / insulation_cells),
            "casing": min(across, layer),
            # the soil only spreads the heat of the pad downwards: a few layers
            "ground": max(layer, cyl.ground_depth / 6.0),
        }
        return active_regions(cyl, targets, heights=heights)

    def _shape(self, center: tuple[float, float] = (0.0, 0.0)) -> CylinderGeometry:
        return CylinderGeometry(
            center_x=center[0], center_y=center[1],
            base_z=self.soil.value() + self.base_z.value(), height=self.height.value(),
            ground_depth=self.soil.value(),
            r_storage=self.radius.value(),
            insulation_thickness=self.insulation_thickness.value(),
            shell_thickness=self.shell_thickness.value(),
            insulation_slab_bottom=self.slab_bottom.value(),
            insulation_slab_top=self.slab_top.value(),
            roof_angle_deg=self.roof_angle.value(),
            steel_slab_top=self.steel_slab.value(),
            fill_cone_with_sand=self.fill_cone.isChecked(),
            enable_cone_roof=self.enable_roof.isChecked(),
            foundation_margin=self.foundation_margin.value(),
        )

    def domain(self) -> tuple[float, float, float]:
        """The box the vessel sits in [m]: the vessel plus :data:`AIR_MARGIN` of air.

        The air is excluded from the problem, so the box only has to hold the vessel and
        one layer of excluded leaves for the outer film to act on; a box the user sets
        was one more thing to keep consistent (and a centre set by hand is what put the
        pipes off the vessel).
        """
        shape = self._shape()
        side = 2.0 * (shape.envelope_radius(shape.foundation_margin) + AIR_MARGIN)
        return (side, side, shape.z_cone_apex + AIR_MARGIN)

    def build_mesh(self) -> AdaptiveMesh:
        """The octree of the current targets (:meth:`adaptive_plan`).

        The tree is kept for the Mesh tab's summary: its leaf count, its levels and its
        leaf edges are measured, never guessed.
        """
        plan = self.adaptive_plan()
        # the budget is a cap on the leaves: the rounds go coarse to fine over the whole
        # tree, so a budget that stops them leaves every region one level short rather
        # than some regions done and others untouched
        mesh = AdaptiveMesh.from_plan(plan, max_cells=int(self.max_cells.value()))
        self._adaptive_mesh = mesh
        self._budget_short = bool(
            float(mesh.extent[:, 0].min()) > 1.01 * min(band.size for band in plan.bands)
            or float(mesh.extent[:, 2].min()) > 1.01 * min(band.height
                                                           for band in plan.bands))
        self._update_mesh_summary()
        return mesh

    def auto_mesh_settings(self) -> dict:
        """Tolerances of the automatic mesh search (see analysis/convergence.py)."""
        return {"delta_temperature": self.auto_dt.value(),
                "delta_power": self.auto_dp.value() / 100.0,
                "max_levels": int(self.auto_levels.value()),
                "refine": self.auto_refine.value(),
                "max_cells": int(self.max_cells.value())}

    def set_plan_targets(self, targets: dict[str, float], note: str = "") -> None:
        """Adopt the a priori plan (analysis/mesh_plan.py): the boxes never get coarser.

        A region whose own conduction length, or whose convective sub-layer 2k/h, is
        finer than the manual target takes the plan value: that is what makes the
        surface flux - and therefore the losses - converge.
        """
        self._plan_targets = dict(targets)
        if note:
            self.plan_info.setText(note)
        self._update_mesh_summary()

    def planned(self, name: str, manual: float) -> float:
        """The tighter of the manual target and the plan target [m]."""
        target = self._plan_targets.get(name)
        return manual if target is None else min(manual, target)

    def auto_spec(self) -> AdaptivePlan | None:
        """The plan the search adopted (None until it converges)."""
        return self._auto_plan

    def set_auto_spec(self, plan: AdaptivePlan | None, message: str = "") -> None:
        """Adopt the plan chosen by the search (None clears it).

        The plan is kept *as the search left it*: re-deriving it would put the box and
        the bands back through the cell budget, and the tree would not be the one whose
        answer converged.
        """
        self._auto_plan = plan if isinstance(plan, AdaptivePlan) else None
        self.auto_result.setText(message or ("adopted" if plan else "not run yet"))
        self._update_mesh_summary()

    # ------------------------------------------------------------ the vessel
    def cylinder(self) -> CylinderGeometry:
        """The vessel, centred in its box."""
        side = self.domain()[0]
        return self._shape((0.5 * side, 0.5 * side))

    def heaters(self) -> HeaterConfig:
        """The plant's electric power, as the geometry records it."""
        return HeaterConfig(power_total=self.power.value())

    def apply_geometry(self, battery) -> None:
        """Attach the built config trees to a BatteryGeometry."""
        battery.cylinder = self.cylinder()
        battery.heaters = self.heaters()

    def set_mesh_info(self, text: str, memory: str) -> None:
        self.mesh_info.setText(text)
        self.memory_info.setText(memory)
