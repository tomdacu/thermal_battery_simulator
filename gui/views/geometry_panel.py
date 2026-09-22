"""Geometry panel: cylinder, insulation, the gas circuit, the pipes and the mesh.

Two of the sub-tabs are the *plant* rather than the vessel: the **Heaters** tab is the
gas circuit the electric resistors heat (``src/solver/fluid.py``), and the **Pipes**
tab is the buried network the gas runs through (``src/core/pipe_network.py``).  They
are the same circuit seen from its two ends: the resistors put power into the gas, the
gas carries it to the pipe walls and the sand takes it up; on discharge the same loop
carries it back out to the exchanger, which sits on the circuit too.  There is no
heater element inside the bed and no second heat exchanger inside the vessel.
"""
from __future__ import annotations


from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from src.analysis.convergence import AdaptivePlan
from src.analysis.mesh_plan import active_regions, region_bands, tree_resolution
from src.constants import (PIPE_SURFACE_POWER_LIMIT_W_CM2,
                           PIPE_SURFACE_POWER_MIN_W_CM2)
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.mesh import Mesh3D
from src.core.pipes import pipe_surface_power_w_cm2
from src.core.refinement import Band, GridSpec
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
)
from src.solver.fluid import Fluid

from ..widgets import FormPanel, button, check, combo, double_spin, hint, int_spin


class GeometryPanel(QWidget):
    """All geometry controls; exposes ``build_*`` accessors returning src objects."""

    mesh_changed = pyqtSignal()
    auto_mesh_requested = pyqtSignal()
    #: the Pipes tab asks the window - which owns the mesh - to build and paint the
    #: network; the window answers through :meth:`set_pipe_network`
    pipe_network_requested = pyqtSignal()

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
        layout = QVBoxLayout(self)
        self._auto_spec: GridSpec | None = None
        #: the tree plan the search adopted, when it ran on a tree (None: the a priori one)
        self._auto_plan: AdaptivePlan | None = None
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
        self._build_circuit_tab()
        self._build_pipes_tab()
        self._build_mesh_tab()

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

    def _build_circuit_tab(self) -> None:
        """The plant's heat source: the electric resistors and the gas circuit.

        The resistors do not sit in the sand: they heat the gas of a closed loop that
        runs through the buried pipes (``src/solver/fluid.py``), and the gas hands the
        power to the bed across the pipe walls.  Everything on this tab is a property
        of the *circuit* - the power in, the gas in it, the flow it carries, the
        pressure it runs at, the temperature the gas comes back to the resistors at,
        the blower and the wall - and the summary below is computed on the network the
        Pipes tab built, because that is the surface the power crosses.
        """
        panel = FormPanel()
        panel.add_hint("The electric resistors heat the GAS, in a tank on the circuit; "
                       "the hot gas enters the distributor, rises through the buried "
                       "pipes, gives its heat to the sand and leaves from the collector. "
                       "On discharge the flow reverses: the gas comes back cold from the "
                       "exchanger, takes the heat from the bed and leaves hot. The bed "
                       "is heated through the tube walls, never by an element inside it.")
        self.power = panel.add("Total power [kW]", double_spin(
            5.0, 0.0, 100000.0, 10.0, 1, on_change=self._update_power,
            tooltip="Electric power of the plant: the resistors put it into the gas, "
                    "and the analyses that do not march the loop read it as the bed's "
                    "own source"))
        self.fluid = panel.add("Gas", combo(self.FLUIDS, 0, self._update_power))
        self.offset_bottom = panel.add("Source from bottom [m]", double_spin(
            0.0, 0.0, 2.0, 0.1, 2,
            tooltip="Band of the storage the lumped bed source covers: an analysis that "
                    "does not march the loop deposits the power over the storage volume "
                    "between these two offsets"))
        self.offset_top = panel.add("Source from top [m]", double_spin(
            0.0, 0.0, 2.0, 0.1, 2))
        self.flow = panel.add("Mass flow [kg/s]", double_spin(
            0.5, 0.0, 200.0, 0.05, 3, on_change=self._update_power,
            tooltip="Total mass flow of the gas loop: the network's branches split it "
                    "by the rule of the Pipes tab"))
        self.circuit_pressure = panel.add("Circuit pressure [bar]", double_spin(
            1.01325, 0.1, 100.0, 0.5, 5, special="1 bar",
            tooltip="Absolute pressure of the loop: at a fixed mass flow the density "
                    "rises with it, so the film coefficient grows and the pressure "
                    "drop falls - the cheap design lever of a gas loop"))
        self.return_t = panel.add("Return to the resistors [°C]", double_spin(
            0.0, 0.0, 700.0, 5.0, 1, special="auto",
            tooltip="Temperature the gas enters the loop at: 0 degC is the temperature "
                    "at which the exchanger returns it on discharge, and 'auto' lets "
                    "the loop solve its own balance from the power instead (the closed "
                    "circuit of a charge)"))
        self.fan_efficiency = panel.add("Fan efficiency [%]", double_spin(
            70.0, 10.0, 100.0, 5.0, 0,
            tooltip="Blower efficiency: the electric power of the fan is the shaft "
                    "power over it"))
        self.circuit_roughness = panel.add("Wall roughness [um]", double_spin(
            0.0, 0.0, 2000.0, 5.0, 1, special="from the tube material",
            tooltip="Absolute roughness of the circuit's walls (tubes, headers and "
                    "ducts): leave it at the minimum to take the tube material's own "
                    "value from the Pipes tab"))
        self.circuit_info = panel.add("Circuit", hint("build the network to size it"))
        panel.add_hint("The summary is the network's: the wetted surface is the "
                       "geometric pi d L of its runs, and the surface power is the "
                       "total power over that surface.  A tube wall is rated by what "
                       "it delivers per square centimetre, exactly as an immersion "
                       "element is, and the 3-8 W/cm2 window is the same one.")
        self.tabs.addTab(panel, "Heaters")


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
            "The label of the tube and the default roughness of its wall: 15 um drawn, "
            "46 um commercial.  The circuit tab's wall roughness overrides it")
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
                       "is not axisymmetric and nothing leaves through the roof. The "
                       "same network is the whole exchange path - the gas charges the "
                       "bed through it (flow up the risers, hot collection at the top) "
                       "and discharges it through it (flow reversed, heat out to the "
                       "exchanger on the circuit) - so there is no second heat "
                       "exchanger inside the vessel. The mass flow, the pressure, the "
                       "gas and the return temperature are on the Heaters tab: they "
                       "are the circuit's.")
        self.tabs.addTab(panel, "Pipes")

    def pipe_network_config(self) -> PipeNetworkConfig:
        """The buried-pipe network the panel describes, in the current vessel.

        The active band is the *storage* band of the cylinder: the risers span the
        sand (and not the insulation slabs under and over it), which is what makes the
        riser length and the bed volume of the module the physical ones.  The wall
        roughness is the circuit's (Heaters tab), falling back to the tube material's.
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
        """Total mass flow of the gas circuit (Heaters tab) [kg/s]."""
        return float(self.flow.value())

    def circuit_fluid(self) -> Fluid:
        """The gas the circuit is filled with [-].

        The pressure is *not* applied here: the loop is handed the gas and the absolute
        pressure and applies it itself (:meth:`~src.solver.fluid.Fluid.at_pressure`),
        so there is one place that decides what the density is.
        """
        return self.fluid.currentData()

    def return_kelvin(self) -> float:
        """Temperature the gas comes back to the resistors at [K].

        The minimum of the slider is the special "auto" value: there the *loop* solves
        the inlet temperature from its own power balance, which is the closed circuit
        of a charge.
        """
        from src.units import c_to_k

        return c_to_k(self.return_t.value())

    def inlet_temperature(self) -> float | None:
        """The prescribed loop inlet [K], or ``None`` for the loop's own balance."""
        return None if self.return_t.value() <= 0.0 else self.return_kelvin()

    def circuit_pressure_pa(self) -> float:
        """Absolute pressure of the loop [Pa]."""
        return float(self.circuit_pressure.value() * 1e5)

    def fan_efficiency_fraction(self) -> float:
        """Blower efficiency as a fraction [-].

        Zero on the slider means an ideal blower: the panel writes 100% at the top of
        the range, so the loop never divides by a zero efficiency.
        """
        return max(self.fan_efficiency.value() / 100.0, 1e-3)

    def pipe_network(self):
        """The network the window built and painted (None until then)."""
        return self._pipe_network

    def pipe_surface_power(self) -> float:
        """Power the network's own wetted surface carries at the design power [W/cm²].

        The surface is the geometric ``pi d L`` of the network's runs - the risers *
        and* the headers, exactly the surface :meth:`PipeNetwork.paint` marks - because
        that is the surface the gas actually hands the power to the sand across.
        """
        network = self._pipe_network
        if network is None:
            return float("nan")
        return pipe_surface_power_w_cm2(self.power.value() * 1000.0, network.total_area)

    def set_pipe_network(self, network, report=None, message: str = "") -> None:
        """Adopt the network the window built on the mesh it owns and report it."""
        self._pipe_network = network
        if network is None:
            self.pipe_info.setText(message or "build the mesh, then the network")
            self.circuit_info.setText(message or "build the network to size it")
            return
        problems = [problem for problem in network.config.validate()
                    if problem.startswith(WARNING)]
        problems.extend(network.validate())
        lines = [network.summary()]
        if report is not None:
            lines.append(report.summary())
        lines.append(self._circuit_line())
        if problems:
            lines.append("check: " + "; ".join(problems))
        if message:
            lines.append(message)
        self.pipe_info.setText("\n".join(lines))
        self.circuit_info.setText(self._circuit_summary())

    def _circuit_line(self) -> str:
        """The surface power of the tubes, next to the window it is rated against."""
        value = self.pipe_surface_power()
        if value != value:                        # NaN: no network yet
            return "surface power: no network"
        state = ("in the 3-8 W/cm2 range"
                 if PIPE_SURFACE_POWER_MIN_W_CM2 <= value <= PIPE_SURFACE_POWER_LIMIT_W_CM2
                 else f"outside the {PIPE_SURFACE_POWER_MIN_W_CM2:.0f}-"
                      f"{PIPE_SURFACE_POWER_LIMIT_W_CM2:.0f} W/cm2 range")
        return (f"surface power: {value:.2f} W/cm2 of tube wall "
                f"({state}) at {self.power.value():.1f} kW")

    def _circuit_summary(self) -> str:
        """The circuit as a whole: the surface power and the power per riser."""
        network = self._pipe_network
        if network is None:
            return "build the network to size it"
        value = self.pipe_surface_power()
        per_riser = self.power.value() / max(network.n_risers, 1)
        return (f"{self.power.value():.1f} kW over {network.total_area:.1f} m2 of tube "
                f"wall = {value:.2f} W/cm2\n"
                f"{per_riser:.3f} kW per riser ({network.n_risers} risers), "
                f"{self.flow.value():.2f} kg/s of {self.fluid.currentText()} at "
                f"{self.circuit_pressure.value():.2f} bar")

    def _build_mesh_tab(self) -> None:
        panel = FormPanel()
        self.refined = panel.add("Refined mesh", check(
            "cells placed where the gradients are", True, on_toggle=self._mesh_mode))
        self.adaptive = panel.add("Adaptive mesh", check(
            "an octree of leaves instead of a graded grid", True, on_toggle=self._mesh_mode,
            tooltip="Default: the same physical targets, refined as boxes of an octree "
                    "instead of as bands of three axes.  The summary then reports leaves "
                    "and levels rather than cells per axis.  Uncheck for the graded grid, "
                    "which stays as the reference road."))
        self.spacing = panel.add("Cell size (uniform) [m]",
                                 double_spin(0.2, 0.02, 1.0, 0.05, 3,
                                             on_change=self.mesh_changed.emit))
        self.cells_storage = panel.add("Cells across storage", int_spin(
            10, 2, 200, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the storage radius at the wall"))
        self.cells_insulation = panel.add("Cells across insulation", int_spin(
            3, 1, 50, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the radial insulation thickness"))
        self.cells_sheath = panel.add("Cells across the tube wall", int_spin(
            2, 1, 20, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the outer diameter of the buried pipes: the tube "
                    "wall is the heat-transfer surface now, and a tube thinner than "
                    "the local cell is painted as one cell whatever its real size"))
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
            tooltip="Ceiling on the realised cell size.  'auto' = the air outside the "
                    "vessel takes the coarsest size the active regions ask for, which "
                    "is the coarsest cell the model needs at all"))
        self.max_cells = panel.add("Cell budget", int_spin(
            400_000, 10_000, 20_000_000, 50_000, tooltip="Every target is scaled up "
            "by a common factor to fit this budget", on_change=self._mesh_mode))
        self.plan_info = panel.add("Active regions", hint(
            "the materials and the film coefficients decide the targets"))
        self.mesh_info = panel.add("Grid", hint("build the mesh to see the grid"))
        self.memory_info = panel.add("Memory", hint("-"))
        panel.add_hint("The refinement covers the *active* model only: the sand, the "
                       "insulation, the shell and the wall of the pipes.  The air "
                       "around the vessel is excluded from the problem, so no band is "
                       "spent on it - the leaves out there stay at the coarse level "
                       "the octree's own balance gives them.  Refined (graded) mesh: "
                       "the targets are physical and do not depend on the domain size. "
                       "Uniform mesh: one cell size everywhere, box snapped to a whole "
                       "number of cells.")
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
            "search before building", False,
            tooltip="Off by default because the search is expensive: on a tree it builds "
                    "and solves several grids (the first probe alone can be five times the "
                    "planned mesh) before it can adopt one.  Build the planned mesh in a "
                    "few seconds with the button, and press *Find the mesh* when you want "
                    "the convergence evidence."))
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
                       self.max_cells):
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
        """The tree the current targets ask for: the *regions* as boxes of an octree.

        The boxes come from :func:`src.analysis.mesh_plan.active_regions` - the sand,
        the insulation ring and the two slabs, the shell, the casing the ambient film
        sits on and the box of the buried pipes - so the tree refines the active model
        and nothing else: a leaf in the air around the vessel would be a cell pinned at
        the ambient temperature, and the octree leaves it at the coarse level its own
        2:1 balance gives it.  The box and its resolution come from the cell budget
        (:func:`src.analysis.mesh_plan.tree_resolution`), which is also the floor under
        every leaf the search then refines.

        When the search ran on a tree its own plan is returned unchanged: a tree has no
        per-axis spec to re-derive the boxes from, and the boxes the search measured
        are the ones that converged.
        """
        if self._auto_plan is not None:
            return self._auto_plan
        lx, ly, lz = (self.domain_lx.value(), self.domain_ly.value(),
                      self.domain_lz.value())
        n_finest, physical_size = tree_resolution((lx, ly, lz), int(self.max_cells.value()))
        return AdaptivePlan(n_finest=n_finest, physical_size=physical_size,
                            bands=region_bands(self.mesh_regions()))

    def mesh_regions(self):
        """The active regions of the model, each with the cell size the panel asks for.

        The mixture per region is the one the Mesh tab shows: the manual counts, never
        coarser than the a priori plan of ``src/analysis/mesh_plan.py`` (see
        :meth:`planned`).  The pipes are the one region the *network* describes - its
        bundle, its active band and its own diameter - so the box is derived from
        :meth:`pipe_network_config` and not from the vessel.
        """
        cyl = self.cylinder()
        targets = {
            "sand": self.planned("storage", cyl.r_storage / max(self.cells_storage.value(), 1)),
            "insulation_radial": self.planned(
                "insulation_radial",
                cyl.insulation_thickness / max(self.cells_insulation.value(), 1)),
            "slab_bottom": self.planned(
                "slab_bottom",
                cyl.insulation_slab_bottom / max(self.cells_insulation.value(), 1)),
            "slab_top": self.planned(
                "slab_top",
                cyl.insulation_slab_top / max(self.cells_insulation.value(), 1)),
            "shell": self.planned(
                "shell", cyl.shell_thickness / max(self.cells_insulation.value(), 1)),
            "pipe_wall": (self.pipe_diameter.value()
                          / max(self.cells_sheath.value(), 1)),
        }
        targets["casing"] = max(targets.values())
        config = self.pipe_network_config()
        reach = config.inner_radius
        pipe_box = ((cyl.center_x - reach, cyl.center_y - reach, config.z_bottom),
                    (cyl.center_x + reach, cyl.center_y + reach, config.z_top))
        return active_regions(cyl, targets, pipe_box=pipe_box)

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
        """Re-size the circuit summary: the network's surface, not a sheath's."""
        self.circuit_info.setText(self._circuit_summary())

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

    def auto_spec(self) -> GridSpec | AdaptivePlan | None:
        """The request the search adopted (None until it converges).

        A graded search answers with a :class:`GridSpec`, a tree search with an
        :class:`AdaptivePlan`: both are the same physical targets in the vocabulary of
        the mesh that was measured.
        """
        return self._auto_plan if self._auto_plan is not None else self._auto_spec

    def set_auto_spec(self, spec: GridSpec | AdaptivePlan | None, message: str = "") -> None:
        """Adopt the mesh chosen by the search (None clears it).

        The plan is kept *as the search left it*: re-deriving it from the spec would put
        the box and the bands back through the cell budget, and the tree would not be the
        one whose answer converged.
        """
        if isinstance(spec, AdaptivePlan):
            self._auto_plan, self._auto_spec = spec, None
        else:
            self._auto_plan, self._auto_spec = None, spec
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
        """Physical refinement targets of the *graded* (structured) road.

        The graded grid covers the whole domain - a Cartesian grid has no way to leave
        a corner out - so its bands are the active model (the storage core, the two
        insulation slabs, the insulation ring, the shell ring and the pipe-header
        junctions) plus **one cap band** over each axis.  The cap is not a far field to
        refine: the air outside the envelope is excluded from the problem, so the cap
        carries the *coarsest* target the active regions ask for (or the explicit
        "largest cell" rail) and the cells out there are no finer than the bed's own.
        The radial bands are per-axis arguments, which is also why this road cannot
        refine the tube wall the way the tree does: a 25 mm band over the bundle would
        be paid along the whole axis and the budget would then coarsen the bed itself.
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
        shell = self.planned("shell",
                            cyl.shell_thickness / max(self.cells_insulation.value(), 1))
        cap = self.max_cell.value() or max(fine, insulation, shell)

        # vertical bands: storage + the two insulation slabs, then the cap
        z = (Band(cyl.z_slab_bottom_start, cyl.z_storage_start,
                  self.planned("slab_bottom", insulation)),
             Band(cyl.z_storage_start, cyl.z_storage_end, fine),
             Band(cyl.z_slab_top_start, cyl.z_slab_top_end,
                  self.planned("slab_top", insulation)),
             Band(0.0, lz, cap))
        # the tube-header junctions: the gas turns there and the surface is singular,
        # so the two header elevations carry the refinement the Pipes tab asks for
        junction = tuple(Band(low, high, target)
                         for low, high, target in self.pipe_junction_bands())
        if junction:
            z = z + junction
        # radial bands: the storage core, the insulation ring, the shell ring, the cap
        radial = []
        for center, extent in ((cyl.center_x, lx), (cyl.center_y, ly)):
            bands = [Band(max(center - cyl.r_storage, 0.0),
                          min(center + cyl.r_storage, extent), fine)]
            # the insulation and the shell are *annuli*: a single band across the whole
            # diameter would ask for insulation cells inside the storage as well
            for low, high in ((center - cyl.r_shell, center - cyl.r_insulation),
                              (center + cyl.r_insulation, center + cyl.r_shell)):
                if high - low > 0:
                    bands.append(Band(max(low, 0.0), min(high, extent), shell))
            for low, high in ((center - cyl.r_insulation, center - cyl.r_storage),
                              (center + cyl.r_storage, center + cyl.r_insulation)):
                if high - low > 0:
                    bands.append(Band(max(low, 0.0), min(high, extent), insulation))
            bands.append(Band(0.0, extent, cap))
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
        """The plant's heat source as the bed sees it: the power and its band."""
        return HeaterConfig(
            power_total=self.power.value(),
            offset_bottom=self.offset_bottom.value(), offset_top=self.offset_top.value(),
        )

    def apply_geometry(self, battery) -> None:
        """Attach the built config trees to a BatteryGeometry."""
        battery.cylinder = self.cylinder()
        battery.heaters = self.heaters()

    def set_mesh_info(self, text: str, memory: str) -> None:
        self.mesh_info.setText(text)
        self.memory_info.setText(memory)
