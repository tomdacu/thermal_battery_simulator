"""Geometry panel: cylinder, insulation, the gas circuit, the pipes and the mesh.

Two of the sub-tabs are the *plant* rather than the vessel: the **Gas circuit** tab is the
closed loop the electric resistors heat (``src/solver/fluid.py``), and the **Pipes** tab
is the buried network the gas runs through (``src/core/pipe_network.py``).  They are the
same circuit seen from its two ends: the resistors put power into the gas, the gas
carries it to the pipe walls and the sand takes it up; on discharge the same loop carries
it back out to the exchanger, which sits on the circuit too.  There is no heater element
inside the bed and no second heat exchanger inside the vessel.

The mesh is the adaptive octree: the a priori plan of every active region (the sand,
the insulation, the shell, the pipe wall) becomes a box the tree refines, and the air
around the vessel is excluded from the problem.
"""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QLabel, QTabWidget, QVBoxLayout, QWidget

from src.analysis.convergence import AdaptivePlan
from src.analysis.mesh_plan import active_regions, region_bands, tree_resolution
from src.constants import (PIPE_SURFACE_POWER_LIMIT_W_CM2,
                           PIPE_SURFACE_POWER_MIN_W_CM2)
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import CylinderGeometry, HeaterConfig
from src.core.pipe_network import (COLLECTION_CENTRAL, COLLECTION_DIRECT,
                                   COLLECTION_REVERSE, COLLECTION_TWO_LEVEL,
                                   LAYOUT_GRID, LAYOUT_RADIAL, LAYOUT_RINGS,
                                   LAYOUT_SPIRAL, LAYOUT_STAGGERED, PIPE_CARBON,
                                   PIPE_STAINLESS, SPLIT_EQUAL, SPLIT_PATH,
                                   SPLIT_RING, SPLIT_SECTOR, WARNING,
                                   PipeNetworkConfig)
from src.core.pipes import pipe_surface_power_w_cm2
from src.solver.fluid import Fluid

from ..widgets import FormPanel, button, check, combo, double_spin, hint, int_spin


class GeometryPanel(QWidget):
    """All geometry controls; exposes accessors returning src objects."""

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
        #: the tree plan the automatic search adopted (None: the a priori one)
        self._auto_plan: AdaptivePlan | None = None
        self._plan_targets: dict[str, float] = {}
        self._pipe_network = None
        #: the tree the last build produced, for the Mesh tab's summary
        self._adaptive_mesh: AdaptiveMesh | None = None
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
        self.enable_roof = panel.add("Conical roof",
                                     check("enable", True, "Cone-shaped steel roof"))
        self.roof_angle = panel.add("Roof angle [deg]", double_spin(15.0, 0.0, 45.0, 1.0, 1))
        self.steel_slab = panel.add("Steel slab [m]", double_spin(0.005, 0.0, 0.2, 0.005, 3))
        self.fill_cone = panel.add("Cone fill", check("fill with sand", False))
        panel.add_hint("The domain only has to contain the vessel: the air around it is "
                       "not simulated (the outer surface carries the ambient film), and "
                       "the apex of the roof must stay below Lz - the build aborts "
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
        pressure it runs at, the blower and the wall - and the summary below is computed
        on the network the Pipes tab built, because that is the surface the power
        crosses.  Every analysis marches this loop: steady, losses and transient.
        """
        panel = FormPanel()
        panel.add_hint("The electric resistors heat the GAS, in a tank on the circuit; "
                       "the hot gas enters the distributor, rises through the buried "
                       "pipes, gives its heat to the sand and leaves from the collector. "
                       "On discharge the exchanger on the same circuit takes the heat "
                       "out of the gas (Analysis > Extraction). The bed is heated "
                       "through the tube walls, never by an element inside it.")
        self.power = panel.add("Total power [kW]", double_spin(
            5.0, 0.0, 100000.0, 10.0, 1, on_change=self._update_power,
            tooltip="Electric power of the resistors: the steady and losses analyses "
                    "put it into the gas; the transient takes its power profile from "
                    "Analysis > Power"))
        self.fluid = panel.add("Gas", combo(self.FLUIDS, 0, self._update_power))
        self.flow = panel.add("Mass flow [kg/s]", double_spin(
            0.5, 0.0, 200.0, 0.05, 3, on_change=self._update_power,
            tooltip="Total mass flow of the gas loop: the network's branches split it "
                    "by the rule of the Pipes tab"))
        self.circuit_pressure = panel.add("Circuit pressure [bar]", double_spin(
            1.01325, 0.1, 100.0, 0.5, 5, special="1 bar",
            tooltip="Absolute pressure of the loop: at a fixed mass flow the density "
                    "rises with it, so the film coefficient grows and the pressure "
                    "drop falls - the cheap design lever of a gas loop"))
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
                       "total power over that surface, rated against the 3-8 W/cm2 "
                       "window of a heated tube wall.")
        self.tabs.addTab(panel, "Gas circuit")

    def _build_pipes_tab(self) -> None:
        """Buried pipe network: the layout, the tube, the plumbing and the paint."""
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
            "The material of the tube and the default roughness of its wall: 15 um "
            "drawn, 46 um commercial.  The circuit tab's wall roughness overrides it")
        self.pipe_duct = panel.add("Duct d [m]", double_spin(0.15, 0.05, 0.6, 0.05, 3))
        self.pipe_insulated = panel.add("Insulated headers", check(
            "lag the distributor and the collector", False,
            tooltip="A lagged header carries the gas and exchanges nothing with the "
                    "bed, so the heat transfer surface is the risers alone"))
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
        panel.add_row(button("Rebuild the network on the mesh",
                             self.pipe_network_requested.emit,
                             "Voxelise the network on the current mesh and mark its "
                             "cells as pipes (Build mesh does it too)"))
        self.pipe_info = panel.add("Network", hint("build the mesh: it paints the network"))
        panel.add_hint("The risers are buried in the sand and the gas goes in from the "
                       "side at the bottom and out from the side at the top. The same "
                       "network is the whole exchange path - the gas charges the bed "
                       "through it and discharges it through it - and the film of every "
                       "pipe cell is computed by the gas loop from the flow, the gas and "
                       "the wall, on the Gas circuit tab.")
        self.tabs.addTab(panel, "Pipes")

    def _build_mesh_tab(self) -> None:
        panel = FormPanel()
        self.cells_storage = panel.add("Cells across storage", int_spin(
            10, 2, 200, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the storage radius"))
        self.cells_insulation = panel.add("Cells across insulation", int_spin(
            3, 1, 50, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the radial insulation thickness (and the slabs)"))
        self.cells_sheath = panel.add("Cells across the tube", int_spin(
            2, 1, 20, 1, on_change=self._mesh_mode,
            tooltip="Cells covering the outer diameter of the buried pipes: the tube "
                    "wall is the heat-transfer surface, and a tube thinner than the "
                    "local cell is painted as one cell whatever its real size"))
        self.max_cells = panel.add("Cell budget", int_spin(
            400_000, 10_000, 20_000_000, 50_000, on_change=self._mesh_mode,
            tooltip="Sets the finest leaf the octree may use: the whole box refined to "
                    "that leaf would hold this many cells"))
        self.plan_info = panel.add("Active regions", hint(
            "the materials and the film coefficients decide the targets"))
        self.mesh_info = panel.add("Mesh", hint("build the mesh to see it"))
        self.memory_info = panel.add("Memory", hint("-"))
        panel.add_hint("The mesh is an octree refined on the *active* model only: the "
                       "sand, the insulation, the shell and the pipe bundle, each to the "
                       "finer of the count above and the a priori plan (layer/N, 2k/h). "
                       "The air around the vessel is excluded from the problem, so its "
                       "leaves stay coarse.")
        separator = QLabel("Automatic mesh: refine until the answer stops moving")
        separator.setToolTip("Runs the steady solve on a sequence of finer trees and "
                             "stops when the storage mean temperature and the heat "
                             "leaving the battery change by less than the tolerances")
        panel.add_row(separator)
        self.auto_dt = panel.add("Temperature tolerance [K]",
                                 double_spin(2.0, 0.05, 100.0, 0.5, 2,
                                             tooltip="Largest change of the storage mean "
                                                     "temperature between two meshes"))
        self.auto_dp = panel.add("Power tolerance [%]",
                                 double_spin(2.0, 0.05, 50.0, 0.25, 2,
                                             tooltip="Largest relative change of the heat "
                                                     "leaving the battery"))
        self.auto_levels = panel.add("Levels", int_spin(
            4, 2, 8, 1, tooltip="Meshes tried, each one finer by the refine factor"))
        self.auto_refine = panel.add("Refine factor", double_spin(
            0.6, 0.2, 0.9, 0.05, 2, tooltip="Target scale from one level to the next"))
        self.auto_btn = panel.add_row(button("Find the mesh", self.auto_mesh_requested.emit,
                                            "Run the search on a background thread"))
        self.auto_result = panel.add("Search", hint("not run yet"))
        panel.add_hint("The search solves the *steady* case with the gas loop: it picks "
                       "the mesh, then Build mesh uses it for every analysis.  If the "
                       "budget stops the refinement, it says so instead of pretending.")
        self._mesh_mode()
        self.tabs.addTab(panel, "Mesh")

    def _mesh_mode(self, *_args) -> None:
        """A mesh setting changed: a plan the search adopted no longer describes it."""
        if getattr(self, "_auto_plan", None) is not None:
            self._auto_plan = None
            self.auto_result.setText("the settings changed: search again")
        self._update_mesh_summary()

    # ----------------------------------------------------------- the pipes
    def pipe_network_config(self) -> PipeNetworkConfig:
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
            duct_diameter=self.pipe_duct.value(),
            insulated_headers=self.pipe_insulated.isChecked(),
            azimuth_in=self.pipe_azimuth_in.value(),
            azimuth_out=self.pipe_azimuth_out.value(),
            split_mode=self.pipe_split.currentData(),
            n_sectors=int(self.pipe_sectors.value()))

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
            self.circuit_info.setText(message or "build the network to size it")
            return
        problems = [problem for problem in network.config.validate()
                    if problem.startswith(WARNING)]
        problems.extend(network.validate())
        lines = [network.summary()]
        if report is not None:
            lines.append(report.summary())
        lines.append(self.circuit_line())
        if problems:
            lines.append("check: " + "; ".join(problems))
        if message:
            lines.append(message)
        self.pipe_info.setText("\n".join(lines))
        self.circuit_info.setText(self._circuit_summary())

    def circuit_line(self) -> str:
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
        lx, ly, lz = self.domain()
        try:
            plan = self.adaptive_plan()
        except (ValueError, RuntimeError) as exc:
            self.mesh_info.setText(f"invalid: {exc}")
            self.memory_info.setText("-")
            return
        box = plan.n_finest * plan.physical_size
        sizes = [band.size for band in plan.bands]
        lines = [
            f"octree: box {box:.3f} m, finest leaf {plan.physical_size * 1000:.1f} mm",
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
        and nothing else.  The box and its resolution come from the cell budget
        (:func:`src.analysis.mesh_plan.tree_resolution`), which is also the floor under
        every leaf the search then refines.

        When the search converged, its own plan is returned unchanged: the boxes the
        search measured are the ones whose answer converged.
        """
        if self._auto_plan is not None:
            return self._auto_plan
        n_finest, physical_size = tree_resolution(self.domain(), int(self.max_cells.value()))
        return AdaptivePlan(n_finest=n_finest, physical_size=physical_size,
                            bands=region_bands(self.mesh_regions()))

    def mesh_regions(self):
        """The active regions of the model, each with the cell size the panel asks for.

        The size per region is the finer of the manual count and the a priori plan of
        ``src/analysis/mesh_plan.py`` (see :meth:`planned`).  The pipes are the one
        region the *network* describes - its bundle, its active band and its own
        diameter - so the box is derived from :meth:`pipe_network_config`.
        """
        cyl = self.cylinder()
        insulation_cells = max(self.cells_insulation.value(), 1)
        targets = {
            "sand": self.planned("storage", cyl.r_storage / max(self.cells_storage.value(), 1)),
            "insulation_radial": self.planned(
                "insulation_radial", cyl.insulation_thickness / insulation_cells),
            "slab_bottom": self.planned(
                "slab_bottom", cyl.insulation_slab_bottom / insulation_cells),
            "slab_top": self.planned("slab_top", cyl.insulation_slab_top / insulation_cells),
            "shell": self.planned("shell", cyl.shell_thickness / insulation_cells),
            "pipe_wall": self.pipe_diameter.value() / max(self.cells_sheath.value(), 1),
        }
        targets["casing"] = max(targets.values())
        config = self.pipe_network_config()
        reach = config.inner_radius
        pipe_box = ((cyl.center_x - reach, cyl.center_y - reach, config.z_bottom),
                    (cyl.center_x + reach, cyl.center_y + reach, config.z_top))
        return active_regions(cyl, targets, pipe_box=pipe_box)

    def domain(self) -> tuple[float, float, float]:
        """The box the vessel sits in [m]."""
        return (self.domain_lx.value(), self.domain_ly.value(), self.domain_lz.value())

    def build_mesh(self) -> AdaptiveMesh:
        """The octree of the current targets (:meth:`adaptive_plan`).

        The tree is kept for the Mesh tab's summary: its leaf count, its levels and its
        leaf edges are measured, never guessed.
        """
        plan = self.adaptive_plan()
        mesh = AdaptiveMesh.from_bands(plan.n_finest, plan.physical_size, plan.bands,
                                       plan.base_level)
        self._adaptive_mesh = mesh
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
        self.plan_info.setText(note or "no plan (build the mesh to compute it)")
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
            foundation_margin=self.foundation_margin.value(),
        )

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
