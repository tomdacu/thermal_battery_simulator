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
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QWidget

from src.analysis.convergence import AdaptivePlan
from src.analysis.mesh_plan import (MAX_TREE_LEVEL, MIN_TREE_LEVEL, active_regions,
                                    region_bands)
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import CylinderGeometry, HeaterConfig
from src.core.pipe_network import (COLLECTION_CENTRAL, COLLECTION_DIRECT,
                                   COLLECTION_REVERSE, COLLECTION_TWO_LEVEL,
                                   LAYOUT_GRID, LAYOUT_RADIAL, LAYOUT_RINGS,
                                   LAYOUT_SPIRAL, LAYOUT_STAGGERED, PIPE_CARBON,
                                   PIPE_STAINLESS, SPLIT_EQUAL, SPLIT_PATH,
                                   SPLIT_RING, SPLIT_SECTOR, WARNING,
                                   PipeNetworkConfig, riser_positions)
from src.core.pipes import pipe_surface_power_w_cm2
from src.solver.fluid import Fluid

from ..widgets import FormPanel, button, check, combo, double_spin, hint, int_spin

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

    mesh_changed = pyqtSignal()
    auto_mesh_requested = pyqtSignal()
    #: the Pipes section asks the window - which owns the mesh - to build and paint the
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
        #: the tree plan the automatic search adopted (None: the a priori one)
        self._auto_plan: AdaptivePlan | None = None
        self._plan_targets: dict[str, float] = {}
        self._pipe_network = None
        #: the tree the last build produced, for the Mesh page's summary
        self._adaptive_mesh: AdaptiveMesh | None = None
        #: True when the cell budget stopped the last build short of its targets
        self._budget_short = False
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
            tooltip="Concrete under the floor of the vessel: the ground face is under it"))
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
             ("Two level rings", COLLECTION_TWO_LEVEL)), 1, self._pipe_mode))
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
        self.pipe_duct = pipes.add("Duct d [m]", double_spin(0.15, 0.05, 0.6, 0.05, 3))
        self.pipe_insulated = pipes.add("Insulated headers", check(
            "lagged", False,
            tooltip="A lagged header carries the gas and exchanges nothing with the bed"))
        self.pipe_azimuth_in = pipes.add("Inlet azimuth [deg]",
                                         double_spin(180.0, 0.0, 360.0, 15.0, 0))
        self.pipe_azimuth_out = pipes.add("Outlet azimuth [deg]",
                                          double_spin(0.0, 0.0, 360.0, 15.0, 0))
        self.pipe_split = pipes.add("Flow split", combo(
            (("Equal per branch", SPLIT_EQUAL), ("From path length", SPLIT_PATH),
             ("Equal per ring main", SPLIT_RING),
             ("Equal per sector", SPLIT_SECTOR)), 0, self._pipe_mode))
        self.pipe_sectors = pipes.add("Sectors", int_spin(
            4, 1, 16, 1, tooltip="The flow is divided between the sectors about the "
                                 "inlet azimuth"))
        pipes.add_row(button("Rebuild the network on the mesh",
                             self.pipe_network_requested.emit,
                             "Voxelise the network on the current mesh (Build mesh "
                             "does it too)"))
        self.pipe_info = pipes.add("Network", hint("build the mesh: it paints the network"))
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
            "The mesh is an octree refined on the active model only: the sand, the "
            "insulation, the shell and a column around every riser, each to the finer "
            "of the count below and the a priori plan.  The air around the vessel is "
            "excluded from the problem, so its leaves stay coarse.  Every target snaps "
            "to the nearest leaf size (leaves are powers of two of the finest one).")
        self.cells_storage = resolution.add("Cells across storage", int_spin(
            10, 2, 200, 1, on_change=self._mesh_mode,
            tooltip="Cells across the storage radius: the bed's own size"))
        self.cells_insulation = resolution.add("Cells across insulation", int_spin(
            3, 1, 50, 1, on_change=self._mesh_mode,
            tooltip="Cells across the radial insulation (and the slabs, and the shell)"))
        self.cells_sheath = resolution.add("Cells across the tube", int_spin(
            1, 1, 20, 1, on_change=self._mesh_mode,
            tooltip="Cells across the outer diameter of a riser: the column of leaves "
                    "the tree refines around every pipe"))
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
        lx, ly, lz = self.domain()
        try:
            plan = self.adaptive_plan()
        except (ValueError, RuntimeError) as exc:
            self.mesh_info.setText(f"invalid: {exc}")
            self.memory_info.setText("-")
            return
        box = plan.n_finest * plan.physical_size
        sizes = [band.size for band in plan.bands]
        lines = [f"box {box:.2f} m, finest leaf {plan.physical_size * 1000:.0f} mm, "
                 f"{len(plan.bands)} regions ({min(sizes) * 1000:.0f}-"
                 f"{max(sizes) * 1000:.0f} mm)"]
        mesh = self._adaptive_mesh
        if mesh is None:
            lines.append("press Build mesh to count the leaves")
            self.memory_info.setText("-")
        else:
            levels = ", ".join(
                f"{plan.physical_size * 2 ** level * 1000:.0f} mm: {count:,}"
                for level, count in mesh.level_histogram().items())
            lines.append(f"{mesh.n_cells:,} leaves - {levels}")
            if self._budget_short:
                lines.append(f"the cell budget stopped the refinement at "
                             f"{mesh.sizes.min() * 1000:.0f} mm: raise it for the "
                             f"{min(sizes) * 1000:.0f} mm the regions ask for")
            self.memory_info.setText(f"~{mesh.n_cells * 98 / 1e6:.1f} MB of fields")
        self.mesh_info.setText("\n".join(lines))

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
        # the finest leaf is the one the finest region asks for (the pipes, usually):
        # the refinement is local, so the floor no longer has to fit the whole box in
        # the budget - the budget caps the leaves the rounds actually make (build_mesh)
        box = max(self.domain())
        finest = min(band.size for band in raw)
        level = int(np.clip(np.round(np.log2(box / finest)), MIN_TREE_LEVEL, MAX_TREE_LEVEL))
        n_finest = 1 << level
        physical_size = box / n_finest
        # a leaf is a power of two of the finest cell: every target is snapped to the
        # largest such size within LEAF_TOLERANCE of it, not to the next one below -
        # otherwise a 200 mm request on 101.6/203 mm leaves refines the whole bed to
        # 101.6 mm, twice finer than asked and eight times the leaves
        bands = tuple(replace(band, size=nearest_leaf(band.size, physical_size))
                      for band in raw)
        return AdaptivePlan(n_finest=n_finest, physical_size=physical_size, bands=bands)

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
            # a thin steel shell needs no cells across it: the painter keeps it one cell
            # thick whatever the mesh, so it takes the insulation's size (and its own
            # film rule, which for steel never binds)
            "shell": cyl.insulation_thickness / insulation_cells,
            "pipe_wall": self.pipe_diameter.value() / max(self.cells_sheath.value(), 1),
        }
        targets["casing"] = max(targets.values())
        config = self.pipe_network_config()
        # one column per riser, as wide as the pipe: the tree refines the leaves the pipe
        # crosses and its own 2:1 balance grades the sand around them.  A single box over
        # the whole bundle refined all the sand to the pipe's size - no octree steps
        # anywhere in the bed, and most of the leaves of the mesh
        half = 0.5 * config.diameter
        pipe_boxes = [((x - half, y - half, config.z_bottom), (x + half, y + half, config.z_top))
                      for x, y in riser_positions(config, (cyl.center_x, cyl.center_y))]
        return active_regions(cyl, targets, pipe_boxes=pipe_boxes)

    def _shape(self, center: tuple[float, float] = (0.0, 0.0)) -> CylinderGeometry:
        return CylinderGeometry(
            center_x=center[0], center_y=center[1],
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
        root = plan.base_level
        if root is None:
            root = int(np.log2(plan.n_finest))
        mesh = AdaptiveMesh.uniform(plan.n_finest, plan.physical_size, root)
        # the budget is a cap on the leaves: the rounds go coarse to fine over the whole
        # tree, so a budget that stops them leaves every region one level short rather
        # than some regions done and others untouched
        mesh.refine_bands(plan.bands, max_cells=int(self.max_cells.value()))
        self._adaptive_mesh = mesh
        self._budget_short = float(mesh.sizes.min()) > 1.01 * min(
            band.size for band in plan.bands)
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
