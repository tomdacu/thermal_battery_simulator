"""Geometry model: configuration dataclasses + voxel painting of a mesh.

The painter is written against the cell protocol of :mod:`src.core.mesh_api`: it reads
where the cells are, how many there are and how big the one at a point is, and it writes
the per-cell fields the protocol names (``material_id``, ``k``, ``rho``, ``cp``,
``Q_source``, ``source_mask``, ``bc_h``, ``bc_T_inf``, ``boundary_type``, ``excluded``,
``h_out``, ``t_ambient``).  It therefore paints either mesh: a :class:`Mesh3D` grid or the
adaptive mesh of :mod:`src.core.adaptive_mesh`.  The few queries that are *not* in the
protocol (the index-free half of a painter: the cell centres, the local cell size) are the
adapters of the section below - they are the whole of the transition and they disappear
with ``Mesh3D`` (step 8 of ``docs/16_ADAPTIVE_MESH_MIGRATION.md``).

Layout painted by :meth:`BatteryGeometry.apply_to_mesh` (later steps win, but the
overlaps that used to corrupt the model are now resolved explicitly):

    1  air everywhere
    2  concrete foundation under the battery footprint
    3  steel lateral shell            (base_z .. cone base, outside the insulation)
    4  radial insulation              (r_storage .. r_insulation, same band)
    5  bottom insulation slab
    6  storage sand (packed bed)      + the lumped bed source of the gas circuit
    7  top insulation slab
    8  optional steel plate under the cone
    9  optional conical roof (steel shell, optionally sand-filled)
   10  domain boundary conditions

The heat-transfer surface itself - the buried pipe network the gas runs through - is
painted by :meth:`src.core.pipe_network.PipeNetwork.paint` on the same mesh: the
resistors live in the gas circuit and heat the gas, and the gas delivers that power to
the sand through the pipes' walls.  Nothing in this module knows of a heater element
inside the bed.

Units: metres, seconds, watts, KELVIN.  ``apply_to_mesh`` validates the geometry
against the mesh and raises ``ValueError`` instead of silently clipping the roof
or the shell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..constants import PACKING_FRACTION_DEFAULT, T_AMBIENT_DEFAULT, T_GROUND_DEFAULT
from .environment import h_out
from .materials import MaterialManager, ThermalProperties
from .mesh import MaterialID, Mesh3D
from .physics import radiation_h

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from .adaptive_mesh import AdaptiveMesh


# --------------------------------------------------------- the painter's mesh view
# The protocol carries the physics per cell and deliberately no index arithmetic and no
# per-axis sizes (see the module docstring of ``src/core/mesh_api.py``).  A painter that
# compares a zone against the cells that are actually there needs exactly the three things
# that leaves out: *where* the cells are, *how big* the cell at a point is, and *how big*
# the box is.  On ``Mesh3D`` each helper below is the structured query the painter used
# before - so nothing that reads a structured mesh moves by an ulp - and on a tree it is a
# leaf lookup.  They are the transition, and they go with ``Mesh3D``.


def _centres(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cell centres ``(X, Y, Z)`` [m], as the masks of the painter index them."""
    if isinstance(mesh, Mesh3D):
        return mesh.X, mesh.Y, mesh.Z
    centres = np.asarray(mesh.centres())
    return centres[:, 0], centres[:, 1], centres[:, 2]


def _axis_size(mesh, axis: int, x: float, y: float, z: float) -> float:
    """Edge of the cell containing ``(x, y, z)`` along one axis [m] (0=x, 1=y, 2=z).

    A tree has cubic leaves, so all three axes answer with the same edge; the *point* is
    what a tree needs (a leaf is located in space) while ``Mesh3D`` reads only the
    coordinate of the axis it is asked about.  That asymmetry is why this is not part of
    the protocol: a per-axis size has no meaning on a tree.
    """
    if isinstance(mesh, Mesh3D):
        return (mesh.size_x(x), mesh.size_y(y), mesh.size_z(z))[axis]
    return mesh.axis_size_at(axis, x, y, z)


def _box(mesh) -> tuple[float, float, float]:
    """The domain box ``(Lx, Ly, Lz)`` [m]; a tree spans a cube."""
    if isinstance(mesh, Mesh3D):
        return mesh.Lx, mesh.Ly, mesh.Lz
    return tuple(float(v) for v in mesh.box)


def _snap_note(mesh) -> str | None:
    """The "box snapped to the grid" note of the build report, if there is one.

    Only a uniform ``Mesh3D`` snaps: a tree spans its box exactly.
    """
    if not isinstance(mesh, Mesh3D):
        return None
    if max(mesh.snapped.values(), default=0.0) <= 1e-9:
        return None
    return (f"domain snapped to the grid: L_y={mesh.Ly:.3f}, L_z={mesh.Lz:.3f} m "
            f"(cell size {mesh.dx.min():.3f} m)")


@dataclass
class CylinderGeometry:
    """Dimensions of the battery, all elevations measured from the domain floor."""

    center_x: float = 5.0
    center_y: float = 5.0
    base_z: float = 0.5
    height: float = 7.0
    r_storage: float = 2.0
    insulation_thickness: float = 0.3
    shell_thickness: float = 0.02
    insulation_slab_bottom: float = 0.2
    insulation_slab_top: float = 0.2
    roof_angle_deg: float = 15.0
    steel_slab_top: float = 0.0
    fill_cone_with_sand: bool = False
    enable_cone_roof: bool = True
    foundation_margin: float = 0.5

    @property
    def r_insulation(self) -> float:
        return self.r_storage + self.insulation_thickness

    @property
    def r_shell(self) -> float:
        return self.r_insulation + self.shell_thickness

    @property
    def roof_angle_rad(self) -> float:
        return float(np.radians(self.roof_angle_deg))

    @property
    def roof_height(self) -> float:
        if not self.enable_cone_roof or self.roof_angle_deg <= 0:
            return 0.0
        return self.r_shell * float(np.tan(self.roof_angle_rad))

    @property
    def z_slab_bottom_start(self) -> float:
        return self.base_z

    @property
    def z_storage_start(self) -> float:
        return self.base_z + self.insulation_slab_bottom

    @property
    def z_storage_end(self) -> float:
        return self.z_storage_start + self.height

    @property
    def z_slab_top_start(self) -> float:
        return self.z_storage_end

    @property
    def z_slab_top_end(self) -> float:
        return self.z_slab_top_start + self.insulation_slab_top

    @property
    def z_steel_slab_end(self) -> float:
        return self.z_slab_top_end + max(self.steel_slab_top, 0.0)

    @property
    def z_cone_base(self) -> float:
        return self.z_steel_slab_end

    @property
    def z_shell_top(self) -> float:
        return self.z_cone_base

    @property
    def z_cone_apex(self) -> float:
        return self.z_cone_base + self.roof_height

    @property
    def total_height(self) -> float:
        return self.z_cone_apex - self.base_z

    def cone_radius(self, z: float) -> float:
        """Radius of the conical roof at elevation ``z``."""
        if self.roof_height <= 0:
            return 0.0
        rel = float(np.clip(z - self.z_cone_base, 0.0, self.roof_height))
        return self.r_shell * (1.0 - rel / self.roof_height)

    def envelope_radius(self, foundation_margin: float = 0.0) -> float:
        """Radius of the outermost solid of the vessel [m].

        The shell, unless the foundation is wider: the concrete under the floor is a
        conduction path to the ground and stays in the problem, so it is part of the
        envelope the mesh has to cover.
        """
        return max(self.r_shell, self.r_shell + max(foundation_margin, 0.0))

    def active_bounds(self, foundation_margin: float = 0.0, margin: float = 0.0
                      ) -> tuple[float, float, float, float, float, float]:
        """Bounding box of the region the *problem* lives in [m].

        ``(x0, x1, y0, y1, z0, z1)`` of the vessel: the shell's footprint and the
        foundation inside it, from the domain floor to the roof apex.  Everything air
        beyond the shell radius or above the apex is *excluded* from the problem by
        :meth:`BatteryGeometry.apply_environment` - those cells are held at the ambient
        temperature and conduct nothing - so this is the region a mesh has a reason to
        refine, and ``margin`` grows it by the distance the refinement must still cover
        around the vessel.
        """
        reach = self.envelope_radius(foundation_margin) + max(margin, 0.0)
        return (self.center_x - reach, self.center_x + reach,
                self.center_y - reach, self.center_y + reach,
                0.0, self.z_cone_apex + max(margin, 0.0))

    def validate(self) -> list[str]:
        problems = []
        if self.r_storage <= 0:
            problems.append("r_storage must be > 0")
        if self.height <= 0:
            problems.append("height must be > 0")
        if self.insulation_thickness <= 0:
            problems.append("insulation_thickness must be > 0")
        if self.shell_thickness < 0:
            problems.append("shell_thickness must be >= 0")
        if min(self.insulation_slab_bottom, self.insulation_slab_top) < 0:
            problems.append("insulation slab thicknesses must be >= 0")
        if self.base_z < 0:
            problems.append("base_z must be >= 0")
        return problems


@dataclass
class HeaterConfig:
    """The electric heat source of the plant, as the bed sees it.

    The resistors do **not** sit in the bed: they heat the gas of the closed circuit
    (``src/solver/fluid.py``), and the gas delivers that power to the sand through the
    walls of the buried pipe network (``src/core/pipe_network.py``).  What the geometry
    still needs is the *lumped* bed source every analysis that does not march the loop
    reads: the electric power, spread uniformly over the storage volume inside the two
    offsets.  The surface the power actually crosses is the pipe network's: the
    per-square-centimetre rating of the tubes is checked on that surface by
    :func:`src.core.pipes.pipe_surface_power_w_cm2`, from the total power and the wetted
    area of the network, not from a sheath diameter that no longer exists.
    """

    power_total: float = 100.0                 # [kW] electric power into the gas
    offset_bottom: float = 0.0                 # [m] of storage left unheated at the floor
    offset_top: float = 0.0                    # [m] of storage left unheated under the roof

    @property
    def power_w(self) -> float:
        return self.power_total * 1000.0


@dataclass
class BuildReport:
    """What ``apply_to_mesh`` actually painted."""

    zone_volumes: dict = field(default_factory=dict)
    zone_masses: dict = field(default_factory=dict)
    n_source_cells: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class BatteryGeometry:
    """Complete geometry description of the storage unit."""

    cylinder: CylinderGeometry = field(default_factory=CylinderGeometry)
    heaters: HeaterConfig = field(default_factory=HeaterConfig)
    storage_material: str = "steatite"
    insulation_material: str = "rock_wool"
    shell_material: str = "carbon_steel"
    packing_fraction: float = PACKING_FRACTION_DEFAULT
    h_top: float = 10.0                     # [W/(m^2*K)]  used at the domain box faces,
    h_lateral: float = 5.0                  # [W/(m^2*K)]  unless the environment below
    t_ambient: float = T_AMBIENT_DEFAULT    # [K]          replaces them with a film
    #: wind speed at the vessel [m/s]: with the film below, the outside becomes a
    #: boundary condition instead of a domain (see ``apply_environment``)
    wind_speed: float = 0.0
    #: nominal surface rise over the ambient used to evaluate the film at a design
    #: point [K]: the correlations need a temperature jump, and right after painting
    #: the shell is still cold, which would null the natural convection
    film_delta_t: float = 30.0
    t_ground: float = T_GROUND_DEFAULT      # [K]

    # ------------------------------------------------------------- validation
    def validate(self, mesh: Mesh3D | AdaptiveMesh) -> list[str]:
        cyl = self.cylinder
        problems = list(cyl.validate())
        if not 0.0 < self.packing_fraction < 1.0:
            problems.append(f"packing_fraction must be in (0,1), got {self.packing_fraction}")
        for h, name in ((self.h_top, "h_top"), (self.h_lateral, "h_lateral")):
            if h < 0:
                problems.append(f"{name} must be >= 0")
        if mesh is not None:
            lx, ly, lz = _box(mesh)
            if cyl.center_x - cyl.r_shell < 0 or cyl.center_x + cyl.r_shell > lx:
                problems.append(
                    f"battery radius {cyl.r_shell:.3f} m does not fit in X "
                    f"[0, {lx:.3f}]: increase Lx or reduce the radius/insulation")
            if cyl.center_y - cyl.r_shell < 0 or cyl.center_y + cyl.r_shell > ly:
                problems.append(
                    f"battery radius {cyl.r_shell:.3f} m does not fit in Y "
                    f"[0, {ly:.3f}]: increase Ly or reduce the radius/insulation")
            if cyl.z_cone_apex > lz + 1e-9:
                problems.append(
                    f"battery height {cyl.z_cone_apex:.3f} m (roof apex) exceeds Lz = "
                    f"{lz:.3f} m: increase Lz or lower the roof/height")
        return problems

    def apply_environment(self, mesh: Mesh3D | AdaptiveMesh,
                          wind_speed: float | None = None,
                          radiation: bool = False) -> dict:
        """Turn the air around the vessel into a boundary condition.

        The exclusion is *geometric*, not by material: a cell is dropped only if it is
        air **and** it lies outside the shell radius or above the roof apex.  The air
        inside an unfilled conical roof, and the concrete of the foundation, stay in the
        problem because they are inside the envelope.

        The film the surface carries is the correlations' convective value (natural +
        wind); the mesh keeps it as ``h_out_conv``, together with the emissivity of the
        shell (``environment_emissivity``), because the solver re-evaluates the film at
        the field it solves.  ``radiation`` mirrors the solver's switch
        (:attr:`~src.solver.steady.SolverConfig.radiation`): with it on, the film painted
        here already carries the linearised radiative share ``h_rad`` of the shell,
        evaluated at the same design-point surface temperature the correlations use.  The
        share is not frozen there - ``h_rad`` grows as ``T^3``, and the surface
        temperature is what the film itself determines - so the film is a *fixed point*:
        the assembly re-evaluates it on the field it assembles for
        (:func:`src.solver.matrix.environment_film`), and **one re-evaluation after a
        solve is enough**.

        Surface temperature for the correlations.  The film is a *design point* closure
        and must not be evaluated on whatever field the mesh happens to hold: right after
        painting the shell is still at the initial temperature, so the driving jump is
        zero, the natural convection vanishes and the film comes out as 0.005 W/(m^2 K) -
        present in the code, absent in the physics.  Use a nominal rise over the ambient
        (or the shell already being hot, if a solve has run).
        """
        cyl = self.cylinder
        wind = self.wind_speed if wind_speed is None else float(wind_speed)
        X, Y, Z = _centres(mesh)
        radius = np.hypot(X - cyl.center_x, Y - cyl.center_y)
        beyond_shell = radius > cyl.r_shell + 1e-9
        above_roof = cyl.z_cone_apex + 1e-9 < Z
        outside = beyond_shell | above_roof
        air = mesh.material_id == int(MaterialID.AIR)
        mesh.excluded = outside & air

        steel = mesh.material_id == int(MaterialID.STEEL)
        hot = float(np.mean(np.asarray(mesh.T)[steel])) if steel.any() else 0.0
        t_surface = max(hot, self.t_ambient + self.film_delta_t)
        film = h_out(t_surface, self.t_ambient,
                     height=max(cyl.z_cone_apex - cyl.base_z, 0.1),
                     width=max(2.0 * cyl.r_shell, 0.1), wind_speed=wind)
        emissivity = float(MaterialManager().get(self.shell_material).emissivity)
        film["radiative"] = (float(radiation_h(t_surface, self.t_ambient, emissivity))
                             if radiation else 0.0)
        film["total"] = film["natural"] + film["wind"] + film["radiative"]
        mesh.t_ambient = self.t_ambient
        mesh.h_out_conv = film["natural"] + film["wind"]
        mesh.h_out = film["total"]
        mesh.environment_emissivity = emissivity
        self.film = film
        return film

    # --------------------------------------------------------------- painting
    def apply_to_mesh(self, mesh: Mesh3D | AdaptiveMesh,
                      materials: MaterialManager = None) -> BuildReport:
        """Paint the geometry onto ``mesh``; raises ``ValueError`` on a bad fit.

        ``mesh`` may be a structured :class:`~src.core.mesh.Mesh3D` or an adaptive
        :class:`~src.core.adaptive_mesh.AdaptiveMesh`: the masks are built over the cell
        centres and the fields written through the protocol, so the same geometry paints
        the same numbers on either (which is what ``tests/test_geometry_on_a_tree.py``
        checks leaf by leaf).
        """
        materials = materials or MaterialManager()
        problems = self.validate(mesh)
        if problems:
            raise ValueError("invalid geometry: " + "; ".join(problems))

        cyl = self.cylinder
        storage_props = materials.compute_packed_bed_properties(self.storage_material,
                                                                self.packing_fraction)
        insul_props = materials.get(self.insulation_material)
        steel_props = materials.get(self.shell_material)
        air_props = materials.get("air")
        concrete_props = materials.get("concrete")

        X, Y, Z = _centres(mesh)
        R = np.sqrt((X - cyl.center_x) ** 2 + (Y - cyl.center_y) ** 2)
        report = BuildReport()
        note = _snap_note(mesh)
        if note is not None:
            report.notes.append(note)

        self._fill(mesh, np.ones(mesh.T.shape, dtype=bool), MaterialID.AIR, air_props)
        self._paint_shell_and_insulation(mesh, R, Z, insul_props, steel_props, concrete_props)
        self._paint_storage(mesh, R, Z, storage_props)
        self._paint_roof(mesh, R, Z, storage_props, steel_props)
        report.n_source_cells = self._paint_source(mesh, Z, R)

        # the air around the vessel becomes a boundary condition: the cells outside the
        # envelope leave the problem and the surface carries the film
        self.apply_environment(mesh)

        self.apply_boundary_conditions(mesh, steel_props)
        report.zone_volumes = self.zone_volumes(mesh)
        report.zone_masses = self.zone_masses(mesh, materials)
        mesh.validate()
        return report

    def apply_source(self, mesh: Mesh3D | AdaptiveMesh) -> int:
        """Re-paint the lumped bed source (the plant power over the storage band).

        The analyses that run without a gas loop read it; a transient overwrites
        ``Q_source`` step by step, so a run that follows one calls this to start from the
        power of the geometry again.  Returns the number of source cells.
        """
        cyl = self.cylinder
        X, Y, Z = _centres(mesh)
        R = np.sqrt((X - cyl.center_x) ** 2 + (Y - cyl.center_y) ** 2)
        return self._paint_source(mesh, Z, R)

    def apply_boundary_conditions(self, mesh: Mesh3D | AdaptiveMesh,
                                  steel_props: ThermalProperties = None) -> None:
        """Air-exposed faces convect (with the shell emissivity), ground is fixed."""
        steel_props = steel_props or MaterialManager().get(self.shell_material)
        for face in ("x_min", "x_max", "y_min", "y_max"):
            mesh.set_convection_bc(face, self.h_lateral, self.t_ambient)
            mesh.face_bc[face].emissivity = steel_props.emissivity
        mesh.set_convection_bc("z_max", self.h_top, self.t_ambient)
        mesh.face_bc["z_max"].emissivity = steel_props.emissivity
        mesh.set_fixed_temperature_bc("z_min", self.t_ground)

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _fill(mesh: Mesh3D | AdaptiveMesh, mask: np.ndarray, material: MaterialID,
              props: ThermalProperties) -> None:
        mesh.material_id[mask] = int(material)
        mesh.k[mask] = props.k
        mesh.rho[mask] = props.rho
        mesh.cp[mask] = props.cp

    @staticmethod
    def _widen(low: float, high: float, minimum: float, grow_up: bool = True
               ) -> tuple[float, float]:
        """Guarantee a band at least ``minimum`` wide, growing away from the core.

        A voxel mesh cannot represent a zone thinner than a cell: without this a
        20 mm shell or a 5 mm plate contains no cell centre and simply disappears
        from the model (the roof used to vanish at any realistic cell size).
        """
        if high - low >= minimum:
            return low, high
        return (low, low + minimum) if grow_up else (high - minimum, high)

    def _paint_shell_and_insulation(self, mesh: Mesh3D | AdaptiveMesh, R, Z, insul, steel,
                                    concrete) -> None:
        cyl = self.cylinder
        z = Z
        # radial thickness of the shell and the vertical thickness of the slabs are
        # compared with the *local* cell size, so a graded mesh measures them with
        # the cells that are actually there
        # the cell is measured *in* the shell (on the +x radius, half a shell inside the
        # outer surface): the point (cx + r, cy + r) this used to probe is the corner of
        # the square, out in the air, where a tree keeps coarse leaves - a 375 mm "cell"
        # there made the shell thicker than the insulation and painted steel over it
        x_shell = cyl.center_x + cyl.r_shell - 0.5 * max(cyl.shell_thickness, 1e-6)
        y_shell = cyl.center_y
        mid_lateral = 0.5 * (cyl.base_z + cyl.z_shell_top)
        cell = max(_axis_size(mesh, 0, x_shell, y_shell, mid_lateral),
                   _axis_size(mesh, 1, x_shell, y_shell, mid_lateral))
        foundation = (z < cyl.base_z) & (cyl.r_shell + cyl.foundation_margin >= R)
        self._fill(mesh, foundation, MaterialID.CONCRETE, concrete)

        lateral = (z >= cyl.base_z) & (z < cyl.z_shell_top)
        # insulation first, shell last: the shell keeps at least one cell of thickness
        self._fill(mesh, lateral & (cyl.r_storage <= R) & (cyl.r_insulation > R),
                   MaterialID.INSULATION, insul)
        shell_inner = max(cyl.r_shell - max(cyl.shell_thickness, cell), 0.0)
        self._fill(mesh, lateral & (shell_inner <= R) & (cyl.r_shell > R),
                   MaterialID.STEEL, steel)
        if cyl.insulation_slab_bottom > 0:
            slab_cell = _axis_size(mesh, 2, cyl.center_x, cyl.center_y,
                                   0.5 * (cyl.z_slab_bottom_start + cyl.z_storage_start))
            bottom_low, bottom_high = self._widen(cyl.z_slab_bottom_start,
                                                  cyl.z_storage_start, slab_cell,
                                                  grow_up=False)
            self._fill(mesh, (z >= bottom_low) & (z < bottom_high) & (cyl.r_storage > R),
                       MaterialID.INSULATION, insul)
        if cyl.insulation_slab_top > 0:
            top_cell = _axis_size(mesh, 2, cyl.center_x, cyl.center_y,
                                  0.5 * (cyl.z_slab_top_start + cyl.z_slab_top_end))
            top_low, top_high = self._widen(cyl.z_slab_top_start, cyl.z_slab_top_end,
                                            top_cell)
            self._fill(mesh, (z >= top_low) & (z < top_high) & (cyl.r_storage > R),
                       MaterialID.INSULATION, insul)
        if cyl.steel_slab_top > 0:
            plate_cell = _axis_size(mesh, 2, cyl.center_x, cyl.center_y,
                                    cyl.z_slab_top_end)
            plate_low, plate_high = self._widen(cyl.z_slab_top_end, cyl.z_steel_slab_end,
                                                plate_cell)
            self._fill(mesh, (z >= plate_low) & (z < plate_high) & (cyl.r_insulation > R),
                       MaterialID.STEEL, steel)

    def _paint_storage(self, mesh: Mesh3D | AdaptiveMesh, R, Z, storage) -> None:
        cyl = self.cylinder
        band = (cyl.z_storage_start <= Z) & (cyl.z_storage_end > Z) & (cyl.r_storage > R)
        self._fill(mesh, band, MaterialID.SAND, storage)

    def _paint_roof(self, mesh: Mesh3D | AdaptiveMesh, R, Z, storage, steel) -> None:
        cyl = self.cylinder
        if cyl.roof_height <= 0:
            return
        in_region = (cyl.z_cone_base <= Z) & (cyl.z_cone_apex >= Z)
        r_cone = cyl.r_shell * (1.0 - np.clip(Z - cyl.z_cone_base, 0.0, cyl.roof_height)
                                / cyl.roof_height)
        if cyl.fill_cone_with_sand:
            self._fill(mesh, in_region & (r_cone > R), MaterialID.SAND, storage)
        # a shell thinner than a cell would vanish: keep at least one cell of steel
        z_roof = cyl.z_cone_base + 0.5 * cyl.roof_height
        thickness = max(cyl.shell_thickness,
                        mesh.cell_size_at(cyl.center_x + 0.7 * cyl.r_shell, cyl.center_y,
                                          z_roof))
        r_inner = np.maximum(r_cone - thickness, 0.0)
        self._fill(mesh, in_region & (r_inner <= R) & (r_cone > R), MaterialID.STEEL, steel)

    def _paint_source(self, mesh: Mesh3D | AdaptiveMesh, Z, R) -> int:
        """The lumped bed source: the gas circuit's power over the storage volume.

        The resistors heat the gas, the gas crosses the pipe walls and the sand takes
        the heat up - and an analysis that does not march the loop (steady, losses) sees
        that power as a uniform volumetric source over the storage band inside the two
        offsets.  It is the *same* number the gas carries: ``power_total`` is the
        electric power of the plant, and the network's own report checks it against the
        wetted surface the pipes actually offer.
        """
        cyl, cfg = self.cylinder, self.heaters
        mesh.Q_source.fill(0.0)
        mesh.source_mask.fill(False)
        z_bottom = cyl.z_storage_start + cfg.offset_bottom
        z_top = cyl.z_storage_end - cfg.offset_top
        if z_top <= z_bottom:
            raise ValueError("heater offsets leave no room inside the storage band")
        mask = (z_bottom <= Z) & (z_top > Z) & (cyl.r_storage > R)
        n = int(np.count_nonzero(mask))
        if n == 0:
            raise ValueError("the heat source covers no cell: refine the mesh")
        mesh.source_mask[mask] = True
        mesh.Q_source[mask] = cfg.power_w / float(mesh.V[mask].sum())
        return n

    # ------------------------------------------------------------- reporting
    def zone_volumes(self, mesh: Mesh3D | AdaptiveMesh = None) -> dict:
        """Analytic zone volumes [m^3] (validated against the mesh when given)."""
        cyl = self.cylinder
        v = {
            "storage": float(np.pi * cyl.r_storage ** 2 * cyl.height),
            "slab_bottom": float(np.pi * cyl.r_storage ** 2 * cyl.insulation_slab_bottom),
            "slab_top": float(np.pi * cyl.r_storage ** 2 * cyl.insulation_slab_top),
            "insulation_radial": float(np.pi * (cyl.r_insulation ** 2 - cyl.r_storage ** 2)
                                       * (cyl.z_shell_top - cyl.base_z)),
            "shell": float(np.pi * (cyl.r_shell ** 2 - cyl.r_insulation ** 2)
                           * (cyl.z_shell_top - cyl.base_z)),
            "cone_shell": self._cone_shell_volume(),
            "steel_slab": float(np.pi * cyl.r_insulation ** 2 * max(cyl.steel_slab_top, 0.0)),
            "foundation": float(np.pi * (cyl.r_shell + cyl.foundation_margin) ** 2
                                * max(cyl.base_z, 0.0)),
        }
        v["insulation"] = v["slab_bottom"] + v["slab_top"] + v["insulation_radial"]
        v["total"] = sum(v[k] for k in ("storage", "insulation", "shell", "cone_shell",
                                        "steel_slab", "foundation"))
        return v

    def _cone_shell_volume(self) -> float:
        cyl = self.cylinder
        if cyl.roof_height <= 0:
            return 0.0
        outer = np.pi * cyl.r_shell ** 2 * cyl.roof_height / 3.0
        inner_radius = max(cyl.r_shell - cyl.shell_thickness, 0.0)
        inner = np.pi * inner_radius ** 2 * cyl.roof_height / 3.0
        return float(max(outer - inner, 0.0))

    def zone_masses(self, mesh: Mesh3D | AdaptiveMesh = None,
                    materials: MaterialManager = None) -> dict:
        """Zone masses [kg] from the analytic volumes and the selected materials."""
        materials = materials or MaterialManager()
        v = self.zone_volumes(mesh)
        storage = materials.compute_packed_bed_properties(self.storage_material,
                                                          self.packing_fraction)
        insul = materials.get(self.insulation_material)
        steel = materials.get(self.shell_material)
        concrete = materials.get("concrete")
        m = {
            "storage": v["storage"] * storage.rho,
            "slab_bottom": v["slab_bottom"] * insul.rho,
            "slab_top": v["slab_top"] * insul.rho,
            "insulation_radial": v["insulation_radial"] * insul.rho,
            "shell": v["shell"] * steel.rho,
            "cone_shell": v["cone_shell"] * steel.rho,
            "steel_slab": v["steel_slab"] * steel.rho,
            "foundation": v["foundation"] * concrete.rho,
        }
        m["insulation"] = m["slab_bottom"] + m["slab_top"] + m["insulation_radial"]
        m["total"] = sum(m[k] for k in ("storage", "insulation", "shell", "cone_shell",
                                        "steel_slab", "foundation"))
        return m

    def estimate_energy_capacity(self, t_high: float, t_low: float,
                                 efficiency: float = 0.87,
                                 materials: MaterialManager = None) -> dict:
        """Stored / usable energy of the storage zone [J] and the sand mass [kg]."""
        materials = materials or MaterialManager()
        store = materials.compute_packed_bed_properties(self.storage_material,
                                                        self.packing_fraction)
        mass = self.zone_masses(materials=materials)["storage"]
        energy = mass * store.cp * max(t_high - t_low, 0.0)
        return {
            "mass_storage_kg": mass,
            "mass_storage_t": mass / 1000.0,
            "E_thermal_J": energy,
            "E_thermal_kWh": energy / 3.6e6,
            "E_usable_J": energy * efficiency,
            "E_usable_kWh": energy * efficiency / 3.6e6,
            "t_high": t_high,
            "t_low": t_low,
        }


def create_small_test_geometry() -> BatteryGeometry:
    """Compact geometry used by the test-suite (roughly a 1:4 scale model)."""
    geom = BatteryGeometry(
        cylinder=CylinderGeometry(center_x=3.0, center_y=3.0, base_z=0.3, height=4.0,
                                  r_storage=2.0, insulation_thickness=0.2,
                                  insulation_slab_bottom=0.2, insulation_slab_top=0.2,
                                  roof_angle_deg=15.0, enable_cone_roof=True),
        heaters=HeaterConfig(power_total=50.0),
    )
    return geom
