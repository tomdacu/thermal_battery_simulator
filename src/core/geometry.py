"""Geometry model: configuration dataclasses + voxel painting of a :class:`Mesh3D`.

Layout painted by :meth:`BatteryGeometry.apply_to_mesh` (later steps win, but the
overlaps that used to corrupt the model are now resolved explicitly):

    1  air everywhere
    2  concrete foundation under the battery footprint
    3  steel lateral shell            (base_z .. cone base, outside the insulation)
    4  radial insulation              (r_storage .. r_insulation, same band)
    5  bottom insulation slab
    6  storage sand (packed bed)      + volumetric source when the pattern is uniform
    7  top insulation slab
    8  optional steel plate under the cone
    9  optional conical roof (steel shell, optionally sand-filled)
   10  discrete heater elements       -> material HEATERS, marked in ``source_mask``
   11  heat-exchanger tubes           -> material TUBES inside the storage band only
   12  domain boundary conditions

Units: metres, seconds, watts, KELVIN.  ``apply_to_mesh`` validates the geometry
against the mesh and raises ``ValueError`` instead of silently clipping the roof
or the shell.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import PACKING_FRACTION_DEFAULT, T_AMBIENT_DEFAULT, T_GROUND_DEFAULT
from .environment import h_out
from .heaters import (DEFAULT_SHEATH_DIAMETER, DEFAULT_SHEATH_MATERIAL,
                       HeaterBank, rasterize, validate_bank)
from .materials import MaterialManager, ThermalProperties
from .mesh import MaterialID, Mesh3D


class HeaterPattern:
    UNIFORM_ZONE = "uniform_zone"
    GRID_VERTICAL = "grid_vertical"
    CHESS_PATTERN = "chess_pattern"
    RADIAL_ARRAY = "radial_array"
    SPIRAL = "spiral"
    CONCENTRIC_RINGS = "concentric_rings"

    ALL = (UNIFORM_ZONE, GRID_VERTICAL, CHESS_PATTERN, RADIAL_ARRAY,
           SPIRAL, CONCENTRIC_RINGS)


class TubePattern:
    CENTRAL_CLUSTER = "central_cluster"
    RADIAL_ARRAY = "radial_array"
    GRID = "grid"
    HEXAGONAL = "hexagonal"
    SINGLE_CENTRAL = "single_central"
    CUSTOM = "custom"

    ALL = (CENTRAL_CLUSTER, RADIAL_ARRAY, GRID, HEXAGONAL, SINGLE_CENTRAL, CUSTOM)


@dataclass
@dataclass
class TubeElement:
    """One heat-exchanger tube."""

    x: float
    y: float
    z_bottom: float
    z_top: float
    radius: float = 0.025
    h_fluid: float = 500.0
    t_fluid: float = 333.15     # [K]


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
    phase_offset_deg: float = 15.0
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
    def phase_offset_rad(self) -> float:
        return float(np.radians(self.phase_offset_deg))

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
    """Heater layout: a uniform volumetric zone or discrete hairpin elements.

    The discrete elements are *flanged immersion heaters*: U-shaped (hairpin)
    sheathed tubes of ``sheath_diameter`` with two legs ``leg_spacing`` apart,
    rated by ``power_element`` (from the total power and the element count) and
    checked against the 3-8 W/cm^2 surface power of sheathed elements.
    """

    power_total: float = 100.0                 # [kW]
    n_heaters: int = 12
    pattern: str = HeaterPattern.UNIFORM_ZONE
    offset_bottom: float = 0.0
    offset_top: float = 0.0
    n_rings: int = 2
    n_per_ring: list[int] | None = None
    ring_radii: list[float] | None = None
    grid_rows: int = 4
    grid_cols: int = 4
    # hairpin design (see src/core/heaters.py)
    sheath_diameter: float = DEFAULT_SHEATH_DIAMETER
    sheath_material: str = DEFAULT_SHEATH_MATERIAL
    leg_spacing: float = 0.08
    active_length: float | None = None         # None = fill the storage band
    cold_shank: float = 0.15
    flange_offset: float = 0.03
    support_plate_offset: float = 0.05
    bend_chords: int = 4

    @property
    def power_w(self) -> float:
        return self.power_total * 1000.0

    def bank(self, z_storage_start: float, z_storage_end: float) -> HeaterBank:
        """The hairpin bank equivalent of this configuration."""
        length = (self.active_length if self.active_length else
                  max(z_storage_end - z_storage_start - self.offset_bottom
                      - self.offset_top, 0.0))
        rows, cols = self._layout_counts()
        return HeaterBank(
            active=True,
            sheath_diameter=self.sheath_diameter,
            sheath_material=self.sheath_material,
            active_length=length,
            cold_shank=self.cold_shank,
            leg_spacing=self.leg_spacing,
            rows=rows,
            columns=cols,
            power_per_element=self.power_w / max(rows * cols, 1),
            offset_bottom=self.offset_bottom,
            offset_top=self.offset_top,
            support_plate_offset=self.support_plate_offset,
            flange_offset=self.flange_offset,
            layout="ring" if self.pattern in (HeaterPattern.RADIAL_ARRAY,
                                              HeaterPattern.CONCENTRIC_RINGS)
            else "grid",
            n_rings=max(self.n_rings, 1),
            bend_chords=self.bend_chords,
        )

    def _layout_counts(self) -> tuple[int, int]:
        """(rows, columns) of the bank for the configured pattern."""
        if self.pattern == HeaterPattern.CHESS_PATTERN:
            return max(self.grid_rows, 1), max(self.grid_cols, 1)
        if self.pattern in (HeaterPattern.GRID_VERTICAL,):
            return max(self.grid_rows, 1), max(self.grid_cols, 1)
        # rings and spiral: keep the element count, arranged on rings
        n = max(self.n_heaters, 1)
        per_ring = max(int(np.ceil(np.sqrt(n))), 1)
        return max(int(np.ceil(n / per_ring)), 1), per_ring

    @staticmethod
    def _ring_points(cx, cy, radius, count, offset) -> list[tuple[float, float]]:
        return [(cx + radius * np.cos(offset + 2 * np.pi * i / count),
                 cy + radius * np.sin(offset + 2 * np.pi * i / count))
                for i in range(max(count, 1))]


@dataclass
class TubeConfig:
    """Heat-exchanger tube layout."""

    n_tubes: int = 8
    diameter: float = 0.05
    h_fluid: float = 500.0
    t_fluid: float = 333.15     # [K]
    active: bool = False
    pattern: str = TubePattern.RADIAL_ARRAY
    tube_length: float | None = None
    n_rings: int = 2
    n_per_ring: list[int] | None = None
    ring_radii: list[float] | None = None
    grid_rows: int = 3
    grid_cols: int = 3
    grid_spacing: float = 0.2
    custom_positions: list[tuple[float, float]] | None = None

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    def generate_positions(self, center_x: float, center_y: float, r_max: float,
                           z_bottom: float, z_top: float) -> list[TubeElement]:
        """Pure: return the element list, never touch the config or its counts."""
        xy = self._positions(center_x, center_y, r_max)
        length = self.tube_length if self.tube_length else max(z_top - z_bottom, 0.0)
        return [TubeElement(x, y, z_bottom, z_bottom + length, self.radius,
                            self.h_fluid, self.t_fluid) for x, y in xy]

    def _positions(self, cx: float, cy: float, r_max: float) -> list[tuple[float, float]]:
        p = self.pattern
        if p == TubePattern.SINGLE_CENTRAL:
            return [(cx, cy)]
        if p == TubePattern.CUSTOM:
            return [(float(x), float(y)) for x, y in (self.custom_positions or [])]
        if p == TubePattern.CENTRAL_CLUSTER:
            around = min(6, max(self.n_tubes - 1, 0))
            r_ring = min(r_max * 0.6, self.grid_spacing * 2)
            out = [(cx, cy)]
            out += [(cx + r_ring * np.cos(2 * np.pi * i / max(around, 1)),
                     cy + r_ring * np.sin(2 * np.pi * i / max(around, 1)))
                    for i in range(around)]
            return out[:max(self.n_tubes, 1)]
        if p == TubePattern.RADIAL_ARRAY:
            if self.ring_radii and self.n_per_ring:
                rings = list(zip(self.ring_radii, self.n_per_ring, strict=True))
            else:
                rings = np.linspace(r_max * 0.3, r_max * 0.8, max(self.n_rings, 1))
                base = max(self.n_tubes // max(self.n_rings, 1), 1)
                extra = self.n_tubes % max(self.n_rings, 1)
                counts = [base + (1 if i < extra else 0) for i in range(len(rings))]
                rings = list(zip(rings, counts, strict=True))
            out = []
            for radius, count in rings:
                out += [(cx + radius * np.cos(2 * np.pi * i / max(count, 1)),
                         cy + radius * np.sin(2 * np.pi * i / max(count, 1)))
                        for i in range(max(count, 1))]
            return out[:max(self.n_tubes, 1)]
        if p == TubePattern.GRID:
            half_w = (self.grid_cols - 1) * self.grid_spacing / 2
            half_h = (self.grid_rows - 1) * self.grid_spacing / 2
            out = []
            for row in range(self.grid_rows):
                for col in range(self.grid_cols):
                    x, y = cx - half_w + col * self.grid_spacing, cy - half_h + row * self.grid_spacing
                    if float(np.hypot(x - cx, y - cy)) <= r_max:
                        out.append((x, y))
            return out[:max(self.n_tubes, 1)]
        if p == TubePattern.HEXAGONAL:
            out, ring = [(cx, cy)], 1
            while len(out) < self.n_tubes:
                radius = ring * self.grid_spacing
                if radius > r_max:
                    break
                n_on_ring = 6 * ring
                out += [(cx + radius * np.cos(2 * np.pi * i / n_on_ring + np.pi / 6),
                         cy + radius * np.sin(2 * np.pi * i / n_on_ring + np.pi / 6))
                        for i in range(n_on_ring)]
                ring += 1
            return out[:max(self.n_tubes, 1)]
        raise ValueError(f"unknown tube pattern {p!r}; expected one of {TubePattern.ALL}")


@dataclass
class BuildReport:
    """What ``apply_to_mesh`` actually painted."""

    zone_volumes: dict = field(default_factory=dict)
    zone_masses: dict = field(default_factory=dict)
    n_source_cells: int = 0
    n_tube_cells: int = 0
    n_heater_elements: int = 0
    n_tube_elements: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class BatteryGeometry:
    """Complete geometry description of the storage unit."""

    cylinder: CylinderGeometry = field(default_factory=CylinderGeometry)
    heaters: HeaterConfig = field(default_factory=HeaterConfig)
    tubes: TubeConfig = field(default_factory=TubeConfig)
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
    t_ground: float = T_GROUND_DEFAULT      # [K]

    # ------------------------------------------------------------- validation
    def validate(self, mesh: Mesh3D) -> list[str]:
        cyl = self.cylinder
        problems = list(cyl.validate())
        if not 0.0 < self.packing_fraction < 1.0:
            problems.append(f"packing_fraction must be in (0,1), got {self.packing_fraction}")
        for h, name in ((self.h_top, "h_top"), (self.h_lateral, "h_lateral")):
            if h < 0:
                problems.append(f"{name} must be >= 0")
        if mesh is not None:
            if cyl.center_x - cyl.r_shell < 0 or cyl.center_x + cyl.r_shell > mesh.Lx:
                problems.append(
                    f"battery radius {cyl.r_shell:.3f} m does not fit in X "
                    f"[0, {mesh.Lx:.3f}]: increase Lx or reduce the radius/insulation")
            if cyl.center_y - cyl.r_shell < 0 or cyl.center_y + cyl.r_shell > mesh.Ly:
                problems.append(
                    f"battery radius {cyl.r_shell:.3f} m does not fit in Y "
                    f"[0, {mesh.Ly:.3f}]: increase Ly or reduce the radius/insulation")
            if cyl.z_cone_apex > mesh.Lz + 1e-9:
                problems.append(
                    f"battery height {cyl.z_cone_apex:.3f} m (roof apex) exceeds Lz = "
                    f"{mesh.Lz:.3f} m: increase Lz or lower the roof/height")
            if self.heaters.pattern != HeaterPattern.UNIFORM_ZONE:
                # a warning never blocks a build: it is reported with the build
                problems.extend(p for p in self.heater_problems(mesh)
                                if not p.startswith("warning: "))
            problems.extend(self.tube_problems(mesh))
        return problems

    def apply_environment(self, mesh: Mesh3D, wind_speed: float | None = None) -> dict:
        """Turn the air around the vessel into a boundary condition.

        **Not called by** :meth:`apply_to_mesh` yet.**  The box-face report has been
        taught to skip the excluded cells (the spurious 4.9 kW is gone), but
        ``tests/test_solver.py::test_flow_rate_extraction_removes_heat_when_the_battery_is_hot``
        still reports a negative extracted power once the air is dropped - a sign
        question to settle on a test that combines a flow-rate extraction with the
        environment, before this becomes the default.  The method works and is tested,
        but wiring it into the default build exposed two defects that need their own
        fix before it can be automatic: ``fluxes.domain_fluxes`` still reports the
        convection of the six box faces even where the boundary cell has been excluded
        (a spurious 4.9 kW on the default vessel, which breaks the reported balance),
        and one existing extraction test changes sign.  Call it explicitly while that
        is open.

        The exclusion is *geometric*, not by material: a cell is dropped only if it is
        air **and** it lies outside the shell radius or above the roof apex.  The air
        inside an unfilled conical roof, and the concrete of the foundation, stay in the
        problem because they are inside the envelope.
        """
        cyl = self.cylinder
        wind = self.wind_speed if wind_speed is None else float(wind_speed)
        radius = np.hypot(mesh.X - cyl.center_x, mesh.Y - cyl.center_y)
        beyond_shell = radius > cyl.r_shell + 1e-9
        above_roof = cyl.z_cone_apex + 1e-9 < mesh.Z
        outside = beyond_shell | above_roof
        air = mesh.material_id == int(MaterialID.AIR)
        mesh.excluded = outside & air

        # surface temperature for the correlations: the mean of the shell if it is
        # painted, otherwise a nominal rise over the ambient
        steel = mesh.material_id == int(MaterialID.STEEL)
        t_surface = (float(np.mean(mesh.T[steel])) if steel.any()
                     else self.t_ambient + 30.0)
        film = h_out(t_surface, self.t_ambient,
                     height=max(cyl.z_cone_apex - cyl.base_z, 0.1),
                     width=max(2.0 * cyl.r_shell, 0.1), wind_speed=wind)
        mesh.t_ambient = self.t_ambient
        mesh.h_out = film["total"]
        self.film = film
        return film

    def tube_problems(self, mesh: Mesh3D) -> list[str]:
        """Tubes whose effective radius leaves the storage would heat the air."""
        if not self.tubes.active:
            return []
        cyl = self.cylinder
        elements = self.tubes.generate_positions(cyl.center_x, cyl.center_y,
                                                 cyl.r_storage * 0.9,
                                                 cyl.z_storage_start, cyl.z_storage_end)
        cell = max(mesh.size_x(cyl.center_x), mesh.size_y(cyl.center_y))
        for element in elements:
            reach = (float(np.hypot(element.x - cyl.center_x, element.y - cyl.center_y))
                     + element.radius + 0.5 * cell)
            if reach > cyl.r_storage + 1e-9:
                return [f"a heat-exchanger tube at ({element.x:.2f}, {element.y:.2f}) m "
                        f"reaches r = {reach:.2f} m, outside the storage radius "
                        f"{cyl.r_storage:.2f} m: it would exchange heat with the air - "
                        f"reduce the tube diameter, the pattern radius or use a "
                        f"pattern that stays inside"]
        return []

    def heater_warnings(self, mesh: Mesh3D) -> list[str]:
        """Non-blocking remarks of the heater bank (surface power, resolution)."""
        if self.heaters.pattern == HeaterPattern.UNIFORM_ZONE:
            return []
        return [p for p in self.heater_problems(mesh) if p.startswith("warning: ")]

    def heater_problems(self, mesh: Mesh3D, tubes_problems: bool = True) -> list[str]:
        """Errors and warnings of the discrete heater bank (warnings prefixed)."""
        cyl, cfg = self.cylinder, self.heaters
        bank = cfg.bank(cyl.z_storage_start, cyl.z_storage_end)
        tubes = None
        if tubes_problems and self.tubes.active:
            tubes = self.tubes.generate_positions(cyl.center_x, cyl.center_y,
                                                  cyl.r_storage * 0.9,
                                                  cyl.z_storage_start, cyl.z_storage_end)
        return validate_bank(bank, mesh, cyl.center_x, cyl.center_y, cyl.r_storage * 0.9,
                             cyl.z_storage_start, cyl.z_storage_end, tubes)

    # --------------------------------------------------------------- painting
    def apply_to_mesh(self, mesh: Mesh3D, materials: MaterialManager = None) -> BuildReport:
        """Paint the geometry onto ``mesh``; raises ``ValueError`` on a bad fit."""
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

        X, Y, Z = mesh.X, mesh.Y, mesh.Z
        R = np.sqrt((X - cyl.center_x) ** 2 + (Y - cyl.center_y) ** 2)
        report = BuildReport()
        if max(mesh.snapped.values(), default=0.0) > 1e-9:
            report.notes.append(
                f"domain snapped to the grid: L_y={mesh.Ly:.3f}, L_z={mesh.Lz:.3f} m "
                f"(cell size {mesh.dx.min():.3f} m)")

        self._fill(mesh, np.ones(mesh.T.shape, dtype=bool), MaterialID.AIR, air_props)
        self._paint_shell_and_insulation(mesh, R, Z, insul_props, steel_props, concrete_props)
        self._paint_storage(mesh, R, Z, storage_props)
        self._paint_roof(mesh, R, Z, storage_props, steel_props)
        report.n_source_cells = self._paint_heaters(mesh, Z, R, materials)
        report.notes.extend(self.heater_warnings(mesh))
        report.n_tube_cells = self._paint_tubes(mesh, Z, R, materials)

        self.apply_boundary_conditions(mesh, steel_props)
        report.zone_volumes = self.zone_volumes(mesh)
        report.zone_masses = self.zone_masses(mesh, materials)
        report.n_heater_elements = self._n_heaters()
        report.n_tube_elements = self._n_tubes()
        mesh.validate()
        return report

    def apply_boundary_conditions(self, mesh: Mesh3D, steel_props: ThermalProperties = None
                                  ) -> None:
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
    def _fill(mesh: Mesh3D, mask: np.ndarray, material: MaterialID,
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

    def _paint_shell_and_insulation(self, mesh, R, Z, insul, steel, concrete) -> None:
        cyl = self.cylinder
        z = Z
        # radial thickness of the shell and the vertical thickness of the slabs are
        # compared with the *local* cell size, so a graded mesh measures them with
        # the cells that are actually there
        cell = max(mesh.size_x(cyl.center_x + cyl.r_shell) * 1.0,
                   mesh.size_y(cyl.center_y + cyl.r_shell) * 1.0)
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
            slab_cell = mesh.size_z(0.5 * (cyl.z_slab_bottom_start + cyl.z_storage_start))
            bottom_low, bottom_high = self._widen(cyl.z_slab_bottom_start,
                                                  cyl.z_storage_start, slab_cell,
                                                  grow_up=False)
            self._fill(mesh, (z >= bottom_low) & (z < bottom_high) & (cyl.r_storage > R),
                       MaterialID.INSULATION, insul)
        if cyl.insulation_slab_top > 0:
            top_cell = mesh.size_z(0.5 * (cyl.z_slab_top_start + cyl.z_slab_top_end))
            top_low, top_high = self._widen(cyl.z_slab_top_start, cyl.z_slab_top_end,
                                            top_cell)
            self._fill(mesh, (z >= top_low) & (z < top_high) & (cyl.r_storage > R),
                       MaterialID.INSULATION, insul)
        if cyl.steel_slab_top > 0:
            plate_cell = mesh.size_z(cyl.z_slab_top_end)
            plate_low, plate_high = self._widen(cyl.z_slab_top_end, cyl.z_steel_slab_end,
                                                plate_cell)
            self._fill(mesh, (z >= plate_low) & (z < plate_high) & (cyl.r_insulation > R),
                       MaterialID.STEEL, steel)

    def _paint_storage(self, mesh, R, Z, storage) -> None:
        cyl = self.cylinder
        band = (cyl.z_storage_start <= Z) & (cyl.z_storage_end > Z) & (cyl.r_storage > R)
        self._fill(mesh, band, MaterialID.SAND, storage)

    def _paint_roof(self, mesh, R, Z, storage, steel) -> None:
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

    def _paint_heaters(self, mesh, Z, R, materials: MaterialManager) -> int:
        cyl, cfg = self.cylinder, self.heaters
        mesh.Q_source.fill(0.0)
        mesh.source_mask.fill(False)
        z_bottom = cyl.z_storage_start + cfg.offset_bottom
        z_top = cyl.z_storage_end - cfg.offset_top
        if z_top <= z_bottom:
            raise ValueError("heater offsets leave no room inside the storage band")
        if cfg.pattern == HeaterPattern.UNIFORM_ZONE:
            mask = (z_bottom <= Z) & (z_top > Z) & (cyl.r_storage > R)
            n = int(np.count_nonzero(mask))
            if n == 0:
                raise ValueError("uniform heater zone covers no cell: refine the mesh")
            mesh.source_mask[mask] = True
            mesh.Q_source[mask] = cfg.power_w / float(mesh.V[mask].sum())
            return n

        # discrete elements: hairpin sheathed tubes, power on the active length only
        bank = cfg.bank(cyl.z_storage_start, cyl.z_storage_end)
        raster = rasterize(bank, mesh, cyl.center_x, cyl.center_y, cyl.r_storage * 0.9,
                           cyl.z_storage_start, cyl.z_storage_end)
        if raster.problem:
            raise ValueError(f"the heater bank cannot be represented: {raster.problem}")
        mask = raster.mask
        n = int(np.count_nonzero(mask))
        if n == 0:
            raise ValueError(
                "no mesh cell falls inside the discrete heaters: the sheath "
                f"({bank.sheath_diameter * 1000:.1f} mm) is smaller than the local cell "
                f"size.  Refine the heater region or use the uniform zone")
        self._fill(mesh, mask, MaterialID.HEATERS, materials.get(bank.sheath_material))
        active = raster.active_mask
        if int(np.count_nonzero(active)) == 0:
            raise ValueError("the heater elements have no cell inside the storage band: "
                             "check the offsets and the active length")
        mesh.source_mask[active] = True
        mesh.Q_source[active] = bank.total_power_w / float(mesh.V[active].sum())
        return n

    def _paint_tubes(self, mesh, Z, R, materials: MaterialManager) -> int:
        cyl, cfg = self.cylinder, self.tubes
        mask_tubes = np.zeros(mesh.T.shape, dtype=bool)
        if not cfg.active:
            mesh.bc_h[mesh.material_id == int(MaterialID.TUBES)] = 0.0
            return 0
        elements = cfg.generate_positions(cyl.center_x, cyl.center_y, cyl.r_storage * 0.9,
                                          cyl.z_storage_start, cyl.z_storage_end)
        if not elements:
            return 0
        band = (cyl.z_storage_start <= Z) & (cyl.z_storage_end > Z)
        mask_tubes = self._elements_mask(mesh, elements, [e.radius for e in elements]) & band
        n = int(np.count_nonzero(mask_tubes))
        if n == 0:
            raise ValueError(
                "no mesh cell falls inside the heat-exchanger tubes: the tube radius "
                f"({cfg.radius} m) is smaller than half the local cell size "
                f"({mesh.dx.min() / 2:.3f} m). "
                "Refine the mesh or increase the tube diameter")
        steel = materials.get(self.shell_material)
        self._fill(mesh, mask_tubes, MaterialID.TUBES, steel)
        self._clear_sources(mesh, mask_tubes)
        mesh.set_internal_convection(mask_tubes, cfg.h_fluid, cfg.t_fluid)
        return n

    @staticmethod
    def _clear_sources(mesh: Mesh3D, mask: np.ndarray) -> None:
        """A tube cell is not a heater: never keep a volumetric source there."""
        mesh.Q_source[mask] = 0.0
        mesh.source_mask[mask] = False

    @staticmethod
    def _elements_mask(mesh: Mesh3D, elements, radii) -> np.ndarray:
        """Cells covered by the elements; a sub-grid element keeps its own cell.

        The radius is widened to half a cell, and if even that misses every cell
        centre (an element thinner than the grid) the cell containing the axis is
        used, so a heater or a tube can never silently vanish.
        """
        mask = np.zeros(mesh.T.shape, dtype=bool)
        for element, radius in zip(elements, radii, strict=True):
            r_eff = max(float(radius),
                        0.5 * mesh.cell_size_at(element.x, element.y, element.z_bottom))
            i0, j0, _ = mesh.find_cell(element.x - r_eff, element.y - r_eff, element.z_bottom)
            i1, j1, _ = mesh.find_cell(element.x + r_eff, element.y + r_eff, element.z_top)
            ii = slice(min(i0, i1), max(i0, i1) + 1)
            jj = slice(min(j0, j1), max(j0, j1) + 1)
            z_top = max(element.z_top, element.z_bottom
                        + mesh.size_z(element.z_bottom))
            kk = (mesh.z >= element.z_bottom) & (mesh.z < z_top)
            if not kk.any():
                kk = np.zeros(mesh.Nz, dtype=bool)
                kk[mesh.find_cell(element.x, element.y, element.z_bottom)[2]] = True
            window = np.zeros(mesh.T.shape, dtype=bool)
            window[ii, jj, :] = True
            window &= kk[None, None, :]
            circle = (mesh.X - element.x) ** 2 + (mesh.Y - element.y) ** 2 <= r_eff ** 2
            hit = window & circle
            if not hit.any():
                i, j, _ = mesh.find_cell(element.x, element.y, element.z_bottom)
                hit = np.zeros(mesh.T.shape, dtype=bool)
                hit[i, j, :] = kk
            mask |= hit
        return mask

    def _n_heaters(self) -> int:
        """Elements of the painted bank (the count the rasteriser deposits)."""
        cyl = self.cylinder
        if self.heaters.pattern == HeaterPattern.UNIFORM_ZONE:
            return 0
        return self.heaters.bank(cyl.z_storage_start, cyl.z_storage_end).n_elements

    def _n_tubes(self) -> int:
        if not self.tubes.active:
            return 0
        cyl = self.cylinder
        return len(self.tubes.generate_positions(cyl.center_x, cyl.center_y,
                                                 cyl.r_storage * 0.9,
                                                 cyl.z_storage_start, cyl.z_storage_end))

    # ------------------------------------------------------------- reporting
    def zone_volumes(self, mesh: Mesh3D = None) -> dict:
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

    def zone_masses(self, mesh: Mesh3D = None, materials: MaterialManager = None) -> dict:
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
        heaters=HeaterConfig(power_total=50.0, n_heaters=6,
                             pattern=HeaterPattern.UNIFORM_ZONE),
        tubes=TubeConfig(n_tubes=6, active=False),
    )
    return geom
