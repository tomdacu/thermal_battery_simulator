"""Hairpin (U-shaped) tubular heater elements and their rasterisation.

The reference design (see ``docs/10_MESH_AND_HEATHERS.md`` and the picture in
``photo/heating_elements_3D.png``) is a *flanged immersion heater bank*: rows of
U-shaped tubular elements welded to a flange on the roof, hanging into the sand and
supported at the bottom by a plate.  Compared with the old straight rods this model

* keeps the **sheath** as the geometric entity that the mesh must resolve
  (a 12 mm tube is invisible in a 200 mm cell);
* sets the power from the **surface power density** ``P / (pi d L_active)`` in
  W/cm^2, which is how immersion heaters are actually rated (3-8 W/cm^2);
* deposits the power on the cells of the *active* length only: the cold shank
  through the insulation is part of the sheath but not a heat source.

Everything here is pure geometry + numpy: no Qt, no solver.  The rasteriser runs on
either mesh - the structured ``Mesh3D`` and the octree-based ``AdaptiveMesh`` - through
the mesh view of :mod:`src.core.pipes`, so a bank lands on the same cells and deposits
the same power on the tree as on the ``Mesh3D`` of the same cells.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .mesh import Mesh3D
# the mesh view the rasterisers share (cell centres, cell sizes, point location) lives
# with the pipe rasteriser, the lowest layer that needs it
from .pipes import cell_centres, cell_index, cell_sizes, flat_cells

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from .adaptive_mesh import AdaptiveMesh

#: rating range of sheathed elements in solids [W/cm^2]
SURFACE_POWER_MIN_W_CM2 = 3.0
SURFACE_POWER_LIMIT_W_CM2 = 8.0
#: default sheath (the user's design: stainless tube, 12 mm outer diameter)
DEFAULT_SHEATH_DIAMETER = 0.012
DEFAULT_SHEATH_MATERIAL = "stainless_steel"   # key of the material database


@dataclass
class HairpinElement:
    """One U-shaped tubular element hanging from the flange.

    ``center_x``/``center_y`` is the middle of the bend, the two legs are
    ``leg_spacing`` apart along ``angle_rad``.  ``active_length`` starts at the
    bottom: the bend and the lower part of the legs are inside the sand, the rest
    (``cold_shank``) crosses the top insulation and the air.
    """

    center_x: float
    center_y: float
    leg_spacing: float = 0.08
    sheath_diameter: float = DEFAULT_SHEATH_DIAMETER
    active_length: float = 2.0
    cold_shank: float = 0.15
    rated_power: float = 1000.0
    angle_rad: float = 0.0
    bend_chords: int = 4

    # ------------------------------------------------------------------ props
    @property
    def radius(self) -> float:
        return 0.5 * self.sheath_diameter

    @property
    def bend_radius(self) -> float:
        return 0.5 * self.leg_spacing

    @property
    def sheath_area(self) -> float:
        """Heated sheath area [m^2]: the two legs plus the bend (not the cold shank)."""
        # the element is a hairpin: two legs of (active_length - bend_radius) each,
        # joined by a semicircle of radius leg_spacing/2
        return (np.pi * self.sheath_diameter
                * (2.0 * max(self.active_length - self.bend_radius, 0.0)
                   + np.pi * self.bend_radius))

    @property
    def surface_power_w_cm2(self) -> float:
        """Rated power per unit of heated sheath area [W/cm^2]."""
        area = self.sheath_area
        return float(self.rated_power / area / 1e4) if area > 0 else float("inf")

    def leg_positions(self) -> tuple[tuple[float, float], tuple[float, float]]:
        dx = 0.5 * self.leg_spacing * np.cos(self.angle_rad)
        dy = 0.5 * self.leg_spacing * np.sin(self.angle_rad)
        return ((self.center_x - dx, self.center_y - dy),
                (self.center_x + dx, self.center_y + dy))

    # --------------------------------------------------------------- geometry
    def segments(self, z_bottom: float, z_top: float, active_until: float
                 ) -> list[tuple[tuple[float, float, float], tuple[float, float, float]]]:
        """Segments of the element from the bend to the flange.

        The U lies in the vertical plane through the two legs: the legs run from the
        bend tangent points (``z_bottom + bend_radius``) up to ``z_top``, and the
        180 deg bend is approximated by ``bend_chords`` chords whose lowest point is
        ``z_bottom``.
        """
        (x1, y1), (x2, y2) = self.leg_positions()
        radius = self.bend_radius
        if radius <= 0:                      # degenerate: two straight legs
            return [((x1, y1, z_bottom), (x1, y1, z_top)),
                    ((x2, y2, z_bottom), (x2, y2, z_top))]
        centre = (0.5 * (x1 + x2), 0.5 * (y1 + y2), z_bottom + radius)
        ux, uy = (x1 - centre[0]) / radius, (y1 - centre[1]) / radius
        points = [
            (centre[0] + radius * np.cos(phi) * ux,
             centre[1] + radius * np.cos(phi) * uy,
             centre[2] - radius * np.sin(phi))
            for phi in (np.pi * i / self.bend_chords
                        for i in range(self.bend_chords + 1))
        ]
        segments = [((x1, y1, centre[2]), (x1, y1, z_top)),
                    ((x2, y2, centre[2]), (x2, y2, z_top))]
        segments += [(points[i], points[i + 1]) for i in range(self.bend_chords)]
        return [(a, b) for a, b in segments if a != b]


@dataclass
class HeaterBank:
    """Rows x columns (or rings) of hairpin elements on a flange."""

    active: bool = True
    sheath_diameter: float = DEFAULT_SHEATH_DIAMETER
    sheath_material: str = DEFAULT_SHEATH_MATERIAL
    active_length: float = 2.0
    cold_shank: float = 0.15
    leg_spacing: float = 0.08
    rows: int = 4
    columns: int = 4
    power_per_element: float = 1000.0
    offset_bottom: float = 0.1
    offset_top: float = 0.1
    support_plate_offset: float = 0.05
    flange_offset: float = 0.03
    layout: str = "grid"
    n_rings: int = 2
    bend_chords: int = 4

    @property
    def n_elements(self) -> int:
        return max(self.rows, 0) * max(self.columns, 0)

    @property
    def total_power_w(self) -> float:
        return self.n_elements * self.power_per_element

    def surface_power_w_cm2(self) -> float:
        """Surface power of one element [W/cm^2] (the bank is uniform by design)."""
        element = HairpinElement(0.0, 0.0, self.leg_spacing, self.sheath_diameter,
                                 self.active_length, self.cold_shank, self.power_per_element,
                                 bend_chords=self.bend_chords)
        return element.surface_power_w_cm2

    # --------------------------------------------------------------- geometry
    def generate_elements(self, center_x: float, center_y: float, r_storage: float,
                          phase_rad: float = 0.0) -> list[HairpinElement]:
        """Element list; positions are always inside ``r_storage`` (validated)."""
        positions: list[tuple[float, float, float]] = []
        if self.layout == "ring":
            for ring in range(max(self.n_rings, 1)):
                radius = r_storage * (0.35 + 0.5 * ring / max(self.n_rings - 1, 1))
                count = max(int(round(2 * np.pi * radius / max(self.leg_spacing * 2.5,
                                                               1e-3))), 1)
                for i in range(count):
                    angle = phase_rad + 2 * np.pi * i / count + 0.5 * ring
                    positions.append((center_x + radius * np.cos(angle),
                                      center_y + radius * np.sin(angle), angle))
        else:
            span_x = (self.columns - 1) * self.leg_spacing * 2.0
            span_y = (self.rows - 1) * self.leg_spacing * 2.0
            for row in range(max(self.rows, 0)):
                for col in range(max(self.columns, 0)):
                    positions.append((center_x - span_x / 2 + col * self.leg_spacing * 2.0,
                                      center_y - span_y / 2 + row * self.leg_spacing * 2.0,
                                      np.pi / 4))
        return [HairpinElement(x, y, self.leg_spacing, self.sheath_diameter,
                               self.active_length, self.cold_shank,
                               self.power_per_element, angle, self.bend_chords)
                for x, y, angle in positions]


# ------------------------------------------------------------------ rasterise
@dataclass
class RasterResult:
    """What the bank occupies in a given mesh and what it will release."""

    mask: np.ndarray                       # sheath cells (material)
    active_mask: np.ndarray                # cells that carry the power
    elements: list[HairpinElement] = field(default_factory=list)
    element_masks: list[np.ndarray] = field(default_factory=list)
    power: float = 0.0                     # total deposited [W]
    surface_power_w_cm2: float = 0.0
    min_cells_across_sheath: int = 0
    problem: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def n_cells_per_element(self) -> list[int]:
        return [int(m.sum()) for m in self.element_masks]


def _distance_to_segment(px: np.ndarray, py: np.ndarray, pz: np.ndarray,
                         a: tuple[float, float, float],
                         b: tuple[float, float, float]) -> np.ndarray:
    ax, ay, az = a
    bx, by, bz = b
    abx, aby, abz = bx - ax, by - ay, bz - az
    norm = abx * abx + aby * aby + abz * abz
    if norm <= 0:
        return np.sqrt((px - ax) ** 2 + (py - ay) ** 2 + (pz - az) ** 2)
    t = ((px - ax) * abx + (py - ay) * aby + (pz - az) * abz) / norm
    t = np.clip(t, 0.0, 1.0)
    return np.sqrt((px - (ax + t * abx)) ** 2 + (py - (ay + t * aby)) ** 2
                   + (pz - (az + t * abz)) ** 2)


def _segment_window(mesh: Mesh3D | AdaptiveMesh, lo: Sequence[float],
                    hi: Sequence[float],
                    radius: float) -> tuple[slice, slice, slice] | np.ndarray:
    """The cells a segment can reach, as an index window or a flat selection.

    Every cell the mask can mark has a centre within ``radius`` of the segment, hence
    inside the box ``[lo - radius, hi + radius]`` the segment spans, so the cells outside
    that box can be left out of the distance evaluation without changing the mask.  A
    ``Mesh3D`` answers with the index block the rasteriser has always used; a tree, which
    has no index arithmetic, with the leaves whose centres fall in the band.
    """
    if isinstance(mesh, Mesh3D):
        i0, j0, k0 = mesh.find_cell(lo[0] - radius, lo[1] - radius, lo[2] - radius)
        i1, j1, k1 = mesh.find_cell(hi[0] + radius, hi[1] + radius, hi[2] + radius)
        return (slice(min(i0, i1), max(i0, i1) + 1),
                slice(min(j0, j1), max(j0, j1) + 1),
                slice(min(k0, k1), max(k0, k1) + 1))
    X, Y, Z = cell_centres(mesh)
    near = np.ones(X.shape, dtype=bool)
    for low, high, coordinate in ((lo[0], hi[0], X), (lo[1], hi[1], Y), (lo[2], hi[2], Z)):
        near &= (low - radius <= coordinate) & (coordinate <= high + radius)
    return np.flatnonzero(near)


def _segment_mask(mesh: Mesh3D | AdaptiveMesh, a: tuple[float, float, float],
                  b: tuple[float, float, float], radius: float) -> np.ndarray:
    """Cells whose centre is within ``radius`` of the segment ``a``-``b``.

    The cell size is also honoured: an element thinner than the local cell still
    owns the cells it passes through (``radius`` is widened to half a cell), so a
    12 mm sheath can never vanish from a coarse mesh without being reported.

    The mask is a mask over the cell centres - a 3-D array on ``Mesh3D`` and a flat one on
    a tree, each shaped like the mesh's own fields - and the cells the axis crosses are
    added cell by cell on either mesh, through the index a mask takes there.
    """
    lo = [min(a[i], b[i]) for i in range(3)]
    hi = [max(a[i], b[i]) for i in range(3)]
    window = _segment_window(mesh, lo, hi, radius)
    X, Y, Z = cell_centres(mesh)
    local = cell_sizes(mesh)
    scale = local[window]
    distance = _distance_to_segment(X[window], Y[window], Z[window], a, b)
    mask = np.zeros(X.shape, dtype=bool)
    mask[window] = distance <= np.maximum(radius, 0.5 * scale)
    # the cells the axis itself crosses: an element thinner than a cell must still
    # form an unbroken chain along its length, otherwise the power would be
    # deposited in a dotted line and the temperature field would be spiky
    length = float(np.sqrt(sum((b[i] - a[i]) ** 2 for i in range(3))))
    # the step is the smallest cell the segment can reach, so the walk cannot step over a
    # cell; a band holding no centre at all - a short segment in a coarse mesh - falls back
    # to the cell that holds the start of the segment
    step = max(float(scale.min()) if scale.size else mesh.cell_size_at(*a), 1e-6)
    for t in np.linspace(0.0, 1.0, max(int(np.ceil(length / step)) + 1, 2)):
        point = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]),
                 a[2] + t * (b[2] - a[2]))
        mask[cell_index(mesh, *point)] = True
    return mask


def _clip_segment(a: tuple[float, float, float], b: tuple[float, float, float],
                  z_max: float):
    """The part of a segment with ``z <= z_max`` (None when it is entirely above)."""
    if min(a[2], b[2]) > z_max:
        return None
    if max(a[2], b[2]) <= z_max:
        return a, b
    t = (z_max - a[2]) / (b[2] - a[2])
    cut = (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), z_max)
    return (a, cut) if a[2] <= b[2] else (cut, b)


def rasterize(bank: HeaterBank, mesh: Mesh3D | AdaptiveMesh, cx: float, cy: float,
              r_storage: float, z_storage_start: float,
              z_storage_end: float) -> RasterResult:
    """Mark the cells of the bank and prepare the power deposit.

    The power of an element is spread over the cells of its *active* length, so
    ``sum(Q_source * V)`` equals the rated power whatever the discretisation.

    The masks are masks over the cell centres and follow the mesh, so the same bank
    rasterised on a tree and on the ``Mesh3D`` of the same cells marks the same cells and
    deposits the same power; the fields the caller writes them into are shaped the same
    way, so ``mesh.Q_source[raster.active_mask] = bank.total_power_w /
    mesh.V[raster.active_mask].sum()`` is the deposit on either mesh.
    """
    elements = bank.generate_elements(cx, cy, r_storage)
    mask = np.zeros(mesh.T.shape, dtype=bool)
    active = np.zeros(mesh.T.shape, dtype=bool)
    element_masks: list[np.ndarray] = []
    z_bottom = z_storage_start + bank.offset_bottom
    z_top = min(z_storage_end - bank.offset_top, z_bottom + bank.active_length)
    z_flange = z_storage_end + bank.flange_offset
    active_until = z_bottom + bank.active_length

    for element in elements:
        own = np.zeros(mesh.T.shape, dtype=bool)
        live = np.zeros(mesh.T.shape, dtype=bool)
        for a, b in element.segments(z_bottom, z_flange, active_until):
            own |= _segment_mask(mesh, a, b, element.radius)
            clipped = _clip_segment(a, b, z_top)      # the cold shank carries no power
            if clipped is not None:
                live |= _segment_mask(mesh, *clipped, element.radius)
        element_masks.append(own)
        mask |= own
        active |= live

    covered = np.array([int(m.sum()) for m in element_masks], dtype=int)
    sheath_cells = _min_cells_across_sheath(mesh, elements, z_bottom)
    if sheath_cells < 1 and mask.sum() != 0:
        result_note = ("the sheath is thinner than a cell here: the element is "
                       "represented by the cells it crosses (refine the heater region "
                       "to resolve its surface)")
    else:
        result_note = None
    result = RasterResult(mask=mask, active_mask=active, elements=elements,
                          element_masks=element_masks,
                          power=bank.total_power_w,
                          surface_power_w_cm2=bank.surface_power_w_cm2(),
                          min_cells_across_sheath=sheath_cells)
    if result_note:
        result.notes.append(result_note)
    if bank.active and active.sum() == 0:
        result.problem = "no heater cell is inside the storage band: check the offsets"
    if not elements:
        result.problem = "the heater bank has no element (rows or columns are zero)"
    elif covered.size and covered.min() < 2:
        result.problem = (f"element {int(np.argmin(covered))} covers "
                          f"{int(covered.min())} cell(s): the sheath "
                          f"({bank.sheath_diameter * 1000:.1f} mm) is too small for the "
                          f"local cell size - refine the mesh")
    return result


def _min_cells_across_sheath(mesh: Mesh3D | AdaptiveMesh, elements: list[HairpinElement],
                             z_bottom: float = 0.0) -> int:
    """Cells the sheath diameter spans across the legs (min over the elements).

    The legs are vertical, so what resolves the sheath is the horizontal cell size
    where the element sits - the two per-axis sizes on ``Mesh3D``, the leaf edge on a
    tree, which is cubic.
    """
    if not elements:
        return 0
    cells = []
    for element in elements:
        if isinstance(mesh, Mesh3D):
            i, j, _ = mesh.find_cell(element.center_x, element.center_y, z_bottom)
            local = min(mesh.dx[i], mesh.dy[j])
        else:
            local = mesh.cell_size_at(element.center_x, element.center_y, z_bottom)
        cells.append(max(int(np.floor(element.sheath_diameter / local)), 0))
    return min(cells)


def validate_bank(bank: HeaterBank, mesh: Mesh3D | AdaptiveMesh, cx: float, cy: float,
                  r_storage: float, z_storage_start: float, z_storage_end: float,
                  tubes: list | None = None) -> list[str]:
    """Problems that make the bank unrepresentable or unsafe ([] if fine).

    ``tubes`` may be a list of objects with ``x``, ``y`` and ``radius`` attributes.
    Warnings (not errors) are prefixed with ``"warning: "``.

    Every check is a check on the cells the bank lands in, so the report is the same on
    either mesh wherever the two meshes have the same cells.
    """
    problems: list[str] = []
    if not bank.active:
        return problems
    if bank.n_elements == 0 and bank.layout != "ring":
        problems.append("the heater bank has no element: rows and columns must be > 0")
        return problems
    if bank.active_length <= 0:
        problems.append("heater active_length must be > 0")
    if bank.leg_spacing <= bank.sheath_diameter:
        problems.append(
            f"the two legs of an element would touch: leg_spacing "
            f"({bank.leg_spacing:.3f} m) must exceed the sheath diameter "
            f"({bank.sheath_diameter:.3f} m)")
    if bank.active_length + bank.cold_shank <= 0:
        problems.append("heater element has no length")

    z_bottom = z_storage_start + bank.offset_bottom
    z_top = min(z_storage_end - bank.offset_top, z_bottom + bank.active_length)
    if z_top <= z_bottom:
        problems.append("heater offsets leave no room inside the storage band")

    elements = bank.generate_elements(cx, cy, r_storage)
    for index, element in enumerate(elements):
        reach = float(np.hypot(element.center_x - cx, element.center_y - cy)) \
            + element.bend_radius
        if reach > r_storage:
            problems.append(
                f"element {index} reaches r = {reach:.3f} m, outside the storage wall "
                f"({r_storage:.3f} m)")
        local = mesh.cell_size_at(element.center_x, element.center_y,
                                  0.5 * (z_bottom + z_top))
        if element.leg_spacing <= element.sheath_diameter + 2.0 * local:
            problems.append(
                f"warning: element {index}: the legs are {element.leg_spacing:.3f} m "
                f"apart with {local:.3f} m cells, so the mesh sees a single rod rather "
                f"than a U")
        (p1, p2) = element.leg_positions()
        z_leg = z_bottom + element.bend_radius
        leg1 = _segment_mask(mesh, (p1[0], p1[1], z_leg), (p1[0], p1[1], z_top),
                             element.radius)
        leg2 = _segment_mask(mesh, (p2[0], p2[1], z_leg), (p2[0], p2[1], z_top),
                             element.radius)
        if np.any(leg1 & leg2):
            problems.append(
                f"element {index}: the two legs share mesh cells (leg spacing "
                f"{element.leg_spacing:.3f} m, mesh {local:.3f} m): increase the spacing "
                f"or refine the heater region")
        for tube in tubes or []:
            distance = float(np.hypot(element.center_x - tube.x, element.center_y - tube.y))
            if distance < element.bend_radius + getattr(tube, "radius", 0.0) + element.radius:
                problems.append(f"element {index} collides with a heat-exchanger tube")
                break

    surface = bank.surface_power_w_cm2()
    if surface > SURFACE_POWER_LIMIT_W_CM2:
        problems.append(f"warning: surface power {surface:.1f} W/cm2 exceeds the "
                        f"{SURFACE_POWER_LIMIT_W_CM2:.0f} W/cm2 limit of sheathed elements")
    elif surface < SURFACE_POWER_MIN_W_CM2:
        problems.append(f"warning: surface power {surface:.1f} W/cm2 is below the "
                        f"{SURFACE_POWER_MIN_W_CM2:.0f} W/cm2 working range")

    # elements sharing cells: the power would be deposited twice
    seen: dict[int, int] = {}
    for index, element in enumerate(elements):
        cells = flat_cells(_element_cells(mesh, element, z_bottom, z_top))
        for cell in np.flatnonzero(cells):
            other = seen.setdefault(int(cell), index)
            if other != index:
                problems.append(f"elements {other} and {index} share a mesh cell: "
                                f"increase the element spacing")
                break
        if len(problems) and "share a mesh cell" in problems[-1]:
            break
    return problems


def _element_cells(mesh: Mesh3D | AdaptiveMesh, element: HairpinElement, z_bottom: float,
                   z_top: float) -> np.ndarray:
    mask = np.zeros(mesh.T.shape, dtype=bool)
    for a, b in element.segments(z_bottom, z_bottom + element.active_length, z_top):
        mask |= _segment_mask(mesh, a, b, element.radius)
    return mask
