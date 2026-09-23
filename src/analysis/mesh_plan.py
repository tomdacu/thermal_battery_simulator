"""A priori mesh plan: how fine each region has to be, from the physics alone.

Two classical criteria decide the cell size *before* any solve:

* a **solid layer** must carry at least ``cells_per_layer`` cells across its own
  thickness, otherwise the temperature drop inside the layer is not represented;
* a **surface with a film coefficient** ``h`` has a conductive sub-layer of thickness
  ``delta = k / h``.  The finite-volume half-cell resistance is ``h_cell / (2 k)`` and
  it must not dominate the film resistance ``1 / h``, so ``h_cell <= 2 k / h``.  This
  is the conduction analogue of the "first cell height" rule of CFD practice: for rock
  wool (k = 0.04 W/(m K)) with h = 5 W/(m^2 K) it asks for 16 mm, which is exactly why
  a 100 mm grid cannot converge on the losses - the surface drop is hidden inside the
  first cell.

The plan is a *starting point*, not the answer: :mod:`src.analysis.convergence` then
measures the real discretisation error with Richardson extrapolation and moves the
targets until the answer stops moving.  Starting from the plan is what makes the
search cheap: the first grid it builds is already in the right neighbourhood.

The plan is also the sharing point between the two meshes: :func:`refinement_bands`
turns the bands a :class:`~src.core.refinement.GridSpec` carries into the boxes an
octree refines, one box per band and with the same target, and :func:`tree_resolution`
says what box a tree needs to hold them.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from ..core.adaptive_mesh import RefinementBand
from ..core.environment import h_out
from ..core.geometry import BatteryGeometry, CylinderGeometry
from ..core.materials import MaterialManager
from ..core.refinement import GridSpec

#: cells across a solid layer whose temperature drop must be represented
CELLS_PER_LAYER = 8.0


@dataclass(frozen=True)
class RegionPlan:
    """One region of the battery and the cell size it asks for."""

    name: str
    thickness: float          # [m] the conduction length the region must resolve
    target: float             # [m] recommended cell size
    reason: str

    def __str__(self) -> str:
        return (f"{self.name:22s} thickness {self.thickness * 1000:7.1f} mm  "
                f"-> target {self.target * 1000:6.1f} mm   ({self.reason})")


def surface_layer(k: float, h: float) -> float:
    """Cell size that keeps the half-cell resistance below the film resistance [m]."""
    if h <= 0 or k <= 0:
        return float("inf")
    return 2.0 * k / h


def layer_target(thickness: float, cells: float = CELLS_PER_LAYER) -> float:
    """Cell size that puts ``cells`` cells across a layer of ``thickness`` [m]."""
    return max(thickness, 1e-9) / max(cells, 1.0)


def plan_regions(geometry: BatteryGeometry, materials: MaterialManager,
                 cells_per_layer: float = CELLS_PER_LAYER) -> list[RegionPlan]:
    """Recommended cell size for every region of the battery.

    The binding criterion is the smaller of the layer rule and the surface rule, so
    the plan is conservative: it never asks for fewer cells than the physics needs.
    """
    cyl = geometry.cylinder
    # the packed bed the painter paints (the packing fraction is the *solid* share: it
    # used to be handed to compute_effective_properties as the porosity, which made the
    # plan's bed 63 % air - 0.15 W/(m K) instead of 0.52 for steatite)
    storage = materials.compute_packed_bed_properties(
        geometry.storage_material, geometry.packing_fraction)
    insulation = materials.get(geometry.insulation_material)
    shell = materials.get(geometry.shell_material)

    plans: list[RegionPlan] = []

    # the storage is heated from inside and cooled only through its own conduction
    # path: its length scale is the distance heat has to travel to the insulation
    storage_length = min(cyl.r_storage, cyl.height)
    plans.append(RegionPlan(
        "storage", storage_length, layer_target(storage_length, cells_per_layer),
        f"{cells_per_layer:.0f} cells across min(r, height), k = {storage.k:.2f}"))

    # radial insulation + shell: the losses cross them and leave through the outer film.
    # The film sits on the *shell* - the steel casing is the surface the air touches - so
    # the surface rule belongs to the shell, with the film the correlations give
    # (``src/core/environment.py``).  The insulation is inside the casing and faces no
    # film: its rule is the layer rule alone.  (It used to take the box-face film
    # ``h_lateral`` and asked 16 mm of cells through the whole insulation for a surface
    # that does not exist in the excluded-air model.)
    film_h = outer_film(geometry)
    radial = [("insulation_radial", cyl.insulation_thickness, insulation.k, 0.0),
              ("shell", cyl.shell_thickness, shell.k, film_h)]
    for name, thickness, k, h in radial:
        if thickness <= 0:
            continue
        film = surface_layer(k, h)
        target = min(layer_target(thickness, cells_per_layer), film)
        reason = (f"{cells_per_layer:.0f} cells across {thickness * 1000:.0f} mm, "
                  f"k = {k:.2f}")
        if film < layer_target(thickness, cells_per_layer):
            reason = (f"surface layer 2k/h = {film * 1000:.1f} mm "
                      f"(k = {k:.2f}, h = {h:.1f})")
        plans.append(RegionPlan(name, thickness, target, reason))

    # the two slabs are inside the casing too: the top one under the steel plate and the
    # roof, the bottom one on the foundation - no film on either
    for name, thickness, h in (("slab_top", cyl.insulation_slab_top, 0.0),
                               ("slab_bottom", cyl.insulation_slab_bottom, 0.0)):
        if thickness <= 0:
            continue
        film = surface_layer(insulation.k, h)
        target = min(layer_target(thickness, cells_per_layer), film)
        reason = (f"{cells_per_layer:.0f} cells across {thickness * 1000:.0f} mm, "
                  f"k = {insulation.k:.2f}")
        if film < layer_target(thickness, cells_per_layer):
            reason = (f"surface layer 2k/h = {film * 1000:.1f} mm "
                      f"(k = {insulation.k:.2f}, h = {h:.1f})")
        plans.append(RegionPlan(name, thickness, target, reason))

    return plans


def outer_film(geometry: BatteryGeometry) -> float:
    """The outer film of the vessel at its design point [W/(m^2 K)].

    The same correlations and the same nominal surface rise the painter uses
    (:meth:`~src.core.geometry.BatteryGeometry.apply_environment`), evaluated without a
    mesh: natural convection on the vessel plus the wind.
    """
    cyl = geometry.cylinder
    film = h_out(geometry.t_ambient + geometry.film_delta_t, geometry.t_ambient,
                 height=max(cyl.z_cone_apex - cyl.base_z, 0.1),
                 width=max(2.0 * cyl.r_shell, 0.1), wind_speed=geometry.wind_speed)
    return float(film["natural"] + film["wind"])


def binding_target(geometry: BatteryGeometry, materials: MaterialManager,
                   cells_per_layer: float = CELLS_PER_LAYER) -> float:
    """Finest cell size any region of the battery asks for [m]."""
    plans = plan_regions(geometry, materials, cells_per_layer)
    return min((p.target for p in plans), default=0.1)


def describe(plans: list[RegionPlan]) -> str:
    """Multi-line summary for the log and the GUI."""
    return "\n".join(str(p) for p in plans)


# ------------------------------------------------------- the regions of the model
@dataclass(frozen=True)
class MeshRegion:
    """One box of the *active* model and the cell size it asks for [m].

    The boxes are what an octree refines - a leaf is split while it intersects a box
    and is still larger than that box's size - so a box is what spends cells.  The
    regions below are the ones the physics needs: the sand, the insulation (the radial
    ring and the two slabs), the shell, the casing (the surfaces the ambient film sits
    on: the roof and the foundation) and the wall of the buried pipes.  The air around
    the vessel is deliberately **not** one of them: it is excluded from the problem by
    :meth:`~src.core.geometry.BatteryGeometry.apply_environment` and is held at the
    ambient temperature, so a leaf refined there would be a leaf spent on a cell that
    carries no flux.
    """

    name: str
    target: float                                  # [m] cell size asked for
    low: tuple[float, float, float]
    high: tuple[float, float, float]
    reason: str = ""
    #: a radial region: the annulus ``r_inner..r_outer`` about the vertical axis through
    #: ``axis`` (``low``/``high`` are its bounding box); ``None`` = the box itself
    axis: tuple[float, float] | None = None
    r_inner: float = 0.0
    r_outer: float = 0.0

    @property
    def volume(self) -> float:
        height = max(self.high[2] - self.low[2], 0.0)
        if self.axis is not None:
            return float(np.pi * (self.r_outer ** 2 - self.r_inner ** 2) * height)
        return float(np.prod([max(h - low, 0.0)
                              for low, h in zip(self.low, self.high, strict=True)]))

    def __str__(self) -> str:
        box = " ".join(f"{axis}[{low:.2f}, {high:.2f}]"
                       for axis, low, high in zip("xyz", self.low, self.high, strict=True))
        return (f"{self.name:18s} target {self.target * 1000:6.1f} mm  {box}"
                + (f"  ({self.reason})" if self.reason else ""))


def active_regions(cylinder: CylinderGeometry, targets: Mapping[str, float], *,
                   pipe_box: tuple[Sequence[float], Sequence[float]] | None = None,
                   pipe_boxes: Sequence[tuple[Sequence[float], Sequence[float]]] = ()
                   ) -> list[MeshRegion]:
    """The boxes of the active model, each with the cell size it asks for [m].

    ``targets`` gives the cell size per region: ``sand``, ``insulation_radial``,
    ``slab_bottom``, ``slab_top``, ``shell``, ``casing`` and ``pipe_wall`` (the last
    one only when ``pipe_box`` is given).  A region whose key is missing keeps its own
    physical length - the sand's radius, the insulation's thickness - so a caller that
    only wants the boxes can pass an empty mapping.  ``pipe_boxes`` are one box per riser
    (the tree then refines a column around each pipe, not the whole bundle);
    ``pipe_box`` is the single box of the whole bundle.

    The sand and the two slabs get one box each; the insulation ring and the shell are
    annuli, covered by four boxes each; the **casing** is the pair of boxes whose
    surfaces carry the ambient film (the conical roof and the foundation under the
    floor) and it is what keeps a 20 mm shell or a 5 mm plate from being widened into a
    metre of steel by the voxel painter when the grid is coarse.  The box of the pipes
    is the caller's: the bundle is the network's business, not the vessel's.
    """
    cyl = cylinder
    cx, cy = cyl.center_x, cyl.center_y

    def target(name: str, length: float) -> float:
        return float(targets.get(name, length))

    def ring(name: str, size: float, inner: float, outer: float, z_low: float,
             z_high: float, reason: str = "") -> MeshRegion:
        """A radial region of the vessel: the annulus (a disc for inner = 0)."""
        return MeshRegion(name, size, (cx - outer, cy - outer, z_low),
                          (cx + outer, cy + outer, z_high), reason,
                          axis=(cx, cy), r_inner=inner, r_outer=outer)

    # the vessel is a cylinder: every region of it is a disc or an annulus about its
    # axis, so a leaf is refined where the region *is* - not in the corners of a box
    regions: list[MeshRegion] = [
        ring("sand", target("sand", cyl.r_storage), 0.0, cyl.r_storage,
             cyl.z_storage_start, cyl.z_storage_end,
             "the storage radius, resolved by the bed's own cell size"),
    ]
    if cyl.insulation_slab_bottom > 0:
        regions.append(ring("slab_bottom", target("slab_bottom", cyl.insulation_slab_bottom),
                            0.0, cyl.r_storage, cyl.z_slab_bottom_start,
                            cyl.z_storage_start))
    if cyl.insulation_slab_top > 0:
        regions.append(ring("slab_top", target("slab_top", cyl.insulation_slab_top),
                            0.0, cyl.r_storage, cyl.z_slab_top_start, cyl.z_slab_top_end))
    for name, inner, outer in (("insulation_radial", cyl.r_storage, cyl.r_shell),
                               ("shell", cyl.r_insulation, cyl.r_shell)):
        if outer - inner <= 0:
            continue
        regions.append(ring(
            name, target(name, outer - inner), inner, outer, cyl.base_z, cyl.z_shell_top,
            f"{1000.0 * (outer - inner):.0f} mm across the "
            f"{'insulation ring' if name == 'insulation_radial' else 'shell'}"))
    casing = target("casing", min(targets.values(), default=cyl.r_storage))
    if cyl.roof_height > 0:
        regions.append(ring("roof", casing, 0.0, cyl.r_shell, cyl.z_cone_base,
                            cyl.z_cone_apex, "the cone roof carries the top film"))
    if cyl.base_z > 0:
        reach = cyl.r_shell + max(cyl.foundation_margin, 0.0)
        regions.append(ring("foundation", casing, 0.0, reach, 0.0, cyl.base_z,
                            "the concrete under the floor carries the ground face"))
    boxes = list(pipe_boxes) + ([pipe_box] if pipe_box is not None else [])
    for low, high in boxes:
        regions.append(MeshRegion("pipe_wall", target("pipe_wall", 0.05),
                                  tuple(low), tuple(high),
                                  "the network's tubes are the heat-transfer surface"))
    return regions


def region_bands(regions: Sequence[MeshRegion]) -> tuple[RefinementBand, ...]:
    """The regions as the boxes an octree refines: one band per region, same target.

    This is the tree's road into the a priori plan: the panel passes the boxes of
    :func:`active_regions` and the octree refines each of them to its own size, so a
    leaf in the air around the vessel - where no region is - is left at the level the
    tree's own balance happens to give it, which is the whole point of planning by
    region instead of by axis.
    """
    bands: list[RefinementBand] = []
    for region in regions:
        if region.target <= 0.0 or region.volume <= 0.0:
            continue
        bands.append(RefinementBand(low=tuple(region.low), high=tuple(region.high),
                                    size=float(region.target), axis=region.axis,
                                    r_inner=float(region.r_inner),
                                    r_outer=float(region.r_outer)))
    if not bands:
        raise ValueError("at least one active region is required")
    return tuple(bands)


# --------------------------------------------------------------- the tree road
#: the octree packs 21 bits per coordinate: this is the finest box it can index
MAX_TREE_LEVEL = 21
#: below four cells a side the flux-jump estimate has no stencil to look at, so a tree
#: that coarse could not be refined on the indicator at all (see
#: :meth:`src.solver.octree_solver.OctreeSteadySolver.indicator`)
MIN_TREE_LEVEL = 2


def refinement_bands(spec: GridSpec, extents: Sequence[float]) -> tuple[RefinementBand, ...]:
    """The bands of a :class:`GridSpec` as tree boxes: one box per band, same target.

    This is what keeps the two roads starting from the same a priori estimate: the
    physical targets - ``thickness / N``, ``2 k / h``, the pitch between the tubes - are
    decided once and land in both vocabularies unchanged, a band of one axis and a box of
    the octree.  A band is a slice of one axis, so its box spans the other two: a leaf
    intersects the box of the band that covers its own coordinate, and
    :meth:`AdaptiveMesh.refine_bands` refines it to the smallest size among the boxes it
    touches - the lower envelope of the three axes' targets, which is the field
    :meth:`GridSpec.edges` builds for the structured mesh.

    ``extents`` is ``(Lx, Ly, Lz)`` of the domain the boxes are written in.  Bands that
    cover nothing are skipped, exactly as :func:`src.core.refinement.partition` skips them.
    """
    boxes: list[RefinementBand] = []
    for axis, bands in enumerate((spec.x, spec.y, spec.z)):
        for band in bands:
            if band.length <= 0 or band.target <= 0:
                continue
            low = [0.0, 0.0, 0.0]
            high = [float(extents[0]), float(extents[1]), float(extents[2])]
            low[axis], high[axis] = float(band.start), float(band.end)
            boxes.append(RefinementBand(low=(low[0], low[1], low[2]),
                                        high=(high[0], high[1], high[2]),
                                        size=float(band.target)))
    if not boxes:
        raise ValueError("a grid spec needs at least one refinement band")
    return tuple(boxes)


def tree_resolution(extents: Sequence[float], max_cells: int) -> tuple[int, float]:
    """``(n_finest, physical_size)`` of the tree a cell budget affords.

    An octree spans a *cube* (the largest extent of the domain) whose leaves are a power of
    two of its finest cell, and that finest cell is the floor under the whole search: the
    bands are the a priori plan and a refinement round goes *below* them wherever the field
    asks for it, so the resolution has to leave room.  The budget sets it - a whole cube
    refined to ``h`` holds ``box^3 / h^3`` leaves - which is the tree's answer to the
    scaling :func:`src.core.refinement.graded_edges` gives a graded grid: the *request* is
    what gets scaled, and this is the floor it may reach down to.

    The level is rounded **down**, and that is not a detail: the a priori plan pushes leaves
    down to the floor wherever a region asks for a fine target, so the floor *is* the worst
    case the user pays for.  Rounding up gave a floor whose whole cube holds up to six times
    the budget (128^3 = 2.1 M leaves for a 400 k budget, measured 670 k leaves on the default
    vessel, three and a half minutes to paint).  Rounding down keeps the worst case inside
    the budget, and the price is that a target finer than the floor cannot be reached - the
    panel reports the targets the tree actually realised, so the gap is visible rather than
    implied.  The level is also clamped at both ends: the octree indexes 21 bits per
    coordinate, and below four cells a side the flux-jump estimate has no stencil to look
    at.
    """
    box = float(max(extents))
    if box <= 0.0:
        raise ValueError(f"the domain has no extent: {tuple(extents)}")
    affordable = (box ** 3 / max(int(max_cells), 1)) ** (1.0 / 3.0)
    level = int(np.floor(np.log2(box / affordable))) if affordable < box else 0
    level = int(np.clip(level, MIN_TREE_LEVEL, MAX_TREE_LEVEL))
    n_finest = 1 << level
    return n_finest, box / n_finest
