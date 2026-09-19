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
"""
from __future__ import annotations

from dataclasses import dataclass

from ..core.geometry import BatteryGeometry
from ..core.materials import MaterialManager

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
    storage = materials.compute_effective_properties(
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

    # radial insulation + shell: the losses cross them and leave through h_lateral
    radial = [("insulation_radial", cyl.insulation_thickness, insulation.k,
               geometry.h_lateral),
              ("shell", cyl.shell_thickness, shell.k, geometry.h_lateral)]
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

    # the two slabs see h_top / the ground
    for name, thickness, h in (("slab_top", cyl.insulation_slab_top, geometry.h_top),
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


def binding_target(geometry: BatteryGeometry, materials: MaterialManager,
                   cells_per_layer: float = CELLS_PER_LAYER) -> float:
    """Finest cell size any region of the battery asks for [m]."""
    plans = plan_regions(geometry, materials, cells_per_layer)
    return min((p.target for p in plans), default=0.1)


def describe(plans: list[RegionPlan]) -> str:
    """Multi-line summary for the log and the GUI."""
    return "\n".join(str(p) for p in plans)
