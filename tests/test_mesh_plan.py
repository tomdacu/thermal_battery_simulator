"""The a priori plan and the regions a mesh refines: the air is not one of them.

The plan answers two questions: how fine does each *region of the battery* have to be
(``plan_regions``: ``thickness / N`` across a layer, ``2 k / h`` under a convective
surface), and *where* are those regions (``active_regions`` as boxes, one
:class:`~src.core.adaptive_mesh.RefinementBand` each).  What the tests below pin is the
second answer, because it is the one that spends cells: the boxes cover the vessel - the
sand, the insulation, the shell, the casing the ambient film sits on - and none of them
covers the air around it, which ``apply_environment`` excludes from the problem and holds
at the ambient temperature.
"""
from __future__ import annotations

from src.analysis.mesh_plan import MeshRegion, active_regions, plan_regions, region_bands
from src.core.adaptive_mesh import RefinementBand
from src.core.geometry import create_small_test_geometry
from src.core.materials import MaterialManager


def vessel_with(**kwargs):
    geometry = create_small_test_geometry()
    for name, value in kwargs.items():
        setattr(geometry.cylinder, name, value)
    return geometry


def test_the_boxes_cover_the_active_model_and_never_the_air():
    """Every box lies inside the vessel's envelope, and the four regions are there."""
    geometry = vessel_with()
    cyl = geometry.cylinder
    regions = active_regions(cyl, {"sand": 0.2, "insulation_radial": 0.02,
                                   "slab_bottom": 0.03, "slab_top": 0.01,
                                   "shell": 0.005, "casing": 0.2},
                             pipe_box=((cyl.center_x - 1.0, cyl.center_y - 1.0, 0.6),
                                       (cyl.center_x + 1.0, cyl.center_y + 1.0, 3.9)))
    names = {region.name for region in regions}
    assert names == {"sand", "insulation_radial", "shell", "slab_bottom", "slab_top",
                     "roof", "foundation", "pipe_wall"}
    for region in regions:
        for axis, (low, high) in enumerate(zip(region.low, region.high, strict=True)):
            assert high > low
            if axis == 2:
                assert low >= 0.0 and high <= cyl.z_cone_apex + 1e-9
            else:
                # the envelope is the shell's footprint (the foundation is wider, and it
                # is a solid: the concrete stays in the problem)
                reach = cyl.r_shell + cyl.foundation_margin
                assert low >= cyl.center_x - reach - 1e-9
                assert high <= cyl.center_x + reach + 1e-9

    # the target of every region is the one the caller asked for (the panel's mixture)
    size_of = {region.name: region.target for region in regions}
    assert size_of["sand"] == 0.2
    assert size_of["insulation_radial"] == 0.02
    assert size_of["shell"] == 0.005
    assert size_of["pipe_wall"] == 0.05          # the region's own default: 50 mm tube
    assert size_of["roof"] == size_of["foundation"] == 0.2   # the coarsest active target


def test_the_air_outside_the_shell_is_outside_every_box():
    """A point in the air down the side of the vessel belongs to no region.

    The check is the one that matters for the cell budget: if no box contains the point,
    the octree leaves the leaf there at whatever level its own 2:1 balance gives it.
    """
    geometry = vessel_with()
    cyl = geometry.cylinder
    regions = active_regions(cyl, {"sand": 0.2, "insulation_radial": 0.02})
    # a metre outside the shell, half way up the wall
    point = (cyl.center_x - cyl.r_shell - 1.0, cyl.center_y, 0.5 * cyl.z_shell_top)
    for region in regions:
        inside = all(low <= value < high
                     for value, low, high in zip(point, region.low, region.high,
                                                 strict=True))
        assert not inside, f"{region.name} covers air outside the vessel"
    # and the sand is covered by the sand box, so the check is not vacuous
    middle = (cyl.center_x, cyl.center_y, 0.5 * (cyl.z_storage_start + cyl.z_storage_end))
    assert any(region.name == "sand" and all(low <= value < high for value, low, high in
                                             zip(middle, region.low, region.high,
                                                 strict=True))
               for region in regions)


def test_region_bands_keep_every_target_and_reject_an_empty_request():
    """One band per region, same box, same size - and nothing to refine is an error."""
    region = MeshRegion("sand", 0.25, (1.0, 1.0, 0.5), (3.0, 3.0, 4.5))
    band, = region_bands([region])
    assert isinstance(band, RefinementBand)
    assert (band.low, band.high, band.size) == (region.low, region.high, 0.25)

    import pytest
    with pytest.raises(ValueError, match="at least one active region"):
        region_bands([MeshRegion("sand", 0.0, (1.0, 1.0, 0.5), (3.0, 3.0, 4.5))])


def test_the_a_priori_plan_of_the_test_battery_is_physical():
    """The plan of the shared fixture: four regions, every target below its thickness.

    A sanity rail rather than a pin on the numbers: whatever the materials database says,
    the rule is ``min(thickness / N, 2 k / h)`` and no region may ask for cells coarser
    than its own conduction length.
    """
    geometry = create_small_test_geometry()
    plans = plan_regions(geometry, MaterialManager())
    names = {plan.name for plan in plans}
    assert {"storage", "insulation_radial", "shell", "slab_top", "slab_bottom"} == names
    for plan in plans:
        assert 0.0 < plan.target <= plan.thickness + 1e-9
