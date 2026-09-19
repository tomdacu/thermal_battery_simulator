"""Hairpin heater bank: geometry, rasterisation, power deposit and validations."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.geometry import (BatteryGeometry, CylinderGeometry, HeaterConfig, HeaterPattern,
                               TubeConfig)
from src.core.heaters import (SURFACE_POWER_LIMIT_W_CM2, SURFACE_POWER_MIN_W_CM2,
                              HeaterBank, HairpinElement,
                              rasterize, validate_bank)
from src.core.mesh import MaterialID, Mesh3D
from src.core.refinement import Band, GridSpec


def bank() -> HeaterBank:
    return HeaterBank(rows=2, columns=2, power_per_element=1000.0, active_length=2.0,
                      sheath_diameter=0.012, leg_spacing=0.08)


def fine_mesh(target: float = 0.04, background: float = 0.12) -> Mesh3D:
    """Graded mesh refined around the element bank (3, 3) inside a 6 m box.

    The background is kept within a factor ~3 of the target: the growth limit means
    a very fine band inside a very coarse box can only be reached through a long
    ramp, which is exactly what the GUI summary warns about.
    """
    return Mesh3D(6.0, 6.0, 5.6, grid=GridSpec(
        x=(Band(0, 6, background), Band(1.6, 4.4, target)),
        y=(Band(0, 6, background), Band(1.6, 4.4, target)),
        z=(Band(0.5, 4.5, 0.06), Band(0, 5.6, background))))


def test_the_power_of_the_bank_is_deposited_exactly():
    """Whatever the grid, sum(Q_source V) must equal the rated power."""
    for mesh in (fine_mesh(), Mesh3D(6.0, 6.0, 5.6, spacing=0.1)):
        result = rasterize(bank(), mesh, 3.0, 3.0, 1.8, 0.5, 4.5)
        q = np.zeros(mesh.T.shape)
        q[result.active_mask] = result.power / float(mesh.V[result.active_mask].sum())
        assert float(np.sum(q * mesh.V)) == pytest.approx(4000.0, rel=1e-12)


def test_a_hairpin_occupies_two_legs_and_a_bend_inside_the_storage():
    """The rasterised element is a U: two vertical legs joined at the bottom."""
    mesh = fine_mesh(0.03)
    result = rasterize(bank(), mesh, 3.0, 3.0, 1.8, 0.5, 4.5)
    assert result.problem is None
    assert len(result.elements) == 4
    assert min(result.n_cells_per_element) > 0

    cells = np.argwhere(result.element_masks[0])
    columns = {(int(i), int(j)) for i, j, _ in cells}
    assert len(columns) >= 2, "the two legs must not collapse into one column"
    k_bottom = int(cells[:, 2].min())
    bend = cells[cells[:, 2] == k_bottom]
    assert len(bend) >= 2, "the bend must cover more than one cell"
    assert cells[:, 2].max() > k_bottom, "the legs must rise above the bend"
    # the bend sits on the storage floor, the legs rise through the storage band
    assert float(mesh.z[cells[cells[:, 2] == k_bottom, 2]].min()) >= 0.5 - 1e-9
    assert float(mesh.Z[result.active_mask].max()) <= 0.5 + 0.1 + 2.0 + mesh.size_z(2.6)


def test_only_the_active_length_carries_power():
    """The cold shank through the insulation is steel, never a heat source."""
    mesh = fine_mesh(0.03)
    result = rasterize(bank(), mesh, 3.0, 3.0, 1.8, 0.5, 4.5)
    assert int(result.active_mask.sum()) < int(result.mask.sum())
    assert not np.any(result.active_mask & ~result.mask)
    # the power stops at offset_bottom + active_length, within the last cell that
    # straddles that elevation
    z_active_top = float(mesh.Z[result.active_mask].max())
    assert z_active_top <= 0.5 + 0.1 + 2.0 + mesh.size_z(2.6)


def test_the_surface_power_is_the_rating_of_the_element():
    """P / (pi d L) drives the 3-8 W/cm2 window.

    The heated length is the whole hairpin: *two* legs of ``active_length - R``
    plus the semicircular bend of radius ``R = leg_spacing / 2``.  Counting one leg
    and half a bend halves the area and doubles the reported W/cm2.
    """
    element = HairpinElement(0.0, 0.0, leg_spacing=0.08, sheath_diameter=0.012,
                             active_length=2.0, rated_power=5000.0)
    bend = 0.5 * 0.08
    area = np.pi * 0.012 * (2.0 * (2.0 - bend) + np.pi * bend)
    assert element.surface_power_w_cm2 == pytest.approx(5000.0 / area / 1e4, rel=1e-12)
    assert element.surface_power_w_cm2 == pytest.approx(3.278, rel=1e-3)
    assert SURFACE_POWER_MIN_W_CM2 < element.surface_power_w_cm2 < SURFACE_POWER_LIMIT_W_CM2
    # with the two legs counted, 8 kW sits at 5.2 W/cm2: inside the window.  The
    # element that really exceeds it is the one above 12 kW.
    stressed = HairpinElement(0.0, 0.0, active_length=2.0, rated_power=13_000.0)
    assert stressed.surface_power_w_cm2 > SURFACE_POWER_LIMIT_W_CM2


def test_degenerate_layouts_are_rejected_with_a_reason():
    mesh = fine_mesh(0.03)
    close = HeaterBank(rows=2, columns=2, leg_spacing=0.015)          # legs touch
    problems = validate_bank(close, mesh, 3.0, 3.0, 1.8, 0.5, 4.5)
    assert any("share mesh cells" in p for p in problems)

    wide = HeaterBank(rows=2, columns=2, leg_spacing=0.08)
    outside = validate_bank(wide, mesh, 3.0, 3.0, 0.05, 0.5, 4.5)     # wall too small
    assert any("outside the storage wall" in p for p in outside)

    empty = HeaterBank(rows=0, columns=0)
    assert validate_bank(empty, mesh, 3.0, 3.0, 1.8, 0.5, 4.5)


def test_a_collision_with_a_heat_exchanger_tube_is_reported():
    class Tube:                                    # only what validate_bank uses
        x, y, radius = 3.08, 3.08, 0.05            # sitting on one element

    mesh = fine_mesh(0.03)
    problems = validate_bank(bank(), mesh, 3.0, 3.0, 1.8, 0.5, 4.5, tubes=[Tube()])
    assert any("collides with a heat-exchanger tube" in p for p in problems)


def test_the_surface_power_window_is_reported_as_a_warning():
    mesh = fine_mesh(0.03)
    weak = HeaterBank(rows=2, columns=2, power_per_element=100.0, active_length=2.0)
    problems = validate_bank(weak, mesh, 3.0, 3.0, 1.8, 0.5, 4.5)
    assert any(p.startswith("warning:") and "below" in p for p in problems)


def test_the_sheath_resolution_is_measured_on_the_mesh():
    """A 12 mm sheath needs ~6 mm cells: the diagnostic must say so."""
    coarse = rasterize(bank(), fine_mesh(0.2), 3.0, 3.0, 1.8, 0.5, 4.5)
    assert coarse.min_cells_across_sheath == 0
    assert any("thinner than a cell" in note for note in coarse.notes)
    # a small box is what makes a 5 mm target affordable (see the cell budget)
    resolved_mesh = Mesh3D(1.2, 1.2, 3.0, grid=GridSpec(
        x=(Band(0, 1.2, 0.02), Band(0.1, 1.1, 0.005)),
        y=(Band(0, 1.2, 0.02), Band(0.1, 1.1, 0.005)),
        z=(Band(0.5, 2.5, 0.05), Band(0, 3.0, 0.2)), max_cells=5_000_000))
    resolved = rasterize(bank(), resolved_mesh, 0.6, 0.6, 0.5, 0.5, 2.5)
    assert resolved.problem is None, resolved.problem
    assert resolved.min_cells_across_sheath >= 2


def test_the_geometry_builds_a_discrete_bank_and_reports_the_surface_power():
    """End to end through BatteryGeometry: mesh, sources and the reported rating."""
    geometry = BatteryGeometry(
        cylinder=CylinderGeometry(center_x=3.0, center_y=3.0, base_z=0.3, height=4.0,
                                  r_storage=2.0, insulation_thickness=0.3,
                                  shell_thickness=0.02, insulation_slab_bottom=0.2,
                                  insulation_slab_top=0.2, enable_cone_roof=False),
        tubes=TubeConfig(active=False),
        heaters=HeaterConfig(power_total=40.0, pattern=HeaterPattern.GRID_VERTICAL,
                             grid_rows=4, grid_cols=4, sheath_diameter=0.012,
                             leg_spacing=0.12, active_length=2.0))
    mesh = Mesh3D(6.0, 6.0, 5.6, grid=GridSpec(
        x=(Band(0, 6, 0.4), Band(1.4, 4.6, 0.04)),
        y=(Band(0, 6, 0.4), Band(1.4, 4.6, 0.04)),
        z=(Band(0.5, 4.3, 0.05), Band(0, 5.6, 0.4))))
    report = geometry.apply_to_mesh(mesh)
    assert report.n_heater_elements == 16
    assert float(np.sum(mesh.Q_source * mesh.V)) == pytest.approx(40_000.0, rel=1e-12)
    heater_cells = mesh.material_id == int(MaterialID.HEATERS)
    assert int(heater_cells.sum()) >= 16          # one element can never be one pixel
    assert np.all(mesh.Q_source[mesh.source_mask] > 0)
    assert np.all(heater_cells[mesh.source_mask]), \
        "power must sit on the sheath cells, not in the sand around them"
