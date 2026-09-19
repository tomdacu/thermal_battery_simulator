"""The fluid loop: effectiveness relation, conservation, and the two regimes.

The point of these tests is the physics the lumped film model cannot represent: a
storage discharges with a large NTU, where the extraction is limited by the *flow*
(``m_dot c_p dT``) and not by the area (``h A dT``).
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.mesh import Mesh3D
from src.core.pipes import rasterize_pipe, vertical_bank
from src.solver.fluid import Fluid, FluidLoop, pipe_h


def uniform_bed(temperature: float = 500.0, spacing: float = 0.5,
                cells: int = 4) -> Mesh3D:
    mesh = Mesh3D(cells * spacing, cells * spacing, cells * spacing, spacing=spacing)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    mesh.T[:] = temperature
    return mesh


def vertical_run(mesh: Mesh3D, diameter: float = 0.05):
    return rasterize_pipe(mesh, [(mesh.Lx / 2, mesh.Ly / 2, 0.0),
                                 (mesh.Lx / 2, mesh.Ly / 2, mesh.Lz)],
                          diameter, name="riser")


# ------------------------------------------------------------------ geometry
def test_the_rasterised_area_is_the_geometric_one():
    """The wetted area must come from pi d L, not from the voxel mask."""
    mesh = uniform_bed()
    run = vertical_run(mesh)
    assert run.total_length == pytest.approx(mesh.Lz, rel=1e-9)
    assert run.total_area == pytest.approx(np.pi * 0.05 * mesh.Lz, rel=1e-9)
    assert float(np.sum(run.length)) == pytest.approx(run.total_length, rel=1e-12)


def test_a_pipe_outside_the_domain_keeps_nothing():
    mesh = uniform_bed()
    run = rasterize_pipe(mesh, [(99.0, 99.0, 0.0), (99.0, 99.0, 1.0)], 0.05)
    assert run.cells.size == 0


def test_a_bank_splits_the_same_total_area_between_the_pipes():
    mesh = uniform_bed()
    bank = vertical_bank(mesh, center=(mesh.Lx / 2, mesh.Ly / 2), radius=0.8,
                         n_pipes=4, z_bottom=0.0, z_top=mesh.Lz, diameter=0.05)
    assert len(bank) == 4
    total = sum(run.total_area for run in bank)
    assert total == pytest.approx(4 * np.pi * 0.05 * mesh.Lz, rel=1e-9)


# --------------------------------------------------------------------- film
def test_the_film_coefficient_follows_the_correlations():
    fluid = Fluid()
    # laminar: Nu = 3.66 -> h = 3.66 k / d
    laminar = pipe_h(1e-6, 0.05, fluid)
    assert laminar == pytest.approx(3.66 * fluid.k / 0.05, rel=1e-9)
    # turbulent: Dittus-Boelter with the Reynolds number of that flow
    mass_flow = 0.2
    area = 0.25 * np.pi * 0.05 ** 2
    velocity = mass_flow / (fluid.rho * area)
    re = fluid.rho * velocity * 0.05 / fluid.mu
    expected = 0.023 * re ** 0.8 * fluid.pr ** 0.4 * fluid.k / 0.05
    assert pipe_h(mass_flow, 0.05, fluid) == pytest.approx(expected, rel=1e-9)
    assert re > 4000.0                     # the case above is really turbulent


# --------------------------------------------------------------- behaviour
def test_a_uniform_wall_gives_the_effectiveness_relation():
    mesh = uniform_bed()
    run = vertical_run(mesh)
    mass_flow, h, t_in = 0.05, 500.0, 300.0
    result = FluidLoop(runs=[run], mass_flow=mass_flow, h_fluid=h, t_in=t_in).solve(mesh)
    ntu = h * run.total_area / (mass_flow * 1005.0)
    expected = 500.0 - (500.0 - 300.0) * np.exp(-ntu)
    assert result.runs[0].ntu == pytest.approx(ntu, rel=1e-9)
    assert result.runs[0].t_out == pytest.approx(expected, rel=1e-9)


def test_the_power_leaving_the_fluid_enters_the_solid():
    """Conservation: the enthalpy drop of the air is exactly what the bed receives."""
    mesh = uniform_bed()
    run = vertical_run(mesh)
    result = FluidLoop(runs=[run], mass_flow=0.05, h_fluid=500.0, t_in=300.0).solve(mesh)
    # the bed is hotter than the incoming air: it *loses* heat, so the source term
    # for the solid is negative and its magnitude is the enthalpy the air gained
    enthalpy = 0.05 * 1005.0 * (result.t_out - result.t_in)
    assert result.power == pytest.approx(-enthalpy, rel=1e-9)
    integrated = float(np.sum(result.q_fluid * mesh.V.ravel(order="F")))
    assert integrated == pytest.approx(-enthalpy, rel=1e-9)


def test_a_large_flow_is_area_limited():
    mesh = uniform_bed()
    run = vertical_run(mesh)
    h, t_in = 500.0, 300.0
    result = FluidLoop(runs=[run], mass_flow=50.0, h_fluid=h, t_in=t_in).solve(mesh)
    assert result.runs[0].ntu < 0.01
    # at NTU = 0.008 the finite-flow correction is NTU/2 = 0.4%: the area limit is
    # reached to within half a percent
    assert abs(result.power) == pytest.approx(h * run.total_area * (500.0 - t_in),
                                              rel=5e-3)


def test_a_small_flow_is_flow_limited():
    """The regime a storage actually discharges in."""
    mesh = uniform_bed()
    run = vertical_run(mesh)
    mass_flow, t_in = 0.002, 300.0
    result = FluidLoop(runs=[run], mass_flow=mass_flow, h_fluid=500.0, t_in=t_in).solve(mesh)
    assert result.runs[0].ntu > 20.0
    assert result.runs[0].t_out == pytest.approx(500.0, rel=1e-6)     # leaves at the bed
    assert abs(result.power) == pytest.approx(mass_flow * 1005.0 * (500.0 - t_in),
                                              rel=1e-6)


def test_a_lumped_model_would_overstate_the_extraction():
    """The number that motivated the 1-D model: NTU/(1 - exp(-NTU))."""
    mesh = uniform_bed()
    run = vertical_run(mesh)
    mass_flow, h, t_in = 0.02, 500.0, 300.0
    result = FluidLoop(runs=[run], mass_flow=mass_flow, h_fluid=h, t_in=t_in).solve(mesh)
    ntu = result.runs[0].ntu
    lumped = h * run.total_area * (500.0 - t_in)
    assert ntu > 3.0
    assert lumped / abs(result.power) == pytest.approx(ntu / (1 - np.exp(-ntu)),
                                                       rel=1e-6)


def test_a_closed_loop_without_external_power_settles_at_the_wall_temperature():
    mesh = uniform_bed(temperature=480.0)
    run = vertical_run(mesh)
    result = FluidLoop(runs=[run], mass_flow=0.05, h_fluid=500.0,
                       external_power=0.0).solve(mesh)
    assert result.t_in == pytest.approx(480.0, rel=1e-9)
    assert result.t_out == pytest.approx(480.0, rel=1e-9)
    assert result.power == pytest.approx(0.0, abs=1e-9)


def test_the_resistors_charge_the_bed_and_the_exchanger_discharges_it():
    mesh = uniform_bed(temperature=500.0)
    run = vertical_run(mesh)
    charge = FluidLoop(runs=[run], mass_flow=0.05, h_fluid=500.0,
                       external_power=+10_000.0).solve(mesh)
    discharge = FluidLoop(runs=[run], mass_flow=0.05, h_fluid=500.0,
                          external_power=-10_000.0).solve(mesh)
    assert charge.power > 0.0 and charge.t_in > 500.0
    assert discharge.power < 0.0 and discharge.t_in < 500.0
    # the loop balance closes in both directions
    assert charge.power == pytest.approx(charge.external_power, rel=1e-9)
    assert discharge.power == pytest.approx(discharge.external_power, rel=1e-9)


def test_the_bank_splits_the_flow_and_keeps_the_total_power():
    mesh = uniform_bed()
    bank = vertical_bank(mesh, center=(mesh.Lx / 2, mesh.Ly / 2), radius=0.8,
                         n_pipes=4, z_bottom=0.0, z_top=mesh.Lz, diameter=0.05)
    result = FluidLoop(runs=bank, mass_flow=0.08, h_fluid=500.0, t_in=300.0).solve(mesh)
    assert len(result.runs) == 4
    assert all(run.mass_flow == pytest.approx(0.02) for run in result.runs)
    enthalpy = 0.08 * 1005.0 * (result.t_out - result.t_in)
    assert result.power == pytest.approx(-enthalpy, rel=1e-9)


def test_an_impossible_operating_point_is_refused():
    """A flow that cannot carry the requested power must say so, not return T < 0."""
    mesh = uniform_bed(temperature=500.0)
    run = vertical_run(mesh)
    with pytest.raises(ValueError, match="cannot carry"):
        FluidLoop(runs=[run], mass_flow=0.005, h_fluid=500.0,
                  external_power=-60_000.0).solve(mesh)


# ------------------------------------------------------------------ hydraulics
def test_the_friction_factor_matches_the_classical_values():
    from src.solver.fluid import friction_factor
    # laminar: f = 64/Re
    assert friction_factor(1000.0) == pytest.approx(0.064, rel=1e-12)
    # smooth pipe, Re = 1e5: Haaland gives 0.0181 (Colebrook: 0.0180)
    assert friction_factor(1e5) == pytest.approx(0.0181, rel=2e-2)
    # rough pipe: more friction than a smooth one at the same Reynolds
    assert friction_factor(1e5, 0.01) > friction_factor(1e5, 0.0)


def test_the_pressure_drop_scales_with_the_square_of_the_flow():
    from src.solver.fluid import pressure_drop
    fluid = Fluid()
    one = pressure_drop(0.05, 0.05, 4.0, fluid)
    two = pressure_drop(0.10, 0.05, 4.0, fluid)
    assert one > 0.0
    # turbulent: dp = (f L/D + K) rho u^2 / 2 with f falling slowly with Re, so the
    # exponent sits between 1.8 (rough, f ~ Re^-0.2) and 2 (f constant)
    assert 1.7 < float(np.log(two / one) / np.log(2.0)) < 2.0
    # local losses add up
    assert pressure_drop(0.05, 0.05, 4.0, fluid, fittings_k=5.0) > one


def test_pressure_is_the_cheap_lever_of_a_gas_loop():
    """What pressure buys, stated exactly.

    At a *fixed mass flow* the Reynolds number is independent of the density, so the
    film coefficient does not move - but the velocity falls as 1/rho, so the pressure
    drop (and the fan power) falls.  At a fixed *velocity* the mass flow and the
    Reynolds number rise with rho, so the film coefficient rises.  That is why a
    pressurised loop reaches the same heat transfer with less fan work.
    """
    from src.solver.fluid import pipe_h, pressure_drop
    cold = Fluid().at(500.0)
    dense = Fluid().at_pressure(20.0 * 101325.0, 500.0)
    assert dense.rho == pytest.approx(cold.rho * 20.0, rel=1e-9)

    # fixed mass flow: same h, much smaller pressure drop
    assert pipe_h(0.05, 0.05, dense) == pytest.approx(pipe_h(0.05, 0.05, cold), rel=1e-9)
    assert pressure_drop(0.05, 0.05, 4.0, dense) < 0.1 * pressure_drop(0.05, 0.05, 4.0, cold)

    # fixed velocity: same pressure drop per unit length, much larger h
    assert pipe_h(0.05 * 20.0, 0.05, dense) > 5.0 * pipe_h(0.05, 0.05, cold)


def test_the_loop_reports_its_circulation_loss():
    mesh = uniform_bed()
    run = vertical_run(mesh)
    result = FluidLoop(runs=[run], mass_flow=0.05, h_fluid=500.0, t_in=300.0,
                       fittings_k=10.0).solve(mesh)
    assert result.delta_p > 0.0
    assert result.fan_power > 0.0
    assert 0.0 < result.circulation_loss < 1.0
    assert "fan" in result.summary()


def test_the_circulation_loss_is_reported_against_both_denominators():
    """The published ~5% is an aggregate; the blower alone is 1-2% of the plant power."""
    mesh = uniform_bed()
    run = vertical_run(mesh)
    result = FluidLoop(runs=[run], mass_flow=0.05, h_fluid=500.0, t_in=300.0,
                       external_power=-2000.0, fittings_k=10.0).solve(mesh)
    assert result.circulation_loss > 0.0
    assert result.circulation_loss_electric == pytest.approx(
        result.fan_power / abs(result.external_power), rel=1e-12)
    assert result.circulation_loss == pytest.approx(
        result.fan_power / abs(result.power), rel=1e-12)


def test_a_pressurised_loop_uses_the_density_of_its_pressure():
    mesh = uniform_bed()
    run = vertical_run(mesh)
    common = dict(runs=[run], mass_flow=0.05, h_fluid=500.0, t_in=300.0,
                  external_power=-2000.0)
    ambient = FluidLoop(**common).solve(mesh)
    dense = FluidLoop(pressure=20.0 * 101325.0, **common).solve(mesh)
    assert dense.delta_p < ambient.delta_p           # lower velocity, less fan work
    assert dense.fan_power < ambient.fan_power


def test_a_circuit_that_drops_its_own_pressure_is_flagged():
    """Above dp/p = 10% the incompressible treatment must be declared invalid."""
    mesh = uniform_bed()
    run = vertical_run(mesh)
    result = FluidLoop(runs=[run], mass_flow=0.5, h_fluid=500.0, t_in=300.0,
                       external_power=-2000.0, fittings_k=200.0,
                       pressure=101325.0).solve(mesh)
    assert result.delta_p > 0.1 * result.pressure
    assert any("incompressible" in note for note in result.notes)


# ------------------------------------------------------------------- layouts
def test_the_staggered_bank_follows_the_published_pitches():
    from src.core.pipes import staggered_bank
    mesh = uniform_bed(spacing=0.25, cells=16)          # 4 m box
    layout = staggered_bank(mesh, center=(2.0, 2.0), radius=1.5, z_bottom=0.0,
                            z_top=2.0, diameter=0.025)
    assert layout.n_pipes > 100
    assert layout.horizontal_pitch == pytest.approx(3 ** 0.5 * 0.025, rel=1e-12)
    assert layout.vertical_pitch == pytest.approx(3 ** 0.5 * 0.025, rel=1e-12)
    # every pipe really sits inside the circle, at the right height
    for run in layout.runs:
        x, y = run.points[0, 0] - 2.0, run.points[0, 1] - 2.0
        assert x ** 2 + y ** 2 <= 1.5 ** 2 + 1e-9
        assert run.points[0, 2] == 0.0 and run.points[-1, 2] == 2.0
    # the specific area is the sizing number: pi d L n / V
    expected = np.pi * 0.025 * 2.0 * layout.n_pipes / (np.pi * 1.5 ** 2 * 2.0)
    assert layout.specific_area == pytest.approx(expected, rel=1e-9)


def test_a_tall_bundle_is_flagged_for_the_header_limit():
    from src.core.pipes import HEADER_LIMIT, staggered_bank
    mesh = uniform_bed(spacing=0.5, cells=12)
    layout = staggered_bank(mesh, center=(3.0, 3.0), radius=2.0, z_bottom=0.0,
                            z_top=4.0, diameter=0.05, vertical_pitch=0.5)
    assert layout.bundle_height > HEADER_LIMIT
    assert any("split into parallel modules" in note for note in layout.notes)


def test_a_pitch_smaller_than_the_tube_is_refused():
    from src.core.pipes import staggered_bank
    mesh = uniform_bed()
    with pytest.raises(ValueError, match="pitch"):
        staggered_bank(mesh, center=(1.0, 1.0), radius=1.0, z_bottom=0.0, z_top=1.0,
                       diameter=0.05, horizontal_pitch=0.02)
