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
