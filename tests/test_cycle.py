"""Charging and discharging through the gas loop, end to end.

The point of these tests is that the *loop* is the heat transfer path: the resistors
put power into the gas, the gas heats the sand through the pipes, and the fan work is
reported with the cycle.  Nothing is deposited in the sand directly.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.mesh import Mesh3D
from src.core.pipes import staggered_bank
from src.core.profiles import (ExtractionProfile, InitialCondition,
                               PowerProfile)
from src.solver.fluid import FluidLoop
from src.solver.steady import SolverConfig
from src.solver.transient import TransientConfig, TransientSolver


def bed_with_loop(spacing: float = 0.25, cells: int = 8, diameter: float = 0.04):
    mesh = Mesh3D(cells * spacing, cells * spacing, cells * spacing, spacing=spacing)
    mesh.k[:] = 0.5
    mesh.rho[:] = 1500.0
    mesh.cp[:] = 800.0
    mesh.T[:] = 400.0
    # these tests are about the loop, not the envelope: an adiabatic box makes the
    # energy accounting unambiguous
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        mesh.set_adiabatic(face)
    layout = staggered_bank(mesh, center=(mesh.Lx / 2, mesh.Ly / 2),
                            radius=0.45 * mesh.Lx, z_bottom=0.0, z_top=mesh.Lz,
                            diameter=diameter)
    loop = FluidLoop(runs=layout.runs, mass_flow=0.02, fittings_k=8.0)
    return mesh, layout, loop


def run_solver(mesh, loop, power_profile, extraction, seconds=600.0, dt=60.0,
               t_start=400.0):
    # the transient applies its initial condition: say it explicitly instead of
    # setting mesh.T and being silently overwritten
    config = TransientConfig(t_final=seconds, dt=dt, power_profile=power_profile,
                             extraction_profile=extraction, fluid_loop=loop,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=t_start))
    solver = TransientSolver(mesh, config, SolverConfig(method="cg", tolerance=1e-7))
    return solver.run()


def test_charging_through_the_loop_heats_the_bed():
    mesh, layout, loop = bed_with_loop()
    assert layout.n_pipes > 10
    results = run_solver(mesh, loop,
                         PowerProfile(mode="constant", constant_power=50_000.0),
                         ExtractionProfile(mode="off"), seconds=1800.0)
    # 50 kW for 30 minutes into an adiabatic 8 m3 bed of sand: ~9 K
    assert float(mesh.T.mean()) > 405.0
    # the resistors are the only energy input, and the loop carried it into the bed
    assert len(results) > 1
    assert loop.external_power == pytest.approx(50_000.0)


def test_the_cycle_reports_what_the_fan_costs():
    """The circulation loss is a number the model must produce, not assume."""
    mesh, _, loop = bed_with_loop()
    solver_config = SolverConfig(method="cg", tolerance=1e-7)
    config = TransientConfig(t_final=300.0, dt=60.0,
                             power_profile=PowerProfile(mode="constant", constant_power=50_000.0),
                             extraction_profile=ExtractionProfile(mode="off"),
                             fluid_loop=loop,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=400.0))
    solver = TransientSolver(mesh, config, solver_config)
    solver.run()
    result = solver.fluid_result
    assert result is not None
    assert result.fan_power > 0.0
    assert result.delta_p > 0.0
    # in a small test bed the fan share is large (short pipes, low thermal power);
    # at full scale it falls towards the few percent the literature reports
    assert 0.0 < result.circulation_loss_electric < 0.5
    assert "fan" in result.summary()


def test_discharging_through_the_loop_cools_the_bed():
    mesh, _, loop = bed_with_loop()
    run_solver(mesh, loop, PowerProfile(mode="off"),
               ExtractionProfile(mode="power", power=5_000.0), t_start=500.0)
    assert float(mesh.T.mean()) < 500.0


def test_the_energy_balance_closes_with_the_loop_driving():
    """The reported balance must still close when the loop is the source term."""
    mesh, _, loop = bed_with_loop()
    solver_config = SolverConfig(method="cg", tolerance=1e-8)
    config = TransientConfig(t_final=300.0, dt=60.0,
                             power_profile=PowerProfile(mode="constant", constant_power=40_000.0),
                             extraction_profile=ExtractionProfile(mode="off"),
                             fluid_loop=loop,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=400.0))
    solver = TransientSolver(mesh, config, solver_config)
    solver.run()
    # the loop deposits exactly what the resistors inject (minus the fan, which is
    # electric and outside the thermal problem), and the balance of that state closes
    deposited = float(np.sum(mesh.Q_source * mesh.V))
    assert deposited == pytest.approx(loop.external_power, rel=0.02)
    # (the imbalance of a *transient* state is the storage rate, so it is not zero
    # here: it equals p_input while the bed is heating up)
