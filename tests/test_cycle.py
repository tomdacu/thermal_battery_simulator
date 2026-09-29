"""Charging and discharging through the gas loop, end to end.

The point of these tests is that the *loop* is the heat transfer path: the resistors
put power into the gas, the gas heats the sand through the pipes, and the fan work is
reported with the cycle.  Nothing is deposited in the sand directly.

The second half drives whole cycles on a small silo: charge, standby, discharge, the
stopping criterion of each phase, and the energy decomposition that has to close.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest

from src.analysis.cycle import CycleSettings, run_cycle
from src.constants import T_AMBIENT_DEFAULT, T_MIN_VALID
from src.core.mesh import MaterialID, Mesh3D
from src.core.pipes import staggered_bank
from src.core.profiles import (ExtractionProfile, InitialCondition,
                               PowerProfile)
from src.solver.fluid import FluidLoop
from src.solver.steady import SolverConfig
from src.solver.transient import TransientConfig, TransientSolver

SOLVER_CONFIG = SolverConfig(method="cg", tolerance=1e-7)

# the small silo of the cycle tests: 3x3x3 m of sand, a bank of narrow tubes (the film
# coefficient of that loop is turbulent, so the gas really does follow the bed), and one
# leaking face, so the balance has a loss to carry
SILO_SIDE = 3.0          # [m]
SILO_SPACING = 0.5       # [m] cells
TUBE_DIAMETER = 0.02     # [m]
TUBE_PITCH = 0.3         # [m]
SILO_FLOW = 0.3          # [kg/s] air through the loop
SAND_K, SAND_RHO, SAND_CP = 0.5, 1500.0, 800.0


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
    # the loop deposits what the resistors inject (minus the fan, which is electric and
    # outside the thermal problem): the pipe film is the balance's input, and the lumped
    # bed source is empty because the gas is the only heat path
    from src.analysis.balance import compute_balance

    assert float(np.sum(mesh.Q_source * mesh.V)) == 0.0
    deposited = compute_balance(mesh, index=solver.index).p_input
    assert deposited == pytest.approx(loop.external_power, rel=0.02)
    # (the imbalance of a *transient* state is the storage rate, so it is not zero
    # here: it equals p_input while the bed is heating up)


def test_a_return_temperature_discharge_delivers_what_the_bed_loses():
    """The exchanger returns the gas at a set temperature: it gets what the bed gives.

    An adiabatic bed has nowhere else to put its heat, so the energy the exchanger
    received is the drop of the stored energy - on a long step too, where the march's
    own estimate (explicit in the wall temperature) would overstate a hot start.
    """
    from src.analysis.balance import compute_balance

    mesh, _, loop = bed_with_loop()
    config = TransientConfig(t_final=6 * 3600.0, dt=3600.0,
                             power_profile=PowerProfile(mode="off"),
                             extraction_profile=ExtractionProfile(
                                 mode="return_temperature", t_inlet=330.0),
                             fluid_loop=loop,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=600.0))
    solver = TransientSolver(mesh, config, SOLVER_CONFIG)
    solver.run()
    # the bed starts uniform, so its stored energy is its capacity times the rise
    stored_start = float(np.sum(mesh.rho * mesh.cp * mesh.V)) * (600.0 - T_AMBIENT_DEFAULT)
    stored_end = compute_balance(mesh, index=solver.index).e_stored
    assert solver.fluid_delivered > 0.0
    assert solver.fluid_delivered == pytest.approx(stored_start - stored_end, rel=1e-3)


def test_the_exchanger_delivers_its_power_while_the_resistors_run():
    """Charging and discharging at once: the exchanger still takes what it asked for."""
    mesh, _, loop = bed_with_loop()
    config = TransientConfig(t_final=1800.0, dt=600.0,
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=20_000.0),
                             extraction_profile=ExtractionProfile(mode="power",
                                                                  power=5_000.0),
                             fluid_loop=loop,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=500.0))
    solver = TransientSolver(mesh, config, SOLVER_CONFIG)
    solver.run()
    assert solver.fluid_delivered == pytest.approx(5_000.0 * 1800.0, rel=1e-9)


# --------------------------------------------------------------------- the cycle


def silo_bed(t_start: float = T_AMBIENT_DEFAULT, leak: float = 1.0,
             mass_flow: float = SILO_FLOW):
    """A 3x3x3 m sand silo with a tube bank, a leaking top face, and the ambient start.

    The bed starts at the ambient reference on purpose: nothing is stored yet, so the
    energy the cycle gains and the energy still in the bed when the discharge stops are
    the same number, and the identity can be read without the ledger of a previous
    cycle.  ``leak`` is the film coefficient [W/(m2 K)] of the top face against the
    ambient: the store is insulated, not sealed.
    """
    mesh = Mesh3D(SILO_SIDE, SILO_SIDE, SILO_SIDE, spacing=SILO_SPACING)
    mesh.k[:] = SAND_K
    mesh.rho[:] = SAND_RHO
    mesh.cp[:] = SAND_CP
    mesh.material_id[:] = int(MaterialID.SAND)
    mesh.T[:] = t_start
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min"):
        mesh.set_adiabatic(face)
    mesh.set_convection_bc("z_max", leak, T_AMBIENT_DEFAULT)
    layout = staggered_bank(mesh, center=(mesh.Lx / 2, mesh.Ly / 2),
                            radius=0.45 * mesh.Lx, z_bottom=0.0, z_top=mesh.Lz,
                            diameter=TUBE_DIAMETER, horizontal_pitch=TUBE_PITCH,
                            vertical_pitch=TUBE_PITCH)
    assert layout.n_pipes > 20
    loop = FluidLoop(runs=layout.runs, mass_flow=mass_flow, fittings_k=8.0)
    return mesh, loop


def cycle_settings(**overrides) -> CycleSettings:
    """The cycle of the small silo: 60 kW of charge, 12 h of standby, 30 kW out."""
    settings = dict(charge_power=60_000.0, t_target=500.0, charge_limit=4.0 * 86400.0,
                    standby_time=12.0 * 3600.0, discharge_power=30_000.0,
                    t_delivery_min=353.15, dt=1800.0, chunk=24.0 * 3600.0)
    settings.update(overrides)
    return CycleSettings(**settings)


def test_the_cycle_closes_the_energy_identity_and_prints_its_numbers():
    """Charge, sit, discharge on a small silo: the identity has to close.

    The cycle starts at the ambient reference and runs to the delivery floor, so every
    term of the identity is a real number: the charge heats the bed through the gas,
    the leak carries a loss, the blower costs electricity, and the tail of the store is
    left behind when the heat stops being usable.
    """
    mesh, loop = silo_bed()
    report = run_cycle(mesh, loop, cycle_settings(), SOLVER_CONFIG)

    # the numbers of the cycle: this is the evidence a report is built from
    print(report.summary())

    assert report.charge.seconds > 0.0
    assert report.standby.seconds == pytest.approx(12.0 * 3600.0)
    assert report.discharge.seconds > 0.0
    # every term of the identity is there and positive on a real cycle
    assert report.energy_delivered > 0.0
    assert report.fan_energy > 0.0
    assert report.energy_loss > 0.0
    assert report.unrecovered > 0.0

    # E_in = dE_stored + delivered + losses + circulation, within the 1% the module
    # promises (the residual is the sum of the solver's own per-step imbalances)
    carried = (report.dE_stored + report.energy_delivered + report.energy_loss
               + report.fan_energy)
    assert report.energy_electric == pytest.approx(carried, rel=0.01)
    assert report.balance_residual() < 0.01

    # the loop is the only path: the electricity of the resistors lands in the bed,
    # and the exchanger hands the user exactly the power it was asked for
    assert report.charge.energy_in == pytest.approx(
        60_000.0 * report.charge.seconds, rel=1e-6)
    assert report.discharge.energy_out == pytest.approx(
        30_000.0 * report.discharge.seconds, rel=1e-6)

    # nothing was stored at the start, so what the cycle gained is what the discharge
    # could not reach
    assert report.unrecovered == pytest.approx(report.dE_stored, rel=1e-9)

    # the field never left the Kelvin contract on the way down
    assert float(mesh.T.min()) > T_MIN_VALID


def test_the_discharge_stops_at_the_delivery_floor_inside_a_chunk():
    """The stop comes from the physics, not from the inspection boundary.

    Three discharges from the same charged state: two inspection intervals of very
    different length, and one finer time step.  The first two must end at the same
    second - the chunk is only where the driver looks - and the fine one must end no
    later than one coarse step after them, since the stop is resolved per step.
    """
    mesh, loop = silo_bed()
    settings = cycle_settings(t_target=450.0, t_delivery_min=400.0, standby_time=0.0,
                              discharge_power=30_000.0)
    # the call charges ``mesh`` in place: the discharges below copy that state
    run_cycle(mesh, loop, replace(settings, discharge_power=0.0), SOLVER_CONFIG)

    def discharge(chunk: float, dt: float):
        bed = copy.deepcopy(mesh)       # the same charged state, another discrete run
        probe = replace(settings, chunk=chunk, dt=dt)
        return run_cycle(bed, loop, probe, SOLVER_CONFIG).discharge

    narrow = discharge(2.0 * 3600.0, 1800.0)
    wide = discharge(24.0 * 3600.0, 1800.0)
    assert narrow.seconds == pytest.approx(wide.seconds, abs=1e-9)
    assert wide.seconds < 24.0 * 3600.0              # shorter than one inspection window
    assert narrow.seconds % (2.0 * 3600.0) != 0.0    # and not on the edge of the narrow one
    assert narrow.t_delivery_end <= settings.t_delivery_min
    assert narrow.t_delivery_end > settings.t_delivery_min - 5.0

    fine = discharge(2.0 * 3600.0, 900.0)
    assert 0.0 <= narrow.seconds - fine.seconds <= 1800.0


def test_the_charge_stops_at_the_target_inside_a_chunk():
    """The charge stops on the target too, one step past it at most."""
    mesh, loop = silo_bed()
    settings = cycle_settings(t_target=400.0, standby_time=0.0, discharge_power=0.0)
    phase = run_cycle(mesh, loop, settings, SOLVER_CONFIG).charge

    assert phase.seconds < settings.chunk                  # inside the inspection window
    assert phase.t_end >= settings.t_target
    # one step of the charge cannot lift the mean by more than it stores
    capacity = float(np.sum(mesh.rho * mesh.cp * mesh.V))  # [J/K]
    assert phase.t_end - settings.t_target < settings.charge_power * settings.dt / capacity
    assert phase.notes and "the target" in phase.notes[0]


def test_a_zero_discharge_power_skips_the_phase_with_a_note():
    mesh, loop = silo_bed()
    settings = cycle_settings(charge_limit=2.0 * 3600.0, standby_time=0.0,
                              discharge_power=0.0, chunk=3600.0)
    report = run_cycle(mesh, loop, settings, SOLVER_CONFIG)

    assert report.discharge.seconds == 0.0
    assert report.discharge.energy_out == 0.0
    assert report.discharge.fan_energy == 0.0
    assert report.discharge.notes == ["no discharge power requested: the phase is skipped"]
    # nothing was taken out, so the cycle is a charge and the identity has three terms
    assert report.energy_delivered == 0.0
    assert report.energy_electric == pytest.approx(
        report.dE_stored + report.energy_loss + report.fan_energy, rel=0.01)


def test_a_discharge_past_the_contract_stops_with_a_note():
    """A pull harder than the store can supply ends the phase; it does not raise.

    The floor of this probe (220 K) is below anything the gas can reach without
    chilling the cold end of the bed under the Kelvin contract.  The criterion has to
    end the phase and say so: the field check of the solver, which raises mid-chunk,
    must never be the one that finds out.
    """
    mesh, loop = silo_bed()
    settings = cycle_settings(t_target=450.0, discharge_power=60_000.0,
                              t_delivery_min=220.0, chunk=1800.0)
    report = run_cycle(mesh, loop, settings, SOLVER_CONFIG)

    assert report.discharge.seconds > 0.0
    assert report.discharge.seconds < settings.max_chunks * settings.chunk
    assert report.discharge.notes and "contract" in report.discharge.notes[0]
    assert float(mesh.T.min()) < T_MIN_VALID      # the pull really was too hard
    assert report.balance_residual() < 0.01       # and the accounting still closes


def test_the_fan_is_counted_once_per_phase():
    """The blower is an electric load: counted once per phase, and never twice."""
    mesh, loop = silo_bed()
    fan_power = loop.solve(mesh).fan_power      # fixed hardware: the same watts always
    assert fan_power > 0.0

    settings = cycle_settings(t_target=450.0, t_delivery_min=440.0)
    report = run_cycle(mesh, loop, settings, SOLVER_CONFIG)

    for phase in report.phases:
        assert phase.seconds > 0.0
        assert phase.fan_energy == pytest.approx(fan_power * phase.seconds, rel=1e-6)
    seconds = sum(phase.seconds for phase in report.phases)
    assert report.fan_energy == pytest.approx(fan_power * seconds, rel=1e-6)
    # and it is one term of the identity, not a term hidden inside another one
    assert report.energy_electric == pytest.approx(
        report.energy_in + report.fan_energy, rel=1e-12)
    assert report.balance_residual() < 0.01


def test_the_field_stays_inside_the_kelvin_contract_on_a_hard_discharge():
    """A small silo discharged hard: the field still stays inside the contract.

    The criterion is asked after every step here (``chunk`` = ``dt``), so the field is
    inspected at every state the solver produced, not only at the phase boundary.
    """
    mesh, loop = silo_bed()
    settings = cycle_settings(t_target=450.0, discharge_power=60_000.0,
                              t_delivery_min=320.0, chunk=1800.0)
    minima: list[float] = []
    report = run_cycle(mesh, loop, settings, SOLVER_CONFIG,
                       progress=lambda _mean: minima.append(float(mesh.T.min())))

    assert report.discharge.seconds > 0.0
    assert report.discharge.t_delivery_end <= settings.t_delivery_min
    assert min(minima) > T_MIN_VALID
    assert float(mesh.T.min()) > T_MIN_VALID
    # the store really was driven down: this is a hard discharge, not a gentle one
    assert float(mesh.T.mean()) < 400.0
