"""The operating cycle: charge, standby, discharge, and where the energy goes.

A storage is judged by a cycle, not by a steady state.  This module drives the three
phases with the transient solver and accounts for the energy in the only way that
cannot hide a term:

    E_in(electric) = dE_stored + E_delivered + E_standby + E_circulation + E_unrecovered

where

* ``E_in`` is what the resistors take from the grid (including the blower, which is an
  electric load of the same system);
* ``E_stored`` is the sensible energy above the ambient, from the same balance the
  solver uses;
* ``E_delivered`` is what the exchanger hands to the user during the discharge;
* ``E_standby`` is what leaks through the envelope while the store sits;
* ``E_circulation`` is the blower work;
* ``E_unrecovered`` is the energy still inside the bed when the discharge stops because
  the delivery temperature fell below what the user can use - the number that decides
  whether a low-temperature tail is worth chasing.

The three phases are driven by the *gas loop*, so the resistors heat the gas and the
gas heats the bed; nothing is deposited in the sand directly.

**Status: work in progress.**  The phases are inspected at chunk boundaries, so a
discharge fast compared with the bed capacity can overshoot the delivery floor inside a
chunk (the driver inspects every 30 minutes for that reason); a per-step stopping
criterion inside the transient solver is the clean fix and is not implemented yet.
The charge and standby phases run and are accounted for; the discharge accounting is
complete but has not yet been exercised on a full cycle.

Rounding of the phases: the charge runs until the storage reaches the target
temperature, the standby for a prescribed time, the discharge until the delivery
temperature drops below its floor.  Every phase runs in chunks so the driver can watch
the state and stop where the physics says so.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import MaterialID, Mesh3D
from ..core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from ..solver.fluid import FluidLoop
from ..solver.steady import SolverConfig
from ..solver.transient import TransientConfig, TransientSolver
from .balance import compute_balance


@dataclass
class CycleSettings:
    """How the cycle is run (all SI, temperatures in K)."""

    charge_power: float = 100_000.0     # [W] electric power of the resistors
    t_target: float = 773.15            # [K] storage mean at the end of the charge
    charge_limit: float = 30.0 * 86400.0   # [s] give up after this
    standby_time: float = 7.0 * 86400.0    # [s] how long the store sits
    discharge_power: float = 50_000.0      # [W] requested at the exchanger
    t_delivery_min: float = 353.15         # [K] below this the heat is not usable
    dt: float = 1800.0                     # [s] time step of the transient
    chunk: float = 12.0 * 3600.0           # [s] how often the driver inspects the state
    max_chunks: int = 240


@dataclass
class CyclePhase:
    """One phase of the cycle and what it cost or delivered."""

    name: str
    seconds: float = 0.0
    energy_in: float = 0.0          # [J] electric
    energy_out: float = 0.0         # [J] delivered to the user
    fan_energy: float = 0.0         # [J]
    stored_start: float = 0.0       # [J]
    stored_end: float = 0.0         # [J]
    t_start: float = float("nan")   # [K] storage mean
    t_end: float = float("nan")
    notes: list[str] = field(default_factory=list)

    @property
    def stored_change(self) -> float:
        return self.stored_end - self.stored_start


@dataclass
class CycleReport:
    """The whole cycle: the three phases and the efficiency decomposition."""

    charge: CyclePhase = field(default_factory=lambda: CyclePhase("charge"))
    standby: CyclePhase = field(default_factory=lambda: CyclePhase("standby"))
    discharge: CyclePhase = field(default_factory=lambda: CyclePhase("discharge"))

    @property
    def energy_in(self) -> float:
        return self.charge.energy_in

    @property
    def energy_delivered(self) -> float:
        return self.discharge.energy_out

    @property
    def circulation_loss(self) -> float:
        """Blower work over the electric energy taken in [-].

        The published circulation figure of a storage cycle is an aggregate (blower,
        controls, hot ducts); this is the blower share alone, computed from the
        pressure drop of the actual pipe network.
        """
        total = self.charge.fan_energy + self.standby.fan_energy + self.discharge.fan_energy
        return total / self.energy_in if self.energy_in > 0 else 0.0

    @property
    def standby_loss(self) -> float:
        """Energy lost while the store sat, over the electric energy taken in [-]."""
        lost = self.standby.stored_change
        return -lost / self.energy_in if self.energy_in > 0 and lost < 0 else 0.0

    @property
    def unrecovered(self) -> float:
        """Energy still in the bed when the discharge had to stop [J]."""
        return self.discharge.stored_end

    @property
    def round_trip(self) -> float:
        """Delivered heat over electric energy taken in [-]."""
        return self.energy_delivered / self.energy_in if self.energy_in > 0 else 0.0

    def summary(self) -> str:
        lines = [
            f"charge    : {self.charge.seconds / 3600:.1f} h, "
            f"in {self.charge.energy_in / 3.6e9:.2f} MWh, "
            f"storage {self.charge.t_start:.1f} -> {self.charge.t_end:.1f} K, "
            f"fan {self.charge.fan_energy / 3.6e6:.1f} kWh",
            f"standby   : {self.standby.seconds / 86400:.1f} d, "
            f"loss {self.standby.stored_change / 3.6e9:+.3f} MWh "
            f"({-self.standby.stored_change / max(self.standby.seconds, 1.0) * 86400 / 1e6:.1f}"
            f" MJ/day)",
            f"discharge : {self.discharge.seconds / 3600:.1f} h, "
            f"delivered {self.discharge.energy_out / 3.6e9:.2f} MWh, "
            f"storage {self.discharge.t_start:.1f} -> {self.discharge.t_end:.1f} K, "
            f"fan {self.discharge.fan_energy / 3.6e6:.1f} kWh",
            f"round trip: {100 * self.round_trip:.1f}% delivered / electric, "
            f"circulation {100 * self.circulation_loss:.1f}%, "
            f"standby {100 * self.standby_loss:.1f}%, "
            f"still stored at the end {self.unrecovered / 3.6e9:.2f} MWh",
        ]
        for phase in (self.charge, self.standby, self.discharge):
            lines.extend(f"  [{phase.name}] {note}" for note in phase.notes)
        return "\n".join(lines)


def _storage_mean(mesh: Mesh3D) -> float:
    """Volume-averaged temperature of the storage material [K]."""
    mask = mesh.material_id == int(MaterialID.SAND)
    if not mask.any():
        mask = mesh.material_id != int(MaterialID.AIR)
    return float(np.mean(mesh.T[mask]))


def _phase_chunk(mesh: Mesh3D, loop: FluidLoop, solver_config: SolverConfig,
                 power: float, extraction: float, seconds: float, dt: float,
                 progress=None) -> tuple[float, float]:
    """Run one chunk; returns (fan energy [J], delivered energy [J])."""
    config = TransientConfig(
        t_final=seconds, dt=dt,
        power_profile=PowerProfile(mode="constant", constant_power=power),
        extraction_profile=ExtractionProfile(mode="power", power=extraction),
        fluid_loop=loop,
        # keep the field the previous chunk left: the cycle is continuous
        initial_condition=InitialCondition(mode="keep"),
    )
    solver = TransientSolver(mesh, config, solver_config)
    solver.run()
    if not np.all(np.isfinite(mesh.T)):
        raise RuntimeError(
            f"the '{power:+.0f} W / {extraction:+.0f} W' chunk diverged: the field "
            f"left the physical range (check the loop set point and the time step)")
    if progress is not None:
        progress(_storage_mean(mesh))
    return solver.fluid_fan_energy, solver.fluid_delivered


def run_cycle(mesh: Mesh3D, loop: FluidLoop, settings: CycleSettings = None,
              solver_config: SolverConfig = None,
              progress=None) -> CycleReport:
    """Charge to the target, let the store sit, discharge to the delivery floor."""
    settings = settings or CycleSettings()
    solver_config = solver_config or SolverConfig(method="cg", preconditioner="amg_rs",
                                                  tolerance=1e-7)
    report = CycleReport()

    def snapshot() -> tuple[float, float]:
        balance = compute_balance(mesh)
        return _storage_mean(mesh), balance.e_stored

    # ------------------------------------------------------------------ charge
    phase = report.charge
    phase.t_start, phase.stored_start = snapshot()
    phase.t_end = phase.t_start
    while phase.seconds < settings.charge_limit:
        fan, _ = _phase_chunk(mesh, loop, solver_config, settings.charge_power, 0.0,
                              settings.chunk, settings.dt, progress)
        phase.fan_energy += fan
        phase.energy_in += settings.charge_power * settings.chunk
        phase.seconds += settings.chunk
        phase.t_end, phase.stored_end = snapshot()
        if phase.t_end >= settings.t_target:
            break
    if phase.t_end < settings.t_target:
        phase.notes.append(
            f"the charge stopped at {phase.t_end:.1f} K, below the target "
            f"{settings.t_target:.1f} K: raise the power or the time limit")

    # ----------------------------------------------------------------- standby
    phase = report.standby
    phase.t_start, phase.stored_start = snapshot()
    phase.t_end = phase.t_start
    remaining = settings.standby_time
    while remaining > 0.0 and phase.seconds < settings.max_chunks * settings.chunk:
        step = min(settings.chunk, remaining)
        _phase_chunk(mesh, loop, solver_config, 0.0, 0.0, step, settings.dt, progress)
        phase.seconds += step
        remaining -= step
    phase.t_end, phase.stored_end = snapshot()

    # --------------------------------------------------------------- discharge
    phase = report.discharge
    phase.t_start, phase.stored_start = snapshot()
    phase.t_end = phase.t_start
    # short inspections near the end: the discharge must stop *at* the delivery floor,
    # and a two-hour step would overshoot it (and could push the field out of the
    # Kelvin contract on a small bed)
    step = min(settings.chunk, 1800.0)
    if settings.discharge_power <= 0.0:
        phase.notes.append("no discharge power requested: the phase is skipped")
    while settings.discharge_power > 0.0 and phase.t_end > settings.t_delivery_min \
            and phase.seconds < settings.max_chunks * settings.chunk:
        fan, delivered = _phase_chunk(mesh, loop, solver_config, 0.0,
                                      settings.discharge_power, step,
                                      settings.dt, progress)
        phase.fan_energy += fan
        phase.energy_out += delivered
        phase.seconds += step
        phase.t_end, phase.stored_end = snapshot()
        if phase.t_end <= settings.t_delivery_min:
            break
        remaining = settings.t_delivery_min - phase.t_end
        if phase.t_end < 250.0:                      # a safety floor, not a target
            phase.notes.append(
                f"the discharge ran the storage down to {phase.t_end:.1f} K: the "
                f"extraction is no longer delivering usable heat")
            break
    if phase.t_end > settings.t_delivery_min:
        phase.notes.append(
            f"the discharge stopped with the storage still at {phase.t_end:.1f} K: "
            f"the time limit was reached before the delivery floor")
    return report
