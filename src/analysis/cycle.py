"""The operating cycle: charge, standby, discharge, and where the energy goes.

A storage is judged by a cycle, not by a steady state.  This module drives the three
phases with the transient solver and accounts for the energy in the only way that
cannot hide a term:

    E_in(electric) = dE_stored + E_delivered + E_standby + E_circulation

where

* ``E_in`` is what the plant draws from the grid, **blower included** (the resistors
  heat the gas, the fan pushes it: both are electric loads of the same system);
* ``dE_stored`` is the sensible energy above the ambient at the end of the cycle minus
  the same at the start.  The energy still inside the bed when the discharge stops
  (``E_unrecovered``) is the part of ``dE_stored`` the user never got: with a store that
  starts at the ambient reference the two are the *same number*, which is why they are
  reported as one term.  Writing ``... + E_circulation + E_unrecovered`` on top of
  ``dE_stored`` counts the stored energy twice and never closes;
* ``E_delivered`` is what the exchanger hands to the user during the discharge;
* ``E_standby`` is what leaks through the envelope, summed over the three phases (the
  standby leg is the quiet one, but the charge and the discharge leak too);
* ``E_circulation`` is the blower work, from the pressure drop of the actual network.

The three phases are driven by the *gas loop*, so the resistors heat the gas and the
gas heats the bed; nothing is deposited in the sand directly.  Every energy above is
*measured* from the balance of the field after each step - what the sources deposit
against what the sinks and the envelope carry away - and not assumed from the power the
profile asked for, so the identity closes to the solver's own imbalance
(``CycleReport.balance_residual``) rather than to a book entry.

Stopping: every phase stops on a **per-step criterion** (``StopWhen``), evaluated after
each step with the state the solver has reached.  A phase is driven in chunks so the
driver can report progress, but the chunks are only reporting boundaries: the criterion
stops the phase *inside* a chunk, which is what lets the discharge end when the gas
leaving the bed drops below the delivery floor instead of hours later - and it is also
what keeps a small bed from being driven towards the Kelvin contract by a discharge
much faster than its capacity.  The reason travels back in ``CyclePhase.notes``, so a
report says why every phase ended.

Rounding of the phases: the charge runs until the storage mean reaches the target
temperature, the standby for a prescribed time, the discharge until the gas delivered to
the user falls below its floor.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..constants import T_MIN_VALID
from ..core.mesh import MaterialID, Mesh3D
from ..core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from ..solver.fluid import FluidLoop, FluidResult
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
    t_delivery_min: float = 353.15         # [K] below this the gas is not usable
    dt: float = 1800.0                     # [s] time step of the transient
    chunk: float = 12.0 * 3600.0           # [s] how often the driver reports progress
    max_chunks: int = 240


@dataclass(frozen=True)
class StepState:
    """What a stop criterion sees after a step: the field and the loop driving it."""

    mesh: Mesh3D
    seconds: float          # [s] elapsed inside the phase
    t_mean: float           # [K] volume-averaged storage temperature
    t_min: float            # [K] coldest cell of the field
    t_delivery: float       # [K] gas leaving the bed, NaN before the first step
    fan_power: float        # [W] blower shaft power of that step


#: returns the reason to stop the phase, or ``None`` to carry on
StopWhen = Callable[[StepState], str | None]


@dataclass
class CyclePhase:
    """One phase of the cycle and what it cost or delivered."""

    name: str
    seconds: float = 0.0
    energy_in: float = 0.0          # [J] electric: what the resistors of the phase put in
    energy_out: float = 0.0         # [J] what the exchanger handed the user
    energy_loss: float = 0.0        # [J] through the envelope
    fan_energy: float = 0.0         # [J] electric, the blower
    stored_start: float = 0.0       # [J]
    stored_end: float = 0.0         # [J]
    t_start: float = float("nan")   # [K] storage mean
    t_end: float = float("nan")
    t_delivery_end: float = float("nan")   # [K] gas leaving the bed at the end
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
    def phases(self) -> tuple[CyclePhase, CyclePhase, CyclePhase]:
        return (self.charge, self.standby, self.discharge)

    @property
    def energy_in(self) -> float:
        """Electric energy the resistors took over the cycle [J] (blower apart)."""
        return sum(phase.energy_in for phase in self.phases)

    @property
    def energy_electric(self) -> float:
        """Electric energy the plant took from the grid [J]: resistors and blower."""
        return self.energy_in + self.fan_energy

    @property
    def fan_energy(self) -> float:
        """Blower work over the whole cycle [J]."""
        return sum(phase.fan_energy for phase in self.phases)

    @property
    def dE_stored(self) -> float:
        """Sensible energy above the ambient gained over the cycle [J] (signed).

        The unrecovered energy is its reading at the end of the discharge; on a cycle
        that starts at the ambient reference the two coincide.
        """
        return self.discharge.stored_end - self.charge.stored_start

    @property
    def circulation_loss(self) -> float:
        """Blower work over the electricity the plant draws [-].

        The published circulation figure of a storage cycle is an aggregate (blower,
        controls, hot ducts); this is the blower share alone, computed from the
        pressure drop of the actual pipe network.
        """
        return self.fan_energy / self.energy_electric if self.energy_electric > 0 else 0.0

    @property
    def standby_loss(self) -> float:
        """Energy lost while the store sat, over the electricity the plant draws [-]."""
        lost = self.standby.stored_change
        return (-lost / self.energy_electric
                if self.energy_electric > 0 and lost < 0 else 0.0)

    @property
    def energy_loss(self) -> float:
        """Heat that left through the envelope over the whole cycle [J]."""
        return sum(phase.energy_loss for phase in self.phases)

    @property
    def energy_delivered(self) -> float:
        """Heat the exchanger handed to the user [J]."""
        return self.discharge.energy_out

    @property
    def unrecovered(self) -> float:
        """Energy still in the bed when the discharge had to stop [J].

        Measured from the ambient reference, like the balance: it is what a following
        cycle starts from, and the reason a low-temperature tail is worth chasing.
        """
        return self.discharge.stored_end

    @property
    def round_trip(self) -> float:
        """Delivered heat over the electricity the plant draws [-] (blower included)."""
        return (self.energy_delivered / self.energy_electric
                if self.energy_electric > 0 else 0.0)

    def balance_residual(self) -> float:
        """Relative mismatch of ``E_in = dE_stored + delivered + loss + fan`` [-].

        The mismatch over the electricity the plant draws: it is the sum of the
        solver's own per-step imbalances, so it is a numerical figure, not a model one.
        """
        if self.energy_electric <= 0:
            return 0.0
        carried = (self.dE_stored + self.energy_delivered + self.energy_loss
                   + self.fan_energy)
        return abs(self.energy_electric - carried) / self.energy_electric

    def summary(self) -> str:
        lines = [
            f"charge    : {self.charge.seconds / 3600:.1f} h, "
            f"heaters {self.charge.energy_in / 3.6e9:.2f} MWh, "
            f"storage {self.charge.t_start:.1f} -> {self.charge.t_end:.1f} K, "
            f"fan {self.charge.fan_energy / 3.6e6:.1f} kWh",
            f"standby   : {self.standby.seconds / 86400:.1f} d, "
            f"loss {self.standby.stored_change / 3.6e9:+.3f} MWh "
            f"({-self.standby.stored_change / max(self.standby.seconds, 1.0) * 86400 / 1e6:.1f}"
            f" MJ/day), fan {self.standby.fan_energy / 3.6e6:.1f} kWh",
            f"discharge : {self.discharge.seconds / 3600:.1f} h, "
            f"delivered {self.discharge.energy_out / 3.6e9:.2f} MWh, "
            f"storage {self.discharge.t_start:.1f} -> {self.discharge.t_end:.1f} K, "
            f"gas delivered {self.discharge.t_delivery_end:.1f} K, "
            f"fan {self.discharge.fan_energy / 3.6e6:.1f} kWh",
            f"balance   : electric {self.energy_electric / 3.6e9:.3f} MWh = stored "
            f"{self.dE_stored / 3.6e9:+.3f} + delivered {self.energy_delivered / 3.6e9:.3f} "
            f"+ losses {self.energy_loss / 3.6e9:.3f} + circulation "
            f"{self.fan_energy / 3.6e9:.3f} MWh, residual "
            f"{100 * self.balance_residual():.3f}%",
            f"round trip: {100 * self.round_trip:.1f}% delivered / electric (heaters "
            f"{self.energy_in / 3.6e9:.3f} + fan {self.fan_energy / 3.6e9:.3f} MWh), "
            f"circulation {100 * self.circulation_loss:.1f}%, "
            f"standby {100 * self.standby_loss:.1f}%, "
            f"still stored at the end {self.unrecovered / 3.6e9:.2f} MWh",
        ]
        for phase in self.phases:
            lines.extend(f"  [{phase.name}] {note}" for note in phase.notes)
        return "\n".join(lines)


def _storage_mean(mesh: Mesh3D) -> float:
    """Volume-averaged temperature of the storage material [K]."""
    mask = mesh.material_id == int(MaterialID.SAND)
    if not mask.any():
        mask = mesh.material_id != int(MaterialID.AIR)
    return float(np.mean(mesh.T[mask]))


def _step_state(mesh: Mesh3D, seconds: float,
                loop_result: FluidResult | None) -> StepState:
    """The state a criterion judges: the field and the loop that is driving it."""
    return StepState(
        mesh=mesh,
        seconds=seconds,
        t_mean=_storage_mean(mesh),
        t_min=float(mesh.T.min()),
        t_delivery=loop_result.t_out if loop_result is not None else float("nan"),
        fan_power=loop_result.fan_power if loop_result is not None else 0.0,
    )


def stop_at_target(settings: CycleSettings) -> StopWhen:
    """Charge until the storage mean reaches ``settings.t_target``."""

    def criterion(state: StepState) -> str | None:
        if state.t_mean >= settings.t_target:
            return (f"the storage mean reached {state.t_mean:.1f} K, the target "
                    f"{settings.t_target:.1f} K, after {state.seconds / 3600:.1f} h")
        return None

    return criterion


def stop_at_delivery_floor(settings: CycleSettings) -> StopWhen:
    """Discharge until the gas leaving the bed can no longer serve the user.

    The delivery temperature is the one the user sees, so it - not the storage mean -
    is what the phase is allowed to be stopped on.  The Kelvin contract is the other
    limit: a loop that pulls harder than the store can supply must stop the phase and
    say so, rather than trip the field check of the solver mid-chunk.
    """

    def criterion(state: StepState) -> str | None:
        if state.t_delivery <= settings.t_delivery_min:
            return (f"the delivery temperature fell to {state.t_delivery:.1f} K, the floor "
                    f"{settings.t_delivery_min:.1f} K, after {state.seconds / 3600:.1f} h: "
                    f"the store has nothing usable left")
        if state.t_min < T_MIN_VALID:
            return (f"the field fell to {state.t_min:.1f} K, below the {T_MIN_VALID:.0f} K "
                    f"contract: the discharge is pulling harder than the store can supply")
        return None

    return criterion


class _StepClock:
    """Reproduces the step boundaries of a run so a callback can weigh them in time.

    ``TransientSolver`` steps ``dt`` and shortens only the last step of the run; a
    per-step callback that integrates a flux needs the length of the step it belongs
    to, which is this - not the formatted message the solver passes along.
    """

    def __init__(self, t_final: float, dt: float) -> None:
        self.t_final = float(t_final)
        self.dt = float(dt)
        self.t = 0.0

    def advance(self) -> float:
        """Length of the step just completed [s], and move the clock past it."""
        step = min(self.dt, self.t_final - self.t)
        self.t += step
        return step


@dataclass
class _Ledger:
    """What one chunk of a phase moved."""

    seconds: float = 0.0
    energy_in: float = 0.0
    energy_out: float = 0.0
    energy_loss: float = 0.0
    fan_energy: float = 0.0
    t_delivery: float = float("nan")
    stop_reason: str | None = None
    loop_result: FluidResult | None = None


def _phase_chunk(mesh: Mesh3D, loop: FluidLoop, solver_config: SolverConfig,
                 power: float, extraction: float, seconds: float, dt: float,
                 progress=None, stop_when: StopWhen | None = None,
                 elapsed: float = 0.0,
                 previous: FluidResult | None = None) -> _Ledger:
    """Run one chunk of a phase and report what it moved.

    The energies are measured, not assumed: every step is weighed by the balance of the
    field it left behind, so the cycle identity closes on the solver's own imbalance
    instead of on a book entry.  ``stop_when`` is evaluated after every step (with
    ``elapsed`` counting the phase from its start) and once before the chunk starts,
    with ``previous`` carrying the loop state the last chunk ended on; when it returns a
    reason the phase ends there - inside the chunk, not at its boundary - and the reason
    travels back in the ledger.
    """
    ledger = _Ledger(loop_result=previous)
    if stop_when is not None:
        # a crossing on the last step of the previous chunk is still a crossing, and
        # the field checks of the solver would raise before the criterion saw it
        reason = stop_when(_step_state(mesh, elapsed, previous))
        if reason is not None:
            ledger.stop_reason = reason
            return ledger

    config = TransientConfig(
        t_final=seconds, dt=dt, save_interval=dt,
        power_profile=PowerProfile(mode="constant", constant_power=power),
        extraction_profile=ExtractionProfile(mode="power", power=extraction),
        fluid_loop=loop,
        # keep the field the previous chunk left: the cycle is continuous
        initial_condition=InitialCondition(mode="keep"),
    )
    solver = TransientSolver(mesh, config, solver_config)
    clock = _StepClock(seconds, dt)

    def after_step(_percent: int, _message: str) -> None:
        step = clock.advance()
        balance = compute_balance(mesh, config.t_ambient, solver.index,
                                  radiation=solver_config.radiation)
        # the loop is solved, not switched: individual cells can go the other way (the
        # cold end of a discharge still takes heat from the gas it meets), but the net
        # is the external device, which is the energy the plant pays for or sells
        net = (balance.p_input - balance.p_extracted) * step
        ledger.seconds += step
        ledger.energy_loss += balance.q_domain * step
        if extraction > 0.0:
            ledger.energy_out += -net
        else:
            ledger.energy_in += net

    def should_stop() -> bool:
        if stop_when is None:
            return False
        reason = stop_when(_step_state(mesh, elapsed + clock.t, current_loop()))
        if reason is None:
            return False
        ledger.stop_reason = reason
        return True

    def current_loop() -> FluidResult | None:
        return solver.fluid_result if solver.fluid_result is not None else previous

    solver.run(progress_callback=after_step, should_stop=should_stop)
    ledger.loop_result = current_loop()
    ledger.fan_energy = solver.fluid_fan_energy
    if ledger.loop_result is not None:
        ledger.t_delivery = ledger.loop_result.t_out
    if not np.all(np.isfinite(mesh.T)):
        raise RuntimeError(
            f"the '{power:+.0f} W / {extraction:+.0f} W' chunk diverged: the field "
            f"left the physical range (check the loop set point and the time step)")
    if progress is not None:
        progress(_storage_mean(mesh))
    return ledger


def run_cycle(mesh: Mesh3D, loop: FluidLoop, settings: CycleSettings = None,
              solver_config: SolverConfig = None,
              progress=None) -> CycleReport:
    """Charge to the target, let the store sit, discharge to the delivery floor.

    The phases are driven in chunks of ``settings.chunk`` for the progress report, but
    they stop on the criterion of the phase, which is checked after every step: a
    discharge ends when the delivered gas reaches its floor, not at the next inspection.
    """
    settings = settings or CycleSettings()
    solver_config = solver_config or SolverConfig(method="cg", preconditioner="amg_rs",
                                                  tolerance=1e-7)
    report = CycleReport()

    def snapshot() -> tuple[float, float]:
        balance = compute_balance(mesh)
        return _storage_mean(mesh), balance.e_stored

    def advance(phase: CyclePhase, ledger: _Ledger) -> bool:
        """Fold a chunk into its phase; True when the criterion stopped the phase."""
        phase.seconds += ledger.seconds
        phase.energy_in += ledger.energy_in
        phase.energy_out += ledger.energy_out
        phase.energy_loss += ledger.energy_loss
        phase.fan_energy += ledger.fan_energy
        if np.isfinite(ledger.t_delivery):
            phase.t_delivery_end = ledger.t_delivery
        phase.t_end, phase.stored_end = snapshot()
        if ledger.stop_reason is not None:
            phase.notes.append(ledger.stop_reason)
            return True
        return False

    def drive(phase: CyclePhase, power: float, extraction: float,
              stop_when: StopWhen | None, limit: float) -> bool:
        """Run a phase in chunks; True when its criterion stopped it."""
        previous: FluidResult | None = None
        while phase.seconds < limit:
            ledger = _phase_chunk(mesh, loop, solver_config, power, extraction,
                                  min(settings.chunk, limit - phase.seconds),
                                  settings.dt, progress, stop_when, phase.seconds,
                                  previous)
            # the loop state is the criterion's other half: without it a chunk that
            # starts on a cold solver would judge the delivery on nothing
            previous = ledger.loop_result
            if advance(phase, ledger):
                return True
        return False

    # ------------------------------------------------------------------ charge
    phase = report.charge
    phase.t_start, phase.stored_start = snapshot()
    phase.t_end, phase.stored_end = phase.t_start, phase.stored_start
    drive(phase, settings.charge_power, 0.0, stop_at_target(settings),
          settings.charge_limit)
    if phase.t_end < settings.t_target:
        phase.notes.append(
            f"the charge stopped at {phase.t_end:.1f} K, below the target "
            f"{settings.t_target:.1f} K: raise the power or the time limit")

    # ----------------------------------------------------------------- standby
    phase = report.standby
    phase.t_start, phase.stored_start = snapshot()
    phase.t_end, phase.stored_end = phase.t_start, phase.stored_start
    drive(phase, 0.0, 0.0, None,
          min(settings.standby_time, settings.max_chunks * settings.chunk))

    # --------------------------------------------------------------- discharge
    phase = report.discharge
    phase.t_start, phase.stored_start = snapshot()
    phase.t_end, phase.stored_end = phase.t_start, phase.stored_start
    if settings.discharge_power <= 0.0:
        phase.notes.append("no discharge power requested: the phase is skipped")
    elif not drive(phase, 0.0, settings.discharge_power, stop_at_delivery_floor(settings),
                   settings.max_chunks * settings.chunk):
        phase.notes.append(
            f"the discharge stopped with the storage still at {phase.t_end:.1f} K and "
            f"the gas leaving the bed at {phase.t_delivery_end:.1f} K: the time limit "
            f"was reached before the delivery floor")
    return report
