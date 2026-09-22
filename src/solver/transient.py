"""Transient solver: backward Euler with a per-volume mass matrix, on either mesh.

Key correctness points (each one was wrong in the previous implementation):

* ``M = diag(rho*cp)`` [J/(m^3*K)].  Including the cell volume scales every time
  constant by ``1/d^3`` and makes the result depend on the mesh resolution.
* Dirichlet rows are replaced by ``T = T_bc`` in both the matrix and the RHS, so
  a fixed-temperature face is exact instead of relaxing with a time constant.
* The power profile drives ``mesh.Q_source`` on the cells flagged in
  ``mesh.source_mask``; a run whose profile asks for power with no source cell
  raises instead of silently heating nothing.
* With a gas loop the profiles drive the loop, never the sand: the resistors add
  ``power`` to the gas, the exchanger takes the extraction out of it (or returns the gas
  at a fixed temperature), and the march is handed to the solid as an *implicit* film
  on the pipe cells (:meth:`~src.solver.fluid.FluidResult.apply`), which is stable at
  any step.  Without a loop the extraction is the legacy tube film or a capped sink.
* One march, either mesh: the operators and the right-hand side are asked of the
  mesh (``build_transient_operators``/``transient_rhs`` dispatch to the tree's own
  assembly), and the driver is left with what the two have in common - the profiles,
  the fluid loop, the per-step ``should_stop`` hook, the balance and the samples.
  ``Mesh3D`` keeps its ``GridIndex`` tables and its 3-D field; a tree carries the flat
  per-leaf vector and no index arithmetic at all.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..analysis.balance import compute_balance
from ..analysis.fluxes import pinned_cells
from ..constants import T_AMBIENT_DEFAULT
from ..core.mesh import MaterialID, Mesh3D
from ..core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from .linear import PreconditionerCache, solve_linear
from ..core.grid import GridIndex
from .fluid import FluidLoop, hold_loop_balance
from .matrix import build_transient_operators, transient_rhs
from .results import TransientResults
from .steady import SolverConfig

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from ..core.adaptive_mesh import AdaptiveMesh


@dataclass
class TransientConfig:
    """Time stepping, initial state and the two driving profiles."""

    t_final: float = 3600.0          # [s]
    dt: float = 60.0                 # [s]
    save_interval: float = 60.0      # [s]
    save_full_field: bool = False    # keep the whole T field per saved sample
    t_ambient: float = T_AMBIENT_DEFAULT
    initial_condition: InitialCondition = field(default_factory=InitialCondition)
    power_profile: PowerProfile = field(default_factory=PowerProfile)
    extraction_profile: ExtractionProfile = field(default_factory=ExtractionProfile)
    #: a closed gas loop through pipes buried in the bed: when present it *is* the
    #: heat transfer path, and the profiles drive its external power (resistors while
    #: charging, exchanger while discharging) instead of depositing heat in the sand
    fluid_loop: FluidLoop | None = None

    def n_steps(self) -> int:
        return int(np.ceil(self.t_final / self.dt))

    def times(self) -> np.ndarray:
        """Sample times, never stepping past ``t_final``."""
        steps = self.n_steps()
        times = np.arange(steps + 1, dtype=float) * self.dt
        return np.clip(times, 0.0, self.t_final) if self.t_final > 0 else np.array([0.0])

    def validate(self, mesh: Mesh3D | AdaptiveMesh = None) -> list[str]:
        problems = []
        if self.t_final <= 0:
            problems.append("t_final must be > 0")
        if self.dt <= 0:
            problems.append("dt must be > 0")
        if self.save_interval <= 0:
            problems.append("save_interval must be > 0")
        problems.extend(self.power_profile.validate())
        problems.extend(self.extraction_profile.validate())
        problems.extend(self.initial_condition.validate(mesh))
        return problems


class TransientSolver:
    """Time integration of the non-linear-free heat equation (Backward Euler).

    Two meshes, one march: a structured :class:`~src.core.mesh.Mesh3D` assembles through
    the ``GridIndex`` tables and keeps its field in the 3-D array, an adaptive mesh
    through its own operator and its flat per-leaf vector, and the profile handling, the
    fluid loop, the ``should_stop`` hook, the balance and the samples below are the same
    code for both.  The two mesh-specific names are decided once, here: the index tables
    (a tree has none) and the volume scaling of the linear layer.
    """

    def __init__(self, mesh: Mesh3D | AdaptiveMesh, config: TransientConfig,
                 solver_config: SolverConfig = None) -> None:
        self.mesh = mesh
        self.config = config
        self.solver_config = solver_config or SolverConfig(method="bicgstab",
                                                           preconditioner="jacobi")
        self.adaptive = not isinstance(mesh, Mesh3D)
        if self.adaptive and not hasattr(mesh, "transient_operators"):
            raise TypeError(f"unsupported mesh {type(mesh).__name__}: a Mesh3D or an "
                            f"adaptive mesh is required")
        self.index = None if self.adaptive else GridIndex.from_mesh(mesh)
        self._cache = PreconditionerCache()
        # per-volume coefficients are symmetric only on a mesh whose cells are all the
        # same size: hand the cell volumes to the iterative solver so it can symmetrise
        # (see solve_linear).  A tree answers with its own leaf volumes, or with None.
        self._scale = None
        if self.adaptive:
            self._scale = mesh.volume_scale()
        elif not mesh.uniform:
            self._scale = mesh.V.ravel(order="F")
        self.notes = []
        #: outcome of the last fluid-loop march (None when no loop is configured)
        self.fluid_result = None
        #: energies carried by the loop over the run [J]: the blower work and what the
        #: exchanger delivered (the two terms a cycle report needs)
        self.fluid_fan_energy = 0.0
        self.fluid_delivered = 0.0

    # ------------------------------------------------------------------ setup
    def apply_initial_condition(self) -> None:
        field = self.config.initial_condition.apply_to_mesh(self.mesh)
        if field is not None:
            self.mesh.T = field

    def _tube_film(self) -> tuple[float, float]:
        """Signature of the tube film the operator must be built with.

        Only the coefficients enter the matrix (the film temperatures are on the
        right-hand side, rebuilt every step), so the signature is two moments of
        ``bc_h`` on the tube cells: cheap, and enough to notice a changed exchange.
        """
        if getattr(self, "_n_tube", 0) == 0:
            return (0.0, 0.0)
        h = self.mesh.bc_h[self._tube]
        return (float(np.sum(h)), float(np.sum(h * h)))

    def _source_masks(self) -> None:
        mask = self.mesh.source_mask
        self._n_source = int(np.count_nonzero(mask))
        if self._n_source == 0:
            self._n_source = None
        self._tube = self.mesh.material_id == int(MaterialID.TUBES)
        self._n_tube = int(np.count_nonzero(self._tube))
        # exchange area the tube cells can offer to the fluid, per unit volume
        self._tube_area_over_v = float(np.sum(self.mesh.V[self._tube]
                                              / self.mesh.h_char[self._tube]))

    def _set_fluid_loop(self, power: float, t: float) -> bool:
        """March the loop and hand it to the solid; True when the loop did the work.

        The external power of the loop is what the resistors inject (``power``) minus
        what the exchanger takes out (the extraction request): the gas carries the heat,
        the sand sees the pipes, and the fan power is reported with the cycle.  With the
        ``return_temperature`` extraction the exchanger returns the gas at a fixed
        temperature instead, and the power it takes is whatever the bed gives the gas.
        """
        loop = self.config.fluid_loop
        if loop is None:
            return False
        profile = self.config.extraction_profile
        if profile.mode == "return_temperature":
            loop.t_in = float(profile.t_inlet)
            loop.external_power = 0.0
        else:
            loop.t_in = self._loop_t_in
            # the resistors add to the gas, the exchanger takes from it
            loop.external_power = float(power) - profile.power_request(t)
        result = loop.solve(self.mesh)
        self.fluid_result = result
        if result.conductance is None:
            return False
        active = result.apply(self.mesh)
        if self.index is not None and np.any(np.asarray(active).ravel(order="F")
                                             & ~self.index.interior_tube):
            # the structured index lists the film cells once: rebuild it when the loop
            # reaches a cell it did not list, or that exchange would be dropped
            self.index = GridIndex.from_mesh(self.mesh)
        return True

    def _set_power(self, power: float) -> None:
        if self._n_source is None:
            if power > 0:
                raise ValueError(
                    "the power profile asks for heating but no cell is flagged as a "
                    "heat source (mesh.source_mask is empty): check the heater pattern"
                )
            return
        self.mesh.Q_source.fill(0.0)
        if power > 0:
            mask = self.mesh.source_mask
            self.mesh.Q_source[mask] = power / float(self.mesh.V[mask].sum())

    def _set_extraction(self, t: float) -> None:
        profile = self.config.extraction_profile
        self.mesh.Q_sink.fill(0.0)
        if profile.mode == "return_temperature":
            raise ValueError("the return-temperature extraction drives the gas loop: build "
                             "and paint the pipe network first")
        if profile.mode == "off" or self._n_tube == 0:
            if profile.mode != "off" and self._n_tube == 0:
                raise ValueError("extraction is enabled but the mesh has no tube cells")
            self.mesh.bc_h[self._tube] = 0.0
            return
        t_inlet = profile.t_inlet
        if profile.mode == "flow_rate":
            # heat leaves through the tube wall: convective BC, capped by physics
            self.mesh.set_internal_convection(self._tube, profile.h_fluid, t_inlet)
            return
        # power mode: volumetric sink, capped by what the tube surface can deliver
        self.mesh.set_internal_convection(self._tube, 0.0, t_inlet)
        requested = profile.power_request(t)
        t_tube = float(self.mesh.T[self._tube].mean())
        available = profile.h_fluid * self._tube_area_over_v * max(t_tube - t_inlet, 0.0)
        power = min(requested, available)
        if power > 0:
            self.mesh.Q_sink[self._tube] = -power / float(self.mesh.V[self._tube].sum())

    def _hold_loop_balance(self, a, linear) -> None:
        """Make the step deposit exactly the loop's external power (in place).

        See :func:`~src.solver.fluid.hold_loop_balance`: the loop inlet is solved with
        the field, at the cost of one more solve with the same (cached) operator.
        """
        if self.fluid_result is None:
            return
        linear.T = hold_loop_balance(
            self.fluid_result, self.mesh, linear.T,
            lambda e: solve_linear(a, e, self.solver_config, cache=self._cache,
                                   scale=self._scale).T,
            pinned_cells(self.mesh, self.index))

    # -------------------------------------------------------------------- run
    def run(self, progress_callback: Callable[[int, str], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> TransientResults:
        """March to ``t_final``, or stop early when ``should_stop`` says so.

        ``should_stop`` is asked before every step, on the mesh the previous step left
        behind, so a caller that watches the field (the cycle's phase criteria) ends the
        run on the state it judges rather than at a saved sample.
        """
        problems = self.config.validate(self.mesh)
        if problems:
            raise ValueError("invalid transient configuration: " + "; ".join(problems))
        self.mesh.validate()
        self.apply_initial_condition()
        self._source_masks()
        loop = self.config.fluid_loop
        self._loop_t_in = None if loop is None else loop.t_in
        if loop is not None:
            # the loop is the only heat path: its film is on the operator from step one,
            # so the matrix is built with it and reused while the film does not change
            try:
                self._set_fluid_loop(self.config.power_profile.power_at(0.0), 0.0)
            except ValueError as exc:
                raise ValueError(f"the gas loop cannot start: {exc}") from exc

        cfg = self.config
        dt = cfg.dt
        stationary_operator = not self.solver_config.radiation
        a = None
        m_diag = None
        # the assembled operator carries the tube film (``a_p[tube] += h/h_char``) while
        # the right-hand side is rebuilt every step from the *current* bc_h: if a loop
        # marches the fluid and changes that film, the cached matrix would keep the old
        # term with no companion on the right-hand side and the tube cells would be
        # driven to a nonsense temperature.  Track the film the operator was built with.
        if stationary_operator:
            a, m_diag = build_transient_operators(self.mesh, dt, self.index)
        film_at_build = self._tube_film()

        results = TransientResults()
        t, t_start = 0.0, time.perf_counter()
        next_save = 0.0
        e_in = e_out = e_loss = 0.0
        e_prev = compute_balance(self.mesh, cfg.t_ambient, self.index,
                                     radiation=self.solver_config.radiation).e_stored

        while t < cfg.t_final - 1e-9:
            if should_stop is not None and should_stop():
                break
            step_dt = min(dt, cfg.t_final - t)
            power = cfg.power_profile.power_at(t)
            # the loop was marched on the field the previous step left behind (at the end
            # of that step, or before the first one): its film is already on the mesh
            loop_used = loop is not None and self.fluid_result is not None
            if loop_used:
                self.fluid_fan_energy += self.fluid_result.fan_power * step_dt
            if not loop_used:
                self._set_power(power)
                self._set_extraction(t)

            film_now = self._tube_film()
            if not stationary_operator or step_dt != dt or film_now != film_at_build:
                # the shortened final step needs operators built for its own dt, and a
                # film that changed (the loop marched) needs a rebuilt diagonal too
                a, m_diag = build_transient_operators(
                    self.mesh, step_dt, self.index,
                    radiation=self.solver_config.radiation)
                film_at_build = film_now
            x = self.mesh.T.ravel(order="F").copy()
            rhs = transient_rhs(self.mesh, m_diag, step_dt, x, index=self.index,
                                radiation=self.solver_config.radiation)
            linear = solve_linear(a, rhs, self.solver_config, x0=x, cache=self._cache,
                                  scale=self._scale)
            if loop_used:
                self._hold_loop_balance(a, linear)
            if not linear.converged:
                self.notes.append(f"t={t:.0f}s: linear solve did not converge "
                                  f"(residual {linear.residual:.2e})")
            self.mesh.T = (np.asarray(linear.T, dtype=float) if self.adaptive
                           else self.mesh.unflatten_field(linear.T))

            balance = compute_balance(self.mesh, cfg.t_ambient, self.index,
                                      radiation=self.solver_config.radiation)
            d_e = balance.e_stored - e_prev
            balance.imbalance = (balance.p_input - balance.p_extracted - balance.q_domain
                                 - d_e / step_dt)
            if loop_used:
                if cfg.extraction_profile.mode == "return_temperature":
                    # the gas comes back at a set temperature and the resistors are off:
                    # the exchanger takes what the bed gave the gas *on the solved field*
                    # (the march's own estimate is explicit in the wall temperature and
                    # overstates a hot start), or heats the gas when the bed is colder
                    net = balance.p_extracted - balance.p_input
                    self.fluid_delivered += max(net, 0.0) * step_dt
                    power = max(-net, 0.0)
                else:
                    # the balance is held: the exchanger takes exactly what it asked for
                    self.fluid_delivered += (cfg.extraction_profile.power_request(t)
                                             * step_dt)
            e_in += power * step_dt
            e_out += balance.p_extracted * step_dt
            e_loss += balance.q_battery * step_dt
            e_prev = balance.e_stored
            t += step_dt

            if t >= next_save - 1e-9 or t >= cfg.t_final - 1e-9:
                # regular samples, plus always the final state so the series ends
                # exactly at t_final
                field = self.mesh.T.copy() if cfg.save_full_field else None
                results.add_timestep(
                    t, field, T_mean_storage=balance.t_mean_storage, T_max=balance.t_max,
                    T_min=balance.t_min, T_mean_insulation=balance.t_mean_insulation,
                    T_mean_shell=balance.t_mean_shell, P_heaters=power,
                    P_extracted=balance.p_extracted, Q_losses_top=balance.q_battery_top,
                    Q_losses_bottom=balance.q_battery_bottom,
                    Q_losses_side=balance.q_battery_side, Q_losses_total=balance.q_battery,
                    E_stored=balance.e_stored, E_in_cumulative=e_in,
                    E_out_cumulative=e_out, E_losses_cumulative=e_loss,
                    Ex_stored=balance.ex_stored, Ex_destroyed=balance.ex_destroyed,
                )
                next_save += cfg.save_interval

            if progress_callback:
                progress_callback(int(100 * min(t, cfg.t_final) / cfg.t_final),
                                  f"t = {t:.0f} s / {cfg.t_final:.0f} s")

            if loop is not None and t < cfg.t_final - 1e-9:
                # march the gas on the new field for the next step: the loop state a
                # caller reads between steps (the delivery temperature a cycle stops on)
                # is then the one of the field it sees, not one step behind it
                try:
                    self._set_fluid_loop(cfg.power_profile.power_at(t), t)
                except ValueError as exc:
                    # an operating point the loop cannot carry (a discharge that has
                    # drained the bed): stop on the last good state, keep the samples
                    self.notes.append(f"t={t:.0f}s: stopped - {exc}")
                    break

        results.wall_time = time.perf_counter() - t_start
        return results


def run_transient_simulation(mesh: Mesh3D | AdaptiveMesh, config: TransientConfig,
                             solver_config: SolverConfig = None,
                             progress_callback: Callable[[int, str], None] | None = None,
                             should_stop: Callable[[], bool] | None = None) -> TransientResults:
    """Convenience wrapper used by the GUI controller, on either mesh."""
    solver = TransientSolver(mesh, config, solver_config)
    return solver.run(progress_callback=progress_callback, should_stop=should_stop)
