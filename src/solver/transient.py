"""Transient solver: backward Euler with a per-volume mass matrix.

Key correctness points (each one was wrong in the previous implementation):

* ``M = diag(rho*cp)`` [J/(m^3*K)].  Including the cell volume scales every time
  constant by ``1/d^3`` and makes the result depend on the mesh resolution.
* Dirichlet rows are replaced by ``T = T_bc`` in both the matrix and the RHS, so
  a fixed-temperature face is exact instead of relaxing with a time constant.
* The power profile drives ``mesh.Q_source`` on the cells flagged in
  ``mesh.source_mask``; a run whose profile asks for power with no source cell
  raises instead of silently heating nothing.
* The extraction profile is applied to the physics (tube-side convection or a
  capped volumetric sink on the tube cells) and the removed power is measured,
  not assumed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

from ..analysis.balance import compute_balance
from ..constants import T_AMBIENT_DEFAULT
from ..core.mesh import MaterialID, Mesh3D
from ..core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from .linear import PreconditionerCache, solve_linear
from ..core.grid import GridIndex
from .matrix import build_transient_operators, transient_rhs
from .results import TransientResults
from .steady import SolverConfig


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

    def n_steps(self) -> int:
        return int(np.ceil(self.t_final / self.dt))

    def times(self) -> np.ndarray:
        """Sample times, never stepping past ``t_final``."""
        steps = self.n_steps()
        times = np.arange(steps + 1, dtype=float) * self.dt
        return np.clip(times, 0.0, self.t_final) if self.t_final > 0 else np.array([0.0])

    def validate(self, mesh: Mesh3D = None) -> list[str]:
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
    """Time integration of the non-linear-free heat equation (Backward Euler)."""

    def __init__(self, mesh: Mesh3D, config: TransientConfig,
                 solver_config: SolverConfig = None) -> None:
        self.mesh = mesh
        self.config = config
        self.solver_config = solver_config or SolverConfig(method="bicgstab",
                                                           preconditioner="jacobi")
        self.index = GridIndex.from_mesh(mesh)
        self._cache = PreconditionerCache()
        # per-volume coefficients are symmetric only on a uniform grid: hand the
        # cell volumes to the iterative solver so it can symmetrise (see solve_linear)
        self._scale = None if self.mesh.uniform else self.mesh.V.ravel(order="F")
        self.notes = []

    # ------------------------------------------------------------------ setup
    def apply_initial_condition(self) -> None:
        field = self.config.initial_condition.apply_to_mesh(self.mesh)
        if field is not None:
            self.mesh.T = field

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

    # -------------------------------------------------------------------- run
    def run(self, progress_callback: Callable[[int, str], None] | None = None,
            should_stop: Callable[[], bool] | None = None) -> TransientResults:
        problems = self.config.validate(self.mesh)
        if problems:
            raise ValueError("invalid transient configuration: " + "; ".join(problems))
        self.mesh.validate()
        self.apply_initial_condition()
        self._source_masks()

        cfg = self.config
        dt = cfg.dt
        stationary_operator = not self.solver_config.radiation
        a = None
        m_diag = None
        if stationary_operator:
            a, m_diag = build_transient_operators(self.mesh, dt, self.index)

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
            self._set_power(power)
            self._set_extraction(t)

            if not stationary_operator or step_dt != dt:
                # the shortened final step needs operators built for its own dt,
                # otherwise the matrix and the right-hand side disagree
                a, m_diag = build_transient_operators(
                    self.mesh, step_dt, self.index,
                    radiation=self.solver_config.radiation)
            x = self.mesh.T.ravel(order="F").copy()
            rhs = transient_rhs(self.mesh, m_diag, step_dt, x, index=self.index,
                                radiation=self.solver_config.radiation)
            linear = solve_linear(a, rhs, self.solver_config, x0=x, cache=self._cache,
                                  scale=self._scale)
            if not linear.converged:
                self.notes.append(f"t={t:.0f}s: linear solve did not converge "
                                  f"(residual {linear.residual:.2e})")
            self.mesh.T = self.mesh.unflatten_field(linear.T)

            balance = compute_balance(self.mesh, cfg.t_ambient, self.index,
                                      radiation=self.solver_config.radiation)
            d_e = balance.e_stored - e_prev
            balance.imbalance = (balance.p_input - balance.p_extracted - balance.q_domain
                                 - d_e / step_dt)
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

        results.wall_time = time.perf_counter() - t_start
        return results


def run_transient_simulation(mesh: Mesh3D, config: TransientConfig,
                             solver_config: SolverConfig = None,
                             progress_callback: Callable[[int, str], None] | None = None,
                             should_stop: Callable[[], bool] | None = None) -> TransientResults:
    """Convenience wrapper used by the GUI controller."""
    solver = TransientSolver(mesh, config, solver_config)
    return solver.run(progress_callback=progress_callback, should_stop=should_stop)
