"""Losses analysis: power needed to hold the storage at a target temperature.

The iterative solver used to live inside the GUI thread (with
``processEvents``), which made it untestable and re-entrant.  It now lives here
and is driven by the GUI controller in a worker thread.

Iteration: secant/Newton on the mean storage temperature, with an
under-relaxation factor and a bounded power floor.  The ground loss path is
applied explicitly (:meth:`LossesConfig.h_ground`), which is what the GUI used
to write by poking ``mesh.bc_h`` without changing the boundary type - and was
therefore silently ignored by the assembly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

from ..constants import T_AMBIENT_DEFAULT, T_GROUND_DEFAULT
from ..core.mesh import MaterialID, Mesh3D
from ..solver.steady import SolverConfig, SteadyStateSolver
from .balance import Balance, compute_balance


@dataclass
class LossesConfig:
    """Target and iteration controls of the losses analysis."""

    t_target: float = 873.15            # [K] mean storage temperature to hold
    t_ambient: float = T_AMBIENT_DEFAULT
    t_ground: float = T_GROUND_DEFAULT
    h_ground: float = 5.0               # [W/(m^2*K)]; 0 keeps the mesh BC as-is
    tolerance: float = 1.0              # [K]
    max_iterations: int = 20
    relaxation: float = 0.7             # 0..1 under-relaxation on the power update
    initial_density: float = 100.0      # [W/m^3] starting volumetric power

    def validate(self) -> list[str]:
        problems = []
        if self.t_target <= self.t_ambient:
            problems.append("t_target must be above t_ambient")
        if self.tolerance <= 0:
            problems.append("tolerance must be > 0")
        if not 1 <= self.max_iterations <= 500:
            problems.append("max_iterations must be in [1, 500]")
        if not 0.0 < self.relaxation <= 1.0:
            problems.append("relaxation must be in (0, 1]")
        if self.h_ground < 0:
            problems.append("h_ground must be >= 0")
        return problems


@dataclass
class LossesResult:
    """Converged operating point of the losses analysis."""

    converged: bool
    iterations: int
    power: float                        # [W] injected to hold the target
    power_density: float                # [W/m^3] over the storage volume
    t_mean_storage: float               # [K]
    balance: Balance = None
    history: list[dict[str, float]] = field(default_factory=list)


def solve_losses(mesh: Mesh3D, config: LossesConfig, solver_config: SolverConfig = None,
                 progress: Callable[[int, str], None] | None = None,
                 should_stop: Callable[[], bool] | None = None) -> LossesResult:
    """Find the heater power that keeps the sand at ``config.t_target``."""
    problems = config.validate()
    if problems:
        raise ValueError("invalid losses configuration: " + "; ".join(problems))
    sand = mesh.material_id == int(MaterialID.SAND)
    if not sand.any():
        raise ValueError("the mesh has no storage (SAND) cells: build the geometry first")
    if config.h_ground > 0:
        mesh.set_convection_bc("z_min", config.h_ground, config.t_ground)

    solver_config = solver_config or SolverConfig(method="direct")
    v_sand = float(np.sum(mesh.V[sand]))
    q_current = max(config.initial_density, 0.0) * v_sand
    q_prev: float | None = None
    t_prev: float | None = None
    history: list[dict[str, float]] = []
    converged = False
    iteration = 0

    for iteration in range(1, config.max_iterations + 1):
        if should_stop is not None and should_stop():
            break
        mesh.Q_source.fill(0.0)
        mesh.source_mask.fill(False)
        mesh.source_mask[sand] = True
        mesh.Q_source[sand] = q_current / v_sand
        result = SteadyStateSolver(mesh, solver_config).solve()
        t_mean = float(mesh.T[sand].mean())
        error = t_mean - config.t_target
        history.append({"iteration": iteration, "power": q_current,
                        "t_mean_storage": t_mean, "error": error})
        if progress:
            progress(int(100 * iteration / config.max_iterations),
                     f"iter {iteration}: P={q_current / 1000:.1f} kW, "
                     f"T={t_mean - 273.15:.1f} degC, err={error:+.2f} K")
        if abs(error) <= config.tolerance:
            converged = True
            break
        if not result.converged:
            break

        if q_prev is not None and t_prev is not None and abs(q_current - q_prev) > 1e-9:
            slope = (t_mean - t_prev) / (q_current - q_prev)
            if abs(slope) > 1e-9:
                q_new = q_current - error / slope
            else:
                q_new = q_current * (1.0 - 0.1 * float(np.sign(error)))
        else:
            q_new = q_current * (1.0 - 0.02 * float(np.sign(error)) or 1.02)

        q_prev, t_prev = q_current, t_mean
        q_new = max(q_new, 0.0)
        q_current = max(q_current + config.relaxation * (q_new - q_current), 1.0)

    balance = compute_balance(mesh, config.t_ambient)
    return LossesResult(converged=converged, iterations=iteration, power=q_current,
                        power_density=q_current / v_sand,
                        t_mean_storage=float(mesh.T[sand].mean()),
                        balance=balance, history=history)
