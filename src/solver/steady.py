"""Steady-state heat solver.

Optional radiation makes the problem non-linear (the linearised coefficient
depends on the surface temperature); it is solved by a Picard sweep that rebuilds
the operators until the field stops moving.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections.abc import Callable

import numpy as np

from ..core.mesh import Mesh3D
from ..units import check_kelvin
from .linear import LinearConfig, PreconditionerCache, set_num_threads, solve_linear
from .matrix import GridIndex, build_steady_matrix


@dataclass
class SolverConfig(LinearConfig):
    """Linear settings + the non-linear/steady-specific options."""

    radiation: bool = False
    max_picard: int = 30
    picard_tolerance: float = 1e-3   # [K] max field change between sweeps
    progress_callback: Callable[[int, str], None] | None = None


@dataclass
class SolverResult:
    """Steady solution and its diagnostics."""

    T: np.ndarray                    # flat field [K]
    converged: bool
    residual: float                  # relative linear residual
    iterations: int                  # Picard sweeps performed
    solve_time: float                # [s]
    notes: list = field(default_factory=list)
    stats: dict[str, float] = field(default_factory=dict)

    def temperature_3d(self, mesh: Mesh3D) -> np.ndarray:
        return mesh.unflatten_field(self.T)


class SteadyStateSolver:
    """Solve ``div(k grad T) + Q = 0`` with Robin/Dirichlet/Neumann faces."""

    def __init__(self, mesh: Mesh3D, config: SolverConfig = None) -> None:
        self.mesh = mesh
        self.config = config or SolverConfig()
        self.threads = set_num_threads(self.config.n_threads)
        self.index = GridIndex.from_mesh(mesh)
        self._cache = PreconditionerCache()
        # per-volume coefficients are symmetric only on a uniform grid: hand the
        # cell volumes to the iterative solver so it can symmetrise (see solve_linear)
        self._scale = None if self.mesh.uniform else self.mesh.V.ravel(order="F")
        self._x0: np.ndarray | None = None

    def solve(self, rebuild: bool = True) -> SolverResult:
        """Solve the current mesh state; warm-starts from the previous solution."""
        self.mesh.validate(check_temperature=not self.config.radiation)
        notes = []
        t_start = time.perf_counter()
        x = self._x0 if self._x0 is not None else np.full(self.mesh.N_total, 293.15)
        residual, iterations = np.inf, 0
        converged = False

        for sweep in range(1, self.config.max_picard + 1):
            iterations = sweep
            a, b = build_steady_matrix(self.mesh, index=self.index,
                                       radiation=self.config.radiation)
            result = solve_linear(a, b, self.config, x0=x, cache=self._cache,
                                  scale=self._scale)
            notes.extend(n for n in result.notes if n not in notes)
            x_new = result.T
            residual = result.residual
            converged = result.converged
            change = float(np.max(np.abs(x_new - x))) if x_new.size else 0.0
            x = x_new
            self.mesh.T = self.mesh.unflatten_field(x)
            if self.config.progress_callback:
                self.config.progress_callback(int(100 * sweep / self.config.max_picard),
                                              f"sweep {sweep}, dT={change:.3g} K")
            if not converged:
                break
            if not self.config.radiation or change <= self.config.picard_tolerance:
                break

        check_kelvin(x, "steady solution")
        self._x0 = x
        return SolverResult(
            T=x, converged=converged, residual=float(residual), iterations=iterations,
            solve_time=time.perf_counter() - t_start, notes=notes,
            stats=self.temperature_stats(x),
        )

    def temperature_stats(self, T_flat: np.ndarray = None) -> dict[str, float]:
        field = self.mesh.T if T_flat is None else self.mesh.unflatten_field(T_flat)
        return {
            "T_min": float(field.min()),
            "T_max": float(field.max()),
            "T_mean": float(field.mean()),
            "T_median": float(np.median(field)),
        }


def solve_steady_state(mesh: Mesh3D, method: str = "direct", radiation: bool = False,
                       verbose: bool = False) -> SolverResult:
    """One-shot convenience wrapper."""
    cfg = SolverConfig(method=method, radiation=radiation, verbose=verbose)
    return SteadyStateSolver(mesh, cfg).solve()
