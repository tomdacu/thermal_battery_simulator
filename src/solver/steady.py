"""Steady-state heat solver, on either mesh.

Optional radiation makes the problem non-linear (the linearised coefficient
depends on the surface temperature); it is solved by a Picard sweep that rebuilds
the operators until the field stops moving.  The gas loop is coupled the same way:
with a :class:`~src.solver.fluid.FluidLoop` every sweep marches the gas on the current
field, hands the exchange to the solid as an implicit film on the pipe cells and solves
again, so the steady state is the plant's own - the resistors' power entering the bed
through the pipe walls - and not a source spread over the sand.  The outer film of an excluded-air
model is non-linear in the same way, and for the same reason: its radiative share
is evaluated on the surface temperature the film itself drives, so every sweep
re-evaluates it and the field and the film converge together.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..core.mesh import Mesh3D
from ..units import check_kelvin
from .fluid import hold_loop_balance
from .linear import LinearConfig, PreconditionerCache, set_num_threads, solve_linear
from .matrix import GridIndex, build_steady_matrix

if TYPE_CHECKING:
    from .fluid import FluidLoop, FluidResult

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from ..core.adaptive_mesh import AdaptiveMesh


@dataclass
class SolverConfig(LinearConfig):
    """Linear settings + the non-linear/steady-specific options."""

    radiation: bool = False
    max_picard: int = 60
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
    """Solve ``div(k grad T) + Q = 0`` with Robin/Dirichlet/Neumann faces.

    Two meshes, one result: a structured :class:`~src.core.mesh.Mesh3D` is assembled by
    :func:`src.solver.matrix.build_steady_matrix` and handed to the linear layer, while an
    adaptive mesh assembles and solves itself (:meth:`AdaptiveMesh.solve_steady`) with the
    same linear layer underneath.  The driver owns what the two have in common: the
    radiation Picard loop (the radiative coefficient depends on the field it drives, and
    so does the outer film it joins), the warm start, the progress callback and the
    :class:`SolverResult` vocabulary - ``T`` is the field flat in the mesh's own cell
    order, leaves included.
    """

    def __init__(self, mesh: Mesh3D | AdaptiveMesh, config: SolverConfig = None,
                 fluid_loop: FluidLoop | None = None) -> None:
        self.mesh = mesh
        self.config = config or SolverConfig()
        #: the gas loop that carries the heat, marched once per sweep (None: the mesh's
        #: own sources and films are the whole problem)
        self.fluid_loop = fluid_loop
        self.fluid_result: FluidResult | None = None
        self.threads = set_num_threads(self.config.n_threads)
        self.adaptive = not isinstance(mesh, Mesh3D)
        if self.adaptive and not hasattr(mesh, "solve_steady"):
            raise TypeError(f"unsupported mesh {type(mesh).__name__}: a Mesh3D or an "
                            f"adaptive mesh is required")
        self.index = None if self.adaptive else GridIndex.from_mesh(mesh)
        self._cache = PreconditionerCache()
        # per-volume coefficients are symmetric only on a uniform grid: hand the
        # cell volumes to the iterative solver so it can symmetrise (see solve_linear).
        # A tree does the same inside solve_steady, where its own cell sizes are read.
        self._scale = None
        if not self.adaptive and not mesh.uniform:
            self._scale = mesh.V.ravel(order="F")
        self._x0: np.ndarray | None = None

    def solve(self, rebuild: bool = True) -> SolverResult:
        """Solve the current mesh state; warm-starts from the previous solution."""
        self.mesh.validate(check_temperature=not self.config.radiation)
        notes = []
        t_start = time.perf_counter()
        # warm start: the previous solution, or the field the mesh carries
        x = (self._x0 if self._x0 is not None
             else np.asarray(self.mesh.T, dtype=float).ravel(order="F").copy())
        residual, iterations = np.inf, 0
        converged = False
        nonlinear = self.config.radiation or self.fluid_loop is not None

        for sweep in range(1, self.config.max_picard + 1):
            iterations = sweep
            if self.fluid_loop is not None:
                self._march_loop()
            x_new, residual, converged = self._sweep(x, notes)
            change = float(np.max(np.abs(x_new - x))) if x_new.size else 0.0
            x = x_new
            self.mesh.T = x if self.adaptive else self.mesh.unflatten_field(x)
            if self.config.progress_callback:
                self.config.progress_callback(int(100 * sweep / self.config.max_picard),
                                              f"sweep {sweep}, dT={change:.3g} K")
            if not converged:
                break
            if not nonlinear or change <= self.config.picard_tolerance:
                break
        else:
            if nonlinear:
                # ``converged`` stays the linear layer's verdict: a caller may ask for a
                # fixed number of sweeps on purpose, and the note says what is left
                notes.append(f"the coupling did not settle in {self.config.max_picard} "
                             f"sweeps (last change {change:.3g} K)")

        check_kelvin(x, "steady solution")
        self._x0 = x
        return SolverResult(
            T=x, converged=converged, residual=float(residual), iterations=iterations,
            solve_time=time.perf_counter() - t_start, notes=notes,
            stats=self.temperature_stats(x),
        )

    def _march_loop(self) -> None:
        """March the gas on the current field and write its film onto the mesh."""
        result = self.fluid_loop.solve(self.mesh)
        self.fluid_result = result
        active = result.apply(self.mesh)
        if self.index is not None and np.any(np.asarray(active).ravel(order="F")
                                             & ~self.index.interior_tube):
            # the structured index lists the film cells once: a cell the loop reaches
            # must be in it, or its exchange would be dropped from the assembly
            self.index = GridIndex.from_mesh(self.mesh)

    def _sweep(self, x: np.ndarray, notes: list) -> tuple[np.ndarray, float, bool]:
        """One assembly + linear solve of the current state, on either mesh.

        Both meshes go through the same linear layer with the solver's own cache, so a
        sweep that leaves the operator unchanged (the gas loop only moves the film
        temperatures, on the right-hand side) reuses the AMG hierarchy.  With a loop whose
        inlet follows its balance, the sweep deposits exactly the loop's external power
        (:func:`~src.solver.fluid.hold_loop_balance`), so the Picard iteration only has to
        settle how the gas distributes it along the pipes.
        """
        if self.adaptive:
            matrix, rhs = self.mesh.assemble(radiation=self.config.radiation)
            scale = self.mesh.volume_scale()
        else:
            matrix, rhs = build_steady_matrix(self.mesh, index=self.index,
                                              radiation=self.config.radiation)
            scale = self._scale
        result = solve_linear(matrix, rhs, self.config, x0=x, cache=self._cache,
                              scale=scale)
        notes.extend(n for n in result.notes if n not in notes)
        field = np.asarray(result.T, dtype=float)
        if self.fluid_result is not None:
            # imported here: the analysis package imports this module
            from ..analysis.fluxes import pinned_cells

            field = hold_loop_balance(
                self.fluid_result, self.mesh, field,
                lambda e: solve_linear(matrix, e, self.config, cache=self._cache,
                                       scale=scale).T,
                pinned_cells(self.mesh, self.index))
        return field, result.residual, result.converged

    def temperature_stats(self, T_flat: np.ndarray = None) -> dict[str, float]:
        values = np.asarray(self.mesh.T if T_flat is None else T_flat, dtype=float)
        return {
            "T_min": float(values.min()),
            "T_max": float(values.max()),
            "T_mean": float(values.mean()),
            "T_median": float(np.median(values)),
        }


def solve_steady_state(mesh: Mesh3D | AdaptiveMesh, method: str = "direct",
                       radiation: bool = False, verbose: bool = False) -> SolverResult:
    """One-shot convenience wrapper."""
    cfg = SolverConfig(method=method, radiation=radiation, verbose=verbose)
    return SteadyStateSolver(mesh, cfg).solve()
