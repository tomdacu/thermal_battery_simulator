"""
src.solver
==========
Solvers for the packed-bed heat equation: assembly, linear backends, drivers.

WHAT IT DOES
------------
Re-exports the public surface of the package: the finite-difference assembly of
:mod:`src.solver.matrix`, the SciPy linear layer of :mod:`src.solver.linear`,
the steady and transient drivers, and the transient result container.
"""

from ..core.grid import GridIndex
from ..core.physics import half_cell_h, harmonic_mean, radiation_h
from .matrix import (
    build_steady_matrix,
    build_transient_operators,
    steady_rhs,
    transient_rhs,
)
from .linear import (
    LinearConfig,
    LinearResult,
    PreconditionerCache,
    fingerprint,
    is_symmetric,
    set_num_threads,
    solve_linear,
)
from .results import TransientResults
from .steady import SolverConfig, SolverResult, SteadyStateSolver, solve_steady_state
from .transient import TransientConfig, TransientSolver, run_transient_simulation

__all__ = [
    "GridIndex", "build_steady_matrix", "build_transient_operators",
    "half_cell_h", "harmonic_mean", "radiation_h", "steady_rhs", "transient_rhs",
    "LinearConfig", "LinearResult", "PreconditionerCache",
    "fingerprint", "is_symmetric", "set_num_threads", "solve_linear",
    "TransientResults", "SolverConfig", "SolverResult", "SteadyStateSolver",
    "solve_steady_state", "TransientConfig", "TransientSolver",
    "run_transient_simulation",
]
