"""
Package solver - Solutori numerici per l'equazione del calore
"""

from .matrix_builder import build_steady_state_matrix, build_transient_matrix
from .steady_state import SteadyStateSolver, SolverConfig, SolverResult, solve_steady_state, set_num_threads

__all__ = [
    'build_steady_state_matrix',
    'build_transient_matrix',
    'SteadyStateSolver',
    'SolverConfig',
    'SolverResult',
    'solve_steady_state',
    'set_num_threads',
]
