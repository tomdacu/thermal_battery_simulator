"""
src/io/__init__.py - Modulo I/O per salvataggio e caricamento
"""

from .state_manager import (
    SimulationState,
    StateManager,
    TransientResults
)

__all__ = [
    'SimulationState',
    'StateManager', 
    'TransientResults'
]
