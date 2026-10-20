"""
src.io
======
State persistence (HDF5).

WHAT IT DOES
------------
Re-exports the state manager, the stored snapshot and its geometry hash, so a
caller reads and writes a state through the package surface.
"""
from .state import (
    FORMAT_VERSION,
    StateError,
    StateManager,
    SimulationState,
    geometry_hash,
)

__all__ = ["FORMAT_VERSION", "StateError", "StateManager", "SimulationState", "geometry_hash"]
