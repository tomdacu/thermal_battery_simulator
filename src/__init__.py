"""
Package src - Sand Battery Thermal Simulation
"""

from .core import (
    Mesh3D, MaterialID, BoundaryType, NodeProperties,
    MaterialManager, ThermalProperties,
    BatteryGeometry, CylinderGeometry,
    create_small_test_geometry
)

from .solver import (
    SteadyStateSolver, SolverConfig, SolverResult,
    solve_steady_state, build_steady_state_matrix
)

from .analysis import (
    PowerBalanceAnalyzer, PowerBalanceResult, ExergyResult
)

# Visualization opzionale
try:
    from .visualization import (
        BatteryRenderer, VisualizationConfig,
        quick_plot, quick_slice, PYVISTA_AVAILABLE
    )
except ImportError:
    PYVISTA_AVAILABLE = False

__version__ = "0.1.0"
__author__ = "Sand Battery Simulation Team"
