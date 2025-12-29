"""
Package core - Moduli fondamentali per la simulazione
"""

from .mesh import Mesh3D, MaterialID, BoundaryType, NodeProperties
from .materials import MaterialManager, ThermalProperties, MaterialType
from .geometry import (
    BatteryGeometry, CylinderGeometry, 
    HeaterConfig, TubeConfig,
    HeaterPattern, TubePattern,
    HeaterElement, TubeElement,
    create_small_test_geometry
)
from .profiles import (
    PowerProfile, ExtractionProfile, 
    InitialCondition, TransientConfig
)

__all__ = [
    'Mesh3D',
    'MaterialID',
    'BoundaryType',
    'NodeProperties',
    'MaterialManager',
    'ThermalProperties',
    'MaterialType',
    'BatteryGeometry',
    'CylinderGeometry',
    'HeaterConfig',
    'TubeConfig',
    'HeaterPattern',
    'TubePattern',
    'HeaterElement',
    'TubeElement',
    'create_small_test_geometry',
]
