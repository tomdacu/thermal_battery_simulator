"""Core domain model: mesh, materials, geometry, time profiles."""
from .materials import MATERIALS, MaterialManager, ThermalProperties
from .mesh import BoundaryType, FaceBC, MaterialID, Mesh3D, NodeProperties
from .geometry import (
    BatteryGeometry,
    BuildReport,
    CylinderGeometry,
    HeaterConfig,
    HeaterPattern,
    TubeConfig,
    TubeElement,
    TubePattern,
    create_small_test_geometry,
)
from .profiles import ExtractionProfile, InitialCondition, PowerProfile

__all__ = [
    "MATERIALS", "MaterialManager", "ThermalProperties",
    "BoundaryType", "FaceBC", "MaterialID", "Mesh3D", "NodeProperties",
    "BatteryGeometry", "BuildReport", "CylinderGeometry", "HeaterConfig",
    "HeaterPattern", "TubeConfig", "TubeElement", "TubePattern",
    "create_small_test_geometry",
    "ExtractionProfile", "InitialCondition", "PowerProfile",
]
