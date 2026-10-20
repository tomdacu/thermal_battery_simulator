"""
src.core
========
Core domain model: mesh, materials, geometry and time profiles.

WHAT IT DOES
------------
Re-exports the public vocabulary of the domain: the structured mesh and its
boundary conditions, the material database and the packed-bed model, the
geometry description and its painters, and the transient driving profiles.
"""
from .materials import MATERIALS, MaterialManager, ThermalProperties
from .mesh import BoundaryType, FaceBC, MaterialID, Mesh3D, NodeProperties
from .geometry import (
    BatteryGeometry,
    BuildReport,
    CylinderGeometry,
    HeaterConfig,
    create_small_test_geometry,
)
from .profiles import ExtractionProfile, InitialCondition, PowerProfile

__all__ = [
    "MATERIALS", "MaterialManager", "ThermalProperties",
    "BoundaryType", "FaceBC", "MaterialID", "Mesh3D", "NodeProperties",
    "BatteryGeometry", "BuildReport", "CylinderGeometry", "HeaterConfig",
    "create_small_test_geometry",
    "ExtractionProfile", "InitialCondition", "PowerProfile",
]
