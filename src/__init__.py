"""Thermal battery simulator core: mesh, materials, geometry, profiles."""

__version__ = "2.1.0"

from .constants import T0, T_AMBIENT_DEFAULT, T_GROUND_DEFAULT, T_INITIAL_DEFAULT
from .units import c_to_k, k_to_c, check_kelvin
from .core.materials import MATERIALS, MaterialManager, ThermalProperties
from .core.mesh import BoundaryType, FaceBC, MaterialID, Mesh3D, NodeProperties
from .core.geometry import (
    BatteryGeometry,
    BuildReport,
    CylinderGeometry,
    HeaterConfig,
    create_small_test_geometry,
)
from .core.profiles import ExtractionProfile, InitialCondition, PowerProfile

__all__ = [
    "T0", "T_AMBIENT_DEFAULT", "T_GROUND_DEFAULT", "T_INITIAL_DEFAULT",
    "c_to_k", "k_to_c", "check_kelvin",
    "MATERIALS", "MaterialManager", "ThermalProperties",
    "BoundaryType", "FaceBC", "MaterialID", "Mesh3D", "NodeProperties",
    "BatteryGeometry", "BuildReport", "CylinderGeometry", "HeaterConfig",
    "create_small_test_geometry",
    "ExtractionProfile", "InitialCondition", "PowerProfile",
]
