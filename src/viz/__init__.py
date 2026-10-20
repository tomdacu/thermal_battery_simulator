"""
src.viz
========
Visualization helpers shared by the GUI and scripts.

WHAT IT DOES
------------
Re-exports the scene helpers: grid construction, field lookup and colour
limits, material names and colours, the geometry preview and the VTK/CSV
exports.
"""
from .scene import (
    MATERIAL_COLORS,
    MATERIAL_NAMES,
    color_limits,
    export_csv,
    export_vtk,
    field_values,
    to_image_data,
)

__all__ = ["MATERIAL_COLORS", "MATERIAL_NAMES", "color_limits", "export_csv",
           "export_vtk", "field_values", "to_image_data"]
