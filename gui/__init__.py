"""GUI package: modular panels, a controller and a thin main window."""
from . import _binding  # noqa: F401  - pins QT_API before QtPy/PyVista import Qt
from .main_window import ThermalBatteryGUI, main

__all__ = ["ThermalBatteryGUI", "main"]
