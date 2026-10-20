"""
gui
===
Modular PySide6 front end: panels, a controller and a thin main window.

WHAT IT DOES
------------
Packages the application's user interface.  ``_binding`` pins the Qt binding
before any QtPy or PyVista import can choose another one; the panels in
``gui.views`` own the controls, ``gui.controller`` owns the worker thread, and
``gui.main_window`` wires them together and owns the built mesh.
"""
from . import _binding  # noqa: F401  - pins QT_API before QtPy/PyVista import Qt
from .main_window import ThermalBatteryGUI, main

__all__ = ["ThermalBatteryGUI", "main"]
