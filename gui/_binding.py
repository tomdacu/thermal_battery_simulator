"""
gui._binding
============
The Qt binding of the application, pinned before anything can choose another one.

WHAT IT DOES
------------
Imports no Qt module: it only sets the ``QT_API`` environment variable to
``pyside6``.  ``gui/__init__`` imports this module first, ahead of any QtPy or
PyVista import.

WHY
---
The GUI is built on **PySide6** (Qt for Python, LGPL): the application code
imports it directly, but PyVista and pyvistaqt reach Qt through **QtPy**, which
picks a binding on its own - and would pick a PyQt6 that happens to be installed
in the same environment.  Two bindings cannot share a process, so the choice is
fixed here, once, before the first ``qtpy`` import.
"""
from __future__ import annotations

import os

os.environ["QT_API"] = "pyside6"
