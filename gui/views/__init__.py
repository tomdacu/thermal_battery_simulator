"""
gui.views
=========
GUI panels: geometry, materials, analysis, solver, results and the 3D view.

WHAT IT DOES
------------
Exposes the six panel classes the window assembles: ``GeometryPanel``,
``MaterialsPanel``, ``AnalysisPanel``, ``SolverPanel``, ``ResultsPanel`` and
``VizView``.
"""
from .analysis_panel import AnalysisPanel
from .geometry_panel import GeometryPanel
from .materials_panel import MaterialsPanel
from .results_panel import ResultsPanel
from .solver_panel import SolverPanel
from .viz_view import VizView

__all__ = ["AnalysisPanel", "GeometryPanel", "MaterialsPanel", "ResultsPanel",
           "SolverPanel", "VizView"]
