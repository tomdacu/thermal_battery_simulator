"""GUI panels: geometry, materials, analysis, solver, results, 3D view."""
from .analysis_panel import AnalysisPanel
from .geometry_panel import GeometryPanel
from .materials_panel import MaterialsPanel
from .results_panel import ResultsPanel
from .solver_panel import SolverPanel
from .viz_view import VizView

__all__ = ["AnalysisPanel", "GeometryPanel", "MaterialsPanel", "ResultsPanel",
           "SolverPanel", "VizView"]
