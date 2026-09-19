"""Solver settings panel: linear method, preconditioner, threading, losses controls."""
from __future__ import annotations

import os

from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from ..widgets import FormPanel, combo, double_spin, int_spin

_METHODS = (
    ("BiCGSTAB (robust)", "bicgstab"),
    ("Conjugate gradient (symmetric only)", "cg"),
    ("GMRES", "gmres"),
    ("Direct LU (small meshes)", "direct"),
)
_PRECONDITIONERS = (
    ("Jacobi (fast, default)", "jacobi"),
    ("None", "none"),
    ("ILU", "ilu"),
    ("AMG Ruge-Stuben", "amg_rs"),
    ("AMG Smoothed aggregation", "amg_sa"),
)
_TOLERANCES = (
    ("1e-10 (high precision)", 1e-10),
    ("1e-8 (default)", 1e-8),
    ("1e-6 (fast)", 1e-6),
    ("1e-4 (very fast)", 1e-4),
)


class SolverPanel(QWidget):
    """Linear solver + losses-iteration controls (no physics here)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        cores = os.cpu_count() or 1
        compute_items = [("Auto (all cores)", 0),
                         (f"All - 1  ({max(cores - 1, 1)} of {cores})", -1),
                         ("2 cores", min(2, cores)),
                         ("1 core", 1)]

        solver = FormPanel()
        self.method = solver.add("Method", combo(_METHODS, 0))
        self.preconditioner = solver.add("Preconditioner", combo(_PRECONDITIONERS, 0))
        self.tolerance = solver.add("Tolerance", combo(_TOLERANCES, 1))
        self.max_iterations = solver.add("Max iterations", int_spin(5000, 100, 200000, 500))
        self.compute = solver.add("Threads", combo(compute_items, 1))
        self.radiation = solver.add("Radiation", combo((("Off", False), ("On", True))))
        solver.add_hint("CG is refused on non-symmetric systems (Dirichlet rows): the "
                        "solver reports the switch instead of diverging silently.")
        self.tabs.addTab(solver, "Solver")

        losses = FormPanel()
        self.losses_tolerance = losses.add("T tolerance [K]",
                                           double_spin(1.0, 0.05, 20.0, 0.1, 2))
        self.losses_max_iterations = losses.add("Max iterations", int_spin(20, 1, 200, 1))
        self.losses_relaxation = losses.add("Under-relaxation", double_spin(0.7, 0.1, 1.0, 0.05, 2))
        self.losses_initial_density = losses.add("Initial power density [W/m³]",
                                                 double_spin(100.0, 1.0, 10000.0, 10.0, 1))
        self.losses_h_ground = losses.add("Ground h [W/(m²·K)]",
                                          double_spin(5.0, 0.0, 1000.0, 1.0, 1,
                                                      tooltip="0 keeps the fixed-temperature ground"))
        losses.add_hint("The losses analysis searches the heater power that holds the "
                        "target storage temperature.")
        self.tabs.addTab(losses, "Losses")

    # -------------------------------------------------------------- accessors
    def threads(self) -> int:
        """Thread budget passed to the BLAS/OMP pool (0 = all cores, -1 = all but one)."""
        return int(self.compute.currentData())

    def settings(self) -> dict:
        return {
            "method": self.method.currentData(),
            "preconditioner": self.preconditioner.currentData(),
            "tolerance": float(self.tolerance.currentData()),
            "max_iterations": self.max_iterations.value(),
            "radiation": bool(self.radiation.currentData()),
        }

    def losses_settings(self) -> dict:
        return {
            "tolerance": self.losses_tolerance.value(),
            "max_iterations": self.losses_max_iterations.value(),
            "relaxation": self.losses_relaxation.value(),
            "initial_density": self.losses_initial_density.value(),
            "h_ground": self.losses_h_ground.value(),
        }
