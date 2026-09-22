"""Solver settings panel: tolerance, threading, radiation, losses controls.

The linear method is not a user choice: every operator the application assembles is
symmetric once the cell volumes scale it (``src/solver/linear.py``), so it is solved by
conjugate gradients with an algebraic-multigrid (Ruge-Stuben) preconditioner - the
fastest of the measured options on the default model and the one that converges in a
handful of iterations whatever the mesh.  The linear layer falls back on its own (to
BiCGSTAB on a non-symmetric operator, to Jacobi without PyAMG) and says so in the log.
"""
from __future__ import annotations

import os

from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from ..widgets import FormPanel, combo, double_spin, int_spin

#: the linear method of every analysis (see the module docstring)
METHOD = "cg"
PRECONDITIONER = "amg_rs"

_TOLERANCES = (
    ("1e-10 (high precision)", 1e-10),
    ("1e-8 (default)", 1e-8),
    ("1e-6 (fast)", 1e-6),
)


class SolverPanel(QWidget):
    """Linear tolerances + losses-iteration controls (no physics here)."""

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
        self.tolerance = solver.add("Tolerance", combo(_TOLERANCES, 1))
        self.max_iterations = solver.add("Max iterations", int_spin(2000, 100, 200000, 500))
        self.compute = solver.add("Threads", combo(compute_items, 1))
        self.radiation = solver.add("Radiation", combo((("Off", False), ("On", True))))
        self.radiation.setToolTip("Linearised radiation of the outer surface, added to the "
                                  "convective film and re-evaluated on the field it drives")
        solver.add_hint("Linear solver: conjugate gradients + AMG (Ruge-Stuben) on the "
                        "volume-symmetrised operator.  It is the fastest option measured "
                        "on the default model and needs a handful of iterations; a "
                        "fallback, if one is ever needed, is reported in the log.")
        self.tabs.addTab(solver, "Solver")

        losses = FormPanel()
        self.losses_tolerance = losses.add("T tolerance [K]",
                                           double_spin(1.0, 0.05, 20.0, 0.1, 2))
        self.losses_max_iterations = losses.add("Max iterations", int_spin(20, 1, 200, 1))
        self.losses_relaxation = losses.add("Under-relaxation", double_spin(0.7, 0.1, 1.0, 0.05, 2))
        self.losses_initial_density = losses.add("Initial power density [W/m³]",
                                                 double_spin(100.0, 1.0, 10000.0, 10.0, 1))
        losses.add_hint("The losses analysis searches the resistors' power that holds the "
                        "target storage temperature, with the gas loop and the model "
                        "exactly as the steady run sees them.")
        self.tabs.addTab(losses, "Losses")

    # -------------------------------------------------------------- accessors
    def threads(self) -> int:
        """Thread budget passed to the BLAS/OMP pool (0 = all cores, -1 = all but one)."""
        return int(self.compute.currentData())

    def settings(self) -> dict:
        return {
            "method": METHOD,
            "preconditioner": PRECONDITIONER,
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
        }
