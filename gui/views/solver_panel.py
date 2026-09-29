"""Solver settings: accuracy in kelvin, threads, radiation, the standby iteration.

The linear method is not a user choice: every operator the application assembles is
symmetric once the cell volumes scale it (``src/solver/linear.py``), so it is solved by
conjugate gradients with an algebraic-multigrid (Ruge-Stuben) preconditioner - the
fastest of the measured options on the default model.  The linear layer falls back on
its own (to BiCGSTAB on a non-symmetric operator, to Jacobi without PyAMG) and says so
in the log.

Accuracy is asked the way it is judged, in kelvin: the coupled iterations (the gas loop,
radiation) and the standby target stop when the temperature moves by less than it.  The
linear residual sits well below that on its own.
"""
from __future__ import annotations

import os

from PySide6.QtWidgets import QWidget

from ..widgets import FormPanel, combo, double_spin, int_spin

#: the linear method of every analysis (see the module docstring)
METHOD = "cg"
PRECONDITIONER = "amg_rs"

_LINEAR = (
    ("1e-5 (fast)", 1e-5),
    ("1e-6 (default)", 1e-6),
    ("1e-8 (reference)", 1e-8),
)


class SolverPanel(QWidget):
    """Accuracy, threads, radiation and the standby iteration (no physics here)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        cores = os.cpu_count() or 1
        compute_items = [("Auto (all cores)", 0),
                         (f"All - 1  ({max(cores - 1, 1)} of {cores})", -1),
                         ("2 cores", min(2, cores)),
                         ("1 core", 1)]

        self.page = FormPanel()
        accuracy = self.page.section(
            "Accuracy",
            "Linear solver: conjugate gradients + AMG (Ruge-Stuben) on the "
            "volume-symmetrised operator - the fastest option measured on the default "
            "model.  The temperature tolerance is what stops the coupled iterations (gas "
            "loop, radiation) and the standby target; the linear residual only has to sit "
            "below it, and 1e-6 already puts the field within hundredths of a kelvin.")
        self.temperature_tolerance = accuracy.add("Temperature tolerance [K]", double_spin(
            0.1, 0.01, 5.0, 0.05, 2,
            tooltip="The coupled iterations stop when the field moves by less than this, "
                    "and the standby holds its target to within it"))
        self.tolerance = accuracy.add("Linear residual", combo(
            _LINEAR, 1, tooltip="Relative residual of every linear solve"))
        self.max_iterations = accuracy.add("Max linear iterations", int_spin(
            2000, 100, 200000, 500))
        self.losses_max_iterations = accuracy.add("Max standby iterations", int_spin(
            20, 1, 200, 1, tooltip="The standby is a linear problem in the power: it "
                                   "needs two or three"))
        other = self.page.section("Performance and physics")
        self.compute = other.add("Threads", combo(
            compute_items, 1, tooltip="The BLAS/OpenMP pool the solves may use"))
        self.radiation = other.add("Radiation", combo(
            (("Off", False), ("On", True)),
            tooltip="Linearised radiation of the outer surface, added to the convective "
                    "film and re-evaluated on the field it drives"))

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
            "picard_tolerance": float(self.temperature_tolerance.value()),
        }

    def losses_settings(self) -> dict:
        return {
            "tolerance": float(self.temperature_tolerance.value()),
            "max_iterations": self.losses_max_iterations.value(),
        }
