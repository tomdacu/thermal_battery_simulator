"""Temperature helpers for the UI layer.

``src/`` works in Kelvin; the GUI shows degC.  Every conversion in the
application goes through these two functions, so a unit mistake has exactly one
place to live.
"""
from __future__ import annotations

import numpy as np

from src.units import c_to_k, k_to_c

__all__ = ["c_to_k", "k_to_c", "fmt_c", "fmt_k", "celsius_span"]


def fmt_c(value_k: float, decimals: int = 1) -> str:
    """Format a Kelvin value as a Celsius string."""
    return f"{k_to_c(value_k):.{decimals}f} °C"


def fmt_k(value_k: float, decimals: int = 2) -> str:
    return f"{value_k:.{decimals}f} K"


def celsius_span(values_k) -> np.ndarray:
    """Whole array converted to Celsius (for statistics and exports)."""
    return np.asarray(values_k, dtype=float) - 273.15
