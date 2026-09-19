"""Temperature unit helpers.

The package contract (see :mod:`src.constants`) is Kelvin everywhere inside
``src/``. These helpers exist for the two legitimate exceptions:

* reading user input expressed in degC,
* formatting values for display or export through ``gui/units.py``.

``check_kelvin`` is the cheap guard that turns "someone passed degC" from a
silently wrong simulation into an immediate ``ValueError``.
"""
from __future__ import annotations

import numpy as np

from .constants import T_MIN_VALID


def c_to_k(t_c: float) -> float:
    """Celsius -> Kelvin."""
    return float(t_c) + 273.15


def k_to_c(t_k: float) -> float:
    """Kelvin -> Celsius."""
    return float(t_k) - 273.15


def check_kelvin(value, name: str = "temperature", strict: bool = True) -> None:
    """Raise if ``value`` (scalar or array) cannot be a Kelvin temperature.

    Catches the classic degC/K mix-up: a Celsius field has values well below
    ``T_MIN_VALID`` (e.g. 20), which no physical absolute temperature reaches.
    With ``strict=False`` only NaN/inf are rejected.
    """
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: non-finite values")
    if strict and arr.min() < T_MIN_VALID:
        raise ValueError(
            f"{name}: minimum {arr.min():.2f} is below {T_MIN_VALID} K - "
            "the field is probably in degC while the package contract is Kelvin"
        )
