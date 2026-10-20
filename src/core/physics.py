"""
src.core.physics
================
Interface coefficients shared by the assembler and the analysis.

WHAT IT DOES
------------
Provides the coefficients that turn a temperature difference into a heat rate
across an interface: the face conductivity between two cells, the effective
film coefficient of a convective surface on a cell, and the linearised
radiative coefficient.

WHY
---
Kept in :mod:`src.core` so that ``src.analysis`` never has to import
``src.solver`` (the solver package imports the analysis for its energy
bookkeeping, and a cycle would break package initialisation).

HOW IT WORKS
------------
-> HARMONIC MEAN <-
   The face conductivity between two cell centres is the harmonic mean of two
   half cells in series, ``k_face = 2 k_a k_b / (k_a + k_b)``.

-> HALF-CELL FILM <-
   A convective surface sits ``d/2`` away from the cell centre, so the film and
   the half-cell conduction are in series: ``h_eff = 2kh / (2k + hd)``.

-> RADIATION <-
   The linearised radiative coefficient is
   ``eps*sigma*(Ts+Tinf)*(Ts^2+Tinf^2)``.

LIMITS
------
``EPS`` is added to each denominator, so a coefficient stays finite when a
conductivity or a film coefficient is zero.
"""
from __future__ import annotations

import numpy as np

from ..constants import EPS, SIGMA


def harmonic_mean(k_a, k_b):
    """Face conductivity between two cell centres (two half cells in series)."""
    return 2.0 * np.asarray(k_a) * np.asarray(k_b) / (np.asarray(k_a) + np.asarray(k_b) + EPS)


def half_cell_h(k, h, d: float):
    """Effective surface conductance of a half cell [W/(m^2*K)].

    A convective surface sits ``d/2`` away from the cell centre, so the film and
    the half-cell conduction are in series: ``h_eff = 2kh / (2k + hd)``.
    """
    return 2.0 * np.asarray(k) * np.asarray(h) / (2.0 * np.asarray(k)
                                                  + np.asarray(h) * d + EPS)


def radiation_h(t_surface, t_inf: float, emissivity):
    """Linearised radiative coefficient ``eps*sigma*(Ts+Tinf)*(Ts^2+Tinf^2)``."""
    ts = np.asarray(t_surface, dtype=float)
    return np.asarray(emissivity, dtype=float) * SIGMA * (ts + t_inf) * (
        ts * ts + t_inf * t_inf)
