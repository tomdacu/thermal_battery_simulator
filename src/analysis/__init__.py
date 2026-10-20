"""
src.analysis
============
Flux evaluation, energy balance, losses optimisation, mesh planning and cycle ledger.

WHAT IT DOES
------------
Re-exports the analysis surface: the flux report and the balance identity, the
standby losses solve, the automatic mesh search and the a-priori mesh plan,
and the charge/standby/discharge ledger.
"""
from .balance import (
    Balance,
    compute_balance,
    storage_capacity,
    thermal_autonomy,
)
from .fluxes import (
    destroyed_exergy,
    domain_face_flux,
    domain_fluxes,
    envelope_fluxes,
    stored_energy,
    stored_exergy,
    tube_flux,
)
from .losses import LossesConfig, LossesResult, solve_losses

__all__ = [
    "Balance", "compute_balance", "storage_capacity", "thermal_autonomy",
    "destroyed_exergy", "domain_face_flux", "domain_fluxes", "envelope_fluxes",
    "stored_energy", "stored_exergy", "tube_flux",
    "LossesConfig", "LossesResult", "solve_losses",
]
