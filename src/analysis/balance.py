"""Energy and exergy bookkeeping - one implementation for the whole application.

Definitions used here (documented once, used everywhere):

* ``p_input``      total power injected [W]: the volumetric sources (``Q_source``)
                   plus the net film of the heat-transfer fluid when it heats the
                   solid - the gas loop charging through the pipe walls.
* ``p_extracted``  power removed [W]: the volumetric sinks plus the net fluid film
                   when it cools the solid.
* ``q_domain``     power leaving through the six box faces [W].  It is computed
                   with the faces and the conductances the assembly used - the mesh's
                   own face list for a tree, the ``GridIndex`` tables for ``Mesh3D``
                   (:mod:`src.analysis.fluxes`) - so
                   ``p_input - p_extracted - q_domain - dE/dt`` closes to
                   numerical precision: that identity is the self-check.
* ``q_battery``    power crossing the battery envelope (sand/insulation/steel/
                   concrete against air) [W] - the physically meaningful loss,
                   which does not scale with the size of the air box.
* ``e_stored``     sensible energy above the ambient [J]; ``ex_stored`` the
                   corresponding exergy.

The same fields are filled from either mesh: only the flux report knows which face list
it is reading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ..constants import T_AMBIENT_DEFAULT
from ..core.mesh import MaterialID, Mesh3D, storage_mask
from ..core.grid import GridIndex
from . import fluxes

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from ..core.adaptive_mesh import AdaptiveMesh


@dataclass
class Balance:
    """Snapshot of the energy balance of a mesh state."""

    t_ambient: float
    p_input: float
    p_extracted: float
    q_domain: float
    q_domain_top: float
    q_domain_bottom: float
    q_domain_side: float
    q_battery: float
    q_battery_top: float
    q_battery_bottom: float
    q_battery_side: float
    e_stored: float
    ex_stored: float
    ex_destroyed: float
    t_mean_storage: float
    t_max: float
    t_min: float
    t_mean_insulation: float
    t_mean_shell: float
    imbalance: float
    eta_energy: float | None = None
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return asdict(self)

    def report(self) -> dict[str, float]:
        """GUI/CSV friendly scalars in SI plus the kWh convenience keys."""
        return {
            "P_input_W": self.p_input,
            "P_extracted_W": self.p_extracted,
            "Q_losses_W": self.q_battery,
            "Q_losses_domain_W": self.q_domain,
            "Q_losses_top_W": self.q_battery_top,
            "Q_losses_bottom_W": self.q_battery_bottom,
            "Q_losses_side_W": self.q_battery_side,
            "E_stored_J": self.e_stored,
            "E_stored_kWh": self.e_stored / 3.6e6,
            "Ex_stored_J": self.ex_stored,
            "Ex_destroyed_J": self.ex_destroyed,
            "T_mean_storage_K": self.t_mean_storage,
            "T_max_K": self.t_max,
            "T_min_K": self.t_min,
            "T_mean_insulation_K": self.t_mean_insulation,
            "T_mean_shell_K": self.t_mean_shell,
            "eta_energy": self.eta_energy if self.eta_energy is not None else float("nan"),
            "imbalance_W": self.imbalance,
        }


def _mask_mean(mesh: Mesh3D | AdaptiveMesh, mask: np.ndarray) -> float:
    return float(mesh.T[mask].mean()) if mask.any() else float("nan")


def compute_balance(mesh: Mesh3D | AdaptiveMesh, t_ambient: float = T_AMBIENT_DEFAULT,
                    index: GridIndex = None, dE_dt: float = 0.0,
                    t_source: float | None = None, radiation: bool = False) -> Balance:
    """Evaluate the energy balance of the current mesh state.

    Either mesh: ``Mesh3D`` reads its ``GridIndex`` tables (the face list its assembly was
    built from) and an adaptive mesh its own ``faces()``, and the flux report of
    :mod:`src.analysis.fluxes` dispatches between them - so the fields below have the
    same meaning on both and the identity ``p_input - p_extracted - q_domain - dE/dt``
    closes at the round-off of the terms on either.

    ``index`` is the structured index of the caller (``SteadyStateSolver.index``,
    ``TransientSolver.index``); a tree needs none.  ``dE_dt`` [W] is the rate of change
    of stored energy; supply it in a transient run so that :attr:`Balance.imbalance`
    stays meaningful.  ``radiation`` must match the setting used by the solver, otherwise
    the reported surface flux would use coefficients the solve never applied.
    """
    index = fluxes.structured_index(mesh, index)
    q_dom = fluxes.domain_fluxes(mesh, radiation=radiation, index=index)
    # the outside film of an excluded-air model is an internal boundary: it belongs to
    # the domain loss, otherwise the balance identity would not close
    environment = fluxes.environment_flux(mesh, index)
    q_dom["total"] += environment
    q_bat = fluxes.envelope_fluxes(mesh, index)
    q_bat["total"] += environment

    # a source (or a sink) inside a cell the elimination pinned never reaches the solver:
    # counting it as input made the reported balance miss the energy the identity rows
    # drop.  A Dirichlet face pins its cells, an environment model the excluded ones.
    free = ~fluxes.pinned_cells(mesh, index)
    q_source = fluxes.as_flat(mesh.Q_source)
    q_sink = fluxes.as_flat(mesh.Q_sink)
    volume = fluxes.as_flat(mesh.V)
    # the fluid film counts where its net rate goes: into the solid it is input (the
    # gas loop charging through the pipe walls), out of it extraction
    fluid = fluxes.tube_flux(mesh, index)
    p_input = float(np.sum(q_source[free] * volume[free])) + max(-fluid, 0.0)
    p_extracted = float(-np.sum(q_sink[free] * volume[free])) + max(fluid, 0.0)

    e_stored = fluxes.stored_energy(mesh, t_ambient)
    ex_stored = fluxes.stored_exergy(mesh, t_ambient)

    battery = mesh.material_id != int(MaterialID.AIR)
    sand = storage_mask(mesh.material_id)
    insulation = mesh.material_id == int(MaterialID.INSULATION)
    steel = mesh.material_id == int(MaterialID.STEEL)

    if t_source is None:
        t_source = _mask_mean(mesh, sand) if sand.any() else _mask_mean(mesh, battery)
    ex_dest = fluxes.destroyed_exergy(p_input, ex_stored, t_ambient, t_source)

    imbalance = p_input - p_extracted - q_dom["total"] - dE_dt
    eta = 1.0 - q_bat["total"] / p_input if p_input > 0 else None

    return Balance(
        t_ambient=t_ambient,
        p_input=p_input, p_extracted=p_extracted,
        q_domain=q_dom["total"], q_domain_top=q_dom["z_max"],
        q_domain_bottom=q_dom["z_min"], q_domain_side=q_dom["x_min"] + q_dom["x_max"]
        + q_dom["y_min"] + q_dom["y_max"],
        q_battery=q_bat["total"], q_battery_top=q_bat["top"],
        q_battery_bottom=q_bat["bottom"], q_battery_side=q_bat["side"],
        e_stored=e_stored, ex_stored=ex_stored, ex_destroyed=ex_dest,
        t_mean_storage=_mask_mean(mesh, sand),
        t_max=float(mesh.T[battery].max()) if battery.any() else float("nan"),
        t_min=float(mesh.T[battery].min()) if battery.any() else float("nan"),
        t_mean_insulation=_mask_mean(mesh, insulation), t_mean_shell=_mask_mean(mesh, steel),
        imbalance=imbalance, eta_energy=eta,
    )


def storage_capacity(mesh: Mesh3D | AdaptiveMesh, t_max: float, t_ambient: float = T_AMBIENT_DEFAULT) -> float:
    """Sensible storage capacity of the sand region between ambient and ``t_max`` [J]."""
    sand = storage_mask(mesh.material_id)
    if not sand.any():
        return 0.0
    return float(np.sum(mesh.rho[sand] * mesh.cp[sand] * mesh.V[sand])
                 * max(t_max - t_ambient, 0.0))


def thermal_autonomy(mesh: Mesh3D | AdaptiveMesh, t_ambient: float = T_AMBIENT_DEFAULT,
                     index: GridIndex = None) -> dict[str, float]:
    """Hours the stored energy can cover the current loss rate.

    The denominator is the loss of the *whole domain*: a Dirichlet face (the ground)
    drains the store as much as the air does, and using the envelope-only rate
    overstates the autonomy by the share of that path.
    """
    bal = compute_balance(mesh, t_ambient, index=index)
    loss = bal.q_domain if bal.q_domain > 0 else bal.q_battery
    hours = bal.e_stored / loss / 3600.0 if loss > 0 else float("inf")
    return {"hours": hours, "loss_W": loss, "stored_J": bal.e_stored}
