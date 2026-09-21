"""Heat-flux evaluation: one implementation shared by solver, balance and checks.

Every flux is expressed with the same half-cell conductances used by the
assembly (``h_eff = 2kh/(2k+hd)`` for convective faces, ``2k/d`` for a Dirichlet
surface ``d/2`` away from the node centre), so the integrals below close the
discrete energy balance instead of approximating it with a different formula.

Sign convention: positive flux LEAVES the domain / the battery.
"""
from __future__ import annotations


import numpy as np

from ..core.mesh import FACES, BoundaryType, MaterialID, Mesh3D
from ..core.grid import (FACE_AXIS, FACE_SLICES, INNER_SLICES, GridIndex,
                         dirichlet_mask)
from ..core.physics import half_cell_h, radiation_h


def _inner_slice(face: str, shape: tuple) -> tuple:
    """First cell inside the domain, clamped for axes with a single cell."""
    out = list(INNER_SLICES[face])
    for axis, item in enumerate(out):
        if isinstance(item, int):
            n = shape[axis]
            out[axis] = max(min(item if item >= 0 else n + item, n - 1), 0)
    return tuple(out)


def domain_face_flux(mesh: Mesh3D, face: str, radiation: bool = False,
                     index: GridIndex = None) -> float:
    """Heat leaving the domain through one of the six box faces [W].

    Dirichlet faces are pinned to ``T_bc``, so their flux CANNOT be evaluated
    from the boundary node (it would be identically zero - a defect that made the
    ground loss disappear from every report).  The one-sided gradient towards the
    first interior cell is used instead, with the same harmonic-mean face
    conductivity as the assembly.

    A node pinned by *another* Dirichlet face is skipped: its own row is replaced by
    the identity, so the exchange it would have had through this face never enters
    the solution and must not be reported either (it is the ground-to-wall edge of
    the battery).
    """
    bc = mesh.face_bc[face]
    sl = FACE_SLICES[face]
    inner = _inner_slice(face, mesh.T.shape)
    axis = FACE_AXIS[face]
    index = index or GridIndex.from_mesh(mesh)
    area = _face_area(mesh, axis, sl)                 # per-cell face area
    # nodes pinned by another Dirichlet face: their row is the identity, so this
    # face's exchange never reaches the solution
    pinned = dirichlet_mask(mesh, index).reshape(mesh.T.shape, order="F")
    # a cell outside the envelope leaves the problem too: it is pinned at the ambient
    # and its exchange is the environment film, counted by ``environment_flux``.  Not
    # skipping it here reported the convection of the box faces as if it happened and
    # broke the balance by kilowatts on the default vessel.
    excluded = mesh.excluded.reshape(mesh.T.shape, order="F")
    free = ~pinned[sl] & ~excluded[sl]
    free_inner = ~pinned[inner] & ~excluded[inner]
    if bc.kind == BoundaryType.DIRICHLET:
        k_self = mesh.k[inner]
        k_nb = mesh.k[sl]
        k_face = 2.0 * k_self * k_nb / (k_self + k_nb)
        # the boundary *cell* is pinned, so the flux towards it uses exactly the
        # centre-to-centre distance of the assembly (the two half cells), otherwise
        # the reported loss disagrees with the operator on a graded mesh
        d_centers = 0.5 * (mesh.axis_size(axis)[inner] + mesh.axis_size(axis)[sl])
        # the corner cells pinned by *another* Dirichlet face are not part of this
        # face's exchange: counting them breaks the balance on a graded grid
        return float(np.sum(k_face * (mesh.T[inner] - bc.value) / d_centers
                            * area * free_inner))
    if bc.kind == BoundaryType.NEUMANN:
        return float(-bc.value * np.sum(area[free]))
    if bc.kind == BoundaryType.CONVECTION and bc.is_active():
        h = np.full(mesh.k[sl].shape, bc.h)
        if radiation and bc.emissivity > 0:
            h = h + radiation_h(mesh.T[sl], bc.value,
                                np.full(mesh.k[sl].shape, bc.emissivity))
        # the exchange happens on the boundary cell: use its own local size, the
        # same one the assembly puts in the half-cell resistance
        h_size = _face_size(mesh, axis, sl)
        return float(np.sum(half_cell_h(mesh.k[sl], h, h_size) * (mesh.T[sl] - bc.value)
                            * area * free))
    return 0.0


def _face_area(mesh: Mesh3D, axis: int, selector) -> np.ndarray:
    """Face areas of a box face (or of a masked set) for a face normal to ``axis``."""
    return (mesh.Ax, mesh.Ay, mesh.Az)[axis][selector]


def _face_size(mesh: Mesh3D, axis: int, selector) -> np.ndarray:
    """Local cell size along ``axis`` on a box face."""
    return mesh.axis_size(axis)[selector]


def environment_flux(mesh: Mesh3D, index: GridIndex = None) -> float:
    """Heat leaving the *active* region through the outside film [W].

    When the air is excluded from the problem, the outer surface is an internal
    boundary: the box faces see nothing of it, so the loss must be integrated here or
    the energy balance of the domain would not close.
    """
    if mesh.h_out <= 0.0:
        return 0.0
    excluded = mesh.excluded.ravel(order="F")
    if not excluded.any():
        return 0.0
    index = index or GridIndex.from_mesh(mesh)
    temperature = mesh.T.ravel(order="F")
    # a cell whose row has been replaced by the Dirichlet elimination does not carry the
    # film: its temperature is imposed, so counting its exchange here made the reported
    # environment flux fall short (0.77 W on a 5 kW input on the default vessel).
    pinned = dirichlet_mask(mesh, index).reshape(mesh.T.shape, order="F").ravel(
        order="F")
    total = 0.0
    for face_index, nb in enumerate(index.neighbours):
        mask = (~excluded & excluded[nb] & ~pinned
                & ~index.on_face[FACES[face_index]])
        if not mask.any():
            continue
        axis = FACE_AXIS[FACES[face_index]]
        area = index.areas[axis][mask]
        total += float(np.sum(mesh.h_out * area
                              * (temperature[mask] - mesh.t_ambient)))
    return total


def domain_fluxes(mesh: Mesh3D, radiation: bool = False) -> dict[str, float]:
    """``{face: W}`` for the six box faces plus the total."""
    index = GridIndex.from_mesh(mesh)
    out = {face: domain_face_flux(mesh, face, radiation, index) for face in FACES}
    out["total"] = float(sum(out.values()))
    return out


def tube_flux(mesh: Mesh3D, index: GridIndex = None) -> float:
    """Heat leaving the domain into the heat-transfer fluid [W]."""
    index = index or GridIndex.from_mesh(mesh)
    mask = index.interior_tube
    if not mask.any():
        return 0.0
    h = mesh.bc_h.ravel(order="F")[mask]
    t_cell = mesh.T.ravel(order="F")[mask]
    t_fluid = mesh.bc_T_inf.ravel(order="F")[mask]
    if mesh.uniform:
        return float(np.sum(h * (t_cell - t_fluid)) * mesh.V_cell / mesh.d)
    v_flat = mesh.V.ravel(order="F")[mask]
    h_flat = mesh.h_char.ravel(order="F")[mask]
    return float(np.sum(h * (t_cell - t_fluid) * v_flat / h_flat))


def _pair_flux(mesh: Mesh3D, battery: np.ndarray, companion: np.ndarray, axis: int,
               shift: int) -> float:
    """Conductive flux across battery/companion cell pairs along ``axis`` [W]."""
    t_nb = np.roll(mesh.T, shift, axis=axis)
    k_nb = np.roll(mesh.k, shift, axis=axis)
    mask = battery & np.roll(companion, shift, axis=axis)
    edge = [slice(None)] * 3
    edge[axis] = -1 if shift < 0 else 0      # np.roll wraps around: drop that row
    mask[tuple(edge)] = False
    if not mask.any():
        return 0.0
    k_self, k_other = mesh.k[mask], k_nb[mask]
    k_face = 2.0 * k_self * k_other / (k_self + k_other)
    size_self = mesh.axis_size(axis)[mask]
    size_other = np.roll(mesh.axis_size(axis), shift, axis=axis)[mask]
    area = (mesh.Ax, mesh.Ay, mesh.Az)[axis][mask]
    flux = k_face * (mesh.T[mask] - t_nb[mask]) / (0.5 * (size_self + size_other)) * area
    return float(np.sum(flux))


def envelope_fluxes(mesh: Mesh3D) -> dict[str, float]:
    """Flux leaving the *battery* through its outer surface, split by direction.

    The envelope is the interface between battery cells (sand, insulation,
    steel, concrete) and the surrounding air cells inside the domain, evaluated
    with the same harmonic-mean conductance as the assembly.  Being an interface
    integral it does not scale with the size of the air box.
    ``top`` = +z, ``bottom`` = -z, ``side`` = +-x and +-y.
    """
    battery = mesh.material_id != int(MaterialID.AIR)
    companion = ~battery
    top = _pair_flux(mesh, battery, companion, 2, -1)
    bottom = _pair_flux(mesh, battery, companion, 2, 1)
    side = sum(_pair_flux(mesh, battery, companion, axis, shift)
               for axis in (0, 1) for shift in (-1, 1))
    return {"top": top, "bottom": bottom, "side": side, "total": top + bottom + side}


def stored_energy(mesh: Mesh3D, t_reference: float) -> float:
    """Sensible energy stored above ``t_reference`` [J]."""
    return float(np.sum(mesh.rho * mesh.cp * (mesh.T - t_reference) * mesh.V))


def stored_exergy(mesh: Mesh3D, t_reference: float) -> float:
    """Exergy stored above ``t_reference`` [J]: ``sum rho cp V [(T-T0) - T0 ln(T/T0)]``."""
    t = mesh.T
    frac = np.maximum(t / t_reference, 1e-9)
    return float(np.sum(mesh.rho * mesh.cp * ((t - t_reference) - t_reference * np.log(frac))
                        * mesh.V))


def destroyed_exergy(p_input: float, ex_stored: float, t_reference: float,
                     t_source: float) -> float:
    """Exergy destroyed between a source at ``t_source`` and the stored exergy [J]."""
    if t_source <= t_reference:
        return 0.0
    ex_input = p_input * (1.0 - t_reference / t_source)
    return float(max(ex_input - ex_stored, 0.0))
