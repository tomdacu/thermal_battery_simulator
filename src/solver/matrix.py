"""Finite-difference assembly for the steady and transient heat equation.

DISCRETIZATION (per unit volume, coefficients in W/(m^3*K))
-----------------------------------------------------------
Balance over a cell of volume ``V = d^3``::

    sum_faces k_eff * (T_nb - T_P) * A / d + Q * V = 0

Divided by ``V`` (``A = d^2``) it becomes the assembled row::

    a_P * T_P - sum_nb a_nb * T_nb = Q ,   a_nb = k_eff / d^2

``k_eff = 2 k_P k_nb / (k_P + k_nb)`` is the series conductance of the two half
cells.  ``Q`` is a volumetric source [W/m^3] - exactly ``Mesh3D.Q_source``.

Boundary conditions
-------------------
* Each domain face carries its own :class:`FaceBC`: a node on an edge or corner
  receives one contribution **per exposed face**, with that face's ``h``/``T_inf``.
* Convection uses the half-cell series conductance ``h_eff = 2kh/(2k+hd)``
  (the surface sits ``d/2`` away from the node centre); ``a = h_eff/d``.
* Dirichlet rows are replaced by ``T = T_bc`` (exact, not relaxed).
* Neumann adds ``q/d`` to the RHS (``q`` [W/m^2] entering the domain).
* Interior cells with ``boundary_type == CONVECTION`` (heat-exchanger tubes)
  exchange ``h*(T_fluid - T_P)`` with the cell volume -> ``h/d``.
* Radiation is opt-in and linearised: ``h_r = eps*sigma*(Ts+Tinf)*(Ts^2+Tinf^2)``,
  added to the convective coefficient of the exposed face.

TRANSIENT
---------
Backward Euler with the mass matrix ``M = diag(rho*cp)`` [J/(m^3*K)]::

    (M/dt + L) T^{n+1} = (M/dt) T^n + b

``M`` must NOT contain the cell volume: ``L`` is already per unit volume, and a
``V`` factor scales every time constant by ``1/d^3``, making the answer depend on
the mesh resolution.
"""
from __future__ import annotations


import numpy as np
from scipy import sparse

from ..constants import EPS
from ..core.grid import FACE_AXIS, GridIndex, dirichlet_mask
from ..core.physics import half_cell_h, radiation_h
from ..core.mesh import FACES, BoundaryType, Mesh3D

def face_coefficients(mesh: Mesh3D, index: GridIndex) -> np.ndarray:
    """(6, N) face conductances per unit volume, zeroed on the box faces.

    ``a = k_face A / (d_centers V)``: ``k_face`` is the harmonic mean of the two
    conductivities (uniform mesh: the classic ``k_eff/d^2``), ``A`` the face area and
    ``d_centers`` the centre-to-centre distance of the two cells.
    """
    k = mesh.k.ravel(order="F")
    excluded = mesh.excluded.ravel(order="F")
    material = mesh.material_id.ravel(order="F")
    out = np.zeros((6, k.size))
    for face_index, nb in enumerate(index.neighbours):
        k_e = k[nb]
        # harmonic mean of the two conductivities = conductivity of the interface
        k_face = 2.0 * k * k_e / (k + k_e + EPS)
        coeff = k_face * index.face_factors[face_index]
        if mesh.h_contact > 0.0:
            # a real interface between two materials is not perfect: the contact
            # resistance sits in series with the two half cells
            axis = FACE_AXIS[FACES[face_index]]
            d_self = index.sizes[axis]
            d_nb = d_self[nb]
            r_series = (0.5 * d_self / (k + EPS) + 1.0 / mesh.h_contact
                        + 0.5 * d_nb / (k_e + EPS))
            k_eff = 0.5 * (d_self + d_nb) / r_series
            coeff = np.where(material != material[nb], k_eff * index.face_factors[face_index],
                             coeff)
        coeff[index.on_face[FACES[face_index]]] = 0.0
        if excluded.any():
            # the excluded cells are not part of the problem: no conduction into them,
            # their interface is a film (added in build_steady_matrix)
            coeff = np.where(excluded[nb], 0.0, coeff)
        out[face_index] = coeff
    return out


def _face_diag_rhs(mesh: Mesh3D, index: GridIndex, face: str, a_p: np.ndarray,
                   b: np.ndarray, radiation: bool) -> None:
    """Add one domain face's contribution to the diagonal and the RHS (in place).

    The face's own ``h``/``T_inf`` are used, so a node on an edge or corner is
    driven by every exposed face with the right coefficients.
    """
    bc = mesh.face_bc[face]
    if not bc.is_active():
        return
    mask = index.on_face[face]
    if bc.kind == BoundaryType.DIRICHLET:
        return  # handled by apply_dirichlet
    size = index.sizes[FACE_AXIS[face]][mask]              # local first-cell size
    if bc.kind == BoundaryType.NEUMANN:
        b[mask] += bc.value / size
        return
    k_face = mesh.k.ravel(order="F")[mask]
    t_face = mesh.T.ravel(order="F")[mask]
    h = np.full(k_face.shape, bc.h)
    if radiation and bc.emissivity > 0:
        h = h + radiation_h(t_face, bc.value, np.full(k_face.shape, bc.emissivity))
    a_conv = half_cell_h(k_face, h, size) / size
    a_p[mask] += a_conv
    b[mask] += a_conv * bc.value


def dirichlet_rows(mesh: Mesh3D, index: GridIndex) -> tuple[np.ndarray, np.ndarray]:
    """(mask, values) of the nodes whose temperature is fixed by a Dirichlet face."""
    mask = dirichlet_mask(mesh, index)
    values = np.zeros(mesh.N_total)
    for face in FACES:
        bc = mesh.face_bc[face]
        if bc.kind == BoundaryType.DIRICHLET:
            values[index.on_face[face]] = bc.value
    return mask, values


def build_steady_matrix(mesh: Mesh3D, index: GridIndex = None, radiation: bool = False,
                        enforce_dirichlet: bool = True,
                        ) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Assemble ``A T = b`` (per unit volume) for the current mesh fields.

    ``enforce_dirichlet=False`` leaves the boundary rows untouched: the transient
    builder needs them intact until the mass diagonal has been added, otherwise
    the identity row is scaled by ``M/dt`` and the fixed temperature decays to
    zero instead of being enforced.
    """
    index = index or GridIndex.from_mesh(mesh)
    coeff = face_coefficients(mesh, index)
    a_p = coeff.sum(axis=0)
    b = (mesh.Q_source + mesh.Q_sink).ravel(order="F").astype(float).copy()

    for face in FACES:
        _face_diag_rhs(mesh, index, face, a_p, b, radiation)

    excluded = mesh.excluded.ravel(order="F")
    if excluded.any() and mesh.h_out > 0.0:
        # the outer surface: every active cell that faces an excluded one exchanges
        # h_out * A * (T_ambient - T) with the environment.  This replaces the air
        # domain entirely - no conduction through the air, no cells spent on it.
        for face_index, nb in enumerate(index.neighbours):
            mask = ~excluded & excluded[nb] & ~index.on_face[FACES[face_index]]
            if not mask.any():
                continue
            axis = FACE_AXIS[FACES[face_index]]
            area_over_v = index.areas[axis][mask] / index.volume[mask]
            a_conv = mesh.h_out * area_over_v
            a_p[mask] += a_conv
            b[mask] += a_conv * mesh.t_ambient

    tube = index.interior_tube
    if tube.any():
        a_tube = mesh.bc_h.ravel(order="F")[tube] / mesh.h_char.ravel(order="F")[tube]
        a_p[tube] += a_tube
        b[tube] += a_tube * mesh.bc_T_inf.ravel(order="F")[tube]

    dirichlet, values = dirichlet_rows(mesh, index)
    if mesh.excluded.any():
        # excluded cells carry no physics: pin them at the ambient temperature so the
        # system stays non-singular and the report can still show the whole box
        flat_excluded = mesh.excluded.ravel(order="F")
        dirichlet = dirichlet | flat_excluded
        values = np.where(flat_excluded, mesh.t_ambient, values)
    a = _assemble(index, coeff, a_p)
    if enforce_dirichlet:
        a = apply_dirichlet(a, b, dirichlet, values)
    return a, b


def build_transient_operators(mesh: Mesh3D, dt: float, index: GridIndex = None,
                              radiation: bool = False) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return ``(A, m_diag)`` for ``(M/dt + L) T = (M/dt) T^n + b``."""
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    index = index or GridIndex.from_mesh(mesh)
    a, _ = build_steady_matrix(mesh, index=index, radiation=radiation,
                               enforce_dirichlet=False)
    m_diag = (mesh.rho * mesh.cp).ravel(order="F").astype(float)
    a = (a + sparse.diags(m_diag / dt)).tocsr()
    mask, values = dirichlet_rows(mesh, index)
    a = apply_dirichlet(a, np.zeros(mesh.N_total), mask, values)
    return a, m_diag


def steady_rhs(mesh: Mesh3D, index: GridIndex = None, radiation: bool = False) -> np.ndarray:
    """The ``b`` of the current state (sources + BC contributions), per volume."""
    _, b = build_steady_matrix(mesh, index=index, radiation=radiation)
    return b


def transient_rhs(mesh: Mesh3D, m_diag: np.ndarray, dt: float, T_prev: np.ndarray,
                  index: GridIndex = None, radiation: bool = False) -> np.ndarray:
    """RHS of one backward-Euler step for the current sources."""
    index = index or GridIndex.from_mesh(mesh)
    rhs = (m_diag / dt) * np.asarray(T_prev, dtype=float).ravel(order="F")
    rhs += steady_rhs(mesh, index=index, radiation=radiation)
    for face in FACES:
        bc = mesh.face_bc[face]
        if bc.kind == BoundaryType.DIRICHLET:
            rhs[index.on_face[face]] = bc.value
    return rhs


def apply_dirichlet(a: sparse.csr_matrix, b: np.ndarray, mask: np.ndarray,
                    values: np.ndarray) -> sparse.csr_matrix:
    """Impose ``T = value`` on the rows selected by ``mask``.

    Returns the modified matrix (``to_csc``/``tocsr`` copy, so the caller must
    keep the result).

    Symmetric elimination: the known contribution is moved to the right-hand side
    and the corresponding *columns* are zeroed before the rows are replaced by the
    identity.  Simply replacing the rows leaves the matrix asymmetric, which makes
    BiCGSTAB break down and CG fundamentally inapplicable.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return a.tocsr()
    a_csc = a.tocsc()
    indptr, indices, data = a_csc.indptr, a_csc.indices, a_csc.data
    for col in np.flatnonzero(mask):
        start, stop = indptr[col], indptr[col + 1]
        rows = indices[start:stop]
        keep = rows != col
        if keep.any():
            b[rows[keep]] -= data[start:stop][keep] * values[col]
        data[start:stop] = 0.0

    a_csr = a_csc.tocsr()
    a_csr.sort_indices()
    row_ptr, row_idx, row_data = a_csr.indptr, a_csr.indices, a_csr.data
    for row in np.flatnonzero(mask):
        start, stop = row_ptr[row], row_ptr[row + 1]
        row_data[start:stop] = 0.0
        pos = start + int(np.searchsorted(row_idx[start:stop], row))
        if pos < stop and row_idx[pos] == row:
            row_data[pos] = 1.0
        else:
            a_csr[row, row] = 1.0
    b[mask] = np.asarray(values, dtype=float)[mask]
    return a_csr


def _assemble(index: GridIndex, coeff: np.ndarray, a_p: np.ndarray) -> sparse.csr_matrix:
    n = a_p.size
    rows = np.tile(index.flat, 7)
    cols = np.concatenate((index.flat, *[index.neighbours[axis] for axis in range(6)]))
    vals = np.concatenate((a_p, *[-coeff[axis] for axis in range(6)]))
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
