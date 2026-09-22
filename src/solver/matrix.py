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
  added to the convective coefficient of the exposed face, and to the environment film
  of an excluded-air model (:func:`environment_film`), whose surface temperature the
  coefficient itself drives.

TRANSIENT
---------
Backward Euler with the mass matrix ``M = diag(rho*cp)`` [J/(m^3*K)]::

    (M/dt + L) T^{n+1} = (M/dt) T^n + b

``M`` must NOT contain the cell volume: ``L`` is already per unit volume, and a
``V`` factor scales every time constant by ``1/d^3``, making the answer depend on
the mesh resolution.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

from ..constants import EPS
from ..core.grid import FACE_AXIS, GridIndex, dirichlet_mask
from ..core.physics import half_cell_h, radiation_h
from ..core.mesh import FACES, BoundaryType, Mesh3D

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from ..core.adaptive_mesh import AdaptiveMesh


def face_conductance(k_a, k_b, area, distance, *, size_a=None, size_b=None,
                     material_a=None, material_b=None, h_contact: float = 0.0,
                     excluded_b=None) -> np.ndarray:
    """The conductance ``g`` [W/K] of one interface: the one rule both meshes assemble.

    ``g = k_face A / d_centers`` with ``k_face`` the harmonic mean of the two
    conductivities - the series conductance of the two half cells, which is what makes a
    two-layer wall exact when its interface sits on a face.  Two corrections ride on the
    same face: a finite contact conductance in series with the two half cells where the
    two sides are *different materials*, and no conduction at all into an excluded side
    (its exchange with the domain is the environment film of :func:`build_steady_matrix`).

    The callers are :func:`face_coefficients` (the six structured neighbour tables of
    :class:`GridIndex`) and :meth:`AdaptiveMesh._face_table` (the index-free face list of
    an octree), so the coefficient the grid assembles and the one the tree assembles
    cannot drift apart.  ``excluded_b`` is the side the flux would flow *into*: the tree
    passes the union of the two sides (its face list holds one entry per pair), the
    structured adapter the destination alone, because the row of an excluded node is
    pinned by the Dirichlet elimination of :func:`build_steady_matrix`.

    ``size_a``/``size_b``/``material_a``/``material_b`` are the two leaf edges and the two
    material ids: the contact correction needs them, a caller that asks for no contact
    resistance (``h_contact=0``, the default) does not.
    """
    k_a = np.asarray(k_a, dtype=float)
    k_b = np.asarray(k_b, dtype=float)
    geometry = np.asarray(area, dtype=float) / np.asarray(distance, dtype=float)
    g = 2.0 * k_a * k_b / (k_a + k_b + EPS) * geometry
    if h_contact > 0.0:
        if size_a is None or size_b is None or material_a is None or material_b is None:
            raise ValueError("a contact conductance needs the two leaf sizes and materials")
        r_series = (0.5 * np.asarray(size_a, dtype=float) / (k_a + EPS) + 1.0 / h_contact
                    + 0.5 * np.asarray(size_b, dtype=float) / (k_b + EPS))
        k_eff = 0.5 * (np.asarray(size_a, dtype=float)
                       + np.asarray(size_b, dtype=float)) / r_series
        g = np.where(np.asarray(material_a) != np.asarray(material_b), k_eff * geometry, g)
    if excluded_b is not None:
        g = np.where(np.asarray(excluded_b, dtype=bool), 0.0, g)
    return g


def face_coefficients(mesh: Mesh3D, index: GridIndex) -> np.ndarray:
    """(6, N) face conductances per unit volume, zeroed on the box faces.

    The structured *adapter* of the shared rule: :class:`GridIndex` supplies the two
    sides of each of the six directions (neighbour index, face area, centre-to-centre
    distance, leaf size, material), :func:`face_conductance` turns them into a
    conductance [W/K], and the cell volume turns that into the per-unit-volume
    coefficient ``a = g / V`` the assembly wants.  The box faces carry no conductance -
    their exchange is the ``face_bc`` film - and an excluded destination carries none
    either.
    """
    k = mesh.k.ravel(order="F")
    excluded = mesh.excluded.ravel(order="F")
    material = mesh.material_id.ravel(order="F")
    out = np.zeros((6, k.size))
    for face_index, nb in enumerate(index.neighbours):
        axis = FACE_AXIS[FACES[face_index]]
        size = index.sizes[axis]
        coeff = face_conductance(
            k, k[nb], index.areas[axis], 0.5 * (size + size[nb]),
            size_a=size, size_b=size[nb], material_a=material, material_b=material[nb],
            h_contact=mesh.h_contact,
            excluded_b=excluded[nb] if excluded.any() else None) / index.volume
        coeff[index.on_face[FACES[face_index]]] = 0.0
        out[face_index] = coeff
    return out


def environment_film(mesh, positions: np.ndarray, areas: np.ndarray,
                     radiation: bool = False) -> float:
    """The film coefficient [W/(m^2 K)] of the outer surface at the current field.

    The convective film is ``mesh.h_out_conv`` when the painter recorded it (the
    correlations alone) and ``mesh.h_out`` otherwise, so the *switch* decides whether the
    surface radiates at all: with ``radiation`` off the film is the convective one even
    where :meth:`~src.core.geometry.BatteryGeometry.apply_environment` painted the
    design-point radiative share into ``h_out``.  With it on, the share is re-evaluated
    here at the surface temperature of the current field, area-weighted over the very
    faces the film acts on: ``h_rad = eps sigma (Ts + T_amb)(Ts^2 + T_amb^2)`` grows as
    ``T^3``, so the film depends on the surface temperature the film itself drives and is
    a *fixed point*.  One re-evaluation per assembly is enough to close it - the steady
    driver repeats it sweep by sweep, the transient once per step - and evaluating from
    the recorded base keeps the re-evaluation idempotent.

    A mesh that never went through
    :meth:`~src.core.geometry.BatteryGeometry.apply_environment` has no emissivity
    recorded and keeps the film it carries.
    """
    base = getattr(mesh, "h_out_conv", None)
    h = float(mesh.h_out if base is None else base)
    emissivity = float(getattr(mesh, "environment_emissivity", 0.0))
    if not radiation or emissivity <= 0.0 or positions.size == 0:
        return h
    values = np.asarray(mesh.T, dtype=float).ravel(order="F")
    t_surface = float(np.sum(areas * values[positions]) / np.sum(areas))
    return h + float(radiation_h(t_surface, mesh.t_ambient, emissivity))


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
        surface = []
        for face_index, nb in enumerate(index.neighbours):
            mask = ~excluded & excluded[nb] & ~index.on_face[FACES[face_index]]
            if not mask.any():
                continue
            axis = FACE_AXIS[FACES[face_index]]
            surface.append((mask, index.areas[axis][mask], index.volume[mask]))
        if surface:
            film = environment_film(
                mesh, np.concatenate([np.flatnonzero(mask) for mask, _a, _v in surface]),
                np.concatenate([area for _m, area, _v in surface]), radiation)
            # the film the operator carries, written back to the mesh: the report
            # (``fluxes.environment_flux``) reads ``h_out``, so the two must be the
            # same number, and the re-evaluation above is idempotent in it
            mesh.h_out = film
            for mask, area, volume in surface:
                a_conv = film * (area / volume)
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


def build_transient_operators(mesh: Mesh3D | AdaptiveMesh, dt: float,
                              index: GridIndex = None,
                              radiation: bool = False) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Return ``(A, m_diag)`` for ``(M/dt + L) T = (M/dt) T^n + b``, on either mesh.

    A structured mesh goes through :func:`build_steady_matrix` and the ``GridIndex``
    tables; a tree assembles and eliminates itself
    (:meth:`~src.core.adaptive_mesh.AdaptiveMesh.transient_operators`) with the same rule
    - the film diagonal, the mass matrix ``rho*cp``, the symmetric elimination of
    :func:`apply_dirichlet` - so a caller that hands either mesh over gets the operator
    the rest of the transient march assumes.
    """
    if not isinstance(mesh, Mesh3D):
        return mesh.transient_operators(dt, radiation=radiation)
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


def transient_rhs(mesh: Mesh3D | AdaptiveMesh, m_diag: np.ndarray, dt: float,
                  T_prev: np.ndarray, index: GridIndex = None,
                  radiation: bool = False) -> np.ndarray:
    """RHS of one backward-Euler step for the current sources, on either mesh.

    A tree rebuilds its own right-hand side
    (:meth:`~src.core.adaptive_mesh.AdaptiveMesh.transient_rhs`): the sources and the
    films of the current state, and the value alone on the rows the elimination pinned.
    """
    if not isinstance(mesh, Mesh3D):
        return mesh.transient_rhs(m_diag, dt, T_prev, radiation=radiation)
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

    Returns a new CSR matrix (the input is not modified, so the caller must
    keep the result).

    Symmetric elimination: the known contribution is moved to the right-hand side
    and the corresponding *columns* are zeroed before the rows are replaced by the
    identity.  Simply replacing the rows leaves the matrix asymmetric, which makes
    BiCGSTAB break down and CG fundamentally inapplicable.
    """
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return a.tocsr()
    values = np.asarray(values, dtype=float)
    a_csr = a.tocsr(copy=True)
    a_csr.sort_indices()
    row = np.repeat(np.arange(a_csr.shape[0]), np.diff(a_csr.indptr))
    col, data = a_csr.indices, a_csr.data
    # the known columns move to the right-hand side of the free rows
    moved = mask[col] & ~mask[row]
    if moved.any():
        np.subtract.at(b, row[moved], data[moved] * values[col[moved]])
    # pinned rows and columns are zeroed *in place*: the sparsity pattern is kept, so a
    # direct factorisation orders the matrix exactly as before the elimination
    data[mask[row] | mask[col]] = 0.0
    on_diagonal = (row == col) & mask[row]
    data[on_diagonal] = 1.0
    missing = np.setdiff1d(np.flatnonzero(mask), row[on_diagonal])
    if missing.size:
        # a pinned row without a stored diagonal: add it
        a_csr = (a_csr + sparse.csr_matrix((np.ones(missing.size), (missing, missing)),
                                           shape=a_csr.shape)).tocsr()
        a_csr.sort_indices()
    b[mask] = values[mask]
    return a_csr


def _assemble(index: GridIndex, coeff: np.ndarray, a_p: np.ndarray) -> sparse.csr_matrix:
    n = a_p.size
    rows = np.tile(index.flat, 7)
    cols = np.concatenate((index.flat, *[index.neighbours[axis] for axis in range(6)]))
    vals = np.concatenate((a_p, *[-coeff[axis] for axis in range(6)]))
    return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
