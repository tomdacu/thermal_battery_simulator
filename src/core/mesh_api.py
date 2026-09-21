"""The cell-mesh protocol the essential consumers of :class:`Mesh3D` really use.

The octree is to become the main mesh, and the port has to be driven by the call sites
rather than by the class: a mesh only has to provide what the solver and the energy
balance *read*.  This module states that contract once, so that
:class:`~src.core.adaptive_mesh.AdaptiveMesh` and (during the transition)
:class:`~src.core.mesh.Mesh3D` can be held against the same list, and so that a consumer
can be ported without wondering which of the two it was written for.

The protocol is *structural* and deliberately small.  It is the intersection of what the
assembly (:mod:`src.solver.matrix`), the transient operators, the linear layer, the flux
report (:mod:`src.analysis.fluxes`) and the balance (:mod:`src.analysis.balance`) read
from a mesh, and nothing more: a consumer that reads three members does not oblige the
interface to name ten.  What it does *not* contain is as deliberate:

* no index arithmetic (``ijk_to_linear``, ``linear_to_ijk``, ``find_cell``) - that is
  the structured layout, and an adaptive mesh answers it with a point location, not with
  a formula (see the census in ``docs/16_ADAPTIVE_MESH_MIGRATION.md``);
* no per-axis sizes (``dx``/``dy``/``dz``, ``edges_x``, ``axis_size``) - a consumer that
  needs a local size needs it *at a cell*, which is what the face list already carries
  through ``d_centers`` and what the film terms take from the cell's own volume;
* no convenience scalars (``d``, ``V_cell``, ``A_cell``, ``uniform``, ``h_char``) - the
  protocol keeps ``V`` and lets the consumer take the cube root or the uniform test;
* no GUI/report vocabulary (``grid_summary``, ``size_label``, ``get_info``) - that is a
  presentation concern and belongs to whatever renders the mesh.

Every member below names the call sites that justify it and the reason it is needed.
The line numbers are those of the census in ``docs/16_ADAPTIVE_MESH_MIGRATION.md``.

The *order* of the cells is part of the contract only through :meth:`MeshAPI.faces`:
every index in the face list indexes the flat per-cell arrays, whatever order the mesh
uses internally (Fortran order for ``Mesh3D``, ``Octree.leaves`` order for the adaptive
mesh).  A consumer that assumes more than that is assuming the structured layout.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from .mesh import BoundaryType, FaceBC

#: one entry of the conservative face list: ``(cell_i, cell_j, axis, area, d_centers)``
#: with the area [m^2] and the centre-to-centre distance [m] of its own pair, each face
#: listed once (the octree splits a coarse face into the fine faces that exist).
FaceRow = tuple[int, int, int, float, float]

#: the members a mesh must provide, in the order they are documented below.
#: Kept as data so that a port can report exactly what is missing from a candidate.
MEMBERS = (
    "n_cells", "T", "k", "rho", "cp", "V", "material_id", "Q_source", "Q_sink",
    "boundary_type", "bc_h", "bc_T_inf", "excluded", "source_mask", "h_out",
    "t_ambient", "h_contact", "face_bc", "faces",
)


@runtime_checkable
class MeshAPI(Protocol):
    """One value per cell, one conductance per face: what the physics reads.

    Every array is indexed by the mesh's cell order and holds one entry per cell
    (a flat vector, or a structured array that a structured consumer ravels itself).
    """

    #: number of cells: allocation and masks.  ``Mesh3D.N_total`` at
    #: ``src/solver/matrix.py:116`` (``np.zeros(mesh.N_total)``),
    #: ``src/solver/steady.py:66`` (the initial guess), ``src/solver/transient.py``.
    n_cells: int

    #: temperature [K] per cell, writable: the solver writes the field back at
    #: ``src/solver/steady.py:82`` (``self.mesh.T = ...``), the radiation path reads the
    #: surface temperature at ``src/solver/matrix.py:104``, the balance at
    #: ``src/analysis/balance.py:87`` and ``src/analysis/fluxes.py:112``.
    T: np.ndarray

    #: conductivity [W/(m K)] per cell: the face coefficients of
    #: ``src/solver/matrix.py:57`` (harmonic mean), the contact resistance at
    #: ``src/solver/matrix.py:72-74`` and every flux in ``src/analysis/fluxes.py:63``.
    k: np.ndarray

    #: density [kg/m^3] and specific heat [J/(kg K)] per cell: the transient mass
    #: matrix ``M = rho * cp`` (``src/solver/matrix.py``, ``build_transient_operators``)
    #: and the stored energy of ``src/analysis/balance.py:153`` and
    #: ``src/analysis/cycle.py:257``.
    rho: np.ndarray
    cp: np.ndarray

    #: cell volume [m^3]: the per-volume coefficients are multiplied back by it in the
    #: balance (``src/analysis/balance.py:112``, ``src/solver/transient.py:161``), it
    #: scales the iterative solver of a graded mesh (``src/solver/steady.py:58``) and it
    #: is the weight of every volume mean (``src/analysis/balance.py:153``).
    V: np.ndarray

    #: material id per cell (``MaterialID``): the contact resistance only acts between
    #: cells of different materials (``src/solver/matrix.py:66-75``) and the balance
    #: masks storage/insulation/shell with it (``src/analysis/balance.py:119-122``).
    material_id: np.ndarray

    #: volumetric source [W/m^3] (>= 0) and sink [W/m^3] (<= 0) per cell: they are the
    #: right-hand side of the assembly (``src/solver/matrix.py:137``) and the input power
    #: of the balance (``src/analysis/balance.py:112``).
    Q_source: np.ndarray
    Q_sink: np.ndarray

    #: boundary type per cell (``BoundaryType`` codes): a cell marked CONVECTION
    #: exchanges with a fluid through the film of ``bc_h``/``bc_T_inf``
    #: (``src/solver/matrix.py:157-161``, ``src/core/grid.py:113``).
    boundary_type: np.ndarray

    #: internal film [W/(m^2 K)] and fluid temperature [K] per cell: the tube-side
    #: exchange of ``src/solver/matrix.py:159-161``, ``src/analysis/fluxes.py:139`` and
    #: the extraction path of ``src/solver/transient.py:112-113``.
    bc_h: np.ndarray
    bc_T_inf: np.ndarray

    #: cells outside the thermal problem (the air that an environment model drops):
    #: no conduction into them (``src/solver/matrix.py:58``), their interface carries the
    #: outside film (``src/solver/matrix.py:143-155``, ``src/analysis/fluxes.py:106``),
    #: and they are pinned at ``t_ambient`` so the system stays non-singular.
    excluded: np.ndarray

    #: cells whose ``Q_source`` a power profile drives: written by the transient march
    #: and the geometry (``src/solver/transient.py:160-161``,
    #: ``src/core/geometry.py``, ``src/core/heaters.py``); it is bookkeeping, not physics,
    #: which is why it is a mask rather than a value.
    source_mask: np.ndarray

    #: film coefficient of the outer surface [W/(m^2 K)] and the ambient temperature [K]
    #: that drives it: the environment model of ``src/solver/matrix.py:143-155`` and
    #: ``src/analysis/fluxes.py:106-121``.
    h_out: float
    t_ambient: float

    #: contact conductance between materials [W/(m^2 K)]: the series resistance of
    #: ``src/solver/matrix.py:66-75``.  Zero means a perfect interface.
    h_contact: float

    #: boundary condition of the six box faces, keyed by ``core.mesh.FACES``:
    #: ``src/solver/matrix.py:93-120`` (diagonal, RHS and the Dirichlet rows),
    #: ``src/analysis/fluxes.py:46`` (the same face read back as a flux) and
    #: ``src/core/grid.py:47`` (which cells a face pins).
    face_bc: Mapping[str, FaceBC]

    def faces(self) -> Sequence[FaceRow]:
        """The conservative face list: ``(cell_i, cell_j, axis, area, d_centers)``.

        One entry per interface, listed once, with the area [m^2] and the
        centre-to-centre distance [m] of *its own* pair - so the heat rate
        ``k_face * area / d_centers * (T_i - T_j)`` is a single number shared by both
        cells and the discrete balance closes to machine precision on any mesh.  This is
        what the structured assembly builds from ``GridIndex.neighbours``/``areas``/
        ``sizes`` and what ``Octree.faces()`` already provides; the boundary of the box
        carries no entry, which is the adiabatic case and the reason a Dirichlet or
        convective face has to be applied to the cells of that face (see
        ``MeshAPI.excluded``, ``face_bc`` and the adaptive assembly).
        """
        ...


def missing_members(mesh: object) -> list[str]:
    """Protocol members ``mesh`` does not provide, empty when it conforms.

    :class:`MeshAPI` is structural, so ``isinstance`` answers the same question the way
    a protocol test can; this returns the *names*, which is what a port wants to print.
    """
    return [name for name in MEMBERS if not hasattr(mesh, name)]


def is_interior_tube(boundary_type: np.ndarray, on_box_face: np.ndarray) -> np.ndarray:
    """Cells exchanging with a fluid that are *not* on a box face.

    The structured index excludes the exposed cells from the tube film
    (``GridIndex.interior_tube``, used by ``src/solver/matrix.py:157`` and
    ``src/analysis/fluxes.py:137``), because a cell on the box face has a face boundary
    condition of its own.  A mesh of either kind can answer this from
    ``boundary_type`` plus "does this cell touch a box face", which is exactly what the
    adaptive mesh keeps.
    """
    tube = np.asarray(boundary_type) == int(BoundaryType.CONVECTION)
    return tube & ~np.asarray(on_box_face, dtype=bool)
