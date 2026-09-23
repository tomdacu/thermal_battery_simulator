"""Heat-flux evaluation: one implementation shared by solver, balance and checks.

Every flux is expressed with the rules the assembly itself applies - the face
conductance :func:`src.solver.matrix.face_conductance` on the mesh's own face list,
``h_eff = 2kh/(2k+hd)`` for convective faces, ``2k/d`` for a Dirichlet surface ``d/2``
away from the node centre - so the integrals below close the discrete energy balance
instead of approximating it with a different formula.

Two meshes, one rule.  A mesh that carries its own conservative face list
(:meth:`src.core.mesh_api.MeshAPI.faces`, i.e. the adaptive mesh) is read through it: the
report and the assembly then share the same faces *and* the same conductances, which is
what keeps the balance honest on a tree with hanging nodes.  ``Mesh3D`` has no face list;
its report reads the ``GridIndex`` neighbour tables, which is the face list its assembly
was built from (``docs/16_ADAPTIVE_MESH_MIGRATION.md`` section 1.2) and whose layout
:meth:`src.core.grid.GridIndex.face_rows` restates.  Wherever a flux is a face law it is
written once, in terms of the shared rule, and called by both roads.

Sign convention: positive flux LEAVES the domain / the battery.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ..core.grid import (FACE_AXIS, FACE_SLICES, INNER_SLICES, GridIndex,
                         dirichlet_mask)
from ..core.mesh import FACES, BoundaryType, MaterialID, Mesh3D
from ..core.mesh_api import is_interior_tube
from ..core.physics import half_cell_h, radiation_h

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from ..core.adaptive_mesh import AdaptiveMesh


def as_flat(values) -> np.ndarray:
    """One-dimensional view of a per-cell field, in the mesh's own cell order.

    ``Mesh3D`` keeps its fields in a 3-D array in Fortran order and an adaptive mesh in
    a flat one; the same call reads either.
    """
    return np.asarray(values).ravel(order="F")


def _face_driven(mesh) -> bool:
    """True for a mesh that carries its own face list (:class:`MeshAPI.faces`).

    ``Mesh3D`` is the structured mesh of the transition: it has no face list, so its
    report reads the neighbour tables of :class:`GridIndex` - the same faces in index
    arithmetic.  Everything else (today: :class:`AdaptiveMesh`) answers ``faces``.
    """
    return hasattr(mesh, "faces")


def structured_index(mesh, index: GridIndex | None = None) -> GridIndex | None:
    """The structured tables of ``mesh`` - from the caller, or built here.

    ``None`` on a mesh that carries its own face list: a tree has no index arithmetic and
    the report must not invent one for it.  A caller that already built the tables (the
    solver hands its own ``index`` to the balance on every transient step) passes them in.
    """
    if index is not None or _face_driven(mesh):
        return index
    return GridIndex.from_mesh(mesh)


def _face_conductance(*args, **kwargs) -> np.ndarray:
    """The shared face rule of :func:`src.solver.matrix.face_conductance`, fetched lazily.

    The import is deferred on purpose: the solver package imports the analysis
    (``src/solver/transient.py`` reads ``src.analysis.balance``), so a module-level
    import of ``src.solver.matrix`` from here would close a cycle while the packages are
    being initialised - the reason :mod:`src.core.physics` exists at all.  The rule is
    needed where it is needed, at the moment a report is drawn up, long after both
    packages are loaded.
    """
    from ..solver.matrix import face_conductance

    return face_conductance(*args, **kwargs)


def _face_coefficients(mesh: Mesh3D, index: GridIndex) -> np.ndarray:
    """The structured ``(6, N)`` face table of the assembly, fetched lazily (see above)."""
    from ..solver.matrix import face_coefficients

    return face_coefficients(mesh, index)


def _faces(mesh) -> tuple[np.ndarray, ...]:
    """``(cell_i, cell_j, axis, area, d_centers, g)`` of a face-driven mesh, in SI units.

    The mesh's own list, in its own order, with the conductance its assembly applies -
    so one entry of the face list is one heat rate ``g (T_i - T_j)``.
    """
    i, j, axis, area, distance, g = mesh.face_rows()
    return (np.asarray(i, dtype=int), np.asarray(j, dtype=int),
            np.asarray(axis, dtype=int), np.asarray(area, dtype=float),
            np.asarray(distance, dtype=float), np.asarray(g, dtype=float))


def pinned_cells(mesh, index: GridIndex | None = None) -> np.ndarray:
    """``(n_cells,)`` mask of the cells whose row the Dirichlet elimination replaced.

    Two sets land there, on either mesh: the cells a ``face_bc`` of kind DIRICHLET
    drives (``GridIndex.on_face`` for a structured mesh, ``wall_cells`` for a tree) and
    the *excluded* cells, which the assembly pins at ``t_ambient`` so the system stays
    non-singular.  A pinned cell carries an identity row, so neither a source deposited
    in it nor a film over it ever reaches the solution: the energy balance must not count
    the first (it would report an input the operator dropped) and the flux report must
    not count the second (it would report an exchange that never happened).  The
    structured and the adaptive reports share this mask, which is what makes the two
    agree on a model with a ground face and an excluded air box.
    """
    if _face_driven(mesh):
        return np.asarray(mesh.fixed_mask(), dtype=bool)
    index = structured_index(mesh, index)
    return dirichlet_mask(mesh, index) | as_flat(mesh.excluded)


def _inner_slice(face: str, shape: tuple) -> tuple:
    """First cell inside the domain, clamped for axes with a single cell."""
    out = list(INNER_SLICES[face])
    for axis, item in enumerate(out):
        if isinstance(item, int):
            n = shape[axis]
            out[axis] = max(min(item if item >= 0 else n + item, n - 1), 0)
    return tuple(out)


def domain_face_flux(mesh: Mesh3D | AdaptiveMesh, face: str, radiation: bool = False,
                     index: GridIndex | None = None) -> float:
    """Heat leaving the domain through one of the six box faces [W].

    Dirichlet faces are pinned to ``T_bc``, so their flux CANNOT be evaluated from the
    boundary node (it would be identically zero - a defect that made the ground loss
    disappear from every report).  The one-sided gradient towards the first interior cell
    is used instead, with the same face conductance as the assembly; on an adaptive mesh
    that gradient is read off the mesh's own face list, so the number reported is the one
    the operator carries on the very faces it built.

    A node pinned by *another* Dirichlet face is skipped: its own row is replaced by
    the identity, so the exchange it would have had through this face never enters
    the solution and must not be reported either (it is the ground-to-wall edge of
    the battery).  The same holds for a cell an environment model excluded.

    A convective face is the half-cell film ``h_eff = 2kh/(2k + hd)`` of the leaf sitting
    on it - the law the assembly puts on the diagonal - and a Neumann face the imposed
    flux over its free cells.  On an adaptive mesh the box face has no radiative share
    yet (``docs/16`` section 6: radiation is a structured feature), so ``radiation`` is
    read only by the structured road and the report always says what that assembly did.
    """
    if _face_driven(mesh):
        return _domain_face_flux_adaptive(mesh, face)
    return _domain_face_flux_structured(mesh, face, radiation, index)


def _domain_face_flux_structured(mesh: Mesh3D, face: str, radiation: bool,
                                 index: GridIndex | None) -> float:
    """The box-face flux of a structured mesh, from the ``GridIndex`` slides."""
    bc = mesh.face_bc[face]
    sl = FACE_SLICES[face]
    inner = _inner_slice(face, mesh.T.shape)
    axis = FACE_AXIS[face]
    index = structured_index(mesh, index)
    area = _face_area(mesh, axis, sl)                 # per-cell face area
    # nodes pinned by a Dirichlet face or by the exclusion: their row is the identity,
    # so this face's exchange never reaches the solution
    pinned = pinned_cells(mesh, index).reshape(mesh.T.shape, order="F")
    # a cell outside the envelope leaves the problem too: it is pinned at the ambient
    # and its exchange is the environment film, counted by ``environment_flux``.  Not
    # skipping it here reported the convection of the box faces as if it happened and
    # broke the balance by kilowatts on the default vessel.
    excluded = as_flat(mesh.excluded).reshape(mesh.T.shape, order="F")
    free = ~pinned[sl]
    free_inner = ~pinned[inner]
    if bc.kind == BoundaryType.DIRICHLET:
        size_inner = mesh.axis_size(axis)[inner]
        size_wall = mesh.axis_size(axis)[sl]
        # the boundary *cell* is pinned, so the flux towards it uses exactly the
        # centre-to-centre distance of the assembly (the two half cells), otherwise
        # the reported loss disagrees with the operator on a graded mesh
        g = _face_conductance(mesh.k[inner], mesh.k[sl], area,
                              0.5 * (size_inner + size_wall),
                              size_a=size_inner, size_b=size_wall,
                              material_a=mesh.material_id[inner],
                              material_b=mesh.material_id[sl],
                              h_contact=mesh.h_contact,
                              excluded_b=excluded[inner] | excluded[sl])
        # the corner cells pinned by *another* Dirichlet face are not part of this
        # face's exchange: counting them breaks the balance on a graded grid
        return float(np.sum(g * (mesh.T[inner] - bc.value) * free_inner))
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


def _domain_face_flux_adaptive(mesh, face: str) -> float:
    """The box-face flux of a face-driven mesh: the film rates of its wall leaves.

    The film is the half-cell law of :meth:`AdaptiveMesh._films` (the same number the
    diagonal carries), the Neumann term the imposed flux over the wall leaves, and the
    Dirichlet term the conduction into the pinned wall leaves - read off the mesh's own
    face list, with the conductance of its assembly, so the loss is the sum of the face
    rates the solver actually applied rather than an integral of a second formula.
    """
    bc = mesh.face_bc[face]
    if not bc.is_active():
        return 0.0
    pinned = pinned_cells(mesh)
    positions = mesh.wall_indices(face)
    # the half-cell depth is the leaf's edge along the face normal, and its face the
    # volume over that edge (a cube's size^2; a flat or tall box's own face)
    size = mesh.extent[positions, FACE_AXIS[face]]
    area = mesh.V[positions] / size
    free = ~pinned[positions]
    if bc.kind == BoundaryType.DIRICHLET:
        wall = np.zeros(mesh.n_cells, dtype=bool)
        wall[positions] = True
        i, j, _axis, _area, _distance, g = _faces(mesh)
        first, second = wall[i], wall[j]
        crossing = first ^ second                  # one side on the wall, one inside
        inner = np.where(first[crossing], j[crossing], i[crossing])
        outer = np.where(first[crossing], i[crossing], j[crossing])
        acting = ~pinned[inner] & ~mesh.excluded[outer]
        return float(np.sum(g[crossing][acting] * (mesh.T[inner][acting] - bc.value)))
    if bc.kind == BoundaryType.NEUMANN:
        return float(-bc.value * np.sum(area[free]))
    if bc.kind == BoundaryType.CONVECTION:
        return float(np.sum(half_cell_h(mesh.k[positions], bc.h, size)
                            * (mesh.T[positions] - bc.value) * area * free))
    return 0.0


def _face_area(mesh: Mesh3D, axis: int, selector) -> np.ndarray:
    """Face areas of a box face (or of a masked set) for a face normal to ``axis``."""
    return (mesh.Ax, mesh.Ay, mesh.Az)[axis][selector]


def _face_size(mesh: Mesh3D, axis: int, selector) -> np.ndarray:
    """Local cell size along ``axis`` on a box face."""
    return mesh.axis_size(axis)[selector]


def environment_flux(mesh: Mesh3D | AdaptiveMesh, index: GridIndex | None = None) -> float:
    """Heat leaving the *active* region through the outside film [W].

    When the air is excluded from the problem, the outer surface is an internal
    boundary: the box faces see nothing of it, so the loss must be integrated here or
    the energy balance of the domain would not close.

    The film is the one the last assembly applied - ``mesh.h_out``, the coefficient
    ``src/solver/matrix.environment_film`` re-evaluated at the surface temperature the
    film itself drives and wrote back - so the report and the operator it describes
    carry the same number and no third evaluation can drift from them.  A cell whose row
    the elimination replaced (``pinned_cells``) does not carry the film: its temperature
    is imposed, so counting its exchange here made the reported environment flux fall
    short (0.77 W on a 5 kW input on the default vessel).
    """
    if mesh.h_out <= 0.0 or not np.any(mesh.excluded):
        return 0.0
    if _face_driven(mesh):
        return _environment_flux_adaptive(mesh)
    index = structured_index(mesh, index)
    excluded = as_flat(mesh.excluded)
    temperature = as_flat(mesh.T)
    pinned = pinned_cells(mesh, index)
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


def _environment_flux_adaptive(mesh) -> float:
    """The same integral, on the active side of the mesh's own active/excluded faces."""
    excluded = as_flat(mesh.excluded)
    temperature = as_flat(mesh.T)
    i, j, _axis, area, _distance, _g = _faces(mesh)
    active_i = ~excluded[i] & excluded[j]
    active_j = ~excluded[j] & excluded[i]
    positions = np.concatenate((i[active_i], j[active_j]))
    areas = np.concatenate((area[active_i], area[active_j]))
    acting = ~pinned_cells(mesh)[positions]
    return float(np.sum(mesh.h_out * areas[acting]
                        * (temperature[positions[acting]] - mesh.t_ambient)))


def domain_fluxes(mesh: Mesh3D | AdaptiveMesh, radiation: bool = False,
                  index: GridIndex | None = None) -> dict[str, float]:
    """``{face: W}`` for the six box faces plus the total."""
    index = structured_index(mesh, index)
    out = {face: domain_face_flux(mesh, face, radiation, index) for face in FACES}
    out["total"] = float(sum(out.values()))
    return out


def tube_flux(mesh: Mesh3D | AdaptiveMesh, index: GridIndex | None = None) -> float:
    """Heat leaving the domain into the heat-transfer fluid [W].

    The mask is the one the assembly applies - ``boundary_type == CONVECTION``
    (:func:`src.core.mesh_api.is_interior_tube`) minus the cells the Dirichlet
    elimination pins, whose identity row never sees the film - and the rate is
    ``bc_h / V^(1/3)`` per cell, the coefficient the assembly puts on the diagonal and
    multiplies back by the volume here.
    """
    mask = (as_flat(is_interior_tube(mesh.boundary_type, mesh.on_box_face))
            if _face_driven(mesh) else structured_index(mesh, index).interior_tube)
    mask = mask & ~pinned_cells(mesh, index)
    if not mask.any():
        return 0.0
    h = as_flat(mesh.bc_h)[mask]
    t_cell = as_flat(mesh.T)[mask]
    t_fluid = as_flat(mesh.bc_T_inf)[mask]
    return float(np.sum(h * (t_cell - t_fluid) * as_flat(mesh.V)[mask]
                        / as_flat(mesh.h_char)[mask]))


def face_fluxes(mesh: Mesh3D | AdaptiveMesh, temperature: np.ndarray | None = None,
                index: GridIndex | None = None) -> np.ndarray:
    """Heat rate [W] through every entry of the mesh's own face list.

    Positive from the first cell of the entry to the second.  The list is the one the
    assembly used - ``faces()`` on an adaptive mesh, the ``GridIndex`` neighbour tables on
    a structured one - and the conductance is the shared rule, so accumulating these
    rates around a cell reproduces the operator the assembly built there: that identity
    is what ``tests/test_balance_on_tree.py`` pins, and it is the reason the report can be
    trusted on a mesh whose cells change size.
    """
    values = as_flat(mesh.T if temperature is None else temperature)
    if _face_driven(mesh):
        i, j, _axis, _area, _distance, g = _faces(mesh)
        return g * (values[i] - values[j])
    index = structured_index(mesh, index)
    i, j, axis, _area, _distance = index.face_rows()
    # the assembly's own conductances: its (6, N) coefficient table is per volume, so the
    # rate across an entry is that coefficient times the volume of the cell it sits in
    g = _face_coefficients(mesh, index)[2 * axis + 1, i] * index.volume[i]
    # no conduction into an excluded cell on this road either: its exchange is the film
    excluded = as_flat(mesh.excluded)
    g = np.where(excluded[i] | excluded[j], 0.0, g)
    return g * (values[i] - values[j])


def _pair_flux(mesh: Mesh3D, battery: np.ndarray, companion: np.ndarray, axis: int,
               shift: int) -> float:
    """Conductive flux across battery/companion cell pairs along ``axis`` [W]."""
    t_nb = np.roll(mesh.T, shift, axis=axis)
    mask = battery & np.roll(companion, shift, axis=axis)
    edge = [slice(None)] * 3
    edge[axis] = -1 if shift < 0 else 0      # np.roll wraps around: drop that row
    mask[tuple(edge)] = False
    if not mask.any():
        return 0.0
    size_self = mesh.axis_size(axis)[mask]
    size_other = np.roll(mesh.axis_size(axis), shift, axis=axis)[mask]
    area = (mesh.Ax, mesh.Ay, mesh.Az)[axis][mask]
    g = _face_conductance(
        mesh.k[mask], np.roll(mesh.k, shift, axis=axis)[mask], area,
        0.5 * (size_self + size_other), size_a=size_self, size_b=size_other,
        material_a=mesh.material_id[mask],
        material_b=np.roll(mesh.material_id, shift, axis=axis)[mask],
        h_contact=mesh.h_contact,
        excluded_b=np.roll(mesh.excluded, shift, axis=axis)[mask] | mesh.excluded[mask])
    return float(np.sum(g * (mesh.T[mask] - t_nb[mask])))


def _envelope_fluxes_adaptive(mesh) -> dict[str, float]:
    """The envelope integral over the mesh's own faces, split by the surface it leaves."""
    battery = as_flat(mesh.material_id) != int(MaterialID.AIR)
    i, j, axis, _area, _distance, g = _faces(mesh)
    cross = battery[i] != battery[j]
    i, j, axis, g = i[cross], j[cross], axis[cross], g[cross]
    rate = g * (as_flat(mesh.T)[i] - as_flat(mesh.T)[j])
    leaving = np.where(battery[i], rate, -rate)       # positive out of the battery
    # the exchange with the air *above* a battery cell is through that cell's +axis
    # surface, the one with the air below through its -axis surface
    centres = mesh.centres()
    outward_plus = np.where(battery[i], centres[j, axis] > centres[i, axis],
                            centres[i, axis] > centres[j, axis])
    up = axis == 2
    top = float(np.sum(leaving[up & outward_plus]))
    bottom = float(np.sum(leaving[up & ~outward_plus]))
    side = float(np.sum(leaving[~up]))
    return {"top": top, "bottom": bottom, "side": side, "total": top + bottom + side}


def envelope_fluxes(mesh: Mesh3D | AdaptiveMesh, index: GridIndex | None = None) -> dict[str, float]:
    """Flux leaving the *battery* through its outer surface, split by direction.

    The envelope is the interface between battery cells (sand, insulation,
    steel, concrete) and the surrounding air, evaluated with the conductance the
    assembly applies to it.  Being an interface integral it does not scale with the
    size of the air box.

    A face whose other side is an *excluded* cell carries no conduction at all - the
    assembly zeroes it, and the exchange is the environment film that
    :func:`environment_flux` reports - so the two are never counted twice, and the
    phantom conduction into a dropped air box (a quarter of the reported envelope loss
    on the default vessel) is gone.

    ``top`` = the battery's +z surface, ``bottom`` = -z, ``side`` = +-x and +-y.
    """
    if _face_driven(mesh):
        return _envelope_fluxes_adaptive(mesh)
    battery = mesh.material_id != int(MaterialID.AIR)
    companion = ~battery
    top = _pair_flux(mesh, battery, companion, 2, -1)
    bottom = _pair_flux(mesh, battery, companion, 2, 1)
    side = sum(_pair_flux(mesh, battery, companion, axis, shift)
               for axis in (0, 1) for shift in (-1, 1))
    return {"top": top, "bottom": bottom, "side": side, "total": top + bottom + side}


def stored_energy(mesh: Mesh3D | AdaptiveMesh, t_reference: float) -> float:
    """Sensible energy stored above ``t_reference`` [J]."""
    return float(np.sum(mesh.rho * mesh.cp * (mesh.T - t_reference) * mesh.V))


def stored_exergy(mesh: Mesh3D | AdaptiveMesh, t_reference: float) -> float:
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
