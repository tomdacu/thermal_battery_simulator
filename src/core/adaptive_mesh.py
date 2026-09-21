"""An adaptive mesh over :class:`~src.core.octree.Octree` that implements :class:`MeshAPI`.

The octree already knows *how* to solve (a conservative face list, the 2:1 balance, the
``k A / (d V)`` assembly); this module gives it the *physics vocabulary* the structured
consumers are written in - a temperature, a conductivity, sources, boundary conditions,
the environment film, the energy balance - so that a consumer can be ported from
``Mesh3D`` to a tree without the physics moving.

What is reused, and from where:

* the mesh, the face list and the operator: :mod:`src.core.octree`
  (:meth:`Octree.diffusion_matrix`, :meth:`Octree.faces`, :meth:`Octree.refine`);
* the physics entry points: :func:`src.solver.octree_solver.wall_cells` (which leaves a
  box face drives), :meth:`src.solver.octree_solver.OctreeSteadySolver.indicator` (the
  flux-jump estimator a refinement round refines on) and :func:`src.core.physics.half_cell_h`
  (the half-cell film the structured solver uses);
* the linear layer and the boundary elimination: :func:`src.solver.linear.solve_linear`
  (method choice, the volume symmetrisation of a graded operator) and
  :func:`src.solver.matrix.apply_dirichlet` (the symmetric elimination, with the known
  columns moved to the right-hand side).

What could not be reused is listed at the end of ``docs/16_ADAPTIVE_MESH_MIGRATION.md``:
``octree_solver`` has no box-face film, no environment film, no contact resistance and no
public per-face conductance array, so those live here - with the *same* formulas as
:mod:`src.solver.matrix`, which is what the equivalence tests measure.

Sameness, in the form the tests check it:

* the conduction coefficient is the octree's ``k_face A / (d_centers V)`` with the
  harmonic mean, so a uniform tree and a uniform ``Mesh3D`` assemble the same operator;
* the box-face film is ``h_eff / d`` with ``h_eff = 2kh/(2k + h d)`` and ``d`` the local
  leaf edge - the structured solver's half-cell law, exact for a flux that is constant
  across the half cell.  A source in the boundary leaf leaves it short by
  ``q d^2 / (8 k)``, and the adaptive mesh reproduces *that* shift leaf by leaf (it is the
  same formula): the tests pin both the shift and the film each leaf carries;
* the environment film acts on the *active* side of an active/excluded face as
  ``h_out A / V`` towards ``t_ambient``, and the excluded leaves are held at
  ``t_ambient`` exactly as ``src/solver/matrix.py`` holds them;
* the balance closes the way the structured one does: ``p_source + p_sink`` equals the
  heat leaving through the fixed (Dirichlet) leaves plus the films, at the round-off of
  the per-leaf closure ``div + film - Q V``.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
from scipy import sparse

from ..constants import (CP_AIR, EPS, K_AIR, RHO_AIR, T_AMBIENT_DEFAULT,
                         T_GROUND_DEFAULT, T_INITIAL_DEFAULT)
from ..solver.linear import LinearConfig, LinearResult, solve_linear
from ..solver.matrix import apply_dirichlet
from ..solver.octree_solver import OctreeSteadySolver, wall_cells
from ..units import check_kelvin
from .mesh import FACES, BoundaryType, FaceBC
from .mesh_api import FaceRow, is_interior_tube
from .octree import Leaf, Octree, uniform_tree
from .physics import half_cell_h

#: the per-leaf fields :meth:`AdaptiveMesh.refine` carries to the new leaves
FIELDS = ("T", "k", "rho", "cp", "Q_source", "Q_sink", "material_id", "boundary_type",
          "bc_h", "bc_T_inf", "excluded", "source_mask")


# --------------------------------------------------------------- construction
@dataclass(frozen=True)
class RefinementBand:
    """A box of the domain that needs cells of a given size [m].

    The adaptive analogue of :class:`~src.core.refinement.Band`, in three dimensions: a
    region (a hot spot, a wall layer) rather than a slice of one axis.  A leaf is refined
    while it intersects the box and is still larger than ``size``.
    """

    low: tuple[float, float, float]
    high: tuple[float, float, float]
    size: float

    def intersects(self, leaf: Leaf, physical_size: float) -> bool:
        """True when the leaf's box overlaps this band (a touch on a plane counts)."""
        corners = (leaf.x, leaf.y, leaf.z)
        for axis in range(3):
            low = corners[axis] * physical_size
            high = (corners[axis] + leaf.size) * physical_size
            if low >= self.high[axis] or high <= self.low[axis]:
                return False
        return True


def _check_bands(bands: Sequence[RefinementBand]) -> None:
    """Reject a band list no refinement can satisfy, before a tree is built."""
    if not bands:
        raise ValueError("at least one refinement band is required")
    for band in bands:
        if band.size <= 0.0:
            raise ValueError(f"band size must be > 0, got {band.size}")
        for low, high in zip(band.low, band.high, strict=True):
            if high <= low:
                raise ValueError(f"empty band {band}")


# ------------------------------------------------------------------- results
@dataclass
class AdaptiveBalance:
    """Energy balance of an adaptive mesh, in the vocabulary of ``src.analysis.balance``.

    ``p_source``/``p_sink`` are the powers the sources deposit in the free leaves [W]
    (positive deposits, negative extractions), ``q_fixed`` the heat leaving them through
    the fixed (Dirichlet) leaves, and ``p_film`` the heat leaving through the films -
    broken down into the box faces, the environment and the fluid.  The identity
    ``p_source + p_sink == q_fixed + p_film`` is the self-check: it holds at the round-off
    of the per-leaf closure, which :attr:`residual` reports in watts.
    """

    t_ambient: float
    p_source: float
    p_sink: float
    q_faces: dict[str, float]
    q_environment: float
    q_tubes: float
    q_fixed: float
    p_film: float
    residual: float
    closure: float
    cells: int

    def summary(self) -> str:
        return (f"source {self.p_source:.6g} W, sink {self.p_sink:.6g} W, "
                f"fixed {self.q_fixed:.6g} W, film {self.p_film:.6g} W "
                f"(closure {self.closure:.2e}, residual {self.residual:.2e} W on "
                f"{self.cells:,} leaves)")


@dataclass
class AdaptiveSteadyResult:
    """Steady field on an adaptive mesh, the fluxes it drives and its balance."""

    T: np.ndarray                 # per-leaf temperature [K], in ``tree.leaves`` order
    face_flux: np.ndarray         # heat rate [W] per entry of ``tree.faces()``
    divergence: np.ndarray        # net conduction rate leaving every leaf [W]
    balance: AdaptiveBalance
    converged: bool
    residual: float               # relative residual of the linear solve
    iterations: int
    solve_time: float
    notes: list[str] = field(default_factory=list)
    cells: int = 0

    def summary(self) -> str:
        return (f"adaptive {self.cells:,} leaves: T {self.T.min():.2f}-{self.T.max():.2f} K, "
                f"{self.balance.summary()}, solved in {self.solve_time * 1000:.1f} ms")


class _FaceTable(NamedTuple):
    """The face list with the conductances the physics asks for (see ``_face_table``)."""

    i: np.ndarray          # first cell of the entry
    j: np.ndarray          # second cell
    axis: np.ndarray       # axis the face is normal to
    area: np.ndarray       # face area [m^2]
    distance: np.ndarray   # centre-to-centre distance [m]
    g_base: np.ndarray     # harmonic-mean conductance of ``Octree.diffusion_matrix`` [W/K]
    g: np.ndarray          # conductance the physics asks for [W/K]


# --------------------------------------------------------------------- the mesh
@dataclass
class AdaptiveMesh:
    """A cell-centred mesh on an octree: per-leaf fields plus the six box faces.

    The tree is the geometry; every field is per leaf and aligned with ``tree.leaves``.
    The defaults mirror :class:`~src.core.mesh.Mesh3D` (air properties, the ambient
    initial field, insulated sides and a ground face at ``T_GROUND_DEFAULT``) so that a
    consumer inspecting a fresh mesh sees what it saw before.
    """

    tree: Octree
    physical_size: float = 1.0

    # per-leaf state
    T: np.ndarray = field(init=False, repr=False)
    k: np.ndarray = field(init=False, repr=False)
    rho: np.ndarray = field(init=False, repr=False)
    cp: np.ndarray = field(init=False, repr=False)
    Q_source: np.ndarray = field(init=False, repr=False)
    Q_sink: np.ndarray = field(init=False, repr=False)
    material_id: np.ndarray = field(init=False, repr=False)
    boundary_type: np.ndarray = field(init=False, repr=False)
    bc_h: np.ndarray = field(init=False, repr=False)
    bc_T_inf: np.ndarray = field(init=False, repr=False)
    excluded: np.ndarray = field(init=False, repr=False)
    source_mask: np.ndarray = field(init=False, repr=False)

    # the environment
    h_out: float = 0.0
    t_ambient: float = T_AMBIENT_DEFAULT
    h_contact: float = 0.0
    face_bc: dict[str, FaceBC] = field(init=False, repr=False, default_factory=dict)

    # derived, rebuilt whenever the tree changes
    sizes: np.ndarray = field(init=False, repr=False)
    V: np.ndarray = field(init=False, repr=False)
    on_box_face: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.physical_size <= 0.0:
            raise ValueError(f"physical_size must be > 0, got {self.physical_size}")
        self.face_bc = {face: FaceBC() for face in FACES}
        self.face_bc["z_min"] = FaceBC(BoundaryType.DIRICHLET, value=T_GROUND_DEFAULT)
        self._allocate(self.tree.n_cells)

    def _allocate(self, n: int) -> None:
        """Fresh per-leaf fields of ``n`` leaves, with the ``Mesh3D`` defaults."""
        self.T = np.full(n, T_INITIAL_DEFAULT)
        self.k = np.full(n, K_AIR)
        self.rho = np.full(n, RHO_AIR)
        self.cp = np.full(n, CP_AIR)
        self.Q_source = np.zeros(n)
        self.Q_sink = np.zeros(n)
        self.material_id = np.zeros(n, dtype=np.int8)
        self.boundary_type = np.zeros(n, dtype=np.int8)
        self.bc_h = np.zeros(n)
        self.bc_T_inf = np.full(n, T_AMBIENT_DEFAULT)
        self.excluded = np.zeros(n, dtype=bool)
        self.source_mask = np.zeros(n, dtype=bool)
        self._rebuild_caches()

    def _rebuild_caches(self) -> None:
        """Leaf edges, volumes and the "touches a box face" mask of the current tree."""
        self.sizes = self.tree.cell_sizes() * self.physical_size
        self.V = self.sizes ** 3
        n = self.tree.n
        self.on_box_face = np.array(
            [any(corner == 0 or corner + leaf.size == n
                 for corner in (leaf.x, leaf.y, leaf.z))
             for leaf in self.tree.leaves], dtype=bool)

    @classmethod
    def uniform(cls, n_finest: int, physical_size: float, level: int) -> AdaptiveMesh:
        """Every leaf of the same level, ``2 ** level`` finest cells on a side."""
        return cls(uniform_tree(n_finest, level), physical_size)

    @classmethod
    def from_bands(cls, n_finest: int, physical_size: float,
                   bands: Sequence[RefinementBand],
                   base_level: int | None = None) -> AdaptiveMesh:
        """A tree refined to the band sizes, starting from ``base_level`` (default: root).

        Refinement is the octree's own: a leaf is split while it intersects a band and is
        still larger than that band's target, and :meth:`Octree.balance` keeps the 2:1 rule
        after every round, so a band boundary is a staircase of at most one level per leaf
        and never a hanging node.
        """
        _check_bands(bands)
        level = Octree(n_finest).max_level if base_level is None else base_level
        mesh = cls(uniform_tree(n_finest, level), physical_size)
        mesh.refine_bands(bands)
        return mesh

    @classmethod
    def from_indicator(cls, n_finest: int, physical_size: float,
                       indicator: Callable[[Leaf], float], threshold: float,
                       levels: int = 1, base_level: int | None = None) -> AdaptiveMesh:
        """A tree refined where a rule of the leaf says so (:meth:`Octree.refine`)."""
        level = Octree(n_finest).max_level if base_level is None else base_level
        mesh = cls(uniform_tree(n_finest, level), physical_size)
        mesh.refine(indicator, threshold, levels=levels)
        return mesh

    # ------------------------------------------------------------- refinement
    def refine_bands(self, bands: Sequence[RefinementBand]) -> None:
        """Refine to the band sizes, in place (see :meth:`from_bands`).

        The rounds run to ``tree.max_level`` at most; the octree stops as soon as no leaf
        is still above its band target, so the cost is the depth actually needed.  The
        per-leaf fields are carried (zeroth order) as in :meth:`refine`.
        """
        _check_bands(bands)

        def indicator(leaf: Leaf) -> float:
            target = min((band.size for band in bands
                          if band.intersects(leaf, self.physical_size)), default=np.inf)
            return 1.0 if leaf.size * self.physical_size > target else 0.0

        self.refine(indicator, 0.5, levels=self.tree.max_level)

    def refine(self, indicator: np.ndarray | Callable[[Leaf], float], threshold: float,
               levels: int = 1) -> None:
        """Refine the tree and carry every per-leaf field to the children.

        ``indicator`` is a per-leaf magnitude in ``tree.leaves`` order (what a solution
        driven round produces) or a rule of the leaf, as :meth:`Octree.refine` takes it.
        A split leaf hands its value to its eight children unchanged - zeroth order, which
        is what refining a *field* wants - so a caller whose properties come from a rule of
        position (a material, a source density) re-evaluates them after the call.  Cost,
        boundary types and the environment survive the round: a leaf that was excluded
        stays excluded.
        """
        if not callable(indicator):
            values = np.asarray(indicator, dtype=float)
            if values.shape != (self.tree.n_cells,):
                raise ValueError(f"indicator must have one value per leaf "
                                 f"({self.tree.n_cells}), got shape {values.shape}")
            table = dict(zip(self.tree.leaves, values, strict=True))
            rule: Callable[[Leaf], float] = table.__getitem__
        else:
            rule = indicator
        previous = {name: getattr(self, name).copy() for name in FIELDS}
        ancestors = {leaf: position for position, leaf in enumerate(self.tree.leaves)}
        self.tree.refine(rule, threshold, levels=levels)
        mapping = np.array([_ancestor_leaf(leaf, ancestors, self.tree.max_level)
                            for leaf in self.tree.leaves])
        for name in FIELDS:
            setattr(self, name, previous[name][mapping])
        self._rebuild_caches()

    # ------------------------------------------------------- boundary conditions
    def set_adiabatic(self, face: str) -> None:
        """Zero-flux (insulated) domain face."""
        self._check_face(face)
        self.face_bc[face] = FaceBC(BoundaryType.INTERNAL)

    def set_fixed_temperature_bc(self, face: str, temperature: float) -> None:
        """Dirichlet temperature on a domain face; ``temperature`` in KELVIN."""
        self._check_face(face)
        check_kelvin(temperature, f"set_fixed_temperature_bc({face})")
        self.face_bc[face] = FaceBC(BoundaryType.DIRICHLET, value=float(temperature))

    def set_convection_bc(self, face: str, h: float, t_inf: float) -> None:
        """Convection on a domain face; ``t_inf`` in KELVIN."""
        self._check_face(face)
        check_kelvin(t_inf, f"set_convection_bc({face}) t_inf")
        self.face_bc[face] = FaceBC(BoundaryType.CONVECTION, h=float(h),
                                    value=float(t_inf))

    def set_heat_flux_bc(self, face: str, q: float) -> None:
        """Imposed heat flux [W/m^2] entering the domain through ``face``."""
        self._check_face(face)
        self.face_bc[face] = FaceBC(BoundaryType.NEUMANN, h=1.0, value=float(q))

    def set_internal_convection(self, mask: np.ndarray, h: float, t_fluid: float) -> None:
        """Convection on interior cells (heat-exchanger tubes); ``t_fluid`` in KELVIN."""
        check_kelvin(t_fluid, "set_internal_convection t_fluid")
        m = np.asarray(mask, dtype=bool)
        if m.shape != (self.tree.n_cells,):
            raise ValueError(f"mask must have one entry per leaf ({self.tree.n_cells}), "
                             f"got shape {m.shape}")
        self.bc_h[m] = float(h)
        self.bc_T_inf[m] = float(t_fluid)
        self.boundary_type[m] = int(BoundaryType.CONVECTION if h > 0.0
                                    else BoundaryType.INTERNAL)

    @staticmethod
    def _check_face(face: str) -> None:
        if face not in FACES:
            raise ValueError(f"invalid face {face!r}; expected one of {FACES}")

    # ---------------------------------------------------------------- geometry
    @property
    def n_cells(self) -> int:
        return self.tree.n_cells

    @property
    def h_char(self) -> np.ndarray:
        """Characteristic leaf size ``V^(1/3)`` [m], as on :class:`Mesh3D`."""
        return np.cbrt(self.V)

    def fixed_leaves(self) -> dict[int, float]:
        """The Dirichlet set as ``{leaf position: temperature}``.

        The leaves a ``face_bc`` of kind DIRICHLET drives (``wall_cells`` of
        :mod:`src.solver.octree_solver`) plus the excluded leaves held at
        :attr:`t_ambient` - the same set and the same values ``src/solver/matrix.py``
        builds for a structured mesh.
        """
        fixed: dict[int, float] = {int(position): float(self.t_ambient)
                                   for position in np.flatnonzero(self.excluded)}
        for face, bc in self.face_bc.items():
            if bc.kind == BoundaryType.DIRICHLET:
                for position in self.wall_indices(face):
                    fixed[int(position)] = float(bc.value)
        return fixed

    def wall_indices(self, face: str) -> np.ndarray:
        """Positions of the leaves touching a box face, in ``tree.leaves`` order.

        The leaves are the ones :func:`src.solver.octree_solver.wall_cells` names - the set
        a face condition drives - and the positions are what the flat per-leaf arrays use.
        """
        positions = {leaf: index for index, leaf in enumerate(self.tree.leaves)}
        return np.array([positions[leaf] for leaf in wall_cells(self.tree, face)],
                        dtype=int)

    def centre(self, position: int) -> tuple[float, float, float]:
        """Centre of one leaf in physical coordinates [m]."""
        return tuple(float(c * self.physical_size)
                     for c in self.tree.leaves[position].centre)

    def centres(self) -> np.ndarray:
        """``(n_cells, 3)`` leaf centres in physical coordinates [m]."""
        return self.tree.cell_centres() * self.physical_size

    def faces(self) -> Sequence[FaceRow]:
        """The conservative face list of :class:`MeshAPI`, straight from the octree."""
        return self.tree.faces()

    # -------------------------------------------------------------------- films
    def _films(self, faces: _FaceTable | None = None
               ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """``(name, positions, coefficient per volume [1/s], reference temperature [K])``.

        The three surface films of the structured solver, with its formulas: a convective
        box face uses the half-cell conductance ``h_eff = 2kh/(2k + h d)`` of the leaf
        sitting on it, the environment film uses ``h_out A / V`` on the active side of an
        active/excluded face, and a tube cell uses ``bc_h / V^(1/3)``.  A box face takes
        its own fluid temperature, the environment film the ambient, a tube cell the fluid
        temperature of its own ``bc_T_inf``.
        """
        out: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for face, bc in self.face_bc.items():
            if bc.kind == BoundaryType.CONVECTION and bc.is_active():
                positions = self.wall_indices(face)
                if positions.size == 0:
                    continue
                size = self.sizes[positions]
                coefficient = half_cell_h(self.k[positions], bc.h, size) / size
                out.append((face, positions, np.asarray(coefficient, dtype=float),
                            np.full(positions.size, float(bc.value))))

        if self.h_out > 0.0 and self.excluded.any():
            table = self._face_table() if faces is None else faces
            active_i = ~self.excluded[table.i] & self.excluded[table.j]
            active_j = ~self.excluded[table.j] & self.excluded[table.i]
            positions = np.concatenate((table.i[active_i], table.j[active_j]))
            if positions.size:
                areas = np.concatenate((table.area[active_i], table.area[active_j]))
                out.append(("environment", positions,
                            self.h_out * areas / self.V[positions],
                            np.full(positions.size, float(self.t_ambient))))

        tube = is_interior_tube(self.boundary_type, self.on_box_face)
        if tube.any():
            positions = np.flatnonzero(tube)
            out.append(("tube", positions, self.bc_h[positions] / self.h_char[positions],
                        self.bc_T_inf[positions]))
        return out

    # ----------------------------------------------------------------- assembly
    def _face_table(self) -> _FaceTable:
        """The face list with the conductance the physics asks for, per entry.

        ``g_base`` is what :meth:`Octree.diffusion_matrix` assembles (the same harmonic
        mean, the same products, the same order); ``g`` is what the physics asks for: a
        face touching an excluded leaf carries no conduction at all (the structured
        assembly zeroes it, ``src/solver/matrix.py:58``), and an interface between two
        materials carries the contact resistance in series with the two half leaves
        (``src/solver/matrix.py:66-75``).  Everywhere else ``g == g_base``, which is what
        keeps this mesh's flux report equal to the octree solver's.
        """
        faces = np.asarray(self.tree.faces(), dtype=float).reshape(-1, 5)
        i = faces[:, 0].astype(int)
        j = faces[:, 1].astype(int)
        area = faces[:, 3] * self.physical_size ** 2
        distance = faces[:, 4] * self.physical_size
        k_face = 2.0 * self.k[i] * self.k[j] / (self.k[i] + self.k[j] + EPS)
        g_base = k_face * area / distance
        g = g_base
        if self.h_contact > 0.0:
            d_i, d_j = self.sizes[i], self.sizes[j]
            r_series = (0.5 * d_i / (self.k[i] + EPS) + 1.0 / self.h_contact
                        + 0.5 * d_j / (self.k[j] + EPS))
            k_eff = 0.5 * (d_i + d_j) / r_series
            g = np.where(self.material_id[i] != self.material_id[j],
                         k_eff * area / distance, g)
        if self.excluded.any():
            g = np.where(self.excluded[i] | self.excluded[j], 0.0, g)
        return _FaceTable(i=i, j=j, axis=faces[:, 2].astype(int), area=area,
                          distance=distance, g_base=g_base, g=g)

    def face_conductances(self, faces: _FaceTable | None = None) -> np.ndarray:
        """The conductance [W/K] of every face entry, as the assembly uses it."""
        table = self._face_table() if faces is None else faces
        return table.g

    def face_fluxes(self, temperature: np.ndarray | None = None,
                    faces: _FaceTable | None = None) -> np.ndarray:
        """Heat rate [W] through every face entry, positive from the first to the second."""
        table = self._face_table() if faces is None else faces
        values = self.T if temperature is None else np.asarray(temperature, dtype=float)
        return table.g * (values[table.i] - values[table.j])

    def divergence(self, temperature: np.ndarray | None = None,
                   faces: _FaceTable | None = None) -> np.ndarray:
        """Net conduction heat rate [W] leaving every leaf.

        Each face rate is accumulated once as an outflow and once as an inflow - the
        one-conductance-per-face bookkeeping :class:`OctreeSteadySolver` reports - so
        summing this over the free leaves leaves only what crosses into the fixed ones.
        """
        table = self._face_table() if faces is None else faces
        values = self.T if temperature is None else np.asarray(temperature, dtype=float)
        flux = table.g * (values[table.i] - values[table.j])
        out = np.zeros(self.tree.n_cells)
        np.add.at(out, table.i, flux)
        np.add.at(out, table.j, -flux)
        return out

    def indicator(self, temperature: np.ndarray | None = None) -> np.ndarray:
        """Per-leaf flux-jump estimate [W], taken from :class:`OctreeSteadySolver`.

        The local error a refinement round attacks: the axial flux imbalance of every
        leaf, i.e. the discrete curvature of the field.  It is asked of the octree solver
        rather than re-derived, so a tree refined here refines the way
        ``refine_on_objective`` would refine it.
        """
        values = self.T if temperature is None else np.asarray(temperature, dtype=float)
        solver = OctreeSteadySolver(self.tree, conductivity=self.k,
                                    physical_size=self.physical_size)
        return solver.indicator(values, self.tree)

    def matrix(self, faces: _FaceTable | None = None) -> sparse.csr_matrix:
        """The conduction operator per unit volume [W/(m^3 K)].

        :meth:`Octree.diffusion_matrix` assembled the octree, plus the correction of
        :meth:`_face_table` - a correction that is empty (the operator *is* the octree
        one) unless ``h_contact > 0`` or a leaf is excluded.

        The correction is added to the triplets of the octree operator rather than to the
        assembled matrix: on a face whose conductance is removed the diagonal cancels to
        zero, a sparse *sum* drops that entry, and the row is then left with no stored
        diagonal for the Dirichlet elimination to replace (see the note in
        ``docs/16_ADAPTIVE_MESH_MIGRATION.md``).
        """
        table = self._face_table() if faces is None else faces
        base = self.tree.diffusion_matrix(self.k, self.physical_size).tocoo()
        changed = table.g != table.g_base
        if not changed.any():
            return base.tocsr()
        i, j = table.i[changed], table.j[changed]
        delta = table.g[changed] - table.g_base[changed]
        delta_i = delta / self.V[i]
        delta_j = delta / self.V[j]
        rows = np.concatenate((base.row, i, i, j, j))
        cols = np.concatenate((base.col, i, j, j, i))
        vals = np.concatenate((base.data, delta_i, -delta_i, delta_j, -delta_j))
        return sparse.coo_matrix((vals, (rows, cols)), shape=base.shape).tocsr()

    def assemble(self, enforce_dirichlet: bool = True
                 ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """``(A, b)`` of ``div(k grad T) + Q = 0`` per unit volume, Dirichlet applied.

        The films go on the diagonal with their reference temperatures on the right-hand
        side, exactly as :mod:`src.solver.matrix` does for a structured mesh; a Neumann
        face adds ``q/d`` with the local leaf edge ``d``.  The Dirichlet rows are the
        leaves :meth:`fixed_leaves` names, eliminated symmetrically by
        :func:`src.solver.matrix.apply_dirichlet` - the same elimination, so a fixed leaf
        is exact instead of relaxed.
        """
        faces = self._face_table()
        a = self.matrix(faces).tocsr()
        b = (self.Q_source + self.Q_sink).astype(float).copy()
        diagonal = np.zeros(self.tree.n_cells)
        for _name, positions, coefficient, reference in self._films(faces):
            np.add.at(diagonal, positions, coefficient)
            np.add.at(b, positions, coefficient * reference)
        for face, bc in self.face_bc.items():
            if bc.kind == BoundaryType.NEUMANN and bc.value != 0.0:
                positions = self.wall_indices(face)
                if positions.size:
                    b[positions] += bc.value / self.sizes[positions]
        if diagonal.any():
            # the film joins the triplet list of the conduction operator instead of being
            # added to the assembled matrix: a sparse *sum* drops the explicit zeros that
            # keep a leaf without conduction (an excluded one) addressable, and the
            # elimination below expects every fixed row to have a stored diagonal
            coo = a.tocoo()
            index = np.arange(self.tree.n_cells)
            a = sparse.coo_matrix(
                (np.concatenate((coo.data, diagonal)),
                 (np.concatenate((coo.row, index)), np.concatenate((coo.col, index)))),
                shape=coo.shape).tocsr()
        if not enforce_dirichlet:
            return a, b
        fixed = self.fixed_leaves()
        mask = np.zeros(self.tree.n_cells, dtype=bool)
        values = np.zeros(self.tree.n_cells)
        for position, value in fixed.items():
            mask[position] = True
            values[position] = value
        return apply_dirichlet(a, b, mask, values), b

    # ------------------------------------------------------------------- solve
    def solve_steady(self, config: LinearConfig | None = None) -> AdaptiveSteadyResult:
        """Solve the steady problem and report the field, the fluxes and the balance.

        The linear layer is the structured one (:func:`src.solver.linear.solve_linear`), so
        a graded tree is solved with the volume symmetrisation the structured solver uses
        and the method and tolerance are the caller's.  The default is a direct
        factorisation, which is exact enough to compare two meshes.
        """
        start = time.perf_counter()
        a, b = self.assemble()
        sizes = self.tree.cell_sizes()
        uniform = float(sizes.min()) == float(sizes.max())
        linear: LinearResult = solve_linear(
            a, b, config or LinearConfig(method="direct"),
            scale=None if uniform else self.V)
        self.T = np.asarray(linear.T, dtype=float)
        return AdaptiveSteadyResult(
            T=self.T.copy(), face_flux=self.face_fluxes(self.T),
            divergence=self.divergence(self.T), balance=self.balance(),
            converged=linear.converged, residual=linear.residual,
            iterations=linear.iterations, solve_time=time.perf_counter() - start,
            notes=list(linear.notes), cells=self.tree.n_cells)

    # ----------------------------------------------------------------- balance
    def balance(self) -> AdaptiveBalance:
        """Energy balance of the current field, closing at the round-off of the scheme.

        Every term is the one the assembly applied: the power of the sources in the free
        leaves, the conduction leaving them into the fixed ones, and the films.  The
        identity ``p_source + p_sink == q_fixed + p_film`` holds at the round-off of the
        per-leaf closure because each term is summed from the same coefficients the
        operator carries - which is what :attr:`AdaptiveBalance.residual` reports.
        """
        faces = self._face_table()
        films = self._films(faces)
        fixed = self.fixed_leaves()
        free = np.ones(self.tree.n_cells, dtype=bool)
        free[list(fixed)] = False

        divergence = self.divergence(self.T, faces)
        per_leaf = divergence.copy()
        rates: dict[str, float] = {}
        for name, positions, coefficient, reference in films:
            rate = coefficient * self.V[positions] * (self.T[positions] - reference)
            np.add.at(per_leaf, positions, rate)
            rates[name] = float(np.sum(rate))
        closure_error = per_leaf[free] - ((self.Q_source + self.Q_sink) * self.V)[free]

        p_source = float(np.sum(self.Q_source[free] * self.V[free]))
        p_sink = float(np.sum(self.Q_sink[free] * self.V[free]))
        p_film = float(sum(rates.values()))
        q_fixed = float(divergence[free].sum())
        # relative to the largest rate the mesh carries, not to the power: a wall that
        # both heats and cools the domain has a source power of zero and a ratio against
        # it would say nothing (the same choice octree_solver makes)
        carried = np.abs(self.face_fluxes(self.T, faces))
        scale = max(abs(p_source), abs(p_sink), abs(q_fixed), abs(p_film),
                    float(carried.max()) if carried.size else 0.0, EPS)
        return AdaptiveBalance(
            t_ambient=float(self.t_ambient), p_source=p_source, p_sink=p_sink,
            q_faces={name: value for name, value in rates.items() if name in FACES},
            q_environment=rates.get("environment", 0.0), q_tubes=rates.get("tube", 0.0),
            q_fixed=q_fixed, p_film=p_film,
            residual=float(np.abs(closure_error).max()) if closure_error.size else 0.0,
            closure=abs(p_source + p_sink - q_fixed - p_film) / scale,
            cells=self.tree.n_cells)

    # -------------------------------------------------------------------- info
    def level_histogram(self) -> dict[int, int]:
        """Leaves per level, as :meth:`Octree.level_histogram` reports them."""
        return self.tree.level_histogram()

    def summary(self) -> str:
        levels = ", ".join(f"L{level}: {count}"
                           for level, count in self.level_histogram().items())
        return (f"adaptive mesh: {self.tree.n_cells} leaves ({levels}), "
                f"leaf edge {self.sizes.min():.4f}-{self.sizes.max():.4f} m, "
                f"{len(self.tree.faces())} faces")


def _ancestor_leaf(leaf: Leaf, ancestors: dict[Leaf, int], max_level: int) -> int:
    """Position of the previous-tree leaf containing ``leaf`` (a split's parent).

    A leaf of the new tree is inside exactly one leaf of the previous one, and the walk
    upwards ends at ``max_level`` at the latest - the leaf that is the whole box, which
    any tree of the same box has.
    """
    for level in range(leaf.level, max_level + 1):
        corner = Leaf(level, (leaf.x >> level) << level, (leaf.y >> level) << level,
                      (leaf.z >> level) << level)
        position = ancestors.get(corner)
        if position is not None:
            return position
    raise ValueError(f"no leaf of the previous tree contains {leaf}")
