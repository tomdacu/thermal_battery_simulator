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
  the per-leaf closure ``div + film - Q V``;
* the transient is the same assembly: :meth:`AdaptiveMesh.transient_operators` adds the
  mass matrix ``rho*cp`` to the steady operator and eliminates the same pinned leaves, and
  :meth:`AdaptiveMesh.transient_rhs` rebuilds the right-hand side of the current state -
  so the backward-Euler march of :mod:`src.solver.transient` runs on a tree without
  knowing it is not on a ``Mesh3D``.
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
from ..solver.matrix import apply_dirichlet, environment_film, face_conductance
from ..solver.octree_solver import (_NOISE_FLOOR, OctreeSteadySolver, _mark,
                                     wall_cells)
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
    #: the convection-only part of ``h_out`` and the emissivity of the outer surface, as
    #: :meth:`~src.core.geometry.BatteryGeometry.apply_environment` paints them.  ``None``
    #: means no base was recorded and the film the mesh carries is the convective one; the
    #: assembly re-evaluates the radiative share of the film at the field it solves
    #: (see :func:`src.solver.matrix.environment_film`)
    h_out_conv: float | None = None
    environment_emissivity: float = 0.0
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
    def refine_bands(self, bands: Sequence[RefinementBand],
                     max_cells: int | None = None) -> None:
        """Refine to the band sizes, in place (see :meth:`from_bands`).

        The rounds run to ``tree.max_level`` at most; the octree stops as soon as no leaf
        is still above its band target, so the cost is the depth actually needed.  The
        per-leaf fields are carried (zeroth order) as in :meth:`refine`.

        ``max_cells`` stops the refinement *between* two rounds, before a round that would
        not fit: a request scaled past the budget of the caller is not answered with a mesh
        the caller cannot afford, and the tree is left balanced and whole.  ``from_bands``
        never passes one - the a priori plan is a request, not a budget - while a search
        that scales the plan does.
        """
        _check_bands(bands)

        def indicator(leaf: Leaf) -> float:
            target = min((band.size for band in bands
                          if band.intersects(leaf, self.physical_size)), default=np.inf)
            return 1.0 if leaf.size * self.physical_size > target else 0.0

        if max_cells is None:
            self.refine(indicator, 0.5, levels=self.tree.max_level)
            return
        for _ in range(self.tree.max_level):
            marked = sum(1 for leaf in self.tree.leaves
                         if leaf.level > 0 and indicator(leaf) > 0.5)
            if marked == 0:
                return                        # every leaf already meets its band
            if self.n_cells + 7 * marked > max_cells:
                return                        # the round would not fit: stop here
            self.refine(indicator, 0.5)

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

    def refine_round(self, refine_fraction: float = 0.1, floor_ratio: float = 0.05,
                     max_cells: int | None = None) -> int:
        """Refine one round on the flux-jump indicator of the current field.

        The rule is the one :func:`src.solver.octree_solver.refine_on_objective` applies,
        asked of the mesh instead of the tree: the leaves whose jump is above
        ``floor_ratio`` of the largest - at most ``refine_fraction`` of the tree and never
        below the round-off level of the fluxes it carries - are split.  ``self.T`` must
        hold a solved field: the indicator is the curvature of *that* field, and a round
        drawn from a field that was never solved would refine round-off.

        One round, not a loop: the estimate belongs to the field in hand, so the caller
        that wants to go deeper has to solve again (which is what the levels of
        :func:`src.analysis.convergence.find_mesh` do) or ask for a finer *request*
        instead - the bands of the plan are the cheaper way to say "this region is not
        resolved yet".

        Returns the number of leaves the round marked, or 0 when nothing was refined -
        either the field carries no feature above the floor, or the round would not fit
        the cell budget, in which case the tree is left exactly as it was.
        """
        values = np.abs(np.asarray(self.indicator(), dtype=float))
        flux = np.abs(self.face_fluxes(self.T))
        threshold, marked = _mark(
            self.tree, values, refine_fraction, floor_ratio,
            noise=_NOISE_FLOOR * (float(flux.max()) if flux.size else 0.0))
        if marked == 0:
            return 0
        # the budget is checked before the round: a marked leaf becomes eight leaves, so
        # the round adds ``7 * marked`` of them (the balance adds a few more along the
        # boundary of the flagged region, which the budget the caller set still bounds)
        if max_cells is not None and self.n_cells + 7 * marked > max_cells:
            return 0
        self.refine(values, threshold)
        return marked

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
    def box_size(self) -> float:
        """Edge of the (cubic) domain [m].

        ``physical_size`` is the edge of one *finest* cell, so the box is ``n_finest``
        of them: this is the ``Lx = Ly = Lz`` a structured consumer reads.
        """
        return float(self.physical_size) * self.tree.n

    @property
    def h_char(self) -> np.ndarray:
        """Characteristic leaf size ``V^(1/3)`` [m], as on :class:`Mesh3D`."""
        return np.cbrt(self.V)

    def fixed_leaves(self) -> dict[int, float]:
        """The Dirichlet set as ``{leaf position: temperature}``.

        The leaves a ``face_bc`` of kind DIRICHLET drives (``wall_cells`` of
        :mod:`src.solver.octree_solver`) plus the excluded leaves held at
        :attr:`t_ambient` - the same set and the same values ``src/solver/matrix.py``
        builds for a structured mesh, which applies the ambient *after* the faces: an
        excluded leaf carries no physics, so where the two rules meet the placeholder
        wins and both meshes hold the same field.
        """
        fixed = {int(position): float(bc.value)
                 for face, bc in self.face_bc.items()
                 if bc.kind == BoundaryType.DIRICHLET
                 for position in self.wall_indices(face)}
        fixed.update({int(position): float(self.t_ambient)
                      for position in np.flatnonzero(self.excluded)})
        return fixed

    def fixed_mask(self) -> np.ndarray:
        """``(n_cells,)`` mask of the leaves the Dirichlet elimination pins.

        The boolean view of :meth:`fixed_leaves`, for a consumer that filters a per-leaf
        array rather than iterating the set: a pinned leaf has an identity row, so an
        exchange with it - a film on it, a source deposited in it - never reaches the
        solution and the reports must not count it.
        """
        mask = np.zeros(self.tree.n_cells, dtype=bool)
        mask[list(self.fixed_leaves())] = True
        return mask

    def _fixed_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(mask, values)`` of the pinned leaves, in the form the elimination takes.

        :meth:`fixed_leaves` as the two per-leaf arrays
        :func:`src.solver.matrix.apply_dirichlet` wants, which is how both assemblies -
        the steady one and the transient one - pin the same set.
        """
        mask = np.zeros(self.tree.n_cells, dtype=bool)
        values = np.zeros(self.tree.n_cells)
        for position, value in self.fixed_leaves().items():
            mask[position] = True
            values[position] = value
        return mask, values

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

    def locate(self, x: float, y: float, z: float) -> int:
        """Position of the leaf containing ``(x, y, z)``, clamped to the box.

        The tree's ``Mesh3D.find_cell``, and by the same convention: a leaf owns its
        lower corner and a point on a leaf face belongs to the leaf *above* it, while a
        point outside the box belongs to the boundary leaf.  A consumer that measures the
        local resolution at a coordinate therefore gets the same answer from either mesh.
        The walk is the octree's own (``Octree._locate``, public at step 6 of the
        migration); the coordinate is converted to the finest-cell index the leaf list is
        written in (``physical_size`` is the edge of one finest cell).
        """
        n = self.tree.n
        corner = tuple(int(np.clip(np.floor(value / self.physical_size), 0, n - 1))
                       for value in (x, y, z))
        position = self.tree._locate(*corner)
        if position < 0:                                  # cannot happen on a full box
            raise ValueError(f"no leaf contains ({x}, {y}, {z}) m")
        return int(position)

    def cell_size_at(self, x: float, y: float, z: float) -> float:
        """Characteristic size ``V^(1/3)`` of the leaf containing a point [m].

        The counterpart of :meth:`src.core.mesh.Mesh3D.cell_size_at`, which is how a
        painter measures a zone against the cells that are actually there.
        """
        return float(self.h_char[self.locate(x, y, z)])

    def faces(self) -> Sequence[FaceRow]:
        """The conservative face list of :class:`MeshAPI`, straight from the octree."""
        return self.tree.faces()

    def face_rows(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                 np.ndarray, np.ndarray]:
        """``faces()`` in metres, with the conductance the assembly applies to each entry.

        ``Octree.faces`` writes its areas and distances in *finest-cell* units (the caller
        multiplies by :attr:`physical_size`), so the protocol's list cannot be read as a
        heat rate without scaling it; this is the same list - the same entries, in the
        same order - with the areas in m^2, the distances in m, and the conductance
        [W/K] of :meth:`_face_table`, i.e. :func:`src.solver.matrix.face_conductance` with
        this mesh's contact resistance and its excluded leaves.  ``g (T_i - T_j)`` is then
        one heat rate per interface, the number the operator carries, which is what the
        flux report of :mod:`src.analysis.fluxes` integrates.
        """
        table = self._face_table()
        return (table.i, table.j, table.axis, table.area, table.distance, table.g)

    # -------------------------------------------------------------------- films
    def _environment_surface(self, faces: _FaceTable | None = None
                             ) -> tuple[np.ndarray, np.ndarray]:
        """``(positions, areas)`` of the leaves the environment film acts on.

        The active side of every active/excluded face: a leaf facing two excluded
        neighbours appears twice, once per face and with that face's area, which is
        exactly how the film is applied and how the flux is reported.
        """
        table = self._face_table() if faces is None else faces
        active_i = ~self.excluded[table.i] & self.excluded[table.j]
        active_j = ~self.excluded[table.j] & self.excluded[table.i]
        return (np.concatenate((table.i[active_i], table.j[active_j])),
                np.concatenate((table.area[active_i], table.area[active_j])))

    def outer_film(self, faces: _FaceTable | None = None,
                   radiation: bool = False) -> float:
        """The film coefficient [W/(m^2 K)] the environment film carries right now.

        Zero when the mesh has no environment film (no excluded leaf, or ``h_out``
        zero).  Otherwise it is :func:`src.solver.matrix.environment_film` evaluated at
        the surface temperature of the current field, area-weighted over
        :meth:`_environment_surface` - the fixed point of the film with its own driving
        temperature, which the assembly re-evaluates once per build.
        """
        if self.h_out <= 0.0 or not self.excluded.any():
            return 0.0
        positions, areas = self._environment_surface(faces)
        if positions.size == 0:
            return 0.0
        return environment_film(self, positions, areas, radiation)

    def _films(self, faces: _FaceTable | None = None
               ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """``(name, positions, coefficient per volume [1/s], reference temperature [K])``.

        The three surface films of the structured solver, with its formulas: a convective
        box face uses the half-cell conductance ``h_eff = 2kh/(2k + h d)`` of the leaf
        sitting on it, the environment film uses ``h_out A / V`` on the active side of an
        active/excluded face, and a tube cell uses ``bc_h / V^(1/3)``.  A box face takes
        its own fluid temperature, the environment film the ambient, a tube cell the fluid
        temperature of its own ``bc_T_inf``.

        The environment coefficient is ``h_out`` - the film the last assembly applied,
        radiative share included when the solve had radiation on (:meth:`outer_film`
        re-evaluates it there) - so the report and the operator it describes carry the
        same number and the per-leaf closure stays at round-off.
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
            positions, areas = self._environment_surface(table)
            if positions.size:
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
        mean, the same products, the same order); ``g`` is what the physics asks for, and
        both come from :func:`src.solver.matrix.face_conductance` - the one rule the
        structured assembly applies to its own neighbour tables: a face touching an
        excluded leaf carries no conduction at all, and an interface between two materials
        carries the contact resistance in series with the two half leaves.  Everywhere
        else ``g == g_base``, which is what keeps this mesh's flux report equal to the
        octree solver's.
        """
        faces = np.asarray(self.tree.faces(), dtype=float).reshape(-1, 5)
        i = faces[:, 0].astype(int)
        j = faces[:, 1].astype(int)
        area = faces[:, 3] * self.physical_size ** 2
        distance = faces[:, 4] * self.physical_size
        g_base = face_conductance(self.k[i], self.k[j], area, distance,
                                  size_a=self.sizes[i], size_b=self.sizes[j],
                                  material_a=self.material_id[i],
                                  material_b=self.material_id[j])
        excluded = self.excluded[i] | self.excluded[j] if self.excluded.any() else None
        g = face_conductance(self.k[i], self.k[j], area, distance,
                             size_a=self.sizes[i], size_b=self.sizes[j],
                             material_a=self.material_id[i], material_b=self.material_id[j],
                             h_contact=self.h_contact, excluded_b=excluded)
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

    def assemble(self, enforce_dirichlet: bool = True, radiation: bool = False
                 ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """``(A, b)`` of ``div(k grad T) + Q = 0`` per unit volume, Dirichlet applied.

        The films go on the diagonal with their reference temperatures on the right-hand
        side, exactly as :mod:`src.solver.matrix` does for a structured mesh; a Neumann
        face adds ``q/d`` with the local leaf edge ``d``.  With ``radiation`` on, the
        environment film carries the linearised radiative share of the current field (see
        :func:`src.solver.matrix.environment_film`).  The Dirichlet rows are the leaves
        :meth:`fixed_leaves` names, eliminated symmetrically by
        :func:`src.solver.matrix.apply_dirichlet` - the same elimination, so a fixed leaf
        is exact instead of relaxed.
        """
        faces = self._face_table()
        a = self.matrix(faces).tocsr()
        b = (self.Q_source + self.Q_sink).astype(float).copy()
        # the film the operator carries, written back to the mesh exactly as the
        # structured assembly does, so ``h_out`` always says what the last solve applied
        film = self.outer_film(faces, radiation)
        if film > 0.0:
            self.h_out = film
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
        mask, values = self._fixed_arrays()
        return apply_dirichlet(a, b, mask, values), b

    # ---------------------------------------------------------------- transient
    def volume_scale(self) -> np.ndarray | None:
        """The per-leaf volumes ``solve_linear`` wants, or ``None`` on a uniform tree.

        The per-volume coefficients are ``diag(V)^-1 K`` with ``K`` symmetric, so on a
        tree whose leaves are all the same size the operator is symmetric already (the
        transformation is a constant) and the linear layer is asked for no
        symmetrisation; on a graded one the leaf volumes are handed over, as the
        structured solver does with ``mesh.V`` on a graded ``Mesh3D``.
        """
        sizes = self.tree.cell_sizes()
        if float(sizes.min()) == float(sizes.max()):
            return None
        return np.asarray(self.V, dtype=float)

    def transient_operators(self, dt: float, radiation: bool = False
                            ) -> tuple[sparse.csr_matrix, np.ndarray]:
        """``(A, m_diag)`` for ``(M/dt + L) T = (M/dt) T^n + b``, as the structured builder.

        The operator :meth:`assemble` builds (conduction, film diagonal, impedance of the
        imposed fluxes) without the elimination, the mass matrix on the diagonal, and then
        the elimination of the pinned leaves - the tree's counterpart of
        :func:`src.solver.matrix.build_transient_operators`, built in the same order and
        for the same reasons:

        * ``M = diag(rho*cp)`` [J/(m^3 K)] and **not** ``rho*cp*V``: ``L`` is already per
          unit volume, and a volume factor scales every time constant by ``1/d^3``, which
          makes the answer depend on the mesh resolution;
        * the mass diagonal is added *before* the elimination, because an identity row
          scaled by ``M/dt`` would let a fixed leaf decay towards zero instead of holding
          its temperature.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
        a, _b = self.assemble(enforce_dirichlet=False, radiation=radiation)
        m_diag = (self.rho * self.cp).astype(float)
        mask, values = self._fixed_arrays()
        a = (a + sparse.diags(m_diag / dt)).tocsr()
        return apply_dirichlet(a, np.zeros(self.tree.n_cells), mask, values), m_diag

    def transient_rhs(self, m_diag: np.ndarray, dt: float, T_prev: np.ndarray,
                      radiation: bool = False) -> np.ndarray:
        """``(M/dt) T^n + b`` of one backward-Euler step, the pinned rows at their value.

        The tree's counterpart of :func:`src.solver.matrix.transient_rhs`, term by term:
        the right-hand side is rebuilt from a full :meth:`assemble` - the sources and the
        films of the current state are what a step changes, and the known columns the
        elimination moves to the right-hand side ride with them (the structured builder
        does exactly the same, through ``build_steady_matrix`` inside its own
        ``transient_rhs``) - and the rows :meth:`fixed_leaves` pins carry their value
        alone, written *after* the mass term, because an identity row must see the fixed
        temperature and not ``M/dt`` riding on it.
        """
        rhs = (m_diag / dt) * np.asarray(T_prev, dtype=float)
        _a, b = self.assemble(enforce_dirichlet=True, radiation=radiation)
        rhs += b
        mask, values = self._fixed_arrays()
        rhs[mask] = values[mask]
        return rhs

    # ------------------------------------------------------------------- solve
    def solve_steady(self, config: LinearConfig | None = None,
                     radiation: bool | None = None,
                     x0: np.ndarray | None = None) -> AdaptiveSteadyResult:
        """Solve the steady problem and report the field, the fluxes and the balance.

        The linear layer is the structured one (:func:`src.solver.linear.solve_linear`), so
        a graded tree is solved with the volume symmetrisation the structured solver uses
        and the method and tolerance are the caller's.  The default is a direct
        factorisation, which is exact enough to compare two meshes.

        ``radiation`` defaults to the flag of ``config`` when it carries one - the
        structured :class:`~src.solver.steady.SolverConfig` does, the linear layer does not
        - so a caller that hands the solver's own config over gets the physics it asked
        for.  ``x0`` is the warm start of an iterative method, in leaf order.
        """
        if radiation is None:
            radiation = bool(getattr(config, "radiation", False))
        start = time.perf_counter()
        a, b = self.assemble(radiation=radiation)
        linear: LinearResult = solve_linear(
            a, b, config or LinearConfig(method="direct"), x0=x0,
            scale=self.volume_scale())
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
        leaves, the conduction leaving them into the fixed ones, and the films - the
        environment one with the coefficient the last :meth:`assemble` stored, radiative
        share included, so the identity ``p_source + p_sink == q_fixed + p_film`` holds at
        the round-off of the per-leaf closure, which :attr:`AdaptiveBalance.residual`
        reports.
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
            # a leaf the elimination pinned (an excluded one, or one on a Dirichlet face)
            # carries no film: its row is the identity, so the exchange never enters the
            # solution and must not be reported either - the rule
            # ``fluxes.environment_flux`` already follows for a structured mesh
            acting = free[positions]
            rate = (coefficient[acting] * self.V[positions[acting]]
                    * (self.T[positions[acting]] - reference[acting]))
            np.add.at(per_leaf, positions[acting], rate)
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

    # ---------------------------------------------------------- validation
    def validate(self, check_temperature: bool = True) -> None:
        """Reject non-physical fields before an expensive solve.

        The same contract :meth:`src.core.mesh.Mesh3D.validate` gives a structured mesh,
        so a painter that ends its work with ``mesh.validate()`` keeps the guarantee while
        the two meshes coexist.
        """
        for name, values in (("k", self.k), ("rho", self.rho), ("cp", self.cp)):
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name}: non-finite values")
            if values.min() <= 0:
                raise ValueError(f"{name}: values must be > 0, min = {values.min():.3g}")
        if not np.all(np.isfinite(self.Q_source)) or not np.all(np.isfinite(self.Q_sink)):
            raise ValueError("Q fields: non-finite values")
        if np.any(self.Q_source < 0):
            raise ValueError("Q_source must be >= 0 (use Q_sink for extractions)")
        if np.any(self.Q_sink > 0):
            raise ValueError("Q_sink must be <= 0")
        if check_temperature:
            check_kelvin(self.T, "mesh.T")
            for face, bc in self.face_bc.items():
                if bc.kind in (BoundaryType.DIRICHLET, BoundaryType.CONVECTION):
                    check_kelvin(bc.value, f"face_bc[{face}].value")
        for face, bc in self.face_bc.items():
            if bc.kind == BoundaryType.CONVECTION and bc.h < 0:
                raise ValueError(f"face_bc[{face}].h must be >= 0")

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
