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
from ..solver.octree_solver import _NOISE_FLOOR, OctreeSteadySolver, _mark
from ..units import check_kelvin
from .mesh import FACES, BoundaryType, FaceBC
from .mesh_api import FaceRow, is_interior_tube
from .box_tree import BoxTree
from .octree import FACE_AXIS, FACE_SIGN, Leaf, Octree, locate_in_tables, uniform_tree
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
    #: the height the band asks for [m] on an anisotropic tree (:class:`BoxTree`);
    #: ``None`` = the same as ``size``.  A cubic octree reads ``min(size, size_z)``
    size_z: float | None = None
    #: a *radial* band: the annulus ``r_inner <= r <= r_outer`` about the vertical axis
    #: through ``axis`` (x, y), between ``low[2]`` and ``high[2]``; ``low``/``high`` are
    #: then its bounding box.  ``None`` = the band is the box itself.  A cylindrical
    #: vessel is annuli and discs: covering them with axis-aligned boxes refined the air
    #: at the corners and missed the ring at 45 degrees
    axis: tuple[float, float] | None = None
    r_inner: float = 0.0
    r_outer: float = 0.0

    @property
    def height(self) -> float:
        """The height the band asks for [m]."""
        return float(self.size if self.size_z is None else self.size_z)

    def intersects(self, leaf: Leaf, physical_size: float) -> bool:
        """True when the leaf's box overlaps this band (a touch on a plane does not)."""
        corner = np.array([[leaf.x, leaf.y, leaf.z]], dtype=float) * physical_size
        edge = np.array([leaf.size * physical_size])
        return bool(self.touching(corner, corner + edge[:, None])[0])

    def touching(self, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        """Which of the boxes ``low[i]..high[i]`` (``(n, 3)`` [m]) overlap the band."""
        hit = np.all((low < np.asarray(self.high)) & (high > np.asarray(self.low)), axis=1)
        if self.axis is None:
            return hit
        cx, cy = self.axis
        # nearest and farthest distance of each box's plan rectangle from the axis
        dx = np.maximum(np.maximum(low[:, 0] - cx, cx - high[:, 0]), 0.0)
        dy = np.maximum(np.maximum(low[:, 1] - cy, cy - high[:, 1]), 0.0)
        near = np.hypot(dx, dy)
        far = np.hypot(np.maximum(np.abs(low[:, 0] - cx), np.abs(high[:, 0] - cx)),
                       np.maximum(np.abs(low[:, 1] - cy), np.abs(high[:, 1] - cy)))
        return hit & (near < self.r_outer) & (far > self.r_inner)


def _check_bands(bands: Sequence[RefinementBand]) -> None:
    """Reject a band list no refinement can satisfy, before a tree is built."""
    if not bands:
        raise ValueError("at least one refinement band is required")
    for band in bands:
        if band.size <= 0.0 or band.height <= 0.0:
            raise ValueError(f"band size must be > 0, got {band.size} x {band.height}")
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

    #: the cubic :class:`Octree` (``physical_size`` is its finest edge) or the
    #: anisotropic :class:`BoxTree` (which carries its own finest cell ``dx`` x ``dz``)
    tree: Octree | BoxTree
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
        if isinstance(self.tree, BoxTree):
            # the finest plan cell: what a consumer that reads one finest size gets
            self.physical_size = float(self.tree.dx)
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

    @property
    def anisotropic(self) -> bool:
        """True on a :class:`BoxTree` (leaves with their own plan edge and height)."""
        return isinstance(self.tree, BoxTree)

    def _rebuild_caches(self) -> None:
        """Leaf edges, volumes and the "touches a box face" mask of the current tree."""
        if self.anisotropic:
            #: ``(n, 3)`` leaf edges [m]: x and y the plan edge, z the height
            self.extent = self.tree.extents()
            self.V = np.prod(self.extent, axis=1)
            self._h_char = np.cbrt(self.V)
            #: the characteristic edge ``V^(1/3)``: a cube's edge on a cubic tree
            self.sizes = self._h_char
            self.on_box_face = self.tree.on_box_face()
            return
        self.sizes = self.tree.cell_sizes() * self.physical_size
        self.V = self.sizes ** 3
        self._h_char = np.cbrt(self.V)
        n = self.tree.n
        corners, levels = self.tree._leaf_arrays()
        upper = corners + np.left_shift(np.int64(1), levels)[:, None]
        self.on_box_face = np.any((corners == 0) | (upper == n), axis=1)
        self.extent = np.repeat(self.sizes[:, None], 3, axis=1)

    @classmethod
    def uniform(cls, n_finest: int, physical_size: float, level: int) -> AdaptiveMesh:
        """Every leaf of the same level, ``2 ** level`` finest cells on a side."""
        return cls(uniform_tree(n_finest, level), physical_size)

    @classmethod
    def from_box(cls, n_xy: int, n_z: int, dx: float, dz: float, lxy: int | None = None,
            lz: int | None = None) -> AdaptiveMesh:
        """A mesh on an anisotropic :class:`BoxTree`, uniform at ``(lxy, lz)``.

        The box is ``n_xy dx`` square in plan and ``n_z dz`` tall; the default levels are
        the single leaf of the whole box.
        """
        tree = BoxTree.uniform(n_xy, n_z, dx, dz,
                               int(np.log2(n_xy)) if lxy is None else int(lxy),
                               int(np.log2(n_z)) if lz is None else int(lz))
        return cls(tree)

    @classmethod
    def from_plan(cls, plan, max_cells: int | None = None) -> AdaptiveMesh:
        """The mesh of a :class:`~src.analysis.convergence.AdaptivePlan`, refined to it.

        An anisotropic plan builds a :class:`BoxTree` from its single root leaf; a cubic
        one the octree of :meth:`from_bands`.  ``max_cells`` caps the leaves.
        """
        if getattr(plan, "anisotropic", False):
            mesh = cls.from_box(plan.n_finest, plan.n_z, plan.physical_size, plan.dz)
            mesh.refine_bands(plan.bands, max_cells=max_cells)
            return mesh
        level = plan.base_level
        if level is None:
            level = int(np.log2(plan.n_finest))
        mesh = cls(uniform_tree(plan.n_finest, level), plan.physical_size)
        mesh.refine_bands(plan.bands, max_cells=max_cells)
        return mesh

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
        if self.anisotropic:
            self._refine_bands_box(bands, max_cells)
            return
        sizes = np.array([min(band.size, band.height) for band in bands], dtype=float)

        def marks() -> np.ndarray:
            """1 for every leaf still larger than the finest band it touches, else 0.

            The same rule as :meth:`RefinementBand.intersects` (a touch on a plane does not
            count), evaluated for all the leaves at once: with one box per riser the band
            list is long, and a per-leaf loop over it was a cost of its own.
            """
            corner = np.array([(leaf.x, leaf.y, leaf.z) for leaf in self.tree.leaves],
                              dtype=float).reshape(-1, 3) * self.physical_size
            edge = self.tree.cell_sizes() * self.physical_size
            upper = corner + edge[:, None]
            target = np.full(edge.size, np.inf)
            for index, band in enumerate(bands):
                touching = band.touching(corner, upper)
                np.minimum(target, np.where(touching, sizes[index], np.inf), out=target)
            levels = self.tree.cell_sizes() > 1.0      # a finest leaf cannot split
            return ((edge > target) & levels).astype(float)

        for _ in range(self.tree.max_level):
            marked = marks()
            count = int(np.count_nonzero(marked))
            if count == 0:
                return                        # every leaf already meets its band
            if max_cells is not None and self.n_cells + 7 * count > max_cells:
                return                        # the round would not fit: stop here
            self.refine(marked, 0.5)

    def _refine_bands_box(self, bands: Sequence[RefinementBand],
                          max_cells: int | None) -> None:
        """:meth:`refine_bands` on a :class:`BoxTree`: plan and height separately.

        A leaf is split in plan while its plan edge is larger than the finest plan
        target of the bands it touches, and in height while its height is larger than
        their finest height target; the rounds stop when every leaf meets its bands or
        before a round that would exceed ``max_cells``.
        """
        plan = np.array([band.size for band in bands], dtype=float)
        tall = np.array([band.height for band in bands], dtype=float)
        tree = self.tree
        for _ in range(tree.max_lxy + tree.max_lz + 1):
            low = tree.lower()
            edge = tree.extents()
            high = low + edge
            want_xy = np.full(tree.n_cells, np.inf)
            want_z = np.full(tree.n_cells, np.inf)
            for index, band in enumerate(bands):
                touching = band.touching(low, high)
                np.minimum(want_xy, np.where(touching, plan[index], np.inf), out=want_xy)
                np.minimum(want_z, np.where(touching, tall[index], np.inf), out=want_z)
            # a leaf exactly at its target is done
            split_xy = (edge[:, 0] > want_xy * (1.0 + 1e-9)) & (tree.lxy > 0)
            split_z = (edge[:, 2] > want_z * (1.0 + 1e-9)) & (tree.lz > 0)
            added = (3 * int(np.count_nonzero(split_xy & ~split_z))
                     + int(np.count_nonzero(split_z & ~split_xy))
                     + 7 * int(np.count_nonzero(split_xy & split_z)))
            if added == 0:
                return
            if max_cells is not None and self.n_cells + added > max_cells:
                return
            self._split_box(split_xy, split_z)

    def _split_box(self, split_xy: np.ndarray, split_z: np.ndarray) -> None:
        """Split a :class:`BoxTree` and carry every per-leaf field to the children."""
        previous = {name: getattr(self, name).copy() for name in FIELDS}
        mapping = self.tree.refine(split_xy, split_z)
        for name in FIELDS:
            setattr(self, name, previous[name][mapping])
        self._rebuild_caches()

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
        if self.anisotropic:
            if callable(indicator):
                raise TypeError("a box tree refines on a per-leaf array, not a rule")
            marked = np.asarray(indicator, dtype=float) > threshold
            self._split_box(marked, marked)
            return
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
        # the old tree's lookup tables: a new leaf lies inside exactly one old leaf (a
        # refinement only splits), found by locating the new leaf's corner in them
        self.tree._leaf_arrays()
        old_tables = self.tree._level_tables
        self.tree.refine(rule, threshold, levels=levels)
        corners, _levels = self.tree._leaf_arrays()
        mapping = locate_in_tables(old_tables, self.tree.n, corners)
        if np.any(mapping < 0):
            raise ValueError("a leaf of the refined tree lies in no leaf of the previous one")
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
        flux = np.abs(self.face_fluxes(self.T))
        noise = _NOISE_FLOOR * (float(flux.max()) if flux.size else 0.0)
        if self.anisotropic:
            return self._refine_round_box(refine_fraction, floor_ratio, noise, max_cells)
        values = np.abs(np.asarray(self.indicator(), dtype=float))
        threshold, marked = _mark(
            self.tree, values, refine_fraction, floor_ratio, noise=noise)
        if marked == 0:
            return 0
        # the budget is checked before the round: a marked leaf becomes eight leaves, so
        # the round adds ``7 * marked`` of them (the balance adds a few more along the
        # boundary of the flagged region, which the budget the caller set still bounds)
        if max_cells is not None and self.n_cells + 7 * marked > max_cells:
            return 0
        self.refine(values, threshold)
        return marked

    def _refine_round_box(self, refine_fraction: float, floor_ratio: float, noise: float,
                          max_cells: int | None) -> int:
        """:meth:`refine_round` on a :class:`BoxTree`: split where the jump is, that way.

        The flux jump is kept per axis: a leaf whose plan share of the jump is at least a
        quarter of it is split in plan, and one whose vertical share is at least a quarter
        in height - so a leaf of the bed whose error is radial stays tall.
        """
        net = self._axis_jumps()
        values = net.sum(axis=1)
        splittable = (self.tree.lxy > 0) | (self.tree.lz > 0)
        live = values[splittable & (values > 0.0)]
        if live.size == 0:
            return 0
        threshold = max(floor_ratio * float(live.max()), noise)
        cap = max(int(np.ceil(refine_fraction * self.n_cells)), 1)
        if int(np.count_nonzero(splittable & (values > threshold))) > cap:
            threshold = max(threshold, float(np.quantile(live, 1.0 - refine_fraction)))
        marked = splittable & (values > threshold)
        count = int(np.count_nonzero(marked))
        if count == 0:
            return 0
        share = np.divide(net, values[:, None], out=np.zeros_like(net),
                          where=values[:, None] > 0.0)
        split_xy = marked & (share[:, 0] + share[:, 1] >= 0.25)
        split_z = marked & (share[:, 2] >= 0.25)
        added = (3 * int(np.count_nonzero(split_xy & ~split_z))
                 + int(np.count_nonzero(split_z & ~split_xy))
                 + 7 * int(np.count_nonzero(split_xy & split_z)))
        if max_cells is not None and self.n_cells + added > max_cells:
            return 0
        self._split_box(split_xy, split_z)
        return count

    def _axis_jumps(self, temperature: np.ndarray | None = None) -> np.ndarray:
        """``(n, 3)`` flux imbalance per leaf and axis [W] (see :meth:`indicator`)."""
        table = self._face_table()
        values = self.T if temperature is None else np.asarray(temperature, dtype=float)
        flux = table.g * (values[table.i] - values[table.j])
        # every entry of a box tree's table has j on the high side of i
        net = np.zeros((self.n_cells, 3))
        np.add.at(net, (table.i, table.axis), flux)
        np.add.at(net, (table.j, table.axis), -flux)
        lower = self.lower_corners()
        upper = lower + self.extent
        box = np.asarray(self.box)
        two_sided = (lower > 1e-12 * box) & (upper < box * (1.0 - 1e-12))
        return np.abs(np.where(two_sided, net, 0.0))

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
    def box(self) -> tuple[float, float, float]:
        """``(Lx, Ly, Lz)`` of the domain [m]."""
        if self.anisotropic:
            return self.tree.box
        size = self.box_size
        return (size, size, size)

    def lower_corners(self) -> np.ndarray:
        """``(n, 3)`` lower corners of the leaves [m]."""
        if self.anisotropic:
            return self.tree.lower()
        corners, _levels = self.tree._leaf_arrays()
        return corners * self.physical_size

    @property
    def box_size(self) -> float:
        """Edge of the (cubic) domain [m]; the largest edge on an anisotropic tree.

        ``physical_size`` is the edge of one *finest* cell, so the box is ``n_finest``
        of them: this is the ``Lx = Ly = Lz`` a structured consumer reads.
        """
        if self.anisotropic:
            return float(max(self.tree.box))
        return float(self.physical_size) * self.tree.n

    @property
    def h_char(self) -> np.ndarray:
        """Characteristic leaf size ``V^(1/3)`` [m], as on :class:`Mesh3D`."""
        return self._h_char

    def fixed_leaves(self) -> dict[int, float]:
        """The Dirichlet set as ``{leaf position: temperature}``.

        The leaves a ``face_bc`` of kind DIRICHLET drives (``wall_cells`` of
        :mod:`src.solver.octree_solver`) plus the excluded leaves held at
        :attr:`t_ambient` - the same set and the same values ``src/solver/matrix.py``
        builds for a structured mesh, which applies the ambient *after* the faces: an
        excluded leaf carries no physics, so where the two rules meet the placeholder
        wins and both meshes hold the same field.
        """
        mask, values = self._fixed_arrays()
        return {int(position): float(values[position])
                for position in np.flatnonzero(mask)}

    def fixed_mask(self) -> np.ndarray:
        """``(n_cells,)`` mask of the leaves the Dirichlet elimination pins.

        The boolean view of :meth:`fixed_leaves`, for a consumer that filters a per-leaf
        array rather than iterating the set: a pinned leaf has an identity row, so an
        exchange with it - a film on it, a source deposited in it - never reaches the
        solution and the reports must not count it.
        """
        return self._fixed_arrays()[0]

    def _fixed_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(mask, values)`` of the pinned leaves, in the form the elimination takes.

        :meth:`fixed_leaves` as the two per-leaf arrays
        :func:`src.solver.matrix.apply_dirichlet` wants, which is how both assemblies -
        the steady one and the transient one - pin the same set.
        """
        mask = np.zeros(self.tree.n_cells, dtype=bool)
        values = np.zeros(self.tree.n_cells)
        for face, bc in self.face_bc.items():
            if bc.kind == BoundaryType.DIRICHLET:
                positions = self.wall_indices(face)
                mask[positions] = True
                values[positions] = float(bc.value)
        # the excluded leaves are applied last: where the two rules meet the ambient wins
        mask |= self.excluded
        values[self.excluded] = float(self.t_ambient)
        return mask, values

    def wall_indices(self, face: str) -> np.ndarray:
        """Positions of the leaves touching a box face, in ``tree.leaves`` order.

        The leaves are the ones :func:`src.solver.octree_solver.wall_cells` names - the set
        a face condition drives - and the positions are what the flat per-leaf arrays use.
        """
        self._check_face(face)
        return self._tree_tables()["walls"][face]

    def _tree_tables(self) -> dict:
        """Per-tree tables every assembly reads: the wall leaves and the face geometry.

        They depend on the leaves alone, so they are built once per tree state (keyed on
        :attr:`Octree.version`, which every mutation bumps) instead of once per call: a
        transient step asks for them several times, and rebuilding the face list each
        time was what made a step cost seconds on a tree of a few ten thousand leaves.
        """
        cache = getattr(self, "_tables", None)
        if cache is not None and cache["version"] == self.tree.version:
            return cache
        if self.anisotropic:
            lower = self.tree.corners
            upper = lower + self.tree.unit_extents()
            limit = (self.tree.n_xy, self.tree.n_xy, self.tree.n_z)
            walls = {}
            for face in FACES:
                axis = FACE_AXIS[face]
                touching = (lower[:, axis] == 0 if FACE_SIGN[face] < 0
                            else upper[:, axis] == limit[axis])
                walls[face] = np.flatnonzero(touching).astype(int)
            i, j, axis, area, distance = self.tree.face_arrays()
            cache = {"version": self.tree.version, "walls": walls, "i": i.astype(int),
                     "j": j.astype(int), "axis": axis.astype(int), "area": area,
                     "distance": distance}
            self._tables = cache
            return cache
        n = self.tree.n
        corners = np.array([(leaf.x, leaf.y, leaf.z) for leaf in self.tree.leaves],
                           dtype=np.int64).reshape(-1, 3)
        edge = self.tree.cell_sizes().astype(np.int64)
        walls = {}
        for face in FACES:
            axis = FACE_AXIS[face]
            touching = (corners[:, axis] == 0 if FACE_SIGN[face] < 0
                        else corners[:, axis] + edge == n)
            walls[face] = np.flatnonzero(touching).astype(int)
        faces = self.tree.face_array()
        i = faces[:, 0].astype(int)
        j = faces[:, 1].astype(int)
        cache = {"version": self.tree.version, "walls": walls, "i": i, "j": j,
                 "axis": faces[:, 2].astype(int),
                 "area": faces[:, 3] * self.physical_size ** 2,
                 "distance": faces[:, 4] * self.physical_size}
        self._tables = cache
        return cache

    def centre(self, position: int) -> tuple[float, float, float]:
        """Centre of one leaf in physical coordinates [m]."""
        if self.anisotropic:
            return tuple(float(c) for c in self.centres()[position])
        return tuple(float(c * self.physical_size)
                     for c in self.tree.leaves[position].centre)

    def centres(self) -> np.ndarray:
        """``(n_cells, 3)`` leaf centres in physical coordinates [m]."""
        if self.anisotropic:
            cache = getattr(self, "_centre_cache", None)
            if cache is None or cache[0] != self.tree.version:
                cache = (self.tree.version, self.tree.centres())
                self._centre_cache = cache
            return cache[1].copy()
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
        if self.anisotropic:
            position = int(self.tree.locate_points(np.array([[x, y, z]]), clamp=True)[0])
            if position < 0:
                raise ValueError(f"no leaf contains ({x}, {y}, {z}) m")
            return position
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

    def axis_size_at(self, axis: int, x: float, y: float, z: float) -> float:
        """Edge along ``axis`` (0 = x, 1 = y, 2 = z) of the leaf containing a point [m]."""
        return float(self.extent[self.locate(x, y, z), axis])

    def locate_points(self, points: np.ndarray, below: bool = False) -> np.ndarray:
        """Leaf of each physical point (``(m, 3)`` [m]), -1 outside the box.

        ``below``: a point on a face belongs to the leaf below it (``ceil(v/d) - 1``, the
        rasteriser's rule); otherwise to the leaf above it (``floor(v/d)``).
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        if self.anisotropic:
            return self.tree.locate_points(points, below=below)
        scaled = points / self.physical_size
        cells = (np.ceil(scaled) - 1 if below else np.floor(scaled)).astype(np.int64)
        return self.tree.locate_many(cells)

    def faces(self) -> Sequence[FaceRow]:
        """The conservative face list of :class:`MeshAPI`, straight from the tree.

        Finest-cell units on a cubic octree (its own convention); metres on a box tree,
        whose axes have different finest cells.
        """
        if self.anisotropic:
            i, j, axis, area, distance = self.tree.face_arrays()
            return list(zip(i.tolist(), j.tolist(), axis.tolist(), area.tolist(),
                            distance.tolist(), strict=True))
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
                size = self.extent[positions, FACE_AXIS[face]]
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
        import hashlib

        digest = hashlib.blake2b(digest_size=16)
        for array in (self.k, self.material_id, self.excluded):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(repr((self.tree.version, self.h_contact)).encode())
        key = digest.hexdigest()
        memo = getattr(self, "_face_memo", None)
        if memo is not None and memo[0] == key:
            # the conductances depend on the properties, not on the field: a balance per
            # step asked for the same table several times
            return memo[1]
        tables = self._tree_tables()
        i, j = tables["i"], tables["j"]
        area, distance = tables["area"], tables["distance"]
        normal_i = self.extent[i, tables["axis"]]
        normal_j = self.extent[j, tables["axis"]]
        g_base = face_conductance(self.k[i], self.k[j], area, distance,
                                  size_a=normal_i, size_b=normal_j,
                                  material_a=self.material_id[i],
                                  material_b=self.material_id[j])
        excluded = self.excluded[i] | self.excluded[j] if self.excluded.any() else None
        g = face_conductance(self.k[i], self.k[j], area, distance,
                             size_a=normal_i, size_b=normal_j,
                             material_a=self.material_id[i], material_b=self.material_id[j],
                             h_contact=self.h_contact, excluded_b=excluded)
        table = _FaceTable(i=i, j=j, axis=tables["axis"], area=area,
                           distance=distance, g_base=g_base, g=g)
        self._face_memo = (key, table)
        return table

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
        if self.anisotropic:
            return self._axis_jumps(temperature).sum(axis=1)
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
        if self.anisotropic:
            # the tree's operator is its face list itself: ``g / V`` on both rows
            n = self.n_cells
            ci, cj = table.g / self.V[table.i], table.g / self.V[table.j]
            rows = np.concatenate((table.i, table.j, np.arange(n)))
            cols = np.concatenate((table.j, table.i, np.arange(n)))
            diagonal = np.zeros(n)
            np.add.at(diagonal, table.i, ci)
            np.add.at(diagonal, table.j, cj)
            vals = np.concatenate((-ci, -cj, diagonal))
            return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
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
        key = None if radiation else self._operator_key()
        entry = getattr(self, "_assembly", None)
        if entry is None or key is None or entry["key"] != key:
            entry = self._build_operator(radiation)
            entry["key"] = key
            self._assembly = entry if key is not None else None
        else:
            # the same operator: the film written back is the one it was built with
            if entry["film"] > 0.0:
                self.h_out = entry["film"]
        b = (self.Q_source + self.Q_sink).astype(float).copy()
        for name, positions, coefficient in entry["films"]:
            np.add.at(b, positions, coefficient * self._film_reference(name, positions))
        for face, bc in self.face_bc.items():
            if bc.kind == BoundaryType.NEUMANN and bc.value != 0.0:
                positions = self.wall_indices(face)
                if positions.size:
                    b[positions] += bc.value / self.extent[positions, FACE_AXIS[face]]
        if not enforce_dirichlet:
            return entry["a"], b
        # the elimination's share of the right-hand side, exactly as apply_dirichlet
        # computes it: the known columns move to the free rows, the pinned rows hold
        rows, contributions, mask, values = entry["moved"]
        if rows.size:
            np.subtract.at(b, rows, contributions)
        b[mask] = values[mask]
        return entry["a_eliminated"], b

    def _operator_key(self) -> str:
        """A digest of everything the operator depends on (not the right-hand side).

        The conduction, the film coefficients and the pinned set change only when a field
        they are made of changes - the conductivities, the materials, the excluded leaves,
        the film coefficients, the box conditions, the outer film - while a transient step
        or a Picard sweep only moves the film *temperatures*.  Keyed on this, the operator
        is assembled once and reused.
        """
        import hashlib

        digest = hashlib.blake2b(digest_size=16)
        for array in (self.k, self.material_id, self.excluded, self.boundary_type,
                      self.bc_h):
            digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(repr((self.tree.version, self.h_out, self.h_out_conv,
                            self.h_contact, self.t_ambient, self.environment_emissivity,
                            sorted((face, bc.kind, bc.h, bc.value, bc.emissivity)
                                   for face, bc in self.face_bc.items()))).encode())
        return digest.hexdigest()

    def _film_reference(self, name: str, positions: np.ndarray) -> np.ndarray:
        """The reference temperature of one film at its positions, from the current state."""
        if name == "environment":
            return np.full(positions.size, float(self.t_ambient))
        if name == "tube":
            return self.bc_T_inf[positions]
        return np.full(positions.size, float(self.face_bc[name].value))

    def _build_operator(self, radiation: bool) -> dict:
        """The operator of the current state, unpinned and pinned, and the films' parts."""
        faces = self._face_table()
        a = self.matrix(faces).tocsr()
        # the film the operator carries, written back to the mesh exactly as the
        # structured assembly does, so ``h_out`` always says what the last solve applied
        film = self.outer_film(faces, radiation)
        if film > 0.0:
            self.h_out = film
        diagonal = np.zeros(self.tree.n_cells)
        films = []
        for name, positions, coefficient, _reference in self._films(faces):
            np.add.at(diagonal, positions, coefficient)
            films.append((name, positions, coefficient))
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
        mask, values = self._fixed_arrays()
        ordered = a.tocsr(copy=True)
        ordered.sort_indices()
        row = np.repeat(np.arange(ordered.shape[0]), np.diff(ordered.indptr))
        col = ordered.indices
        moved = mask[col] & ~mask[row]
        contributions = ordered.data[moved] * values[col[moved]]
        eliminated = apply_dirichlet(a, np.zeros(self.tree.n_cells), mask, values)
        return {"a": a, "a_eliminated": eliminated, "films": films, "film": film,
                "moved": (row[moved], contributions, mask, values)}

    # ---------------------------------------------------------------- transient
    def volume_scale(self) -> np.ndarray | None:
        """The per-leaf volumes ``solve_linear`` wants, or ``None`` on a uniform tree.

        The per-volume coefficients are ``diag(V)^-1 K`` with ``K`` symmetric, so on a
        tree whose leaves are all the same size the operator is symmetric already (the
        transformation is a constant) and the linear layer is asked for no
        symmetrisation; on a graded one the leaf volumes are handed over, as the
        structured solver does with ``mesh.V`` on a graded ``Mesh3D``.
        """
        volume = np.asarray(self.V, dtype=float)
        if float(volume.min()) == float(volume.max()):
            return None
        return volume

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
    def level_histogram(self) -> dict:
        """Leaves per level (a cubic tree) or per ``(lxy, lz)`` pair (a box tree)."""
        if self.anisotropic:
            return self.tree.level_pairs()
        return self.tree.level_histogram()

    def size_histogram(self) -> dict[tuple[float, float], int]:
        """Leaves per ``(plan edge, height)`` [m], finest first."""
        pairs, counts = np.unique(np.round(self.extent[:, [0, 2]], 9), axis=0,
                                  return_counts=True)
        return {(float(a), float(b)): int(c)
                for (a, b), c in zip(pairs, counts, strict=True)}

    def summary(self) -> str:
        if self.anisotropic:
            sizes = ", ".join(f"{a * 1000:.0f}x{b * 1000:.0f} mm: {count}"
                              for (a, b), count in self.size_histogram().items())
            return f"adaptive mesh: {self.n_cells} leaves ({sizes}; plan x height)"
        levels = ", ".join(f"L{level}: {count}"
                           for level, count in self.level_histogram().items())
        # no face count: the face list costs seconds on a large tree and the summary is
        # printed after every build, before anything needs the faces
        return (f"adaptive mesh: {self.tree.n_cells} leaves ({levels}), "
                f"leaf edge {self.sizes.min():.4f}-{self.sizes.max():.4f} m")


