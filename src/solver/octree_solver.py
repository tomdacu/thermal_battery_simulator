"""Steady conduction on the adaptive octree, and the refinement that pays for itself.

:mod:`src.core.octree` builds the mesh and assembles the operator; this module gives it
the physics the structured solver has - a conductivity and a source per leaf, the leaves
held at a fixed temperature - and the diagnostics an energy balance needs (the heat rate
through every face, the total leaving the domain, the per-cell closure error).  It also
drives the a posteriori refinement: solve coarse, look at the flux jump between
neighbours, refine where it is large, solve again, and stop when the *objective* - the
loss flux, the storage mean - stops moving by more than the tolerance the caller asked
for, the same criterion :mod:`src.analysis.convergence` applies to a sequence of
structured grids (its :class:`ConvergenceTarget` and :class:`ConvergenceLevel` are the
vocabulary of the report this module hands back).

PHYSICS CONTRACT
----------------
The coefficients are exactly the ones of :mod:`src.solver.matrix`: a face is
``k_face A / (d_centers V)`` with ``k_face`` the harmonic mean of the two leaves, so a
tree of uniform leaves and a uniform :class:`~src.core.mesh.Mesh3D` of the same cell size
assemble **the same matrix** and the two solvers agree to the tolerance of the linear
solve: measured on a two-layer wall (1-D) and on a slab with a source in half the box
(2-D) the largest temperature difference is 2.5e-10 K and 2.0e-11 K, against a CG
tolerance of 1e-12 (``tests/test_octree_solver.py``).  The walls of an octree carry no
face at all, so they are adiabatic by construction: the only way heat enters or leaves is
through the leaves the caller holds at a fixed temperature, which is what
:func:`wall_cells` and :func:`dirichlet_walls` build.

WHAT THE FLUX REPORT IS
-----------------------
Every entry of ``Octree.faces()`` describes one interface with one conductance
``g`` [W/K], so the heat rate ``q = g (T_i - T_j)`` is a single number shared by both
cells: the flux field is conservative *by construction*, at machine precision, on any
combination of levels.  ``OctreeSteadyResult.divergence`` is the divergence of that
field, so summing it over the mesh cancels face by face - measured at 4.5e-13 W on a
graded test tree (288 leaves, 768 faces, up to 459 W per face), which is a part in 1e-15
of the largest rate and the round-off of the sum rather than a truncation error - while
``total_flux``, the rate leaving the free region through the fixed cells, equals the power
the sources deposit there to the residual of the linear solve (measured 1.1e-15 of the
largest face rate for a 10 kW model).  The heat rate is reported positive from the first cell
of the entry to the second; a caller wanting the flow through one wall wants
``wall_cells`` and the flux of the entries on it.

WHEN THE ADAPTIVE MESH PAYS
---------------------------
It pays when the error is *concentrated* and the objective is **local**: a hot spot in a
sand bed, a boundary layer, a pipe wall.  Cells are then spent on the feature and the far
field stays coarse.  Measured on the battery of ``tests/test_octree_solver.py`` - a
0.25 m cube of source of 100 W in a 1 m box, every wall at 300 K, the objective the
volume-weighted mean temperature of the source region - the cycle starting from 64 leaves
reaches 0.88 K of the 4096-leaf answer after two rounds and 176 leaves, against 6.83 K
for a uniform tree of 512 leaves: **7.8x closer with 2.9x fewer cells**.  It does *not*
pay, and this module says so rather than burning cells, when:

* the field is smooth everywhere (a uniform source in a box): the local error is spread
  evenly, so refining a region moves the error somewhere else, and the objective stalls
  on the first round;
* the objective is an integral over a smooth far field: on the battery above the same
  adaptive tree that answers the hot spot 7.8x better than the uniform one is *worse* on
  the whole field (3.60 K of root-mean-square error against 2.31 K) and on the mean of
  the whole box, because the far field, which adaptivity leaves coarse, is what those
  numbers weigh.  A local objective is what it buys, not a global one;
* the feature is invisible to the solution: a source or a material boundary falling
  *inside* a leaf changes the physics the estimator cannot see (leaves are homogeneous by
  definition).  No residual estimator finds it - the caller must start from a mesh whose
  leaf faces already sit on the interface;
* the answer is an integral that conservation fixes anyway: the total loss of a closed
  box is the source power on *any* mesh, so ``total_flux`` is not an objective
  refinement can improve.  Use the storage mean or the peak;
* the tree is too coarse to read: the estimator is a second difference, so a leaf must
  have a face on both sides of an axis to say anything.  A tree of two cells per side has
  one face per axis, and every leaf of it touches a wall - the honest answer there is
  "nothing to refine", which is what the report says;
* the linear algebra dominates: see the cost below.

MEASURED COST
-------------
Windows x64, Python 3.13, SciPy CG, i3-1125G4, with three other test processes running
at the same time (so the numbers are pessimistic by a factor of two): a uniform tree of
64 leaves (144 faces) solves in 4 ms, 512 leaves (1344 faces) in 29-45 ms, 4096 leaves
(11_520 faces) in 0.29-0.48 s and 32_768 leaves (95_232 faces) in about 1.5 s.  That is
**45 to 120 us per leaf, flat in the tree size**, and the split on the 4096-leaf tree is
68 ms of face list plus 109 ms of assembly against 64 ms of CG - three quarters the
geometry, one quarter the solve.  The refinement round on the battery above - solve,
estimate, refine, balance, solve again - took 66 ms for 64 -> 176 leaves.

The cost per leaf is flat because the Python loops over the face list and the linear
solve grow together: a tree twice as large costs twice as much and no level of refinement
is intrinsically more expensive than another.  The practical limit of this module is
therefore the Python face list (about 10^5 faces per second), not the solver; past that,
the block refinement of Afivo noted in :mod:`src.core.octree` is the next step, and it
changes no formula here.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ..analysis.convergence import ConvergenceLevel, ConvergenceTarget
from ..constants import EPS
from ..core.octree import FACE_AXIS, FACE_SIGN, FACES, Leaf, Octree

#: a field given per leaf: a mapping keyed by :class:`Leaf`, a callable, or an array
PerLeaf = Mapping[Leaf, float] | Callable[[Leaf], float] | np.ndarray | float
#: the Dirichlet set: a static mapping of leaves, or a rule re-evaluated on every tree
FixedSet = Mapping[Leaf, float] | Callable[[Octree], Mapping[Leaf, float]]


# --------------------------------------------------------------------- geometry
def wall_cells(tree: Octree, face: str) -> list[Leaf]:
    """The leaves touching a box face: the cells a face condition would drive.

    The face list of the octree stops at the wall, so a wall is adiabatic unless the
    leaves sitting on it are held at a temperature; this is the set to hold.
    """
    if face not in FACES:
        raise ValueError(f"invalid face {face!r}; expected one of {FACES}")
    axis = FACE_AXIS[face]
    low_side = FACE_SIGN[face] < 0
    out = []
    for leaf in tree.leaves:
        corner = (leaf.x, leaf.y, leaf.z)[axis]
        if corner == 0 if low_side else corner + leaf.size == tree.n:
            out.append(leaf)
    return out


def dirichlet_walls(tree: Octree, walls: Mapping[str, float]) -> dict[Leaf, float]:
    """Fixed-temperature leaves of the box walls, e.g. ``{"z_min": 300.0}``.

    A leaf on an edge or a corner belongs to two walls and keeps the value of the last
    wall in iteration order; give consistent values when that matters.
    """
    fixed: dict[Leaf, float] = {}
    for face, value in walls.items():
        for leaf in wall_cells(tree, face):
            fixed[leaf] = float(value)
    return fixed


def mean_temperature(tree: Octree, temperature: np.ndarray,
                     mask: Callable[[Leaf], bool] | None = None) -> float:
    """Volume-weighted mean temperature [K] of the leaves a mask keeps.

    Weighting by the volume is not cosmetic: the arithmetic mean of the temperatures
    changes when a region is refined, so a *cell count* would move the objective and the
    convergence test would chase the mesh instead of the physics.  This is the
    ``t_mean_storage`` of :mod:`src.analysis.convergence`, measured on a tree.
    """
    temperature = np.asarray(temperature, dtype=float)
    if temperature.shape != (tree.n_cells,):
        raise ValueError(f"temperature must have one value per leaf ({tree.n_cells})")
    volumes = tree.cell_sizes() ** 3
    if mask is not None:
        volumes = np.where([bool(mask(leaf)) for leaf in tree.leaves], volumes, 0.0)
        if volumes.sum() <= 0.0:
            raise ValueError("the mask selects no leaf")
    return float((volumes * temperature).sum() / volumes.sum())


def storage_mean_objective(mask: Callable[[Leaf], bool] | None = None
                           ) -> Callable[[Octree, OctreeSteadyResult], float]:
    """Objective for :func:`refine_on_objective`: the storage mean temperature [K].

    ``mask`` keeps the leaves of the storage (the sand bed, say); ``None`` averages the
    whole box.
    """
    def objective(tree: Octree, result: OctreeSteadyResult) -> float:
        return mean_temperature(tree, result.T, mask)

    return objective


def _per_leaf(tree: Octree, values: PerLeaf | None, default: float,
              name: str) -> np.ndarray:
    """Resolve a per-leaf field to an array aligned with ``tree.leaves``.

    The mapping form is checked against the tree, because a map left over from an
    earlier refinement is the one mistake this API makes easy: its keys are leaves the
    tree no longer has, and every one of them would be silently dropped.
    """
    if values is None:
        return np.full(tree.n_cells, float(default))
    if isinstance(values, Mapping):
        index = {leaf: i for i, leaf in enumerate(tree.leaves)}
        unknown = [leaf for leaf in values if leaf not in index]
        if unknown:
            first = unknown[0]
            raise ValueError(
                f"{name} names {len(unknown)} leaf/leaves that are not in the tree, "
                f"e.g. {first}; rebuild the map after refining")
        return np.array([float(values.get(leaf, default)) for leaf in tree.leaves])
    if callable(values) and not isinstance(values, np.ndarray):
        return np.array([float(values(leaf)) for leaf in tree.leaves])
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(tree.n_cells, float(array))
    if array.shape != (tree.n_cells,):
        raise ValueError(f"{name} must have one value per leaf ({tree.n_cells}), "
                         f"got shape {array.shape}")
    return array.astype(float, copy=True)


# --------------------------------------------------------------------- results
@dataclass
class OctreeSteadyResult:
    """Steady field on a tree and the fluxes it drives."""

    T: np.ndarray                 # per-leaf temperature [K], in ``tree.leaves`` order
    face_flux: np.ndarray         # heat rate [W] per face entry, from entry[0] to [1]
    divergence: np.ndarray        # net heat rate leaving every leaf [W]
    jump: np.ndarray              # per-leaf flux-jump estimate [W], for refinement
    total_flux: float             # rate leaving the free region through the fixed cells
    source_power: float           # power deposited by the sources in the free cells [W]
    residual: float               # largest per-cell closure error [W]
    balance_error: float          # |total_flux - source_power| / largest face rate
    cells: int
    solve_time: float             # [s]

    def summary(self) -> str:
        return (f"octree {self.cells} leaves: T {self.T.min():.2f}-{self.T.max():.2f} K, "
                f"source {self.source_power:.4g} W, boundary flux {self.total_flux:.4g} W "
                f"(balance {self.balance_error:.2e}), residual {self.residual:.2e} W, "
                f"{self.solve_time * 1000:.1f} ms")


class OctreeSteadySolver:
    """Solve ``div(k grad T) + Q = 0`` on an octree, holding a set of leaves fixed.

    ``conductivity`` and ``source`` are per leaf - a mapping, a callable, an array, or a
    scalar for the whole tree - and the fixed cells are a mapping of leaves to their
    temperature or a rule ``tree -> mapping`` re-evaluated on every tree.  The rule form
    is what a refinement round needs: a leaf that gets split is replaced by eight leaves
    that are not in a static map, and the wall would stop being a wall.
    """

    def __init__(self, tree: Octree, conductivity: PerLeaf | None = None,
                 source: PerLeaf | None = None, fixed: FixedSet | None = None,
                 physical_size: float = 1.0, tolerance: float = 1e-12,
                 conductivity_default: float = 1.0,
                 source_default: float = 0.0) -> None:
        if physical_size <= 0.0:
            raise ValueError(f"physical_size must be > 0, got {physical_size}")
        if tolerance <= 0.0:
            raise ValueError(f"tolerance must be > 0, got {tolerance}")
        self.tree = tree
        self.conductivity_arg = conductivity
        self.source_arg = source
        self.fixed_arg = fixed
        self.physical_size = float(physical_size)
        self.tolerance = float(tolerance)
        self.conductivity_default = float(conductivity_default)
        self.source_default = float(source_default)

    def __call__(self, tree: Octree) -> OctreeSteadyResult:
        """Solve on ``tree``: the solver is the ``solve`` argument of the refinement."""
        return self.solve(tree)

    # ---------------------------------------------------------------- the physics
    def conductivities(self, tree: Octree | None = None) -> np.ndarray:
        """Conductivity [W/(m K)] of every leaf, in ``tree.leaves`` order."""
        tree = self.tree if tree is None else tree
        return _per_leaf(tree, self.conductivity_arg, self.conductivity_default,
                         "conductivity")

    def sources(self, tree: Octree | None = None) -> np.ndarray:
        """Volumetric source [W/m^3] of every leaf, in ``tree.leaves`` order."""
        tree = self.tree if tree is None else tree
        return _per_leaf(tree, self.source_arg, self.source_default, "source")

    def fixed_cells(self, tree: Octree | None = None) -> dict[int, float]:
        """The Dirichlet set as ``{leaf index: temperature}`` for ``Octree.solve``."""
        tree = self.tree if tree is None else tree
        fixed = self.fixed_arg(tree) if callable(self.fixed_arg) else self.fixed_arg
        if not fixed:
            raise ValueError(
                "the octree walls are adiabatic: a pure Neumann problem has no unique "
                "solution, hold at least one leaf at a fixed temperature")
        index = {leaf: i for i, leaf in enumerate(tree.leaves)}
        out: dict[int, float] = {}
        for leaf, value in fixed.items():
            position = index.get(leaf)
            if position is None:
                raise ValueError(f"fixed leaf {leaf} is not in the tree; rebuild the "
                                 f"fixed set after refining")
            out[position] = float(value)
        return out

    # ------------------------------------------------------------------- assembly
    def _face_data(self, tree: Octree, conductivity: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(i, j, axis, conductance)`` of the whole face list [W/K].

        The conductance is ``k_face A / d_centers`` built with the same products as
        :meth:`Octree.diffusion_matrix`, so ``conductance * (T_i - T_j)`` is the heat
        rate the assembled operator implies and not a second, slightly different one.
        """
        faces = np.asarray(tree.faces(), dtype=float).reshape(-1, 5)
        i = faces[:, 0].astype(int)
        j = faces[:, 1].astype(int)
        axis = faces[:, 2].astype(int)
        k_face = 2.0 * conductivity[i] * conductivity[j] / (
            conductivity[i] + conductivity[j] + EPS)
        area = faces[:, 3] * self.physical_size ** 2
        distance = faces[:, 4] * self.physical_size
        return i, j, axis, k_face * area / distance

    def face_fluxes(self, temperature: np.ndarray, tree: Octree | None = None,
                    conductivity: np.ndarray | PerLeaf | None = None) -> np.ndarray:
        """Heat rate [W] through every face entry, positive from the first to the second."""
        tree = self.tree if tree is None else tree
        k = self._resolve_conductivity(tree, conductivity)
        i, j, _axis, g = self._face_data(tree, k)
        return g * (np.asarray(temperature, dtype=float)[i]
                    - np.asarray(temperature, dtype=float)[j])

    def divergence(self, temperature: np.ndarray, tree: Octree | None = None,
                   conductivity: np.ndarray | PerLeaf | None = None) -> np.ndarray:
        """Net heat rate [W] leaving every leaf: the discrete divergence of the flux."""
        tree = self.tree if tree is None else tree
        k = self._resolve_conductivity(tree, conductivity)
        i, j, _axis, g = self._face_data(tree, k)
        temperature = np.asarray(temperature, dtype=float)
        flux = g * (temperature[i] - temperature[j])
        return _divergence(tree.n_cells, i, j, flux)

    def indicator(self, temperature: np.ndarray, tree: Octree | None = None,
                  conductivity: np.ndarray | PerLeaf | None = None) -> np.ndarray:
        """Per-leaf flux-jump estimate [W]: the local error the refined mesh attacks.

        For every leaf and every axis the heat rate arriving through the low face is
        compared with the rate leaving through the high face; the indicator is the sum
        of the absolute differences over the axes the leaf has a face on both sides of.
        It is the discrete curvature of the flux field - the truncation error of the
        two-point flux the scheme is built on - so it vanishes on a linear field, it
        equals the source power on a leaf that generates heat (the two sides of its
        balance differ by exactly that), and it is nonzero in the shell around a hot
        spot, where the heat arriving from one face leaves through the others.  A leaf of
        a tree of two cells per side has one face per axis and no stencil, so it shows
        nothing: the estimate needs at least four cells per side to have anywhere to
        look.  One pass over the face list, no extra solve.
        """
        tree = self.tree if tree is None else tree
        k = self._resolve_conductivity(tree, conductivity)
        i, j, axis, g = self._face_data(tree, k)
        temperature = np.asarray(temperature, dtype=float)
        flux = g * (temperature[i] - temperature[j])
        return _flux_jump(tree, i, j, axis, flux)

    def _resolve_conductivity(self, tree: Octree,
                              conductivity: np.ndarray | PerLeaf | None) -> np.ndarray:
        if conductivity is None:
            return self.conductivities(tree)
        if isinstance(conductivity, np.ndarray):
            if conductivity.shape != (tree.n_cells,):
                raise ValueError(f"conductivity must have one value per leaf "
                                 f"({tree.n_cells}), got shape {conductivity.shape}")
            return conductivity.astype(float, copy=False)
        return _per_leaf(tree, conductivity, self.conductivity_default, "conductivity")

    # --------------------------------------------------------------------- solve
    def solve(self, tree: Octree | None = None) -> OctreeSteadyResult:
        """Solve the current tree and report the field, the fluxes and the balance."""
        tree = self.tree if tree is None else tree
        start = time.perf_counter()
        k = self.conductivities(tree)
        if not np.all(k > 0.0):
            worst = float(np.min(k))
            raise ValueError(f"conductivity must be > 0 on every leaf, got {worst}")
        source = self.sources(tree)
        fixed = self.fixed_cells(tree)

        temperature = tree.solve(source, conductivity=k,
                                 physical_size=self.physical_size, fixed=fixed,
                                 tolerance=self.tolerance)

        i, j, axis, g = self._face_data(tree, k)
        flux = g * (temperature[i] - temperature[j])
        divergence = _divergence(tree.n_cells, i, j, flux)
        jump = _flux_jump(tree, i, j, axis, flux)

        free = np.ones(tree.n_cells, dtype=bool)
        free[list(fixed)] = False
        volumes = (tree.cell_sizes() * self.physical_size) ** 3
        balance = divergence[free] - source[free] * volumes[free]
        source_power = float((source[free] * volumes[free]).sum())
        total_flux = float(divergence[free].sum())
        return OctreeSteadyResult(
            T=temperature, face_flux=flux, divergence=divergence, jump=jump,
            total_flux=total_flux, source_power=source_power,
            residual=float(np.abs(balance).max()) if balance.size else 0.0,
            # relative to the largest rate the mesh carries, not to the power: a wall
            # that both heats and cools the domain has a source power of zero and a
            # ratio against it would say nothing
            balance_error=abs(total_flux - source_power) / max(
                float(np.abs(flux).max()), EPS),
            cells=tree.n_cells, solve_time=time.perf_counter() - start)


def _divergence(n_cells: int, i: np.ndarray, j: np.ndarray,
                flux: np.ndarray) -> np.ndarray:
    """Net heat rate leaving every cell, accumulated face by face.

    Each face rate is added once as an outflow and once as an inflow, so summing the
    result over the mesh cancels face by face and leaves only what crosses the cells the
    caller did not fix: the conservation of the scheme, at machine precision.
    """
    out = np.zeros(n_cells)
    np.add.at(out, i, flux)
    np.add.at(out, j, -flux)
    return out


def _flux_jump(tree: Octree, i: np.ndarray, j: np.ndarray, axis: np.ndarray,
               flux: np.ndarray) -> np.ndarray:
    """Axial flux imbalance per leaf [W] - see :meth:`OctreeSteadySolver.indicator`.

    An axis counts only where the leaf has a face on *both* sides.  At the wall of the
    box the outer face does not exist (the boundary is adiabatic as far as the operator
    knows), so the rate arriving from the interior has nothing to cancel against and the
    whole flux would be charged to a cell whose error is not the question: the second
    difference of a field is not defined without its two neighbours.
    """
    corners = np.array([(leaf.x, leaf.y, leaf.z) for leaf in tree.leaves], dtype=float)
    sizes = np.array([leaf.size for leaf in tree.leaves], dtype=float)
    rows = np.arange(i.size)
    high = corners[j][rows, axis] > corners[i][rows, axis]
    sign = np.where(high, 1.0, -1.0)          # +1: the neighbour is on the high side
    net = np.zeros((tree.n_cells, 3))
    np.add.at(net, (i, axis), sign * flux)
    np.add.at(net, (j, axis), -sign * flux)
    two_sided = (corners > 0.0) & (corners + sizes[:, None] < tree.n)
    return np.abs(np.where(two_sided, net, 0.0)).sum(axis=1)


# ------------------------------------------------------------------ refinement
#: a flux jump below this fraction of the largest face heat rate is round-off and not
#: curvature: a linear field leaves residuals of about 1e-14 of the flux it carries
_NOISE_FLOOR = 1e-12


@dataclass
class OctreeAdaptiveReport:
    """Outcome of the objective-driven refinement: the levels tried and the tree."""

    tree: Octree
    levels: list[ConvergenceLevel] = field(default_factory=list)
    converged: bool = False
    message: str = ""
    objective: str = "objective"
    rounds: int = 0

    @property
    def chosen(self) -> ConvergenceLevel:
        """The last level tried, i.e. the tree the report hands back."""
        return self.levels[-1]

    def summary(self) -> str:
        lines = [self.message]
        for level in self.levels:
            lines.append(
                f"  round {level.level}: {level.cells:,} leaves "
                f"({level.min_size * 1000:.1f}-{level.max_size * 1000:.1f} mm)  "
                f"{self.objective} {level.power:.6g}  "
                f"d {level.d_temperature:.3g} ({100 * level.d_power:.2f}%)")
        return "\n".join(lines)


def refine_on_objective(tree: Octree, solve: Callable[[Octree], OctreeSteadyResult],
                        objective: Callable[[Octree, OctreeSteadyResult], float],
                        target: ConvergenceTarget,
                        indicator: Callable[[Octree, OctreeSteadyResult], np.ndarray]
                        | None = None, *,
                        relative: bool = True, refine_fraction: float = 0.1,
                        floor_ratio: float = 0.05, name: str = "objective",
                        physical_size: float | None = None,
                        progress: Callable[[int, str], None] | None = None
                        ) -> OctreeAdaptiveReport:
    """Refine ``tree`` until the objective stops moving more than ``target`` allows.

    One round is: solve, estimate the local error (``indicator``, by default the flux
    jump of :class:`OctreeSteadySolver`), split the leaves whose estimate is above
    ``floor_ratio`` of the largest one - at most ``refine_fraction`` of the tree, and
    never below the round-off level of the fluxes it carries - and solve again.  The loop
    stops when the objective changes by less than ``target.delta_power`` (``relative``,
    the default) or ``target.delta_temperature`` (absolute) between two rounds, or when
    the cell budget, the round budget or the leaves themselves run out - the last one is
    the honest answer for a field with no feature left to resolve.

    ``solve`` is any callable returning an :class:`OctreeSteadyResult` (an
    :class:`OctreeSteadySolver` is one), which keeps this driver testable without a
    model; ``physical_size`` [m] only scales the sizes the level records report, and is
    taken from the solver when the caller does not pass one.  ``tree`` is refined **in
    place** and returned in the report.
    """
    problems = target.validate()
    if problems:
        raise ValueError("invalid convergence target: " + "; ".join(problems))
    if indicator is None:
        if not isinstance(solve, OctreeSteadySolver):
            raise ValueError("indicator is required when solve is not an "
                             "OctreeSteadySolver")
        indicator = lambda t, r: solve.indicator(r.T, t)      # noqa: E731
    if physical_size is None:
        physical_size = getattr(solve, "physical_size", 1.0)

    result = solve(tree)
    value = float(objective(tree, result))
    levels = [_level_record(tree, 1, value, physical_size)]
    converged, rounds, message = False, 0, ""

    while rounds < max(target.max_levels - 1, 0):
        if tree.n_cells >= target.max_cells:
            message = (f"the cell budget ({target.max_cells:,} leaves) is full at "
                       f"{tree.n_cells:,} leaves")
            break
        values = np.asarray(indicator(tree, result), dtype=float)
        if values.shape != (tree.n_cells,):
            raise ValueError(f"the indicator must return one value per leaf "
                             f"({tree.n_cells}), got shape {values.shape}")
        # the estimate is a magnitude: a signed estimator is as significant below zero
        magnitude = np.abs(values)
        threshold, marked = _mark(tree, magnitude, refine_fraction, floor_ratio,
                                  noise=_NOISE_FLOOR * float(np.abs(result.face_flux).max()))
        if marked == 0:
            message = (f"converged after {rounds} round(s): no leaf shows a flux jump "
                       f"above {threshold:.3g} W - the field carries no feature left to "
                       f"resolve") if rounds else (
                       f"nothing to refine: no leaf shows a flux jump above "
                       f"{threshold:.3g} W ({floor_ratio:.2f} of the maximum, or the "
                       f"round-off of the {np.abs(result.face_flux).max():.3g} W the "
                       f"faces carry), so refining cannot improve the objective")
            converged = rounds > 0
            break
        if tree.n_cells + 7 * marked > target.max_cells:
            message = (f"the cell budget ({target.max_cells:,} leaves) is the limit: the "
                       f"next round would add {7 * marked:,} leaves to {tree.n_cells:,}")
            break
        if progress is not None:
            progress(int(100 * rounds / max(target.max_levels - 1, 1)),
                     f"round {rounds + 1}: refining {marked} of {tree.n_cells} leaves")
        tree.refine(dict(zip(tree.leaves, magnitude, strict=True)).__getitem__, threshold)
        rounds += 1
        result = solve(tree)
        value = float(objective(tree, result))
        levels.append(_level_record(tree, rounds + 1, value, physical_size,
                                    levels[-1].power))
        if _reached(levels[-1], target, relative):
            converged = True
            message = (f"converged after {rounds} round(s): {tree.n_cells:,} leaves, "
                       f"{name} {value:.6g}, last change {levels[-1].d_temperature:.3g}"
                       f" ({100 * levels[-1].d_power:.2f}%) within the tolerance")
            break

    if not message:
        message = (f"not converged after {rounds} round(s): {tree.n_cells:,} leaves, "
                   f"{name} {value:.6g} still moving by "
                   f"{levels[-1].d_temperature:.3g} ({100 * levels[-1].d_power:.2f}%)")
    return OctreeAdaptiveReport(tree=tree, levels=levels, converged=converged,
                                message=message, objective=name, rounds=rounds)


def _mark(tree: Octree, values: np.ndarray, refine_fraction: float, floor_ratio: float,
          noise: float = 0.0) -> tuple[float, int]:
    """Threshold and number of splittable leaves marking the largest flux jumps.

    ``values`` are magnitudes.  The threshold is relative to the *field* on purpose: a
    flux has units of the model,
    and a fixed threshold would be right for one problem and useless for the next.  The
    top ``refine_fraction`` of the cells is added at most, and a leaf is only marked
    above ``floor_ratio`` of the maximum - but never below ``noise``, the round-off
    level of the estimator: a linear field leaves residuals of ``1e-14`` of the flux it
    carries, and a rule that looks only at its own maximum would refine those.
    """
    if not 0.0 < refine_fraction < 1.0:
        raise ValueError(f"refine_fraction must be in (0, 1), got {refine_fraction}")
    if not 0.0 <= floor_ratio < 1.0:
        raise ValueError(f"floor_ratio must be in [0, 1), got {floor_ratio}")
    splittable = np.array([leaf.level > 0 for leaf in tree.leaves], dtype=bool)
    live = values[splittable & (values > 0.0)]
    if live.size == 0:
        return max(noise, 0.0), 0
    threshold = max(floor_ratio * float(live.max()), noise)
    marked = int(np.count_nonzero(splittable & (values > threshold)))
    cap = max(int(np.ceil(refine_fraction * tree.n_cells)), 1)
    if marked > cap:
        # a signal that is broad has to be cut: the cap keeps one round from
        # multiplying most of the tree by eight
        threshold = max(threshold, float(np.quantile(live, 1.0 - refine_fraction)))
        marked = int(np.count_nonzero(splittable & (values > threshold)))
    return threshold, marked


def _level_record(tree: Octree, number: int, value: float, physical_size: float,
                  previous: float | None = None) -> ConvergenceLevel:
    """One round of the sequence, in the vocabulary of the structured driver.

    The objective travels in ``power`` (the field the tolerance is applied to), the
    axial change in ``d_temperature``.  ``order`` and ``gci`` stay undefined: an
    adaptive sequence has no single refinement ratio, so there is no Richardson
    extrapolation to report - the objective change *is* the criterion here.
    """
    sizes = tree.cell_sizes() * physical_size
    record = ConvergenceLevel(level=number, cells=tree.n_cells,
                              min_size=float(sizes.min()), max_size=float(sizes.max()),
                              worst_ratio=float(sizes.max() / sizes.min()), power=value)
    if previous is not None:
        record.d_temperature = abs(value - previous)
        record.d_power = record.d_temperature / max(abs(previous), 1e-9)
    return record


def _reached(level: ConvergenceLevel, target: ConvergenceTarget,
             relative: bool) -> bool:
    """True when the objective moved less than the tolerance between two rounds."""
    if not (np.isfinite(level.d_power) and np.isfinite(level.d_temperature)):
        return False
    if relative:
        return level.d_power <= target.delta_power
    return level.d_temperature <= target.delta_temperature
