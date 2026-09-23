"""Automatic mesh selection: choose the grid from the *accuracy* you ask for.

There is no way to know the right cell size a priori: what a mesher can do is make
the discrete answer stop moving.  This module drives a sequence of grids and declares
the mesh good enough when the answer stops moving by more than the tolerances asked
for:

* ``delta_temperature`` - change of the storage mean temperature [K];
* ``delta_power`` - *relative* change of the heat leaving the domain.

The method is the classical one for discretisation-error estimation (Richardson
extrapolation, reported as Roache's Grid Convergence Index; ASME V&V 20 and Celik et
al. 2008 recommend exactly this three-grid procedure, with a refinement factor above
1.3 and a factor of safety of 1.25):

1. **probe** - three cheap coarse grids (``start_scale`` times the requested targets,
   each one ``refine`` times finer) give the *observed order of accuracy* ``p``;
2. **predict** - the error model ``E = C h^p`` calibrated on those three grids gives
   the cell size that meets the tolerance, so the search *jumps* to the grid it needs
   instead of refining one level at a time;
3. **verify** - the predicted grid is solved and the change against the previous grid
   is checked; if it still moves, the prediction is repeated with the new pair.

``p`` is floored at ``min_order`` (default 1): a mixed-order scheme - second order
inside the domain, first order at the convective faces - has an observed order below
its formal one, and a conservative order asks for a *finer* grid when in doubt.  The
report carries the observed order and the GCI of the chosen grid, so the discretisation
uncertainty is stated rather than implied.

Energy conservation is *not* a criterion here: the finite-volume scheme conserves
energy to machine precision on **any** grid (see ``tests/test_graded_mesh.py``); what
changes with the grid is the accuracy of the field itself.

The same search drives an adaptive mesh, and the request says which road it is walking
(see :class:`AdaptivePlan`).  A level still asks for a *scale*: a graded
:class:`~src.core.refinement.GridSpec` realises it by scaling its targets
(:meth:`GridSpec.scaled`), while a tree realises it by scaling the bands of the a priori
plan (:func:`src.analysis.mesh_plan.refinement_bands`) and adding one round of refinement
on the flux-jump indicator of the field already solved.  That is what keeps the reported
order, the prediction and the GCI the same code on both meshes - and it is why the plan
keeps its say: the flux-jump estimate is blind at the box faces, which is exactly where a
film puts the largest error of the scheme, so the indicator alone would leave the wall
layer coarse.  :func:`find_mesh` states it in full.

The driver is pure: the caller supplies ``build_mesh(spec)`` and ``observables(mesh)``,
which keeps it testable without a GUI or a solver.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import numpy as np

from ..core.mesh_api import MeshAPI
from ..core.refinement import GridSpec

# the tree module imports this one (through ``octree_solver``), so its names can only be
# named here, never imported: the driver reaches a tree through the methods it is asked for
if TYPE_CHECKING:
    from ..core.adaptive_mesh import AdaptiveMesh, RefinementBand


@dataclass(frozen=True)
class AdaptivePlan:
    """The tree a search level builds: the box, the refinement bands, and nothing else.

    This is the adaptive counterpart of :class:`~src.core.refinement.GridSpec`: the same
    physical targets (the panel's bands map to :class:`RefinementBand`s one to one, see
    :func:`src.analysis.mesh_plan.refinement_bands`), the same a priori estimate, expressed
    as the boxes an octree refines.  ``n_finest`` and ``physical_size`` fix the box the way
    :meth:`AdaptiveMesh.from_bands` takes them - ``n_finest`` cells of ``physical_size``
    [m] on a side, ``n_finest`` a power of two - which is also the resolution floor: no
    leaf of the tree can be smaller than one finest cell.
    """

    n_finest: int
    physical_size: float
    bands: tuple[RefinementBand, ...]
    base_level: int | None = None
    #: an anisotropic tree (:class:`~src.core.box_tree.BoxTree`): ``n_finest`` and
    #: ``physical_size`` are then the plan's, and these the height's (``None``: a cubic
    #: octree)
    n_z: int | None = None
    dz: float | None = None

    @property
    def anisotropic(self) -> bool:
        return self.n_z is not None and self.dz is not None

    def scaled(self, factor: float) -> AdaptivePlan:
        """The same plan with every band target multiplied by ``factor``.

        The tree's answer to :meth:`GridSpec.scaled`, and the way a search level asks the
        tree for a resolution: the bands are the physical request, so scaling them is what
        makes level *k* finer than level *k-1*.  Only the bands move - a tree has no min
        or max size rails to scale, because a leaf edge is a power of two of
        ``physical_size`` and never goes below it.
        """
        return replace(self, bands=tuple(
            replace(band, size=band.size * factor,
                    size_z=None if band.size_z is None else band.size_z * factor)
            for band in self.bands))


@dataclass(frozen=True)
class ConvergenceTarget:
    """How accurately the user wants the answer (and what may be spent)."""

    delta_temperature: float = 2.0        # [K] on the storage mean
    delta_power: float = 0.02             # relative on the domain loss
    max_levels: int = 5                   # solves allowed, probe included
    refine: float = 0.7                   # target scale from one level to the next
    min_size: float = 0.004               # [m] never refine below this
    max_cells: int = 400_000
    #: the probe starts this much coarser than the requested targets (cheap levels);
    #: too coarse and the probe is outside the asymptotic range, where no order fits
    start_scale: float = 1.0
    #: a single prediction may not jump more than this many refine steps
    max_jump: int = 8
    #: Roache's factor of safety for the reported GCI (1.25 for three grids)
    safety: float = 1.25
    #: conservative floor on the observed order of accuracy
    min_order: float = 1.0
    max_order: float = 3.0
    probe_levels: int = 3

    def validate(self) -> list[str]:
        problems = []
        if self.delta_temperature <= 0:
            problems.append("delta_temperature must be > 0")
        if self.delta_power <= 0:
            problems.append("delta_power must be > 0")
        if not 0.2 <= self.refine < 1.0:
            problems.append("refine must be in [0.2, 1)")
        if self.max_levels < 2:
            problems.append("max_levels must be >= 2 (one level cannot converge)")
        if self.start_scale < 1.0:
            problems.append("start_scale must be >= 1")
        if self.safety < 1.0:
            problems.append("safety must be >= 1")
        if self.min_order <= 0 or self.max_order < self.min_order:
            problems.append("orders must satisfy 0 < min_order <= max_order")
        if self.max_jump < 1:
            problems.append("max_jump must be >= 1")
        return problems


@dataclass
class ConvergenceLevel:
    """One grid of the sequence and what it produced."""

    level: int
    cells: int
    min_size: float
    max_size: float
    worst_ratio: float
    scale: float = 1.0
    t_mean_storage: float = float("nan")
    t_max: float = float("nan")
    power: float = float("nan")
    d_temperature: float = float("nan")
    d_power: float = float("nan")
    order: float = float("nan")
    gci: float = float("nan")


@dataclass
class ConvergenceReport:
    """Outcome of the search: the levels tried and the mesh to use.

    ``spec`` is the request of the chosen level on a graded grid (``build_mesh(spec)``
    rebuilds it), and the plan the search started from on a tree: a tree level is a round
    on a solved field, so what a caller rebuilds there is the *band* level, while the mesh
    the search actually chose is ``mesh``, the tree the last level left behind.
    """

    spec: GridSpec | AdaptivePlan
    levels: list[ConvergenceLevel] = field(default_factory=list)
    converged: bool = False
    message: str = ""
    order: float = float("nan")
    gci_temperature: float = float("nan")
    gci_power: float = float("nan")
    mesh: MeshAPI | AdaptiveMesh | None = None

    @property
    def chosen(self) -> ConvergenceLevel:
        return self.levels[-1]

    def summary(self) -> str:
        unit = "leaves" if isinstance(self.spec, AdaptivePlan) else "cells"
        lines = [self.message]
        for level in self.levels:
            extra = ""
            if np.isfinite(level.order):
                extra = f"  p {level.order:.2f}"
            if np.isfinite(level.gci):
                extra += f"  GCI {100 * level.gci:.2f}%"
            lines.append(
                f"  level {level.level}: {level.cells:,} {unit} "
                f"({level.min_size * 1000:.1f}-{level.max_size * 1000:.0f} mm)  "
                f"T_mean {level.t_mean_storage:.2f} K  "
                f"dT {level.d_temperature:.3f} K  dP {100 * level.d_power:.2f}%"
                f"{extra}")
        if np.isfinite(self.gci_temperature):
            lines.append(
                f"  discretisation uncertainty of the chosen mesh "
                f"(GCI, Fs = 1.25): dT {self.gci_temperature:.3f} K, "
                f"dP {100 * self.gci_power:.3f}%")
        return "\n".join(lines)


def find_mesh(build_mesh: Callable[[GridSpec | AdaptivePlan], MeshAPI],
              observables: Callable[[MeshAPI], dict],
              spec: GridSpec | AdaptivePlan,
              target: ConvergenceTarget,
              progress: Callable[[int, str], None] | None = None,
              should_stop: Callable[[], bool] | None = None) -> ConvergenceReport:
    """Find the coarsest mesh whose answer has stopped moving.

    ``observables(mesh)`` must return at least ``t_mean_storage`` and ``power``
    (the heat leaving the domain [W]); ``t_max`` is reported but not used as a
    criterion, because a peak value converges more slowly than an integral and
    would force an unnecessary fine mesh everywhere.

    ``spec`` is the request, and both roads ask for the same thing: a *scale* - the level's
    cell size relative to the first one.  A :class:`GridSpec` turns it into a graded grid
    (:meth:`GridSpec.scaled`), an :class:`AdaptivePlan` into a tree: the first level is
    ``build_mesh(plan)`` (the a priori bands themselves), and every following one refines
    the tree already in hand in two steps - the bands scaled by that factor
    (:meth:`AdaptivePlan.scaled`, the same physical request the graded road makes) and one
    round on the flux-jump indicator of the field just solved
    (:meth:`AdaptiveMesh.refine_round`).

    Why both steps, and not the indicator alone: the estimate is the *second* difference of
    the field, so ``OctreeSteadySolver`` reports it as zero for a leaf with no neighbour on
    both sides of an axis - and a leaf sitting on a box face has none on one side.  The
    wall layer, which is exactly where a film or a fixed temperature puts the largest
    truncation error of a cell-centred scheme, is therefore invisible to the estimator, and
    the a priori plan is what keeps it refined: the bands carry the physics (``thickness /
    N``, ``2 k / h``, the tube pitch) on both roads, and the indicator rounds add the
    refinement no a priori rule can know about.  The scale keeps meaning what it means on a
    graded grid, so the observed order, the prediction, the verification and the GCI below
    are the same code for both meshes.

    ``build_mesh`` is asked for the first level only on the tree road: a round is a rule of
    the field, and the field belongs to the tree the driver already holds.  The per-leaf
    fields survive the rounds the way :meth:`AdaptiveMesh.refine` documents - a split leaf
    hands its material, sources and temperature to its children - so a caller that paints
    the geometry inside ``build_mesh`` keeps the model it painted.
    """
    problems = target.validate()
    if problems:
        raise ValueError("invalid convergence target: " + "; ".join(problems))

    tree = isinstance(spec, AdaptivePlan)
    unit = "leaves" if tree else "cells"
    levels: list[ConvergenceLevel] = []
    chosen: GridSpec | AdaptivePlan = spec
    held: MeshAPI | AdaptiveMesh | None = None      # the mesh of the last level
    blocked = False
    attempts = 0
    limit = target.max_levels + 8                 # a blocked level costs no solve
    probe = int(np.clip(target.probe_levels, 2, max(target.max_levels - 1, 2)))
    # How much of the tree one round may refine: everything the floor flags.  A search has
    # ``max_levels`` solves, not a long walk - the 10% that :meth:`AdaptiveMesh.refine_round`
    # inherits from :func:`refine_on_objective` would move the objective too slowly to
    # reach a tolerance in that budget - and what bounds the round is the cell budget,
    # checked before it is applied.  The cap stays a safety valve.
    round_fraction = 0.9

    def report(**kwargs) -> ConvergenceReport:
        """A report of the levels tried; a tree search hands its mesh back with it."""
        return ConvergenceReport(mesh=held if tree else None, **kwargs)

    def stopped() -> ConvergenceReport:
        return report(spec=chosen, levels=levels, converged=False,
                      message="cancelled by the user")

    def solve_at(scale: float) -> ConvergenceLevel | None:
        """Build or refine and solve one level; ``None`` when it is not finer than the last."""
        nonlocal chosen, held
        if progress is not None:
            progress(int(100 * len(levels) / max(target.max_levels, 1)),
                     f"level {len(levels) + 1}: building the mesh")
        if tree:
            if held is None:                  # the first level: the a priori bands
                mesh = build_mesh(spec)
            else:
                before = held.n_cells
                # the plan at this level's scale, under the same cell budget the rounds
                # below are checked against: a request the caller cannot afford is met as
                # closely as the budget allows, and the search sees the stop
                held.refine_bands(spec.scaled(scale).bands, max_cells=target.max_cells)
                # one round on the field in hand: what the plan cannot know, and the one
                # step of a level that is a rule of the solution rather than of the request
                if held.refine_round(refine_fraction=round_fraction,
                                     max_cells=target.max_cells) == 0 and \
                        held.n_cells <= before:
                    return None               # no band left to meet, no feature to resolve
                mesh = held
        else:
            spec_level = spec.scaled(scale, min_size=target.min_size,
                                     max_cells=target.max_cells)
            mesh = build_mesh(spec_level)
            chosen = spec_level
        held = mesh
        cells, min_size, max_size, worst_ratio = _mesh_summary(mesh)
        if levels and cells <= levels[-1].cells * 1.02:
            return None                    # no finer: asking again would tell us nothing
        if progress is not None:
            progress(int(100 * len(levels) / max(target.max_levels, 1)),
                     f"level {len(levels) + 1}: solving {cells:,} {unit}")
        values = observables(mesh)
        entry = ConvergenceLevel(
            level=len(levels) + 1,
            # The scale the level *reached*.  A graded grid can declare it - every target
            # was scaled by exactly that factor - while a tree can only measure it: what a
            # round does to the cell size depends on where the field asked for refinement,
            # so the level records the volume weighted mean leaf edge [m], the equivalent
            # uniform cell size of the mesh.  Only the ratio between two levels is read
            # from it, and that ratio is then what actually happened rather than what was
            # asked for - the request is realised as far as the box and the budget allow.
            scale=_tree_scale(mesh) if tree else float(scale),
            cells=cells, min_size=min_size, max_size=max_size, worst_ratio=worst_ratio,
            t_mean_storage=float(values.get("t_mean_storage", float("nan"))),
            t_max=float(values.get("t_max", float("nan"))),
            power=float(values.get("power", float("nan"))),
        )
        if levels:
            entry.d_temperature = _change(levels[-1].t_mean_storage, entry.t_mean_storage)
            entry.d_power = _relative_change(levels[-1].power, entry.power)
        levels.append(entry)
        return entry

    def converged(entry: ConvergenceLevel, how: str) -> ConvergenceReport:
        order, gci_t, gci_p = _fit(levels, target)
        entry.order = order
        entry.gci = gci_t if gci_t >= gci_p else gci_p
        error_t, error_p = _errors(levels, target)
        message = (f"converged at level {entry.level} ({how}): {entry.cells:,} {unit}, "
                   f"estimated error dT {error_t:.3f} K <= "
                   f"{target.delta_temperature:.3f} K, dP "
                   f"{100 * error_p / max(abs(entry.power), 1e-9):.2f}% <= "
                   f"{100 * target.delta_power:.2f}%")
        if np.isfinite(order):
            message += f"; observed order p = {order:.2f}"
        return report(spec=chosen, levels=levels, converged=True,
                      message=message, order=order,
                      gci_temperature=gci_t, gci_power=gci_p)

    # ---------------------------------------------------------------- probe
    scale = max(target.start_scale, 1.0)
    while len(levels) < probe and attempts < limit:
        attempts += 1
        if should_stop is not None and should_stop():
            return stopped()
        entry = solve_at(scale)
        if entry is None:
            if levels:
                blocked = True
                break
            scale *= target.refine
            continue
        if entry.level > 1 and _reached(levels, target):
            return converged(entry, "probe")
        scale *= target.refine

    # ------------------------------------------------- predict and verify
    while attempts < limit and len(levels) < target.max_levels:
        prediction = _prediction(levels, target)
        if prediction is None:
            break
        attempts += 1
        if should_stop is not None and should_stop():
            return stopped()
        entry = solve_at(prediction)
        if entry is None:
            blocked = True                 # cannot go finer: the limit is elsewhere
            break
        if _reached(levels, target):
            return converged(entry, "predicted")
        # still moving: the next pass re-fits with the new pair

    if not levels:
        return report(spec=chosen, levels=levels, converged=False,
                      message=("no mesh could be built: the cell budget and the minimum "
                               "cell size reject every level"))

    order, gci_t, gci_p = _fit(levels, target)
    last = levels[-1]
    error_t, error_p = _errors(levels, target)
    budget_limited = _budget_limited(levels, target) or blocked
    if np.isfinite(error_t):
        message = (f"not converged after {len(levels)} levels: the finest mesh "
                   f"({last.cells:,} {unit}) still carries an estimated error of "
                   f"{error_t:.3f} K / "
                   f"{100 * error_p / max(abs(last.power), 1e-9):.2f}%")
    else:
        message = (f"not converged: the first mesh ({last.cells:,} {unit}) already "
                   f"fills the cell budget")
    if budget_limited:
        message += (f".  The cell budget ({target.max_cells:,} cells) is the limit: "
                    f"raise it or loosen the tolerances")
    else:
        message += (".  Raise max_levels, loosen the tolerances, or lower the "
                    f"minimum cell size ({target.min_size * 1000:.1f} mm)")
    if np.isfinite(gci_t):
        message += (f"  The estimated discretisation uncertainty of the finest mesh "
                    f"is dT {gci_t:.3f} K / dP {100 * gci_p:.2f}% (GCI).")
    return report(spec=chosen, levels=levels, converged=False, message=message,
                  order=order, gci_temperature=gci_t, gci_power=gci_p)


# --------------------------------------------------------------------- helpers
def _reached(levels: list[ConvergenceLevel], target: ConvergenceTarget) -> bool:
    """True when the *estimated error* of the finest grid is inside the tolerances.

    The raw change between two grids only measures the error when the grids are
    adjacent; with a predicted jump it is dominated by the coarse grid and would
    reject a grid that is already good.  The Richardson estimate divides the change by
    ``r^p - 1``, which is what makes the criterion independent of the jump size.
    """
    error_t, error_p = _errors(levels, target)
    if not (np.isfinite(error_t) and np.isfinite(error_p)):
        return False
    reference = max(abs(levels[-1].power), 1e-9)
    return (error_t <= target.delta_temperature
            and error_p <= target.delta_power * reference)


def _errors(levels: list[ConvergenceLevel], target: ConvergenceTarget
            ) -> tuple[float, float]:
    """(absolute error on T, absolute error on the power) of the finest grid [K, W]."""
    if len(levels) < 2:
        return float("inf"), float("inf")
    recent = levels[-3:] if len(levels) >= 3 else levels[-2:]
    # the error estimate belongs to the *finest* pair: with predicted jumps the
    # ratios are not all equal, so the last one is the one that matters
    ratio = recent[-2].scale / recent[-1].scale
    if ratio <= 1.0:
        return float("inf"), float("inf")
    t_values = [level.t_mean_storage for level in recent]
    p_values = [level.power for level in recent]
    order = target.min_order
    if len(recent) >= 3:
        orders = [o for o in (_observed_order(t_values, ratio, target),
                              _observed_order(p_values, ratio, target)) if o is not None]
        if orders:
            order = min(orders)
    return (_richardson_error(t_values, ratio, order),
            _richardson_error(p_values, ratio, order))


def _observed_order(values: list[float], ratio: float,
                    target: ConvergenceTarget) -> float | None:
    """Order of accuracy from three grids (Richardson, Celik et al. 2008)."""
    if len(values) < 3 or ratio <= 1.0:
        return None
    d1 = values[1] - values[0]
    d2 = values[2] - values[1]
    if not (np.isfinite(d1) and np.isfinite(d2)) or d1 == 0.0 or d2 == 0.0:
        return None
    if d1 * d2 <= 0.0:                     # not monotone: no order can be fitted
        return None
    order = float(np.log(abs(d1 / d2)) / np.log(ratio))
    return float(np.clip(order, target.min_order, target.max_order))


def _richardson_error(values: list[float], ratio: float, order: float) -> float:
    """Estimated error of the finest grid against the extrapolated limit."""
    if len(values) < 2 or not np.isfinite(order):
        return float("inf")
    denominator = ratio ** order - 1.0
    if denominator <= 0:
        return float("inf")
    return float(abs(values[-1] - values[-2]) / denominator)


def _fit(levels: list[ConvergenceLevel], target: ConvergenceTarget
         ) -> tuple[float, float, float]:
    """(order, GCI on T, GCI on the power) of the finest grid (relative)."""
    if len(levels) < 3:
        return float("nan"), float("nan"), float("nan")
    recent = levels[-3:]
    ratio = recent[0].scale / recent[1].scale
    if ratio <= 1.0:
        return float("nan"), float("nan"), float("nan")
    t_values = [level.t_mean_storage for level in recent]
    p_values = [level.power for level in recent]
    orders = [o for o in (_observed_order(t_values, ratio, target),
                          _observed_order(p_values, ratio, target)) if o is not None]
    if not orders:
        return float("nan"), float("nan"), float("nan")
    order = min(orders)                    # the most conservative of the two
    error_t, error_p = _errors(levels, target)
    reference = max(abs(p_values[-1]), 1e-9)
    return (order, target.safety * error_t / max(abs(t_values[-1]), 1e-9),
            target.safety * error_p / reference)


def _prediction(levels: list[ConvergenceLevel], target: ConvergenceTarget) -> float | None:
    """Scale the requested targets must reach to meet the tolerances.

    Calibrates ``E = C h^p`` on the last three levels and inverts it for the
    tolerance: the search jumps straight to the grid it needs.  Returns ``None``
    when no order could be fitted (the caller then refines the plain way).
    """
    if len(levels) < 3:
        return None
    recent = levels[-3:]
    ratio = recent[0].scale / recent[1].scale
    if ratio <= 1.0:
        return None
    t_values = [level.t_mean_storage for level in recent]
    p_values = [level.power for level in recent]
    p_t = _observed_order(t_values, ratio, target)
    p_p = _observed_order(p_values, ratio, target)
    if p_t is None and p_p is None:
        return None
    order = min(o for o in (p_t, p_p) if o is not None)

    # how far each observable is from its tolerance, in units of the tolerance
    errors = []
    if p_t is not None:
        errors.append(_richardson_error(t_values, ratio, p_t) / target.delta_temperature)
    if p_p is not None:
        denominator = target.delta_power * max(abs(p_values[-1]), 1e-9)
        errors.append(_richardson_error(p_values, ratio, p_p) / denominator)
    worst = max(errors)
    if not np.isfinite(worst):
        return None
    if worst <= 1.0:                       # the fit says it is already accurate enough
        return recent[-1].scale * target.refine
    # E ~ h^p and h ~ scale, so the scale that meets the tolerance is worst^(-1/p) times
    # the current one.  The jump is bounded on both sides: at least one refine step
    # (otherwise the search would stall) and at most max_jump steps (a prediction made
    # outside the asymptotic range must not be trusted blindly).
    current = recent[-1].scale
    predicted = current * worst ** (-1.0 / order)
    return float(np.clip(predicted, current * target.refine ** target.max_jump,
                         current * target.refine))


def _budget_limited(levels: list[ConvergenceLevel], target: ConvergenceTarget) -> bool:
    """True when the finest grid already sits on the cell budget."""
    return levels[-1].cells >= 0.98 * target.max_cells


def _tree_scale(mesh: AdaptiveMesh) -> float:
    """Resolution one tree level reached: its volume weighted mean leaf edge [m].

    The tree's counterpart of the factor a graded grid scales its targets by, and measured
    rather than declared: a round refines the leaves the field flags, so what it does to
    the cell size is a property of the solution.  The *volume weighted* mean is the
    equivalent uniform cell size - the resolution the integrals see - and unlike the
    finest leaf edge it cannot be dragged down by a handful of leaves the estimator
    deepened: it moves when the mesh really got finer, which is what the error model's
    ratio has to mean.
    """
    volume = np.asarray(mesh.V, dtype=float)
    sizes = np.cbrt(volume)
    total = float(volume.sum())
    return float((sizes * volume).sum() / total) if total > 0.0 else float(sizes.mean())


def _mesh_summary(mesh: MeshAPI) -> tuple[int, float, float, float]:
    """``(cells, min_size, max_size, worst ratio)`` of the mesh one level produced.

    A graded :class:`~src.core.mesh.Mesh3D` answers with its own ``grid_summary`` - the
    same numbers the GUI shows - so a structured search records exactly what it always
    did.  A tree is read from the members the protocol does carry, ``n_cells`` and the
    cell volumes, plus its face list: ``mesh_api`` leaves the report vocabulary out of the
    contract on purpose, so the search takes the numbers it needs instead of asking a
    mesh to present them.  The neighbour ratio of a tree is bounded by the 2:1 balance the
    octree keeps after every mutation, and it is measured here rather than assumed.
    """
    summary = getattr(mesh, "grid_summary", None)
    if summary is not None:
        realised = summary()
        return (int(realised["cells"]), float(realised["min_size"]),
                float(realised["max_size"]), float(realised["worst_ratio"]))
    sizes = np.cbrt(np.asarray(mesh.V, dtype=float))
    faces = mesh.faces()
    first = np.fromiter((face[0] for face in faces), dtype=int, count=len(faces))
    second = np.fromiter((face[1] for face in faces), dtype=int, count=len(faces))
    ratio = (np.maximum(sizes[first], sizes[second])
             / np.minimum(sizes[first], sizes[second]))
    return (int(mesh.n_cells), float(sizes.min()), float(sizes.max()),
            float(ratio.max()) if ratio.size else 1.0)


def _change(previous: float, current: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(current):
        return float("inf")
    return abs(current - previous)


def _relative_change(previous: float, current: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(current):
        return float("inf")
    return abs(current - previous) / max(abs(previous), 1e-9)
