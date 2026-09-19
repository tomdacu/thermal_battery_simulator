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

The driver is pure: the caller supplies ``build_mesh(spec)`` and ``observables(mesh)``,
which keeps it testable without a GUI or a solver.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh3D
from ..core.refinement import GridSpec


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
    """Outcome of the search: the levels tried and the mesh to use."""

    spec: GridSpec
    levels: list[ConvergenceLevel] = field(default_factory=list)
    converged: bool = False
    message: str = ""
    order: float = float("nan")
    gci_temperature: float = float("nan")
    gci_power: float = float("nan")

    @property
    def chosen(self) -> ConvergenceLevel:
        return self.levels[-1]

    def summary(self) -> str:
        lines = [self.message]
        for level in self.levels:
            extra = ""
            if np.isfinite(level.order):
                extra = f"  p {level.order:.2f}"
            if np.isfinite(level.gci):
                extra += f"  GCI {100 * level.gci:.2f}%"
            lines.append(
                f"  level {level.level}: {level.cells:,} cells "
                f"({level.min_size * 1000:.1f}-{level.max_size * 1000:.0f} mm)  "
                f"T_mean {level.t_mean_storage:.2f} K  "
                f"dT {level.d_temperature:.3f} K  dP {100 * level.d_power:.2f}%"
                f"{extra}")
        if np.isfinite(self.gci_temperature):
            lines.append(
                f"  discretisation uncertainty of the chosen grid "
                f"(GCI, Fs = 1.25): dT {self.gci_temperature:.3f} K, "
                f"dP {100 * self.gci_power:.3f}%")
        return "\n".join(lines)


def find_mesh(build_mesh: Callable[[GridSpec], Mesh3D],
              observables: Callable[[Mesh3D], dict],
              spec: GridSpec,
              target: ConvergenceTarget,
              progress: Callable[[int, str], None] | None = None,
              should_stop: Callable[[], bool] | None = None) -> ConvergenceReport:
    """Find the coarsest grid whose answer has stopped moving.

    ``observables(mesh)`` must return at least ``t_mean_storage`` and ``power``
    (the heat leaving the domain [W]); ``t_max`` is reported but not used as a
    criterion, because a peak value converges more slowly than an integral and
    would force an unnecessary fine mesh everywhere.
    """
    problems = target.validate()
    if problems:
        raise ValueError("invalid convergence target: " + "; ".join(problems))

    levels: list[ConvergenceLevel] = []
    chosen = spec
    blocked = False
    attempts = 0
    limit = target.max_levels + 8                 # a blocked level costs no solve
    probe = int(np.clip(target.probe_levels, 2, max(target.max_levels - 1, 2)))

    def solve_at(scale: float) -> ConvergenceLevel | None:
        """Build and solve one level; ``None`` when the grid is not finer than the last."""
        nonlocal chosen
        spec_level = spec.scaled(scale, min_size=target.min_size,
                                 max_cells=target.max_cells)
        if progress is not None:
            progress(int(100 * len(levels) / max(target.max_levels, 1)),
                     f"level {len(levels) + 1}: building the mesh")
        mesh = build_mesh(spec_level)
        chosen = spec_level
        summary = mesh.grid_summary()
        cells = int(summary["cells"])
        if levels and cells <= levels[-1].cells * 1.02:
            return None                    # no finer: asking again would tell us nothing
        if progress is not None:
            progress(int(100 * len(levels) / max(target.max_levels, 1)),
                     f"level {len(levels) + 1}: solving {cells:,} cells")
        values = observables(mesh)
        entry = ConvergenceLevel(
            level=len(levels) + 1, scale=scale, cells=cells,
            min_size=float(summary["min_size"]), max_size=float(summary["max_size"]),
            worst_ratio=float(summary["worst_ratio"]),
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
        message = (f"converged at level {entry.level} ({how}): {entry.cells:,} cells, "
                   f"estimated error dT {error_t:.3f} K <= "
                   f"{target.delta_temperature:.3f} K, dP "
                   f"{100 * error_p / max(abs(entry.power), 1e-9):.2f}% <= "
                   f"{100 * target.delta_power:.2f}%")
        if np.isfinite(order):
            message += f"; observed order p = {order:.2f}"
        return ConvergenceReport(spec=chosen, levels=levels, converged=True,
                                 message=message, order=order,
                                 gci_temperature=gci_t, gci_power=gci_p)

    # ---------------------------------------------------------------- probe
    scale = max(target.start_scale, 1.0)
    while len(levels) < probe and attempts < limit:
        attempts += 1
        if should_stop is not None and should_stop():
            return ConvergenceReport(spec=chosen, levels=levels, converged=False,
                                     message="cancelled by the user")
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
            return ConvergenceReport(spec=chosen, levels=levels, converged=False,
                                     message="cancelled by the user")
        entry = solve_at(prediction)
        if entry is None:
            blocked = True                 # cannot go finer: the limit is elsewhere
            break
        if _reached(levels, target):
            return converged(entry, "predicted")
        # still moving: the next pass re-fits with the new pair

    if not levels:
        return ConvergenceReport(spec=chosen, levels=levels, converged=False,
                                 message=("no grid could be built: the cell budget and "
                                          "the minimum cell size reject every level"))

    order, gci_t, gci_p = _fit(levels, target)
    last = levels[-1]
    error_t, error_p = _errors(levels, target)
    budget_limited = _budget_limited(levels, target) or blocked
    if np.isfinite(error_t):
        message = (f"not converged after {len(levels)} levels: the finest mesh "
                   f"({last.cells:,} cells) still carries an estimated error of "
                   f"{error_t:.3f} K / "
                   f"{100 * error_p / max(abs(last.power), 1e-9):.2f}%")
    else:
        message = (f"not converged: the first grid ({last.cells:,} cells) already "
                   f"fills the cell budget")
    if budget_limited:
        message += (f".  The cell budget ({target.max_cells:,} cells) is the limit: "
                    f"raise it or loosen the tolerances")
    else:
        message += (".  Raise max_levels, loosen the tolerances, or lower the "
                    f"minimum cell size ({target.min_size * 1000:.1f} mm)")
    if np.isfinite(gci_t):
        message += (f"  The estimated discretisation uncertainty of the finest grid "
                    f"is dT {gci_t:.3f} K / dP {100 * gci_p:.2f}% (GCI).")
    return ConvergenceReport(spec=chosen, levels=levels, converged=False, message=message,
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
    # E ~ h^p and h ~ scale, so the scale that meets the tolerance is e^(-1/p) times
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


def _change(previous: float, current: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(current):
        return float("inf")
    return abs(current - previous)


def _relative_change(previous: float, current: float) -> float:
    if not np.isfinite(previous) or not np.isfinite(current):
        return float("inf")
    return abs(current - previous) / max(abs(previous), 1e-9)
