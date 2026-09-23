"""Linear solve layer: method selection, preconditioners, convergence reporting.

Everything runs on the CPU through SciPy.  ``cg`` on a non-symmetric matrix is
refused explicitly: an unguarded CG silently diverges on an asymmetric operator.
Every substitution is reported in :attr:`LinearResult.notes` instead of being
swallowed.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as splinalg
import weakref


try:
    import pyamg

    HAS_PYAMG = True
except Exception:  # pragma: no cover
    pyamg = None
    HAS_PYAMG = False

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

METHODS = ("direct", "cg", "bicgstab", "gmres")
PRECONDITIONERS = ("none", "jacobi", "ilu", "amg", "amg_rs", "amg_sa")


@dataclass
class LinearConfig:
    """Linear solver settings (GUI-facing)."""

    method: str = "direct"
    tolerance: float = 1e-8
    max_iterations: int = 5000
    preconditioner: str = "jacobi"
    n_threads: int = 0
    verbose: bool = False

    def validate(self) -> list[str]:
        notes = []
        if self.method not in METHODS:
            notes.append(f"unknown method {self.method!r}, using 'direct'")
            self.method = "direct"
        if self.preconditioner not in PRECONDITIONERS:
            notes.append(f"unknown preconditioner {self.preconditioner!r}, using 'jacobi'")
            self.preconditioner = "jacobi"
        if self.tolerance <= 0:
            notes.append("tolerance must be > 0, using 1e-8")
            self.tolerance = 1e-8
        if self.max_iterations < 1:
            notes.append("max_iterations must be >= 1, using 5000")
            self.max_iterations = 5000
        return notes


@dataclass
class LinearResult:
    """Outcome of a linear solve."""

    T: np.ndarray
    converged: bool
    iterations: int
    residual: float
    method: str
    notes: list[str] = field(default_factory=list)


def set_num_threads(n_threads: int) -> int:
    """Set the BLAS/OMP thread budget; 0 = all cores, -1 = all but one.

    The environment variables only reach a library that has not been loaded yet; NumPy's
    BLAS is loaded by the time a solver runs, so the running pools are limited through
    ``threadpoolctl`` as well - without it the setting did nothing.
    """
    n_cpu = os.cpu_count() or 1
    if n_threads == 0:
        actual = n_cpu
    elif n_threads == -1:
        actual = max(1, n_cpu - 1)
    else:
        actual = max(1, min(int(n_threads), n_cpu))
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(actual)
    if threadpool_limits is not None:
        threadpool_limits(limits=actual)
    return actual


def is_symmetric(a: sparse.spmatrix, tolerance: float = 0.0) -> bool:
    """True if the matrix equals its transpose (structure and values)."""
    diff = (a - a.T).tocsr()
    if diff.nnz == 0:
        return True
    return float(np.abs(diff.data).max()) <= tolerance


def fingerprint(a: sparse.spmatrix) -> str:
    """Stable content hash of a sparse matrix (cache key, provenance)."""
    a = a.tocsr()
    h = hashlib.blake2b(digest_size=16)
    h.update(np.asarray(a.shape, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a.indices, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a.data, dtype=np.float64).tobytes())
    return h.hexdigest()


#: relative change of every entry under which an AMG hierarchy is reused as it is
NEAR_REUSE = 0.2


def _structure(a: sparse.csr_matrix) -> str:
    """Hash of the sparsity pattern alone."""
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(a.indptr, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(a.indices, dtype=np.int64).tobytes())
    return h.hexdigest()


class PreconditionerCache:
    """Keeps the expensive AMG hierarchy alive while the matrix is unchanged."""

    def __init__(self) -> None:
        self._key: str | None = None
        self._object = None
        #: the last matrix the hierarchy was built for: the same object needs no hash
        self._last = None
        #: the symmetrised operator of the last (matrix, scale) pair
        self._scaled = None

    def scaled(self, a, s: np.ndarray, source=None):
        """``diag(s) A diag(1/s)``, built once per (matrix, scale) pair and reused.

        ``source`` is the array ``s`` was computed from (the cell volumes): the memo is
        keyed on it and on the matrix object, both of which a solver keeps between calls.
        """
        memo = self._scaled
        source = s if source is None else source
        if memo is not None and memo[0]() is a and memo[1] is source:
            return memo[2]
        product = (sparse.diags(s) @ a @ sparse.diags(1.0 / s)).tocsr()
        try:
            self._scaled = (weakref.ref(a), source, product)
        except TypeError:                      # a matrix type without weak references
            self._scaled = None
        return product

    def get(self, a: sparse.csr_matrix, cfg: LinearConfig, notes: list[str]):
        if cfg.preconditioner == "none":
            return None
        if cfg.preconditioner == "jacobi":
            diag = a.diagonal().astype(float)
            diag[np.abs(diag) < 1e-300] = 1.0
            return sparse.diags(1.0 / diag).tocsr()
        if cfg.preconditioner == "ilu":
            return _ilu_operator(a, notes)
        if not HAS_PYAMG:
            notes.append("pyamg not installed: falling back to Jacobi preconditioner")
            return PreconditionerCache.get(self, a, _jacobi_cfg(cfg), notes)

        # the key is the *content* of the matrix: the symmetrised operator of a graded
        # mesh is a new object at every call, and keying on the object rebuilt the
        # hierarchy - the dominant cost - at every step of a transient
        if self._last is not None and self._last() is a and self._object is not None                 and self._key is not None and self._key.startswith(cfg.preconditioner + ":"):
            return self._object                # the very same operator: no hash needed
        key = f"{cfg.preconditioner}:{fingerprint(a)}"
        if key == self._key:
            notes.append("reusing cached AMG hierarchy")
            self._last = weakref.ref(a)
            return self._object
        if self._near(a, cfg):
            # the same sparsity and entries within NEAR_REUSE of the ones the hierarchy
            # was built for: a preconditioner only has to be *close* to the operator
            # (CG still converges to the tolerance asked), so a transient whose films
            # drift step by step keeps one hierarchy for several steps
            notes.append("reusing the AMG hierarchy of a nearby operator")
            self._last = weakref.ref(a)
            return self._object
        if cfg.preconditioner == "amg_sa":
            solver = pyamg.smoothed_aggregation_solver(a, max_coarse=500, max_levels=10)
        else:
            # a V(1,1) cycle - one forward Gauss-Seidel sweep down, one backward sweep up,
            # which keeps the preconditioner symmetric for CG - instead of pyamg's
            # symmetric sweeps: measured on the default tree (214 089 unknowns) 0.93 s of
            # setup and 0.65 s per solve against 1.36 s and 1.09 s, for 11 iterations
            solver = pyamg.ruge_stuben_solver(
                a, max_coarse=500, max_levels=10,
                presmoother=("gauss_seidel", {"sweep": "forward"}),
                postsmoother=("gauss_seidel", {"sweep": "backward"}))
        self._key, self._object = key, solver.aspreconditioner()
        self._last = weakref.ref(a)
        self._built = (cfg.preconditioner, a.shape, _structure(a), a.data.copy())
        return self._object

    def _near(self, a: sparse.csr_matrix, cfg: LinearConfig) -> bool:
        """Whether ``a`` has the built operator's pattern and entries within NEAR_REUSE."""
        built = getattr(self, "_built", None)
        if built is None or self._object is None or built[0] != cfg.preconditioner:
            return False
        if built[1] != a.shape or built[3].size != a.data.size:
            return False
        if built[2] != _structure(a):
            return False
        old = built[3]
        change = np.abs(a.data - old)
        scale = np.maximum(np.abs(old), 1e-300)
        return bool(np.all(change <= NEAR_REUSE * scale))


def _symmetry_tolerance(a: sparse.spmatrix) -> float:
    """Rounding-level tolerance for the symmetry check of a scaled operator.

    The per-volume coefficients of a graded mesh are `diag(V)^-1 K` with `K`
    symmetric, so the symmetrised operator is symmetric *up to rounding* of the row
    and column scaling: comparing with zero tolerance would reject it.  A relative
    threshold still catches a structurally asymmetric operator (which differs by
    orders of magnitude, not by an ulp).
    """
    if a.nnz == 0:
        return 0.0
    return float(np.abs(a.data).max()) * 1e-12


def _jacobi_cfg(cfg: LinearConfig) -> LinearConfig:
    out = LinearConfig(**{**cfg.__dict__, "preconditioner": "jacobi"})
    return out


def _ilu_operator(a: sparse.csr_matrix, notes: list[str]):
    try:
        ilu = splinalg.spilu(a.tocsc(), drop_tol=1e-5, fill_factor=10)
        return splinalg.LinearOperator(a.shape, ilu.solve)
    except Exception as exc:  # singular or badly scaled: keep going with Jacobi
        notes.append(f"ILU failed ({exc}); using Jacobi preconditioner")
        diag = a.diagonal().astype(float)
        diag[np.abs(diag) < 1e-300] = 1.0
        return sparse.diags(1.0 / diag).tocsr()


def solve_linear(a: sparse.csr_matrix, b: np.ndarray, cfg: LinearConfig = None,
                 x0: np.ndarray = None, cache: PreconditionerCache = None,
                 scale: np.ndarray = None) -> LinearResult:
    """Solve ``A x = b`` with the requested method.

    ``scale`` is an optional per-cell vector (the cell volumes): with per-volume
    coefficients ``A = diag(V)^-1 K`` the *operator* ``K`` is symmetric while ``A``
    is not, and the two are related by the similarity ``diag(V)^1/2``.  Passing
    ``scale`` makes the symmetric methods solve the symmetrised system and map the
    solution back, which is what keeps CG valid on a graded mesh.  A uniform mesh
    needs no scaling (the transformation is a constant there) and passes ``None``.
    """
    cfg = cfg or LinearConfig()
    notes = cfg.validate()
    b = np.asarray(b, dtype=float).ravel()
    x0_flat = None if x0 is None else np.asarray(x0, dtype=float).ravel()
    if cfg.method == "direct":
        return _solve_direct(a, b, notes)
    free, a_free, scale_free = _decoupled(a, cache, scale)
    if free is not None:
        # the rows a symmetric elimination pinned (the excluded air, a Dirichlet wall) are
        # an identity and nothing else: their value is known, and the iterative method
        # only sees the rows that are really coupled - a quarter fewer unknowns on the
        # default vessel, whose excluded air is pinned at the ambient
        a = a.tocsr()
        x = b / a.diagonal()
        start = None if x0_flat is None else x0_flat[free]
        sub = _solve_iterative(a_free, b[free], cfg, start, cache, notes, scale_free)
        x[free] = np.asarray(sub.T, dtype=float).ravel()
        return LinearResult(x, sub.converged, sub.iterations, _relative_residual(a, b, x),
                            sub.method, sub.notes)
    return _solve_iterative(a, b, cfg, x0_flat, cache, notes, scale)


def _decoupled(a, cache, scale):
    """``(free rows, A[free, free], scale[free])`` when rows decouple, else Nones.

    A row is decoupled when its only entry is its diagonal and its column holds nothing
    else either - what :func:`src.solver.matrix.apply_dirichlet` leaves of a pinned cell.
    Built once per matrix (and scale) object when a cache is given.
    """
    memo = getattr(cache, "_reduced", None) if cache is not None else None
    if (memo is not None and memo[0]() is a and memo[1] is scale):
        return memo[2], memo[3], memo[4]
    csr = a.tocsr()
    n = csr.shape[0]
    rows = np.diff(csr.indptr)
    columns = np.diff(csr.tocsc().indptr)
    diagonal = csr.diagonal()
    alone = (rows == 1) & (columns == 1) & (diagonal != 0.0)
    if int(np.count_nonzero(alone)) < max(1, int(0.02 * n)) or alone.all():
        result = (None, None, None)
    else:
        free = np.flatnonzero(~alone)
        a_free = csr[free][:, free].tocsr()
        scale_free = None if scale is None else np.asarray(scale, dtype=float).ravel()[free]
        result = (free, a_free, scale_free)
    if cache is not None:
        try:
            cache._reduced = (weakref.ref(a), scale, *result)
        except TypeError:
            cache._reduced = None
    return result


def _solve_direct(a, b, notes) -> LinearResult:
    try:
        x = splinalg.spsolve(a.tocsc(), b)
    except Exception as exc:
        notes.append(f"direct solve failed ({exc}); falling back to BiCGSTAB")
        cfg = LinearConfig(method="bicgstab")
        return _solve_iterative(a, b, cfg, None, None, notes)
    if not np.all(np.isfinite(x)):
        notes.append("direct solve returned non-finite values")
        return LinearResult(np.zeros_like(b), False, 0, np.inf, "direct", notes)
    return LinearResult(x, True, 0, _relative_residual(a, b, x), "direct (LU sparse)", notes)


def _solve_iterative(a, b, cfg: LinearConfig, x0, cache, notes, scale=None) -> LinearResult:
    method = cfg.method
    if scale is not None and method in ("cg", "gmres"):
        # solve the symmetrised system diag(s) A diag(1/s) y = diag(s) b, y = diag(s) x
        s = np.sqrt(np.asarray(scale, dtype=float).ravel())
        if not np.all(np.isfinite(s)) or np.any(s <= 0):
            notes.append("invalid volume scaling: solved without symmetrisation")
        else:
            a_orig, b_orig = a, b
            a = (cache.scaled(a, s, scale) if cache is not None
                 else sparse.diags(s) @ a @ sparse.diags(1.0 / s))
            b = s * b
            x0 = None if x0 is None else x0 * s
            result = _solve_iterative(a, b, cfg, x0, cache, notes, None)
            x = np.asarray(result.T, dtype=float).ravel() / s
            residual = _relative_residual(a_orig, b_orig, x)
            return LinearResult(x, result.converged, result.iterations, residual,
                                result.method, result.notes)
    if method == "cg" and not is_symmetric(a, tolerance=_symmetry_tolerance(a)):
        notes.append("cg requires a symmetric matrix (Dirichlet rows break symmetry): "
                     "switched to bicgstab")
        method = "bicgstab"
    m = (cache or PreconditionerCache()).get(a, cfg, notes)
    count = [0]

    def counter(*_args) -> None:
        count[0] += 1

    kwargs = dict(rtol=cfg.tolerance, maxiter=cfg.max_iterations, M=m, callback=counter)
    if x0 is not None:
        kwargs["x0"] = x0
    if method == "cg":
        x, info = splinalg.cg(a, b, **kwargs)
    elif method == "bicgstab":
        x, info = splinalg.bicgstab(a, b, **kwargs)
    else:
        x, info = splinalg.gmres(a, b, callback_type="pr_norm", **kwargs)
    residual = _relative_residual(a, b, x)
    converged = info == 0 and np.isfinite(residual) and residual <= max(cfg.tolerance * 20, 1e-6)
    if not converged and not np.isfinite(residual):
        x = np.zeros_like(b)
        notes.append(f"{method} diverged (info={info})")
    elif not converged:
        notes.append(f"{method} reached info={info} after {cfg.max_iterations} iterations "
                     f"(residual {residual:.2e})")
    return LinearResult(x, converged, int(count[0]), residual,
                        f"{method} + {cfg.preconditioner}", notes)


def _relative_residual(a, b, x) -> float:
    r = a @ x - b
    denom = np.linalg.norm(b)
    return float(np.linalg.norm(r) / denom) if denom > 0 else float(np.linalg.norm(r))
