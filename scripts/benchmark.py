"""Benchmark of the current pipeline: the tree's assembly, steady solve, transient step.

    python scripts/benchmark.py [--max-cells 200000] [--scales 2.0 1.4 1.0 0.7 0.5]

Timings are wall-clock on this machine; the point is to compare refinement levels and
solver settings, not to produce absolute numbers.  The sweep is over the *refinement* of
the a priori request - one row per scale of the bands the physics plan asks for, which is
the level a search asks a tree for - because a tree is chosen by how far it is refined and
not by a cell spacing: each row is the small test geometry on the tree that request
produces, and the budget both sets the resolution floor and stops the refinement.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.convergence import AdaptivePlan                     # noqa: E402
from src.analysis.mesh_plan import refinement_bands, tree_resolution  # noqa: E402
from src.core.adaptive_mesh import AdaptiveMesh                       # noqa: E402
from src.core.geometry import create_small_test_geometry              # noqa: E402
from src.core.octree import Octree                                    # noqa: E402
from src.core.refinement import Band, GridSpec                        # noqa: E402
from src.solver.linear import LinearConfig, solve_linear              # noqa: E402

#: the octree spans a cube: the largest extent of the small geometry's 8 x 8 x 7 m domain
BOX = 8.0
#: the a priori request, in the vocabulary the GUI's Mesh tab builds it in: a target size
#: per region - the two insulation slabs, the storage core and the air around the battery
CORE_SIZE = 0.2
SLAB_SIZE = 0.1
FAR_SIZE = 0.4


def spec() -> GridSpec:
    """Refinement request of the small test geometry: three bands of z, one far field."""
    return GridSpec(x=(Band(0.0, BOX, FAR_SIZE),),
                    y=(Band(0.0, BOX, FAR_SIZE),),
                    z=(Band(0.0, 1.4, SLAB_SIZE), Band(1.4, 5.4, CORE_SIZE),
                       Band(5.4, 7.0, FAR_SIZE)))


def build_model(scale: float, budget: int) -> AdaptiveMesh:
    """The small geometry on a tree refined to ``scale`` times the a priori request.

    ``scale`` is the search's own knob (:meth:`AdaptivePlan.scaled`): the bands are the
    physical request, so shrinking them is what makes one level finer than the last.  The
    budget is the floor under every leaf and the stop of the refinement, so a row costs
    what it says it costs.
    """
    n_finest, physical_size = tree_resolution((BOX, BOX, BOX), budget)
    plan = AdaptivePlan(n_finest=n_finest, physical_size=physical_size,
                        bands=refinement_bands(spec(), (BOX, BOX, BOX))).scaled(scale)
    mesh = AdaptiveMesh.uniform(n_finest, physical_size, Octree(n_finest).max_level)
    mesh.refine_bands(plan.bands, max_cells=budget)
    create_small_test_geometry().apply_to_mesh(mesh)
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cells", type=int, default=60_000,
                        help="leaves the sweep may build: the floor and the stop")
    parser.add_argument("--scales", type=float, nargs="*",
                        default=(4.0, 3.0, 2.0, 1.4, 1.0),
                        help="band scales of the sweep, coarse to fine")
    args = parser.parse_args()

    print(f"{'scale':>8} {'leaves':>10} {'levels':>12} {'assemble':>10} {'direct':>10} "
          f"{'bicgstab+jacobi':>16} {'amg':>10}")
    last: AdaptiveMesh | None = None
    for scale in args.scales:
        mesh = build_model(scale, args.max_cells)
        if mesh.n_cells > args.max_cells:
            print(f"{scale:8.2f} {mesh.n_cells:10,d}   skipped: over the "
                  f"{args.max_cells:,} leaf budget")
            continue

        t0 = time.perf_counter()
        matrix, rhs = mesh.assemble()
        t_assemble = time.perf_counter() - t0
        scale_volume = mesh.volume_scale()

        t0 = time.perf_counter()
        direct = solve_linear(matrix, rhs, LinearConfig(method="direct"))
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        iterative = solve_linear(matrix, rhs, LinearConfig(method="bicgstab",
                                                           preconditioner="jacobi",
                                                           tolerance=1e-8),
                                 x0=direct.T, scale=scale_volume)
        t_iterative = time.perf_counter() - t0

        t0 = time.perf_counter()
        amg = solve_linear(matrix, rhs, LinearConfig(method="cg", preconditioner="amg_rs",
                                                     tolerance=1e-8),
                           x0=direct.T, scale=scale_volume)
        t_amg = time.perf_counter() - t0

        levels = "/".join(str(count) for _level, count in
                          sorted(mesh.level_histogram().items()))
        print(f"{scale:8.2f} {mesh.n_cells:10,d} {levels:>12} {t_assemble:9.3f}s "
              f"{t_direct:9.3f}s {t_iterative:15.3f}s {t_amg:9.3f}s   "
              f"(residuals {direct.residual:.1e}/{iterative.residual:.1e}/"
              f"{amg.residual:.1e})")
        last = mesh

    if last is not None:
        t0 = time.perf_counter()
        last.transient_operators(60.0)
        print(f"\ntransient operators ({last.n_cells:,} leaves): "
              f"{time.perf_counter() - t0:.3f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
