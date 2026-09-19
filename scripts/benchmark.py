"""Benchmark of the current pipeline: assembly, steady solve, transient step.

    python scripts/benchmark.py [--max-cells 200000]

Timings are wall-clock on this machine; the point is to compare mesh sizes and
solver settings, not to produce absolute numbers.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.geometry import create_small_test_geometry  # noqa: E402
from src.core.grid import GridIndex  # noqa: E402
from src.core.mesh import Mesh3D  # noqa: E402
from src.solver.linear import LinearConfig, solve_linear  # noqa: E402
from src.solver.matrix import build_steady_matrix, build_transient_operators  # noqa: E402


def build_model(spacing: float) -> Mesh3D:
    mesh = Mesh3D(Lx=8.0, Ly=8.0, Lz=7.0, spacing=spacing)
    create_small_test_geometry().apply_to_mesh(mesh)
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-cells", type=int, default=200_000)
    args = parser.parse_args()

    print(f"{'spacing':>8} {'cells':>10} {'assemble':>10} {'direct':>10} "
          f"{'bicgstab+jacobi':>16} {'amg':>10}")
    for spacing in (0.4, 0.3, 0.25, 0.2, 0.15):
        mesh = build_model(spacing)
        if mesh.N_total > args.max_cells:
            continue
        index = GridIndex.from_mesh(mesh)
        t0 = time.perf_counter()
        matrix, rhs = build_steady_matrix(mesh, index=index)
        t_assemble = time.perf_counter() - t0

        t0 = time.perf_counter()
        direct = solve_linear(matrix, rhs, LinearConfig(method="direct"))
        t_direct = time.perf_counter() - t0

        t0 = time.perf_counter()
        iterative = solve_linear(matrix, rhs, LinearConfig(method="bicgstab",
                                                           preconditioner="jacobi",
                                                           tolerance=1e-8),
                                 x0=direct.T)
        t_iterative = time.perf_counter() - t0

        t0 = time.perf_counter()
        amg = solve_linear(matrix, rhs, LinearConfig(method="cg", preconditioner="amg_rs",
                                                     tolerance=1e-8), x0=direct.T)
        t_amg = time.perf_counter() - t0

        print(f"{mesh.d:8.3f} {mesh.N_total:10,d} {t_assemble:9.3f}s {t_direct:9.3f}s "
              f"{t_iterative:15.3f}s {t_amg:9.3f}s   "
              f"(residuals {direct.residual:.1e}/{iterative.residual:.1e}/{amg.residual:.1e})")

    mesh = build_model(0.3)
    t0 = time.perf_counter()
    build_transient_operators(mesh, 60.0)
    print(f"\ntransient operators ({mesh.N_total:,} cells): {time.perf_counter() - t0:.3f} s")
    _ = np
    return 0


if __name__ == "__main__":
    sys.exit(main())
