"""Heat-transfer pipes buried in the storage medium: geometry and rasterisation.

Polar Night Energy's architecture (their published description) is a closed air loop
through pipes buried in the sand: the electric resistors heat the *air*, the air heats
the bed through the pipe walls, and on discharge the same loop delivers the heat to an
exchanger.  The pipes are the only heat-transfer surface, so their **wetted area** is a
physical quantity and must not be read off the voxel mask: the mask is a staircase and
its surface is a mesh artefact.

This module therefore does two things:

* builds pipe layouts (vertical banks in a silo, rings, serpentines, arbitrary
  polylines);
* rasterises a polyline onto the grid returning, for every cell it crosses, the
  **length of pipe inside that cell** and the wetted area ``pi * d * length``.

The sum of the areas equals ``pi * d * total length`` to machine precision, which is
the invariant the fluid solve relies on: distributing the exchange over the cells must
not create or destroy surface.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from .mesh import Mesh3D


@dataclass
class PipeRun:
    """One pipe run: a polyline of diameter ``diameter`` and where it crosses the grid."""

    name: str
    points: np.ndarray                 # (n, 3) polyline vertices [m]
    diameter: float                    # [m]
    cells: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    length: np.ndarray = field(default_factory=lambda: np.empty(0))   # [m] per cell

    @property
    def perimeter(self) -> float:
        return float(np.pi * self.diameter)

    @property
    def area(self) -> np.ndarray:
        """Wetted area of the pipe inside each crossed cell [m^2]."""
        return self.perimeter * self.length

    @property
    def total_length(self) -> float:
        segments = np.diff(self.points, axis=0)
        return float(np.sum(np.linalg.norm(segments, axis=1)))

    @property
    def total_area(self) -> float:
        return self.perimeter * self.total_length


def _cell_of(mesh: Mesh3D, x: float, y: float, z: float) -> int:
    """Flat (Fortran-order) index of the cell containing a point, or -1 outside."""
    i = int(np.searchsorted(mesh.edges_x, x) - 1)
    j = int(np.searchsorted(mesh.edges_y, y) - 1)
    k = int(np.searchsorted(mesh.edges_z, z) - 1)
    if not (0 <= i < mesh.Nx and 0 <= j < mesh.Ny and 0 <= k < mesh.Nz):
        return -1
    return i + j * mesh.Nx + k * mesh.Nx * mesh.Ny


def rasterize_pipe(mesh: Mesh3D, points: Sequence[Sequence[float]], diameter: float,
                   name: str = "run", substeps: float = 4.0) -> PipeRun:
    """Split a polyline into the cells it crosses, with the length inside each cell.

    The polyline is sampled at ``substeps`` points per *local cell size* so that the
    sampling cannot skip a cell, and the length is accumulated by cell.  The area that
    comes out is the geometric ``pi d L``: no voxel surface is involved.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("a pipe run needs at least two 3-D points")
    run = PipeRun(name=name, points=pts, diameter=float(diameter))

    lengths: dict[int, float] = {}
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        span = float(np.linalg.norm(b - a))
        if span <= 0.0:
            continue
        cell_here = mesh.cell_size_at(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1]),
                                      0.5 * (a[2] + b[2]))
        n = max(int(substeps * span / max(cell_here, 1e-9)), 2)
        for t0, t1 in zip(np.linspace(0.0, 1.0, n + 1)[:-1],
                          np.linspace(0.0, 1.0, n + 1)[1:], strict=True):
            mid = a + 0.5 * (t0 + t1) * (b - a)
            index = _cell_of(mesh, float(mid[0]), float(mid[1]), float(mid[2]))
            if index >= 0:
                lengths[index] = lengths.get(index, 0.0) + (t1 - t0) * span

    order = sorted(lengths, key=lambda idx: float(mesh.Z.ravel(order="F")[idx]))
    run.cells = np.asarray(order, dtype=np.int64)
    run.length = np.asarray([lengths[idx] for idx in order], dtype=float)
    return run


def vertical_bank(mesh: Mesh3D, center: tuple[float, float], radius: float,
                  n_pipes: int, z_bottom: float, z_top: float, diameter: float,
                  phase_deg: float = 0.0, name: str = "bank") -> list[PipeRun]:
    """``n_pipes`` vertical pipes on a circle of ``radius``: the silo layout.

    Air is fed at the bottom and collected at the top, so every pipe is a separate run
    and the flow splits between them (the caller decides how much of it goes to each).
    """
    runs = []
    phase = np.deg2rad(phase_deg)
    for index in range(max(int(n_pipes), 1)):
        angle = phase + 2.0 * np.pi * index / max(int(n_pipes), 1)
        x = center[0] + radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        runs.append(rasterize_pipe(mesh, [(x, y, z_bottom), (x, y, z_top)], diameter,
                                   name=f"{name}_{index}"))
    return runs


def serpentine(mesh: Mesh3D, x_range: tuple[float, float], y_range: tuple[float, float],
               z: float, passes: int, diameter: float,
               name: str = "serpentine") -> list[PipeRun]:
    """One pipe zig-zagging in a horizontal plane (a single run, in series)."""
    x0, x1 = x_range
    y0, y1 = y_range
    points: list[tuple[float, float, float]] = []
    for index in range(max(int(passes), 1)):
        t = index / max(int(passes) - 1, 1)
        y = y0 + t * (y1 - y0)
        if index % 2 == 0:
            points.extend([(x0, y, z), (x1, y, z)])
        else:
            points.extend([(x1, y, z), (x0, y, z)])
    return [rasterize_pipe(mesh, points, diameter, name=name)]


def describe(runs: Iterable[PipeRun]) -> str:
    """One line per run: cells crossed, length, wetted area."""
    lines = []
    for run in runs:
        lines.append(f"{run.name:16s} {len(run.cells):6d} cells  "
                     f"L {run.total_length:7.2f} m  A {run.total_area:7.3f} m2  "
                     f"d {run.diameter * 1000:.1f} mm")
    return "\n".join(lines)
