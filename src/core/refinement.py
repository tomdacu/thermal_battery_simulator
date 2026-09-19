"""Mesh refinement: turn *physical* targets into cell sizes.

The user asks for cells of a given size in a region of the domain (the storage, the
insulation, the sheath of a heater); this module turns that request into the edges of
a graded grid.  The rules are:

* a **band** declares the size a slice of an axis needs;
* outside the bands the size follows a **linear ramp** ``h + (growth-1)*distance``, so
  the ratio between neighbouring cells never exceeds ``growth`` (a cell of size ``h``
  can grow by ``(growth-1)*h`` per step) — an exponential ramp would look shorter but
  its per-cell ratio grows with the cell size;
* the size field is a plain function of the coordinate, so a request and its mirror
  image give mirrored grids: symmetric targets, symmetric mesh;
* inside a band the cells are placed by **equidistributing the density** ``1/h``, so
  the local size follows the field, a uniform field gives an exactly uniform grid and
  the last cell of a band absorbs the remainder (never a sliver);
* band boundaries stay **exactly on grid lines**, so the geometry masks never split a
  cell, and the axis ends exactly on the domain;
* a **cell budget** only limits the request: if the targets do not fit, every target is
  scaled by one common factor until they do (the finest grid that fits is returned).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Band:
    """A slice of one axis with the cell size that region needs [m]."""

    start: float
    end: float
    target: float

    @property
    def length(self) -> float:
        return max(self.end - self.start, 0.0)


def partition(length: float, bands: Sequence[Band]) -> list[Band]:
    """Split overlapping bands into contiguous, non-overlapping ones.

    A refinement band may be declared *inside* a wider one (fine storage inside a
    coarse domain); the partition keeps every boundary and assigns the finest target
    to the overlap, which is what a user expects.
    """
    points = sorted({0.0, length}
                    | {b.start for b in bands if 0.0 < b.start < length}
                    | {b.end for b in bands if 0.0 < b.end < length})
    out: list[Band] = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        covering = [b.target for b in bands
                    if b.start <= start + 1e-12 and b.end >= end - 1e-12]
        if covering:
            out.append(Band(start, end, min(covering)))
        elif out:
            out.append(Band(start, end, out[-1].target))
        else:
            out.append(Band(start, end, bands[0].target))
    return out


def _size_field(bands: Sequence[Band], growth: float, scale: float = 1.0,
                min_size: float | None = None, max_size: float | None = None):
    """Lower envelope of the graded targets, as a function of the coordinate [m].

    ``h(x) = min_k clip(h_k + (growth-1) * dist(x, band_k), min_size, max_size)``.
    """
    ordered = sorted(bands, key=lambda b: b.start)
    coarsest = max(band.target * scale for band in ordered)
    slope = max(growth - 1.0, 0.0)

    def field(x):
        values = np.asarray(x, dtype=float)
        best = np.full(values.shape, coarsest)
        for band in ordered:
            h_band = max(band.target * scale, 1e-9)
            distance = np.where(values < band.start, band.start - values,
                                np.where(values > band.end, values - band.end, 0.0))
            best = np.minimum(best, h_band + slope * distance)
        if min_size:
            best = np.maximum(best, min_size)
        if max_size:
            best = np.minimum(best, max_size)
        return best

    return field


def _band_edges(start: float, end: float, field) -> np.ndarray:
    """Cell boundaries of one band, equi-distributing the density ``1/h``.

    ``n = round(∫ 1/h)`` cells are placed where the cumulative density crosses an
    integer, so the local size follows the field and a uniform field gives an exactly
    uniform grid.  The last cell absorbs the remainder.
    """
    width = end - start
    if width <= 0:
        return np.empty(0)

    # the density must be resolved at the *finest* size of the band, otherwise the
    # ramp is sampled too coarsely and the realised cells jump
    probe = np.linspace(start, end, 257)
    fine = float(np.min(field(probe)))
    samples = int(np.clip(4.0 * width / max(fine, 1e-9), 64, 200_000))
    x = np.linspace(start, end, samples)
    h = field(x)
    if float(h.max() - h.min()) <= 1e-12 * float(h.mean()):      # constant: exact
        n = max(int(round(width / float(h[0]))), 1)
        return np.linspace(start, end, n + 1)
    density = 1.0 / h
    cumulative = np.concatenate(([0.0], np.cumsum(0.5 * (density[1:] + density[:-1])
                                                  * np.diff(x))))
    total = cumulative[-1]
    n = max(int(round(total)), 1)
    targets = np.arange(1, n) * (total / n)
    return np.concatenate(([start], np.interp(targets, cumulative, x), [end]))


def graded_edges(length: float, bands: Sequence[Band], growth: float = 1.3,
                 max_cells: int = 200_000, min_size: float | None = None,
                 max_size: float | None = None) -> np.ndarray:
    """Edges of a graded 1-D grid covering ``[0, length]``.

    The axis is split into the partitioned bands, every band gets the number of cells
    its own *density* integral asks for and the cells inside it follow the size field.
    Band boundaries therefore stay exactly on grid lines, the realised neighbour ratio
    is bounded by ``growth`` by construction of the field, and the result is symmetric
    whenever the request is.
    """
    if length <= 0:
        raise ValueError(f"length must be > 0, got {length}")
    declared = [band for band in bands if band.length > 0 and band.target > 0]
    if not declared:
        raise ValueError("at least one refinement band is required")
    if min_size and max_size and min_size > max_size:
        raise ValueError(f"min_size {min_size:g} m is larger than max_size {max_size:g} m")
    ordered = partition(length, declared)

    scale = 1.0
    for _ in range(60):
        field = _size_field(ordered, max(growth, 1.0), scale, min_size, max_size)
        edges = [0.0]
        for band in ordered:
            piece = _band_edges(band.start, band.end, field)
            edges.extend(piece[1:])
        grid = np.asarray(edges, dtype=float)
        if grid.size - 1 <= max_cells:
            return grid
        scale *= (grid.size - 1) / max_cells * 1.05
    raise RuntimeError("refinement failed to fit the cell budget")


def uniform_edges(length: float, size: float) -> np.ndarray:
    """Uniform grid with the requested cell size (legacy spacing API)."""
    if size <= 0:
        raise ValueError(f"cell size must be > 0, got {size}")
    n = max(1, int(round(length / size)))
    return np.linspace(0.0, length, n + 1)


def edges_to_centers(edges: np.ndarray) -> np.ndarray:
    return 0.5 * (edges[:-1] + edges[1:])


def edges_to_sizes(edges: np.ndarray) -> np.ndarray:
    return np.diff(edges)


def worst_ratio(edges: np.ndarray) -> float:
    """Largest neighbour size ratio (1.0 for a uniform grid)."""
    sizes = edges_to_sizes(edges)
    if sizes.size < 2:
        return 1.0
    step = sizes[1:] / np.maximum(sizes[:-1], 1e-12)
    return float(max(step.max(), 1.0 / max(step.min(), 1e-12)))


def describe(edges: np.ndarray) -> dict:
    """Summary for the GUI and the log."""
    sizes = edges_to_sizes(edges)
    return {"cells": int(sizes.size), "min_size": float(sizes.min()),
            "max_size": float(sizes.max()), "worst_ratio": worst_ratio(edges)}


def size_at(edges: np.ndarray, position: float) -> float:
    """Cell size at a coordinate (used to size element masks and probes)."""
    index = int(np.clip(np.searchsorted(edges, position) - 1, 0, len(edges) - 2))
    return float(edges[index + 1] - edges[index])


def bands_from_triples(items: Iterable[tuple[float, float, float]]) -> list[Band]:
    """Build bands from ``(start, end, target)`` triples, skipping empty ones."""
    return [Band(float(s), float(e), float(t)) for s, e, t in items
            if e - s > 0 and t > 0]


@dataclass(frozen=True)
class GridSpec:
    """Refinement targets of the three axes plus the shared knobs.

    An axis is described by *bands* in domain coordinates; the coarsest band acts as
    the far-field cap, so a background band over the whole axis is the usual way to
    say "outside the refined regions use this size".
    """

    x: tuple[Band, ...] = ()
    y: tuple[Band, ...] = ()
    z: tuple[Band, ...] = ()
    growth: float = 1.3
    #: total cell budget: it only limits the request (see :meth:`edges`)
    max_cells: int = 1_500_000
    #: hard limits on the realised cell size [m] (None = only the bands decide)
    min_size: float | None = None
    max_size: float | None = None

    # ------------------------------------------------------------- properties
    def targets(self) -> tuple[float, float]:
        """(finest, coarsest) declared target size [m]."""
        values = [band.target for axis in (self.x, self.y, self.z) for band in axis
                  if band.target > 0]
        if not values:
            raise ValueError("a grid spec needs at least one refinement band")
        return min(values), max(values)

    def scaled(self, factor: float, min_size: float | None = None,
               max_size: float | None = None,
               max_cells: int | None = None) -> GridSpec:
        """The same request with every target multiplied by ``factor``.

        Used by the automatic mesh search: level *k* is ``scaled(refine ** k)``, so the
        grid keeps its *shape* and only gets finer.
        """
        def bands(values):
            return tuple(Band(b.start, b.end, b.target * factor) for b in values)

        return GridSpec(x=bands(self.x), y=bands(self.y), z=bands(self.z),
                        growth=self.growth,
                        max_cells=self.max_cells if max_cells is None else max_cells,
                        min_size=self.min_size if min_size is None else min_size,
                        max_size=self.max_size if max_size is None else max_size)

    def with_limits(self, min_size: float | None = None,
                    max_size: float | None = None) -> GridSpec:
        """The same request with explicit size limits (GUI convenience)."""
        return GridSpec(x=self.x, y=self.y, z=self.z, growth=self.growth,
                        max_cells=self.max_cells,
                        min_size=min_size if min_size is not None else self.min_size,
                        max_size=max_size if max_size is not None else self.max_size)

    def describe(self, edges: tuple[np.ndarray, np.ndarray, np.ndarray]) -> dict:
        """Summary of the realised grid, for the GUI and the log."""
        nx, ny, nz = (e.size - 1 for e in edges)
        sizes = [edges_to_sizes(e) for e in edges]
        return {
            "cells_axis": (nx, ny, nz),
            "cells": nx * ny * nz,
            "min_size": float(min(s.min() for s in sizes)),
            "max_size": float(max(s.max() for s in sizes)),
            "worst_ratio": max(worst_ratio(e) for e in edges),
            "memory_MB": nx * ny * nz * 66 / 1e6,
        }

    # ------------------------------------------------------------------ build
    def edges(self, Lx: float, Ly: float, Lz: float) -> tuple[np.ndarray, np.ndarray,
                                                             np.ndarray]:
        """Graded edges of the three axes, respecting the cell budget."""
        fallback = max(max((b.target for b in axis), default=0.0)
                       for axis in (self.x, self.y, self.z))
        budget = max(int(self.max_cells), 1)

        def build(scale: float):
            out = []
            for bands, length in ((self.x, Lx), (self.y, Ly), (self.z, Lz)):
                declared = tuple(bands) or (Band(0.0, length, fallback),)
                scaled = [Band(b.start, b.end, b.target * scale) for b in declared]
                out.append(graded_edges(length, scaled, self.growth,
                                        max_cells=1_000_000_000,
                                        min_size=self.min_size,
                                        max_size=self.max_size))
            cells = 1
            for edge in out:
                cells *= edge.size - 1
            return tuple(out), cells

        # the analytic count (band length / target) decides whether the budget is an
        # issue at all: it ignores the ramp, so it is a lower bound of the real count
        estimate = 1.0
        for bands, length in ((self.x, Lx), (self.y, Ly), (self.z, Lz)):
            declared = tuple(bands) or (Band(0.0, length, fallback),)
            estimate *= max(sum(max(b.length, 0.0) / max(b.target, 1e-9)
                                for b in declared), 1.0)
        if estimate <= budget:
            grid, cells = build(1.0)
            if cells <= budget:                        # the request defines the grid
                return grid
        scale = (estimate / budget) ** (1.0 / 3.0) if estimate > budget else 1.0
        best: tuple | None = None
        best_cells = -1
        for _ in range(6):                             # coarsen until it fits
            grid, cells = build(scale)
            if cells <= budget:
                best, best_cells = grid, cells
                break
            scale *= max((cells / budget) ** (1.0 / 3.0), 1.1) * 1.05
        if best is None:
            raise RuntimeError("refinement failed to fit the cell budget")
        # the correction above is conservative (the ramp adds cells), so walk back in:
        # only here, where the request did not fit, is the budget there to be used
        for _ in range(4):
            scale *= 0.85
            grid, cells = build(scale)
            if cells > budget or cells <= best_cells:
                break
            best, best_cells = grid, cells
        return best
