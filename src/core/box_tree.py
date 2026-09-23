"""An anisotropic tree of boxes: leaves split in plan and in height independently.

Why not the cubic octree: the vessel is a vertical cylinder with vertical tubes.  Across
a riser, a wall or the insulation the temperature changes over centimetres; along them
it changes over metres.  A cube has to be as short as it is narrow, so a leaf that is
fine where the plan needs it is fine in height too, and most of those leaves carry no
information.  Here every leaf has two levels:

* ``lxy`` - ``log2`` of its edge in plan (x and y together, a square), in finest plan
  cells of ``dx`` [m];
* ``lz``  - ``log2`` of its height, in finest vertical cells of ``dz`` [m].

A leaf splits into four (plan), two (height) or eight (both) children, so a column can
be refined in height where the headers, the slabs and the roof are and stay tall along
the bed, and the ratio height/width of a leaf is free (``2**lz dz / 2**lxy dx``).  This
is a restricted anisotropic octree, the arrangement of the "2.5-D" meshes of the
reservoir and ground-heat literature (a quadtree in plan, layered in the vertical;
Aziz and Settari, *Petroleum Reservoir Simulation*, 1979, ch. 4; Heidemann et al. for
borehole fields) with the layers allowed to change from column to column.

**Balance.**  Two face neighbours differ by at most one level in plan *and* one level in
height (the 2:1 rule of p4est, Burstedde et al., SIAM J. Sci. Comput. 33, 2011, applied
per direction).  It keeps the graded stencil local and guarantees that the face of a
leaf is tiled by neighbours whose tangential extents are at least half its own, which is
what makes the four-probe face search below complete.

**Faces.**  A leaf's positive face is probed just outside it at the four points
``corner + {0, half} x {0, half}`` of its tangential extents; every neighbour tiling the
face contains at least one probe, so the unique hits are the face neighbours.  The
entry of a pair carries the *overlap* area (the intersection of the two faces) and the
centre distance along the normal, ``(e_i + e_j) / 2``: each interface appears once, with
one conductance, and the discrete balance closes to round-off on any combination of
levels - the same conservative two-point flux as :mod:`src.core.octree`.

**Arrays only.**  The leaves are five integer arrays - corner ``x, y, z`` in finest
cells, ``lxy`` and ``lz`` - and every query (locate, balance, faces) is a handful of
``searchsorted`` calls over per-level sorted codes, the linear-octree idea of p4est
(Morton-style integer codes, no pointers, no per-leaf Python objects).  A tree of a few
hundred thousand leaves is built and balanced in seconds.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: bits per coordinate in a location code: 3 * 21 = 63
BITS = 21


@dataclass
class _Tables:
    """Per (lxy, lz) pair: sorted level-normalised codes and the leaf positions."""

    pairs: list[tuple[int, int]]
    codes: list[np.ndarray]
    positions: list[np.ndarray]


def _code(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    return x | (y << BITS) | (z << (2 * BITS))


class BoxTree:
    """Leaves of a box ``n_xy dx`` x ``n_xy dx`` x ``n_z dz``, 2:1 balanced per direction."""

    def __init__(self, n_xy: int, n_z: int, dx: float, dz: float,
                 corners: np.ndarray | None = None, lxy: np.ndarray | None = None,
                 lz: np.ndarray | None = None) -> None:
        for name, n in (("n_xy", n_xy), ("n_z", n_z)):
            if n < 1 or (n & (n - 1)):
                raise ValueError(f"{name} must be a power of two, got {n}")
        if dx <= 0.0 or dz <= 0.0:
            raise ValueError(f"the finest cell must be > 0, got {dx} x {dz}")
        self.n_xy, self.n_z = int(n_xy), int(n_z)
        self.dx, self.dz = float(dx), float(dz)
        self.max_lxy = int(np.log2(self.n_xy))
        self.max_lz = int(np.log2(self.n_z))
        self.version = 0
        if corners is None:
            corners = np.zeros((1, 3), dtype=np.int64)
            lxy = np.array([self.max_lxy], dtype=np.int64)
            lz = np.array([self.max_lz], dtype=np.int64)
        self._set(np.asarray(corners, dtype=np.int64).reshape(-1, 3),
                  np.asarray(lxy, dtype=np.int64), np.asarray(lz, dtype=np.int64))

    # ---------------------------------------------------------------- building
    @classmethod
    def uniform(cls, n_xy: int, n_z: int, dx: float, dz: float, lxy: int,
                lz: int) -> BoxTree:
        """Every leaf ``2**lxy`` finest cells wide and ``2**lz`` tall."""
        sx, sz = 1 << lxy, 1 << lz
        xs = np.arange(0, n_xy, sx, dtype=np.int64)
        zs = np.arange(0, n_z, sz, dtype=np.int64)
        gx, gy, gz = np.meshgrid(xs, xs, zs, indexing="ij")
        corners = np.column_stack((gx.ravel(), gy.ravel(), gz.ravel()))
        count = corners.shape[0]
        return cls(n_xy, n_z, dx, dz, corners, np.full(count, lxy, dtype=np.int64),
                   np.full(count, lz, dtype=np.int64))

    def _set(self, corners: np.ndarray, lxy: np.ndarray, lz: np.ndarray,
             order: np.ndarray | None = None) -> np.ndarray:
        """Adopt new leaves in the canonical order; returns the permutation applied.

        The order is by height first, then y, then x: the leaves of a layer sit next to
        each other, which keeps the bandwidth of the operator small.
        """
        if not (np.all(lxy >= 0) and np.all(lxy <= self.max_lxy)
                and np.all(lz >= 0) and np.all(lz <= self.max_lz)):
            raise ValueError("a leaf level does not fit the box")
        perm = np.lexsort((corners[:, 0], corners[:, 1], corners[:, 2]))
        self.corners = corners[perm]
        self.lxy = lxy[perm]
        self.lz = lz[perm]
        self.version += 1
        self._tables: _Tables | None = None
        self._faces: tuple[np.ndarray, ...] | None = None
        return perm

    # ---------------------------------------------------------------- geometry
    @property
    def n_cells(self) -> int:
        return int(self.corners.shape[0])

    @property
    def box(self) -> tuple[float, float, float]:
        """``(Lx, Ly, Lz)`` of the box [m]."""
        side = self.n_xy * self.dx
        return (side, side, self.n_z * self.dz)

    def unit_extents(self) -> np.ndarray:
        """``(n, 3)`` leaf edges in finest cells of their own axis."""
        sxy = np.left_shift(np.int64(1), self.lxy)
        return np.column_stack((sxy, sxy, np.left_shift(np.int64(1), self.lz)))

    def extents(self) -> np.ndarray:
        """``(n, 3)`` leaf edges [m]."""
        return self.unit_extents() * np.array([self.dx, self.dx, self.dz])

    def lower(self) -> np.ndarray:
        """``(n, 3)`` lower corners [m]."""
        return self.corners * np.array([self.dx, self.dx, self.dz])

    def centres(self) -> np.ndarray:
        """``(n, 3)`` leaf centres [m]."""
        return self.lower() + 0.5 * self.extents()

    def volumes(self) -> np.ndarray:
        return np.prod(self.extents(), axis=1)

    def on_box_face(self) -> np.ndarray:
        """Whether each leaf touches one of the six faces of the box."""
        upper = self.corners + self.unit_extents()
        limit = np.array([self.n_xy, self.n_xy, self.n_z])
        return np.any((self.corners == 0) | (upper == limit), axis=1)

    def level_pairs(self) -> dict[tuple[int, int], int]:
        """Leaves per ``(lxy, lz)`` pair."""
        pairs, counts = np.unique(np.column_stack((self.lxy, self.lz)), axis=0,
                                  return_counts=True)
        return {(int(a), int(b)): int(c) for (a, b), c in zip(pairs, counts, strict=True)}

    # ---------------------------------------------------------------- location
    def _lookup(self) -> _Tables:
        if self._tables is None:
            pairs, codes, positions = [], [], []
            key = self.lxy * (self.max_lz + 1) + self.lz
            order = np.argsort(key, kind="stable")
            bounds = np.flatnonzero(np.diff(key[order])) + 1
            for group in np.split(order, bounds):
                if group.size == 0:
                    continue
                a, b = int(self.lxy[group[0]]), int(self.lz[group[0]])
                c = self.corners[group]
                code = _code(c[:, 0] >> a, c[:, 1] >> a, c[:, 2] >> b)
                sort = np.argsort(code, kind="stable")
                pairs.append((a, b))
                codes.append(code[sort])
                positions.append(group[sort])
            self._tables = _Tables(pairs, codes, positions)
        return self._tables

    def locate_many(self, points: np.ndarray) -> np.ndarray:
        """Leaf holding each point given in finest cells (``(m, 3)`` ints), -1 outside."""
        points = np.asarray(points, dtype=np.int64).reshape(-1, 3)
        out = np.full(points.shape[0], -1, dtype=np.int64)
        inside = np.all((points >= 0) & (points < np.array([self.n_xy, self.n_xy,
                                                            self.n_z])), axis=1)
        tables = self._lookup()
        for (a, b), codes, positions in zip(tables.pairs, tables.codes, tables.positions,
                                            strict=True):
            pending = np.flatnonzero(inside & (out < 0))
            if pending.size == 0:
                break
            p = points[pending]
            wanted = _code(p[:, 0] >> a, p[:, 1] >> a, p[:, 2] >> b)
            where = np.minimum(np.searchsorted(codes, wanted), codes.size - 1)
            hit = codes[where] == wanted
            out[pending[hit]] = positions[where[hit]]
        return out

    def locate_points(self, points: np.ndarray, below: bool = False,
                      clamp: bool = False) -> np.ndarray:
        """Leaf holding each physical point [m] (``(m, 3)``), -1 outside the box.

        ``below`` is the rasteriser's convention (a point on a face belongs to the leaf
        below it: ``ceil(v / d) - 1``); otherwise a leaf owns its lower face
        (``floor(v / d)``).  ``clamp`` moves a point outside the box onto its boundary.
        """
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        size = np.array([self.dx, self.dx, self.dz])
        scaled = points / size
        cells = (np.ceil(scaled) - 1 if below else np.floor(scaled)).astype(np.int64)
        if clamp:
            cells = np.clip(cells, 0, np.array([self.n_xy, self.n_xy, self.n_z]) - 1)
        return self.locate_many(cells)

    # -------------------------------------------------------------- refinement
    def split(self, split_xy: np.ndarray, split_z: np.ndarray) -> np.ndarray:
        """Split the marked leaves; returns, per new leaf, the index of its old leaf.

        A leaf marked in plan becomes four children (x and y halved), in height two, in
        both eight.  A leaf already at the finest level of a direction is not split in
        that direction.  The tree is *not* balanced here (see :meth:`refine`).
        """
        split_xy = np.asarray(split_xy, dtype=bool) & (self.lxy > 0)
        split_z = np.asarray(split_z, dtype=bool) & (self.lz > 0)
        corners, lxy, lz = self.corners, self.lxy, self.lz
        pieces_c, pieces_xy, pieces_z, parents = [], [], [], []
        keep = ~(split_xy | split_z)
        index = np.arange(self.n_cells)
        pieces_c.append(corners[keep])
        pieces_xy.append(lxy[keep])
        pieces_z.append(lz[keep])
        parents.append(index[keep])
        for plan, height in ((True, False), (False, True), (True, True)):
            rows = np.flatnonzero((split_xy == plan) & (split_z == height))
            if rows.size == 0:
                continue
            hxy = (np.left_shift(np.int64(1), lxy[rows] - 1) if plan
                   else np.zeros(rows.size, dtype=np.int64))
            hz = (np.left_shift(np.int64(1), lz[rows] - 1) if height
                  else np.zeros(rows.size, dtype=np.int64))
            offsets = [(ox, oy, oz) for ox in ((0, 1) if plan else (0,))
                       for oy in ((0, 1) if plan else (0,))
                       for oz in ((0, 1) if height else (0,))]
            for ox, oy, oz in offsets:
                child = corners[rows].copy()
                child[:, 0] += ox * hxy
                child[:, 1] += oy * hxy
                child[:, 2] += oz * hz
                pieces_c.append(child)
                pieces_xy.append(lxy[rows] - (1 if plan else 0))
                pieces_z.append(lz[rows] - (1 if height else 0))
                parents.append(rows)
        new_parent = np.concatenate(parents)
        perm = self._set(np.concatenate(pieces_c), np.concatenate(pieces_xy),
                         np.concatenate(pieces_z))
        return new_parent[perm]

    def refine(self, split_xy: np.ndarray, split_z: np.ndarray) -> np.ndarray:
        """Split the marked leaves and restore the balance; returns the parent map.

        The map gives, for every leaf of the new tree, the index of the leaf of the old
        tree that contains it - which is how a mesh carries its fields over a refinement.
        """
        before = self.n_cells
        parent = self.split(split_xy, split_z)
        fresh = np.zeros(self.n_cells, dtype=bool)
        # the children are the leaves whose parent was split: they are the seeds of the
        # balance (only next to them can the 2:1 rule be broken)
        counts = np.bincount(parent, minlength=before)
        fresh[counts[parent] > 1] = True
        return parent[self.balance_from(fresh)]

    def balance_from(self, seeds: np.ndarray) -> np.ndarray:
        """Restore the per-direction 2:1 rule around the ``seeds`` mask.

        A split only makes leaves finer, so a violation always has a new leaf on its fine
        side: the faces of the seeds are probed (four points per face, covering it) and
        a neighbour more than one level coarser - in plan or in height - is split in that
        direction.  Its children are the next seeds.  Returns the map from the balanced
        tree to the tree the call started with.
        """
        mapping = np.arange(self.n_cells)
        frontier = np.asarray(seeds, dtype=bool).copy()
        while frontier.any():
            rows = np.flatnonzero(frontier)
            mark_xy = np.zeros(self.n_cells, dtype=bool)
            mark_z = np.zeros(self.n_cells, dtype=bool)
            for i, j in self._probe_faces(rows, both_sides=True):
                too_wide = self.lxy[j] > self.lxy[i] + 1
                too_tall = self.lz[j] > self.lz[i] + 1
                mark_xy[j[too_wide]] = True
                mark_z[j[too_tall]] = True
            if not (mark_xy.any() or mark_z.any()):
                break
            before = self.n_cells
            parent = self.split(mark_xy, mark_z)
            counts = np.bincount(parent, minlength=before)
            frontier = counts[parent] > 1
            mapping = mapping[parent]
        return mapping

    def _probe_faces(self, rows: np.ndarray, both_sides: bool):
        """Yield ``(i, j)`` pairs: leaf ``rows`` and the leaves across its faces.

        Each face is probed one finest cell outside the leaf, at the four points
        ``corner + {0, half} x {0, half}`` of its tangential extents (one point where the
        extent is a single finest cell).  ``both_sides`` probes the negative faces too.
        """
        corners = self.corners[rows]
        ext = self.unit_extents()[rows]
        limit = np.array([self.n_xy, self.n_xy, self.n_z])
        signs = (1, -1) if both_sides else (1,)
        for axis in range(3):
            t1, t2 = [a for a in range(3) if a != axis]
            half1, half2 = ext[:, t1] // 2, ext[:, t2] // 2
            for sign in signs:
                plane = corners[:, axis] + ext[:, axis] if sign > 0 else corners[:, axis] - 1
                valid = (plane >= 0) & (plane < limit[axis])
                if not valid.any():
                    continue
                found_rows, found = [], []
                for o1 in (0, 1):
                    for o2 in (0, 1):
                        point = corners.copy()
                        point[:, axis] = plane
                        point[:, t1] += o1 * half1
                        point[:, t2] += o2 * half2
                        use = valid & ((o1 == 0) | (half1 > 0)) & ((o2 == 0) | (half2 > 0))
                        hit = np.full(rows.size, -1, dtype=np.int64)
                        hit[use] = self.locate_many(point[use])
                        ok = hit >= 0
                        found_rows.append(rows[ok])
                        found.append(hit[ok])
                i = np.concatenate(found_rows)
                j = np.concatenate(found)
                if i.size:
                    pair = np.unique(np.column_stack((i, j)), axis=0)
                    yield pair[:, 0], pair[:, 1]

    # ------------------------------------------------------------------- faces
    def face_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                   np.ndarray]:
        """``(i, j, axis, area [m^2], distance [m])``: every interface once.

        Built from the positive faces of every leaf, so a pair is found from the side
        whose positive face it is and never twice; sorted by ``(i, axis, j)``.  The
        area is the overlap of the two faces and the distance the centre-to-centre
        distance along the normal.
        """
        if self._faces is None:
            rows = np.arange(self.n_cells)
            ext = self.unit_extents()
            lower = self.corners
            size = np.array([self.dx, self.dx, self.dz])
            parts = []
            for axis in range(3):
                t1, t2 = [a for a in range(3) if a != axis]
                plane = lower[:, axis] + ext[:, axis]
                valid = plane < (self.n_z if axis == 2 else self.n_xy)
                sub = rows[valid]
                pairs = []
                for o1 in (0, 1):
                    for o2 in (0, 1):
                        half1, half2 = ext[sub, t1] // 2, ext[sub, t2] // 2
                        use = ((o1 == 0) | (half1 > 0)) & ((o2 == 0) | (half2 > 0))
                        point = lower[sub].copy()
                        point[:, axis] = plane[sub]
                        point[:, t1] += o1 * half1
                        point[:, t2] += o2 * half2
                        hit = np.full(sub.size, -1, dtype=np.int64)
                        hit[use] = self.locate_many(point[use])
                        ok = hit >= 0
                        pairs.append(np.column_stack((sub[ok], hit[ok])))
                pair = np.unique(np.concatenate(pairs), axis=0)
                if pair.size == 0:
                    continue
                i, j = pair[:, 0], pair[:, 1]
                overlap = np.ones(i.size)
                for t in (t1, t2):
                    low = np.maximum(lower[i, t], lower[j, t])
                    high = np.minimum(lower[i, t] + ext[i, t], lower[j, t] + ext[j, t])
                    overlap *= (high - low) * size[t]
                distance = 0.5 * (ext[i, axis] + ext[j, axis]) * size[axis]
                parts.append((i, j, np.full(i.size, axis, dtype=np.int64), overlap,
                              distance))
            if parts:
                i, j, axis, area, distance = (np.concatenate(c) for c in zip(*parts,
                                                                            strict=True))
                order = np.lexsort((j, axis, i))
                self._faces = (i[order], j[order], axis[order], area[order],
                               distance[order])
            else:
                empty = np.empty(0, dtype=np.int64)
                self._faces = (empty, empty, empty, np.empty(0), np.empty(0))
        return self._faces

    def summary(self) -> str:
        pairs = ", ".join(f"{(1 << a) * self.dx * 1000:.0f}x{(1 << b) * self.dz * 1000:.0f}"
                          f" mm: {count}" for (a, b), count in self.level_pairs().items())
        return f"box tree {self.n_xy}^2 x {self.n_z}: {self.n_cells} leaves ({pairs})"
