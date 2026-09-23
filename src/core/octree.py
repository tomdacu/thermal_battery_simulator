"""A cell-level octree with the 2:1 balance, for adaptive finite volumes.

Why an octree: the battery spans four metres of sand, a 20 mm shell and a few
millimetres of pipe wall.  A uniform Cartesian grid able to resolve the wall would need
10^8 cells; grading helps but is still a global choice.  An octree spends cells where
the solution asks for them, and this module provides the two things that make it safe:

* the **2:1 balance** - neighbouring leaves never differ by more than one level, which
  keeps the stencil local and the number of hanging nodes bounded (a coarse face is
  shared with at most four finer faces);
* a **conservative face list** - a coarse face is split into the finer faces that
  actually exist, each carrying its own conductance ``k A/d``, so every flux appears
  with opposite signs in the two cells that share it and the discrete balance closes to
  machine precision on *any* combination of levels.

**Status: complete.**  A face is probed at its centre and, when the leaf found there is
finer, at the centre of every sub-face of *half* the leaf size: a coarse face against
``n`` finer leaves comes back as those ``n`` leaves with the fine area, a pair of equal
leaves as one entry, and each pair is listed once - by the coarse side of a coarse/fine
pair, which is the side that sees the sub-faces, and by the lower index of an equal pair.
What the earlier passes got wrong: the probes walked the tangential offsets at the *leaf*
size, so a probe landed beyond the neighbouring sub-face and returned edge and corner
neighbours as well as face neighbours (738 entries on a uniform 4x4x4 tree where 144
exist), while the deduplication key (pair, axis) turned those extra neighbours into extra
entries; and the level was read as a refinement depth by `split` (which handed out
*larger* children and refused to split the root) while `uniform_tree`, `size` and
`coarsen` read it as the logarithm of the edge length.  The convention is the latter: the
level of a leaf is ``log2`` of its edge in finest cells, so level 0 is a single finest
cell and ``Octree.max_level`` is the leaf that is the whole box.  The Morton code packs 21
bits per coordinate now, so the round trip is exact for leaves of mixed level; it
interleaved only ``level + 1`` bits before and dropped the high bits of any leaf whose
coordinates did not fit in its own level.

The leaf list is indexed per level by the level-normalised corner, so a neighbour query
is a couple of dictionary lookups rather than a walk over every leaf: a uniform tree of
32768 leaves and its 95232 faces are done in about half a second, which is the budget the
tests pin.  The blocks of Afivo (one dense leaf per octant) are the next step if a
*refined* tree of that size has to go faster - the balance rounds, not the face list, are
what costs seconds there - and they change no formula in this file.

The representation is a *linear* octree in the sense of p4est: every leaf is an integer
triple ``(level, x, y, z)`` with the coordinates counted in the finest cells, so
neighbour finding is integer arithmetic and a Morton code gives a canonical order.

The assembly reuses the harmonic mean of the conductivities and the
``k A / (d_centers V)`` coefficient of the structured solver, so an octree mesh and a
graded Cartesian mesh agree on the same solution where their cells agree.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
from scipy import sparse

EPS = 1e-30

#: bits per coordinate in the Morton code: 3 * 21 = 63, the 64-bit linear index of p4est
MORTON_BITS = 21


@dataclass(frozen=True, order=True)
class Leaf:
    """One octree leaf: ``level`` and its lower corner in finest-cell units.

    ``level`` is ``log2`` of the edge length in finest cells, so level 0 is a single
    finest cell and the coarsest leaf of a box of ``n`` finest cells has level
    ``log2(n)``.  The lower corner is a multiple of the edge length.
    """

    level: int
    x: int
    y: int
    z: int

    @property
    def size(self) -> int:
        """Edge length in finest cells."""
        return 1 << self.level

    @property
    def centre(self) -> tuple[float, float, float]:
        """Centre in finest-cell units (float)."""
        half = 0.5 * self.size
        return (self.x + half, self.y + half, self.z + half)

    def morton(self) -> int:
        """Interleaved-bit code: a canonical order that keeps neighbours close.

        Every coordinate gets a full ``MORTON_BITS`` field, so leaves of different level
        can be compared and recovered from their codes.
        """
        code = 0
        for bit in range(MORTON_BITS):
            for axis, value in enumerate((self.x, self.y, self.z)):
                code |= ((value >> bit) & 1) << (3 * bit + axis)
        return code


def locate_in_tables(tables: list[tuple[np.ndarray, np.ndarray]], n: int,
                     points: np.ndarray) -> np.ndarray:
    """Index of the leaf covering each point, from per-level sorted corner tables.

    ``tables[level]`` is ``(sorted level-normalised codes, leaf positions)`` as
    :meth:`Octree._leaf_arrays` builds them; the lookup is one ``searchsorted`` per level,
    coarsest first.  A table of a *previous* tree answers "which old leaf holds this
    point", which is how a refinement carries its fields.  Points outside get -1.
    """
    points = np.asarray(points, dtype=np.int64).reshape(-1, 3)
    out = np.full(points.shape[0], -1, dtype=np.int64)
    inside = np.all((points >= 0) & (points < n), axis=1)
    for level in range(len(tables) - 1, -1, -1):
        codes, positions = tables[level]
        pending = np.flatnonzero(inside & (out < 0))
        if pending.size == 0:
            break
        if codes.size == 0:
            continue
        shifted = points[pending] >> level
        wanted = shifted[:, 0] | (shifted[:, 1] << 21) | (shifted[:, 2] << 42)
        where = np.minimum(np.searchsorted(codes, wanted), codes.size - 1)
        hit = codes[where] == wanted
        out[pending[hit]] = positions[where[hit]]
    return out


def _leaf_key(leaf: Leaf) -> tuple[int, int, int, int]:
    """The sort key of a leaf: its dataclass order as a tuple."""
    return (leaf.level, leaf.x, leaf.y, leaf.z)


def _morton_roundtrip(leaf: Leaf) -> Leaf:
    """Recover a leaf from its Morton code (used by the tests and by sorting)."""
    code = leaf.morton()
    x = y = z = 0
    for bit in range(MORTON_BITS):
        x |= ((code >> (3 * bit + 0)) & 1) << bit
        y |= ((code >> (3 * bit + 1)) & 1) << bit
        z |= ((code >> (3 * bit + 2)) & 1) << bit
    return Leaf(leaf.level, x, y, z)


# ---------------------------------------------------------------------------
#: face order: west, east, south, north, bottom, top (same as the structured solver)
FACES = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
FACE_AXIS = {"x_min": 0, "x_max": 0, "y_min": 1, "y_max": 1, "z_min": 2, "z_max": 2}
FACE_SIGN = {"x_min": -1, "x_max": +1, "y_min": -1, "y_max": +1, "z_min": -1, "z_max": +1}
#: the two axes a face spans, in the order the sub-face probes walk them
_FACE_TANGENTIAL = {"x_min": (1, 2), "x_max": (1, 2), "y_min": (0, 2), "y_max": (0, 2),
                    "z_min": (0, 1), "z_max": (0, 1)}


class Octree:
    """A set of leaves covering a box of ``n^3`` finest cells, kept 2:1 balanced."""

    def __init__(self, n_finest: int, leaves: Iterable[Leaf] | None = None) -> None:
        if n_finest < 2 or (n_finest & (n_finest - 1)):
            raise ValueError("n_finest must be a power of two")
        self.n = int(n_finest)
        self.max_level = int(np.log2(self.n))
        # the default tree is the single leaf that is the whole box
        self._reindex(leaves if leaves else [Leaf(self.max_level, 0, 0, 0)])

    # ------------------------------------------------------------- structure
    @property
    def n_cells(self) -> int:
        return len(self.leaves)

    def volume(self, leaf: Leaf) -> float:
        """Volume in finest-cell units cubed."""
        return float(leaf.size ** 3)

    def _reindex(self, leaves: Iterable[Leaf]) -> None:
        """Rebuild the leaf list and the lookups, dropping duplicates.

        Duplicates are not hypothetical: a split that reaches the same leaf twice (two
        parents asking for it, or a refinement round revisiting it) would put the same
        leaf in the list several times, ``_index`` would keep only the last position, and
        every earlier copy would look like a leaf with no faces - which is what made the
        face list lose faces and the matrix singular.  The per-level tables key a leaf by
        its corner shifted down to its own level, which is what the neighbour search
        needs to be a couple of dictionary lookups instead of a walk over every leaf.
        """
        # the key is the dataclass order (level, x, y, z) as a plain tuple: the same order,
        # without a Python ``__lt__`` call per comparison (millions on a large tree)
        self.leaves = sorted(set(leaves), key=_leaf_key)
        self._index = {leaf: i for i, leaf in enumerate(self.leaves)}
        #: bumped by every mutation: consumers key their per-tree caches on it
        self.version = getattr(self, "version", 0) + 1
        # the face list is a function of the leaves alone: computed once per tree state
        # (every mutation ends here), because a transient asks for it several times a step
        self._faces: list[tuple[int, int, int, float, float]] | None = None
        self._face_array: np.ndarray | None = None
        self._sizes: np.ndarray | None = None
        self._centres: np.ndarray | None = None
        by_level: list[dict[tuple[int, int, int], int]] = [
            {} for _ in range(self.max_level + 1)]
        for index, leaf in enumerate(self.leaves):
            if not 0 <= leaf.level <= self.max_level:
                raise ValueError(f"leaf {leaf} does not fit a box of {self.n} finest cells")
            by_level[leaf.level][(leaf.x >> leaf.level, leaf.y >> leaf.level,
                                  leaf.z >> leaf.level)] = index
        self._level_cells = by_level

    def split(self, leaf: Leaf) -> list[Leaf]:
        """The eight children of a leaf: one level finer, half the edge length."""
        if leaf.level <= 0:
            raise ValueError("the leaf is already at the finest level")
        half = leaf.size // 2
        return [Leaf(leaf.level - 1, leaf.x + dx, leaf.y + dy, leaf.z + dz)
                for dx in (0, half) for dy in (0, half) for dz in (0, half)]

    def _locate(self, x: int, y: int, z: int, coarsest: int = -1) -> int:
        """Index of the leaf covering a point (in finest-cell units), or -1.

        The walk starts at ``coarsest`` - the coarsest level a neighbour can have, which
        is ``leaf.level + 1`` under the 2:1 rule - and goes towards the fine levels:
        leaves are disjoint, so the first level that has a leaf over the point holds the
        (unique) leaf that covers it.  Neighbours must be searched from the coarser bound
        *down*, so that a finer neighbour is reached as well; a search that stopped at the
        caller's own level was blind to refined neighbours, which is what left leaves of a
        graded tree without any face.
        """
        top = self.max_level if coarsest < 0 else min(coarsest, self.max_level)
        cells = self._level_cells
        for level in range(top, -1, -1):
            index = cells[level].get((x >> level, y >> level, z >> level))
            if index is not None:
                return index
        return -1

    def locate(self, x: float, y: float, z: float) -> Leaf | None:
        """The leaf covering a point given in *finest-cell* units, or ``None`` outside.

        The public face of the walk ``_locate`` does, and the point location the painters
        of ``docs/16_ADAPTIVE_MESH_MIGRATION.md`` step 6 need: a rasteriser asks the tree
        what is *there* instead of counting indices.  The coordinate is floored to the
        finest cell, so a leaf owns its lower corner: a point on a leaf's low face belongs
        to that leaf and a point on its high face to the neighbour above it.
        """
        corner = (int(np.floor(x)), int(np.floor(y)), int(np.floor(z)))
        if min(corner) < 0 or max(corner) >= self.n:
            return None
        index = self._locate(*corner)
        return None if index < 0 else self.leaves[index]

    def cell_size_at(self, x: float, y: float, z: float,
                     physical_size: float = 1.0) -> float:
        """Edge of the leaf covering a point, in finest-cell units times ``physical_size``.

        The tree's ``Mesh3D.cell_size_at``: a leaf is cubic, so its edge *is* the
        characteristic size ``V**(1/3)``, and the point (not the cell index) is what a
        rasteriser measures a zone against.  Finest-cell units by default, the convention
        of :meth:`faces`; ``physical_size`` [m] is the edge of one finest cell.
        """
        leaf = self.locate(x, y, z)
        if leaf is None:
            raise ValueError(f"no leaf contains the point ({x}, {y}, {z})")
        return float(leaf.size) * physical_size

    def _face_neighbours(self, leaf: Leaf, face: str) -> list[int]:
        """Indices of the leaves sharing this face, in sub-face order.

        The face is probed at its centre.  A leaf that covers the centre and is not finer
        than this one covers the whole face (it is aligned and at least as large), so the
        face belongs to a single pair; when the leaf found there is finer it covers only
        part of the face, and the face is then tiled by leaves of at most half the edge
        length (2:1), which the four sub-face probes collect.  Every probe sits strictly
        on the far side of the plane, so the leaf itself can never come back and no edge
        or corner neighbour can pass for a face neighbour.
        """
        axis = FACE_AXIS[face]
        corner = (leaf.x, leaf.y, leaf.z)
        size = leaf.size
        near = corner[axis]
        # every probe sits one finest cell inside the neighbour: the neighbour's first
        # cell on the high side, its last cell on the low side.  A point on the plane
        # itself belongs to this leaf, because the search floors the coordinates, so a
        # probe placed at ``near - size`` (the far side of a finer neighbour) would walk
        # past the sub-face it means to look at.
        if FACE_SIGN[face] > 0:
            if near + size >= self.n:
                return []                              # the far wall of the box
            axis_coord = near + size
        else:
            if near <= 0:
                return []                              # the near wall of the box
            axis_coord = near - 1
        coarsest = leaf.level + 1          # the 2:1 rule allows no coarser neighbour
        tangent_a, tangent_b = _FACE_TANGENTIAL[face]
        centre = [corner[0] + size // 2, corner[1] + size // 2, corner[2] + size // 2]
        centre[axis] = axis_coord
        first = self._locate(centre[0], centre[1], centre[2], coarsest)
        if first < 0:
            return []
        if self.leaves[first].level >= leaf.level:
            # a leaf at least this large that covers the centre of the face covers the
            # whole face, so the face belongs to one pair only
            return [first]
        step = max(size // 2, 1)          # the finest a neighbour can be under 2:1
        found: list[int] = []
        for offset_a in (0, step):
            for offset_b in (0, step):
                point = [corner[0], corner[1], corner[2]]
                point[axis] = axis_coord
                point[tangent_a] += offset_a
                point[tangent_b] += offset_b
                index = self._locate(point[0], point[1], point[2], coarsest)
                if index >= 0 and index not in found:
                    found.append(index)
        return found

    def neighbours(self, leaf: Leaf, face: str) -> list[Leaf]:
        """Leaves sharing this face: one, or up to four when the neighbour is finer."""
        return [self.leaves[index] for index in self._face_neighbours(leaf, face)]

    # -------------------------------------------------------------- balancing
    def balance_from(self, seeds: Iterable[Leaf]) -> int:
        """Enforce the 2:1 rule around ``seeds`` (the leaves a split just made).

        A refinement can only break the rule next to the leaves it created, so only
        their neighbours are looked at: every face of a seed is probed just outside its
        centre, the leaf found there is the neighbour covering the whole face whenever it
        is coarser, and it is split when it is more than one level coarser.  Its children
        are the next seeds, so a cascade runs as far as it has to and no further.  The
        result is the same tree :meth:`balance` gives (the minimal balanced refinement is
        unique); the cost is the size of the change, not of the tree - a full sweep of
        every face of every leaf per round is what made a large build take tens of
        seconds.  Returns how many leaves were split.
        """
        splits = 0
        frontier = {leaf for leaf in seeds if leaf in self._index}
        while frontier:
            # every seed's three outward faces, probed at once: a seed is a child of a
            # split, so along each axis one face is shared with a sibling (same level,
            # never a violation) and only the other one looks outside the parent
            seeds_array = np.array([(leaf.x, leaf.y, leaf.z, leaf.level)
                                    for leaf in frontier], dtype=np.int64).reshape(-1, 4)
            corner, level = seeds_array[:, :3], seeds_array[:, 3]
            size = np.left_shift(np.int64(1), level)
            centre = corner + (size // 2)[:, None]
            corners_all, levels_all = self._leaf_arrays()
            to_split_index: list[np.ndarray] = []
            for axis in range(3):
                outward = (corner[:, axis] >> level) & 1
                point = centre.copy()
                point[:, axis] = np.where(outward == 1, corner[:, axis] + size,
                                          corner[:, axis] - 1)
                inside = (point[:, axis] >= 0) & (point[:, axis] < self.n)
                found = self.locate_many(point[inside])
                own = level[inside]
                ok = found >= 0
                # a neighbour two or more levels coarser breaks the rule
                coarse = ok & (levels_all[np.where(ok, found, 0)] > own + 1)
                to_split_index.append(found[coarse])
            indices = np.unique(np.concatenate(to_split_index)) if to_split_index else []
            to_split = {self.leaves[int(index)] for index in indices}
            if not to_split:
                break
            new: list[Leaf] = []
            frontier = set()
            for leaf in self.leaves:
                if leaf in to_split:
                    children = self.split(leaf)
                    new.extend(children)
                    frontier.update(children)
                    splits += 1
                else:
                    new.append(leaf)
            self._reindex(new)
        return splits

    def balance(self, rounds: int = 32) -> int:
        """Enforce the 2:1 rule on the whole tree; returns how many leaves were split."""
        splits = 0
        for _ in range(rounds):
            to_split: set[Leaf] = set()
            for leaf in self.leaves:
                for face in FACES:
                    for neighbour in self.neighbours(leaf, face):
                        if abs(leaf.level - neighbour.level) > 1:
                            # the coarser leaf has the larger level: it is the one that
                            # gets refined until the pair is one level apart
                            coarser = leaf if leaf.level > neighbour.level else neighbour
                            to_split.add(coarser)
            if not to_split:
                break
            new: list[Leaf] = []
            for leaf in self.leaves:
                if leaf in to_split:
                    new.extend(self.split(leaf))
                    splits += 1
                else:
                    new.append(leaf)
            self._reindex(new)
        return splits

    # ------------------------------------------------------------ refinement
    def refine(self, indicator: Callable[[Leaf], float], threshold: float,
               levels: int = 1) -> None:
        """Split every leaf whose indicator exceeds ``threshold``, then balance."""
        for _ in range(max(levels, 0)):
            marked = [leaf for leaf in self.leaves
                      if leaf.level > 0 and indicator(leaf) > threshold]
            if not marked:
                break
            marked_set = set(marked)
            new: list[Leaf] = []
            children: list[Leaf] = []
            for leaf in self.leaves:
                if leaf in marked_set:
                    split = self.split(leaf)
                    new.extend(split)
                    children.extend(split)
                else:
                    new.append(leaf)
            self._reindex(new)
            # only the new leaves can have broken the 2:1 rule: balance around them
            self.balance_from(children)

    def coarsen(self, indicator: Callable[[Leaf], float], threshold: float) -> None:
        """Merge the eight children of a parent when all of them are below threshold."""
        parents: dict[Leaf, list[Leaf]] = defaultdict(list)
        for leaf in self.leaves:
            if leaf.level >= self.max_level:
                continue                               # the whole box has no parent
            half = leaf.size
            parent = Leaf(leaf.level + 1, (leaf.x // (2 * half)) * 2 * half,
                          (leaf.y // (2 * half)) * 2 * half,
                          (leaf.z // (2 * half)) * 2 * half)
            parents[parent].append(leaf)
        merged = {parent for parent, children in parents.items()
                  if len(children) == 8 and all(indicator(c) <= threshold for c in children)}
        if not merged:
            return
        kept: list[Leaf] = [leaf for leaf in self.leaves
                            if not any(leaf in parents[p] for p in merged)]
        kept.extend(merged)
        self._reindex(kept)
        self.balance()

    # ------------------------------------------------------------------ view
    def cell_centres(self) -> np.ndarray:
        """(n_cells, 3) centres in finest-cell units."""
        if self._centres is None:
            self._centres = np.array([leaf.centre for leaf in self.leaves],
                                     dtype=float).reshape(-1, 3)
        return self._centres.copy()

    def cell_sizes(self) -> np.ndarray:
        """Edge length of every leaf, in finest-cell units."""
        if self._sizes is None:
            self._sizes = np.array([leaf.size for leaf in self.leaves], dtype=float)
        return self._sizes.copy()

    def level_histogram(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for leaf in self.leaves:
            counts[leaf.level] = counts.get(leaf.level, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        levels = ", ".join(f"L{level}: {count}" for level, count
                           in self.level_histogram().items())
        return (f"octree {self.n}^3 finest cells: {self.n_cells} leaves ({levels}), "
                f"finest size {1.0 / self.n:.4f} of the box")

    # ---------------------------------------------------------- conservative faces
    def faces(self) -> list[tuple[int, int, int, float, float]]:
        """Conservative face list: (cell_i, cell_j, axis, area, centre distance).

        A coarse face is decomposed into the finer faces that exist opposite it, so a
        coarse face against ``n`` finer leaves is listed as ``n`` entries, each with the
        area of the fine sub-face and the centre-to-centre distance of its own pair, while
        a pair of equal leaves is listed once.  The coarse side of a coarse/fine pair is
        the one that lists the sub-faces, and the fine side finds them already listed; an
        equal pair is listed by its lower index.  Areas and distances are in finest-cell
        units, so the caller multiplies by the physical cell size to get metres.  The tree
        must be 2:1 balanced - every mutation in this class ends in `balance`.
        """
        if self._faces is not None:
            return self._faces
        i, j, axis, area, distance = self._face_arrays()
        self._faces = list(zip(i.tolist(), j.tolist(), axis.tolist(), area.tolist(),
                               distance.tolist(), strict=True))
        return self._faces

    # ------------------------------------------------ vectorised neighbour search
    def _leaf_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """``(corners (n, 3), levels (n,))`` as integer arrays, cached per tree state."""
        if getattr(self, "_arrays_version", None) != self.version:
            leaves = self.leaves
            self._corner_array = np.array([(leaf.x, leaf.y, leaf.z) for leaf in leaves],
                                          dtype=np.int64).reshape(-1, 3)
            self._level_array = np.array([leaf.level for leaf in leaves], dtype=np.int64)
            tables = []
            for level in range(self.max_level + 1):
                positions = np.flatnonzero(self._level_array == level)
                shifted = self._corner_array[positions] >> level
                codes = shifted[:, 0] | (shifted[:, 1] << 21) | (shifted[:, 2] << 42)
                order = np.argsort(codes, kind="stable")
                tables.append((codes[order], positions[order]))
            self._level_tables = tables
            self._arrays_version = self.version
        return self._corner_array, self._level_array

    def locate_many(self, points: np.ndarray) -> np.ndarray:
        """Index of the leaf covering each point (finest-cell units, ``(m, 3)``), or -1.

        :meth:`_locate` for many points at once: every level is one sorted table of
        level-normalised corners, and a point is looked up in all of them with one
        ``searchsorted`` per level, from the coarsest down.  Points outside the box get -1.
        """
        self._leaf_arrays()
        return locate_in_tables(self._level_tables, self.n, points)

    def _face_arrays(self) -> tuple[np.ndarray, ...]:
        """The conservative face list as arrays, built without a loop over the leaves.

        The same probes as :meth:`_face_neighbours` - the centre of every face just
        outside the leaf, then the four sub-face points when the leaf found there is finer
        - evaluated for all the leaves of a face direction at once with
        :meth:`locate_many`, and the same entry rules: an equal pair is listed by its lower
        index, a coarse/fine pair by the coarse side.  The entries come out in the order
        the per-leaf loop produced them (leaf, face, sub-face), so the list is the same list.
        """
        corners, levels = self._leaf_arrays()
        sizes = np.left_shift(np.int64(1), levels)
        parts: list[tuple[np.ndarray, ...]] = []
        for face_index, face in enumerate(FACES):
            axis = FACE_AXIS[face]
            tangent_a, tangent_b = _FACE_TANGENTIAL[face]
            near = corners[:, axis]
            if FACE_SIGN[face] > 0:
                coord = near + sizes
                valid = coord < self.n
            else:
                coord = near - 1
                valid = near > 0
            rows = np.flatnonzero(valid)
            if rows.size == 0:
                continue
            centre = corners[rows] + (sizes[rows] // 2)[:, None]
            centre[:, axis] = coord[rows]
            first = self.locate_many(centre)
            found = first >= 0
            rows, first = rows[found], first[found]
            own, other = levels[rows], levels[first]
            # a neighbour at least as large covers the whole face: one pair, listed only
            # by the lower index of an equal pair (a coarser neighbour lists it itself)
            equal = (other == own) & (first > rows)
            if equal.any():
                size = sizes[rows[equal]]
                parts.append((rows[equal], first[equal], np.full(size.size, axis),
                              (size * size).astype(float), size.astype(float),
                              np.full(size.size, face_index), np.zeros(size.size, int)))
            # a finer neighbour: the face is tiled by the leaves the sub-face probes find
            coarse = rows[other < own]
            if coarse.size == 0:
                continue
            step = np.maximum(sizes[coarse] // 2, 1)
            previous: list[np.ndarray] = []
            for sub, (offset_a, offset_b) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                point = corners[coarse].copy()
                point[:, axis] = coord[coarse]
                point[:, tangent_a] += offset_a * step
                point[:, tangent_b] += offset_b * step
                neighbour = self.locate_many(point)
                keep = neighbour >= 0
                for earlier in previous:
                    keep &= neighbour != earlier
                previous.append(neighbour)
                if not keep.any():
                    continue
                i, j = coarse[keep], neighbour[keep]
                side = np.minimum(sizes[i], sizes[j])
                parts.append((i, j, np.full(i.size, axis), (side * side).astype(float),
                              0.5 * (sizes[i] + sizes[j]).astype(float),
                              np.full(i.size, face_index), np.full(i.size, sub)))
        if not parts:
            empty = np.empty(0)
            return (empty.astype(np.int64), empty.astype(np.int64), empty.astype(np.int64),
                    empty, empty)
        i, j, axis, area, distance, face_key, sub_key = (np.concatenate(column)
                                                         for column in zip(*parts,
                                                                           strict=True))
        order = np.lexsort((sub_key, face_key, i))
        return (i[order], j[order], axis[order].astype(np.int64), area[order],
                distance[order])

    def face_array(self) -> np.ndarray:
        """:meth:`faces` as one ``(n_faces, 5)`` float array, cached with the list.

        Columns ``i, j, axis, area, distance`` in finest-cell units.  The array is shared
        and read-only: callers index it and never write into it.
        """
        if self._face_array is None:
            if self._faces is None:
                self._face_array = np.column_stack(
                    [np.asarray(column, dtype=float) for column in self._face_arrays()]
                ).reshape(-1, 5)
            else:
                self._face_array = np.asarray(self.faces(), dtype=float).reshape(-1, 5)
            self._face_array.setflags(write=False)
        return self._face_array

    # ------------------------------------------------------------------ assembly  # noqa: E501
    def diffusion_matrix(self, conductivity: np.ndarray | None = None,
                         physical_size: float = 1.0) -> sparse.csr_matrix:
        """The Laplacian on the octree, in the same per-volume form as the solver.

        ``k`` is per leaf (default one); ``physical_size`` [m] is the size of one
        finest cell, so a leaf of size ``s`` has edge ``s * physical_size``.  The
        coefficient of a face is ``k_face A / (d_centers V)`` with the harmonic mean of
        the two conductivities, exactly as in :mod:`src.solver.matrix`.
        """
        n = self.n_cells
        k = np.ones(n) if conductivity is None else np.asarray(conductivity, dtype=float)
        faces = self.face_array()
        i = faces[:, 0].astype(np.int64)
        j = faces[:, 1].astype(np.int64)
        k_face = 2.0 * k[i] * k[j] / (k[i] + k[j] + EPS)
        # area in metres^2, distance in metres, volume in metres^3
        a_face = faces[:, 3] * physical_size ** 2
        d_centers = faces[:, 4] * physical_size
        volume = (self.cell_sizes() * physical_size) ** 3
        coeff_i = k_face * a_face / (d_centers * volume[i])
        coeff_j = k_face * a_face / (d_centers * volume[j])
        # face by face, (i, j) interleaved: the summation order of the per-face loop this
        # replaced, so the operator is the same to the last bit
        pair_rows = np.column_stack((i, j)).ravel()
        pair_cols = np.column_stack((j, i)).ravel()
        pair_vals = np.column_stack((coeff_i, coeff_j)).ravel()
        diagonal = np.zeros(n)
        np.add.at(diagonal, pair_rows, pair_vals)
        rows = np.concatenate((pair_rows, np.arange(n)))
        cols = np.concatenate((pair_cols, np.arange(n)))
        vals = np.concatenate((-pair_vals, diagonal))
        return sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()

    def solve(self, source: np.ndarray, conductivity: np.ndarray | None = None,
              physical_size: float = 1.0, fixed: dict[int, float] | None = None,
              tolerance: float = 1e-10) -> np.ndarray:
        """Solve the steady diffusion problem, holding ``fixed`` cells at their value.

        ``source`` is the volumetric source per leaf [W/m^3]; the fixed cells are the
        Dirichlet set, which is how the shell of a storage is driven - their rows become
        the identity and their columns leave the operator, so it stays symmetric.  A
        graded operator is symmetrised by its cell volumes before CG sees it, as in
        :mod:`src.solver.linear`.  Returns the per-leaf temperature.
        """
        from scipy.sparse.linalg import cg

        b = np.asarray(source, dtype=float).copy()
        a = _apply_dirichlet(self.diffusion_matrix(conductivity, physical_size), b,
                             fixed or {})
        sizes = self.cell_sizes()
        if sizes.min() == sizes.max():
            solution, _ = cg(a, b, rtol=tolerance, maxiter=50 * self.n_cells)
            return np.asarray(solution, dtype=float)
        # the per-volume rows are symmetric only where the volumes agree: as in
        # src.solver.linear, a graded operator is symmetrised by its cell volumes before
        # CG sees it, which keeps the method applicable and the solution unchanged
        scale = np.sqrt((sizes * physical_size) ** 3)
        symmetrised = (sparse.diags(scale) @ a @ sparse.diags(1.0 / scale)).tocsr()
        solution, _ = cg(symmetrised, scale * b, rtol=tolerance, maxiter=50 * self.n_cells)
        return np.asarray(solution / scale, dtype=float)

    def flux_balance(self, temperature: np.ndarray, conductivity: np.ndarray | None = None,
                     physical_size: float = 1.0) -> float:
        """Net heat rate through the boundary of the box, weighed by the cell volumes.

        Every face carries a single conductance and both cells that share it build their
        balance from the same numbers, so summing the per-cell balances over the mesh
        cancels each interior face against itself and leaves only the flux through the
        domain boundary - zero here, because the face list stops at the wall.  That is the
        cancellation that closes the discrete energy balance to machine precision on any
        combination of levels; it is taken from the assembled operator, so a face whose
        two sides disagree about its conductance leaves a remainder here.
        """
        residual = self.diffusion_matrix(conductivity, physical_size) @ np.asarray(
            temperature, dtype=float)
        volumes = (self.cell_sizes() * physical_size) ** 3
        return float(volumes @ residual)


def _apply_dirichlet(a: sparse.csr_matrix, b: np.ndarray,
                     fixed: dict[int, float]) -> sparse.csr_matrix:
    """Impose ``T = value`` on the fixed cells, leaving the operator symmetric.

    The same symmetric elimination as :func:`src.solver.matrix.apply_dirichlet`: the
    coupling of a fixed cell to its neighbours moves to the right-hand side and its
    column leaves the operator before the row becomes the identity.  Replacing the row
    alone leaves the operator asymmetric, and CG is not applicable to one of those - it
    diverges silently instead of failing.
    """
    if not fixed:
        return a.tocsr()
    a = a.tocsc()
    for index, value in fixed.items():
        start, stop = a.indptr[index], a.indptr[index + 1]
        rows = a.indices[start:stop]
        coupling = a.data[start:stop]
        keep = rows != index
        b[rows[keep]] -= coupling[keep] * value
        a.data[start:stop] = 0.0
    a = a.tocsr().tolil()
    for index, value in fixed.items():
        a.rows[index] = [index]
        a.data[index] = [1.0]
        b[index] = value
    return a.tocsr()


def refine_by_gradient(tree: Octree, values: np.ndarray, threshold: float,
                       levels: int = 1) -> None:
    """Refine where the field jumps between neighbours: the cheap a posteriori rule.

    The jump across a face is the first-order indicator the AMR literature uses when a
    residual estimator would be too expensive; it is monotone in the local error and
    cheap to compute on the face list already built for the assembly.
    """
    per_cell: dict[int, float] = {}
    for i, j, _axis, _area, _distance in tree.faces():
        jump = abs(values[i] - values[j])
        per_cell[i] = max(per_cell.get(i, 0.0), jump)
        per_cell[j] = max(per_cell.get(j, 0.0), jump)

    def indicator(leaf: Leaf) -> float:
        return per_cell.get(tree._index.get(leaf, -1), 0.0)

    tree.refine(indicator, threshold, levels=levels)


def uniform_tree(n_finest: int, level: int) -> Octree:
    """Every leaf at the same level, ``2 ** level`` finest cells on a side."""
    size = 1 << level
    leaves = [Leaf(level, x, y, z)
              for x in range(0, n_finest, size)
              for y in range(0, n_finest, size)
              for z in range(0, n_finest, size)]
    return Octree(n_finest, leaves)
