"""The adaptive octree: balance, conservative faces, and a real solution.

The claims this module makes have to be checkable, so the tests here pin them:

* the 2:1 balance really holds after refining (no neighbour more than one level away);
* the face list is conservative: every interface between two finest cells of the box is
  claimed by exactly one entry and the area of that entry is the number of finest
  interfaces it covers, a coarse face against ``n`` finer faces is ``n`` entries with the
  area of the fine sub-face, a pair of equal leaves exactly one;
* a one-dimensional conduction problem is solved to the analytic answer, including a
  two-material wall whose interface falls on a leaf face;
* the refinement criterion puts cells where the gradient is and nowhere else;
* a tree of tens of thousands of leaves is built, and its face list with it, in well
  under a second.

Every leaf is ``(level, x, y, z)`` with the level the ``log2`` of the edge length in
finest cells, so level 0 is a single finest cell and ``Octree.max_level`` is the whole
box.
"""
from __future__ import annotations

import time
from collections import Counter
from itertools import chain

import numpy as np
import pytest

from src.core.octree import FACES, Leaf, Octree, refine_by_gradient, uniform_tree


def _layered_tree(below: int = 8, levels: int = 3) -> Octree:
    """The box refined below a plane: level 1 underneath it, level 2 above."""
    tree = Octree(16)
    tree.refine(lambda leaf: 1.0 if leaf.z < below else 0.0, threshold=0.5, levels=levels)
    return tree


def _fixed_ends(tree: Octree, hot: float, cold: float) -> dict[int, float]:
    """Hold the bottom plane at ``hot`` and the top plane at ``cold``."""
    return {index: (hot if leaf.z == 0 else cold)
            for index, leaf in enumerate(tree.leaves)
            if leaf.z == 0 or leaf.z + leaf.size == tree.n}


def _unit_interfaces(tree: Octree) -> dict[tuple, tuple[int, int]]:
    """(axis, cell) -> the two leaves that share every interface of finest cells."""
    n = tree.n
    owner = -np.ones((n, n, n), dtype=int)
    for index, leaf in enumerate(tree.leaves):
        size = leaf.size
        owner[leaf.x:leaf.x + size, leaf.y:leaf.y + size, leaf.z:leaf.z + size] = index
    assert (owner >= 0).all(), "the leaf list must cover the box"
    interfaces: dict[tuple, tuple[int, int]] = {}
    for axis in range(3):
        low = [slice(None)] * 3
        low[axis] = slice(0, n - 1)
        high = [slice(None)] * 3
        high[axis] = slice(1, n)
        below, above = owner[tuple(low)], owner[tuple(high)]
        for cell in np.argwhere(below != above):
            key = (axis, *(int(c) for c in cell))
            interfaces[key] = (int(below[tuple(cell)]), int(above[tuple(cell)]))
    return interfaces


def _face_claims(tree: Octree) -> dict[tuple, list[tuple[int, tuple[int, int]]]]:
    """(axis, cell) -> the face entries that claim each interface of finest cells."""
    claims: dict[tuple, list[tuple[int, tuple[int, int]]]] = {}
    for entry, (i, j, axis, area, _distance) in enumerate(tree.faces()):
        a, b = tree.leaves[i], tree.leaves[j]
        low = [max(a.x, b.x), max(a.y, b.y), max(a.z, b.z)]
        high = [min(a.x + a.size, b.x + b.size),
                min(a.y + a.size, b.y + b.size),
                min(a.z + a.size, b.z + b.size)]
        assert high[axis] == low[axis], "an entry must join two leaves face to face"
        tangent = [t for t in range(3) if t != axis]
        covered = (high[tangent[0]] - low[tangent[0]]) * (
            high[tangent[1]] - low[tangent[1]])
        assert covered == area, f"area {area} does not match the overlap {covered}"
        for u in range(low[tangent[0]], high[tangent[0]]):
            for v in range(low[tangent[1]], high[tangent[1]]):
                cell = [0, 0, 0]
                cell[axis] = low[axis] - 1
                cell[tangent[0]] = u
                cell[tangent[1]] = v
                key = (axis, *cell)
                claims.setdefault(key, []).append((entry, (min(i, j), max(i, j))))
    return claims


# ------------------------------------------------------------------ structure
def test_a_leaf_survives_the_morton_round_trip():
    from src.core.octree import _morton_roundtrip
    for level in range(5):
        # the coordinates of a mixed-level leaf do not fit in its own level's bits: the
        # code has to spend a fixed field per coordinate, not ``level + 1``
        leaf = Leaf(level, 3 << level, 1 << level, 2 << level)
        assert _morton_roundtrip(leaf) == leaf
    for leaf in (Leaf(0, 0, 0, 0), Leaf(3, 24, 8, 16), Leaf(5, 1 << 20, 0, 0)):
        assert _morton_roundtrip(leaf) == leaf


def test_a_uniform_tree_covers_the_box_exactly():
    tree = uniform_tree(8, 1)
    assert tree.n_cells == 4 ** 3
    assert tree.leaves[0].size == 2
    total = sum(tree.volume(leaf) for leaf in tree.leaves)
    assert total == pytest.approx(8 ** 3)


def test_splitting_a_leaf_replaces_it_with_eight_children():
    tree = Octree(8)
    root = tree.leaves[0]
    assert (root.level, root.size) == (tree.max_level, 8)
    children = tree.split(root)
    assert len(children) == 8
    assert all(child.size == 4 for child in children)
    assert sum(tree.volume(child) for child in children) == tree.volume(root)
    assert sum(child.level for child in children) == 8 * (root.level - 1)
    with pytest.raises(ValueError):
        tree.split(Leaf(0, 0, 0, 0))               # already a single finest cell


# ------------------------------------------------------------------- balance
def test_the_2_to_1_balance_holds_after_refining_one_corner():
    tree = Octree(16)
    tree.refine(lambda leaf: 1.0 if (leaf.x, leaf.y, leaf.z) == (0, 0, 0) else 0.0,
                threshold=0.5, levels=2)
    levels = tree.level_histogram()
    assert len(levels) >= 2, "the corner must be finer than the rest"
    assert min(levels) == tree.max_level - 2
    for leaf in tree.leaves:                       # no neighbour may be two levels apart
        for face in FACES:
            for neighbour in tree.neighbours(leaf, face):
                assert abs(leaf.level - neighbour.level) <= 1


def test_a_coarse_face_touches_at_most_four_finer_faces():
    tree = _layered_tree(below=8, levels=3)
    fine = 0
    for leaf in tree.leaves:
        for face in FACES:
            neighbours = tree.neighbours(leaf, face)
            assert len(neighbours) <= 4
            for neighbour in neighbours:
                assert neighbour != leaf
                assert abs(neighbour.level - leaf.level) <= 1
            fine += sum(1 for neighbour in neighbours if neighbour.level < leaf.level)
    assert fine > 0, "a graded tree must have coarse faces against finer ones"


# ------------------------------------------------------------------- faces
def test_the_face_list_is_conservative_and_covers_the_interface_area():
    tree = uniform_tree(8, 1)
    faces = tree.faces()
    # a 4x4x4 uniform grid of leaves: 3 planes x 4x4 faces x 3 axes
    assert len(faces) == 3 * 16 * 3
    for i, j, _axis, area, distance in faces:
        assert tree.leaves[i].size == tree.leaves[j].size == 2
        assert area == pytest.approx(4.0)           # 2 x 2 finest cells
        assert distance == pytest.approx(2.0)       # centre to centre


def test_a_coarse_face_against_four_fine_faces_is_four_entries():
    """One entry per sub-face, carrying the area of the fine face and its distance."""
    tree = _layered_tree(below=8, levels=3)
    faces = tree.faces()
    coarse = [leaf for leaf in tree.leaves if leaf.z == 8]
    assert coarse, "the interface must have a coarse side"
    kinds = Counter()
    for leaf in coarse:
        index = tree._index[leaf]
        neighbours = tree.neighbours(leaf, "z_min")
        assert len(neighbours) == 4                  # 4 x 4 opposed by 2 x 2 faces
        assert all(n.size == leaf.size // 2 for n in neighbours)
        assert all(n.z + n.size == leaf.z for n in neighbours)
        expected = {tree._index[n] for n in neighbours}
        entries = [entry for entry in faces if entry[0] == index and entry[1] in expected]
        assert len(entries) == 4, "one entry per fine sub-face"
        assert all(area == 4.0 for _, _, _, area, _ in entries)     # 2 x 2 finest cells
        assert all(distance == 3.0 for *_, distance in entries)     # (4 + 2) / 2
        assert {entry[1] for entry in entries} == expected
        kinds.update("coarse/fine" for _ in entries)
    # and a pair of equal leaves is exactly one entry, however often it is looked at
    same = [entry for entry in faces if tree.leaves[entry[0]].size == tree.leaves[entry[1]].size]
    assert same and len(same) == len({(entry[0], entry[1], entry[2]) for entry in same})
    assert kinds["coarse/fine"] == 4 * len(coarse)


def test_the_flux_balance_is_zero_on_a_conservative_face_list():
    """Every face flux has an opposite twin: the sum must vanish identically."""
    tree = _layered_tree(below=8, levels=3)
    temperature = np.random.default_rng(3).normal(size=tree.n_cells) * 100.0 + 500.0
    assert abs(tree.flux_balance(temperature, physical_size=0.1)) < 1e-9 * 500.0


def test_a_graded_tree_loses_no_interface_and_reproduces_a_linear_field():
    """Conservation on mixed levels: the interface coverage and the node stencil."""
    tree = _layered_tree(below=8, levels=3)
    assert len(tree.level_histogram()) > 1, "the test needs mixed levels"
    for leaf in tree.leaves:                       # no leaf may be left without a face
        assert any(tree.neighbours(leaf, face) for face in FACES)

    interfaces = _unit_interfaces(tree)
    claims = _face_claims(tree)
    assert set(claims) == set(interfaces), "every interface must be claimed"
    for key, entries in claims.items():
        assert len(entries) == 1, f"{key} is claimed {len(entries)} times"
        assert entries[0][1] == tuple(sorted(interfaces[key]))
    areas = {entry: face[3] for entry, face in enumerate(tree.faces())}
    claimed = Counter(entry for entry, _pair in chain.from_iterable(claims.values()))
    assert claimed == Counter(areas), "an entry must cover exactly its own area"

    # a linear field is reproduced exactly: the two-point flux k A (T_i - T_j) / d is
    # exact for it, so every leaf that has all six sides listed sees zero net flux
    centres = tree.cell_centres()
    linear = 300.0 + 25.0 * centres[:, 2]
    residual = tree.diffusion_matrix(physical_size=0.02) @ linear
    interior = [i for i, leaf in enumerate(tree.leaves)
                if all(tree.neighbours(leaf, face) for face in FACES)]
    assert len(interior) > 100
    assert np.abs(residual[interior]).max() < 1e-8, "the node stencil must be exact"


# --------------------------------------------------------------- conduction
def test_one_dimensional_conduction_against_the_analytic_solution():
    """Two fixed temperatures at the ends: the profile is linear in the node centres."""
    size = 0.02
    tree = _layered_tree(below=12, levels=3)
    centres = tree.cell_centres() * size
    temperature = tree.solve(np.zeros(tree.n_cells), physical_size=size,
                             fixed=_fixed_ends(tree, 400.0, 300.0), tolerance=1e-12)
    # a conduction problem with no source and one conductivity has a linear profile,
    # whatever the levels: the nodes are the cell centres, so they sit on the line
    # between the two Dirichlet nodes
    first, last = centres[:, 2].min(), centres[:, 2].max()
    expected = 400.0 - 100.0 * (centres[:, 2] - first) / (last - first)
    assert np.allclose(temperature, expected, atol=0.5)
    assert temperature.min() >= 300.0 and temperature.max() <= 400.0


def test_a_two_material_wall_matches_the_analytic_flux():
    """The flux through a layered wall is the analytic one, at either interface."""
    size = 0.02
    interface = 8                                  # the material change falls on a face
    tree = _layered_tree(below=12, levels=3)
    conductivity = np.array([1.0 if leaf.z < interface else 5.0 for leaf in tree.leaves])
    temperature = tree.solve(np.zeros(tree.n_cells), conductivity=conductivity,
                             physical_size=size, fixed=_fixed_ends(tree, 400.0, 300.0),
                             tolerance=1e-12)
    # the two cells that meet at the interface are the same size, so the harmonic mean of
    # the FV flux is the exact series resistance; the nodes are the cell centres
    hot = 0.5 * 2 * size                           # centre of the bottom (size 2) cells
    cold = (16 - 0.5 * 4) * size                   # centre of the top (size 4) cells
    resistance = (interface * size - hot) / 1.0 + (cold - interface * size) / 5.0
    analytic = (400.0 - 300.0) / resistance * (16 * size) ** 2
    for plane in (interface, 12):                  # the material change and the level change
        flux = 0.0
        for i, j, axis, area, distance in tree.faces():
            if axis != 2:
                continue
            below, above = (i, j) if tree.leaves[i].z < tree.leaves[j].z else (j, i)
            if tree.leaves[below].z + tree.leaves[below].size != plane:
                continue
            k_face = 2.0 * conductivity[below] * conductivity[above] / (
                conductivity[below] + conductivity[above])
            flux += k_face * area * size ** 2 * (
                temperature[below] - temperature[above]) / (distance * size)
        assert flux == pytest.approx(analytic, rel=1e-6)


# ------------------------------------------------------------- refinement
def test_the_gradient_indicator_refines_the_steep_region_only():
    tree = uniform_tree(16, 2)                     # 8x8x8 leaves of four finest cells
    temperature = np.array([500.0 if leaf.z < 8 else 320.0 for leaf in tree.leaves])
    refine_by_gradient(tree, temperature, threshold=50.0, levels=1)
    assert 1 in tree.level_histogram()             # one level finer than the uniform mesh
    coarse = [leaf for leaf in tree.leaves if leaf.level == 2]
    assert coarse, "the flat regions must not be refined"
    for leaf in coarse:
        assert leaf.z + leaf.size <= 8 or leaf.z >= 8


def test_coarsening_puts_the_eight_children_back_into_one_parent():
    tree = uniform_tree(16, 3)                     # 2x2x2 leaves of eight finest cells
    start = list(tree.leaves)
    tree.refine(lambda leaf: 1.0 if leaf.z == 0 else 0.0, threshold=0.5, levels=1)
    refined = tree.n_cells
    assert refined == 4 + 4 * 8, "four parents became eight children each"
    tree.coarsen(lambda leaf: 1.0 if leaf.z >= 8 else 0.0, threshold=0.5)
    assert tree.n_cells == 8
    assert tree.leaves == start, "the eight children must go back into their parent"


# ------------------------------------------------------------- performance
def test_a_tree_of_twenty_thousand_leaves_builds_and_lists_its_faces_in_under_a_second():
    start = time.perf_counter()
    tree = uniform_tree(64, 1)                     # 32x32x32 leaves of two finest cells
    faces = tree.faces()
    elapsed = time.perf_counter() - start
    assert tree.n_cells == 32 ** 3 >= 20_000
    assert len(faces) == 3 * 31 * 32 ** 2          # 3 axes x 31 planes x 32x32 faces
    assert elapsed < 1.0, f"{elapsed:.3f} s for {tree.n_cells} leaves"
