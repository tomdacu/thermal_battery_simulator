"""The anisotropic tree of boxes (``src/core/box_tree.py``) and the mesh built on it.

What is pinned: the face list of a uniform box tree is the octree's; after a refinement
in plan and in height the 2:1 rule holds per direction and the faces tile every leaf
face exactly; a uniform box tree and a uniform octree solve the same problem to the same
field; a field that varies along one axis is exact on leaves graded in the other two;
the balance closes on a graded, anisotropic tree.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand
from src.core.box_tree import BoxTree
from src.core.octree import uniform_tree


def _refined_tree() -> BoxTree:
    tree = BoxTree.uniform(16, 32, 0.1, 0.05, 3, 4)
    for _ in range(4):
        centre = tree.centres()
        tree.refine(np.hypot(centre[:, 0] - 0.3, centre[:, 1] - 0.3) < 0.3,
                    np.zeros(tree.n_cells, dtype=bool))
        tree.refine(np.zeros(tree.n_cells, dtype=bool), tree.lower()[:, 2] < 0.2)
    return tree


def test_a_uniform_box_tree_has_the_octree_faces():
    tree = BoxTree.uniform(4, 4, 1.0, 1.0, 0, 0)
    i, j, axis, area, distance = tree.face_arrays()
    octree = uniform_tree(4, 0).face_array()
    assert i.size == octree.shape[0] == 144
    assert np.allclose(area, 1.0) and np.allclose(distance, 1.0)
    # every pair once, from its low side
    assert np.all(i < j)


def test_refinement_keeps_the_balance_per_direction_and_tiles_every_face():
    tree = _refined_tree()
    assert len(tree.level_pairs()) > 4                  # flat, tall and cubic leaves
    i, j, axis, area, _distance = tree.face_arrays()
    assert np.abs(tree.lxy[i] - tree.lxy[j]).max() <= 1
    assert np.abs(tree.lz[i] - tree.lz[j]).max() <= 1
    extent, lower = tree.extents(), tree.lower()
    for direction in range(3):
        own = np.prod(np.delete(extent, direction, axis=1), axis=1)
        high = np.zeros(tree.n_cells)
        np.add.at(high, i[axis == direction], area[axis == direction])
        low = np.zeros(tree.n_cells)
        np.add.at(low, j[axis == direction], area[axis == direction])
        inner_high = lower[:, direction] + extent[:, direction] < tree.box[direction] - 1e-12
        inner_low = lower[:, direction] > 1e-12
        assert np.allclose(high[inner_high], own[inner_high], rtol=1e-12)
        assert np.allclose(low[inner_low], own[inner_low], rtol=1e-12)
    assert tree.volumes().sum() == pytest.approx(np.prod(tree.box), rel=1e-12)


def test_a_uniform_box_tree_solves_like_the_uniform_octree():
    def paint(mesh):
        centre = mesh.centres()
        mesh.k[:] = np.where(centre[:, 0] < 0.5, 2.0, 0.5)
        mesh.Q_source[:] = np.where(np.hypot(centre[:, 0] - 0.5, centre[:, 1] - 0.5)
                                    < 0.3, 1000.0, 0.0)
        mesh.set_fixed_temperature_bc("z_min", 300.0)
        mesh.set_convection_bc("x_max", 10.0, 290.0)

    box = AdaptiveMesh.from_box(8, 8, 1.0 / 8, 1.0 / 8, 0, 0)
    cube = AdaptiveMesh(uniform_tree(8, 0), 1.0 / 8)
    for mesh in (box, cube):
        paint(mesh)
        mesh.solve_steady()
    # the two leaf orders differ: compare by position
    order_box = np.lexsort(box.centres().T)
    order_cube = np.lexsort(cube.centres().T)
    assert np.allclose(box.T[order_box], cube.T[order_cube], rtol=0, atol=1e-8)


def test_a_vertical_profile_is_exact_on_leaves_graded_in_plan():
    """Dirichlet at the bottom and the top, insulated sides: T is linear in z, whatever
    the plan does - the vertical faces carry no flux and the horizontal ones the exact
    one, also between leaves of different widths."""
    mesh = AdaptiveMesh.from_box(16, 16, 0.1, 0.05, 3, 0)
    centre = mesh.centres()
    mesh._split_box(np.hypot(centre[:, 0] - 0.4, centre[:, 1] - 0.4) < 0.4,
                    np.zeros(mesh.n_cells, dtype=bool))
    mesh.k[:] = 1.5
    mesh.set_fixed_temperature_bc("z_min", 300.0)
    mesh.set_fixed_temperature_bc("z_max", 400.0)
    mesh.solve_steady()
    z = mesh.centres()[:, 2]
    free = ~mesh.fixed_mask()
    # the pinned wall leaves sit at their value; the free ones on the line through them
    first, last = 0.5 * 0.05, 0.8 - 0.5 * 0.05
    expected = 300.0 + 100.0 * (z - first) / (last - first)
    assert np.allclose(mesh.T[free], expected[free], atol=1e-8)


def test_the_balance_closes_on_a_graded_anisotropic_tree():
    mesh = AdaptiveMesh.from_box(16, 32, 0.1, 0.05)
    band = RefinementBand(low=(0.4, 0.4, 0.3), high=(1.2, 1.2, 1.0), size=0.1,
                          size_z=0.4)
    thin = RefinementBand(low=(0.0, 0.0, 0.0), high=(1.6, 1.6, 0.2), size=0.4,
                          size_z=0.05)
    mesh.refine_bands([band, thin])
    assert mesh.extent[:, 2].max() / mesh.extent[:, 0].min() >= 4.0
    assert mesh.extent[:, 0].max() / mesh.extent[:, 2].min() >= 4.0
    centre = mesh.centres()
    mesh.k[:] = np.where(centre[:, 2] < 0.2, 0.04, 1.0)
    mesh.Q_source[:] = np.where(np.hypot(centre[:, 0] - 0.8, centre[:, 1] - 0.8) < 0.3,
                                500.0, 0.0)
    mesh.set_fixed_temperature_bc("z_min", 283.15)
    mesh.set_convection_bc("z_max", 5.0, 293.15)
    result = mesh.solve_steady()
    assert result.balance.closure < 1e-9
