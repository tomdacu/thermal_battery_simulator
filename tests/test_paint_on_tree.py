"""The painters that need a point, on a tree: the pipe rasteriser and the network.

Step 6 of ``docs/16_ADAPTIVE_MESH_MIGRATION.md``.  A mask over ``(i, j, k)`` becomes a mask
over the cell centres and the "cells the axis crosses" walk measures the leaf edge instead
of ``dx``/``dy``/``dz``; what the module pins is that the model painted does not move:

* **the rasteriser splits a centreline the same way** - the same cells with the same length
  in each, on the uniform tree and on the ``Mesh3D`` of the same cells (the ``box_pair``
  fixture of ``tests/conftest.py``) - including a run that sits exactly on a cell face and
  one that leaves the box;
* **the network paints the same leaves** - ``PipeNetwork.paint`` marks the same tube cells
  with the same material, the same gas film and the same report, on either mesh;
* **the wetted area stays geometric** - ``sum(pi d L_cell) = pi d L_total`` on the tree as
  on the grid, and the cells the paint leaves out (outside the vessel, outside the domain)
  are the same ones on both roads;
* **a refined tree paints too** - where no structured twin exists the invariants are the
  check: the area is still ``pi d L_total`` and every painted leaf is a cell the network's
  centrelines reach.

Nothing here reads an area off the mask: the mask paints material, the centreline measures
surface.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand
from src.core.materials import MaterialManager
from src.core.mesh import BoundaryType, MaterialID, Mesh3D
from src.core.pipe_network import PipeNetworkConfig, build_pipe_network
from src.core.pipes import rasterize_pipe

#: the per-cell state the painters write through the protocol
PAINTED = ("material_id", "k", "rho", "cp", "Q_source", "source_mask", "bc_h",
           "bc_T_inf", "boundary_type")

#: sample centrelines: generic, exactly on a cell face, along an axis, diagonal, one that
#: starts inside the box and ends outside it, and one entirely outside
CENTRELINES = [
    ([(4.2, 4.35, 1.0), (4.2, 4.35, 7.0)], "vertical"),
    ([(4.5, 4.3, 1.0), (4.5, 4.3, 7.0)], "on-a-face"),
    ([(1.0, 4.0, 1.0), (7.0, 4.0, 1.0)], "horizontal"),
    ([(1.0, 1.0, 1.0), (7.0, 7.0, 7.0)], "diagonal"),
    ([(4.0, 4.0, 1.0), (4.0, 4.0, 9.0)], "leaving-the-box"),
    ([(9.0, 9.0, 1.0), (9.0, 9.0, 7.0)], "outside"),
]


# ------------------------------------------------------------------- helpers
def vessel(**kwargs) -> PipeNetworkConfig:
    """A small vessel centred in the box; the pitches are explicit to keep it quick."""
    base = dict(radius=1.5, height=5.0, band_bottom=0.3, diameter=0.04,
                horizontal_pitch=0.2, vertical_pitch=0.25)
    base.update(kwargs)
    return PipeNetworkConfig(**base)


def leaf_cells(tree: AdaptiveMesh, structured: Mesh3D) -> np.ndarray:
    """Flat ``Mesh3D`` cell of every leaf centre.

    A uniform tree and a uniform ``Mesh3D`` of the same spacing have the same cells, so
    the map is one to one: it is what reads a structured field back as a per-leaf vector
    (the counterpart of ``tests/test_geometry_on_a_tree.py::cell_of``).
    """
    return np.array([structured.ijk_to_linear(*structured.find_cell(*centre))
                     for centre in tree.centres()])


def painted(mesh, index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The painted fields of ``mesh`` as per-cell vectors, in structured cell order."""
    out = {}
    for name in PAINTED:
        values = np.asarray(getattr(mesh, name))
        out[name] = values if values.ndim == 1 else values.reshape(-1, order="F")[index]
    return out


def differences(tree: AdaptiveMesh, structured: Mesh3D) -> dict[str, int]:
    """Leaves that disagree with their ``Mesh3D`` cell, per field (zero when equal)."""
    on_tree, on_grid = painted(tree), painted(structured, leaf_cells(tree, structured))
    return {name: int(np.count_nonzero(on_tree[name] != on_grid[name])) for name in PAINTED}


def same_paint(on_tree, on_grid) -> None:
    """The two paint reports describe the same paint.

    The areas are summed over the cells in the mesh's own order, which differs between a
    tree and a grid, so the floats are compared at the round-off of the sum and the
    counts, the film and the notes - the part of the report that is not a sum - exactly.
    """
    assert on_tree.cells == on_grid.cells
    assert on_tree.riser_cells == on_grid.riser_cells
    assert on_tree.area == pytest.approx(on_grid.area, rel=1e-12)
    assert on_tree.dropped == pytest.approx(on_grid.dropped, rel=1e-12)
    assert on_tree.insulated == pytest.approx(on_grid.insulated, rel=1e-12)
    assert on_tree.h_fluid == on_grid.h_fluid
    assert on_tree.t_fluid == on_grid.t_fluid
    assert on_tree.material == on_grid.material
    assert on_tree.notes == on_grid.notes


def refined_tree(finest: int = 32, physical_size: float = 0.25) -> AdaptiveMesh:
    """A tree refined around the vessel: 0.125 m leaves, two across a tube spacing."""
    return AdaptiveMesh.from_bands(
        finest, physical_size,
        [RefinementBand((1.5, 1.5, 0.0), (6.5, 6.5, 6.5), 0.125)])


def distance_to_runs(centres: np.ndarray, runs) -> np.ndarray:
    """Distance from every centre to the nearest centreline of ``runs`` [m]."""
    out = np.full(centres.shape[0], np.inf)
    for run in runs:
        for a, b in zip(run.points[:-1], run.points[1:], strict=True):
            span = b - a
            norm = float(span @ span)
            t = (np.clip(((centres - a) @ span) / norm, 0.0, 1.0) if norm > 0.0
                 else np.zeros(centres.shape[0]))
            out = np.minimum(out, np.linalg.norm(centres - (a + t[:, None] * span), axis=1))
    return out


# ------------------------------------------------------------- the point location
def test_the_tree_answers_a_point_with_its_leaf_and_the_leaf_size(box_pair):
    """The prerequisite of the step: the octree locates a point and measures its leaf.

    A painter that works on cells rather than on index arithmetic needs exactly this much
    (``docs/16`` step 6): which leaf holds a point, and how big it is.  Both the tree and
    the structured mesh put a point on a cell *face* in the cell above it - the placement
    the rasteriser overrides with its own convention - and a point outside the box belongs
    to no leaf at all.
    """
    tree, structured = box_pair()
    octree, size = tree.tree, tree.physical_size
    for point in ((0.0, 0.0, 0.0), (4.3, 4.7, 6.1), (7.99, 7.99, 7.99)):
        finest = tuple(value / size for value in point)
        leaf = octree.locate(*finest)
        assert leaf is not None
        assert octree.cell_size_at(*finest) == pytest.approx(float(leaf.size))
        assert octree.cell_size_at(*finest, size) == pytest.approx(leaf.size * size)
        # the same cell the structured mesh names, at the same size, in metres
        assert (leaf.x, leaf.y, leaf.z) == structured.find_cell(*point)
        assert octree.cell_size_at(*finest, size) == pytest.approx(
            structured.cell_size_at(*point))

    # a point on a leaf face belongs to the leaf above it, on either mesh
    assert octree.locate(1.0, 1.0, 1.0) == octree.locate(1.25, 1.0, 1.0)
    assert octree.locate(1.0, 1.0, 1.0).x == structured.find_cell(0.5, 0.5, 0.5)[0] == 1
    # outside the box there is no leaf, and no size either
    for outside in ((-0.5, 0.0, 0.0), (float(octree.n), 0.0, 0.0), (0.0, -1e-9, 0.0)):
        assert octree.locate(*outside) is None
        with pytest.raises(ValueError, match="no leaf contains"):
            octree.cell_size_at(*outside)


# ------------------------------------------------------------- the pipe rasteriser
@pytest.mark.parametrize("points", [case[0] for case in CENTRELINES],
                         ids=[case[1] for case in CENTRELINES])
def test_the_rasteriser_splits_a_centreline_the_same_way_on_a_tree(box_pair, points):
    """The same cells with the same length in each: one rule, two point locations.

    The runs are the ones that make the placement convention visible: a pipe exactly on a
    cell face (``searchsorted`` books it in the cell below) and one that leaves the box (a
    sample outside belongs to no cell, so no length is booked there).
    """
    tree, structured = box_pair()
    index = leaf_cells(tree, structured)
    on_tree = rasterize_pipe(tree, points, 0.05)
    on_grid = rasterize_pipe(structured, points, 0.05)

    grid_length = np.zeros(structured.N_total)
    grid_length[on_grid.cells] = on_grid.length
    assert np.array_equal(index[on_tree.cells], on_grid.cells)
    assert on_tree.length == pytest.approx(grid_length[index[on_tree.cells]], rel=1e-12)
    assert float(np.sum(on_tree.length)) == pytest.approx(float(on_grid.length.sum()),
                                                          rel=1e-12)
    # the geometric surface is the polyline's, and the grid can only measure a part of it
    assert float(on_tree.total_area) == pytest.approx(on_grid.total_area, rel=1e-15)
    assert float(np.sum(on_tree.area)) <= on_tree.total_area * (1.0 + 1e-12)


# --------------------------------------------------------------- the network paint
def test_the_network_paints_the_same_leaves_on_a_tree(box_pair):
    """The tube cells, their material and their gas film: one model, two meshes."""
    tree, structured = box_pair()
    config = vessel()
    net = build_pipe_network(tree, config)
    same_paint(net.paint(tree, h_fluid=350.0, t_fluid=333.15),
               build_pipe_network(structured, config).paint(structured, h_fluid=350.0,
                                                            t_fluid=333.15))

    assert differences(tree, structured) == dict.fromkeys(PAINTED, 0)
    tubes = tree.material_id == int(MaterialID.TUBES)
    assert tubes.sum() > 0 and int(tubes.sum()) == int(
        np.count_nonzero(structured.material_id == int(MaterialID.TUBES)))
    # the paint marked the pipes and wrote the gas film, and no source; the cells keep
    # the properties they had (a thin pipe inside a cell of the bed - air in this box)
    props = MaterialManager().get("air")
    assert tree.rho[tubes].min() == pytest.approx(props.rho, rel=1e-12)
    assert tree.k[tubes].min() == pytest.approx(props.k, rel=1e-12)
    assert tree.bc_h[tubes].min() == pytest.approx(350.0, rel=1e-12)
    assert np.all(tree.bc_T_inf[tubes] == pytest.approx(333.15, rel=1e-12))
    assert np.all(tree.boundary_type[tubes] == int(BoundaryType.CONVECTION))
    # the film went to the tube cells and to nobody else
    assert not np.any((tree.bc_h > 0.0) & ~tubes)
    assert not tree.source_mask.any() and not np.any(tree.Q_source)


def test_a_lagged_header_is_painted_without_a_film_on_a_tree(box_pair):
    """The lagging is a fact of the model on either mesh: a pipe the gas does not heat.

    The lattice is a wide one - 0.8 m between the risers in a 0.5 m grid - so the headers
    and the jumpers own cells of their own between the tubes, which is where the lagging
    is visible in the film rather than only in the report.
    """
    tree, structured = box_pair()
    config = vessel(horizontal_pitch=0.8, vertical_pitch=0.8, insulated_headers=True)
    net = build_pipe_network(tree, config)
    report = net.paint(tree)
    same_paint(report, build_pipe_network(structured, config).paint(structured))

    assert differences(tree, structured) == dict.fromkeys(PAINTED, 0)
    tubes = tree.material_id == int(MaterialID.TUBES)
    # the paint writes no film value (the gas loop does, per cell): an exchanging cell is
    # one the assembly lists as convective, a lagged one is not
    film = tubes & (tree.boundary_type == int(BoundaryType.CONVECTION))
    assert 0 < int(film.sum()) < int(tubes.sum())   # the headers have cells of their own
    # the lagged surface is what the report says is left without a film
    assert report.insulated == pytest.approx(net.total_area - net.exchange_area,
                                             rel=1e-12)
    assert report.insulated > 0.0 and net.exchange_area > 0.0


def test_the_wetted_area_stays_geometric_on_a_tree(box_pair):
    """``sum(pi d L_cell) = pi d L_total``: the mask never contributes a square metre."""
    tree, structured = box_pair()
    net = build_pipe_network(tree, vessel())
    cells, lengths, areas = net.voxelize(tree)

    assert cells.size > 0
    assert float(np.sum(lengths)) == pytest.approx(net.total_length, rel=1e-12)
    assert float(np.sum(areas)) == pytest.approx(net.total_area, rel=1e-12)
    # the same centrelines on the structured mesh measure the same physical surface
    grid_lengths, grid_areas = net.voxelize(structured)[1:]
    assert float(np.sum(grid_lengths)) == pytest.approx(net.total_length, rel=1e-12)
    assert float(np.sum(grid_areas)) == pytest.approx(net.total_area, rel=1e-12)


def test_the_cells_outside_the_vessel_are_left_out_on_both_roads(box_pair):
    """The stubs that cross the wall are not painted, and the report says how much.

    The wetted area of a painted network is a geometric one: what the paint left out is
    the centreline surface minus the painted cells, so the stubs have to be accounted for
    - on a tree as on a grid, where the same cells are excluded.
    """
    tree, structured = box_pair()
    config = vessel()
    center = (2.0, 4.0)                      # vessel clear of the wall: nothing leaves it
    net = build_pipe_network(tree, config, center=center)
    report = net.paint(tree)
    same_paint(report, build_pipe_network(structured, config,
                                          center=center).paint(structured))

    assert report.dropped > 0.0              # the nozzle stubs cross the vessel wall
    assert report.area + report.dropped == pytest.approx(net.total_area, rel=1e-12)
    assert report.area == pytest.approx(float(np.sum(net.voxelize(tree)[2]))
                                        - report.dropped, rel=1e-12)
    # nothing outside the vessel was painted
    centres = tree.centres()[tree.material_id == int(MaterialID.TUBES)]
    assert np.all(np.hypot(centres[:, 0] - center[0], centres[:, 1] - center[1])
                  <= config.radius + 1e-9)
    assert differences(tree, structured) == dict.fromkeys(PAINTED, 0)


def test_a_network_that_leaves_the_domain_books_no_length_there_on_a_tree(box_pair):
    """A tube pushed through the wall books its length in no cell, on either road.

    The second exclusion a rasteriser has: a sample outside the box belongs to no cell at
    all, so the surface it carries is not painted and not reported as dropped either - it
    is simply not in the mesh.  The two roads must lose exactly the same surface, or the
    fluid solve would see a different wetted area on a tree than on the grid.
    """
    tree, structured = box_pair()
    config = vessel(wall_clearance=0.05)     # the outer risers reach past the wall
    center = (1.2, 4.0)
    net = build_pipe_network(tree, config, center=center)
    report = net.paint(tree)
    same_paint(report, build_pipe_network(structured, config,
                                          center=center).paint(structured))

    painted = float(np.sum(net.voxelize(tree)[2]))
    on_grid = float(np.sum(build_pipe_network(structured, config,
                                              center=center).voxelize(structured)[2]))
    assert painted == pytest.approx(on_grid, rel=1e-12)
    assert painted < net.total_area                       # some of it left the domain
    assert report.area + report.dropped == pytest.approx(painted, rel=1e-12)
    assert differences(tree, structured) == dict.fromkeys(PAINTED, 0)


# ------------------------------------------------------------------ the refined tree
def test_the_network_paints_a_refined_tree_and_keeps_its_area(box_pair):
    """Where no structured twin exists, the invariants are the check."""
    coarse, _structured = box_pair()
    tree = refined_tree()
    config = vessel()
    net = build_pipe_network(tree, config)
    report = net.paint(tree)
    _cells, _lengths, areas = net.voxelize(tree)
    tubes = tree.material_id == int(MaterialID.TUBES)

    assert report.cells == int(np.count_nonzero(tubes)) > 0
    assert float(np.sum(areas)) == pytest.approx(net.total_area, rel=1e-12)
    assert report.area + report.dropped == pytest.approx(net.total_area, rel=1e-12)
    # the paint follows the centrelines: a painted leaf is a cell the axis crosses, so its
    # centre is at most half a diagonal away from the pipe it was painted for
    centres = tree.centres()[tubes]
    reach = 0.5 * np.sqrt(3.0) * np.cbrt(tree.V[tubes]) + 0.5 * config.diameter
    assert float(np.max(distance_to_runs(centres, net.runs) - reach)) <= 0.0
    # the same network on the coarse tree paints fewer, fatter cells
    coarse_net = build_pipe_network(coarse, config)
    assert coarse_net.paint(coarse).cells < report.cells
