"""Scene-building tests: the rendering path is exercised head-less.

PyVista can build and render a scene off-screen, so every field, clip plane and
colormap combination is checked without a display.  This is the test that would
have caught the "list of RGB tuples as cmap" crash (PyVista wants strings).
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand  # noqa: E402
from src.core.geometry import create_small_test_geometry  # noqa: E402
from src.core.mesh import Mesh3D  # noqa: E402
from src.viz import scene  # noqa: E402


@pytest.fixture(scope="module")
def model():
    mesh = Mesh3D(Lx=6.0, Ly=6.0, Lz=5.6, spacing=0.5)
    battery = create_small_test_geometry()
    battery.apply_to_mesh(mesh)
    return mesh, battery


@pytest.fixture(scope="module")
def tree_model():
    """The shared geometry on a tree: an 8 m cube, refined in one corner.

    The box is the one ``tests/conftest.py`` paints its models in (an 8 m cube, 0.25 m
    finest cell), and the band refines a 2 m corner down to that floor, so the scene
    carries the step between leaf sizes a real model has - a hanging node - instead of a
    uniform grid of cubes.
    """
    mesh = AdaptiveMesh.from_bands(
        32, 0.25, (RefinementBand(low=(0.0, 0.0, 0.0), high=(2.0, 2.0, 2.0), size=0.25),))
    create_small_test_geometry().apply_to_mesh(mesh)
    return mesh


@pytest.fixture(scope="module")
def plotter():
    try:
        plot = pv.Plotter(off_screen=True, window_size=(320, 240))
    except Exception as exc:                                    # pragma: no cover
        pytest.skip(f"no off-screen OpenGL available: {exc}")
    yield plot
    plot.close()


@pytest.mark.parametrize("field", scene.FIELD_ARRAYS)
@pytest.mark.parametrize("axis", ["x", "y", "z", None])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_every_field_and_clip_combination_renders(plotter, model, field, axis, fraction):
    """No combination may raise; an empty cut returns None instead of crashing."""
    mesh, _ = model
    plotter.clear()
    actor = scene.add_field(plotter, mesh, field, axis=axis, fraction=fraction)
    assert actor is not None or (field == "Material" and 0.5 * mesh.Lz or True)
    if actor is None:
        assert field == "Material" and axis is not None and fraction > 0.9


def test_material_field_uses_a_string_colormap(plotter, model):
    """PyVista raises TypeError for a list of RGB tuples: colours must be strings."""
    mesh, _ = model
    cmap = scene.material_cmap()
    assert all(isinstance(entry, str) for entry in cmap)
    assert len(cmap) == len(scene.MATERIAL_COLORS)
    plotter.clear()
    scene.add_field(plotter, mesh, "Material")


@pytest.mark.parametrize("cmap", ["coolwarm", "jet", "viridis", "plasma", "inferno", "turbo"])
def test_every_colormap_offered_by_the_gui(plotter, model, cmap):
    mesh, _ = model
    plotter.clear()
    assert scene.add_field(plotter, mesh, "Temperature", cmap=cmap) is not None


def test_legend_and_geometry_preview(plotter, model):
    mesh, battery = model
    plotter.clear()
    scene.add_material_legend(plotter)
    scene.add_geometry_preview(plotter, battery, mesh)
    plotter.render()


def test_field_values_are_aligned_with_the_cells(plotter, model):
    """A field varying along z must land in the cell whose centre has that value."""
    mesh, _ = model
    mesh.T = 273.15 + 100.0 * mesh.Z / mesh.Lz
    grid = scene.to_image_data(mesh, "Temperature")
    values = np.asarray(grid.cell_data["Temperature"])
    centres = np.asarray(grid.cell_centers().points)[:, 2]
    expected = 100.0 * centres / mesh.Lz
    assert np.abs(values - expected).max() < 1e-9


def test_exports_write_readable_files(tmp_path, model):
    mesh, _ = model
    vtk_path = tmp_path / "field.vti"
    csv_path = tmp_path / "field.csv"
    scene.export_vtk(mesh, str(vtk_path))
    scene.export_csv(str(csv_path), None, mesh=mesh)
    assert vtk_path.stat().st_size > 0
    loaded = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    assert loaded.shape[0] == mesh.N_total


def test_clip_grid_never_leaves_the_domain(model):
    mesh, _ = model
    for fraction in (0.0, 1.0, -5.0, 42.0):
        grid = scene.clip_grid(mesh, "Temperature", "z", fraction)
        assert grid.n_cells > 0
        assert np.all(np.asarray(grid.cell_centers().points)[:, 2] >= 0.0)


def test_geometry_preview_covers_the_whole_stack(plotter, model):
    """The preview must be continuous: storage, slabs, plate, shell and roof."""
    mesh, battery = model
    battery.cyl = battery.cylinder
    cyl = battery.cylinder
    plotter.clear()
    scene.add_geometry_preview(plotter, battery, mesh)
    bounds = [actor.GetBounds() for actor in plotter.renderer.actors.values()] \
        if hasattr(plotter, "renderer") else []
    zs = [b[4] for b in bounds] + [b[5] for b in bounds if len(b) == 6]
    assert min(zs) <= 1e-6 and max(zs) >= cyl.z_cone_apex - 1e-6


# ------------------------------------------------------------------ an adaptive tree
def test_a_tree_becomes_an_unstructured_grid_of_its_leaves(tree_model):
    """The leaves *are* the cells: one hexahedron each, in the mesh's own leaf order."""
    mesh = tree_model
    grid = scene.to_image_data(mesh, "Temperature")
    assert isinstance(grid, pv.UnstructuredGrid)
    assert grid.n_cells == mesh.n_cells
    assert np.asarray(grid.cell_data["Temperature"]).shape == (mesh.n_cells,)


def test_tree_field_values_are_aligned_with_the_leaves(tree_model):
    """A field varying along z must land in the leaf whose centre has that value.

    The comparison runs against the centres of the grid that was built, so the hexahedra
    have to be where the leaves are *and* the cell data in leaf order: a rotated corner
    order or a re-shuffled field fails it.
    """
    mesh = tree_model
    centres = scene.cell_centers(mesh)
    mesh.T = 273.15 + 100.0 * centres[:, 2] / mesh.box_size
    grid = scene.to_image_data(mesh, "Temperature")
    values = np.asarray(grid.cell_data["Temperature"])
    grid_centres = np.asarray(grid.cell_centers().points)[:, 2]
    assert np.abs(values - 100.0 * grid_centres / mesh.box_size).max() < 1e-9


def test_tree_leaf_volumes_are_the_cell_volumes_of_the_grid(tree_model):
    """Leaves of several sizes must be drawn at their own size, not at an average one."""
    mesh = tree_model
    grid = scene.to_image_data(mesh, "Temperature")
    sizes = np.asarray(mesh.V) ** (1.0 / 3.0)
    assert len(np.unique(np.round(sizes, 9))) > 1, "the fixture needs hanging nodes"
    assert grid.volume == pytest.approx(float(mesh.V.sum()), rel=1e-9)


@pytest.mark.parametrize("field", scene.FIELD_ARRAYS)
def test_every_field_of_a_tree_renders(plotter, tree_model, field):
    """Every field the GUI offers, at a cut, on a mesh with hanging nodes."""
    mesh = tree_model
    plotter.clear()
    assert scene.add_field(plotter, mesh, field, axis="z", fraction=0.5) is not None
    plotter.render()


@pytest.mark.parametrize("axis", ["x", "y", "z", None])
def test_every_clip_of_a_tree_renders(plotter, tree_model, axis):
    mesh = tree_model
    plotter.clear()
    scene.add_field(plotter, mesh, "Temperature", axis=axis, fraction=0.5)
    plotter.render()


def test_exports_write_readable_files_for_a_tree(tmp_path, tree_model):
    """The export follows the grid: a tree is written as the ``.vtu`` it can be read as."""
    mesh = tree_model
    written = scene.export_vtk(mesh, str(tmp_path / "field.vti"))
    assert written.endswith(".vtu") and (tmp_path / "field.vtu").stat().st_size > 0
    assert pv.read(written).n_cells == mesh.n_cells
    csv_path = tmp_path / "field.csv"
    scene.export_csv(str(csv_path), None, mesh=mesh)
    assert np.loadtxt(csv_path, delimiter=",", skiprows=1).shape[0] == mesh.n_cells
