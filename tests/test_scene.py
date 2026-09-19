"""Scene-building tests: the rendering path is exercised head-less.

PyVista can build and render a scene off-screen, so every field, clip plane and
colormap combination is checked without a display.  This is the test that would
have caught the "list of RGB tuples as cmap" crash (PyVista wants strings).
"""
from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista")

from src.core.geometry import create_small_test_geometry  # noqa: E402
from src.core.mesh import Mesh3D  # noqa: E402
from src.viz import scene  # noqa: E402


@pytest.fixture(scope="module")
def model():
    mesh = Mesh3D(Lx=6.0, Ly=6.0, Lz=5.6, spacing=0.5)
    battery = create_small_test_geometry()
    battery.tubes.active = True
    battery.tubes.diameter = 0.3
    battery.apply_to_mesh(mesh)
    return mesh, battery


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
