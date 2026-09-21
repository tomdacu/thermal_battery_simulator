"""Shared rendering helpers (PyVista grid construction, colour tables).

The GUI and any script build their scene from the same :func:`to_image_data`,
so a script and the interactive view can never disagree on orientation, units or
colour mapping.  Either mesh renders: a structured grid becomes the ``ImageData`` or
``RectilinearGrid`` it always was, a tree becomes an ``UnstructuredGrid`` of its leaves,
and the report vocabulary a tree cannot answer (``grid_summary``, ``size_label``,
``Nx x Ny x Nz``) is replaced here, once, by :func:`grid_lines` - the presentation side
of the contract that ``src/core/mesh_api.py`` deliberately leaves out.
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np

from ..core.geometry import HeaterPattern
from ..core.mesh import MaterialID, Mesh3D

if TYPE_CHECKING:                      # the tree is the target, not a runtime dependency
    from ..core.adaptive_mesh import AdaptiveMesh

MATERIAL_NAMES: dict[int, str] = {
    int(MaterialID.AIR): "Air",
    int(MaterialID.SAND): "Sand (storage)",
    int(MaterialID.INSULATION): "Insulation",
    int(MaterialID.STEEL): "Steel",
    int(MaterialID.TUBES): "Tubes",
    int(MaterialID.HEATERS): "Heaters",
    int(MaterialID.GROUND): "Ground",
    int(MaterialID.CONCRETE): "Concrete",
}

MATERIAL_COLORS: dict[int, tuple[float, float, float]] = {
    int(MaterialID.AIR): (0.85, 0.90, 1.00),
    int(MaterialID.SAND): (0.93, 0.79, 0.36),
    int(MaterialID.INSULATION): (0.85, 0.60, 0.30),
    int(MaterialID.STEEL): (0.60, 0.62, 0.68),
    int(MaterialID.TUBES): (0.20, 0.55, 0.85),
    int(MaterialID.HEATERS): (0.90, 0.20, 0.15),
    int(MaterialID.GROUND): (0.42, 0.32, 0.22),
    int(MaterialID.CONCRETE): (0.70, 0.70, 0.68),
}

FIELD_ARRAYS = ("Temperature", "Material", "Sources", "Conductivity")

FIELD_UNITS = {"Temperature": "°C", "Sources": "W/m³", "Conductivity": "W/(m·K)"}

AXIS_INDEX = {"x": (0, (1.0, 0.0, 0.0)), "y": (1, (0.0, 1.0, 0.0)), "z": (2, (0.0, 0.0, 1.0))}

#: the extension PyVista picks the writer from, per grid type it builds here
GRID_SUFFIX = {"ImageData": ".vti", "RectilinearGrid": ".vtr", "UnstructuredGrid": ".vtu"}


def domain_extent(mesh: Mesh3D | AdaptiveMesh) -> tuple[float, float, float]:
    """Edge lengths of the domain [m], on either mesh.

    A structured mesh has three of them; a tree spans a cube whose edge is ``box_size``
    (``physical_size`` times the finest cells a side), which is the ``Lx = Ly = Lz`` the
    view and the clip plane are written against.
    """
    box = getattr(mesh, "box_size", None)
    if box is not None:
        return (float(box),) * 3
    return (float(mesh.Lx), float(mesh.Ly), float(mesh.Lz))


def grid_lines(mesh: Mesh3D | AdaptiveMesh) -> list[str]:
    """The mesh block of the reports: one line per fact, in the mesh's own vocabulary.

    A structured mesh answers with the numbers it always reported (cells per axis, the
    realised cell size, the box); a tree answers with what it has instead - its leaves,
    its level histogram and its leaf edges, which is ``AdaptiveMesh.summary``.  The
    report vocabulary is deliberately not part of ``MeshAPI``: the two roads differ here
    and the difference is stated, not hidden behind a grid that pretends to be uniform.
    """
    lx, ly, lz = domain_extent(mesh)
    report = getattr(mesh, "summary", None)
    if report is not None:                       # a tree reports itself
        return [report(), f"domain      {lx:.3f} x {ly:.3f} x {lz:.3f} m"]
    return [f"grid        {mesh.Nx} x {mesh.Ny} x {mesh.Nz} = {mesh.N_total:,} cells",
            f"cell size   {mesh.size_label()}",
            f"domain      {lx:.3f} x {ly:.3f} x {lz:.3f} m"]


def material_cmap() -> list:
    """Material colours as hex strings.

    PyVista accepts a *list of strings* for ``cmap``; a list of RGB tuples raises
    ``TypeError: When inputting a list as a cmap, each item should be a string``.
    """
    return ["#{:02x}{:02x}{:02x}".format(*(int(round(255 * c))
                                          for c in MATERIAL_COLORS[index]))
            for index in sorted(MATERIAL_COLORS)]


def clip_grid(mesh: Mesh3D | AdaptiveMesh, field: str, axis: str = "z", fraction: float = 0.5):
    """Grid of ``field`` clipped at ``fraction`` of ``axis`` (or unclipped)."""
    grid = to_image_data(mesh, field)
    if axis is None:
        return grid
    index, normal = AXIS_INDEX[axis]
    length = domain_extent(mesh)[index]
    position = min(max(fraction, 0.02), 0.98) * length
    origin = [0.0, 0.0, 0.0]
    origin[index] = position
    return grid.clip(normal=normal, origin=origin, invert=False)


def add_field(plotter, mesh: Mesh3D | AdaptiveMesh, field: str = "Temperature",
              cmap: str = "coolwarm", opacity: float = 0.8, axis: str = None,
              fraction: float = 0.5, show_colorbar: bool = True):
    """Add one scalar field to any PyVista plotter; returns the actor, or ``None``.

    ``None`` means the clip plane left nothing to draw (for instance the material
    view above the battery) - PyVista refuses to plot an empty mesh, so the
    caller is expected to show a message instead of an actor.
    Shared by the interactive view and by scripts/tests, so a scene built
    off-screen and one built by the GUI cannot diverge.
    """
    grid = clip_grid(mesh, field, axis, fraction)
    if grid.n_cells == 0:
        return None
    if field == "Material":
        grid = grid.threshold(0.5, scalars="Material")
        if grid.n_cells == 0:
            return None
        return plotter.add_mesh(grid, cmap=material_cmap(), clim=[0, mesh.material_id.max()],
                                show_scalar_bar=False, opacity=opacity)
    low, high = color_limits(mesh, field)
    scalar_args = None
    if show_colorbar:
        unit = FIELD_UNITS.get(field, "")
        scalar_args = {"title": f"{field} [{unit}]" if unit else field,
                       "position_x": 0.62, "position_y": 0.02,
                       "width": 0.36, "height": 0.07, "label_font_size": 10,
                       "n_labels": 5, "fmt": "%.1f" if abs(high - low) < 10 else "%.0f"}
    return plotter.add_mesh(grid, cmap=cmap, clim=[low, high], opacity=opacity,
                            scalar_bar_args=scalar_args)


def add_material_legend(plotter, loc: str = "upper left", size=(0.17, 0.34)):
    """Colour legend of the material ids (hex strings, as PyVista expects)."""
    labels = [(MATERIAL_NAMES.get(index, str(index)), MATERIAL_COLORS[index])
              for index in sorted(MATERIAL_COLORS)]
    return plotter.add_legend(labels=labels, bcolor="white", size=size, loc=loc)


def _clip_dataset(dataset, clip):
    """Apply the cutting plane of the view controls to a preview primitive."""
    if clip is None:
        return dataset
    axis, position = clip
    index, normal = AXIS_INDEX[axis]
    origin = [0.0, 0.0, 0.0]
    origin[index] = position
    return dataset.clip(normal=normal, origin=origin, invert=False)


def _disk(plotter, cx, cy, z0, z1, radius, colour, opacity, clip=None, opacity_scale=1.0):
    import pyvista as pv

    if z1 <= z0 or radius <= 0:
        return
    dataset = pv.Cylinder(center=(cx, cy, (z0 + z1) / 2), direction=(0, 0, 1),
                          radius=radius, height=z1 - z0, resolution=64)
    dataset = _clip_dataset(dataset, clip)
    if dataset.n_points == 0:
        return
    plotter.add_mesh(dataset, color=colour, opacity=min(opacity * opacity_scale, 1.0))


def add_geometry_preview(plotter, battery, mesh: Mesh3D | AdaptiveMesh = None, clip=None,
                         opacity_scale: float = 1.0):
    """Schematic view of the configured battery, zone by zone.

    Every part is drawn between the same elevations ``apply_to_mesh`` uses, so the
    picture is a faithful section of the model: skipping the top slab and the
    steel plate used to leave a visible gap under the roof.  ``clip=(axis_name,
    position [m])`` and ``opacity_scale`` apply the same view controls as the field
    view.
    """
    import pyvista as pv

    cyl = battery.cylinder
    cx, cy = cyl.center_x, cyl.center_y
    sand = MATERIAL_COLORS[int(MaterialID.SAND)]
    insul = MATERIAL_COLORS[int(MaterialID.INSULATION)]
    steel = MATERIAL_COLORS[int(MaterialID.STEEL)]
    scale = opacity_scale

    # storage and the two insulation slabs (r < r_storage)
    _disk(plotter, cx, cy, cyl.z_storage_start, cyl.z_storage_end, cyl.r_storage, sand,
          0.6, clip, scale)
    _disk(plotter, cx, cy, cyl.z_slab_bottom_start, cyl.z_storage_start, cyl.r_storage,
          insul, 0.7, clip, scale)
    _disk(plotter, cx, cy, cyl.z_slab_top_start, cyl.z_slab_top_end, cyl.r_storage,
          insul, 0.7, clip, scale)
    _disk(plotter, cx, cy, cyl.base_z, cyl.z_shell_top, cyl.r_insulation, insul, 0.25,
          clip, scale)
    _disk(plotter, cx, cy, cyl.base_z, cyl.z_shell_top, cyl.r_shell, steel, 0.15, clip, scale)
    _disk(plotter, cx, cy, 0.0, cyl.base_z, cyl.r_shell + cyl.foundation_margin,
          MATERIAL_COLORS[int(MaterialID.CONCRETE)], 0.35, clip, scale)
    if cyl.steel_slab_top > 0:
        _disk(plotter, cx, cy, cyl.z_slab_top_end, cyl.steel_slab_top + cyl.z_slab_top_end,
              cyl.r_insulation, steel, 0.5, clip, scale)
    if cyl.roof_height > 0:
        cone = pv.Cone(center=(cx, cy, cyl.z_cone_base + cyl.roof_height / 2),
                       direction=(0, 0, 1), radius=cyl.r_shell, height=cyl.roof_height,
                       resolution=64)
        cone = _clip_dataset(cone, clip)
        if cone.n_points:
            if cyl.fill_cone_with_sand:
                plotter.add_mesh(cone, color=sand, opacity=min(0.4 * scale, 1.0),
                                 label="sand fill")
            plotter.add_mesh(cone, color=steel, opacity=min(0.35 * scale, 1.0),
                             label="roof")

    if battery.tubes.active:
        for element in battery.tubes.generate_positions(cx, cy, cyl.r_storage * 0.9,
                                                        cyl.z_storage_start,
                                                        cyl.z_storage_end):
            _disk(plotter, element.x, element.y, element.z_bottom, element.z_top,
                  max(element.radius, 0.01), MATERIAL_COLORS[int(MaterialID.TUBES)], 0.95,
                  clip, scale)
    if battery.heaters.pattern != HeaterPattern.UNIFORM_ZONE:
        _add_heater_bank(plotter, battery, cyl, clip, scale)

    lx, ly, lz = domain_extent(mesh) if mesh else (cyl.r_shell * 2 + 1,
                                                   cyl.r_shell * 2 + 1,
                                                   cyl.z_cone_apex + 0.5)
    bounds = (0, lx, 0, ly, 0, lz)
    plotter.add_mesh(pv.Box(bounds=bounds), style="wireframe", color="grey")
    return plotter


def _add_heater_bank(plotter, battery, cyl, clip, scale) -> None:
    """Hairpin elements: flange, support plate, two legs and the bottom bend."""
    import pyvista as pv

    bank = battery.heaters.bank(cyl.z_storage_start, cyl.z_storage_end)
    colour = MATERIAL_COLORS[int(MaterialID.HEATERS)]
    steel = MATERIAL_COLORS[int(MaterialID.STEEL)]
    elements = bank.generate_elements(cyl.center_x, cyl.center_y, cyl.r_storage * 0.9)
    z_bottom = cyl.z_storage_start + bank.offset_bottom
    z_top = cyl.z_storage_end + bank.flange_offset
    radius = max(bank.sheath_diameter * 0.5, 0.006)

    for element in elements:
        for a, b in element.segments(z_bottom, z_top, z_bottom + bank.active_length):
            if a[2] == b[2]:                        # a bend chord
                points = np.array([a, b], dtype=float)
                tube = pv.Tube(points=points, radius=radius, n_sides=8)
            else:
                tube = pv.Cylinder(center=((a[0] + b[0]) / 2, (a[1] + b[1]) / 2,
                                           (a[2] + b[2]) / 2),
                                   direction=(0, 0, 1), radius=radius,
                                   height=abs(b[2] - a[2]), resolution=10)
            tube = _clip_dataset(tube, clip)
            if tube.n_points:
                plotter.add_mesh(tube, color=colour, opacity=min(0.95 * scale, 1.0))

    # flange on the roof and support plate at the bottom of the sand
    reach = cyl.r_storage * 0.45
    _disk(plotter, cyl.center_x, cyl.center_y, cyl.z_cone_base - 0.02,
          cyl.z_cone_base + 0.02, min(reach * 1.2, cyl.r_storage * 0.9), steel, 0.8,
          clip, scale)
    _disk(plotter, cyl.center_x, cyl.center_y, z_bottom - 0.01, z_bottom + 0.01,
          min(reach * 1.2, cyl.r_storage * 0.9), steel, 0.6, clip, scale)


def field_values(mesh: Mesh3D | AdaptiveMesh, field: str) -> np.ndarray:
    """Cell array, in the mesh's own cell order, for one of the displayable fields.

    ``Mesh3D`` keeps a 3-D array and a tree a flat one; the caller flattens either with
    the mesh's own order, which is what every per-cell field of the protocol is in.
    """
    if field == "Temperature":
        return mesh.T - 273.15            # displayed in degC
    if field == "Material":
        return mesh.material_id.astype(float)
    if field == "Sources":
        return mesh.Q_source.copy()
    if field == "Conductivity":
        return mesh.k.copy()
    raise ValueError(f"unknown field {field!r}; expected one of {FIELD_ARRAYS}")


def _leaf_grid(mesh: AdaptiveMesh, pv):
    """``UnstructuredGrid`` with one hexahedron per leaf, in leaf order.

    A leaf is a cube, so its eight corners come from its centre and its volume, and the
    cell order is the leaf order the per-leaf fields are in: the array handed to VTK is
    the array the solver wrote, cell for cell.  The corners follow the VTK hexahedron
    order (the lower face counter-clockwise seen from above, then the upper one).
    """
    centres = np.asarray(mesh.centres(), dtype=float)
    half = np.cbrt(np.asarray(mesh.V, dtype=float)) / 2.0
    corners = np.array([[-1.0, -1.0, -1.0], [1.0, -1.0, -1.0],
                        [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
                        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0],
                        [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
    points = (centres[:, None, :] + half[:, None, None] * corners[None, :, :]).reshape(-1, 3)
    order = np.arange(8 * centres.shape[0], dtype=np.int64).reshape(-1, 8)
    cells = np.hstack([np.full((centres.shape[0], 1), 8, dtype=np.int64), order]).ravel()
    types = np.full(centres.shape[0], int(pv.CellType.HEXAHEDRON), dtype=np.uint8)
    return pv.UnstructuredGrid(cells, types, points)


def to_image_data(mesh: Mesh3D | AdaptiveMesh, field: str = "Temperature"):
    """Build the PyVista grid of the mesh (cell data in the mesh's own cell order).

    A graded mesh becomes a ``RectilinearGrid`` carrying the per-axis edges and a
    uniform mesh stays an ``ImageData`` so the exported file keeps its ``.vti``
    extension; a tree has no axes to describe it, so it becomes an ``UnstructuredGrid``
    of its leaves - the leaves *are* the cells, and the step between two levels is drawn
    as the staircase it is, face against face, never interpolated.
    """
    import pyvista as pv

    if hasattr(mesh, "faces"):                    # a tree: one hexahedron per leaf
        grid = _leaf_grid(mesh, pv)
    elif mesh.uniform:
        grid = pv.ImageData(dimensions=(mesh.Nx + 1, mesh.Ny + 1, mesh.Nz + 1),
                            spacing=(float(mesh.dx[0]), float(mesh.dy[0]),
                                     float(mesh.dz[0])), origin=(0.0, 0.0, 0.0))
    else:
        grid = pv.RectilinearGrid(mesh.edges_x, mesh.edges_y, mesh.edges_z)
    grid.cell_data[field] = np.asarray(field_values(mesh, field)).ravel(order="F")
    return grid


def color_limits(mesh: Mesh3D | AdaptiveMesh, field: str = "Temperature") -> tuple[float, float]:
    values = field_values(mesh, field)
    low, high = float(np.min(values)), float(np.max(values))
    return (low, high) if high > low else (low, low + 1.0)


def cell_centers(mesh: Mesh3D | AdaptiveMesh) -> np.ndarray:
    """(N, 3) cell centres in the same order as the flattened fields."""
    centres = getattr(mesh, "centres", None)
    if centres is not None:                       # a tree knows where its leaves are
        return np.asarray(centres(), dtype=float)
    ii, jj, kk = np.meshgrid(mesh.x, mesh.y, mesh.z, indexing="ij")
    return np.column_stack([ii.ravel(order="F"), jj.ravel(order="F"),
                            kk.ravel(order="F")])


def export_vtk(mesh: Mesh3D | AdaptiveMesh, path: str, field: str = "Temperature") -> str:
    """Write the grid of ``mesh``; returns the file that was written.

    PyVista picks the writer from the extension and a grid type has exactly one it can
    be read back from (``.vti``, ``.vtr``, ``.vtu``), so the extension follows the grid
    the mesh produces rather than the file dialog: a tree writes an ``.vtu`` even when
    the caller asked for ``.vti``, instead of failing at the end of a save the user
    asked for.
    """
    grid = to_image_data(mesh, field)
    suffix = GRID_SUFFIX[type(grid).__name__]
    if not path.lower().endswith(suffix):
        path = os.path.splitext(path)[0] + suffix
    grid.save(path)
    return path


def export_csv(path: str, results, mesh: Mesh3D | AdaptiveMesh = None) -> str:
    """Write either a transient time series or the current mesh fields."""
    if mesh is not None and not hasattr(results, "to_arrays"):
        rows = np.column_stack([mesh.T.ravel(order="F"),
                                mesh.material_id.ravel(order="F"),
                                mesh.Q_source.ravel(order="F")])
        np.savetxt(path, rows, delimiter=",",
                   header="T_K,material_id,Q_source_W_m3", comments="")
        return path
    return results.export_csv(path)
