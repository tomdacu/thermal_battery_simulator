"""Shared fixtures: small but physically complete models."""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import BatteryGeometry, create_small_test_geometry
from src.core.mesh import Mesh3D

#: The geometry fixtures paint one model on two meshes, and the two need a box they can
#: share: an octree spans a *cube* with a power-of-two number of cells per side, while
#: ``storage_model`` below is a 6 x 6 x 5.6 m box the grid snaps to 5.5 m.  The pair is
#: therefore an 8 m cube at the same 0.5 m spacing (16 leaves a side), carrying the same
#: test geometry.
BOX = 8.0
SPACING = 0.5
FINEST = 16                              # finest cells per side: BOX / SPACING


@pytest.fixture
def slab() -> Mesh3D:
    """1 m tall column: 1D conduction with adiabatic sides, ground at the bottom."""
    mesh = Mesh3D(Lx=0.5, Ly=0.5, Lz=1.0, spacing=0.25)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    for face in ("x_min", "x_max", "y_min", "y_max", "z_max"):
        mesh.set_adiabatic(face)
    mesh.set_fixed_temperature_bc("z_min", 300.0)
    mesh.set_fixed_temperature_bc("z_max", 400.0)
    return mesh


@pytest.fixture
def adiabatic_box() -> Mesh3D:
    """Closed box with uniform properties: pure storage + source."""
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=0.25)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        mesh.set_adiabatic(face)
    return mesh


@pytest.fixture
def storage_model() -> Mesh3D:
    """Complete battery built from the shared test geometry."""
    mesh = Mesh3D(Lx=6.0, Ly=6.0, Lz=5.6, spacing=0.5)
    create_small_test_geometry().apply_to_mesh(mesh)
    return mesh


@pytest.fixture
def box_pair() -> Callable[[], tuple[AdaptiveMesh, Mesh3D]]:
    """``() -> (tree, structured)``: a *fresh* empty pair of ``BOX``, one per call.

    A tree spans a cube with a power-of-two number of cells a side, so the pair has to be
    a cube the structured mesh can meet exactly: ``BOX`` at ``SPACING``.  The painters of
    the migration need the two meshes *before* anything is written on them (a pipe
    network, a heater bank, the battery model), and a test that paints, solves and then
    paints again needs two pairs - the same reason ``paint_pair`` is a factory, a used
    mesh is not the same mesh.
    """
    def pair() -> tuple[AdaptiveMesh, Mesh3D]:
        return (AdaptiveMesh.uniform(FINEST, SPACING, level=0),
                Mesh3D(Lx=BOX, Ly=BOX, Lz=BOX, spacing=SPACING))

    return pair


@pytest.fixture
def paint_pair(box_pair) -> Callable[[BatteryGeometry | None],
                                     tuple[AdaptiveMesh, Mesh3D]]:
    """``(geometry) -> (tree, structured)``: one model painted on both meshes.

    The same geometry is painted twice - once on a uniform
    :class:`~src.core.adaptive_mesh.AdaptiveMesh` and once on the
    :class:`~src.core.mesh.Mesh3D` of the same cells (``BOX``, ``SPACING``) - so a test
    can compare the result leaf by leaf.  Steps 2-6 of the migration need the same pair
    (a battery model on a tree beside its structured twin), which is why the factory
    lives here and not in one test module.
    """
    def paint(geometry: BatteryGeometry | None = None) -> tuple[AdaptiveMesh, Mesh3D]:
        battery = create_small_test_geometry() if geometry is None else geometry
        tree, structured = box_pair()
        battery.apply_to_mesh(tree)
        battery.apply_to_mesh(structured)
        return tree, structured

    return paint


@pytest.fixture
def tree_model(paint_pair) -> AdaptiveMesh:
    """The shared test geometry on a uniform tree: 16^3 leaves of 0.5 m in an 8 m box."""
    return paint_pair()[0]


@pytest.fixture
def structured_twin() -> Callable[[AdaptiveMesh], tuple[Mesh3D, np.ndarray]]:
    """``(tree) -> (Mesh3D, leaves)``: the structured twin of a *uniform* tree.

    The same cells carrying the same per-cell state and the same face conditions,
    *copied* rather than re-painted - the recipe of ``tests/test_adaptive_mesh.py`` and
    ``tests/test_steady_on_tree.py`` - so a transient march, or a whole cycle, run on the
    tree beside the same run on ``Mesh3D`` measures the two assemblies and nothing else.
    ``leaves`` is the flat ``Mesh3D`` cell of every leaf: it reads a structured field back
    as a per-leaf vector and re-indexes a pipe run.  A refined tree has no cell-by-cell
    twin, and a test that needs one compares against the analytic solution or against the
    tree's own steady field instead.
    """
    #: the per-cell state the two meshes share - the protocol's fields, one value per leaf
    shared = ("T", "k", "rho", "cp", "Q_source", "Q_sink", "material_id", "boundary_type",
              "bc_h", "bc_T_inf", "excluded", "source_mask")

    def twin(tree: AdaptiveMesh) -> tuple[Mesh3D, np.ndarray]:
        levels = {leaf.level for leaf in tree.tree.leaves}
        assert len(levels) == 1, "a cell-by-cell twin needs a uniform tree"
        edge = float(tree.physical_size * (1 << levels.pop()))
        side = tree.box_size
        structured = Mesh3D(Lx=side, Ly=side, Lz=side, spacing=edge)
        index = tuple(np.array(axis) for axis in
                      zip(*[structured.find_cell(*tree.centre(position))
                            for position in range(tree.n_cells)], strict=True))
        flat = np.array([structured.ijk_to_linear(i, j, k)
                         for i, j, k in zip(*index, strict=True)])
        for name in shared:
            getattr(structured, name)[index] = getattr(tree, name)
        structured.h_out = tree.h_out
        structured.t_ambient = tree.t_ambient
        structured.h_contact = tree.h_contact
        structured.h_out_conv = tree.h_out_conv
        structured.environment_emissivity = tree.environment_emissivity
        structured.face_bc = dict(tree.face_bc)
        return structured, flat

    return twin


def relative_residual(matrix, b, x) -> float:
    r = matrix @ x - b
    return float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-30))
