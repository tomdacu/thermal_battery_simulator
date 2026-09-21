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
def paint_pair() -> Callable[[BatteryGeometry | None], tuple[AdaptiveMesh, Mesh3D]]:
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
        tree = AdaptiveMesh.uniform(FINEST, SPACING, level=0)
        structured = Mesh3D(Lx=BOX, Ly=BOX, Lz=BOX, spacing=SPACING)
        battery.apply_to_mesh(tree)
        battery.apply_to_mesh(structured)
        return tree, structured

    return paint


@pytest.fixture
def tree_model(paint_pair) -> AdaptiveMesh:
    """The shared test geometry on a uniform tree: 16^3 leaves of 0.5 m in an 8 m box."""
    return paint_pair()[0]


def relative_residual(matrix, b, x) -> float:
    r = matrix @ x - b
    return float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-30))
