"""Shared fixtures: small but physically complete models."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.geometry import create_small_test_geometry
from src.core.mesh import Mesh3D


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


def relative_residual(matrix, b, x) -> float:
    r = matrix @ x - b
    return float(np.linalg.norm(r) / max(np.linalg.norm(b), 1e-30))
