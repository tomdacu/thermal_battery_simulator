"""Graded grid: coefficient exactness, resolution independence, balance closure.

The acceptance criteria of ``docs/10_MESH_AND_HEATERS.md``: a graded grid must
reproduce the uniform results when asked for a uniform grid, must solve the
analytic slab exactly, must not change the answer when every target is halved,
and must keep the energy balance closing.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.balance import compute_balance
from src.core.geometry import create_small_test_geometry
from src.core.mesh import Mesh3D
from src.core.refinement import Band, GridSpec
from src.solver.steady import SolverConfig, SteadyStateSolver


def slab(k: float = 1.0, length: float = 1.0) -> GridSpec:
    """1 m column, fine near the bottom, coarse at the top."""
    return GridSpec(x=(Band(0, 1, 0.25),), y=(Band(0, 1, 0.25),),
                    z=(Band(0, 0.25, 0.05), Band(0, 1, 0.25)))


def test_uniform_gridspec_reproduces_the_legacy_uniform_mesh():
    """A spec asking for one size everywhere must give the same grid and answer.

    The box divides exactly by the requested size, so the legacy mesh has nothing
    to snap and the two grids differ only in the code path.
    """
    grid = GridSpec(x=(Band(0, 6, 0.5),), y=(Band(0, 6, 0.5),), z=(Band(0, 5.5, 0.5),))
    graded = Mesh3D(6.0, 6.0, 5.5, grid=grid)
    uniform = Mesh3D(6.0, 6.0, 5.5, spacing=0.5)
    assert (graded.Nx, graded.Ny, graded.Nz) == (uniform.Nx, uniform.Ny, uniform.Nz)
    assert np.allclose(graded.dx, uniform.dx, rtol=1e-12)
    assert np.allclose(graded.edges_z, uniform.edges_z, rtol=1e-12)

    for mesh in (graded, uniform):
        create_small_test_geometry().apply_to_mesh(mesh)
        SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
    assert np.abs(graded.T - uniform.T).max() < 1e-9


def test_a_linear_profile_is_exact_on_a_graded_grid():
    """Constant conductivity and Dirichlet ends: the FV solution is exact anywhere."""
    mesh = Mesh3D(1.0, 1.0, 1.0, grid=slab())
    mesh.k[:] = 0.7
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    mesh.set_fixed_temperature_bc("z_min", 300.0)
    mesh.set_fixed_temperature_bc("z_max", 500.0)
    assert mesh.Nx * mesh.Ny * mesh.Nz > 0 and not mesh.uniform

    SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
    # the boundary *cells* are pinned, not the box faces, so the exact discrete
    # solution is the line through the first and the last cell centre
    exact = 300.0 + 200.0 * (mesh.Z - mesh.z[0]) / (mesh.z[-1] - mesh.z[0])
    assert np.abs(mesh.T - exact).max() < 1e-9


def test_the_interface_flux_matches_the_series_resistance():
    """Two cells of very different size: the face conductance is the series one."""
    # growth 10 makes the walk jump straight from the fine band to the coarse one
    grid = GridSpec(x=(Band(0, 1, 0.5),), y=(Band(0, 1, 0.5),),
                    z=(Band(0, 0.2, 0.2), Band(0, 1.0, 0.8)), growth=10.0)
    mesh = Mesh3D(1.0, 1.0, 1.0, grid=grid)
    h = mesh.dz
    assert h.size == 2 and h[1] > 3.0 * h[0], h      # the grid really is graded
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    mesh.set_fixed_temperature_bc("z_min", 300.0)    # closes the energy balance
    flux = 25.0                                      # W enters through the top
    mesh.set_heat_flux_bc("z_max", flux)
    SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
    # across the interface between two cells of different size the temperature drop
    # is the series resistance of the two half cells
    resistance = (h[0] / 2 + h[1] / 2) / float(mesh.k[0, 0, 0])
    drop = float(mesh.T[0, 0, 1] - mesh.T[0, 0, 0])   # heat enters from the top
    assert drop == pytest.approx(flux * resistance, rel=1e-9)


def _storage_grid(fine: float, background: float) -> GridSpec:
    """Refinement targets of the test battery: fine storage + shell, coarse far field."""
    cyl = create_small_test_geometry().cylinder
    return GridSpec(
        x=(Band(0, 6, background),
           Band(cyl.center_x - cyl.r_shell - 0.3, cyl.center_x + cyl.r_shell + 0.3, fine)),
        y=(Band(0, 6, background),
           Band(cyl.center_y - cyl.r_shell - 0.3, cyl.center_y + cyl.r_shell + 0.3, fine)),
        z=(Band(cyl.z_storage_start, cyl.z_storage_end, fine), Band(0, 5.6, background)))


def test_a_graded_mesh_reproduces_the_uniform_loss():
    """The heat leaving the domain is a property of the model, not of the grid.

    Checked against a *uniform* mesh with the same fine size: the graded grid puts
    the same cells only where the gradients are, so the two must agree on a global
    flux.  (The stored energy is a much sharper quantity on this 1:4 scale test
    model, where the near-wall gradient dominates, and is not used here.)
    """
    losses = {}
    for label, mesh in (
            ("graded", Mesh3D(6.0, 6.0, 5.6, grid=_storage_grid(0.15, 0.35))),
            ("uniform", Mesh3D(6.0, 6.0, 5.6, spacing=0.15))):
        geom = create_small_test_geometry()
        geom.heaters.power_total = 5.0
        geom.apply_to_mesh(mesh)
        SteadyStateSolver(mesh, SolverConfig(method="cg", preconditioner="amg_rs")).solve()
        balance = compute_balance(mesh)
        losses[label] = balance.q_domain
    assert losses["graded"] == pytest.approx(losses["uniform"], rel=0.01)


def test_the_energy_balance_closes_on_a_graded_mesh():
    """The assembled operator and the flux integrals must agree cell by cell."""
    mesh = Mesh3D(6.0, 6.0, 5.6, grid=_storage_grid(0.2, 0.4))
    geom = create_small_test_geometry()
    geom.heaters.power_total = 5.0
    geom.apply_to_mesh(mesh)
    SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
    balance = compute_balance(mesh)
    assert abs(balance.imbalance) < 1e-9 * balance.p_input

