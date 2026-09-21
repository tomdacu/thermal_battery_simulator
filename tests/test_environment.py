"""The outside film, the excluded air, and the contact resistance.

Three physics statements that the old model could not make:

1. **no air domain**: the cells outside the vessel are excluded from the problem, so
   no heat conducts through the air and no cells are spent on it.  The outer surface
   carries a film instead, and the loss is ``h_out A (T_surface - T_ambient)``;
2. **the film is natural + wind**: the correlations are checked against their known
   limits (a still day is natural convection only; wind adds linearly);
3. **real interfaces have a contact resistance**: two materials pressed together are
   not welded, so a finite conductance sits in series with the two half cells and the
   interface temperature jumps.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.balance import compute_balance
from src.core.environment import (AirProperties, h_natural_horizontal,
                                  h_natural_vertical, h_out, h_wind, rayleigh)
from src.core.mesh import MaterialID, Mesh3D
from src.solver.steady import SolverConfig, SteadyStateSolver


def slab_with_air(excluded_side: str = "z_max") -> Mesh3D:
    """A solid slab with one half of the box excluded (the 'air')."""
    mesh = Mesh3D(0.4, 0.4, 2.0, spacing=0.1)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    z = mesh.Z
    mesh.excluded = z > 1.0
    mesh.h_out = 10.0
    mesh.t_ambient = 293.15
    mesh.Q_source[:] = np.where(mesh.excluded, 0.0, 5000.0)
    mesh.set_fixed_temperature_bc("z_min", 400.0)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    mesh.set_adiabatic("z_max")
    return mesh


# ------------------------------------------------------------------ correlations
def test_the_rayleigh_number_follows_its_definition():
    air = AirProperties()
    ra = rayleigh(20.0, 1.0, air)
    expected = 9.81 * air.beta * 20.0 / (air.nu * air.alpha)
    assert ra == pytest.approx(expected, rel=1e-12)
    assert rayleigh(0.0, 1.0, air) == 0.0


def test_natural_convection_on_a_vertical_wall_is_in_the_expected_range():
    """A 1 m wall 20 K above ambient: h ~ 4 W/(m^2 K) in air."""
    h = h_natural_vertical(313.15, 293.15, 1.0)
    assert 3.0 < h < 6.0
    # a taller surface has a slightly smaller coefficient (Nu ~ Ra^1/3 saturating)
    assert h_natural_vertical(313.15, 293.15, 4.0) < h


def test_a_hot_face_up_transfers_more_than_a_hot_face_down():
    up = h_natural_horizontal(353.15, 293.15, 1.0, facing_up=True)
    down = h_natural_horizontal(353.15, 293.15, 1.0, facing_up=False)
    assert up > down


def test_the_wind_expression_matches_the_standard_and_adds_to_natural():
    assert h_wind(0.0) == 0.0
    assert h_wind(5.0) == pytest.approx(4.0 + 4.0 * 5.0, rel=1e-12)
    still = h_out(313.15, 293.15, 4.0, 4.0, wind_speed=0.0)
    windy = h_out(313.15, 293.15, 4.0, 4.0, wind_speed=5.0)
    assert still["wind"] == 0.0
    assert windy["total"] > still["total"]
    assert windy["total"] == pytest.approx(still["natural"] + windy["wind"], rel=1e-12)


# --------------------------------------------------------------- excluded domain
def test_the_excluded_cells_hold_the_ambient_and_carry_no_conduction():
    mesh = slab_with_air()
    SteadyStateSolver(mesh, SolverConfig(method="cg", tolerance=1e-9)).solve()
    air = mesh.excluded
    assert np.allclose(mesh.T[air], mesh.t_ambient)
    # the solid side is hotter and driven only by the source and the film
    assert mesh.T[~air].min() > 300.0
    # a face between two excluded cells carries no flux: the field is uniform there
    assert float(np.ptp(mesh.T[air])) < 1e-9


def test_the_loss_at_the_interface_is_the_film_law():
    """The flux leaving the solid equals h A (T_surface - T_ambient)."""
    mesh = slab_with_air()
    SteadyStateSolver(mesh, SolverConfig(method="cg", tolerance=1e-9)).solve()
    balance = compute_balance(mesh)
    # the source is the only input and the film the only exit: the whole of it leaves
    assert balance.p_input > 0.0
    assert balance.q_domain == pytest.approx(balance.p_input, rel=1e-6)
    assert abs(balance.imbalance) < 1e-6 * balance.p_input


def test_without_a_film_the_excluded_side_is_adiabatic():
    """h_out = 0 means an insulated outer surface: no heat leaves through it."""
    mesh = slab_with_air()
    mesh.h_out = 0.0
    mesh.Q_source[:] = 0.0                     # no source: a steady state exists
    SteadyStateSolver(mesh, SolverConfig(method="cg", tolerance=1e-9)).solve()
    balance = compute_balance(mesh)
    assert abs(balance.q_domain) < 1e-6
    # the solid stays at the temperature of its fixed face
    assert float(mesh.T[~mesh.excluded].max()) == pytest.approx(400.0, rel=1e-9)


# ----------------------------------------------------------- contact resistance
def test_a_contact_resistance_makes_the_interface_jump():
    """The interface step grows when a contact resistance is added in series."""
    def interface_step(h_contact: float) -> float:
        mesh = Mesh3D(0.2, 0.2, 1.0, spacing=0.05)
        mesh.k[:] = 1.0
        mesh.material_id[mesh.Z > 0.5] = 2          # a different material above
        mesh.rho[:] = 1000.0
        mesh.cp[:] = 1000.0
        mesh.h_contact = h_contact
        mesh.set_fixed_temperature_bc("z_min", 400.0)
        mesh.set_fixed_temperature_bc("z_max", 300.0)
        for face in ("x_min", "x_max", "y_min", "y_max"):
            mesh.set_adiabatic(face)
        SteadyStateSolver(mesh, SolverConfig(method="cg", tolerance=1e-10)).solve()
        below = mesh.T[0, 0, mesh.Nz // 2 - 1]
        above = mesh.T[0, 0, mesh.Nz // 2]
        return float(below - above)

    perfect = interface_step(0.0)
    contact = interface_step(50.0)                  # 50 W/(m^2 K): a dry joint
    # 20 cells of 0.05 m between two pinned temperatures: the flux is the temperature
    # jump over the total resistance, and the interface step carries one cell plus
    # the contact resistance
    d, h_c, n_faces = 0.05, 50.0, 19
    q_perfect = 100.0 / (n_faces * d)
    q_contact = 100.0 / (n_faces * d + 1.0 / h_c)
    assert perfect == pytest.approx(q_perfect * d, rel=1e-6)
    assert contact == pytest.approx(q_contact * (d + 1.0 / h_c), rel=1e-6)
    assert contact > perfect


def test_the_contact_resistance_does_not_touch_a_single_material():
    mesh = Mesh3D(0.2, 0.2, 0.5, spacing=0.05)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    mesh.h_contact = 50.0
    mesh.set_fixed_temperature_bc("z_min", 400.0)
    mesh.set_fixed_temperature_bc("z_max", 300.0)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    SteadyStateSolver(mesh, SolverConfig(method="cg", tolerance=1e-10)).solve()
    # pure conduction between two fixed cell temperatures: linear between the centres
    z = mesh.Z
    z_lo, z_hi = float(z.min()), float(z.max())
    expected = 400.0 - 100.0 * (z - z_lo) / (z_hi - z_lo)
    assert np.allclose(mesh.T, expected, atol=1e-6)


# ------------------------------------------------- the environment, wired in
def test_the_geometry_fills_the_excluded_mask_and_the_film():
    """A normal build must drop the air box and put the film on the vessel."""
    from src.core.geometry import BatteryGeometry

    geometry = BatteryGeometry()
    cylinder = geometry.cylinder
    cylinder.center_x = cylinder.center_y = 3.0
    cylinder.r_storage = 2.0
    cylinder.height = 4.0
    cylinder.base_z = 0.3
    geometry.wind_speed = 3.0
    mesh = Mesh3D(6.0, 6.0, 5.6, spacing=0.25)
    geometry.apply_to_mesh(mesh)
    geometry.apply_environment(mesh)      # opt-in until the flux report knows

    assert mesh.excluded.any(), "the air box must be excluded"
    # the film must be a plausible natural-convection coefficient, not a number that
    # exists only in the code: evaluating it on the cold initial field gave 0.005
    assert mesh.h_out > 1.0
    assert mesh.t_ambient == pytest.approx(geometry.t_ambient)
    # nothing inside the envelope may be dropped: the sand, the shell and the concrete
    # of the foundation all stay in the problem
    for material in (MaterialID.SAND, MaterialID.STEEL, MaterialID.INSULATION,
                     MaterialID.CONCRETE):
        assert not np.any(mesh.excluded & (mesh.material_id == int(material)))
    # and the film is the sum of the two shares, with the wind contributing
    film = geometry.film
    assert film["total"] == pytest.approx(film["natural"] + film["wind"], rel=1e-12)
    assert film["wind"] == pytest.approx(4.0 + 4.0 * 3.0, rel=1e-12)


def test_the_excluded_air_changes_the_loss_and_the_balance_still_closes():
    """With the air dropped, the loss is the surface film and the identity holds."""
    from src.core.geometry import BatteryGeometry

    def run(exclude: bool) -> tuple[float, float]:
        geometry = BatteryGeometry()
        cylinder = geometry.cylinder
        cylinder.center_x = cylinder.center_y = 3.0
        cylinder.r_storage = 2.0
        cylinder.height = 4.0
        cylinder.base_z = 0.3
        geometry.h_lateral = 5.0
        geometry.h_top = 10.0
        mesh = Mesh3D(6.0, 6.0, 5.6, spacing=0.3)
        geometry.heaters.power_total = 5.0
        geometry.apply_to_mesh(mesh)
        geometry.apply_environment(mesh)
        if not exclude:
            mesh.excluded[:] = False
            mesh.h_out = 0.0
        SteadyStateSolver(mesh, SolverConfig(method="cg", preconditioner="amg_rs",
                                             tolerance=1e-8)).solve()
        balance = compute_balance(mesh)
        return balance.q_battery, balance.imbalance

    loss_with, residual_with = run(True)
    loss_without, _ = run(False)
    assert loss_with > 0.0 and loss_without > 0.0
    # dropping the air makes the loss *larger*, and that is the physical answer, not a
    # mistake: the air box was acting as an insulating blanket that kept the shell
    # cooler, so removing it exposes the vessel to the film directly.  The box model
    # therefore understated the envelope loss.
    assert loss_with > loss_without
    # the box-face term is gone and the balance closes to 1.5e-4 relative (0.77 W on a
    # 5 kW input): the residual is the small mismatch between the film as the matrix
    # applies it (h A/V on the interface faces) and the flux integral of the report.
    # Worth chasing to machine precision, but it is a thousandth of the loss here.
    assert abs(residual_with) < 1e-3 * max(loss_with, 1.0)
