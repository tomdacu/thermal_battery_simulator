"""Solver tests: assembly exactness, analytic regressions, linear backends."""
from __future__ import annotations

import numpy as np
import pytest

from src.core.mesh import BoundaryType, MaterialID, Mesh3D
from src.core.physics import half_cell_h
from src.core.profiles import InitialCondition, PowerProfile, ExtractionProfile
from src.solver.linear import HAS_PYAMG, LinearConfig, is_symmetric, solve_linear
from src.solver.matrix import build_steady_matrix
from src.solver.steady import SolverConfig, SteadyStateSolver
from src.solver.transient import TransientConfig, TransientSolver


def column(mesh, field=None):
    field = mesh.T if field is None else field
    return field[0, 0, :]


# ------------------------------------------------------------------ assembly
def test_interior_stencil_is_the_textbook_7_point_scheme():
    """Interior node of a fully insulated box: six equal couplings, diagonal 6a."""
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=0.25)
    mesh.k[:] = 1.0
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        mesh.set_adiabatic(face)
    matrix, _ = build_steady_matrix(mesh)
    dense = matrix.toarray()
    a = 1.0 / mesh.d ** 2
    node = mesh.ijk_to_linear(1, 1, 1)
    assert dense[node, node] == pytest.approx(6 * a)
    for offset in (1, -1, mesh.Nx, -mesh.Nx, mesh.Nx * mesh.Ny, -mesh.Nx * mesh.Ny):
        assert dense[node, node + offset] == pytest.approx(-a)


def test_matrix_matches_an_independent_assembly(slab):
    """Cross-check every row against a directly written, unvectorised reference."""
    matrix, rhs = build_steady_matrix(slab)
    mesh = slab
    nx, ny, nz, d = mesh.Nx, mesh.Ny, mesh.Nz, mesh.d
    reference = np.zeros((mesh.N_total, mesh.N_total))
    reference_rhs = np.zeros(mesh.N_total)

    def index(i, j, k):
        return i + j * nx + k * nx * ny

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                row = index(i, j, k)
                bc = mesh.face_bc
                pinned = ((k == 0 and bc["z_min"].kind == BoundaryType.DIRICHLET)
                          or (k == nz - 1 and bc["z_max"].kind == BoundaryType.DIRICHLET))
                if pinned:
                    value = bc["z_min"].value if k == 0 else bc["z_max"].value
                    reference[row, row] = 1.0
                    reference_rhs[row] = value
                    continue
                diag = 0.0
                for di, dj, dk, face, exposed in (
                        (-1, 0, 0, "x_min", i == 0), (1, 0, 0, "x_max", i == nx - 1),
                        (0, -1, 0, "y_min", j == 0), (0, 1, 0, "y_max", j == ny - 1),
                        (0, 0, -1, "z_min", k == 0), (0, 0, 1, "z_max", k == nz - 1)):
                    if exposed and bc[face].kind == BoundaryType.INTERNAL:
                        continue
                    if exposed and bc[face].kind == BoundaryType.DIRICHLET:
                        continue
                    if exposed:
                        h = bc[face].h
                        a_conv = half_cell_h(mesh.k[i, j, k], h, d) / d
                        diag += a_conv
                        reference_rhs[row] += a_conv * bc[face].value
                        continue
                    a_face = mesh.k[i, j, k] / d ** 2
                    diag += a_face
                    reference[row, index(i + di, j + dj, k + dk)] = -a_face
                reference[row, row] = diag
    # mirror the symmetric elimination: zero the Dirichlet columns, move them to b
    for row in range(mesh.N_total):
        for col in range(mesh.N_total):
            if reference[col, col] == 1.0 and row != col and reference[row, col] != 0:
                reference_rhs[row] -= reference[row, col] * reference_rhs[col]
                reference[row, col] = 0.0
    assert np.abs(reference - matrix.toarray()).max() < 1e-9
    assert np.abs(reference_rhs - rhs).max() < 1e-9


def test_steady_solution_is_exact_for_dirichlet_and_source(slab):
    """Uniform k with two Dirichlet faces: the parabola is reproduced exactly."""
    slab.Q_source[:] = 1000.0
    result = SteadyStateSolver(slab, SolverConfig(method="direct")).solve()
    z = slab.z
    z0, z1 = z[0], z[-1]
    t0, t1 = slab.face_bc["z_min"].value, slab.face_bc["z_max"].value
    linear = t0 + (t1 - t0) * (z - z0) / (z1 - z0)
    parabola = 1000.0 / (2 * slab.k[0, 0, 0]) * (z - z0) * (z1 - z)
    analytic = linear + parabola
    assert result.converged
    assert np.abs(column(slab) - analytic).max() < 1e-8


def test_convective_face_flux_matches_the_analytic_resistance(slab):
    """Robin face: q = dT / (L/k + 1/h) with the half-cell correction."""
    h, t_inf = 25.0, 500.0
    slab.set_convection_bc("z_max", h, t_inf)
    SteadyStateSolver(slab, SolverConfig(method="direct")).solve()
    q_mid = slab.k[0, 0, 0] * (column(slab)[-2] - column(slab)[-1]) / slab.d
    total = (slab.Lz - slab.d) / slab.k[0, 0, 0] + 1.0 / half_cell_h(
        slab.k[0, 0, 0], h, slab.d)
    q_analytic = (slab.face_bc["z_min"].value - t_inf) / total
    assert q_mid == pytest.approx(q_analytic, rel=2e-2)


def test_imposed_flux_face_adds_exactly_that_power(adiabatic_box):
    """Neumann faces were dead code in the old builder; now they carry the flux."""
    box = adiabatic_box
    flux, area = 100.0, box.Lx * box.Ly
    box.set_heat_flux_bc("z_min", flux)
    box.set_fixed_temperature_bc("z_max", 300.0)
    SteadyStateSolver(box, SolverConfig(method="direct")).solve()
    from src.analysis.fluxes import domain_face_flux

    assert domain_face_flux(box, "z_min") == pytest.approx(-flux * area, rel=1e-6)


def test_dirichlet_elimination_keeps_the_matrix_symmetric(slab):
    """Row replacement used to break symmetry; symmetric elimination does not."""
    matrix, rhs = build_steady_matrix(slab)
    assert is_symmetric(matrix)
    cg = solve_linear(matrix, rhs, LinearConfig(method="cg", tolerance=1e-12))
    direct = solve_linear(matrix, rhs, LinearConfig(method="direct"))
    assert cg.converged and cg.notes == []
    assert np.abs(cg.T - direct.T).max() < 1e-6


def test_an_asymmetric_operator_is_solved_without_cg(slab):
    matrix, rhs = build_steady_matrix(slab)
    skewed = matrix.tolil()
    skewed[0, 1] = 5.0                     # one-sided coupling: not symmetric
    skewed = skewed.tocsr()
    assert not is_symmetric(skewed)
    result = solve_linear(skewed, rhs, LinearConfig(method="cg", tolerance=1e-8))
    assert any("bicgstab" in note for note in result.notes)


def test_direct_and_iterative_agree(slab):
    matrix, rhs = build_steady_matrix(slab)
    direct = solve_linear(matrix, rhs, LinearConfig(method="direct"))
    iterative = solve_linear(matrix, rhs, LinearConfig(method="bicgstab", tolerance=1e-12))
    assert np.abs(direct.T - iterative.T).max() < 1e-6


@pytest.mark.skipif(not HAS_PYAMG,
                    reason="PyAMG is not installed: the layer falls back to Jacobi")
def test_amg_is_used_when_available(slab):
    """AMG is *used*: no fallback note, and the same answer as the direct solve."""
    matrix, rhs = build_steady_matrix(slab)
    result = solve_linear(matrix, rhs, LinearConfig(method="bicgstab",
                                                    preconditioner="amg_rs",
                                                    tolerance=1e-10))
    assert result.converged
    fallback = [note for note in result.notes
                if "falling back" in note or "not installed" in note]
    assert fallback == [], fallback
    direct = solve_linear(matrix, rhs, LinearConfig(method="direct"))
    assert np.abs(result.T - direct.T).max() < 1e-8


# ----------------------------------------------------------------- transient
@pytest.mark.parametrize("spacing", [0.25, 0.125])
def test_transient_rate_is_mesh_independent(adiabatic_box, spacing):
    """dT/dt = Q/(rho*cp).  With a cell-volume factor the rate scales as 1/d^3."""
    n = int(round(1.0 / spacing))
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=spacing)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        mesh.set_adiabatic(face)
    power_density, t_final, dt = 1000.0, 10000.0, 1000.0
    total_power = power_density * mesh.V_cell * mesh.N_total
    config = TransientConfig(t_final=t_final, dt=dt, save_interval=t_final,
                             initial_condition=InitialCondition(mode="uniform", t_uniform=293.15),
                             power_profile=PowerProfile(mode="constant", constant_power=total_power))
    mesh.source_mask[:] = True
    TransientSolver(mesh, config).run()
    expected = 293.15 + power_density / (1000.0 * 1000.0) * t_final
    assert mesh.T.mean() == pytest.approx(expected, rel=1e-6), f"n={n} cells per axis"


def test_transient_enforces_dirichlet_faces_exactly(slab):
    """The fixed-temperature ground must not decay: the row is identity + M/dt."""
    config = TransientConfig(t_final=600.0, dt=600.0, save_interval=600.0,
                             initial_condition=InitialCondition(mode="uniform", t_uniform=400.0),
                             power_profile=PowerProfile(mode="off"))
    TransientSolver(slab, config).run()
    assert slab.T[:, :, 0].min() == pytest.approx(300.0, abs=1e-2)
    assert slab.T[:, :, -1].max() == pytest.approx(400.0, abs=1e-2)


def test_transient_matches_the_analytic_slab_series():
    """1D Dirichlet slab vs the Fourier series, backward Euler, 5% band."""
    mesh = Mesh3D(Lx=0.5, Ly=0.5, Lz=1.0, spacing=0.05)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    mesh.set_fixed_temperature_bc("z_min", 373.15)
    mesh.set_fixed_temperature_bc("z_max", 373.15)
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_adiabatic(face)
    alpha = 1e-6
    t_end, dt = 200_000.0, 10_000.0
    config = TransientConfig(t_final=t_end, dt=dt, save_interval=t_end,
                             initial_condition=InitialCondition(mode="uniform", t_uniform=273.15),
                             power_profile=PowerProfile(mode="off"))
    TransientSolver(mesh, config).run()
    z = mesh.z
    analytic = np.full_like(z, 373.15)
    for n in range(1, 200):
        b = 2 * (273.15 - 373.15) * (1 - (-1) ** n) / (n * np.pi)
        analytic += b * np.sin(n * np.pi * z) * np.exp(-(n * np.pi) ** 2 * alpha * t_end)
    # the pinned boundary nodes carry a first-order surface artifact; the interior
    # must follow the series within a few percent
    interior = slice(2, -2)
    error = np.abs(column(mesh)[interior] - analytic[interior]).max()
    assert error < 0.05 * (373.15 - 273.15), f"max interior error {error:.2f} K"


def test_extraction_is_capped_by_availability(storage_model):
    """Target power is limited by h*A*dT: no heat is pumped from a cold body."""
    mesh = storage_model
    mesh.T[:] = 293.15
    config = TransientConfig(
        t_final=600.0, dt=600.0, save_interval=600.0, t_ambient=293.15,
        initial_condition=InitialCondition(mode="uniform", t_uniform=293.15),
        power_profile=PowerProfile(mode="off"),
        extraction_profile=ExtractionProfile(mode="power", power=1e6,
                                             t_inlet=333.15, h_fluid=500.0))
    mesh.material_id[0, 0, 0] = 4                     # one tube cell
    mesh.set_internal_convection(mesh.material_id == 4, 500.0, 333.15)
    results = TransientSolver(mesh, config).run()
    assert max(results.P_extracted) == 0.0            # tube colder than the inlet


def test_power_profile_without_sources_raises(storage_model):
    mesh = storage_model
    mesh.source_mask.fill(False)
    mesh.Q_source.fill(0.0)
    config = TransientConfig(t_final=60.0, dt=60.0, save_interval=60.0,
                             power_profile=PowerProfile(mode="constant", constant_power=1000.0))
    with pytest.raises(ValueError, match="source"):
        TransientSolver(mesh, config).run()


def test_transient_stops_cleanly_on_request(adiabatic_box):
    mesh = adiabatic_box
    mesh.source_mask[:] = True
    mesh.Q_source[:] = 1000.0
    calls = {"n": 0}

    def stop() -> bool:
        calls["n"] += 1
        return calls["n"] > 2

    config = TransientConfig(t_final=1e6, dt=1000.0, save_interval=1000.0)
    results = TransientSolver(mesh, config).run(should_stop=stop)
    assert len(results) < 10


#: With the environment active the air box is excluded and the film of the outer
#: surface replaces the domain-face boundary conditions: the radiative share of the
#: shell has to ride on that film, not on the box faces (which carry no flux once
#: their cells are excluded).  ``apply_environment`` paints the design-point share and
#: the assembly re-evaluates it on the field it solves, so the two tests below measure
#: the radiative loss where it physically happens.
def test_radiation_increases_the_steady_losses(storage_model):
    mesh = storage_model
    plain = SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
    from src.analysis.balance import compute_balance

    loss_plain = compute_balance(mesh).q_battery
    solver = SteadyStateSolver(mesh, SolverConfig(method="direct", radiation=True,
                                                  max_picard=20))
    result = solver.solve()
    loss_radiant = compute_balance(mesh, radiation=True).q_battery
    assert result.converged
    assert result.iterations >= 1
    assert loss_radiant > loss_plain
    assert plain.T.max() > 0


# ------------------------------------------------------- transient edge cases
def test_last_step_is_shortened_when_t_final_is_not_a_multiple_of_dt(adiabatic_box):
    mesh = adiabatic_box
    mesh.source_mask[:] = True
    total_power = 1000.0 * mesh.V_cell * mesh.N_total
    config = TransientConfig(t_final=750.0, dt=600.0, save_interval=10_000.0,
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=total_power))
    results = TransientSolver(mesh, config).run()
    assert results.times[-1] == pytest.approx(750.0)          # no overshoot
    assert len(results) == 2                                  # 600 s + shortened 150 s
    expected = 293.15 + 1000.0 / 1e6 * 750.0
    assert mesh.T.mean() == pytest.approx(expected, rel=1e-6)


def test_dt_larger_than_t_final_still_runs_one_step(adiabatic_box):
    mesh = adiabatic_box
    mesh.source_mask[:] = True
    total_power = 1000.0 * mesh.V_cell * mesh.N_total
    config = TransientConfig(t_final=600.0, dt=3600.0, save_interval=600.0,
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=total_power))
    results = TransientSolver(mesh, config).run()
    assert results.times[-1] == pytest.approx(600.0)
    assert mesh.T.mean() == pytest.approx(293.15 + 1000.0 / 1e6 * 600.0, rel=1e-6)


def test_save_interval_finer_than_dt_saves_every_step(adiabatic_box):
    mesh = adiabatic_box
    mesh.source_mask[:] = True
    total_power = 500.0 * mesh.V_cell * mesh.N_total
    config = TransientConfig(t_final=1800.0, dt=600.0, save_interval=100.0,
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=total_power))
    results = TransientSolver(mesh, config).run()
    assert len(results) == 3
    assert results.times == sorted(results.times)
    assert results.times[-1] == pytest.approx(1800.0)


def test_flow_rate_extraction_removes_heat_when_the_battery_is_hot(storage_model):
    """Hot tubes must lose energy to the fluid, and never more than it can carry."""
    from src.core.mesh import MaterialID

    mesh = storage_model
    mesh.T[:] = 500.0
    tubes = mesh.material_id == int(MaterialID.TUBES)
    # inside the vessel, not in the air box: with the environment active the cells
    # outside the envelope are excluded from the problem and pinned at the ambient, so
    # a synthetic tube placed at the corner of the domain would be an isothermal block
    # at 293 K and the extracted power would come out with the wrong sign
    i, j = mesh.Nx // 2, mesh.Ny // 2
    mesh.material_id[i - 1:i + 1, j - 1:j + 1, :6] = int(MaterialID.TUBES)
    mesh.boundary_type[i - 1:i + 1, j - 1:j + 1, :6] = 0
    mesh.set_internal_convection(mesh.material_id == int(MaterialID.TUBES), 500.0, 300.0)
    assert (mesh.material_id == int(MaterialID.TUBES)).any()

    config = TransientConfig(
        t_final=600.0, dt=600.0, save_interval=600.0, t_ambient=293.15,
        initial_condition=InitialCondition(mode="uniform", t_uniform=500.0),
        power_profile=PowerProfile(mode="off"),
        extraction_profile=ExtractionProfile(mode="flow_rate", mass_flow=0.5,
                                             t_inlet=300.0, h_fluid=500.0))
    results = TransientSolver(mesh, config).run()
    assert results.P_extracted[-1] > 0.0
    assert results.T_mean_storage[-1] < 500.0
    _ = tubes


def test_transient_with_radiation_stays_finite_and_loses_more(storage_model):
    mesh = storage_model
    config = TransientConfig(
        t_final=600.0, dt=600.0, save_interval=600.0,
        initial_condition=InitialCondition(mode="uniform", t_uniform=800.0),
        power_profile=PowerProfile(mode="off"))
    plain = TransientSolver(mesh, config, SolverConfig(method="bicgstab")).run()
    loss_plain = plain.Q_losses_total[-1]

    mesh.T[:] = 800.0
    radiant_solver = TransientSolver(mesh, config, SolverConfig(method="bicgstab",
                                                               radiation=True))
    radiant = radiant_solver.run()
    assert np.all(np.isfinite(mesh.T))
    assert radiant.Q_losses_total[-1] > loss_plain


def test_a_loop_that_changes_the_tube_film_rebuilds_the_operator():
    """Regression: the cached operator used to keep a film the RHS had lost.

    With a fluid loop the transient zeroes the tube film, but the matrix built at the
    start still carried it: the tube cells were then solved as
    ``(m/dt + a_p_stale) T = (m/dt) T_prev + b`` and drifted to a nonsense temperature
    (-89 C in the reported case) while the bed sat at +20 C.  The operator must be
    rebuilt whenever the film it was built with changes.
    """
    from src.core.pipes import rasterize_pipe
    from src.solver.fluid import FluidLoop
    from src.solver.transient import TransientConfig, TransientSolver

    mesh = Mesh3D(1.0, 1.0, 1.0, spacing=0.2)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1500.0
    mesh.cp[:] = 800.0
    mesh.T[:] = 293.15
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        mesh.set_adiabatic(face)
    run = rasterize_pipe(mesh, [(0.5, 0.5, 0.0), (0.5, 0.5, 1.0)], 0.05)
    tube = np.zeros(mesh.T.shape, dtype=bool)
    flat = tube.ravel(order="F")
    flat[run.cells] = True
    tube = flat.reshape(mesh.T.shape, order="F")
    mesh.material_id[tube] = int(MaterialID.TUBES)
    mesh.set_internal_convection(tube, 500.0, 333.15)        # a film before the loop
    loop = FluidLoop(runs=[run], mass_flow=0.01, h_fluid=500.0, t_in=300.0)
    config = TransientConfig(t_final=1800.0, dt=300.0,
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=5000.0),
                             fluid_loop=loop)
    TransientSolver(mesh, config, SolverConfig(method="cg", tolerance=1e-8)).run()
    # the field must stay physical: the bed is heated, nothing dives to negative C
    assert float(mesh.T.min()) > 280.0, float(mesh.T.min())
    assert float(mesh.T.max()) < 500.0
    assert np.all(np.isfinite(mesh.T))
