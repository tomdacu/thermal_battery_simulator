"""Analysis and IO tests: flux closure, losses iteration, HDF5 round trip."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.balance import compute_balance, storage_capacity, thermal_autonomy
from src.analysis.fluxes import domain_fluxes, envelope_fluxes, stored_energy
from src.analysis.losses import LossesConfig, solve_losses
from src.constants import T_AMBIENT_DEFAULT
from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand
from src.core.geometry import create_small_test_geometry
from src.core.mesh import Mesh3D
from src.io.state import FORMAT_VERSION, StateError, StateManager, geometry_hash
from src.solver.steady import SolverConfig, SteadyStateSolver

#: the box the tree fixtures of the suite paint their models in: an 8 m cube whose finest
#: cell is 0.5 m, i.e. 16 of them a side (``tests/conftest.py``)
BOX = 8.0
FINEST = 16
SPACING = 0.5


# ------------------------------------------------------------------- fluxes
def test_balance_closes_for_an_all_convective_box(adiabatic_box):
    """With convective faces only, sum(fluxes) equals the injected power exactly."""
    box = adiabatic_box
    box.Q_source[:] = 500.0
    box.source_mask[:] = True
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        box.set_convection_bc(face, 8.0, 293.15)
    SteadyStateSolver(box, SolverConfig(method="direct")).solve()
    balance = compute_balance(box)
    assert balance.imbalance == pytest.approx(0.0, abs=1e-6 * balance.p_input)
    assert balance.q_domain > 0


def test_dirichlet_face_flux_is_not_identically_zero(storage_model):
    """The old formula evaluated T-T_bc on pinned nodes and always returned 0."""
    SteadyStateSolver(storage_model, SolverConfig(method="direct")).solve()
    fluxes = domain_fluxes(storage_model)
    assert abs(fluxes["z_min"]) > 1.0        # ground path carries heat


def test_stored_energy_and_exergy_are_referenced_to_ambient(adiabatic_box):
    box = adiabatic_box
    box.T[:] = T_AMBIENT_DEFAULT
    assert stored_energy(box, T_AMBIENT_DEFAULT) == pytest.approx(0.0, abs=1e-6)
    box.T[:] = T_AMBIENT_DEFAULT + 100.0
    thermal = stored_energy(box, T_AMBIENT_DEFAULT)
    exergy = compute_balance(box).ex_stored
    assert thermal > 0
    assert 0 < exergy < thermal          # exergy is always below the energy


def test_envelope_loss_does_not_scale_with_the_air_box():
    """Two identical batteries in different domains must report the same loss."""
    losses = []
    for size in (6.0, 9.0):
        mesh = Mesh3D(Lx=size, Ly=size, Lz=5.6, spacing=0.5)
        geometry = create_small_test_geometry()
        geometry.cylinder.center_x = geometry.cylinder.center_y = size / 2
        geometry.apply_to_mesh(mesh)
        SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
        losses.append(envelope_fluxes(mesh)["total"])
    # the air box changes the surface temperature slightly, but not the
    # order of magnitude as the old domain-wide loss integral did
    assert losses[0] == pytest.approx(losses[1], rel=0.3)


def test_storage_capacity_and_autonomy(storage_model):
    capacity = storage_capacity(storage_model, t_max=873.15, t_ambient=293.15)
    assert capacity > 0
    SteadyStateSolver(storage_model, SolverConfig(method="direct")).solve()
    autonomy = thermal_autonomy(storage_model)
    assert autonomy["hours"] > 0
    assert autonomy["loss_W"] > 0


# ------------------------------------------------------------------- losses
def test_losses_analysis_holds_the_target_temperature(storage_model):
    mesh = storage_model
    target = 473.15
    config = LossesConfig(t_target=target, t_ambient=mesh.face_bc["z_max"].value,
                          tolerance=2.0, max_iterations=12, h_ground=5.0)
    result = solve_losses(mesh, config, SolverConfig(method="direct"))
    assert result.power > 0
    assert abs(result.t_mean_storage - target) <= 5.0
    assert result.balance.q_battery > 0
    assert result.history[0]["iteration"] == 1


def test_losses_analysis_validates_input(storage_model):
    with pytest.raises(ValueError):
        solve_losses(storage_model, LossesConfig(t_target=200.0, t_ambient=293.15))


# ---------------------------------------------------------------------- IO
def test_state_round_trip(storage_model):
    manager = StateManager(directory="results/states")
    path = manager.save_state(storage_model, name="pytest_state")
    state = manager.load_state(path)
    assert state.version == FORMAT_VERSION
    assert state.temperature_unit == "K"
    assert state.T.shape == storage_model.T.shape
    mesh = Mesh3D(Lx=storage_model.Lx, Ly=storage_model.Ly, Lz=storage_model.Lz,
                  spacing=storage_model.d)
    create_small_test_geometry().apply_to_mesh(mesh)
    notes = manager.apply(mesh, state)
    assert np.allclose(mesh.T, storage_model.T)
    assert notes is not None


def test_state_refuses_a_different_geometry(storage_model):
    manager = StateManager(directory="results/states")
    path = manager.save_state(storage_model, name="pytest_state_2")
    state = manager.load_state(path)
    other = Mesh3D(Lx=7.0, Ly=7.0, Lz=5.6, spacing=0.5)
    create_small_test_geometry().apply_to_mesh(other)
    problems = manager.verify(other, state)
    assert any("hash" in problem or "grid" in problem for problem in problems)
    with pytest.raises(StateError):
        manager.apply(other, state)


def test_state_hash_covers_geometry_parameters(storage_model):
    base = geometry_hash(storage_model, {"r_storage": 2.0, "height": 4.0})
    moved = geometry_hash(storage_model, {"r_storage": 2.5, "height": 4.0})
    assert base != moved


def test_state_rejects_a_missing_file():
    with pytest.raises(StateError):
        StateManager().load_state("results/states/does_not_exist.h5")


def test_state_upgrades_a_celsius_file(tmp_path):
    import h5py

    path = tmp_path / "legacy.h5"
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=0.5)
    with h5py.File(path, "w") as handle:
        handle.attrs["version"] = 1
        handle.attrs["temperature_unit"] = "degC"
        handle.create_dataset("material_id", data=mesh.material_id)
        handle.create_dataset("T", data=mesh.T - 273.15)
    state = StateManager().load_state(str(path))
    assert state.temperature_unit == "K"
    assert state.T.min() == pytest.approx(mesh.T.min(), abs=1e-6)
    assert any("converted" in note for note in state.notes)


# ------------------------------------------------------------------ IO (a tree)
def test_state_round_trip_keeps_a_tree_field_exact(tree_model):
    """A tree's state is its leaf list: the field comes back leaf by leaf, exactly."""
    manager = StateManager(directory="results/states")
    tree_model.T = tree_model.T + np.linspace(0.0, 40.0, tree_model.n_cells)
    path = manager.save_state(tree_model, name="pytest_state_tree")
    state = manager.load_state(path)
    assert state.grid["leaves"] is not None
    assert len(state.grid["leaves"]) == tree_model.n_cells
    assert state.T.shape == tree_model.T.shape

    fresh = AdaptiveMesh.uniform(FINEST, SPACING, level=0)
    create_small_test_geometry().apply_to_mesh(fresh)
    manager.apply(fresh, state)
    assert np.array_equal(fresh.T, tree_model.T)


def test_state_refuses_a_tree_that_is_not_the_one_it_stored(tree_model):
    """Refining the model moves the cell count and the leaf list: the state is refused."""
    manager = StateManager(directory="results/states")
    path = manager.save_state(tree_model, name="pytest_state_tree_2")
    state = manager.load_state(path)
    refined = AdaptiveMesh.uniform(FINEST, SPACING, level=2)      # 2 m leaves to start from
    create_small_test_geometry().apply_to_mesh(refined)
    refined.refine_bands((RefinementBand(low=(0.0, 0.0, 0.0),
                                         high=(BOX, BOX, BOX / 2.0), size=SPACING),))
    assert refined.n_cells != tree_model.n_cells
    problems = manager.verify(refined, state)
    assert any("hash" in problem or "grid" in problem for problem in problems)
    with pytest.raises(StateError):
        manager.apply(refined, state)
