"""Core model tests: units contract, mesh, materials, geometry, profiles."""
from __future__ import annotations

import numpy as np
import pytest

from src.constants import T_AMBIENT_DEFAULT, T_GROUND_DEFAULT
from src.core.geometry import (BatteryGeometry, CylinderGeometry,
                               create_small_test_geometry)
from src.core.materials import MaterialManager
from src.core.mesh import BoundaryType, MaterialID, Mesh3D
from src.core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from src.units import c_to_k, check_kelvin, k_to_c


# --------------------------------------------------------------------- units
def test_celsius_kelvin_round_trip():
    assert c_to_k(0.0) == pytest.approx(273.15)
    assert k_to_c(373.15) == pytest.approx(100.0)


def test_check_kelvin_rejects_a_celsius_field():
    """A Celsius field is the classic unit bug: it must fail loudly, never silently."""
    with pytest.raises(ValueError, match="degC"):
        check_kelvin(np.array([20.0, 300.0]), "T")
    check_kelvin(np.array([T_AMBIENT_DEFAULT, 900.0]), "T")


def test_mesh_defaults_are_kelvin():
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=0.5)
    assert mesh.T.min() == pytest.approx(T_AMBIENT_DEFAULT)
    assert mesh.face_bc["z_min"].value == pytest.approx(T_GROUND_DEFAULT)
    mesh.validate()


def test_boundary_conditions_record_the_face_not_the_node():
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=0.25)
    mesh.set_convection_bc("z_max", 12.0, 300.0)
    mesh.set_fixed_temperature_bc("x_min", 350.0)
    mesh.set_adiabatic("y_min")
    assert mesh.face_bc["z_max"].h == 12.0
    assert mesh.face_bc["x_min"].kind == BoundaryType.DIRICHLET
    assert mesh.face_bc["y_min"].kind == BoundaryType.INTERNAL
    with pytest.raises(ValueError):
        mesh.set_convection_bc("top", 1.0, 300.0)
    with pytest.raises(ValueError):
        mesh.set_convection_bc("z_max", 1.0, 20.0)      # Celsius passed as Kelvin


def test_domain_snaps_to_whole_cells():
    mesh = Mesh3D(Lx=6.0, Ly=6.0, Lz=5.6, spacing=0.5)
    assert (mesh.Nx, mesh.Ny, mesh.Nz) == (12, 12, 11)
    assert mesh.Lz == pytest.approx(5.5)
    assert mesh.snapped["Lz"] == pytest.approx(0.1)


def test_index_round_trip_and_cell_lookup():
    mesh = Mesh3D(Lx=4.0, Ly=4.0, Lz=4.0, spacing=0.5)
    for (i, j, k) in [(0, 0, 0), (1, 2, 3), (7, 7, 7)]:
        assert mesh.linear_to_ijk(mesh.ijk_to_linear(i, j, k)) == (i, j, k)
    assert mesh.find_cell(-1.0, 2.0, 2.0) == (0, 4, 4)
    assert mesh.find_cell(99.0, 99.0, 99.0) == (mesh.Nx - 1, mesh.Ny - 1, mesh.Nz - 1)


def test_validate_rejects_non_physical_fields():
    mesh = Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=0.5)
    mesh.rho[0, 0, 0] = -1.0
    with pytest.raises(ValueError, match="rho"):
        mesh.validate()


# ----------------------------------------------------------------- materials
def test_material_database_is_single_and_consistent():
    manager = MaterialManager()
    steatite = manager.get("steatite")
    assert (steatite.k, steatite.rho, steatite.cp) == (3.0, 2700.0, 980.0)
    assert steatite.t_max > 1200.0
    with pytest.raises(KeyError):
        manager.get("does_not_exist")
    assert "steatite" in manager.list_materials("storage")


def test_packed_bed_lowers_conductivity_and_density():
    manager = MaterialManager()
    solid = manager.get("silica_sand")
    bed = manager.compute_packed_bed_properties("silica_sand", 0.63)
    assert 0.0 < bed.k < solid.k
    assert bed.rho < solid.rho
    assert bed.cp > 0


@pytest.mark.parametrize("packing", [0.0, 1.0, -0.2, 1.5])
def test_packing_fraction_is_validated(packing):
    with pytest.raises(ValueError):
        MaterialManager().compute_packed_bed_properties("silica_sand", packing)


def test_energy_density_scales_with_temperature_span():
    manager = MaterialManager()
    single = manager.get_energy_density("steatite", t_high=873.15, t_low=293.15)
    double = manager.get_energy_density("steatite", t_high=1453.15, t_low=293.15)
    assert double == pytest.approx(2 * single)


# ------------------------------------------------------------------ geometry
def test_apply_to_mesh_marks_sources_and_keeps_the_power_budget(storage_model):
    """The volumetric sources must integrate to the rated heater power."""
    mesh = storage_model
    assert mesh.source_mask.any()
    assert mesh.T.min() > 250.0                        # Kelvin, not Celsius
    power = float(np.sum(mesh.Q_source) * mesh.V_cell)
    assert power == pytest.approx(50_000.0, rel=1e-9)


def test_geometry_validation_rejects_a_clipped_roof():
    mesh = Mesh3D(Lx=6.0, Ly=6.0, Lz=3.0, spacing=0.5)
    geometry = create_small_test_geometry()
    with pytest.raises(ValueError, match="Lz"):
        geometry.apply_to_mesh(mesh)


def test_geometry_validation_rejects_a_battery_larger_than_the_domain():
    mesh = Mesh3D(Lx=3.0, Ly=3.0, Lz=8.0, spacing=0.5)
    geometry = BatteryGeometry(cylinder=CylinderGeometry(center_x=1.5, center_y=1.5,
                                                         r_storage=2.0))
    with pytest.raises(ValueError, match="radius"):
        geometry.apply_to_mesh(mesh)


def test_zone_volumes_and_masses_are_consistent(storage_model):
    geometry = create_small_test_geometry()
    volumes = geometry.zone_volumes()
    masses = geometry.zone_masses()
    assert volumes["storage"] == pytest.approx(np.pi * 2.0 ** 2 * 4.0, rel=1e-9)
    assert masses["storage"] > 0
    assert set(masses) >= {"storage", "insulation", "shell", "cone_shell", "foundation"}
    capacity = geometry.estimate_energy_capacity(t_high=873.15, t_low=293.15)
    assert capacity["E_usable_J"] < capacity["E_thermal_J"]
    assert capacity["mass_storage_kg"] == pytest.approx(masses["storage"])


# ------------------------------------------------------------------ profiles
def test_power_profile_modes_and_extrapolation():
    assert PowerProfile(mode="off").power_at(10.0) == 0.0
    assert PowerProfile(mode="constant", constant_power=1234.0).power_at(0.0) == 1234.0
    schedule = PowerProfile(mode="schedule", times=[0.0, 100.0], powers=[0.0, 1000.0])
    assert schedule.power_at(50.0) == pytest.approx(500.0)
    assert schedule.power_at(5000.0) == pytest.approx(1000.0)   # holds the last value


def test_power_profile_rejects_broken_input():
    with pytest.raises(ValueError):
        PowerProfile(mode="schedule", times=[10.0, 5.0], powers=[1.0, 2.0])
    with pytest.raises(ValueError):
        PowerProfile(mode="schedule", times=[0.0], powers=[1.0, 2.0])
    with pytest.raises(ValueError):
        PowerProfile(mode="csv").validate()
    assert PowerProfile(mode="off").validate() == []


def test_extraction_profile_has_no_free_energy_placeholder():
    profile = ExtractionProfile(mode="power", power=5000.0)
    assert profile.power_request(0.0) == 5000.0
    assert profile.validate() == []
    flow = ExtractionProfile(mode="flow_rate", mass_flow=0.0)
    assert any("mass_flow" in problem for problem in flow.validate())
    assert ExtractionProfile(mode="off").power_request(0.0) == 0.0


def test_initial_condition_modes_return_kelvin(storage_model):
    mesh = storage_model
    uniform = InitialCondition(mode="uniform", t_uniform=373.15)
    field = uniform.apply_to_mesh(mesh)
    assert field.min() == field.max() == pytest.approx(373.15)
    check_kelvin(field, "IC")
    by_material = InitialCondition(mode="by_material",
                                   t_by_material={int(MaterialID.SAND): 873.15})
    field = by_material.apply_to_mesh(mesh)
    assert field[mesh.material_id == int(MaterialID.SAND)].min() == pytest.approx(873.15)
    with pytest.raises(ValueError):
        InitialCondition(mode="by_material", t_by_material={99: 300.0}).apply_to_mesh(mesh)


def test_initial_condition_rejects_celsius_values():
    with pytest.raises(ValueError):
        InitialCondition(mode="uniform", t_uniform=20.0).apply_to_mesh(
            Mesh3D(Lx=1.0, Ly=1.0, Lz=1.0, spacing=0.5))
