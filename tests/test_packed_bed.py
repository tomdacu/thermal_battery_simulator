"""The conductivity of the packed bed: Zehner-Bauer-Schlunder with radiation.

Pinned: the model reduces to the gas when the grains are the gas (and the radiation is
switched off), grows with the temperature and the grain size (the radiation across the
voids), stays between the two phases' series and parallel bounds without radiation,
and lands in the measured range of dry sand beds at room temperature (0.25-0.45
W/(m K) in the sources, e.g. VDI Heat Atlas D6.3 fig. 3 and the dry-sand data of
Farouki, CRREL 81-1; the assert is looser, 0.25-0.5, because the models differ).
The solvers then evaluate it on the field: a steady bed at 500 degC carries the
conductivity of 500 degC.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.materials import MaterialManager, PackedBed


def test_a_bed_of_gas_grains_without_radiation_is_the_gas():
    law = PackedBed(solid_k=0.0263, porosity=0.4, diameter=1e-12, fluid_k=0.0263)
    assert law(300.0)[0] == pytest.approx(law.gas_k(np.array([300.0]))[0], rel=1e-9)


def test_without_radiation_the_bed_lies_between_the_series_and_parallel_bounds():
    law = PackedBed(solid_k=3.0, porosity=0.37, diameter=1e-12)
    k_f = law.gas_k(np.array([300.0]))[0]
    k = law(300.0)[0]
    series = 1.0 / (0.63 / 3.0 + 0.37 / k_f)
    parallel = 0.63 * 3.0 + 0.37 * k_f
    assert series < k < parallel


def test_the_bed_conducts_better_hot_and_coarse():
    manager = MaterialManager()
    fine = manager.packed_bed("steatite", 0.63, 0.5e-3)
    coarse = manager.packed_bed("steatite", 0.63, 3e-3)
    temperatures = np.array([293.15, 573.15, 773.15, 973.15])
    k = coarse(temperatures)
    assert np.all(np.diff(k) > 0.0)
    assert np.all(coarse(temperatures[1:]) > fine(temperatures[1:]))


def test_dry_sand_at_room_temperature_is_in_the_measured_range():
    manager = MaterialManager()
    for medium in ("silica_sand", "steatite", "quartzite"):
        k = manager.packed_bed(medium, 0.63)(293.15)[0]
        assert 0.25 < k < 0.5, (medium, k)


def test_the_steady_bed_takes_the_conductivity_of_its_temperature():
    from src.core.adaptive_mesh import AdaptiveMesh
    from src.core.geometry import BatteryGeometry, CylinderGeometry
    from src.core.mesh import storage_mask
    from src.solver.steady import SolverConfig, SteadyStateSolver

    cylinder = CylinderGeometry(center_x=1.6, center_y=1.6, base_z=0.3, height=1.5,
                                r_storage=1.0, insulation_thickness=0.3,
                                insulation_slab_bottom=0.2, insulation_slab_top=0.2,
                                enable_cone_roof=False)
    battery = BatteryGeometry(cylinder=cylinder)
    mesh = AdaptiveMesh.from_box(16, 16, 0.2, 0.25, 0, 0)
    battery.apply_to_mesh(mesh)
    bed = storage_mask(mesh.material_id)
    mesh.Q_source[bed] = 2000.0 / float(mesh.V[bed].sum())
    result = SteadyStateSolver(mesh, SolverConfig(method="direct")).solve()
    assert result.converged and result.iterations > 1
    law = battery.bed_law()
    # every bed cell carries its law at its own temperature, to the update step
    assert np.allclose(mesh.k[bed], law(mesh.T[bed]), rtol=0.021)
