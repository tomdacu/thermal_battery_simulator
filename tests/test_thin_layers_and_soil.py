"""Thin steel on coarse cells, and the soil under the foundation.

A 20 mm shell on 100 mm cells used to be painted as 100 mm of steel over the insulation:
five times the steel's heat capacity, and a fifth of the insulation gone.  The outer cell
is now a blend - steel and insulation in series across it, their capacities by volume -
so the insulation keeps its resistance and the shell its own mass.  The soil layer puts
the deep-ground temperature a few metres down instead of right under the pad.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import BatteryGeometry, CylinderGeometry
from src.core.materials import MaterialManager
from src.core.mesh import BoundaryType, MaterialID


def _battery(ground: float = 0.0) -> tuple[BatteryGeometry, AdaptiveMesh]:
    cylinder = CylinderGeometry(center_x=1.6, center_y=1.6, base_z=0.3 + ground,
                                height=1.5, r_storage=1.0, insulation_thickness=0.3,
                                shell_thickness=0.02, insulation_slab_bottom=0.2,
                                insulation_slab_top=0.2, enable_cone_roof=False,
                                ground_depth=ground)
    battery = BatteryGeometry(cylinder=cylinder)
    mesh = AdaptiveMesh.from_box(32, 32, 0.1, 0.25 if ground else 0.125, 0, 0)
    battery.apply_to_mesh(mesh)
    return battery, mesh


def test_a_thin_shell_blends_with_the_insulation_behind_it():
    battery, mesh = _battery()
    materials = MaterialManager()
    steel = materials.get(battery.shell_material)
    insulation = materials.get(battery.insulation_material)
    shell = mesh.material_id == int(MaterialID.STEEL)
    assert shell.any()
    f = 0.02 / 0.1
    series = 1.0 / (f / steel.k + (1.0 - f) / insulation.k)
    assert np.allclose(mesh.k[shell], series)
    assert np.allclose(mesh.rho[shell] * mesh.cp[shell],
                       f * steel.rho * steel.cp + (1 - f) * insulation.rho * insulation.cp)
    # the steel mass of the model is the shell's, to the staircase of the circle
    painted = float(np.sum(mesh.V[shell] * f)) * steel.rho
    expected = battery.zone_masses()["shell"]
    assert painted == pytest.approx(expected, rel=0.25)


def test_the_soil_carries_the_ground_temperature_down_to_the_box_floor():
    _battery_, mesh = _battery(ground=2.0)
    soil = mesh.material_id == int(MaterialID.GROUND)
    z = mesh.centres()[:, 2]
    assert soil.any() and np.all(z[soil] < 2.0)
    assert not np.any(mesh.excluded[soil])
    for face in ("x_min", "x_max", "y_min", "y_max"):
        assert mesh.face_bc[face].kind == BoundaryType.INTERNAL   # the soil goes on
    assert mesh.face_bc["z_min"].kind == BoundaryType.DIRICHLET
