"""The geometry painter on a tree: the same model, leaf by leaf on both meshes.

Step 1 of the migration (``docs/16_ADAPTIVE_MESH_MIGRATION.md``): ``apply_to_mesh`` takes
a ``Mesh3D`` or an ``AdaptiveMesh``, and the model it paints is the same on both.  The
comparison is the one ``tests/test_adaptive_mesh.py::structured_partner`` makes for the
solver - read the structured field back through the cell of every leaf - applied to the
painter: the two meshes are painted *independently* (the same geometry object twice) and
every per-cell field is compared, so a mask that only knew the structured layout shows up
as a difference and not as a wrong number nobody looks at.

The pair is the one of ``tests/conftest.py``: a uniform tree of 16^3 leaves of 0.5 m in an
8 m cube and the ``Mesh3D`` of the same cells.  The structured fixture of the suite
(``storage_model``, 6 x 6 x 5.6 m snapped to 5.5 m) has no tree twin - a tree spans a cube
with a power-of-two number of cells per side - which is why the comparison uses the cube.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import (BatteryGeometry, CylinderGeometry,
                               create_small_test_geometry)
from src.core.mesh import MaterialID, Mesh3D

#: the per-cell state ``apply_to_mesh`` writes through the protocol
PAINTED = ("material_id", "k", "rho", "cp", "Q_source", "Q_sink", "source_mask", "bc_h",
           "bc_T_inf", "boundary_type", "excluded")

#: the entries of :data:`PAINTED` that are identities or masks, not measurements: a
#: single cell of disagreement is a painter that only knew one of the two meshes
DISCRETE = ("material_id", "source_mask", "boundary_type", "excluded")


# ------------------------------------------------------------------- helpers
def cell_of(tree: AdaptiveMesh, structured: Mesh3D) -> np.ndarray:
    """Flat position of the ``Mesh3D`` cell every leaf centre falls in.

    A uniform tree and a uniform ``Mesh3D`` of the same spacing have the same cells, so
    the map is one to one; it is what reads a structured field back as a per-leaf vector
    (the counterpart of ``tests/test_adaptive_mesh.py::cells_of``).
    """
    return np.array([structured.ijk_to_linear(*structured.find_cell(*centre))
                     for centre in tree.centres()])


def painted(mesh, index: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The painted fields of ``mesh`` as per-cell vectors, in structured cell order."""
    out = {}
    for name in PAINTED:
        values = np.asarray(getattr(mesh, name))
        out[name] = values if values.ndim == 1 else values.reshape(-1, order="F")[index]
    return out


def differences(tree: AdaptiveMesh, structured: Mesh3D) -> dict[str, int]:
    """Leaves that disagree with their ``Mesh3D`` cell, per field (zero when equal).

    The masks and the identities must agree exactly.  The measurements cannot be
    compared bit for bit: the two painters evaluate the same rule over arrays of
    different shapes, so their last bits may round apart - a 1e-12 relative bound says
    the same number was meant, while a painter that read the wrong cell is off by a
    factor, not by a rounding (found by the CI: 20 leaves differed by one ulp on
    ``rho``, on a runner whose BLAS rounds differently from this workstation).
    """
    index = cell_of(tree, structured)
    on_tree, on_grid = painted(tree), painted(structured, index)
    out = {}
    for name in PAINTED:
        left, right = on_tree[name], on_grid[name]
        bad = left != right if name in DISCRETE else ~np.isclose(left, right, rtol=1e-12,
                                                                atol=1e-300)
        out[name] = int(np.count_nonzero(bad))
    return out


def same_mapping(left: dict, right: dict, rel: float = 1e-12) -> None:
    """Two dictionaries of numbers agree key by key, to a rounding.

    The same rule as :func:`differences`: the volumes and the masses of a zone are
    sums over differently shaped arrays on the two meshes.
    """
    assert left.keys() == right.keys()
    for key, value in left.items():
        assert value == pytest.approx(right[key], rel=rel), key


def battery_with_power(power_kw: float = 50.0) -> BatteryGeometry:
    """The shared test geometry with a different plant power."""
    geometry = create_small_test_geometry()
    geometry.heaters.power_total = power_kw
    return geometry


# ------------------------------------------------------------------ the painter
def test_the_painter_writes_the_same_leaves_on_a_tree(paint_pair):
    """Every field the painter writes is the same, leaf by leaf, on either mesh."""
    tree, structured = paint_pair()

    assert differences(tree, structured) == dict.fromkeys(PAINTED, 0)
    # the same statement from the other side: the tree carries a real model, not air
    assert tree.material_id.shape == (tree.n_cells,)
    assert set(np.unique(tree.material_id)) == {int(MaterialID.AIR),
                                                int(MaterialID.CONCRETE),
                                                int(MaterialID.INSULATION),
                                                int(MaterialID.STEEL),
                                                int(MaterialID.SAND)}
    assert tree.excluded.sum() == structured.excluded.sum() > 0
    assert not (tree.excluded & (tree.material_id != int(MaterialID.AIR))).any()


def test_the_bed_source_carries_the_rated_power_on_a_tree(paint_pair):
    """The bed source is a mask over the storage band: same cells, same power density."""
    power_kw = 50.0
    geometry = battery_with_power(power_kw)
    tree, structured = paint_pair(geometry)

    assert differences(tree, structured) == dict.fromkeys(PAINTED, 0)
    driven = tree.source_mask
    assert driven.any()
    assert np.array_equal(driven, tree.Q_source > 0.0)
    assert tree.Q_source[driven].min() > 0.0 and tree.Q_source[~driven].max() == 0.0
    assert float(np.sum(tree.Q_source * tree.V)) == pytest.approx(power_kw * 1000.0,
                                                                  rel=1e-12)
    # the zone is inside the storage band, and it is one volumetric density
    assert tree.Q_source[driven].min() == pytest.approx(tree.Q_source[driven].max())
    assert tree.material_id[driven].min() == int(MaterialID.SAND)


def test_the_build_report_is_the_same_on_either_mesh(paint_pair):
    """The report is the contract of ``apply_to_mesh``: same fields, same numbers."""
    tree, structured = paint_pair()
    geometry = create_small_test_geometry()

    report_tree = geometry.apply_to_mesh(tree)
    report_grid = geometry.apply_to_mesh(structured)

    assert report_tree.n_source_cells == report_grid.n_source_cells > 0
    assert report_tree.notes == report_grid.notes == []
    same_mapping(report_tree.zone_volumes, report_grid.zone_volumes)
    same_mapping(report_tree.zone_masses, report_grid.zone_masses)
    assert differences(tree, structured) == dict.fromkeys(PAINTED, 0)


def test_the_box_snapping_note_stays_where_the_box_snaps(paint_pair):
    """A tree spans its box exactly, so only a snapped ``Mesh3D`` reports a note."""
    small = Mesh3D(Lx=6.0, Ly=6.0, Lz=5.6, spacing=0.5)
    report = create_small_test_geometry().apply_to_mesh(small)
    assert any("snapped" in note for note in report.notes)

    tree, structured = paint_pair()
    assert create_small_test_geometry().apply_to_mesh(structured).notes == []
    assert create_small_test_geometry().apply_to_mesh(tree).notes == []


# ------------------------------------------------------- the excluded air + film
def test_the_air_leaves_the_problem_on_a_tree_and_the_film_is_the_same(paint_pair):
    """``apply_environment`` drops the same air leaves and fits the same film."""
    tree, structured = paint_pair()
    index = cell_of(tree, structured)

    on_grid = structured.excluded.reshape(-1, order="F")[index]
    assert np.array_equal(tree.excluded, on_grid) and tree.excluded.any()
    assert not (tree.excluded & (tree.material_id != int(MaterialID.AIR))).any()
    assert tree.h_out == pytest.approx(structured.h_out, rel=1e-12) and tree.h_out > 0.0
    assert tree.t_ambient == structured.t_ambient

    # the same call with a wind speed moves both films the same way
    geometry = create_small_test_geometry()
    film_tree = geometry.apply_environment(tree, wind_speed=5.0)
    film_grid = geometry.apply_environment(structured, wind_speed=5.0)
    same_mapping(film_tree, film_grid)
    assert tree.h_out == pytest.approx(structured.h_out, rel=1e-12)
    assert tree.h_out == pytest.approx(film_tree["total"], rel=1e-12)
    assert film_tree["total"] > film_tree["natural"]


# --------------------------------------------------------------- the validation
@pytest.mark.parametrize("bad, match", [
    (lambda: BatteryGeometry(cylinder=CylinderGeometry(center_x=3.0, center_y=3.0,
                                                       r_storage=4.5)),
     "does not fit in X"),
    (lambda: BatteryGeometry(cylinder=CylinderGeometry(center_x=3.0, center_y=3.0,
                                                       r_storage=2.0, height=7.5,
                                                       insulation_slab_bottom=0.2)),
     "roof apex"),
], ids=["too wide", "too tall"])
def test_the_geometry_is_validated_against_the_box_of_either_mesh(paint_pair, bad, match):
    """The painter's own refusals read the box, which a tree reports as a cube."""
    tree, structured = paint_pair()
    for mesh in (tree, structured):
        with pytest.raises(ValueError, match=match):
            bad().apply_to_mesh(mesh)
