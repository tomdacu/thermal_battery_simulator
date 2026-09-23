"""Buried pipe networks: the layout rules, the header plumbing and the accounting.

The point of these tests is that a network is a *design*, not a drawing: the risers
must sit inside the vessel, every riser must run from the bottom header to the top one,
the rings must be linked to each other and to the duct, the branch flows must add up to
the whole, the wetted area must be the geometric ``pi d L`` of the centreline (never
the staircase surface of the mask), and the same configuration must always build the
same network.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from src.core.adaptive_mesh import AdaptiveMesh
from src.core.materials import MaterialManager
from src.core.mesh import BoundaryType, MaterialID, Mesh3D
from src.core.pipe_network import (COLLECTION_CENTRAL, COLLECTION_DIRECT,
                                   COLLECTION_REVERSE, COLLECTION_TWO_LEVEL,
                                   LAYOUT_GRID, LAYOUT_RADIAL, LAYOUT_RINGS,
                                   LAYOUT_SPIRAL, LAYOUT_STAGGERED, LAYOUTS,
                                   MODULE_HEIGHT_LIMIT, PIPE_CARBON, PIPE_STAINLESS,
                                   ROUGHNESS_LIMIT, SPLIT_EQUAL, SPLIT_PATH,
                                   SPLIT_RING, SPLIT_SECTOR, PipeNetwork,
                                   PipeNetworkConfig, build_pipe_network)
from src.core.pipes import (HEADER_LIMIT, HEADER_SAFE, PITCH_TRIANGULAR,
                            rasterize_pipe)
from src.solver.fluid import Fluid, FluidLoop, pressure_drop


def box(spacing: float = 0.5, cells: int = 16) -> Mesh3D:
    """An 8 m cubic domain: the test vessel and its nozzle stubs fit with room over."""
    return Mesh3D(cells * spacing, cells * spacing, cells * spacing, spacing=spacing)


def vessel(**kwargs) -> PipeNetworkConfig:
    """A small but complete vessel; the pitches are explicit to keep the tests quick."""
    base = dict(radius=1.5, height=5.0, band_bottom=0.3, diameter=0.04,
                horizontal_pitch=0.2, vertical_pitch=0.25)
    base.update(kwargs)
    return PipeNetworkConfig(**base)


def network(config: PipeNetworkConfig | None = None, mesh: Mesh3D | None = None
            ) -> PipeNetwork:
    return build_pipe_network(box() if mesh is None else mesh, config or vessel())


def combinations() -> list[tuple[str, str]]:
    """Every (layout, collection) pair the design allows."""
    pairs = [(layout, collection) for layout in LAYOUTS
             for collection in (COLLECTION_DIRECT, COLLECTION_REVERSE,
                                COLLECTION_CENTRAL)]
    pairs.append((LAYOUT_RINGS, COLLECTION_TWO_LEVEL))
    return pairs


# ----------------------------------------------------------------------- geometry
@pytest.mark.parametrize("layout,collection", combinations())
def test_every_riser_sits_inside_the_vessel_between_the_two_headers(layout, collection):
    """The bundle is inside the wall and every riser spans the active band exactly."""
    config = vessel(layout=layout, collection=collection)
    net = network(config)
    assert net.validate() == []
    assert net.n_risers > 20
    x, y = net.plan_xy()
    assert np.all(np.hypot(x, y) <= config.inner_radius + 1e-9)
    assert net.bundle_radius <= config.radius - config.clearance + 1e-9
    level = config.level_gap if collection == COLLECTION_TWO_LEVEL else 0.0
    shifts = [0.0, level] if level else [0.0]
    for run in net.risers:
        assert any(run.points[0, 2] == pytest.approx(config.z_bottom + shift, abs=1e-12)
                   for shift in shifts)
        assert any(run.points[-1, 2] == pytest.approx(config.z_top + shift, abs=1e-12)
                   for shift in shifts)
        assert run.points[-1, 2] - run.points[0, 2] == pytest.approx(
            config.z_top - config.z_bottom, rel=1e-12)


def test_the_default_pitches_are_the_published_ones():
    """The lattice defaults stay the ones of ``src.core.pipes``."""
    staggered = PipeNetworkConfig(layout=LAYOUT_STAGGERED, diameter=0.05)
    assert staggered.pitch_h == pytest.approx(PITCH_TRIANGULAR * 0.05, rel=1e-12)
    assert staggered.pitch_v == pytest.approx(PITCH_TRIANGULAR * 0.05, rel=1e-12)
    # the other layouts keep the bundle convention: 2.0 d across, 2.5 d between rows
    rings = PipeNetworkConfig(layout=LAYOUT_RINGS, diameter=0.05)
    assert rings.pitch_h == pytest.approx(2.0 * 0.05, rel=1e-12)
    assert rings.pitch_v == pytest.approx(2.5 * 0.05, rel=1e-12)


def rows_of(net: PipeNetwork) -> dict[float, list[float]]:
    """The risers grouped by row: the y coordinate of the row maps to its x values."""
    rows: dict[float, list[float]] = {}
    for run in net.risers:
        rows.setdefault(round(run.points[0, 1] - net.center[1], 9), []).append(
            run.points[0, 0] - net.center[0])
    return rows


def test_the_lattice_is_one_lattice_and_the_stagger_is_half_a_pitch():
    """A grid lines its columns up; a staggered bundle puts them half a pitch across."""
    for layout, shift in ((LAYOUT_GRID, 0.0), (LAYOUT_STAGGERED, 0.5)):
        config = vessel(layout=layout)
        net = network(config)
        rows = rows_of(net)
        ys = sorted(rows)
        assert np.diff(ys) == pytest.approx(
            np.full(len(ys) - 1, config.pitch_v), rel=1e-9)
        for xs in rows.values():
            assert np.diff(sorted(xs)) == pytest.approx(
                np.full(len(xs) - 1, config.pitch_h), rel=1e-9)
        across = min(abs(first - second) for first in rows[ys[0]]
                     for second in rows[ys[1]])
        assert across == pytest.approx(shift * config.pitch_h, abs=1e-9)
        x, y = net.plan_xy()
        distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
        np.fill_diagonal(distance, np.inf)
        assert float(np.min(distance)) >= min(config.pitch_h, config.pitch_v) - 1e-9


def test_every_ring_carries_its_own_risers_and_the_rings_are_one_pitch_apart():
    """A ring main with no tube on it would be a pipe that only costs pressure."""
    config = vessel(layout=LAYOUT_RINGS)
    net = network(config)
    assert len(net.distributors) == int(np.floor(config.inner_radius / config.pitch_v
                                                 + 1e-12))
    radii = []
    for run in net.distributors:
        x = run.points[:, 0] - net.center[0]
        y = run.points[:, 1] - net.center[1]
        radii.append(float(np.max(np.hypot(x, y))))
        assert float(np.ptp(np.hypot(x, y))) == pytest.approx(0.0, abs=1e-9)
    assert radii == pytest.approx([(index + 1) * config.pitch_v
                                   for index in range(len(radii))], rel=1e-12)
    carried = {id(run): 0 for run in net.headers}
    for branch in net.branches:
        carried[id(branch.distributor)] += 1
        carried[id(branch.collector)] += 1
    assert min(carried.values()) > 0


def test_the_radial_files_run_along_the_radius_at_the_lattice_pitch():
    """A radial file is a spoke of risers: same azimuth, one pitch after the other."""
    config = vessel(layout=LAYOUT_RADIAL)
    net = network(config)
    x, y = net.plan_xy()
    angles, radii = np.arctan2(y, x), np.hypot(x, y)
    files = np.unique(np.round(angles, 9))
    assert files.size > 3
    for angle in files[:3]:
        inside = np.sort(radii[np.round(angles, 9) == angle])
        assert np.diff(inside) == pytest.approx(
            np.full(inside.size - 1, config.pitch_h), rel=1e-9)
    # the tightest pair of the whole bundle is never closer than the pitch
    distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(distance, np.inf)
    assert float(np.min(distance)) >= config.pitch_h - 1e-9


def test_the_two_level_collection_stacks_the_rings_at_two_elevations():
    """The droppers between neighbouring rings need the two levels to clear each other."""
    config = vessel(layout=LAYOUT_RINGS, collection=COLLECTION_TWO_LEVEL)
    net = network(config)
    assert net.validate() == []
    bottom = sorted({round(float(run.points[0, 2]), 9) for run in net.distributors})
    assert len(bottom) == 2
    assert bottom[1] - bottom[0] == pytest.approx(config.level_gap, rel=1e-12)
    assert len({round(run.total_length, 9) for run in net.risers}) == 1
    assert net.header_span > config.z_top - config.z_bottom
    # the two levels change the plumbing, not the balance of the chain
    assert net.path_spread() == pytest.approx(1.0, abs=1e-9)
    assert not any("direct return" in note for note in net.notes)


def test_the_central_header_takes_its_flow_along_the_vessel_axis():
    """The duct rises on the axis and the headers branch off it (the derivazioni)."""
    config = vessel(collection=COLLECTION_CENTRAL)
    net = network(config)
    for run in (net.inlet, net.outlet):
        plan = np.hypot(run.points[:, 0] - net.center[0], run.points[:, 1] - net.center[1])
        assert float(np.min(plan)) < 1e-9                 # the pipe passes the axis
    # the inlet climbs the axis from the low nozzle up to the distributor, and the
    # outlet climbs from the collector up to the high nozzle: both cross the wall
    assert float(np.max(np.abs(np.diff(net.inlet.points[:, 2])))) == pytest.approx(
        config.z_bottom - config.inlet_elevation, rel=1e-12)
    assert float(np.max(np.abs(np.diff(net.outlet.points[:, 2])))) == pytest.approx(
        config.outlet_elevation - config.z_top, rel=1e-12)
    # the wall-fed reference reaches the same bundle from the side instead of the axis
    side = network(vessel(collection=COLLECTION_DIRECT))
    assert float(np.min(np.hypot(side.inlet.points[:, 0] - side.center[0],
                                 side.inlet.points[:, 1] - side.center[1]))) > 1.0


# --------------------------------------------------------------------- the ducts
def test_the_two_nozzles_cross_the_wall_and_stay_under_the_roof():
    """Both ducts come in through the lateral wall: the roof carries the insulation."""
    config = vessel()
    net = network(config)
    for run in (net.inlet, net.outlet):
        radius = np.hypot(run.points[:, 0] - net.center[0],
                          run.points[:, 1] - net.center[1])
        assert float(np.max(radius)) > config.radius          # the stub comes from out
        assert float(np.min(radius)) < config.radius          # and reaches inside
        assert np.any(np.isclose(radius, config.radius, atol=1e-9))   # crosses the wall
        assert run.points[0, 2] == pytest.approx(run.points[1, 2], abs=1e-12)
        assert float(np.max(run.points[:, 2])) < config.roof_z
    assert config.outlet_azimuth == pytest.approx(config.azimuth_in)
    assert vessel(collection=COLLECTION_REVERSE).outlet_azimuth == pytest.approx(
        vessel().azimuth_in + 180.0)


def test_an_outlet_through_the_roof_is_refused():
    """The nozzle band is the wall: an outlet above it is a different design."""
    config = vessel(elevation_out=vessel().roof_z - 0.01)
    problems = config.validate()
    assert any("roof" in problem and "outlet" in problem for problem in problems)
    with pytest.raises(ValueError, match="roof"):
        build_pipe_network(box(), config)


def test_an_outlet_below_the_inlet_is_refused():
    """Cold in at the bottom, hot out at the top: the headers say which is which."""
    problems = vessel(elevation_in=3.0, elevation_out=2.0).validate()
    assert any("above the inlet" in problem for problem in problems)


# ---------------------------------------------------------------- the accounting
def test_the_wetted_area_is_geometric_and_adds_up_to_the_centreline_length():
    """``sum(pi d L_cell) = pi d L_total``: the mask never contributes a square metre."""
    net = network()
    cells, lengths, areas = net.voxelize()
    assert cells.size and np.all(np.diff(cells) > 0)
    assert cells.min() >= 0 and cells.max() < box().N_total
    assert np.all(lengths > 0.0) and np.all(areas > 0.0)
    assert float(np.sum(lengths)) == pytest.approx(
        sum(run.total_length for run in net.runs), rel=1e-12)
    assert float(np.sum(areas)) == pytest.approx(net.total_area, rel=1e-12)
    for run in net.runs:
        assert float(np.sum(run.area)) == pytest.approx(run.total_area, rel=1e-12)
        assert float(np.sum(run.length)) == pytest.approx(run.total_length, rel=1e-12)
    # the headers and the ducts are runs: their surface is part of the design
    assert net.total_area > net.riser_area > 0.0


def test_the_specific_area_is_the_sizing_number():
    config = vessel()
    net = network(config)
    band = config.z_top - config.z_bottom
    assert net.bed_volume == pytest.approx(np.pi * config.radius ** 2 * band, rel=1e-12)
    assert net.specific_area == pytest.approx(net.total_area / net.bed_volume, rel=1e-12)
    assert 0.5 < net.specific_area < 100.0          # a bundle worth building


def test_the_split_gives_every_branch_a_share_that_sums_to_one():
    """What ``FluidLoop`` needs: one fraction per riser, adding up to the whole flow."""
    for layout in LAYOUTS:
        for mode in (SPLIT_EQUAL, SPLIT_PATH):
            net = network(vessel(layout=layout, split_mode=mode))
            split = net.split()
            assert split.size == net.n_risers
            assert np.all(split > 0.0)
            assert float(np.sum(split)) == pytest.approx(1.0, rel=1e-15)
            if mode == SPLIT_EQUAL:
                assert split == pytest.approx(np.full(split.size, 1.0 / split.size))
            else:
                assert split == pytest.approx(
                    net.paths() / float(np.sum(net.paths())), rel=1e-12)


def test_the_reverse_return_equalises_the_branches():
    """Tichelmann plumbing: every branch runs the same header length, so it flows alike."""
    for layout in LAYOUTS:
        direct = network(vessel(layout=layout, collection=COLLECTION_DIRECT))
        reverse = network(vessel(layout=layout, collection=COLLECTION_REVERSE))
        assert direct.path_spread() > 1.05
        assert any("direct return" in note for note in direct.notes)
        assert reverse.path_spread() == pytest.approx(1.0, abs=1e-9)
        assert not any("direct return" in note for note in reverse.notes)
        # equal paths mean the path-weighted split is the equal split
        weighted = network(vessel(layout=layout, collection=COLLECTION_REVERSE,
                                  split_mode=SPLIT_PATH))
        assert weighted.split() == pytest.approx(
            np.full(weighted.n_risers, 1.0 / weighted.n_risers), rel=1e-9)


def test_the_three_metre_rule_is_reported_with_the_collector_elevations():
    """Below 1 m no care, 1-3 m check the distribution, above 3 m split the bundle."""
    tall = network(vessel(height=7.0, band_bottom=0.5))
    assert tall.header_span == pytest.approx(tall.config.z_top - tall.config.z_bottom,
                                             rel=1e-12)
    assert tall.header_span > HEADER_LIMIT
    assert "split into parallel modules" in tall.header_rule()
    assert any("split into identical parallel modules" in note for note in tall.notes)
    text = tall.summary()
    assert f"{tall.config.z_bottom:.2f} m" in text and f"{tall.config.z_top:.2f} m" in text
    assert "3 m rule" in text

    middle = network(vessel(height=3.0))
    assert HEADER_SAFE < middle.header_span <= HEADER_LIMIT
    assert "check the flow distribution" in middle.header_rule()
    assert any("check the flow distribution" in note for note in middle.notes)

    short = network(vessel(height=1.4, band_bottom=0.2))
    assert short.header_span == pytest.approx(1.0, rel=1e-12)
    assert short.header_rule() == f"no care needed below {HEADER_SAFE:.0f} m"
    assert not any("split" in note for note in short.notes)


def test_a_tall_bed_is_flagged_for_the_module_height():
    """The published practice keeps one module below four metres of bed."""
    net = network(vessel(height=6.0, band_bottom=0.3))
    assert net.header_span > MODULE_HEIGHT_LIMIT
    assert any("module" in note for note in net.notes)


def test_the_summary_reports_the_design_numbers():
    net = network(vessel(layout=LAYOUT_RINGS, collection=COLLECTION_REVERSE))
    text = net.summary()
    assert f"{net.n_risers} risers" in text
    assert f"{net.config.pitch_h * 1000:.0f} mm horizontal" in text
    assert f"{net.specific_area:.2f} m2/m3" in text
    assert f"{net.total_length:.0f} m" in text
    assert "split equal" in text
    data = net.summarize()
    assert data["n_risers"] == net.n_risers
    assert data["header_rule"] == net.header_rule()
    assert data["path_spread"] == pytest.approx(net.path_spread(), rel=1e-12)


# ------------------------------------------------------------------- the guards
def test_a_ring_with_no_riser_on_it_is_refused():
    """A ring main that feeds nothing is a piece of plumbing that cannot work."""
    net = network(vessel(layout=LAYOUT_RINGS))
    assert net.validate() == []
    ring = net.distributors[0]
    net.branches = [branch for branch in net.branches if branch.distributor is not ring]
    problems = net.validate()
    assert any(ring.name in problem and "carries no riser" in problem
               for problem in problems)


def test_a_ring_nobody_linked_to_the_duct_is_refused():
    """Nested rings are a chain: a ring that hangs in the sand is not connected."""
    net = network(vessel(layout=LAYOUT_RINGS))
    stray = rasterize_pipe(box(), [(0.5, 0.5, 1.0), (0.9, 0.5, 1.0)], 0.04,
                           name="bottom_ring_99")
    net.distributors.append(stray)
    problems = net.validate()
    assert any("bottom_ring_99" in problem and "not connected" in problem
               for problem in problems)


def test_a_riser_that_misses_its_collector_is_refused():
    """Every riser runs from the bottom header to the top one: no floating tubes."""
    net = network()
    branch = net.branches[0]
    moved = rasterize_pipe(box(), [(0.5, 0.5, 0.2), (0.5, 0.5, 0.4)], 0.04,
                           name="riser_0")
    net.branches[0] = replace(branch, riser=moved)
    net.risers[0] = moved
    problems = net.validate()
    assert any("riser 0" in problem and "does not" in problem for problem in problems)
    # the branch keeps the header arcs it had: only the tube moved
    assert net.branches[0].feed_length == pytest.approx(branch.feed_length, rel=1e-12)


@pytest.mark.parametrize("changes,expected", [
    ({"horizontal_pitch": 0.01}, "pitch"),
    ({"vertical_pitch": 0.01}, "pitch"),
    ({"collection": COLLECTION_TWO_LEVEL}, "layout"),
    ({"layout": LAYOUT_RINGS, "n_rings": 30}, "n_rings"),
    ({"layout": "elliptical"}, "unknown layout"),
    ({"collection": "thermosiphon"}, "unknown collection"),
    ({"split_mode": "proportional"}, "unknown split mode"),
    ({"split_mode": SPLIT_RING}, "needs layout"),
    ({"material": "copper"}, "unknown tube material"),
    ({"wall_thickness": -0.001}, "wall thickness"),
    ({"wall_thickness": 0.03}, "no bore"),
    ({"roughness": -1e-5}, "absolute roughness"),
    ({"junction_refinement": 0.0}, "junction refinement"),
    ({"n_sectors": 0}, "n_sectors"),
    ({"band_top": 9.0}, "above the vessel wall"),
    ({"band_bottom": 0.3, "band_top": 0.1}, "no height"),
    ({"radius": 0.05}, "clearance"),
    ({"wall_clearance": 4.0}, "clearance"),
    ({"wall_clearance": -0.5}, "clearance"),
    ({"elevation_in": -1.0}, "does not fit inside the vessel"),
])
def test_a_configuration_that_cannot_be_built_says_which_parameter_to_change(
        changes, expected):
    """``validate()`` is the designer's guard rail: it names the knob, not the symptom."""
    config = vessel(**changes)
    problems = config.validate()
    assert any(expected in problem for problem in problems)
    with pytest.raises(ValueError):
        build_pipe_network(box(), config)


def test_a_warning_does_not_stop_the_build():
    """A duct that crosses the active band is legal plumbing, so it only warns."""
    config = vessel(elevation_in=1.2)
    problems = config.validate()
    assert any(problem.startswith("warning:") for problem in problems)
    net = build_pipe_network(box(), config)
    assert net.n_risers > 0
    assert not any(problem.startswith("warning:") for problem in net.validate())


# -------------------------------------------------------------- reproducibility
def test_the_same_configuration_always_builds_the_same_network():
    """No randomness, no dependence on the history: a design is reproducible."""
    config = vessel(layout=LAYOUT_RINGS, collection=COLLECTION_REVERSE)
    first = network(config)
    second = network(config)
    assert first.summary() == second.summary()
    cells_a, length_a, area_a = first.voxelize()
    cells_b, length_b, area_b = second.voxelize()
    assert np.array_equal(cells_a, cells_b)
    assert np.array_equal(length_a, length_b)
    assert np.array_equal(area_a, area_b)
    for run_a, run_b in zip(first.runs, second.runs, strict=True):
        assert run_a.name == run_b.name
        assert np.array_equal(run_a.points, run_b.points)
        assert np.array_equal(run_a.cells, run_b.cells)


def test_the_vessel_can_be_put_anywhere_in_the_domain():
    """The centre is a parameter: the same design translated is the same design."""
    config = vessel()
    mesh = box(cells=20)
    centred = build_pipe_network(mesh, config)
    moved = build_pipe_network(mesh, config, center=(6.0, 5.0))
    shift = np.asarray([6.0 - centred.center[0], 5.0 - centred.center[1], 0.0])
    for run_a, run_b in zip(centred.runs, moved.runs, strict=True):
        assert run_b.points == pytest.approx(run_a.points + shift, abs=1e-12)


def test_the_voxelisation_can_move_to_another_grid():
    """A finer grid re-rasterises the same centrelines: the area is the same physical one."""
    config = vessel(radius=1.0, horizontal_pitch=0.25, vertical_pitch=0.25)
    coarse = network(config)
    fine_mesh = box(spacing=0.25, cells=32)
    fine = coarse.voxelize(fine_mesh)
    assert fine[0].size > coarse.voxelize()[0].size
    assert float(np.sum(fine[2])) == pytest.approx(coarse.total_area, rel=1e-12)


# --------------------------------------------------------- the fluid integration
def test_the_network_drives_the_fluid_loop_with_its_branch_split():
    """The risers and their split are what the 1-D march needs; the bed splits the flow."""
    mesh = box()
    config = vessel(radius=1.0, height=2.0, band_bottom=0.2, horizontal_pitch=0.3,
                    vertical_pitch=0.3, elevation_out=1.8)
    net = build_pipe_network(mesh, config)
    for face in ("x_min", "x_max", "y_min", "y_max", "z_max"):
        mesh.set_adiabatic(face)
    mesh.T[:] = 500.0
    result = FluidLoop(runs=net.risers, mass_flow=0.05, h_fluid=500.0, t_in=300.0,
                       split=net.split()).solve(mesh)
    assert len(result.runs) == net.n_risers
    share = net.split()
    assert [run.mass_flow for run in result.runs] == pytest.approx(list(0.05 * share),
                                                                  rel=1e-9)
    enthalpy = 0.05 * 1005.0 * (result.t_out - result.t_in)
    assert result.power == pytest.approx(-enthalpy, rel=1e-9)


# ------------------------------------------------------- the tube and its options
def test_the_wall_thickness_sets_the_bore_the_gas_flows_through():
    """The outer diameter is the geometry, the bore is the hydraulics: both reported."""
    config = vessel(wall_thickness=0.003, duct_diameter=0.1)
    assert config.inner_diameter == pytest.approx(
        config.diameter - 2.0 * config.wall_thickness, rel=1e-12)
    assert config.duct_inner_diameter == pytest.approx(
        config.duct_d - 2.0 * config.wall_thickness, rel=1e-12)
    net = network(config)
    assert net.inner_diameter == pytest.approx(config.inner_diameter, rel=1e-12)
    data = net.summarize()
    assert data["inner_diameter"] == pytest.approx(config.inner_diameter, rel=1e-12)
    assert f"{config.inner_diameter * 1000:.1f} mm" in net.summary()
    # the geometry keeps the outer diameter: the pitches, the clearance and the area
    assert net.risers[0].diameter == pytest.approx(config.diameter, rel=1e-12)
    assert net.total_area == pytest.approx(
        sum(run.perimeter * run.total_length for run in net.runs), rel=1e-12)
    # a thicker wall is a smaller bore and a rougher pipe, relative to its own bore
    thin = vessel(wall_thickness=0.001)
    thick = vessel(wall_thickness=0.005)
    assert thick.inner_diameter < thin.inner_diameter
    assert thick.relative_roughness > thin.relative_roughness


def test_the_tube_material_carries_the_label_and_the_roughness():
    """Stainless or carbon: the same geometry, a different wall and a different eps."""
    steel = vessel(material=PIPE_STAINLESS)
    carbon = vessel(material=PIPE_CARBON)
    assert steel.absolute_roughness < carbon.absolute_roughness
    assert steel.relative_roughness == pytest.approx(
        steel.absolute_roughness / steel.inner_diameter, rel=1e-12)
    text = network(carbon).summary()
    assert "carbon steel" in text and "relative roughness" in text
    # an explicit roughness overrides the material, and the material still labels it
    rough = vessel(material=PIPE_STAINLESS, roughness=2e-3)
    assert rough.absolute_roughness == pytest.approx(2e-3, rel=1e-12)
    problems = rough.validate()
    assert any(problem.startswith("warning:") and "Moody" in problem
               for problem in problems)
    assert rough.relative_roughness > ROUGHNESS_LIMIT


def test_the_lagged_headers_leave_the_risers_as_the_exchange_surface():
    """An insulated header carries the gas and nothing else: the tubes do the work."""
    bare = network(vessel())
    lagged = network(vessel(insulated_headers=True))
    assert bare.exchange_area == pytest.approx(bare.total_area, rel=1e-12)
    assert lagged.total_area == pytest.approx(bare.total_area, rel=1e-12)
    assert lagged.exchange_area == pytest.approx(
        lagged.total_area - sum(run.total_area for run in lagged.headers), rel=1e-12)
    assert lagged.exchange_area < lagged.total_area
    assert "insulated" in lagged.summary()
    assert lagged.summarize()["insulated_headers"] is True


def test_the_junction_refinement_names_the_band_the_mesh_must_resolve():
    """The tube-header junction is a surface and a turn: the band around it is refined."""
    plain = vessel()
    assert plain.junction_bands() == []
    config = vessel(junction_refinement=0.02)
    net = network(config)
    bands = config.junction_bands()
    assert len(bands) == 2
    assert [band[2] for band in bands] == pytest.approx([0.02, 0.02])
    assert bands[0][0] < config.z_bottom < bands[0][1]
    assert bands[1][0] < config.z_top < bands[1][1]
    assert f"{config.junction_refinement * 1000:.1f} mm" in net.summary()
    # a refinement coarser than the tube it refines cannot resolve the junction
    coarse = vessel(junction_refinement=3.0 * vessel().diameter)
    assert any(problem.startswith("warning:") and "coarser" in problem
               for problem in coarse.validate())


# ------------------------------------------------------------- the distributions
def test_the_sector_split_gives_every_sector_the_same_flow():
    """A sector valve per quadrant: the flow follows the sector, not the lattice."""
    config = vessel(n_sectors=4, split_mode=SPLIT_SECTOR)
    net = network(config)
    assert config.validate() == []
    split = net.split()
    assert float(np.sum(split)) == pytest.approx(1.0, rel=1e-12)
    x, y = net.plan_xy()
    sector = np.floor(((np.arctan2(y, x) - np.deg2rad(config.azimuth_in))
                       % (2.0 * np.pi)) / (2.0 * np.pi / config.n_sectors))
    for index in range(config.n_sectors):
        inside = sector == index
        assert np.count_nonzero(inside) > 0
        # the whole sector carries 1/n_sectors of the flow, its risers share it
        assert float(np.sum(split[inside])) == pytest.approx(
            1.0 / config.n_sectors, rel=1e-12)
        assert float(np.std(split[inside])) < 1e-12
    # a sector nobody feeds is refused with the parameter to change
    empty = vessel(n_sectors=64, split_mode=SPLIT_SECTOR)
    problems = empty.validate()
    assert any("hold no riser" in problem and "n_sectors" in problem
               for problem in problems)
    with pytest.raises(ValueError):
        build_pipe_network(box(), empty)


def test_the_ring_split_gives_every_ring_main_the_same_flow():
    """A manifold per ring: the ring main gets its share, its taps divide it."""
    config = vessel(layout=LAYOUT_RINGS, split_mode=SPLIT_RING)
    net = network(config)
    assert config.validate() == []
    split = net.split()
    assert float(np.sum(split)) == pytest.approx(1.0, rel=1e-12)
    rings = np.asarray(net.groups)
    assert rings.size == net.n_risers
    counts = {}
    for ring, share in zip(rings, split, strict=True):
        counts.setdefault(int(ring), []).append(float(share))
    assert len(counts) == len(net.distributors)
    for shares in counts.values():
        assert float(np.sum(shares)) == pytest.approx(
            1.0 / len(net.distributors), rel=1e-12)
    # the inner rings have fewer taps, so each of their tubes carries more
    inner = min(counts, key=lambda ring: len(counts[ring]))
    outer = max(counts, key=lambda ring: len(counts[ring]))
    assert counts[inner][0] > counts[outer][0]


# ------------------------------------------------------------ painting the mesh
def test_the_paint_marks_the_pipes_inside_the_vessel_and_keeps_the_area():
    """What ``paint`` marks is a pipe cell of the bed, never a cell outside the wall."""
    mesh = box(spacing=0.25, cells=32)
    config = vessel(insulated_headers=True)
    net = build_pipe_network(mesh, config)
    report = net.paint(mesh, h_fluid=350.0, t_fluid=333.15)
    tubes = mesh.material_id == int(MaterialID.TUBES)
    assert report.cells == int(np.count_nonzero(tubes))
    assert report.cells > 0 and report.riser_cells > 0
    # every marked cell has its centre inside the vessel
    centre_z = mesh.Z
    radius = np.hypot(mesh.X - net.center[0], mesh.Y - net.center[1])
    low, high = config.base_z - 1e-9, config.roof_z + 1e-9
    inside = ((radius <= config.radius + 1e-9) & (low <= centre_z)
              & (centre_z <= high))
    assert not np.any(tubes & ~inside)
    # the area is still the geometric one: what was not painted is accounted for
    assert report.area + report.dropped == pytest.approx(net.total_area, rel=1e-12)
    assert report.area == pytest.approx(float(np.sum(net.voxelize(mesh)[2])) -
                                        report.dropped, rel=1e-12)
    assert net.total_area == pytest.approx(
        sum(run.perimeter * run.total_length for run in net.runs), rel=1e-12)
    # the film, the material and the sources
    assert mesh.bc_h[tubes].max() == pytest.approx(350.0, rel=1e-12)
    assert np.all(mesh.bc_T_inf[tubes] == pytest.approx(333.15, rel=1e-12))
    assert mesh.boundary_type[tubes].max() == int(BoundaryType.CONVECTION)
    # the cell stays what it was (a thin pipe in the bed): the box is air here
    assert mesh.rho[tubes].min() == pytest.approx(
        MaterialManager().get("air").rho, rel=1e-12)
    assert float(np.abs(mesh.Q_source[tubes]).sum()) == 0.0
    assert not mesh.source_mask[tubes].any()
    assert report.material == config.material


def test_the_paint_marks_the_pipes_of_a_tree_inside_the_vessel_and_keeps_the_area():
    """The same paint on the adaptive mesh: the tube cells of the bed, outside the wall no.

    What this pins is what the mesh's own fields end up holding on the mesh the port is
    for - a pipe material, a gas film, no source - and that the report accounts for the
    stubs exactly as it does on a grid.  The cells are the leaves, so the mask, the film
    and the volumes it is written through are the mesh's own.
    """
    # the tree of the same 8 m cube the `box()` helper spans: 16 leaves a side at 0.5 m
    mesh = AdaptiveMesh.uniform(16, 0.5, level=0)
    config = vessel(insulated_headers=True)
    net = build_pipe_network(mesh, config)
    report = net.paint(mesh, h_fluid=350.0, t_fluid=333.15)
    tubes = mesh.material_id == int(MaterialID.TUBES)
    assert report.cells == int(np.count_nonzero(tubes))
    assert report.cells > 0 and report.riser_cells > 0
    # every marked leaf has its centre inside the vessel
    centres = mesh.centres()
    radius = np.hypot(centres[:, 0] - net.center[0], centres[:, 1] - net.center[1])
    low, high = config.base_z - 1e-9, config.roof_z + 1e-9
    inside = ((radius <= config.radius + 1e-9) & (low <= centres[:, 2])
              & (centres[:, 2] <= high))
    assert not np.any(tubes & ~inside)
    # the area is still the geometric one: the stubs are accounted for, not invented
    assert report.area + report.dropped == pytest.approx(net.total_area, rel=1e-12)
    # the film, the material and the sources
    assert mesh.bc_h[tubes].max() == pytest.approx(350.0, rel=1e-12)
    assert np.all(mesh.bc_T_inf[tubes] == pytest.approx(333.15, rel=1e-12))
    assert mesh.boundary_type[tubes].max() == int(BoundaryType.CONVECTION)
    # the cell stays what it was (a thin pipe in the bed): the box is air here
    assert mesh.rho[tubes].min() == pytest.approx(
        MaterialManager().get("air").rho, rel=1e-12)
    assert float(np.abs(mesh.Q_source[tubes]).sum()) == 0.0
    assert not mesh.source_mask[tubes].any()
    assert report.material == config.material


def test_a_lagged_header_is_painted_without_a_gas_film():
    """The lagging is a fact of the model: a pipe the gas does not heat the sand with."""
    mesh = box(spacing=0.125, cells=64)
    config = vessel(radius=1.5, height=5.0, band_bottom=0.4, diameter=0.1,
                    horizontal_pitch=0.8, vertical_pitch=0.8,
                    insulated_headers=True)
    net = build_pipe_network(mesh, config)
    report = net.paint(mesh)
    tubes = mesh.material_id == int(MaterialID.TUBES)
    # the paint writes no film value (the gas loop does, per cell): an exchanging cell
    # is one the assembly lists as convective, a lagged one is not
    film = tubes & (mesh.boundary_type == int(BoundaryType.CONVECTION))
    assert report.riser_cells < report.cells          # the headers have cells of their own
    assert int(np.count_nonzero(tubes & ~film)) > 0   # and they carry no film
    assert report.insulated == pytest.approx(
        net.total_area - net.exchange_area, rel=1e-12)
    bare = build_pipe_network(mesh, vessel(radius=1.5, height=5.0, band_bottom=0.4,
                                          diameter=0.1, horizontal_pitch=0.8,
                                          vertical_pitch=0.8))
    bare.paint(mesh)
    assert int(np.count_nonzero(
        mesh.boundary_type[tubes] == int(BoundaryType.CONVECTION))) == int(tubes.sum())


# ------------------------------------------------------- the circuit of the loop
def test_the_hydraulics_adds_the_riser_to_the_pipes_the_branches_share():
    """The branch pays for its tube, its header arcs and the two ducts it crosses."""
    net = network(vessel(duct_diameter=0.1))
    mass_flow = 0.05
    circuit = net.hydraulics(mass_flow)
    assert circuit.branch_drop.size == net.n_risers
    assert np.all(circuit.riser_drop > 0.0) and np.all(circuit.shared_drop > 0.0)
    assert circuit.duct_length == pytest.approx(net.inlet.total_length +
                                                net.outlet.total_length, rel=1e-12)
    branch = net.branches[0]
    share = net.split()[0]
    fluid = Fluid()
    expected = pressure_drop(mass_flow * share, net.inner_diameter,
                             branch.riser.total_length, fluid, net.roughness)
    assert circuit.riser_drop[0] == pytest.approx(expected, rel=1e-12)
    # the equivalent fittings reproduce the shared drop at the velocity of the bore
    velocity = mass_flow * share / (fluid.rho * 0.25 * np.pi * net.inner_diameter ** 2)
    assert circuit.fittings_k * 0.5 * fluid.rho * velocity ** 2 == pytest.approx(
        float(np.mean(circuit.shared_drop)), rel=1e-9)
    assert circuit.mean_drop > circuit.riser_drop.mean()
    assert circuit.spread >= 1.0
    assert "circuit" in circuit.summary()
    with pytest.raises(ValueError):
        net.hydraulics(0.0)


def test_the_loop_built_from_the_network_marches_with_its_split_and_keeps_enthalpy():
    """One run per branch, the network's split, the circuit's pressure drop."""
    mesh = box(spacing=0.25, cells=32)
    config = vessel(radius=1.0, height=2.0, band_bottom=0.2, horizontal_pitch=0.3,
                    vertical_pitch=0.3, duct_diameter=0.1, wall_thickness=0.003,
                    elevation_out=1.8)
    net = build_pipe_network(mesh, config)
    net.paint(mesh)
    for face in ("x_min", "x_max", "y_min", "y_max", "z_max"):
        mesh.set_adiabatic(face)
    mesh.T[:] = 500.0
    mass_flow = 0.05
    # the reference: the risers in parallel at constant gas properties
    loop = net.fluid_loop(mass_flow, t_in=300.0, well_model=False,
                          variable_properties=False)
    assert loop.runs[0].diameter == pytest.approx(config.inner_diameter, rel=1e-12)
    assert [run.name for run in loop.runs] == [run.name for run in net.risers]
    assert loop.split == pytest.approx(list(net.split()), rel=1e-12)
    result = loop.solve(mesh)
    assert len(result.runs) == net.n_risers
    assert [run.mass_flow for run in result.runs] == pytest.approx(
        list(mass_flow * net.split()), rel=1e-9)
    enthalpy = mass_flow * loop.fluid.cp * (result.t_out - result.t_in)
    assert result.power == pytest.approx(-enthalpy, rel=1e-9)
    # the circuit's pressure drop is the one the hydraulics reports, fittings included
    assert result.delta_p == pytest.approx(net.hydraulics(mass_flow).mean_drop,
                                           rel=1e-9)
    # the exchange lands in the pipe cells the paint marked
    tubes = (mesh.material_id == int(MaterialID.TUBES)).ravel(order="F")
    assert result.q_fluid is not None
    assert np.all(result.q_fluid[~tubes] == 0.0)
    assert float(np.sum(result.q_fluid[tubes])) != 0.0


def test_the_loop_can_be_built_on_the_mesh_the_solve_will_use():
    """A mesh built after the design re-rasterises the risers, the split does not move."""
    coarse = box(spacing=0.5)
    config = vessel(radius=1.0, height=2.0, band_bottom=0.2, horizontal_pitch=0.3,
                    vertical_pitch=0.3, elevation_out=1.8)
    net = build_pipe_network(coarse, config)
    fine = box(spacing=0.25, cells=32)
    loop = net.fluid_loop(0.05, mesh=fine)
    assert loop.runs[0].cells.size > net.risers[0].cells.size
    assert loop.split == pytest.approx(list(net.split()), rel=1e-12)
    assert loop.runs[0].area.sum() == pytest.approx(
        net.inner_diameter * np.pi * net.risers[0].total_length, rel=1e-12)


# ------------------------------------------------------------------- the spiral
def test_the_spiral_layout_winds_the_tubes_one_pitch_per_turn():
    """An Archimedean spiral: the arc steps are horizontal pitches, the turns vertical."""
    config = vessel(layout=LAYOUT_SPIRAL)
    net = network(config)
    assert net.validate() == []
    assert net.n_risers > 20
    x, y = net.plan_xy()
    radii = np.hypot(x, y)
    assert np.all(np.diff(radii) <= 1e-12)                # the walk winds inwards
    assert int(np.argmax(radii)) == 0                     # and starts at the wall
    assert float(radii[0]) == pytest.approx(config.inner_radius, rel=1e-12)
    assert float(radii[-1]) >= 0.5 * config.pitch_h - 1e-9
    assert float(radii[0]) - float(radii[-1]) > 2.0 * config.pitch_v   # several turns
    # the turns are one vertical pitch apart, the steps along the walk one horizontal
    theta = np.unwrap(np.arctan2(y, x))
    assert np.all(np.diff(theta) <= 1e-12)                # one way round, no zig-zag
    assert float(abs(theta[0] - theta[-1])) == pytest.approx(
        2.0 * np.pi * (float(radii[0]) - float(radii[-1])) / config.pitch_v, rel=1e-9)
    step = np.hypot(np.diff(x), np.diff(y))
    assert float(np.max(step)) <= config.pitch_h + 1e-9
    assert float(np.min(step)) >= 0.5 * config.pitch_h - 1e-9
    distance = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    np.fill_diagonal(distance, np.inf)
    assert float(np.min(distance)) >= 0.5 * min(config.pitch_h, config.pitch_v) - 1e-9
    assert float(np.min(distance)) > config.diameter      # the tubes never touch
    assert LAYOUT_SPIRAL in LAYOUTS
    assert "spiral" in net.summary()


# ------------------------------------------------------- the surface power rating
def test_the_power_of_the_plant_is_rated_on_the_surface_of_the_pipes():
    """The wall that delivers the power is the tubes': P over the wetted area.

    The rating window the immersion-element practice uses (3-8 W/cm2) is applied to the
    surface the gas actually crosses - the geometric ``pi d L`` of the network, which is
    not read off the voxel mask - and a network with no surface has no rating to report.
    """
    from src.constants import (PIPE_SURFACE_POWER_LIMIT_W_CM2,
                               PIPE_SURFACE_POWER_MIN_W_CM2)
    from src.core.pipes import pipe_surface_power_w_cm2

    net = network()
    area = net.total_area                                    # ~ pi d L of every run
    assert area > 0.0
    # a busbar's power over this surface is far past the window, a kettle's far below it
    assert pipe_surface_power_w_cm2(1.0e7, area) > PIPE_SURFACE_POWER_LIMIT_W_CM2
    assert pipe_surface_power_w_cm2(1.0e5, area) < PIPE_SURFACE_POWER_MIN_W_CM2
    # 1 m2 carrying 10 kW is exactly 1 W/cm2: the conversion is the point of the helper
    assert pipe_surface_power_w_cm2(10_000.0, 1.0) == pytest.approx(1.0)
    assert pipe_surface_power_w_cm2(1.0e6, 0.0) == float("inf")


def test_a_ring_count_spreads_the_rings_over_the_whole_bed():
    """Fewer rings than the pitch fits are spread over the radius, not packed at the axis."""
    from src.core.pipe_network import riser_positions

    config = vessel(layout=LAYOUT_RINGS, n_rings=4)
    points = np.asarray(riser_positions(config, (0.0, 0.0)))
    radii = np.unique(np.round(np.hypot(points[:, 0], points[:, 1]), 9))
    assert len(radii) == 4
    assert radii[-1] == pytest.approx(config.inner_radius, rel=1e-9)
    assert radii[0] == pytest.approx(config.inner_radius / 4, rel=1e-9)
    # the risers on a ring are spaced about like the rings: similar sand per riser
    step = config.inner_radius / 4
    for radius in radii:
        on_ring = int(np.sum(np.isclose(np.hypot(points[:, 0], points[:, 1]), radius)))
        assert 2 * np.pi * radius / on_ring >= step - 1e-9
        assert 2 * np.pi * radius / on_ring < 2.5 * step
