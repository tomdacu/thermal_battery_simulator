"""The circuit's hydraulics and the header engine (``src/solver/hydraulics.py``).

Pinned: a single pipe drops what Darcy-Weisbach says; two parallel pipes share the flow
as their resistances say; a hot riser draws at a low flow and starves at a high one;
the balancing orifices put every riser exactly on its target; and the engine, on a ring
network, picks the radial manifold over the chain, keeps every header under the velocity
limit, lifts the headers into the sand and hands back a network whose own hydraulics
deliver the target flows.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.solver.fluid import Fluid, pressure_drop
from src.solver.hydraulics import HydraulicNetwork, Pipe, voronoi_shares


def _pipe(name, a, b, start, end, d, k=0.0, riser=-1, group=""):
    return Pipe(name=name, a=a, b=b, start=np.asarray(start, float),
                end=np.asarray(end, float), bore=d, outer=d + 0.004, k_local=k,
                group=group, kind="riser" if riser >= 0 else "header", riser=riser)


def test_one_pipe_drops_what_darcy_weisbach_says():
    net = HydraulicNetwork([_pipe("p", 0, 1, (0, 0, 0), (10, 0, 0), 0.05)], 2,
                           roughness=4.5e-5)
    fluid = Fluid()
    state = net.solve(0.05, fluid, 300.0)
    expected = pressure_drop(0.05, 0.05, 10.0, fluid.at_pressure(101325.0, 300.0),
                             roughness=4.5e-5)
    assert state.converged
    assert state.delta_p == pytest.approx(expected, rel=2e-2)   # Haaland vs max(64/Re)


def test_two_parallel_pipes_share_the_flow_by_their_resistance():
    # 0 -> 2 through a short and a long pipe, then 2 -> 1
    net = HydraulicNetwork([
        _pipe("short", 0, 2, (0, 0, 0), (5, 0, 0), 0.05, riser=0),
        _pipe("long", 0, 2, (0, 0, 0), (20, 0, 0), 0.05, riser=1),
        _pipe("out", 2, 1, (5, 0, 0), (6, 0, 0), 0.2)], 3)
    state = net.solve(0.2, Fluid(), 300.0)
    short, long = state.riser_flow
    assert short + long == pytest.approx(0.2, rel=1e-9)
    # turbulent: dp ~ f L m^2, so m ~ 1/sqrt(L) up to the friction factor
    assert short / long == pytest.approx(2.0, rel=0.1)


def test_the_hot_riser_draws_at_low_flow_and_starves_at_high_flow():
    """Two parallel risers, one at 900 K and one at 300 K.  The hot gas is three times
    lighter: its column draws ((rho_cold - rho_hot) g H ~ 39 Pa over 5 m), but at the same
    mass flow it runs three times faster and loses three times more to friction.  At a low
    flow the draught wins and the hot riser takes more; at a high flow the friction wins
    and it takes less - the hot-channel starvation of forced flow."""
    risers = [_pipe(f"r{i}", 2, 3, (i, 0, 0), (i, 0, 5), 0.046, k=2.0, riser=i)
              for i in range(2)]
    net = HydraulicNetwork([_pipe("in", 0, 2, (0, 0, 0), (0, 0, 0.1), 0.2),
                            *risers,
                            _pipe("out", 3, 1, (0, 0, 5), (0, 0, 5.1), 0.2)], 4)
    temps = np.array([300.0, 900.0, 300.0, 300.0])       # riser 0 hot, riser 1 cold
    low = net.solve(0.002, Fluid(), temps)
    assert low.riser_flow[0] > 1.05 * low.riser_flow[1]
    high = net.solve(0.03, Fluid(), temps)
    assert high.riser_flow[0] < 0.95 * high.riser_flow[1]
    assert np.sum(high.riser_flow) == pytest.approx(0.03, rel=1e-9)


def test_the_orifices_put_every_riser_on_its_target():
    # a ladder of five risers fed from one end, returned from the same end (direct)
    pipes = []
    feet, heads = list(range(2, 7)), list(range(7, 12))
    pipes.append(_pipe("in", 0, feet[0], (-1, 0, 0), (0, 0, 0), 0.08, group="d"))
    for i in range(4):
        pipes.append(_pipe(f"d{i}", feet[i], feet[i + 1], (i, 0, 0), (i + 1, 0, 0),
                           0.08, group="d"))
        pipes.append(_pipe(f"c{i}", heads[i + 1], heads[i], (i + 1, 0, 4), (i, 0, 4),
                           0.08, group="c"))
    for i in range(5):
        pipes.append(_pipe(f"r{i}", feet[i], heads[i], (i, 0, 0), (i, 0, 4), 0.046,
                           k=2.0, riser=i))
    pipes.append(_pipe("out", heads[0], 1, (0, 0, 4), (-1, 0, 4), 0.08, group="c"))
    net = HydraulicNetwork(pipes, 12)
    targets = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    fluid = Fluid()
    free = net.solve(0.1, fluid, 600.0)
    assert np.max(np.abs(free.riser_flow / (0.1 * targets) - 1.0)) > 0.1
    balancing = net.balance(0.1, fluid, targets, 600.0)
    assert np.min(balancing.k_orifice) == pytest.approx(0.0, abs=1e-9)
    balanced = net.with_orifices(balancing.k_orifice).solve(0.1, fluid, 600.0)
    assert np.allclose(balanced.riser_flow, 0.1 * targets, rtol=1e-8)
    assert balanced.delta_p == pytest.approx(balancing.delta_p, rel=1e-6)


def test_voronoi_shares_cover_the_disc():
    points = np.array([[0.5, 0.0], [-0.5, 0.0], [0.0, 0.5], [0.0, -0.5]])
    shares = voronoi_shares(points, 1.0)
    assert np.allclose(shares, 0.25, atol=0.01)


@pytest.fixture(scope="module")
def ring_design():
    from src.core.pipe_network import (COLLECTION_REVERSE, LAYOUT_RINGS,
                                       PipeNetworkConfig, design_headers)

    config = PipeNetworkConfig(radius=1.4, height=3.4, base_z=1.3, band_bottom=0.2,
                               band_top=3.2, diameter=0.05, wall_thickness=0.002,
                               layout=LAYOUT_RINGS, collection=COLLECTION_REVERSE,
                               n_rings=3, duct_diameter=0.15, azimuth_in=180.0,
                               azimuth_out=0.0)
    design = design_headers(config, (2.5, 2.5), 0.25, Fluid(), tolerance=0.05,
                            max_velocity=20.0, temperature=773.15)
    return config, design


def test_the_engine_prefers_the_manifold_and_meets_the_velocity(ring_design):
    config, design = ring_design
    assert design.config.collection == "radial_manifold"
    assert design.sizing.feasible
    assert design.sizing.max_velocity <= 20.0 + 1e-6
    assert any("also tried reverse_return" in note for note in design.sizing.notes)


def test_the_engine_lifts_the_headers_into_the_sand(ring_design):
    config, design = ring_design
    assert design.config.z_bottom == pytest.approx(config.z_bottom + design.lift)
    assert design.config.z_top == pytest.approx(config.z_top - design.lift)


def test_the_designed_network_delivers_its_targets(ring_design):
    from src.core.adaptive_mesh import AdaptiveMesh
    from src.core.pipe_network import build_pipe_network

    _config, design = ring_design
    network = build_pipe_network(AdaptiveMesh.uniform(2, 2.5, level=0), design.config,
                                 center=(2.5, 2.5))
    state = network.hydraulic_network().solve(0.25, Fluid(), 773.15)
    assert np.allclose(state.riser_flow / 0.25, design.sizing.targets,
                       rtol=0.05 + 1e-9)
    # the split the loop uses is the hydraulics'
    assert np.allclose(network.split(), state.riser_flow / 0.25, rtol=1e-6)


def test_the_loop_follows_the_hydraulics_of_the_hot_gas(ring_design):
    from src.core.adaptive_mesh import AdaptiveMesh
    from src.core.pipe_network import build_pipe_network

    _config, design = ring_design
    mesh = AdaptiveMesh.from_box(16, 16, 0.3125, 0.3125, 1, 1)
    network = build_pipe_network(mesh, design.config, center=(2.5, 2.5))
    mesh.T[:] = 700.0
    loop = network.fluid_loop(0.25, Fluid(), mesh=mesh, external_power=20_000.0)
    result = loop.solve(mesh)
    assert result.power == pytest.approx(20_000.0, rel=1e-9)
    assert loop.hydraulic_state is not None and loop.hydraulic_state.converged
    loop.graph.check(tolerance=1e-6)


def test_an_orifice_hole_gives_its_loss_back():
    from src.solver.hydraulics import orifice_bore, orifice_loss

    k = np.array([0.0, 0.5, 5.0, 25.0, 200.0])
    holes = orifice_bore(k, 0.046)
    assert holes[0] == pytest.approx(0.046)
    assert np.all(np.diff(holes) < 0.0)                  # more loss, smaller hole
    assert np.allclose(orifice_loss((holes[1:] / 0.046) ** 2), k[1:], rtol=1e-6)
