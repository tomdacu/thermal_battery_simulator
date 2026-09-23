"""The tube in its cell (Peaceman's well model) and the whole gas circuit.

A tube is a line inside a cell of the bed.  Without a correction the exchange depends
on the cell: the bed between the wall and the cell centre is counted as whatever the
mesh makes of it.  With the well model the exchange of one tube in a square of sand held
at a fixed temperature is the conduction shape factor of a cylinder centred in a square
(Incropera et al., *Fundamentals of Heat and Mass Transfer*, Table 4.1:
``S = 2 pi L / ln(1.08 w / D)``) in series with the gas film, on a coarse and on a fine
mesh alike.  The circuit tests pin the network graph: mass conserved at every node, the
headers exchanging, the bed receiving exactly the external power.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.core.adaptive_mesh import AdaptiveMesh
from src.core.mesh import BoundaryType
from src.core.pipes import rasterize_pipe
from src.solver.fluid import Fluid, FluidLoop, pipe_h

SIDE = 2.4          # [m] the square of sand
BORE = 0.046        # [m]
WALL = 0.002        # [m]
FLOW = 0.01         # [kg/s]
T_GAS = 700.0       # [K] gas in
T_SAND = 300.0      # [K] the sides of the square


def _tube_in_a_square(cell: float, well: bool) -> float:
    """Steady power [W] of 1 m of tube in the middle of the square, coupled to the gas."""
    n = int(round(SIDE / cell))
    mesh = AdaptiveMesh.from_box(n, 4, SIDE / n, 0.25, 0, 0)
    mesh.k[:] = 0.5
    for face in ("x_min", "x_max", "y_min", "y_max"):
        mesh.set_fixed_temperature_bc(face, T_SAND)
    mesh.set_adiabatic("z_min")
    mesh.set_adiabatic("z_max")
    centre = 0.5 * SIDE + 1e-6
    run = rasterize_pipe(mesh, [(centre, centre, 1e-6), (centre, centre, 1.0 - 1e-6)],
                         BORE)
    mesh.boundary_type[run.cells] = int(BoundaryType.CONVECTION)
    loop = FluidLoop(runs=[run], mass_flow=FLOW, fluid=Fluid(), t_in=T_GAS,
                     well_model=well, wall_thickness=WALL)
    mesh.T[:] = T_SAND
    for _ in range(25):
        result = loop.solve(mesh)
        result.apply(mesh)
        mesh.solve_steady()
    return float(result.power)


def _analytic() -> float:
    fluid = Fluid()
    outer = BORE + 2.0 * WALL
    film = 1.0 / (pipe_h(FLOW, BORE, fluid) * np.pi * BORE)
    wall = np.log(outer / BORE) / (2.0 * np.pi * 21.0)
    bed = np.log(1.08 * SIDE / outer) / (2.0 * np.pi * 0.5)
    ntu = 1.0 / (film + wall + bed) / (FLOW * fluid.cp)
    return FLOW * fluid.cp * (T_GAS - T_SAND) * (1.0 - np.exp(-ntu))


def test_the_well_model_makes_the_exchange_independent_of_the_cell():
    coarse, fine = _tube_in_a_square(0.3, True), _tube_in_a_square(0.15, True)
    reference = _analytic()
    assert coarse == pytest.approx(reference, rel=0.05)
    assert fine == pytest.approx(reference, rel=0.03)
    assert abs(coarse - fine) < 0.03 * reference


def test_without_the_well_model_the_exchange_follows_the_mesh():
    coarse, fine = _tube_in_a_square(0.3, False), _tube_in_a_square(0.15, False)
    assert abs(coarse - fine) > 0.15 * _analytic()


def test_the_well_model_leaves_a_cell_smaller_than_the_tube_uncorrected():
    """Below 2.53 d the equivalent radius falls inside the tube: no correction."""
    assert _tube_in_a_square(0.075, True) == pytest.approx(
        _tube_in_a_square(0.075, False), rel=1e-12)


def test_the_whole_circuit_conserves_mass_and_gives_the_bed_the_external_power():
    from src.core.geometry import BatteryGeometry, CylinderGeometry
    from src.core.pipe_network import (COLLECTION_REVERSE, LAYOUT_RINGS, PipeNetworkConfig,
                                       build_pipe_network)

    cylinder = CylinderGeometry(center_x=2.0, center_y=2.0, base_z=0.3, height=3.0,
                                r_storage=1.2, insulation_thickness=0.2,
                                insulation_slab_bottom=0.2, insulation_slab_top=0.2,
                                enable_cone_roof=False)
    mesh = AdaptiveMesh.from_box(16, 16, 0.25, 0.25, 1, 1)
    battery = BatteryGeometry(cylinder=cylinder)
    battery.apply_to_mesh(mesh)
    config = PipeNetworkConfig(
        radius=cylinder.r_storage, height=cylinder.z_cone_base - cylinder.base_z,
        base_z=cylinder.base_z,
        band_bottom=cylinder.z_storage_start - cylinder.base_z,
        band_top=cylinder.z_storage_end - cylinder.base_z, diameter=0.05,
        layout=LAYOUT_RINGS, collection=COLLECTION_REVERSE, n_rings=3,
        duct_diameter=0.1)
    network = build_pipe_network(mesh, config, center=(2.0, 2.0))
    network.paint(mesh)
    mesh.T[:] = 600.0
    graph = network.gas_graph(mesh)
    graph.check()                                            # mass at every node
    names = [segment.name for segment in graph.segments]
    assert any(name.startswith("bottom_ring") for name in names)
    assert any(name.startswith("top_ring") for name in names)
    loop = network.fluid_loop(0.3, mesh=mesh, external_power=20_000.0)
    result = loop.solve(mesh)
    # the balance is imposed on the exchange: the bed takes the resistors' power
    assert result.power == pytest.approx(20_000.0, rel=1e-9)
    header_power = sum(run.power for run in result.runs if "ring" in run.name)
    assert header_power > 0.0                     # the bare headers charge the bed too
    # the loop comes back colder than it left, by the power over the flow
    assert result.t_in > result.t_out
