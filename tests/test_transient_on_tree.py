"""The transient march and the cycle on a tree: one driver, two assemblies.

Step 5 of the migration (``docs/16_ADAPTIVE_MESH_MIGRATION.md``): the transient runs on
``Mesh3D`` and on an :class:`~src.core.adaptive_mesh.AdaptiveMesh` through the same
driver, and the cycle's ledger closes on either.  What this module pins, in the order the
claims matter:

* **the slab** - the 1-D Dirichlet slab of
  ``tests/test_solver.py::test_transient_matches_the_analytic_slab_series`` run on a tree:
  the interior column follows the Fourier series in the same 5 % band, and every leaf
  carries the temperature the matching ``Mesh3D`` carries, to the tolerance of the linear
  solve.  The two operators are *equal* (the same face conductance, the same film, the
  same elimination), so what the assertion measures is the march;
* **the lumped case** - an adiabatic box with a uniform source, where ``rho*cp`` alone
  sets the rate (the exactness ``dT/dt = Q/(rho*cp)`` of
  ``tests/test_solver.py::test_transient_rate_is_mesh_independent``): it holds on a tree,
  on a *refined* tree with hanging nodes, and on ``Mesh3D`` - the mass matrix is the same
  number on both assemblies, and it does not carry the cell volume;
* **the environment film and the excluded air** - the painted battery model on both
  meshes: the same field on every active leaf and the same environment flux.  The
  excluded leaves are the one place where the two roads differ, and they differ by rule:
  the tree's elimination pins them at the ambient (its steady rule, and the reason its
  balance stays readable), while the structured transient leaves them decoupled - no
  conductance, no film - so they evolve on their own and feed nothing back;
* **the refined tree** - hanging nodes: the march settles on the tree's *own* steady
  field (``solve_steady``), which is what says the operator, the films and the elimination
  the transient builds are the ones the verified steady assembly builds;
* **the cycle** - charge, standby and discharge on a tree bed with a gas loop: the ledger
  closes exactly as on ``Mesh3D``, blower and energy delivered by the loop included, and
  the phase criteria (the per-step ``should_stop`` hook) end the phases on the tree the
  way they end them on the grid;
* **the profile branches** - a ``flow_rate`` extraction through tube leaves (the film and
  the sink it drives), and the guard that refuses a power profile with no source leaf:
  both read the tree's per-cell arrays, which are one flat vector instead of a 3-D field.

The tree of the slab cases is the 1 m cube of ``tests/test_adaptive_mesh.py`` (16 finest
cells a side, so every level is a power-of-two subdivision of the box); the battery case
is the 8 m cube of ``tests/conftest.py``; the cycle bed is a 2 m silo built here.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.balance import compute_balance
from src.analysis.cycle import CycleSettings, run_cycle
from src.analysis.fluxes import environment_flux
from src.core.adaptive_mesh import AdaptiveMesh, RefinementBand
from src.core.mesh import MaterialID, Mesh3D
from src.core.pipes import PipeRun
from src.core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from src.solver.fluid import FluidLoop
from src.solver.linear import LinearConfig
from src.solver.steady import SolverConfig
from src.solver.transient import TransientConfig, TransientSolver

#: the linear layer of the marches compared leaf by leaf: tight enough that the two
#: fields are the same to the last decimals the solver can resolve
MARCH = SolverConfig(method="bicgstab", tolerance=1e-10)
#: the cycle is driven with the settings ``tests/test_cycle.py`` uses
CYCLE_MARCH = SolverConfig(method="cg", tolerance=1e-7)

BOX = 1.0
FINEST = 16                       # finest cells a side of the slab box
H = BOX / FINEST                  # edge of one finest cell [m]
T_HOT, T_COLD = 373.15, 273.15
SLAB_T_END, SLAB_DT = 200_000.0, 20_000.0


# ------------------------------------------------------------------- helpers
def slab_tree() -> AdaptiveMesh:
    """The 1-D slab of ``tests/test_solver.py`` as a cube with adiabatic sides.

    A tree spans a cube, so the 1-D problem is posed the way it is posed for a cylinder
    of the structured solver: the sides carry no flux and the field is uniform across
    them, which leaves the column as the solution of the series below.
    """
    tree = AdaptiveMesh.uniform(FINEST, H, 0)
    tree.k[:] = 1.0
    tree.rho[:] = 1000.0
    tree.cp[:] = 1000.0
    for face in ("x_min", "x_max", "y_min", "y_max"):
        tree.set_adiabatic(face)
    tree.set_fixed_temperature_bc("z_min", T_HOT)
    tree.set_fixed_temperature_bc("z_max", T_HOT)
    return tree


def column(tree: AdaptiveMesh) -> tuple[np.ndarray, np.ndarray]:
    """``(z, T)`` of the leaves on the ``x = y = 0`` column, bottom to top.

    The slab is one-dimensional, so one column carries the whole solution and the series
    can be sampled at the leaf centres the march produced.
    """
    centres = tree.centres()
    first = 0.5 * tree.physical_size
    mask = np.isclose(centres[:, 0], first) & np.isclose(centres[:, 1], first)
    z = centres[mask, 2]
    order = np.argsort(z)
    return z[order], tree.T[mask][order]


def fourier(z: np.ndarray, t_end: float) -> np.ndarray:
    """The analytic slab of the structured test, sampled at ``z``."""
    alpha = 1.0 / (1000.0 * 1000.0)
    out = np.full_like(z, T_HOT)
    for n in range(1, 200):
        coefficient = 2 * (T_COLD - T_HOT) * (1 - (-1) ** n) / (n * np.pi)
        out += (coefficient * np.sin(n * np.pi * z / BOX)
                * np.exp(-(n * np.pi / BOX) ** 2 * alpha * t_end))
    return out


def march(mesh, config: TransientConfig, solver_config: SolverConfig = MARCH):
    """Run the transient on either mesh; returns the results."""
    return TransientSolver(mesh, config, solver_config).run()


def lumped_tree(level: int = 3) -> AdaptiveMesh:
    """A coarse uniform tree of the slab box: 2 leaves a side at ``level`` 3."""
    tree = AdaptiveMesh.uniform(FINEST, H, level)
    tree.k[:] = 1.0
    tree.rho[:] = 1000.0
    tree.cp[:] = 1000.0
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        tree.set_adiabatic(face)
    tree.source_mask[:] = True
    return tree


def lumped_refined_tree() -> AdaptiveMesh:
    """The same box with the middle band refined once: hanging nodes, same lumped rate."""
    tree = AdaptiveMesh.from_bands(
        FINEST, H, [RefinementBand((0.0, 0.0, 0.25), (BOX, BOX, 0.75), 2 * H)],
        base_level=2)
    tree.k[:] = 1.0
    tree.rho[:] = 1000.0
    tree.cp[:] = 1000.0
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        tree.set_adiabatic(face)
    tree.source_mask[:] = True
    return tree


def lumped_grid() -> Mesh3D:
    """The ``Mesh3D`` of the coarse tree: the same cells, the same adiabatic box."""
    mesh = Mesh3D(BOX, BOX, BOX, spacing=0.5)
    mesh.k[:] = 1.0
    mesh.rho[:] = 1000.0
    mesh.cp[:] = 1000.0
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        mesh.set_adiabatic(face)
    mesh.source_mask[:] = True
    return mesh


# ------------------------------------------------------------------- the slab
def test_the_slab_matches_the_fourier_series_and_the_structured_mesh(structured_twin):
    """1-D conduction on a tree: the series in the interior, ``Mesh3D`` leaf by leaf."""
    tree = slab_tree()
    config = TransientConfig(t_final=SLAB_T_END, dt=SLAB_DT, save_interval=SLAB_T_END,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=T_COLD),
                             power_profile=PowerProfile(mode="off"))
    march(tree, config)

    z, temperature = column(tree)
    analytic = fourier(z, SLAB_T_END)
    # the pinned boundary nodes carry a first-order surface artifact; the interior must
    # follow the series within the same few percent the structured test allows
    error = float(np.abs(temperature[2:-2] - analytic[2:-2]).max())
    assert error < 0.05 * (T_HOT - T_COLD), f"max interior error {error:.2f} K"

    structured, leaves = structured_twin(tree)
    structured.T = np.full(structured.T.shape, T_COLD)
    march(structured, config)
    field = structured.T.reshape(-1, order="F")[leaves]
    difference = float(np.abs(tree.T - field).max())
    assert difference < 1e-4, f"the two assemblies differ by {difference:.2e} K"


# ------------------------------------------------------------- the lumped case
def test_the_lumped_rate_is_exact_on_a_tree_on_a_refined_tree_and_on_the_grid():
    """An adiabatic box with a uniform source: ``rho*cp`` alone sets the rate.

    ``dT/dt = Q/(rho*cp)`` is exact on any mesh - a constant field is in the kernel of
    the conduction operator, whatever the face list looks like (a hanging node included) -
    which makes it the sharpest test of the mass matrix: a ``rho*cp*V`` diagonal, or one
    that dropped the leaf volumes, would move the rate, and the three meshes would stop
    agreeing with the analytic number.
    """
    density, t_final, dt = 1000.0, 10_000.0, 1_000.0
    refined = lumped_refined_tree()
    assert len(refined.level_histogram()) >= 2, "the refined tree must carry hanging nodes"

    expected = 293.15 + density / (1000.0 * 1000.0) * t_final
    for name, mesh in (("tree", lumped_tree()), ("refined tree", refined),
                       ("structured grid", lumped_grid())):
        power = density * float(np.sum(mesh.V))
        config = TransientConfig(t_final=t_final, dt=dt, save_interval=t_final,
                                 power_profile=PowerProfile(mode="constant",
                                                            constant_power=power))
        march(mesh, config)
        assert float(np.mean(mesh.T)) == pytest.approx(expected, rel=1e-6), name


# ------------------------------------------------------------ the environment
def test_the_environment_film_and_the_excluded_air_on_a_tree(paint_pair):
    """The painted battery on both meshes: same active field, same flux, pinned air.

    The air box is dropped, not meshed: its leaves carry no conduction at all and behave
    one way on one mesh and another way on the other - the tree's elimination pins them
    at ``t_ambient`` (the rule its steady assembly states), while the structured transient
    leaves them out of every elimination, so they evolve decoupled and feed nothing back.
    Everything that *does* feed back - the active field and the film the outer surface
    exchanges through - is the same number on both roads.
    """
    tree, structured = paint_pair()
    leaves = np.array([structured.ijk_to_linear(*structured.find_cell(*centre))
                       for centre in tree.centres()])
    config = TransientConfig(t_final=2 * 1800.0, dt=1800.0, save_interval=2 * 1800.0,
                             t_ambient=tree.t_ambient,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=400.0),
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=50_000.0))
    march(tree, config)
    march(structured, config)

    field = structured.T.reshape(-1, order="F")[leaves]
    active = ~tree.excluded
    assert 0 < int(active.sum()) < tree.n_cells
    difference = float(np.abs(tree.T[active] - field[active]).max())
    assert difference < 1e-3, f"the active leaves differ by {difference:.2e} K"
    assert np.allclose(tree.T[~active], tree.t_ambient), "the excluded leaves stay ambient"

    balance = compute_balance(tree, tree.t_ambient)
    reference = compute_balance(structured, tree.t_ambient)
    assert balance.p_input == pytest.approx(reference.p_input, rel=1e-9)
    assert environment_flux(tree) == pytest.approx(environment_flux(structured), rel=1e-6)
    assert balance.q_domain == pytest.approx(reference.q_domain, rel=1e-6)
    assert balance.t_mean_storage == pytest.approx(reference.t_mean_storage, rel=1e-6)


# -------------------------------------------------------------- the refined tree
def refined_tree() -> AdaptiveMesh:
    """A slab box refined once around the middle: two levels, hanging nodes."""
    tree = AdaptiveMesh.from_bands(
        FINEST, H, [RefinementBand((0.0, 0.0, 0.25), (BOX, BOX, 0.75), 2 * H)],
        base_level=2)
    tree.k[:] = 2.0
    tree.rho[:] = 1000.0
    tree.cp[:] = 1000.0
    tree.material_id[:] = 1
    tree.source_mask[:] = True
    tree.Q_source[:] = 3000.0
    for face in ("x_min", "x_max", "y_min", "y_max"):
        tree.set_adiabatic(face)
    tree.set_fixed_temperature_bc("z_min", 350.0)
    tree.set_convection_bc("z_max", 25.0, 293.15)
    return tree


def test_a_refined_tree_with_hanging_nodes_settles_on_its_own_steady_field():
    """A source, a film and a fixed face on a refined tree: the march finds its steady field.

    The transient operator is ``M/dt + L`` and the steady operator is ``L``; both carry
    the same conduction, the same film and the same elimination, so the long-time limit of
    a march is the steady solution of the very mesh it ran on.  Comparing the two is a
    test of the whole operator - the hanging-node faces, the film on the refined leaves,
    the pinned rows - against an assembly that five equivalence tests already pin, and it
    needs no structured twin: a refined tree has none.
    """
    tree = refined_tree()
    assert len(tree.level_histogram()) >= 2, "the test needs hanging nodes"
    steady = tree.solve_steady(LinearConfig(method="direct"))

    steps, dt = 80, 10_000.0
    config = TransientConfig(t_final=steps * dt, dt=dt, save_interval=steps * dt,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=300.0),
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=float(
                                                            np.sum(tree.Q_source * tree.V))))
    march(tree, config, SolverConfig(method="bicgstab"))
    residual = float(np.abs(tree.T - steady.T).max())
    assert residual < 1e-3, f"the march settled {residual:.2e} K away from its own steady field"


# -------------------------------------------------------------------- the cycle
def silo_tree(box: float = 2.0, n_finest: int = 4) -> tuple[AdaptiveMesh, np.ndarray]:
    """A small sand silo on a tree: adiabatic except the top face, four tube columns.

    The bed of ``tests/test_cycle.py`` in miniature, with the tube bank the tree can carry
    today: whole leaves flagged as tubes (step 6 of the migration paints the rasterised
    network), one vertical run per column, the gas loop marching them bottom to top.
    """
    spacing = box / n_finest
    tree = AdaptiveMesh.uniform(n_finest, spacing, 0)
    tree.k[:] = 0.5
    tree.rho[:] = 1500.0
    tree.cp[:] = 800.0
    tree.material_id[:] = int(MaterialID.SAND)
    tree.T[:] = 293.15
    tree.source_mask[:] = True
    for face in ("x_min", "x_max", "y_min", "y_max", "z_min"):
        tree.set_adiabatic(face)
    tree.set_convection_bc("z_max", 1.0, 293.15)
    centres = tree.centres()
    axis = np.unique(np.round(centres[:, 0], 9))
    tube = np.zeros(tree.n_cells, dtype=bool)
    for x in (axis[1], axis[-2]):
        for y in (axis[1], axis[-2]):
            tube |= np.isclose(centres[:, 0], x) & np.isclose(centres[:, 1], y)
    tree.material_id[tube] = int(MaterialID.TUBES)
    tree.set_internal_convection(tube, 0.0, 293.15)
    return tree, tube


def tube_runs(tree: AdaptiveMesh, tube: np.ndarray, diameter: float = 0.1,
              index: np.ndarray | None = None) -> list[PipeRun]:
    """One vertical :class:`PipeRun` per tube column, in the cell numbering of ``index``."""
    centres = tree.centres()
    columns: dict[tuple[float, float], list[int]] = {}
    for position in np.flatnonzero(tube):
        key = (round(float(centres[position, 0]), 9), round(float(centres[position, 1]), 9))
        columns.setdefault(key, []).append(int(position))
    runs = []
    for (x, y), leaves in sorted(columns.items()):
        leaves = sorted(leaves, key=lambda position: centres[position, 2])
        cells = np.array(leaves, dtype=np.int64)
        if index is not None:
            cells = index[cells]
        runs.append(PipeRun(name=f"tube_{x}_{y}",
                            points=np.array([[x, y, 0.0], [x, y, tree.box_size]]),
                            diameter=diameter, cells=cells,
                            length=np.full(cells.size, float(tree.sizes[leaves[0]]))))
    return runs


def silo_settings(**overrides) -> CycleSettings:
    """The cycle of the small silo: 20 kW in, half an hour of standby, 10 kW out.

    The charge target is ~22 K above the start on purpose: a step of the miniature
    silo adds ~3.7 K to the store (the pipe columns included - they are bed cells), so
    the criterion fires after six steps, *inside* the second chunk - which is what
    shows the per-step hook of the march doing the stopping on a tree - while the
    discharge runs to its own floor on the gas.
    """
    settings = dict(charge_power=20_000.0, t_target=315.0, charge_limit=6 * 3600.0,
                    standby_time=1800.0, discharge_power=10_000.0, t_delivery_min=250.0,
                    dt=1800.0, chunk=7200.0, max_chunks=2)
    settings.update(overrides)
    return CycleSettings(**settings)


def test_the_cycle_closes_its_ledger_on_a_tree(structured_twin):
    """Charge, stand, discharge on a tree: the identity closes, and as on ``Mesh3D``.

    Every term is measured from the field the step left behind (never assumed from the
    profile), the blower is the pressure drop of the actual runs, and the energy delivered
    is what the exchanger took out of the gas - so the same number has to come out of the
    two assemblies, phase by phase, and the tree's identity has to close on its own
    solver's imbalance.
    """
    tree, tube = silo_tree()
    loop = FluidLoop(runs=tube_runs(tree, tube), mass_flow=0.3, fittings_k=8.0)
    settings = silo_settings()
    structured, leaves = structured_twin(tree)
    twin_loop = FluidLoop(runs=tube_runs(tree, tube, index=leaves), mass_flow=0.3,
                          fittings_k=8.0)

    report = run_cycle(tree, loop, settings, CYCLE_MARCH)
    reference = run_cycle(structured, twin_loop, settings, CYCLE_MARCH)

    # the phases really ran, and the loop carried the energy: blower and delivered heat
    # are terms of the ledger, not annotations
    assert report.charge.seconds > 0.0
    assert report.standby.seconds == pytest.approx(settings.standby_time)
    assert report.discharge.seconds > 0.0
    assert report.fan_energy > 0.0
    assert report.energy_delivered > 0.0
    assert report.energy_loss > 0.0
    # both phases end on their own criterion, the charge inside a chunk: that is the
    # per-step ``should_stop`` hook of the march, running on a tree
    assert report.charge.seconds < settings.charge_limit
    assert report.charge.seconds % settings.chunk != 0.0
    assert report.charge.notes and "the target" in report.charge.notes[0]
    assert report.discharge.seconds < settings.max_chunks * settings.chunk
    assert report.discharge.t_delivery_end <= settings.t_delivery_min
    assert report.discharge.notes and "floor" in report.discharge.notes[0]

    carried = (report.dE_stored + report.energy_delivered + report.energy_loss
               + report.fan_energy)
    assert report.energy_electric == pytest.approx(carried, rel=0.01)
    assert report.balance_residual() < 0.01
    assert report.charge.energy_in == pytest.approx(
        settings.charge_power * report.charge.seconds, rel=1e-6)
    assert report.discharge.energy_out == pytest.approx(
        settings.discharge_power * report.discharge.seconds, rel=1e-6)
    assert report.fan_energy == pytest.approx(
        loop.solve(tree).fan_power * sum(phase.seconds for phase in report.phases), rel=1e-6)

    # the same cycle on the structured twin: same phases, same ledger, to the round-off of
    # the two assemblies
    for on_tree, on_grid in zip(report.phases, reference.phases, strict=True):
        assert on_tree.seconds == pytest.approx(on_grid.seconds, abs=1e-9), on_tree.name
        assert on_tree.stored_start == pytest.approx(on_grid.stored_start, rel=1e-6), on_tree.name
        assert on_tree.stored_end == pytest.approx(on_grid.stored_end, rel=1e-6), on_tree.name
        assert on_tree.energy_in == pytest.approx(on_grid.energy_in, rel=1e-6, abs=1e-6), on_tree.name
        assert on_tree.energy_out == pytest.approx(on_grid.energy_out, rel=1e-6, abs=1e-6), on_tree.name
        assert on_tree.energy_loss == pytest.approx(on_grid.energy_loss, rel=1e-6, abs=1e-6), on_tree.name
        assert on_tree.fan_energy == pytest.approx(on_grid.fan_energy, rel=1e-6, abs=1e-6), on_tree.name
        assert on_tree.t_end == pytest.approx(on_grid.t_end, rel=1e-9), on_tree.name
    assert reference.balance_residual() < 0.01


def test_the_flow_rate_extraction_drains_a_tree_through_its_tube_leaves(structured_twin):
    """The tube film and the sink it drives, on a tree: same field, same extracted power.

    ``flow_rate`` extraction is the profile branch that writes ``bc_h``/``bc_T_inf`` per
    cell and lets the physics cap what leaves: the film of the tube leaves, the
    ``Q_sink`` of a ``power`` request and the report that adds them up all index the
    per-cell arrays, and on a tree those arrays are one flat vector.  The tube leaves sit
    inside the box on purpose: a tube leaf on a box face carries the face condition
    instead (``is_interior_tube``), on either mesh.
    """
    tree = slab_tree()
    for face in ("z_min", "z_max"):
        tree.set_adiabatic(face)
    tube = np.zeros(tree.n_cells, dtype=bool)
    centres = tree.centres()
    axis = np.unique(np.round(centres[:, 0], 9))
    for x in (axis[1], axis[-2]):
        for y in (axis[1], axis[-2]):
            tube |= np.isclose(centres[:, 0], x) & np.isclose(centres[:, 1], y)
    tube &= ~tree.on_box_face
    assert tube.any() and not tree.on_box_face[tube].any(), (
        "the test needs tube leaves off the box faces")
    tree.material_id[tube] = int(MaterialID.TUBES)
    tree.set_internal_convection(tube, 500.0, 300.0)

    config = TransientConfig(t_final=1800.0, dt=600.0, save_interval=1800.0,
                             initial_condition=InitialCondition(mode="uniform",
                                                                t_uniform=500.0),
                             power_profile=PowerProfile(mode="off"),
                             extraction_profile=ExtractionProfile(mode="flow_rate",
                                                                  mass_flow=0.5,
                                                                  t_inlet=300.0,
                                                                  h_fluid=500.0))
    results = march(tree, config)
    structured, leaves = structured_twin(tree)
    structured.T = np.full(structured.T.shape, 500.0)
    reference = march(structured, config)

    assert results.P_extracted[-1] > 0.0
    assert float(tree.T.mean()) < 500.0
    assert results.P_extracted[-1] == pytest.approx(reference.P_extracted[-1], rel=1e-6)
    field = structured.T.reshape(-1, order="F")[leaves]
    difference = float(np.abs(tree.T - field).max())
    assert difference < 1e-3, f"the tube film differs by {difference:.2e} K"


def test_a_tree_bed_refuses_a_power_profile_with_no_source_leaf():
    """The guard of ``tests/test_solver.py`` reads the tree's own mask.

    A profile that asks for heat while no leaf is flagged as a source is a model mistake,
    and the march has to say so instead of heating nothing - on a tree the mask is one
    flat array, and this is the one place that assumption could quietly break.
    """
    tree = lumped_tree(level=0)
    tree.source_mask[:] = False
    config = TransientConfig(t_final=60.0, dt=60.0,
                             power_profile=PowerProfile(mode="constant",
                                                        constant_power=1000.0))
    with pytest.raises(ValueError, match="source"):
        TransientSolver(tree, config).run()


def test_a_mesh_neither_assembly_knows_is_refused():
    """A mesh that is neither a ``Mesh3D`` nor a tree is rejected before any step runs."""
    class NoAssembly:
        """Neither the structured tables nor the tree's transient operators."""

    with pytest.raises(TypeError, match="adaptive mesh"):
        TransientSolver(NoAssembly(), TransientConfig())
