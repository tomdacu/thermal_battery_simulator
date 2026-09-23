"""Run configuration and the worker/state machine that executes a simulation.

The GUI never runs physics: it assembles a :class:`RunConfig` from its widgets
and hands it to :class:`SimulationController`, which owns the threads, the
progress reporting and the cancel flag.  All the numerics happen in ``src/``.

Every analysis runs the plant: when the window has painted a pipe network, the gas
loop of that network is the heat path of the run - the steady state and the losses
analysis couple it (``SteadyStateSolver(fluid_loop=...)``), the transient marches it
step by step - and each run sets the state it needs on the mesh itself, so nothing
a previous run left behind (a transient's last sources, its films) leaks into it.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.analysis.balance import compute_balance
from src.analysis.convergence import AdaptivePlan, ConvergenceTarget, find_mesh
from src.analysis.losses import LossesConfig, solve_losses
from src.analysis.mesh_plan import describe as describe_plan
from src.analysis.mesh_plan import plan_regions
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import BatteryGeometry
from src.core.materials import MaterialManager
from src.core.mesh import Mesh3D
from src.core.pipe_network import PipeNetwork, PipeNetworkConfig, build_pipe_network
from src.core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from src.solver.steady import SolverConfig
from src.solver.transient import TransientConfig, TransientSolver

ANALYSIS_TYPES = ("standby", "transient")


@dataclass
class RunConfig:
    """Everything a run needs, already validated and in SI/Kelvin units."""

    analysis: str = "steady"
    battery: BatteryGeometry = field(default_factory=BatteryGeometry)
    #: the tree the automatic mesh search refines (``None``: no search)
    mesh_spec: AdaptivePlan | None = None
    #: tolerances of the search: delta_temperature [K], delta_power [-], levels, budget
    convergence: dict = field(default_factory=dict)
    method: str = "cg"
    preconditioner: str = "amg_rs"
    tolerance: float = 1e-8
    max_iterations: int = 2000
    n_threads: int = -1
    radiation: bool = False
    losses: dict = field(default_factory=dict)
    transient: dict = field(default_factory=dict)
    initial_condition: InitialCondition = field(default_factory=InitialCondition)
    power_profile: PowerProfile = field(default_factory=PowerProfile)
    extraction_profile: ExtractionProfile = field(default_factory=ExtractionProfile)
    start_from_steady: bool = False
    #: the buried pipe network the window built and painted on the run's mesh (None =
    #: no network: the geometry's lumped bed source alone drives the run)
    pipe_network: PipeNetwork | None = None
    #: the recipe of that network, for the meshes the automatic search builds
    pipe_config: PipeNetworkConfig | None = None
    #: total mass flow of the gas circuit [kg/s]
    pipe_flow: float = 0.0
    #: the gas the circuit is filled with (``src.solver.fluid.Fluid``); None = air
    pipe_fluid: object = None
    #: absolute pressure of the loop [Pa]: it sets the density the gas marches at
    pipe_pressure: float = 101325.0
    #: blower efficiency [-]: the electric power of the fan is the shaft power over it
    pipe_fan_efficiency: float = 0.7

    def solver_config(self) -> SolverConfig:
        return SolverConfig(method=self.method, preconditioner=self.preconditioner,
                            tolerance=self.tolerance, max_iterations=self.max_iterations,
                            n_threads=self.n_threads, radiation=self.radiation)


class SimulationJob(QThread):
    """Runs one callable in a worker thread with progress and cancellation."""

    progressed = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, work: Callable, parent=None) -> None:
        super().__init__(parent)
        self._work = work
        self._stop = False

    def cancel(self) -> None:
        self._stop = True
        self.requestInterruption()

    def should_stop(self) -> bool:
        return self._stop or self.isInterruptionRequested()

    def run(self) -> None:  # pragma: no cover - thread body
        try:
            result = self._work(self.progressed.emit, self.should_stop)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a message
            self.failed.emit(f"{type(exc).__name__}: {exc}")
            return
        self.completed.emit(result)


class SimulationController(QObject):
    """Owns the running job, the button state and the progress fan-out."""

    progressed = pyqtSignal(int, str)
    log = pyqtSignal(str)
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str)
    running_changed = pyqtSignal(bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._job: SimulationJob | None = None
        #: the gas-loop result of the last run, for the results panel (None: no loop)
        self.last_loop = None

    @property
    def running(self) -> bool:
        return self._job is not None and self._job.isRunning()

    # ------------------------------------------------------------------ api
    def start(self, config: RunConfig, mesh: Mesh3D | AdaptiveMesh | None) -> None:
        if self.running:
            self.failed.emit("a simulation is already running")
            return
        work = self._make_work(config, mesh)
        self._job = SimulationJob(work, parent=self)
        self._job.progressed.connect(self.progressed.emit)
        self._job.completed.connect(lambda result: self._finish(config.analysis, result))
        self._job.failed.connect(self._fail)
        self._job.finished.connect(self._cleanup)
        self.running_changed.emit(True)
        self._job.start()

    def cancel(self) -> None:
        if self.running:
            self.log.emit("[cancel] stop requested")
            self._job.cancel()

    def wait(self, timeout_ms: int = 5000) -> bool:
        """Block until the worker stops (closing the window must not kill a QThread)."""
        if self._job is None:
            return True
        return bool(self._job.wait(timeout_ms))

    # --------------------------------------------------------------- engine
    def _make_work(self, config: RunConfig, mesh: Mesh3D | AdaptiveMesh) -> Callable:
        solver_config = config.solver_config()

        def hold(target, network, progress=None, should_stop=None, tolerance=None):
            """The standby state of ``target``: the power that holds the set temperature.

            A storage has one steady state worth asking for - the one where the power in
            equals the losses - so it is asked as a temperature and solved for the power
            (a secant iteration on coupled steady solves, ``solve_losses``).
            """
            settings = dict(config.losses)
            if tolerance is not None:
                settings["tolerance"] = tolerance
            loop = self._fluid_loop(config, target, network)
            result = solve_losses(target, LossesConfig(**settings), solver_config,
                                  progress, should_stop, fluid_loop=loop)
            self.last_loop = loop.solve(target) if loop is not None else None
            return result

        def standby(progress, should_stop):
            t_hold = config.losses.get("t_target", 773.15) - 273.15
            self.log.emit(f"[standby] holding the storage at {t_hold:.0f} degC")
            result = hold(mesh, config.pipe_network, progress, should_stop)
            self.log.emit(f"[standby] converged={result.converged} in {result.iterations} "
                          f"iterations: {result.power / 1000:.2f} kW holds "
                          f"{result.t_mean_storage - 273.15:.1f} degC")
            self._log_loop()
            return result

        def automesh(progress, should_stop):
            plan = config.mesh_spec
            if plan is None:
                raise ValueError("the automatic mesh search needs the tree plan of the "
                                 "Mesh tab")
            settings = {k: v for k, v in (config.convergence or {}).items()
                        if k in ConvergenceTarget.__dataclass_fields__}
            target = ConvergenceTarget(**settings)
            self.log.emit(f"[automesh] tolerances: dT {target.delta_temperature} K, "
                          f"dP {100 * target.delta_power:.1f}%, budget "
                          f"{target.max_cells:,} cells")
            for line in describe_plan(plan_regions(config.battery,
                                                   MaterialManager())).splitlines():
                self.log.emit(f"[plan] {line}")
            networks: dict[int, PipeNetwork] = {}

            def build(level: AdaptivePlan):
                """One search level: the tree, the battery and the network painted on it."""
                tree = AdaptiveMesh.from_bands(level.n_finest, level.physical_size,
                                               level.bands, level.base_level)
                config.battery.apply_to_mesh(tree)
                if config.pipe_config is not None:
                    cyl = config.battery.cylinder
                    network = build_pipe_network(tree, config.pipe_config,
                                                 center=(cyl.center_x, cyl.center_y))
                    network.paint(tree)
                    networks[id(tree)] = network
                return tree

            def observables(tree):
                # the standby state of every level: the storage held at the set
                # temperature, and the power that holds it - the losses - is what the
                # mesh has to get right (a tight hold, so its own tolerance does not
                # pass for a discretisation error)
                result = hold(tree, networks.get(id(tree)), tolerance=0.02)
                balance = compute_balance(tree, config.battery.t_ambient,
                                          radiation=config.radiation)
                return {"t_mean_storage": balance.t_mean_storage,
                        "t_max": balance.t_max, "power": result.power}

            report = find_mesh(build, observables, plan, target, progress, should_stop)
            for line in report.summary().splitlines():
                self.log.emit(f"[automesh] {line}")
            return report

        def transient(progress, should_stop):
            loop = self._fluid_loop(config, mesh, config.pipe_network)
            if loop is None:
                config.battery.apply_source(mesh)
            if config.start_from_steady:
                self.log.emit("[transient] standby pre-run for the initial field")
                hold(mesh, config.pipe_network)
            settings = dict(config.transient)
            settings.setdefault("save_full_field", False)
            t_cfg = TransientConfig(
                initial_condition=config.initial_condition,
                power_profile=config.power_profile,
                extraction_profile=config.extraction_profile,
                t_ambient=config.battery.t_ambient,
                fluid_loop=loop,
                **settings,
            )
            solver = TransientSolver(mesh, t_cfg, solver_config)
            results = solver.run(progress, should_stop)
            self.last_loop = solver.fluid_result
            self._log_loop()
            for note in solver.notes:
                self.log.emit(f"[transient] {note}")
            if loop is not None:
                self.log.emit(f"[transient] fan {solver.fluid_fan_energy / 3.6e6:.3f} kWh, "
                              f"delivered by the exchanger "
                              f"{solver.fluid_delivered / 3.6e6:.3f} kWh")
            return results

        self.last_loop = None
        return {"standby": standby, "transient": transient,
                "automesh": automesh}[config.analysis]

    def _fluid_loop(self, config: RunConfig, mesh: Mesh3D | AdaptiveMesh,
                    network: PipeNetwork | None, power: float = 0.0):
        """The gas circuit of ``network`` on ``mesh``, or None without a network.

        With a network the gas *is* the heat transfer path: the loop marches the pipes
        on the mesh (the cells the paint marked) at the circuit's own settings - the gas,
        its pressure, the blower - and ``power`` is the resistors' power it starts from.
        """
        if network is None:
            return None
        if config.pipe_flow <= 0.0:
            raise ValueError(
                "the buried pipe network needs a circuit mass flow > 0 kg/s: set it on "
                "the Gas circuit tab")
        circuit = network.hydraulics(config.pipe_flow, config.pipe_fluid)
        self.log.emit(f"[pipes] {circuit.summary()}")
        return network.fluid_loop(config.pipe_flow, fluid=config.pipe_fluid, mesh=mesh,
                                  external_power=power,
                                  fan_efficiency=config.pipe_fan_efficiency,
                                  pressure=config.pipe_pressure)

    def _log_loop(self) -> None:
        if self.last_loop is not None:
            self.log.emit(f"[loop] {self.last_loop.summary()}")
            for note in self.last_loop.notes:
                self.log.emit(f"[loop] {note}")

    def _finish(self, analysis: str, result) -> None:
        self.finished.emit(analysis, result)

    def _fail(self, message: str) -> None:
        self.log.emit(f"[error] {message}")
        self.failed.emit(message)

    def _cleanup(self) -> None:
        self._job = None
        self.running_changed.emit(False)
