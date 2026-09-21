"""Run configuration and the worker/state machine that executes a simulation.

The GUI never runs physics: it assembles a :class:`RunConfig` from its widgets
and hands it to :class:`SimulationController`, which owns the threads, the
progress reporting and the cancel flag.  All the numerics happen in ``src/``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from src.analysis.balance import compute_balance
from src.analysis.convergence import AdaptivePlan, ConvergenceTarget, find_mesh
from src.analysis.mesh_plan import describe as describe_plan, plan_regions
from src.core.materials import MaterialManager
from src.analysis.losses import LossesConfig, solve_losses
from src.core.adaptive_mesh import AdaptiveMesh
from src.core.geometry import BatteryGeometry
from src.core.mesh import Mesh3D
from src.core.pipe_network import PipeNetwork
from src.core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from src.solver.steady import SolverConfig, SteadyStateSolver
from src.solver.transient import TransientConfig, TransientSolver

ANALYSIS_TYPES = ("steady", "losses", "transient")


@dataclass
class RunConfig:
    """Everything a run needs, already validated and in SI/Kelvin units."""

    analysis: str = "steady"
    battery: BatteryGeometry = field(default_factory=BatteryGeometry)
    domain: tuple = (6.0, 6.0, 5.6, 0.2)
    #: refinement spec of the automatic mesh search (None = uniform mesh)
    mesh_spec: object = None
    #: tolerances of the search: delta_temperature [K], delta_power [-], levels, budget
    convergence: dict = field(default_factory=dict)
    method: str = "bicgstab"
    preconditioner: str = "jacobi"
    tolerance: float = 1e-8
    max_iterations: int = 5000
    n_threads: int = -1
    radiation: bool = False
    losses: dict = field(default_factory=dict)
    transient: dict = field(default_factory=dict)
    initial_condition: InitialCondition = field(default_factory=InitialCondition)
    power_profile: PowerProfile = field(default_factory=PowerProfile)
    extraction_profile: ExtractionProfile = field(default_factory=ExtractionProfile)
    start_from_steady: bool = False
    #: the buried pipe network the Pipes tab built and painted (None = the lumped
    #: tube bank of the geometry is the heat exchanger)
    pipe_network: PipeNetwork | None = None
    #: total mass flow of the gas circuit of that network [kg/s]
    pipe_flow: float = 0.0

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
        self._mesh: Mesh3D | AdaptiveMesh | None = None
        self._config: RunConfig | None = None

    @property
    def running(self) -> bool:
        return self._job is not None and self._job.isRunning()

    # ------------------------------------------------------------------ api
    def start(self, config: RunConfig, mesh: Mesh3D | AdaptiveMesh | None) -> None:
        if self.running:
            self.failed.emit("a simulation is already running")
            return
        self._config, self._mesh = config, mesh
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

        def steady(progress, should_stop):
            self.log.emit("[steady] solving")
            result = SteadyStateSolver(mesh, solver_config).solve()
            self.log.emit(f"[steady] converged={result.converged} "
                          f"residual={result.residual:.2e} in {result.solve_time:.2f} s")
            for note in result.notes:
                self.log.emit(f"[solver] {note}")
            return result

        def losses(progress, should_stop):
            cfg = LossesConfig(**config.losses)
            self.log.emit("[losses] iterating on the heater power")
            return solve_losses(mesh, cfg, solver_config, progress, should_stop)

        def automesh(progress, should_stop):
            spec = config.mesh_spec
            if spec is None:
                raise ValueError("the automatic mesh search needs a refined mesh spec")
            settings = {k: v for k, v in (config.convergence or {}).items()
                        if k in ConvergenceTarget.__dataclass_fields__}
            target = ConvergenceTarget(**settings)
            self.log.emit(f"[automesh] tolerances: dT {target.delta_temperature} K, "
                          f"dP {100 * target.delta_power:.1f}%, budget "
                          f"{target.max_cells:,} cells")

            plans = plan_regions(config.battery, MaterialManager())
            for line in describe_plan(plans).splitlines():
                self.log.emit(f"[plan] {line}")

            def build(spec_level):
                """One search level, on the mesh its request describes.

                The panel hands the search an ``AdaptivePlan`` when the Mesh tab builds a
                tree and a ``GridSpec`` when it builds a graded grid: the same physical
                targets in the vocabulary of their own mesh, so the search refines - and
                the chosen level rebuilds - the mesh mode that is selected.
                """
                if isinstance(spec_level, AdaptivePlan):
                    mesh = AdaptiveMesh.from_bands(spec_level.n_finest,
                                                   spec_level.physical_size,
                                                   spec_level.bands, spec_level.base_level)
                else:
                    mesh = Mesh3D(Lx=lx, Ly=ly, Lz=lz, grid=spec_level)
                config.battery.apply_to_mesh(mesh)
                return mesh

            def observables(mesh):
                SteadyStateSolver(mesh, solver_config).solve()
                balance = compute_balance(mesh)
                return {"t_mean_storage": balance.t_mean_storage,
                        "t_max": balance.t_max, "power": balance.q_battery}

            lx, ly, lz, _ = config.domain
            report = find_mesh(build, observables, spec, target, progress, should_stop)
            for line in report.summary().splitlines():
                self.log.emit(f"[automesh] {line}")
            return report

        def transient(progress, should_stop):
            if config.start_from_steady:
                self.log.emit("[transient] steady pre-run for the initial field")
                SteadyStateSolver(mesh, solver_config).solve()
            settings = dict(config.transient)
            settings.setdefault("save_full_field", False)
            t_cfg = TransientConfig(
                initial_condition=config.initial_condition,
                power_profile=config.power_profile,
                extraction_profile=config.extraction_profile,
                t_ambient=config.battery.t_ambient,
                fluid_loop=self._fluid_loop(config, mesh),
                **settings,
            )
            return TransientSolver(mesh, t_cfg, solver_config).run(progress, should_stop)

        return {"steady": steady, "losses": losses, "transient": transient,
                "automesh": automesh}[config.analysis]

    def _fluid_loop(self, config: RunConfig, mesh: Mesh3D | AdaptiveMesh):
        """The gas circuit of a network the Pipes tab built, or None.

        With a network the gas *is* the heat transfer path: the loop marches the pipes
        on the mesh (the cells the paint marked) and the profiles drive its external
        power instead of depositing heat in the sand.  Without one the lumped tube bank
        of the geometry keeps doing the job, as it always did.
        """
        network = config.pipe_network
        if network is None:
            return None
        if config.pipe_flow <= 0.0:
            raise ValueError(
                "the buried pipe network needs a circuit mass flow > 0 kg/s: set it in "
                "the Pipes tab")
        circuit = network.hydraulics(config.pipe_flow)
        self.log.emit(f"[pipes] {circuit.summary()}")
        return network.fluid_loop(config.pipe_flow, mesh=mesh)

    def _finish(self, analysis: str, result) -> None:
        self.finished.emit(analysis, result)

    def _fail(self, message: str) -> None:
        self.log.emit(f"[error] {message}")
        self.failed.emit(message)

    def _cleanup(self) -> None:
        self._job = None
        self.running_changed.emit(False)
