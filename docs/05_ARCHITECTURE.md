# 5. Architecture

## 1. Layers and the one rule that matters

```
        gui/            widgets, panels, threads, 3D view      (Qt)
          │  reads widgets, writes RunConfig
          ▼
      controller.py     run state machine + worker threads
          │  calls the public API of src/
          ▼
        src/            physics, numerics, IO, rendering helpers
```

**`src/` never imports `gui/`.**  Everything that can be computed without a
display lives in `src/` and is exercised by the test suite; the GUI only reads
widgets, assembles a `RunConfig` and displays results.  The consequence is
practical: a simulation can be scripted, profiled and tested head-less, and a
regression cannot hide behind a modal dialog.

`analysis/convergence.py` holds the automatic mesh search: it drives a sequence
of grids through the caller's own build/solve closures and stops when the storage
mean temperature and the heat leaving the battery stop moving by more than the
tolerances the user asked for (grid self-convergence, the idea behind Roache's
Grid Convergence Index).  It is pure: no Qt, no solver import.

Inside `src/` the dependency order is `constants/units → core → solver →
analysis → io/viz`, with one deliberate exception: `analysis` and `solver` both
need the interface coefficients, so those live in `core/physics.py`, and the
index tables both need live in `core/grid.py`.  This keeps the package graph
acyclic (`analysis.balance` is imported by `solver.transient`; the shared helpers
must not live in the solver package or the import order would decide whether the
package loads).

Two layers of work live beside that main path:

* the **gas-side redesign** (`core/pipes.py`, `core/pipe_network.py`,
  `solver/fluid.py`, `analysis/cycle.py`, `core/environment.py`): self-contained
  modules with their own tests, called by scripts and by the GUI's *Pipes* tab, and not
  yet on the `Mesh3D` → solver path - `BatteryGeometry` neither paints the network nor
  fills the excluded-air mask ([13](13_REDESIGN.md) §4-5);
* the **adaptive mesh** (`core/octree.py`): a balanced octree with a conservative face
  list and its own Laplacian, again with its own tests and not yet the mesh of
  `SteadyStateSolver`.

`analysis/convergence.py` holds the automatic mesh search: it drives a sequence
of grids through the caller's own build/solve closures and stops when the storage
mean temperature and the heat leaving the battery stop moving by more than the
tolerances the user asked for (grid self-convergence, the idea behind Roache's
Grid Convergence Index).  It is pure: no Qt, no solver import.
`analysis/mesh_plan.py` computes the a priori cell size per region
(`thickness/N`, `2k/h`) that seeds the search, and `gui/main_window.py` merges the two
(the plan tightens the manual targets, the search moves them).

## 2. Contracts

| Contract | Where | Enforced by |
|---|---|---|
| Temperatures are **Kelvin** inside `src/` | `src/constants.py`, `src/units.py` | `check_kelvin()` on mesh fields, face conditions, profiles; tests |
| Powers in W, energies in J, lengths in m, time in s | everywhere | dataclass type hints + validation |
| Matrix coefficients **per unit volume**; `Q` in W/m³ | `src/solver/matrix.py` | analytic tests; mesh independence of the transient |
| Sources in `Q_source ≥ 0`, sinks in `Q_sink ≤ 0`, driven cells flagged in `source_mask` | `src/core/mesh.py` | `Mesh3D.validate()`, geometry tests |
| Boundary conditions belong to **faces** (`FaceBC`), tube exchange to cells | `src/core/mesh.py` | reference-matrix test |
| Excluded cells carry no conduction; the outer film is `h_out·A·(T − T_ambient)` at active/excluded interfaces; a contact resistance sits between two materials | `src/core/mesh.py` (`excluded`, `h_out`, `h_contact`), `src/solver/matrix.py`, `src/analysis/fluxes.py` | `tests/test_environment.py` (the mask is built by hand: `BatteryGeometry` does not fill it yet) |
| A buried riser's wetted area is the **geometric** `π d L` of the centreline, never the voxel surface | `src/core/pipes.py`, `src/core/pipe_network.py` | `tests/test_pipes.py`, `tests/test_fluid.py` |
| The octree face list is conservative: every interface is claimed exactly once and the flux balance on it is zero | `src/core/octree.py` | `tests/test_octree.py` |
| One flux evaluator for solver, balance and reports | `src/analysis/fluxes.py` | balance closure test |
| Results containers are never `None`-initialised lists | `src/solver/results.py` | transient tests |
| Every state file carries the temperature unit and a geometry hash | `src/io/state.py` | round-trip and mismatch tests |

## 3. Data flow of a run

1. The GUI panels expose *pure getters* (`GeometryPanel.cylinder()`,
   `MaterialsPanel.conditions()`, `AnalysisPanel.transient_settings()`,
   `GeometryPanel.pipe_network_config()`, ...).
2. `RunConfig` (dataclass in `gui/controller.py`) collects them in SI/Kelvin.
3. `SimulationController` builds the domain objects: `Mesh3D` (uniform or graded from
   `mesh_spec`) → `BatteryGeometry.apply_to_mesh` → `SteadyStateSolver` / `solve_losses` /
   `TransientSolver`; the automatic mesh is a fourth job kind (`analysis="automesh"`)
   that runs the same steady solve on a sequence of grids before the mesh is built.
4. The solvers mutate the mesh (`mesh.T`, `mesh.Q_source`, `mesh.Q_sink`) and
   return result objects; the analysis modules read the mesh and compute the
   reported quantities.
5. The GUI displays; nothing in the GUI recomputes physics or converses units
   except through `gui/units.py`.

## 4. Threading and cancellation

One `SimulationJob(QThread)` per run, created by the controller:

* the worker receives a callable `(progress_callback, should_stop)`; every long
  loop (`losses` iteration, transient stepping) checks `should_stop()` and
  returns early instead of blocking the application;
* progress and results cross the thread boundary only through Qt signals
  (queued delivery: the slots run on the GUI thread, so touching widgets there is
  safe);
* `SimulationController.running_changed` drives button enablement centrally - a
  second run cannot start while one is in flight;
* closing the window cancels the job first.

Slots are wrapped with `gui/safe.py::safe_slot`: by default an unhandled
exception inside a PyQt slot **aborts the process**, so the guard turns a mistake
in one handler into a logged error.

## 5. Error handling

* Validation problems are collected as human-readable lists
  (`BatteryGeometry.validate`, `TransientConfig.validate`, `PowerProfile.validate`,
  `LossesConfig.validate`), and the entry points raise `ValueError` with the
  joined list.  The GUI shows them in a dialog and stops.
* Numeric solvers report `converged`, `residual` and a list of `notes`
  (preconditioner fallbacks when PyAMG is missing or an ILU factorisation fails, a
  direct solve that falls back to BiCGSTAB, the switch away from CG, non-converged
  steps) instead of failing silently.
* `StateError` distinguishes "file missing/corrupt" from "incompatible model" and
  carries the reasons.
* Nothing is silently substituted: a profile asking for power with no source cell
  raises, a geometry that does not fit raises, a Celsius field raises.

## 6. Extension points

| Task | Where to change |
|---|---|
| new material | `MATERIALS` in `src/core/materials.py` |
| new heater/tube layout | a branch in `HeaterConfig._positions` / `TubeConfig._positions` (pure functions) |
| new boundary condition type | `BoundaryType` + `FaceBC.is_active` + one branch in `_face_diag_rhs` + `domain_face_flux` |
| new analysis quantity | add it to `Balance` and compute it in `compute_balance` / `fluxes` |
| new linear solver | `METHODS` + one branch in `_solve_iterative` |
| new pipe layout or collection mode | `LAYOUTS` / `COLLECTIONS` and the planner in `src/core/pipe_network.py` (see [15](15_PIPE_NETWORKS.md)) |
| new outer film correlation | `src/core/environment.py` (`h_natural_*`, `h_wind`, `h_out`) |
| new GUI control | the owning panel (getter) + `RunConfig` if it configures the run |
| new reported figure | `ResultsPanel.update_*`, fed from `analysis` |
| a new figure for the documentation | a function in `scripts/figures.py` and a row in the README table |

## 7. What was removed and why

* `gui/transient_results_widget.py`: never imported and broken (it read fields
  that did not exist).  Time series are shown in the *Transient* results tab and
  exported to CSV.
* `src/visualization/renderer.py`: a second, standalone PyVista scene builder
  that the GUI never used; `src/viz/scene.py` is now the single source of the
  grid, the colour tables and the exports.  (The empty `src/visualization/` directory
  is left in the tree.)
* `gui/analysis_tab.py::AnalysisTab`: an unused container whose only real
  content was the wiring the window now does.
* `materials_database.py` (repository root) and `config/default_config.yaml`:
  both orphaned and numerically divergent from the code actually used; the docs
  used to point at them, which made them actively misleading.  (The empty `config/`
  directory is left in the tree.)
* the Numba JIT assembly path: untestable here and worth 1.1-1.5x over the
  vectorised NumPy assembly that remains.
* the GPU backends (CuPy/CUDA and PyOpenCL), the device CG and `accelerators.py`:
  one code path instead of three, no `float32` fallback surprises, no device-specific
  bugs.  Everything runs on CPU through SciPy; the thread budget is still selectable.
* the rod-heater path (`HeaterElement`, `HeaterConfig.generate_positions`,
  `heater_radius`, `heater_length`, `custom_positions`, `cold_shank` as a config
  field): the hairpin bank replaced it ([10](10_MESH_AND_HEATERS.md) §3).
* the GUI colormap selector: the field view is fixed to `coolwarm`, so one scale is
  used everywhere and the colour bar cannot lie.
* Mesh3D helpers that no caller used (`cell_volume`, `get_position`,
  `flatten_field`, `get_temperature_slice`) and the dual uniform/graded assembly path.
* the loop-based matrix builder: kept as a "reference" but never executed; the
  reference now lives in the test suite, where it is actually run.
