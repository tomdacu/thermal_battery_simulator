# 7. Code structure

Two levels of API are worth knowing:

* the names a package **re-exports** (`src/__init__.py`, `src/core/__init__.py`,
  `src/solver/__init__.py`, `src/analysis/__init__.py`, `src/io/__init__.py`,
  `src/viz/__init__.py`, `gui/views/__init__.py`) - the convenient surface;
* the module-level names below - what a caller imports from the module itself.
  Several modules are deliberately *not* re-exported (`core/grid.py`,
  `core/physics.py`, `core/refinement.py`, `core/octree.py`, `core/heaters.py`,
  `core/pipes.py`, `core/pipe_network.py`, `core/environment.py`,
  `solver/fluid.py`, `analysis/cycle.py`, `analysis/convergence.py`,
  `analysis/mesh_plan.py`), so the package surface stays small and a rename inside
  those modules cannot ripple through the application.

## `src/`

| module | responsibility | public API |
|---|---|---|
| `constants.py` | physical constants, default temperatures, Kelvin contract | `T0`, `SIGMA`, `T_AMBIENT_DEFAULT`, `T_GROUND_DEFAULT`, `T_INITIAL_DEFAULT`, `T_MIN_VALID`, `K_AIR`, `RHO_AIR`, `CP_AIR`, `GRAVITY`, `PACKING_FRACTION_DEFAULT`, `DEFAULT_SPACING`, `MIN_CELLS_PER_AXIS`, `EPS` |
| `units.py` | degC ↔ K conversion and the Kelvin guard | `c_to_k`, `k_to_c`, `check_kelvin` |
| `core/mesh.py` | the grid (per-axis sizes, `excluded`, `h_out`, `h_contact`), its fields and the face conditions | `Mesh3D`, `MaterialID`, `BoundaryType`, `FaceBC`, `NodeProperties`, `FACES` |
| `core/grid.py` | linear-index tables shared by assembler and analysis, cached face factors | `GridIndex`, `FACE_SLICES`, `INNER_SLICES`, `FACE_AXIS`, `dirichlet_mask` |
| `core/physics.py` | interface coefficients (harmonic mean, half cell, radiation) | `harmonic_mean`, `half_cell_h`, `radiation_h` |
| `core/materials.py` | material database + packed-bed model | `MATERIALS`, `STORAGE_MATERIALS`, `INSULATION_MATERIALS`, `CATEGORIES`, `ThermalProperties`, `MaterialManager` |
| `core/geometry.py` | geometry configuration and voxel painting | `BatteryGeometry`, `CylinderGeometry`, `HeaterConfig`, `HeaterPattern`, `TubeConfig`, `TubePattern`, `TubeElement`, `BuildReport`, `create_small_test_geometry` |
| `core/heaters.py` | hairpin (U-shaped) sheathed elements, their bank, rasteriser and validation | `HairpinElement`, `HeaterBank`, `RasterResult`, `rasterize`, `validate_bank`, `SURFACE_POWER_MIN_W_CM2`, `SURFACE_POWER_LIMIT_W_CM2`, `DEFAULT_SHEATH_DIAMETER`, `DEFAULT_SHEATH_MATERIAL` |
| `core/refinement.py` | physical refinement targets → graded rectilinear grid | `Band`, `GridSpec`, `partition`, `graded_edges`, `uniform_edges`, `edges_to_centers`, `edges_to_sizes`, `bands_from_triples`, `size_at`, `worst_ratio`, `describe` |
| `core/octree.py` | cell-level octree with the 2:1 balance, conservative face list and its own Laplacian | `Octree`, `Leaf`, `uniform_tree`, `refine_by_gradient`, `MORTON_BITS`, `FACES`, `FACE_AXIS`, `FACE_SIGN` |
| `core/pipes.py` | buried pipe runs: geometry, rasterisation, bank layouts, design pitches | `PipeRun`, `BankLayout`, `rasterize_pipe`, `vertical_bank`, `serpentine`, `staggered_bank`, `describe`, `PITCH_HORIZONTAL`, `PITCH_VERTICAL`, `PITCH_TRIANGULAR`, `HEADER_SAFE`, `HEADER_LIMIT` |
| `core/pipe_network.py` | vessel-scale networks: layouts (incl. the spiral), collection modes, wall thickness/material/roughness, insulated headers, the branch split, voxelisation (`paint`), hydraulics and the `FluidLoop` built from a network - see [15](15_PIPE_NETWORKS.md) | `PipeNetworkConfig`, `PipeNetwork`, `PipeMaterial`, `Branch`, `Hydraulics`, `PaintReport`, `build_pipe_network`, `LAYOUTS`, `COLLECTIONS`, `BALANCED_COLLECTIONS`, `SPLITS`, `LAYOUT_*`, `COLLECTION_*`, `SPLIT_*`, `MODULE_HEIGHT_LIMIT`, `WALL_STUB`, `TWO_LEVEL_GAP`, `WARNING` |
| `core/environment.py` | the outside film (natural + wind) without simulating the air | `AirProperties`, `rayleigh`, `h_natural_vertical`, `h_natural_horizontal`, `h_wind`, `h_out` |
| `core/profiles.py` | time profiles and initial conditions | `PowerProfile`, `ExtractionProfile`, `InitialCondition`, `POWER_MODES`, `EXTRACTION_MODES`, `IC_MODES` |
| `solver/matrix.py` | steady/transient assembly, exclusion and contact handling, Dirichlet elimination | `face_coefficients`, `build_steady_matrix`, `build_transient_operators`, `steady_rhs`, `transient_rhs`, `apply_dirichlet`, `dirichlet_rows` |
| `solver/linear.py` | method/preconditioner selection, diagnostics | `LinearConfig`, `LinearResult`, `solve_linear`, `PreconditionerCache`, `is_symmetric`, `fingerprint`, `set_num_threads`, `METHODS`, `PRECONDITIONERS` |
| `solver/steady.py` | steady driver (radiation Picard sweeps) | `SteadyStateSolver`, `SolverConfig`, `SolverResult`, `solve_steady_state` |
| `solver/transient.py` | backward-Euler driver, source/extraction handling, cancellation | `TransientSolver`, `TransientConfig`, `run_transient_simulation` |
| `solver/fluid.py` | the gas loop as a 1-D network: effectiveness, film coefficient, pressure, fan | `Fluid`, `FluidLoop`, `FluidResult`, `RunResult`, `pipe_h`, `friction_factor`, `pressure_drop`, `fan_power` |
| `solver/results.py` | transient time-series container | `TransientResults`, `SCALAR_FIELDS` |
| `solver/octree_solver.py` | steady conduction on the adaptive octree: wall sets, the leaf assembly, the flux report and the objective-driven refinement (`refine_on_objective`) - the first piece that puts the octree on the physics path, not yet the mesh of `SteadyStateSolver` | `OctreeSteadySolver`, `OctreeSteadyResult`, `OctreeAdaptiveReport`, `refine_on_objective`, `wall_cells`, `dirichlet_walls`, `mean_temperature`, `storage_mean_objective` |
| `analysis/fluxes.py` | the single flux evaluator (envelope, excluded-air film, box faces, tubes) | `domain_fluxes`, `domain_face_flux`, `envelope_fluxes`, `environment_flux`, `tube_flux`, `stored_energy`, `stored_exergy`, `destroyed_exergy` |
| `analysis/balance.py` | energy/exergy snapshot of a mesh state | `Balance`, `compute_balance`, `storage_capacity`, `thermal_autonomy` |
| `analysis/losses.py` | losses analysis (power needed to hold a set point) | `LossesConfig`, `LossesResult`, `solve_losses` |
| `analysis/convergence.py` | automatic mesh search (Richardson/GCI) | `ConvergenceTarget`, `ConvergenceLevel`, `ConvergenceReport`, `find_mesh` |
| `analysis/mesh_plan.py` | a priori cell size per region | `RegionPlan`, `plan_regions`, `surface_layer`, `layer_target`, `binding_target`, `describe`, `CELLS_PER_LAYER` |
| `analysis/cycle.py` | charge / standby / discharge accounting | `CycleSettings`, `CyclePhase`, `CycleReport`, `run_cycle` |
| `io/state.py` | HDF5 state, geometry hash, unit upgrade | `StateManager`, `SimulationState`, `StateError`, `geometry_hash`, `face_bc_from_state`, `FORMAT_VERSION`, `TEMPERATURE_UNIT` |
| `viz/scene.py` | rendering helpers shared by GUI and scripts | `to_image_data`, `field_values`, `color_limits`, `cell_centers`, `clip_grid`, `material_cmap`, `add_field`, `add_material_legend`, `add_geometry_preview`, `export_vtk`, `export_csv`, `FIELD_ARRAYS`, `FIELD_UNITS`, `AXIS_INDEX`, `MATERIAL_NAMES`, `MATERIAL_COLORS` |

## `gui/`

| module | responsibility |
|---|---|
| `main_window.py` | window layout and wiring only: `ThermalBatteryGUI`, `main`, `HELP_TEXT` |
| `controller.py` | `RunConfig`, `SimulationJob`, `SimulationController` (threads, cancel, state), `ANALYSIS_TYPES` |
| `widgets.py` | widget factories (`double_spin`, `int_spin`, `combo`, `check`, `button`, `hint`), `FormPanel` with its register and value reader; combos carry their value in `itemData` |
| `assets.py` | repository assets by path, not by cwd: `repo_root`, `asset`, `window_icon_path` |
| `_binding.py` | pins `QT_API=pyside6` before PyVista/QtPy picks a binding, imported first by the package |
| `units.py` | display conversions and formatting: `fmt_c`, `fmt_k`, `celsius_span` |
| `safe.py` | `safe_slot`, the guard that reports a failing slot in the log and the status bar |
| `views/geometry_panel.py` | Cylinder, Insulation, Heaters, Tubes, Mesh, Pipes sub-tabs |
| `views/materials_panel.py` | storage medium, insulation, ambient conditions (single authority) |
| `views/analysis_panel.py` | analysis type, initial condition, power profile, extraction, save/load |
| `views/solver_panel.py` | linear solver settings + losses iteration controls |
| `views/results_panel.py` | statistics, energy balance, materials, transient, log |
| `views/viz_view.py` | PyVista scene, cut/position/opacity controls, geometry preview, graceful degradation |

## `tests/` and `scripts/`

Collected cases as reported by `python -m pytest tests/ --collect-only -q`
(measured 2026-09-20; see [09](09_TESTING.md) for what each file checks):

| file | cases | content |
|---|---|---|
| `tests/conftest.py` | - | fixtures: 1D slab, adiabatic box, complete storage model |
| `tests/test_pipes.py` | 73 | buried pipe networks: layouts, collection, splits, paint, hydraulics |
| `tests/test_scene.py` | 72 | rendering path off-screen (fields, cuts, colormaps, exports) |
| `tests/test_core.py` | 23 | units contract, mesh, materials, geometry, profiles |
| `tests/test_fluid.py` | 23 | the gas loop and the pipe banks |
| `tests/test_solver.py` | 23 | assembly exactness, reference matrix, analytic regressions, backends |
| `tests/test_analysis.py` | 14 | flux closure, losses iteration, HDF5 round trip |
| `tests/test_octree.py` | 13 | 2:1 balance, conservative faces, an octree Laplacian |
| `tests/test_cycle.py` | 13 | charge and discharge through the gas loop |
| `tests/test_paint_on_tree.py` | 13 | the painters on the tree against their structured twin |
| `tests/test_octree_solver.py` | 12 | the octree wired to the physics: equivalence, conservation, refinement |
| `tests/test_adaptive_mesh.py` | 11 | the ``MeshAPI`` protocol, analytic cases, refinement |
| `tests/test_environment.py` | 11 | outside film, excluded air, contact resistance |
| `tests/test_gui_sweep.py` | 11 | systematic control sweep (head-less, slow) |
| `tests/test_hydraulics.py` | 10 | the network's hydraulics and the header engine |
| `tests/test_refinement.py` | 10 | graded-grid generation |
| `tests/test_balance_on_tree.py` | 9 | the energy balance on the tree and on the structured twin |
| `tests/test_convergence.py` | 9 | automatic mesh search and its report |
| `tests/test_gui.py` | 9 | head-less GUI smoke test (skipped without PySide6) |
| `tests/test_transient_on_tree.py` | 8 | the transient march on the tree |
| `tests/test_convergence_on_tree.py` | 7 | the same search driven by an ``AdaptivePlan`` |
| `tests/test_geometry_on_a_tree.py` | 7 | the painter on the tree against the structured twin |
| `tests/test_box_tree.py` | 5 | the anisotropic tree of boxes against the octree |
| `tests/test_graded_mesh.py` | 5 | graded grid inside the solver |
| `tests/test_mesh_plan.py` | 5 | the a priori plan: active regions and bands |
| `tests/test_packed_bed.py` | 5 | the Zehner-Bauer-Schluender bed |
| `tests/test_steady_on_tree.py` | 5 | the steady solve on the tree and its structured twin |
| `tests/test_well_model.py` | 4 | the Peaceman well model and the gas circuit |
| `tests/test_thin_layers_and_soil.py` | 2 | a thin shell in series, the soil layer |
| `scripts/benchmark.py` | - | assembly/solve timings per mesh size (`build_model`, `main`) |
| `scripts/figures.py` | - | regenerates `docs/figures/`: `battery_with_pipes`, `temperature_field`, `graded_mesh`, `adaptive_concept`, `gas_loop_schema`, `design_curves`, `main` |

The counts move while work is in flight; the table is the reading of
`python -m pytest tests/ --collect-only -q` on **2026-09-29 (412 total, 401 without the
GUI sweep, after the audit removed the cases that could not fail)**; earlier the same
day the same command said 418/406, and on 2026-09-20 it said 311/300, with 295/284 and 270/259
earlier in that session as work landed.  Re-run the command instead of trusting it;
[09](09_TESTING.md) §1 records the same reading with its caveat and what each case
checks.

## Entry point

`run_gui.py` adds the repository root to `sys.path` and calls
`gui.main_window.main()`.  Scripts import `src` directly; no GUI import is
needed for any simulation.
