# 7. Code structure

Public names are the ones re-exported by the packages
(`src/__init__.py`, `src/core/__init__.py`, `src/solver/__init__.py`,
`src/analysis/__init__.py`, `src/io/__init__.py`, `src/viz/__init__.py`).

## `src/`

| module | responsibility | public API |
|---|---|---|
| `constants.py` | physical constants, default temperatures, Kelvin contract | `T0`, `SIGMA`, `T_AMBIENT_DEFAULT`, `T_GROUND_DEFAULT`, `T_INITIAL_DEFAULT`, `K_AIR`, `RHO_AIR`, `CP_AIR`, `PACKING_FRACTION_DEFAULT`, `DEFAULT_SPACING` |
| `units.py` | degC ↔ K conversion and the Kelvin guard | `c_to_k`, `k_to_c`, `check_kelvin` |
| `core/mesh.py` | the grid, its fields and the face conditions | `Mesh3D`, `MaterialID`, `BoundaryType`, `FaceBC`, `NodeProperties` |
| `core/grid.py` | linear-index tables shared by assembler and analysis | `GridIndex`, `FACE_SLICES`, `INNER_SLICES` |
| `core/physics.py` | interface coefficients (harmonic mean, half cell, radiation) | `harmonic_mean`, `half_cell_h`, `radiation_h` |
| `core/materials.py` | material database + packed-bed model | `MATERIALS`, `ThermalProperties`, `MaterialManager` |
| `core/geometry.py` | geometry configuration and voxel painting | `BatteryGeometry`, `CylinderGeometry`, `HeaterConfig`, `HeaterPattern`, `HeaterElement`, `TubeConfig`, `TubePattern`, `TubeElement`, `BuildReport`, `create_small_test_geometry` |
| `core/profiles.py` | time profiles and initial conditions | `PowerProfile`, `ExtractionProfile`, `InitialCondition` |
| `solver/matrix.py` | steady/transient assembly, Dirichlet elimination | `build_steady_matrix`, `build_transient_operators`, `steady_rhs`, `transient_rhs`, `apply_dirichlet`, `dirichlet_rows`, `face_coefficients` |
| `solver/linear.py` | method/preconditioner selection, diagnostics | `LinearConfig`, `LinearResult`, `solve_linear`, `PreconditionerCache`, `is_symmetric`, `fingerprint`, `set_num_threads` |
| `solver/steady.py` | steady driver (radiation Picard sweeps) | `SteadyStateSolver`, `SolverConfig`, `SolverResult`, `solve_steady_state` |
| `solver/transient.py` | backward-Euler driver, source/extraction handling, cancellation | `TransientSolver`, `TransientConfig`, `run_transient_simulation` |
| `solver/results.py` | transient time-series container | `TransientResults` |
| `analysis/fluxes.py` | the single flux evaluator (envelope, box faces, tubes) | `domain_fluxes`, `domain_face_flux`, `envelope_fluxes`, `tube_flux`, `stored_energy`, `stored_exergy`, `destroyed_exergy` |
| `analysis/balance.py` | energy/exergy snapshot of a mesh state | `Balance`, `compute_balance`, `storage_capacity`, `thermal_autonomy` |
| `analysis/losses.py` | losses analysis (power needed to hold a set point) | `LossesConfig`, `LossesResult`, `solve_losses` |
| `io/state.py` | HDF5 state, geometry hash, unit upgrade | `StateManager`, `SimulationState`, `StateError`, `geometry_hash`, `face_bc_from_state` |
| `viz/scene.py` | rendering helpers shared by GUI and scripts | `to_image_data`, `field_values`, `color_limits`, `cell_centers`, `MATERIAL_NAMES`, `MATERIAL_COLORS`, `export_vtk`, `export_csv` |

## `gui/`

| module | responsibility |
|---|---|
| `main_window.py` | window layout and wiring only (~330 lines) |
| `controller.py` | `RunConfig`, `SimulationJob`, `SimulationController` (threads, cancel, state) |
| `widgets.py` | widget factories; combos carry their value in `itemData` |
| `assets.py` | repository assets (window icon) resolved by path, not by cwd |
| `units.py` | display conversions (degC) |
| `safe.py` | `safe_slot` guard against PyQt aborting on slot exceptions |
| `views/geometry_panel.py` | cylinder, insulation, heaters, tubes, mesh sub-tabs |
| `views/materials_panel.py` | storage medium, insulation, ambient conditions (single authority) |
| `views/analysis_panel.py` | analysis type, initial condition, power profile, extraction, save/load |
| `views/solver_panel.py` | linear solver settings + losses iteration controls |
| `views/results_panel.py` | statistics, energy balance, materials, transient, log |
| `views/viz_view.py` | PyVista scene, clip/opacity/colormap, geometry preview, graceful degradation |

## `tests/` and `scripts/`

| file | content |
|---|---|
| `conftest.py` | shared fixtures: 1D slab, adiabatic box, complete storage model |
| `test_core.py` | units contract, mesh, materials, geometry, profiles |
| `test_solver.py` | assembly exactness, reference matrix, analytic regressions, backends |
| `test_analysis.py` | flux closure, losses iteration, HDF5 round trip |
| `test_gui.py` | head-less GUI smoke test (skipped without PyQt6) |
| `scripts/benchmark.py` | assembly/solve timings per mesh size |

## Entry point

`run_gui.py` adds the repository root to `sys.path` and calls
`gui.main_window.main()`.  Scripts import `src` directly; no GUI import is
needed for any simulation.
