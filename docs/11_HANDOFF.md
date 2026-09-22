# 11. Handoff — state of the work

Written so that a **fresh session** (after `/compact` or in a new chat) can
continue without re-reading the whole conversation.  Read this first, then
[docs/13](13_REDESIGN.md) for the architecture the project is moving to, and
[docs/15](15_PIPE_NETWORKS.md) for the buried-pipe networks.

## 1. Current state

| | |
|---|---|
| Repository | `C:\Users\tomma\OneDrive - Politecnico di Torino\Documenti\Progetti_prova_VisualStudio\2)Big_energy_self_projects\sand_battery_storage\battery_simulation` |
| Tests | `python -m pytest tests/ -q --ignore=tests/test_gui_sweep.py` → **300 collected cases** (311 with the GUI sweep); measured 2026-09-20, and 270/259 earlier the same day.  Counts move while work is in flight - re-run the command |
| Lint | `python -m ruff check src tests gui --select F,E9,B,SIM,UP` → **all checks passed** (reading of 2026-09-20, after the six findings in the landing octree-solver files were fixed) |
| Code size | reading of 2026-09-20 (before the octree solver landed): `src` 8659 lines, `gui` 2351, `tests` 3292, `scripts` 392 - python, excluding caches |
| Docs | `docs/00_INDEX.md` … `docs/15_PIPE_NETWORKS.md` |
| Commits | **nothing is committed**: the whole refactor lives in the working tree |
| Safety copy | the pre-refactor tree is in `%TEMP%\refactor_baseline` |

## 2. Architecture in one page

```
src/ (no Qt, scriptable)
  constants.py units.py          Kelvin contract + constants
  core/    mesh grid physics materials geometry profiles
           refinement            graded grid from physical targets
           heaters               hairpin (U-shaped) sheathed elements
           pipes pipe_network    buried pipe runs and vessel-scale networks
           environment           outside film without an air domain
           octree                balanced octree, its own Laplacian (standalone)
  solver/  matrix linear steady transient results
           fluid                 the gas loop: 1-D network, effectiveness, fan
           octree_solver         steady solve + refinement on the leaf list
  analysis/fluxes balance losses
           convergence mesh_plan automatic mesh + a priori cell size
           cycle                 charge / standby / discharge accounting
  io/state.py                    HDF5 + geometry hash
  viz/scene.py                   PyVista scene builders (shared with tests)
gui/   main_window.py (wiring) controller.py (threads, run state) widgets.py
       units.py safe.py assets.py views/{geometry,materials,analysis,solver,results,viz}
scripts/benchmark.py (timings) figures.py (regenerates docs/figures/)
```

Rules that everything follows: temperatures in **Kelvin** inside `src/` (the GUI
converts), matrix coefficients **per unit volume**, sources in `Q_source` with the
driven cells in `source_mask`, boundary conditions attached to **faces**
(`FaceBC`), one flux evaluator shared by solver, balance and reports.
`src/` never imports `gui/`.

## 3. What was done (chronological, all verified)

1. **Numerical audit** of the original code: reference-matrix comparison, analytic
   solutions, execution of the GUI paths → the defect list in `CHANGELOG.md`.
2. **Full refactor**: units, transient (`1/d³` bug), Dirichlet handling, geometry
   validation, GUI modularisation, tests, docs.
3. **Bug-fix pass from the user's screenshots**: material `cmap` crash, empty-cut
   crash, save/load geometry-hash mismatch, squeezed sliders, missing icon,
   legend/colour-bar overlap, preview gap, **GPU code removed** (CPU only).
4. **Thin-zone voxelization**: zones thinner than a cell (shell, plate, roof) are
   widened to one cell - the roof no longer disappears from the mesh.
5. **Preview reacts to the view controls** (cut axis/position, opacity); the
   colormap selector was removed (coolwarm only).
6. **Graded-mesh core implemented**: `src/core/refinement.py` + `GridSpec` +
   `tests/test_refinement.py` and `tests/test_graded_mesh.py`.
7. **Graded grid wired into the whole pipeline**: `Mesh3D` with per-axis sizes,
   per-volume coefficients `k_face A/(d_centers V)` cached in `GridIndex`,
   symmetrised CG for graded meshes, flux integrals and balance on the local sizes,
   `RectilinearGrid` rendering, per-axis edges in the state hash.
8. **Energy-balance bug fixed**: sources in pinned cells and the exchange of nodes
   pinned by another Dirichlet face are no longer counted.
9. **Hairpin heater bank**: `src/core/heaters.py` with
   `HairpinElement`/`HeaterBank`/`rasterize`/`validate_bank`, wired into
   `BatteryGeometry` and the GUI, drawn in the preview; `tests/test_heaters.py`.
10. **Simplification pass**: one assembly path (the uniform fast paths are gone),
    dead code removed (Mesh3D helpers, the rod-heater path), the mesher rewritten
    (linear ramp + density equidistribution, symmetric grids, min/max cell size)
    and the **automatic mesh search** added (`src/analysis/convergence.py`).
11. **Redesign, electrically: the gas loop.**  `src/solver/fluid.py` (1-D network,
    effectiveness relation, film coefficient from the flow, Sutherland viscosity,
    friction factor, pressure drop, fan power, circulation loss) and
    `src/core/pipes.py` (polyline runs, geometric per-cell wetted area, bank
    layouts with the published pitches); `tests/test_fluid.py`.
12. **Vessel-scale pipe networks.**  `src/core/pipe_network.py` with four layouts,
    four collection modes, the two nozzles through the wall, the branch split, the
    header rules and the voxelisation; the *Geometry → Pipes* tab drives it;
    `tests/test_pipes.py`; documented in [15](15_PIPE_NETWORKS.md).
13. **Outside film without the air domain.**  `src/core/environment.py`
    (Churchill-Chu natural convection, `4 + 4 v` wind, `h_out` as their sum),
    `Mesh3D.excluded/h_out/t_ambient/h_contact`, the film on active/excluded
    interfaces in `src/solver/matrix.py`, the matching term in
    `src/analysis/fluxes.py::environment_flux` and the audit in `balance.py`;
    `tests/test_environment.py`.
14. **Adaptive mesh core.**  `src/core/octree.py`: cell-level linear octree with the
    2:1 balance, a conservative face list, a Laplacian on the same per-volume form
    and a gradient indicator; `tests/test_octree.py`.
15. **Cycle accounting.**  `src/analysis/cycle.py` drives charge, standby and
    discharge through the gas loop and decomposes the energy
    (`E_in = ΔE_stored + E_delivered + E_standby + E_circulation + E_unrecovered`),
    stopping each phase where the physics says so; `tests/test_cycle.py`.
16. **Reproducible figures.**  `scripts/figures.py` writes every figure in
    `docs/figures/` offscreen; the README lists them with the function that draws
    each one.
17. **Octree on the physics path.**  `src/solver/octree_solver.py` solves steady
    conduction on a leaf list (equivalence with the structured solver on a matching
    uniform grid, a conservative flux report and an objective-driven refinement
    cycle), `tests/test_octree_solver.py`; the octree is not yet the mesh of
    `Mesh3D`/`SteadyStateSolver`.
18. **Documentation catch-up.**  `README.md` and `docs/00`-`docs/13` rewritten against
    the code that exists: module maps and test counts are measurements (command and
    date recorded in [09](09_TESTING.md)), the GUI defaults were read from the live
    panels, the figures section lists the files that exist, and the status of every
    redesign piece is stated with the file and the test behind it
    ([13](13_REDESIGN.md) §8).  See `CHANGELOG.md` for the list and for what stayed
    open.

## 3b. 2026-09-23: the solver checked, the GUI cut to the plant

Read `CHANGELOG.md` (top section) for the list with the evidence.  In short: the octree
caches its face list (a transient step went from seconds to tens of milliseconds on the
test tree), the gas loop is coupled implicitly with its balance held per step, the steady
and losses runs couple the loop too (no more uniform source plus a 60 degC pipe sink),
every run sets its own state on the mesh, the AMG cache and the thread setting work, and
the GUI shows only the plant (octree, Gas circuit, Pipes, an exchanger) with CG + AMG as
its linear solver.  Suite: 386 passed (375 without the GUI sweep), ruff clean.

Open, in order: (1) gas properties that follow the gas temperature along the loop
([12](12_METHODS.md) §11 "Limits"); (2) the standby paths of [13](13_REDESIGN.md) §7;
(3) the loop march is a Python loop over the cells of each run - fine for the default
network (42 risers), worth vectorising for a network of thousands of runs.

## 4. Work in progress / next

* **Octree in the main solver**: `src/core/octree.py` (mesh, balance, conservative
  faces) plus `src/solver/octree_solver.py` (steady solve, flux report, the objective
  refinement `refine_on_objective`) are the adaptive path and are tested
  (`tests/test_octree.py`, `tests/test_octree_solver.py`); `Mesh3D`/`SteadyStateSolver`
  still do not route a geometry into it, so the graded Cartesian grid remains the
  production mesh ([13](13_REDESIGN.md) §5).
* **Air box removal**: `mesh.excluded`/`h_out` work and are tested, but nothing fills
  them from `BatteryGeometry`, which still paints the air and sets box-face
  convection.  This is the step that removes the air from a default run
  ([13](13_REDESIGN.md) §4).
* **Pipe network**: the extension has landed (spiral layout, wall thickness/material,
  roughness, insulated headers, `paint`, hydraulics, the `FluidLoop` built from the
  network); what is left is whatever [docs/15](15_PIPE_NETWORKS.md) still lists as open.
* **Cycle**: the per-step stopping criterion and the energy decomposition are in
  (`tests/test_cycle.py` covers the charge stop, the discharge floor, the fan per phase
  and the identity); a full cycle of the default machine has not been reported yet.
* **Part 4 of [docs/10](10_MESH_AND_HEATERS.md)**: the graded-vs-uniform accuracy and
  cost measurements, and a possible sub-grid heater model.
* **Mesher**: the ramp length is `(h_coarse - h_fine)/(growth - 1)`, so a very fine
  band inside a very coarse box still needs room for it; the GUI summary says when the
  budget coarsened the targets.
* **Automatic mesh**: the default geometry (5 kW in a 4 m x 4 m cylinder) does not
  converge to 2 K within the default 400 k-cell budget - the search says so; raising
  the budget or the tolerance is the user's call.
* **Standby paths**: the insulation thermal bridges (pipe penetrations, foundation)
  are not modelled ([13](13_REDESIGN.md) §7).
* **Balance reporting**: `q_battery` (the envelope integral) is still a separate
  quantity from the domain faces; a single conservation report would be clearer.
* **Lint**: clean as of the 2026-09-20 reading.  The one failing test on this machine is a
  wall-clock budget, not physics:
  `tests/test_octree.py::test_a_tree_of_twenty_thousand_leaves_builds_and_lists_its_faces_in_under_a_second`
  measured 1.30-1.55 s against an assertion of 1.0 s with the machine idle (2.06 s under load);
  the owner of that module has it.

## 5. User decisions already taken

* GPU: **removed**, CPU only.
* Colormap: **coolwarm only**, no selector.
* Mesh: the user wants non-uniform sizes driven by *physical targets* chosen in
  the GUI (not a cell count) and results that do not change with the resolution.
* Architecture: the machine is the **closed gas loop through buried pipes**
  ([13](13_REDESIGN.md)) - the resistors belong in the gas circuit, and the bed is
  heated and discharged by the pipe network.
* Docstrings and comments: English; the interface labels are English too.

## 6. How to verify quickly

```bash
python -m pytest tests/ -q --ignore=tests/test_gui_sweep.py   # 259 cases, head-less
python -m pytest tests/ -q                                    # + the GUI control sweep
python -m ruff check src tests gui --select F,E9,B,SIM,UP
python run_gui.py                                             # GUI
python scripts/benchmark.py --max-cells 60000                 # timings
python scripts/figures.py                                     # regenerate docs/figures/
```

Off-screen renders for visual checks (used throughout):
`pyvista.OFF_SCREEN = True` then `src.viz.scene.add_field(...)` /
`add_geometry_preview(...)` and `plotter.screenshot(path)`; read the PNG back.

## 7. Where the redesign stands

The architecture, the per-piece state and the measured numbers are in
[13_REDESIGN.md](13_REDESIGN.md) §8 - that section is the single place to look, and it
names the file and the test behind every row.  In one line: the gas loop, the pipe
networks, the graded mesh, the outside-film *capability* and the octree *core* exist;
the air box is still in the default geometry, the octree is not yet the solver's mesh,
and the cycle is being completed.
