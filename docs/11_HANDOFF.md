# 11. Handoff — state of the work

Written so that a **fresh session** (after `/compact` or in a new chat) can
continue without re-reading the whole conversation.  Read this first, then
`docs/10_MESH_AND_HEATERS.md` for the plan of the work in progress.

## 1. Current state

| | |
|---|---|
| Repository | `C:\Users\tomma\OneDrive - Politecnico di Torino\Documenti\Progetti_prova_VisualStudio\2)Big_energy_self_projects\sand_battery_storage\battery_simulation` |
| Tests | `python -m pytest tests/ -q` → **169 passed** (no display, no GPU) |
| Lint | `python -m ruff check src gui tests scripts --select F,E9,B,SIM,UP` → clean |
| Docs | `docs/00_INDEX.md` … `docs/11_HANDOFF.md` (this file) |
| Commits | **nothing is committed**: the whole refactor lives in the working tree |
| Safety copy | the pre-refactor tree is in `%TEMP%\refactor_baseline` |

## 2. Architecture in one page

```
src/ (no Qt, scriptable)
  constants.py units.py          Kelvin contract + constants
  core/    mesh grid physics materials geometry profiles refinement
  solver/  matrix linear steady transient results
  analysis/fluxes balance losses
  io/state.py                    HDF5 + geometry hash
  viz/scene.py                   PyVista scene builders (shared with tests)
gui/   main_window.py (wiring) controller.py (threads, run state) widgets.py
       units.py safe.py assets.py views/{geometry,materials,analysis,solver,results,viz}
```

Rules that everything follows: temperatures in **Kelvin** inside `src/` (the GUI
converts), matrix coefficients **per unit volume**, sources in `Q_source` with the
driven cells in `source_mask`, boundary conditions attached to **faces**
(`FaceCB`), one flux evaluator shared by solver, balance and reports.
`src/` never imports `gui/`.

## 3. What was done (chronological, all verified)

1. **Numerical audit** of the original code: reference-matrix comparison, analytic
   solutions, execution of the GUI paths → the defect list in `CHANGELOG.md`.
2. **Full refactor**: units, transient (`1/d³` bug), Dirichlet handling, geometry
   validation, GUI modularisation, tests, docs.  Lines: 13 346 → ~6 900 python.
3. **Bug-fix pass from the user's screenshots**: material `cmap` crash, empty-cut
   crash, save/load geometry-hash mismatch, squeezed sliders, missing icon,
   legend/colour-bar overlap, preview gap, **GPU code removed** (CPU only).
4. **Thin-zone voxelization**: zones thinner than a cell (shell, plate, roof) are
   widened to one cell - the roof no longer disappears from the mesh.
5. **Preview reacts to the view controls** (cut axis/position, opacity); the
   colormap selector was removed (coolwarm only).
6. **Graded-mesh core implemented**: `src/core/refinement.py` + `GridSpec` +
   `tests/test_refinement.py` and `tests/test_graded_mesh.py` - refinement targets →
   graded grid with exact band boundaries, bounded growth, no slivers, cell budget.
7. **Graded grid wired into the whole pipeline** (part 2 of `docs/10`): `Mesh3D`
   with per-axis sizes, per-volume coefficients `k_face A/(d_centers V)`, symmetrised
   CG for graded meshes, flux integrals and balance on the local sizes, RectilinearGrid
   rendering, per-axis edges in the state hash.  Uniform meshes keep the old scalar
   path, so their results are unchanged.
8. **Energy-balance bug fixed**: sources in pinned cells and the exchange of nodes
   pinned by another Dirichlet face are no longer counted (the balance used to miss
   0.3 % on the battery and up to 17 % on convection/Dirichlet boxes).
9. **Hairpin heater bank** (part 3 of `docs/10`): `src/core/heaters.py` with
   `HairpinElement`/`HeaterBank`/`rasterize`/`validate_bank`, wired into
   `BatteryGeometry` and the GUI, drawn in the preview; `tests/test_heaters.py`.

10. **Simplification pass**: one assembly path (the uniform fast paths are gone, the
    geometric face factors are cached in `GridIndex`), dead code removed (Mesh3D
    helpers, the rod-heater path), the mesher rewritten (linear ramp + density
    equidistribution, symmetric grids, min/max cell size) and the **automatic mesh
    search** added (`src/analysis/convergence.py`, GUI button *Find the mesh*).

## 4. Work in progress — read `docs/10_MESH_AND_HEATERS.md`

Parts 1-3 are done.  What is left:

* **Part 4 (next): optimisation** - fill `scripts/benchmark.py` with the graded-vs-
  uniform accuracy/cost comparison, check the assembly overhead on graded grids, and
  keep the transient operator rebuild tied to a changed `dt` only.
* **Mesher**: the ramp length is `(h_coarse - h_fine)/(growth - 1)`, so a very fine
  band inside a very coarse box still needs room for it; the GUI summary says when the
  budget coarsened the targets.  A two-level ramp would make deep refinement cheaper.
* **Automatic mesh**: the default geometry (5 kW in a 4 m x 4 m cylinder) does not
  converge to 2 K within the default 400 k-cell budget - the near-wall gradient is too
  steep for that.  The search says so; raising the budget or the tolerance is the
  user's call.  A *sub-grid* heater model would remove the steepest gradient.
* **Heaters in the field view**: the preview draws the hairpins, but the temperature
  field still needs a mesh that resolves a 12 mm sheath (`cells across sheath` = 2 →
  6 mm cells); a *sub-grid* heater model (source spread over the containing cell with a
  contact resistance) would remove that requirement.
* **Balance reporting**: `q_battery` (the envelope integral) is still a separate
  quantity from the domain faces; a single conservation report would be clearer.

## 5. User decisions already taken

* GPU: **removed**, CPU only.
* Colormap: **coolwarm only**, no selector.
* Mesh: the user wants non-uniform sizes driven by *physical targets* chosen in
  the GUI (not a cell count) and results that do not change with the resolution.
* Docstrings and comments: English; the interface labels are English too.

## 6. How to verify quickly

```bash
python -m pytest tests/ -q                       # 147 tests
python run_gui.py                                # GUI
python scripts/benchmark.py --max-cells 60000    # timings
```
Off-screen renders for visual checks (used throughout):
`pyvista.OFF_SCREEN = True` then `src.viz.scene.add_field(...)` /
`add_geometry_preview(...)` and `plotter.screenshot(path)`; read the PNG back.
