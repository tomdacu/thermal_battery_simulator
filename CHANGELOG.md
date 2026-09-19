# Changelog

## Refactor 2026 - correctness, modularisation, line reduction

The code base was reviewed end to end (numerical audit against analytic
solutions and an independently assembled reference matrix, plus static review of
every module).  The list below is what changed and why; each item was a real
defect with a reproduction, not a stylistic preference.

### Blocking defects fixed

| # | Defect | Evidence | Fix |
|---|--------|----------|-----|
| 1 | Transient dynamics were wrong by a factor `1/d^3` (mass matrix carried the cell volume while the stiffness matrix is per unit volume) | adiabatic test: +80000 K instead of +10 K at d=0.05; ratio tracked `1/d^3` for d = 0.1/0.05/0.025 | `M = diag(rho*cp)` per unit volume; test asserts mesh independence |
| 2 | The transient entry point the GUI called raised `AttributeError` on the first save (`TransientResults.times` was `None`, `T_mean` did not exist, `MaterialID.HEATER` did not exist) | executing the GUI path: `AttributeError: 'NoneType' object has no attribute 'append'` | one transient implementation; results built with `add_timestep` |
| 3 | Fixed-temperature faces did not hold: the Dirichlet row was scaled by `M/dt` in the transient and the steady row replacement broke symmetry | ground plane decayed to 0.08 K instead of 283.15 K; BiCGSTAB reported `info=-10` | symmetric elimination (`apply_dirichlet`) applied after the mass diagonal; CG is now valid |
| 4 | No single temperature-unit convention (degC in the mesh, materials and power balance; Kelvin in the loss analysis and the GUI) | `mesh.py` field default 20.0 vs `energy_balance` ambient `+273.15` | Kelvin everywhere inside `src/`; conversion only in `gui/units.py`; `check_kelvin()` rejects a Celsius field |
| 5 | Heat sources were invisible: the default heater pattern wrote `Q` into SAND cells while the transient and the balance looked for `MaterialID.HEATERS` | `P_input = 0.0` in the balance of a heated model | `mesh.source_mask` + `Q_source`/`Q_sink`; a profile asking for power with no source cell raises |
| 6 | Loss accounting did not describe the simulation: Dirichlet faces always reported zero, convective faces used `h` instead of the half-cell `h_eff`, and losses were integrated over the whole air box | balance imbalance 35 kW on a 50 kW model; `Q_losses_bottom = 0` | one flux evaluator in `src/analysis/fluxes.py` (envelope losses + box-face audit, both solver-consistent) |
| 7 | GUI combo→value maps were keyed on Italian labels while the items were English: tolerance, precision and thread selectors silently fell back to defaults | `git show HEAD:gui/main_window.py` (labels matched before the unfinished translation) | combos carry their value in `itemData`; no string keys anywhere |
| 8 | Extraction was computed but never applied to the physics; `P_extracted` was hardcoded to 0 with a TODO | `transient.py` `'P_extracted': 0.0  # TODO` | extraction drives the solver (tube convection or a capped volumetric sink) and the removed power is measured |
| 9 | The per-face boundary condition was built from the *node's* `bc_h`, so edges/corners used the wrong coefficient | independent reference matrix: 8 rows differ when `h_top != h_lateral` | each face carries its own `FaceBC` |
| 10 | Geometry was silently clipped (roof cut by `Lz`, radius outside the domain), sub-grid heaters/tubes vanished, `estimate_energy_capacity` raised `KeyError` | roof apex 5.33 m in a 5.0 m domain; `KeyError: 'sand_total'` | validation with explicit errors, cell coverage guaranteed, zone/mass keys consistent |
| 11 | `bc_q` (imposed flux) was defined and never used | repo-wide grep: only the definition | `set_heat_flux_bc` implemented and tested |
| 12 | Radiation absent although every material declares an emissivity | ~14 kW at 600 degC on the default envelope | opt-in linearised radiation (`h_r`), Picard sweeps in the steady solver |

### Structural changes

* `src/` no longer depends on the GUI: the losses iteration, the mesh
  construction and the balance moved out of `main_window.py`.
* `main_window.py` 3428 → ~330 lines; panels in `gui/views/`, run orchestration
  in `gui/controller.py` (threads, cancel, state machine).
* One transient implementation (three existed), one balance module (two existed),
  one material database (two existed), one flux evaluator (three existed).
* Removed: `materials_database.py` (orphan duplicate), `config/default_config.yaml`
  (never loaded), `gui/transient_results_widget.py` (never imported, broken),
  `gui/analysis_tab.py::AnalysisTab` (unused), `src/visualization/renderer.py`
  (superseded by `src/viz/scene.py`), the Numba JIT path (untestable here, 1.1-1.5x
  gain over the vectorised NumPy assembly that remains).
* Benchmarks and diagnostics moved from `tests/` to `scripts/`.

### GUI fixes and GPU removal (second pass)

| Defect | Fix |
|---|---|
| `Field → Material` crashed: PyVista wants a list of colour *strings*, not RGB tuples | `src/viz/scene.py::material_cmap()` + the whole scene build moved out of the widget, so it is covered by head-less tests |
| A cut that leaves nothing to draw made PyVista raise on an empty mesh | `add_field` returns `None` and the view prints "nothing to show" |
| Save/load refused every file: the geometry hash broke on string parameters and the window passed different parameters on save and load | type-agnostic canonical hash + `MainWindow._geometry_params()` used symmetrically |
| The slice/opacity sliders were squeezed to dots and unlabelled | two-row control panel with full-width sliders, value labels and tooltips |
| The window icon was missing | `gui/assets.py` resolves `photo/` by path (the old code had a file-name typo) |
| Legend and colour bar overlapped the plot edges | legend top-left, colour bar bottom-right with fixed tick formatting |
| The geometry preview showed a gap under the roof | the preview now draws every zone between the same elevations `apply_to_mesh` uses (slabs, steel plate, foundation, tubes, heaters) |
| GPU code (CuPy/CUDA, PyOpenCL, device CG) | removed on request: one CPU code path through SciPy, ~260 lines and two optional dependencies lighter |

### Graded mesh - refinement core (third pass)

`src/core/refinement.py` + `tests/test_refinement.py`: physical refinement targets
(cells across a region, growth ratio, cell budget) are turned into a graded
rectilinear grid: band boundaries exactly on grid lines, exact axis length,
bounded neighbour growth, no sliver cells, budget respected, overlapping bands
partitioned.  The solver still runs on a uniform grid; wiring the graded grid and
the realistic hairpin heater bank is specified in `docs/10_MESH_AND_HEATERS.md`.

### Fourth pass: preview, thin zones, colormap

* **Thin zones no longer disappear**: a shell, plate or roof thinner than a cell is
  widened to one cell growing away from the storage (the conical roof was missing
  from every voxel model at realistic cell sizes).
* **The geometry preview obeys the view controls**: cut axis, cut position and
  opacity now apply to it exactly as to the field view.
* **Colour map selector removed** (coolwarm only).

### Fifth pass: graded mesh wired into the solver + hairpin heaters

* **Graded mesh live in the solver.**  `Mesh3D` accepts a `GridSpec` (physical
  refinement targets per axis) next to the legacy `spacing`; `dx/dy/dz` are per-axis
  arrays and `V`, `Ax/Ay/Az` the cell volumes and face areas.  The assembly, the
  transient powers, the flux integrals, the balance, the geometry masks, the state
  hash and the 3D view (RectilinearGrid) all use the local sizes.  On a uniform grid
  the code takes the old scalar path, so every previous result is bit-identical.
* **CG is symmetrised on a graded mesh.**  Per-volume coefficients are `diag(V)^-1 K`
  with `K` symmetric: `solve_linear(..., scale=V)` solves the similar symmetric system,
  so the AMG/CG path stays valid.  (On a uniform mesh the transformation is a constant
  and is not applied.)
* **Energy-balance fix.**  Sources inside Dirichlet (pinned) cells were counted as
  input although the identity rows drop them, and the convection/Neumann flux of a node
  pinned by *another* face was reported although the solution never sees it.  Both are
  now excluded, and the Dirichlet flux uses the same centre-to-centre distance as the
  assembly: the balance closes to machine precision (was 0.3 % on the battery, 5-17 %
  on convection/Dirichlet test boxes).
* **Hairpin heaters** (`src/core/heaters.py`): U-shaped sheathed elements, Ø12 mm by
  default, two legs + a 180 deg bend, flange and support plate, rated by the surface
  power density in W/cm² (3-8), rasterised on the graded grid with the active length
  carrying the power and the cold shank only sheath material.  Validations reject
  layouts the mesh cannot represent (legs or elements sharing cells, an element outside
  the wall, a tube collision) and warn about the rest.
* **GUI**: *Geometry → Mesh* now exposes the refinement targets (cells across storage /
  insulation / sheath, far-field size, growth, budget) with a live grid summary;
  *Geometry → Heaters* exposes the hairpin design with a live surface power readout; the
  preview draws the flange, the support plate, the legs and the bends.

### Sixth pass: simplification + symmetric mesher + automatic mesh

* **Simpler assembly**: the dual uniform/graded code path is gone - one set of
  coefficients (`k_face * A/(d_centers V)`) with the geometric factors cached in the
  `GridIndex`.  A uniform grid now builds its edges from a constant step, so its cells
  are bit-identical and the operator stays exactly symmetric without a special case.
* **Dead code removed**: `Mesh3D.cell_volume/get_position/flatten_field/`
  `get_temperature_slice`, the whole rod-heater path (`HeaterElement`,
  `HeaterConfig.generate_positions/_positions/_ring_counts/_ring_points`) and the
  fields it fed (`heater_radius`, `heater_length`, `grid_spacing`, `custom_positions`,
  `cold_shank`); `_n_heaters` counts the bank instead.
* **Mesher rewritten**: size field with a *linear* ramp (per-cell ratio <= growth by
  construction), density equidistribution inside each band, optional min/max cell size,
  budget as a limit of the request.  A symmetric request now gives a **symmetric grid**
  (the old directional walk refined one side more than the other - visible in the 3D
  view as an off-centre halo).
* **Automatic mesh** (`src/analysis/convergence.py` + *Find the mesh* in the GUI):
  refine until the steady answer stops moving by more than the requested dT [K] and
  dP [%], then adopt that grid.  It reports the levels it tried and refuses to call two
  identical grids "converged" when the budget or the size floor is the real limit.
* **Symmetry check in the solver** uses a rounding-level tolerance, so the symmetrised
  operator of a graded mesh keeps CG (and its AMG preconditioner) instead of silently
  falling back to BiCGSTAB.

### Verification

* 58 pytest tests (was 31) including analytic regressions: exact 1D conduction
  and source solutions, an independently assembled reference matrix, mesh
  independence of the transient rate, Dirichlet invariance, balance closure,
  HDF5 round trip and unit upgrade, plus a head-less GUI smoke test.
* `README`, `requirements.txt`, `.gitignore` updated; `LICENSE` (PolyForm-Noncommercial-1.0.0) and
  the README license statement now agree.
