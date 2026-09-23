# Changelog

## 2.0.0 - 2026-09-24

The first tagged version.  Everything since the last push is in it: the anisotropic tree
of boxes, the well model, the whole gas circuit and its hydraulics, the header engine and
the radial manifold, the Zehner-Bauer-Schlünder bed, the thin layers and the soil, and:

* **the mesh coupled to the model**: the window keeps a digest of the model the mesh was
  built from; an edit on the vessel, the materials, the site, the mesh settings or the
  plant marks it out of date (the 3D view and the Mesh tab say so) and *Run* rebuilds it
  before it solves - before, a run after an edit solved the old mesh with the new
  panels;
* the house-style banner and icon (`scripts/banner.py`, `assets/`), the README restyled;
* the repository renamed `thermal_battery_simulator`.

## 2026-09-23 (night) - the flows from the hydraulics, the headers from an engine

* `src/solver/hydraulics.py`: the circuit as a looped pipe network (Darcy-Weisbach, a
  continuous friction factor, Idelchik's tee losses, the draught of the hot gas), node
  pressures by Newton.  The riser flows are solved; the loop re-solves them at the gas
  temperatures and turns a segment round if its flow reverses.
* The **header engine** (`design_headers`): targets = each riser's share of the bed
  (Voronoi), header groups sized on EN 10220 nominal sizes under a velocity limit and
  grown while they pay in pressure drop, calibrated riser orifices (K, and the plate hole
  by Idelchik) where needed, the headers lifted into the sand and iterated with the sizes;
  tries the chosen collection and the **radial manifold** and keeps the better.
* Default plant: the reverse-return ring chain, solved, gave riser flows from 0.01 to 5x
  the mean with 150 mm headers and needs 52 m/s at any size that fits; the manifold runs
  at 17 m/s and 751 Pa, orifices on 124 risers, headers 243 mm into the sand.
* The preview draws every pipe at its own diameter; *Geometry* is an entry of the 3D
  field selector, and the fields come back from it (the view was stuck on the geometry).
* The duct diameter control is gone (the engine's); the collection defaults to the
  radial manifold, the split to the hydraulics.
* [docs/19_HYDRAULICS.md](docs/19_HYDRAULICS.md); 15, 17, 18, 06, 09, 12, README updated.

## 2026-09-23 (evening) - phase 1 and 2: the mesh follows the vessel, the physics the plant

| # | Change | Evidence |
|---|--------|----------|
| 1 | **Tree of boxes** (`src/core/box_tree.py`): leaves with their own plan edge and height, 2:1 per direction, arrays only; the GUI's mesh | default 36 268 leaves in 2.3 s (the cubic octree: 214 089 in 12 s) |
| 2 | **Well model** (Peaceman): the bed between the tube wall and the cell centre in the tube's `UA`; no refinement around the tubes any more | one tube in a square of sand: within 3.5 % of the shape factor on 300 and 150 mm cells (without: +26 % / +5 %, -9 % on 75 mm) |
| 3 | **The whole gas circuit** marched: inlet duct, distributor, risers, collector, outlet duct, each segment with its flow, enthalpy mixing at the nodes; the balance on the exchange | the headers carry 28 % of the wetted area and now exchange |
| 4 | **Gas properties** at the gas's own temperature per segment (Incropera A.4/A.6), 10 K hysteresis | - |
| 5 | The reverse-return ring outlet rises above the rings instead of crossing them | - |
| 6 | **Pinned rows out of the Krylov solve** (excluded air, Dirichlet walls); **AMG reused** for operators within 20 % per entry | standby 19 -> 10 s; 6 h transient 21 -> 12 s |
| 7 | **Thin shell blended** in series with the insulation (the widened shell ate up to 100 mm of insulation); roof and plate keep the steel's own mass | standby losses at 500 degC 8.3 -> 6.1 kW with the soil |
| 8 | **Soil under the pad** (3 m of moist sand, VDI 4640), box sides a symmetry | - |
| 9 | **Zehner-Bauer-Schlünder bed with radiation**, re-evaluated on the field (steady per sweep, transient per step, 2 % hysteresis) | steatite 0.30 (20 degC) / 0.57 (500 degC) W/(m K); standby 6.4 kW |
| 10 | Silica sand carried the bed's own k and density, diluted twice by the packing | now the quartz grain: bed 0.39 W/(m K) at 20 degC |
| 11 | GUI: layers in the bed height, soil depth, grain size, the bed's k at 20 and 500 degC, the network report behind an info mark | - |
| 12 | Docs: [17 References](docs/17_REFERENCES.md), [18 The solver, end to end](docs/18_SOLVER.md), README rewritten, citations in 01, 02, 12, 15, 16 | - |

Suite: 406 passed (395 without the GUI sweep), ruff clean.

## 2026-09-23 - the solver checked, the GUI cut to the plant

Every item was measured before it was changed; the numbers are in the commits and in
docs/12 §3 and §11.

### Solver defects fixed

| # | Defect | Evidence | Fix |
|---|--------|----------|-----|
| 1 | The octree rebuilt its face list at every request: a transient step cost seconds | 27 face-list builds in 4 steps, 2.2 s each on 25 000 leaves; 80 s for 4 steps | face list, sizes, centres and wall tables cached per tree state (`Octree.version`); matrix and Dirichlet elimination vectorised: 1.15 s for the same 4 steps, same answer |
| 2 | The gas loop was coupled explicitly (its marched power a fixed source for the step): a pipe cell with a large exchange overshoots | the bed driven below 100 K in the cycle runs of 2026-09-21 | implicit film `q = G (T_gas - T_wall)` per pipe cell, the loop inlet solved with the field so each step deposits exactly the external power, the gas re-marched at the end of the step |
| 3 | The steady and losses runs used the lumped tube model: a uniform source in the sand plus a fixed 500 W/(m2 K) film to a 60 degC "gas" on the pipes, so the pipes were a sink | default model: 967 W of the 5 kW left through the pipes, storage mean 110 degC instead of 150 | the steady state couples the loop (Picard + the balance held per sweep: 11 sweeps, exactly 5000 W into the bed) |
| 4 | Painting the network zeroed the source in the pipe cells without rescaling it | 4758 W of the 5 kW | the remaining source is rescaled |
| 5 | A transient left its loop sources and films on the mesh, and the next steady run solved them | a steady run after a 50 kW transient: sand at 4392 degC | every run sets the state it needs (the loop's film, or the lumped source repainted) |
| 6 | The AMG hierarchy was keyed on the matrix object; the symmetrised operator is a new object every call | the hierarchy rebuilt at every transient step | keyed on the matrix content |
| 7 | The thread setting only wrote environment variables, after BLAS was loaded | no effect | `threadpoolctl` |
| 8 | The losses analysis rewrote the ground condition of the mesh and left it changed | the steady run after a losses run had a convective ground | `h_ground` defaults to 0 (the mesh's own ground) |
| 9 | A pipe film on a box-face cell was dropped (a legacy of the lumped tubes) | risers reaching the box lost their end cells | the film is kept; pinned cells drop it on their own |
| 10 | "Load a saved HDF5 state" as an initial condition read the `.h5` file with `np.load` | could not work | replaced by "Current field" (a state loaded in Save / Load, or the last run) |
| 11 | The transient power default was 10 000 kW (value 10 000 in the kW unit) | – | 5 kW, the circuit's power |
| 12 | The pipe cells were painted as solid tube material: a 94 mm cell of steel for a 50 mm pipe with a 2 mm wall | default network: 1.6 m3 of "pipe" steel for 0.05 m3 of real wall; a 24 h, 50 kW charge raised the sand by 18 K instead of 27 | the cells keep the bed's properties and carry the pipe's exchange |
| 13 | The exchanger's delivered energy was `max(-net loop power, 0)`: zero whenever the resistors ran at the same time, and the explicit march estimate in the return-temperature mode | a 24 h discharge "delivered" 1344 kWh from 1137 kWh stored | the exchanger delivers its set power; in the return-temperature mode, the net exchange of the solved field |
| 14 | The coupled steady state drifted from the resistors' power (60 sweeps, 2069 W of 5000 into the bed) | default model | the loop balance is held in every sweep: 11 sweeps, 5000.0 W |

### Third pass (GUI layout, resolution, speed)

* **GUI**: six tabs, one per thing - Vessel (shape and layer materials), Plant (gas
  circuit, pipes), Site (ambient, ground, wind - it was "Conditions" under Materials),
  Mesh, Analysis (Type, Initial, Charge, Discharge, Save/Load), Solver.  Every page scrolls;
  the window opens inside the screen's free area; explanations are tooltips behind an ⓘ
  instead of paragraphs; the controls a layout does not read are disabled; Help is a menu.
* **Removed as redundant**: Lx, Ly, Lz, Centre X/Y (the box is the vessel plus 0.3 m of
  excluded air, the vessel centred in it), the method/preconditioner, under-relaxation and
  initial-density knobs, the 3-8 W/cm2 judgement of a gas-heated tube wall.
* **Accuracy in kelvin**: 0.1 K for the coupled iterations and the standby, 1e-6 linear.
* **Resolution where it matters**: the finest leaf is the finest region's target (52 mm
  around the pipes) and the budget caps the leaves; targets snap to the largest leaf
  within 1.6x.  Default: 214 089 leaves.
* **Speed**: vectorised face list (identical to the loop, pinned), operator cached per
  property state, AMG V(1,1) Gauss-Seidel cycle, the balance solve done once, the loop
  march vectorised (the area property was recomputed per cell), rasterised pipes cached,
  vectorised octree balance, field mapping and rasteriser.  Default model: build 12 s,
  standby 25 s, transient ~0.5 s per step - at 2.4x the leaves of the previous pass.
* **Found on the way**: the mesh plan passed the packing fraction as the *porosity* (its
  bed was 63 % air, 0.15 W/(m K) instead of 0.52; the simulation itself always used 0.52).

### Second pass (the owner's review of the running app)

| # | Found | Evidence | Fix |
|---|-------|----------|-----|
| 15 | The steady analysis at a fixed power is not a state a storage reaches at its charging power | a steady run at 5 kW sat at 458 degC, at the 200 kW of a real plant it would be thousands | "Steady standby": hold the bed at T, get the holding power (= the losses); the constant power profile is the circuit's rated power |
| 16 | The ring layout packed a set number of rings at the pitch: 3 rings inside the central 0.4 m of a 2 m vessel | the hot square in the centre of the owner's slice | a ring count spreads the rings over the radius, risers spaced like the rings |
| 17 | The network was centred on the box, not on the cylinder | correct only while the centre is the box centre | `build_pipe_network(center=cylinder centre)` everywhere |
| 18 | The mesh refined the whole bundle as one box: every leaf of the bed at the finest size, no octree step in the sand | 143 396 leaves, 39 s | one column per riser |
| 19 | The insulation ring and the shell were covered by four axis-aligned boxes: air refined at their corners, the ring at 45 degrees not at all | 36 000 excluded-air leaves at the finest size | radial bands (discs and annuli about the vessel axis) |
| 20 | The plan asked 16 mm through the whole insulation with the box-face film `h_lateral`, a film the insulation never faces | the insulation always at the floor | the surface rule only where the film sits (the casing, with the correlations' film) |
| 21 | The painter measured the shell's cell at (cx + r, cy + r), out in the air | on a tree: a 375 mm shell painted over the insulation | the cell is measured in the shell |
| 22 | Targets were rounded down to the next leaf size | a 200 mm request on 101.6 / 203 mm leaves refined the whole bed to 101.6 | nearest leaf size in log scale |
| 23 | The 2:1 balance swept every face of the tree after every round, in Python | 22 s of a 33 s build | balance only around the new leaves (the same tree, pinned by a test) |
| 24 | `h_char` recomputed a cube root of every volume per call; the rasteriser bisected Python objects | 3 s and 5 s of the build | cached; the tree's own position table |
| 25 | A discharge whose gas would have to come back at 28 K was accepted | the default network at 100 kW | the exchanger's return temperature is the floor: the run stops there with a note |
| 26 | The pipe cells were left out of the storage statistics once they became bed cells | a store whose pipe columns were all its leaves had "no sand" | `storage_mask` = sand + pipe cells |
| 27 | The geometry preview did not follow the panels | the owner's report | every edit redraws the preview (250 ms debounce), with the network as the panels describe it now |

Default model after the pass (Kankaanpää scale: 4 m, 5 m of bed, ~95 t): build and paint
3.3 s (90 630 leaves), standby 25 s (7.9 kW hold 500 degC), 48 h of charge at 200 kW in
35 s (gas in at 721 degC), a 100 kW discharge that runs 28 h and delivers 2800 kWh before
the bed around the pipes can no longer give it with gas returning at 60 degC.

### GUI: what was removed, and why

* The mesh is the octree only: the uniform and graded modes, the growth ratio, the
  min/max cell rails and "search before building" are gone (`Mesh3D` stays in `src/` as
  the reference of the equivalence tests).
* Gas circuit: the "Source from bottom/top" offsets and "Return to the resistors"
  (it silently ignored the power) are gone; a set return temperature is now an
  extraction mode of the exchanger.
* Pipes: the fixed "Gas h / Gas T" film and the graded-only "Junction refinement".
* Materials: "h top / h lateral" (films on the box faces, which only touch excluded air)
  and the second "Radiation" switch (never read); "Wind speed" feeds the outer film.
* Extraction: "Fluid flow rate" and "Tube h" (ignored whenever the loop runs).
* Cylinder: "Tubes/heaters phase" (read by nothing).
* Solver: the method and preconditioner selectors - the GUI solves with CG + AMG
  Ruge-Stuben, the fastest of the measured options, with the linear layer's own
  fallbacks.
* Added: the gas loop in the energy balance (inlet/outlet, bed power, NTU, pressure
  drop, fan), the VTK export button, the outer film in the log.

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

---

## Documentation catch-up (seventh pass, 2026-09-20)

The documentation had fallen several passes behind the code: it still described the
GPU backends, a 4-tab GUI with a colormap selector, the rod-heater path, 147/169 test
counts and a "planned" redesign whose first four pieces had already landed.  This pass
rewrites the documents against the code that exists, and every number quoted in them is
now a **measurement with the command that produced it**.

### Rewritten for the current code

| file | what it now says |
|---|---|
| `README.md` | the real project structure (`core/{octree,pipe_network,pipes,environment}.py`, `solver/{fluid,octree_solver}.py`, `analysis/{cycle,convergence,mesh_plan}.py`, `scripts/figures.py`), the document table 00-15, a **figures section** with the seven files that exist in `docs/figures/` and the function that draws each, the real GUI structure, the real default solver (BiCGSTAB + Jacobi in the GUI, direct LU in `LinearConfig`), no GPU recommendation, and the test count with its command and date |
| `docs/00_INDEX.md` | refreshed descriptions (10, 13, 14, 15) and an explicit statement that every quoted count is a measurement |
| `docs/01_THEORY.md` | the internal film and the contact resistance, and the NTU/effectiveness + fan-work definitions of the gas loop (`solver/fluid.py`) |
| `docs/02_FDM_DISCRETIZATION.md` | per-volume coefficients with the cached `GridIndex.face_factors`, contact resistance, the excluded-air film at active/excluded interfaces, GPU claims removed, graded/octree validation rows |
| `docs/03_GEOMETRY.md` | the true heater pattern → bank mapping (the checkerboard is the grid bank, the spiral is a rectangular bank, there is no `custom` heater pattern), the reality that `apply_to_mesh` never fills `mesh.excluded`, and a section on the buried pipe networks |
| `docs/04_GUI_DESIGN.md` | the real layout (four left tabs with their sub-tabs), the getter tables including the mesh/pipes getters, the run cycle with the `automesh` job, `backend()` removed, the 3D view controls and its fixed `coolwarm` scale |
| `docs/05_ARCHITECTURE.md` | the two workstreams beside the main path (gas-side redesign, adaptive mesh) with what is and is not wired, the new contracts (excluded film, contact resistance, geometric wetted area, conservative face list), the `automesh` job kind, and the removed code list extended (rod-heater path, colormap selector, `accelerators.py`) |
| `docs/06_GUI_CONFIGURATION.md` | every control with the default **read from the live panels off-screen** (heater power 5 kW, six heater patterns and the bank each maps to, the *Pipes* tab, the automatic-mesh tolerances, the 3D view sliders) and the truth about the exports (the time-series CSV has a button, the VTK/CSV field export does not) |
| `docs/07_CODE_STRUCTURE.md` | the module map taken from the source, marking which names the package `__init__` files re-export, and the test table with the collected count per file |
| `docs/08_ANALYSIS_WORKFLOWS.md` | the automatic mesh search, the cycle with its energy identity and per-step stops, and the gas loop as the heat path, plus the new reported quantities (`environment_flux`, circulation, delivered/unrecovered) |
| `docs/09_TESTING.md` | the real collected count per file with the command and the date, the GPU backend and GUI-colormap claims removed, the scene combination count corrected (4 fields x 4 cuts x 3 fractions = 48) |
| `docs/10_MESH_AND_HEATERS.md` | status per part (1-3 implemented, 4 open), the hairpin bank is **no longer the design path** (the resistors belong in the gas circuit), one element type instead of the promised selector, and no unkept promises |
| `docs/11_HANDOFF.md` | the state of the work: current counts, the passes that landed (fluid loop, networks, environment, octree, cycle, octree solver, figures), the real work-in-progress list and the quick verification commands |
| `docs/13_REDESIGN.md` | the duplicated patent subsection merged, and §4/§5/§6 rewritten with their true status plus a new **§8 status of the redesign** with the file and test behind every row |

### Verification of this pass

* `python -m pytest tests/ --collect-only -q` → **311 collected**, 300 without
  `tests/test_gui_sweep.py` (2026-09-20).  The same command reported 270/259 and later
  295/284 in the same session, which is why the documents record the command and the
  date next to every count instead of a bare number;
* `python -m pytest tests/ -q --ignore=tests/test_gui_sweep.py` → **299 passed, 1
  failed**, and the failure is a wall-clock budget, not physics:
  `tests/test_octree.py::test_a_tree_of_twenty_thousand_leaves_builds_and_lists_its_faces_in_under_a_second`
  asserts under a second for a 32768-leaf tree and measured 1.30-1.55 s with the machine
  idle (2.06 s under load).  It is reported to the owner of that module; the documentation
  pass itself changes no code, so nothing here can have caused or fixed it;
* `python -m ruff check src tests gui --select F,E9,B,SIM,UP` → **all checks passed**
  (the six findings in `src/solver/octree_solver.py` / `tests/test_octree_solver.py`
  that appeared while that module was landing were fixed by its author);
* every path written into these documents was checked to exist in the tree (the only
  exceptions are the three files listed as *removed* in `docs/05`, which is the point of
  that section);
* the GUI defaults were read by instantiating `GeometryPanel`, `MaterialsPanel`,
  `AnalysisPanel`, `SolverPanel`, `VizView` and `ResultsPanel` off-screen
  (`QT_QPA_PLATFORM=offscreen`, `THERMAL_DISABLE_3D=1`), not copied from the previous
  tables.

### Still open (documented, not fixed)

* `BatteryGeometry` still paints the air box: `mesh.excluded`/`mesh.h_out` are never
  filled from a run, so the outside film is a tested capability, not the default model;
* the octree is not the mesh of the main solver: `src/solver/octree_solver.py` exists and
  is tested, `SteadyStateSolver` does not call it and there is no transient driver;
* the graded-vs-uniform cost/accuracy measurements of `docs/10` part 4, and the possible
  sub-grid heater model;
* the automatic mesh search still cannot reach the default 2 K tolerance inside the
  default 400 000-cell budget on the default geometry (it says so);
* docs/12, 14 and 15 are owned by other agents in the same cycle and are not touched here;
* one test fails on this machine and is **not** a physics failure:
  `tests/test_octree.py::test_a_tree_of_twenty_thousand_leaves_builds_and_lists_its_faces_in_under_a_second`
  (a one-second wall-clock budget for a 32768-leaf tree; measured 1.55 s idle).  The
  budget or the test needs the owner's decision; the rest of its file passes.
