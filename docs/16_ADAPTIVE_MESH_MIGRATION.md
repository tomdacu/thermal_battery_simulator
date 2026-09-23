# 16 - The adaptive mesh: the interface, the equivalence, and the way off `Mesh3D`

Status: **steps 1-7 are done, step 8's criterion is evaluated and `Mesh3D` stays** (section
7). This document is the census of what the consumers actually read from a mesh, the
contract that grew out of it (`src/core/mesh_api.py`, `src/core/adaptive_mesh.py`) and the
ordered plan that moves the application onto a tree. Line numbers are those of the census,
taken on the commit before the port starts (`grep -n` over `src/`, `gui/`, `tests/`,
`scripts/`).

Who reads what, in one line: the solver assembles `k A / (d V)` from a face list and
per-cell fields, the balance reads the same fields back, the geometry/heaters/pipes
*paint* the per-cell fields from 3-D boolean masks, the GUI and the IO render and store
the 3-D arrays. Only the first two are physics; the rest is either structural (index
arithmetic) or presentation.

---

## 1. The census

Classification: **essential** - the solver and the balance need it, so a tree must provide
it with the same meaning; **structural** - it assumes 3-D indices, so a consumer that uses
it has to be rewritten for an adaptive mesh; **convenience** - only the tests, the GUI or
a report use it, so an adapter or a summary can carry it; **delete** - nothing needs it
once the port is done.

### 1.1 Per-cell state (essential)

| member | who reads it | what it means for a tree |
|---|---|---|
| `T` | `src/solver/steady.py:82`, `:100`; `src/solver/matrix.py:104` (radiation); `src/solver/transient.py:102`, `:246`; `src/analysis/balance.py:87`, `:141-142`, `:150-153`; `src/analysis/cycle.py:236-239`, `:249`, `:390`; `src/analysis/fluxes.py:112`; `src/core/geometry.py:438`; `src/core/profiles.py:156-178`; `src/io/state.py:181-195`; `src/viz/scene.py:232`, `:282`; `gui/views/results_panel.py:50-57` | one temperature per leaf [K]; the solver writes it back |
| `k` | `src/solver/matrix.py:57`, `:63-75`, `:103`; `src/analysis/fluxes.py:63-64`, `:84`; `src/core/geometry.py:542`; `src/core/pipe_network.py:1224`; `src/viz/scene.py:238` | one conductivity per leaf [W/(m K)] |
| `rho`, `cp` | `src/solver/matrix.py` (`build_transient_operators`, `m_diag = rho*cp`); `src/analysis/balance.py:153`; `src/analysis/cycle.py:257`; `src/core/geometry.py:543-544`; `src/core/pipe_network.py:1224`; `tests/test_solver.py:169-170`, `:380` | density and specific heat per leaf: the transient mass matrix and the stored energy |
| `Q_source`, `Q_sink` | `src/solver/matrix.py:137`; `src/analysis/balance.py:112-113`; `src/solver/transient.py:144-146`, `:158`, `:165`, `:183`; `src/core/geometry.py:621`, `:636-637`, `:658-659`, `:690`; `src/viz/scene.py:236`, `:284` | volumetric source/sink per leaf [W/m^3]. The split keeps `Q_source >= 0 >= Q_sink`, which `Mesh3D.validate` enforces |
| `source_mask` | `src/solver/transient.py:117`, `:160-161`; `src/core/geometry.py:622`, `:626`, `:636`, `:658`, `:691`; `src/core/heaters.py:269`, `:279`; `src/core/pipe_network.py:1226`; `src/io/state.py:83` | the cells a power profile is allowed to drive; pure bookkeeping, and the same meaning on a tree |
| `material_id` | `src/solver/matrix.py:59`, `:74` (contact resistance between materials); `src/analysis/balance.py:119-122`, `:150`; `src/solver/transient.py:121`; `src/core/geometry.py:432`, `:437`, `:541`; `src/core/pipe_network.py:1223`; `src/viz/scene.py:234`, `:283`; `gui/views/results_panel.py:49` | `MaterialID` per leaf |
| `V` | `src/analysis/balance.py:112-113`, `:153`; `src/analysis/cycle.py:257`; `src/solver/transient.py:124-125`, `:161`, `:183`; `src/core/geometry.py:637`, `:659`; `src/core/grid.py:119`; `src/core/heaters.py`; `tests/test_heaters.py:39` | cell volume [m^3]: the weight of every mean, the factor back from per-volume coefficients, and the scale of the graded linear solve (`src/solver/steady.py:58`, `src/solver/transient.py:89`) |
| `bc_h`, `bc_T_inf` | `src/solver/matrix.py:159-161`; `src/analysis/fluxes.py:139-142`; `src/solver/transient.py:112-113`, `:147`, `:169`, `:177`; `src/core/pipe_network.py:1227-1230`; `src/core/mesh.py:set_internal_convection` | internal film [W/(m^2 K)] and fluid temperature [K] |
| `boundary_type` | `src/core/grid.py:113` (`interior_tube`); `src/core/mesh.py:set_internal_convection`; `src/core/geometry.py` (tubes); `tests/test_core.py:163`; `tests/test_pipes.py:617` | which cells exchange with a fluid |
| `excluded` | `src/solver/matrix.py:58`, `:142-168`; `src/analysis/fluxes.py:59`, `:108`, `:120`; `src/core/geometry.py:433`; `tests/test_environment.py:33`, `:81` | the cells outside the thermal problem |
| `h_out`, `t_ambient` | `src/solver/matrix.py:143-155`; `src/analysis/fluxes.py:106`, `:120-121`; `src/core/geometry.py:443-444` | the outside film and the ambient that drives it |
| `h_contact` | `src/solver/matrix.py:66-75` | the contact conductance between materials |
| `face_bc` | `src/solver/matrix.py:93-120`; `src/analysis/fluxes.py:46`; `src/core/grid.py:47`; `src/core/geometry.py:531-535`; `src/io/state.py:124`; `tests/test_core.py:38-53` | the six box faces: kind, `h`, `value`, `emissivity` |

`h_char` is `V**(1/3)`: derived, read at `src/solver/matrix.py:159`, `src/solver/transient.py:125`
and `src/analysis/fluxes.py:145`. It is *not* in the protocol - a tree provides `V`.

### 1.2 The face list (essential)

There is no `faces()` on `Mesh3D`: the structured consumer builds one where it needs it,
through `GridIndex.from_mesh` (`src/core/grid.py:85-119`), which gathers

* `neighbours` (six flat neighbour indices), `on_face` (six masks), `interior_tube`
  (`boundary_type == CONVECTION` and not on a box face), `sizes` (per-axis cell sizes),
  `areas` (per-axis face areas), `volume` (flattened `V`),

and `src/solver/matrix.py:face_coefficients` turns that into one conductance per direction:
`k_face A / (d_centers V)` with the harmonic mean, the contact resistance where two
materials meet, zeros at the box boundary and zeros toward an excluded neighbour.
`Octree.faces()` already provides exactly the conservative version of the same list
(`(cell_i, cell_j, axis, area, d_centers)`, one entry per interface).

### 1.3 Structural (assumes 3-D indices)

| member | who reads it |
|---|---|
| `Nx`, `Ny`, `Nz`, `N_total` | `src/solver/matrix.py:116`; `src/solver/steady.py:66`; `src/solver/transient.py`; `src/core/grid.py:92`; `src/io/state.py:77`; `src/viz/scene.py:252`; `gui/main_window.py:195`; `gui/views/results_panel.py:66`; `tests/test_core.py:58-68` |
| `unflatten_field` | `src/solver/steady.py:82`, `:100`; `src/solver/transient.py:246` |
| `ijk_to_linear`, `linear_to_ijk` | `tests/test_core.py:66`; `tests/test_octree_solver.py:87` |
| `find_cell` | `src/core/heaters.py:222-223`, `:245`, `:327`; `src/core/pipes.py:_flat_cell` (`:69-74`, from `edges_x/y/z`); `src/core/geometry.py:614`; `tests/test_octree_solver.py:87` |
| `dx`, `dy`, `dz`, `edges_x`, `edges_y`, `edges_z`, `axis_size`, `size_x`, `size_y`, `size_z` | `src/analysis/fluxes.py:91-96`, `:163-165`; `src/core/grid.py:114`; `src/core/geometry.py:456`, `:508`, `:569-570`, `:582-595`, `:710`; `src/core/heaters.py:231-233`, `:328`; `src/core/pipes.py:69-74`; `src/io/state.py:80`, `:120-122` |
| `Ax`, `Ay`, `Az` | `src/core/grid.py:115`; `src/analysis/fluxes.py:91`, `:164` |
| `uniform` | `src/analysis/fluxes.py:142`; `src/solver/steady.py:58`; `src/solver/transient.py:89`; `src/viz/scene.py:251`; `tests/test_graded_mesh.py:53`; `tests/test_gui_sweep.py:276-281` |

A tree answers these three ways: `N_total` becomes `n_cells`; `find_cell`/`cell_size_at`
become a point location on the leaves (today `Octree._locate` is private - the port has to
make it public, see step 6); `dx`/`axis_size`/`Ax` become the leaf's own edge and face area,
which the face list and `cell_sizes()` already carry.

### 1.4 Convenience (tests, GUI, reports)

* `grid_summary` - `src/analysis/convergence.py:182` (the cell count of a level),
  `gui/views/geometry_panel.py`, `tests/test_convergence.py:136`.
* `size_label` - `gui/main_window.py:196`, `gui/views/results_panel.py:67`.
* `get_info` - `gui/views/geometry_panel.py`, `gui/controller.py`.
* `snapped` - `src/core/geometry.py:505`; `tests/test_core.py:60`.
* `validate` - `src/solver/steady.py:63`, `src/solver/transient.py:191`,
  `src/io/state.py:103`, `src/core/geometry.py:527`.
* `d`, `V_cell`, `A_cell` - `tests/test_solver.py:30-33`, `:117`, `:174`,
  `scripts/benchmark.py:64`; uniform grids only.

A tree answers them with `summary()`, `level_histogram()`, `sizes`, `V` and a validation
that mirrors `Mesh3D.validate`.

### 1.5 Delete

* the legacy uniform constructor: `Mesh3D(spacing=...)`, `_build_uniform`, `_step`,
  `snapped` - the graded/adaptive path supersedes it. Call sites:
  `gui/views/geometry_panel.py:508-513` (the `refined` checkbox), `tests/conftest.py:14`,
  `tests/test_solver.py:24`, `scripts/benchmark.py:40` and about thirty more test
  fixtures (they all construct a *uniform* box, which is `AdaptiveMesh.uniform`).
* `d`, `V_cell`, `A_cell`: only the uniform tests and the benchmark use them.
* the `Mesh3D.uniform` *flag*: needed during the transition, gone at the end - the tree's
  analogue is a property of the mesh, not a mode.

---

## 2. The protocol

`src/core/mesh_api.py` states the contract the **essential** rows above add up to, as a
`typing.Protocol` (`MeshAPI`) plus `MEMBERS` and `missing_members()`. It is deliberately
the intersection and not the union:

* **in**: `n_cells`, `T`, `k`, `rho`, `cp`, `V`, `material_id`, `Q_source`, `Q_sink`,
  `source_mask`, `boundary_type`, `bc_h`, `bc_T_inf`, `excluded`, `h_out`, `t_ambient`,
  `h_contact`, `face_bc`, and `faces()` returning `(cell_i, cell_j, axis, area, d_centers)`.
* **out**: the index arithmetic, the per-axis sizes, the uniform-grid shortcuts, the
  summary/report vocabulary and `h_char` (derived from `V`), each for the reason stated in
  the module docstring. A consumer that reads three members does not make the interface
  ten members wide.
* **the one rule that is not a field**: `is_interior_tube(boundary_type, on_box_face)`,
  the "a tube cell on a box face carries a face condition instead" rule that today lives
  in `GridIndex.from_mesh`.

`AdaptiveMesh` conforms (`missing_members(AdaptiveMesh.uniform(...)) == []`, pinned in
`tests/test_adaptive_mesh.py`). `Mesh3D` conforms in everything but the two names
`n_cells` (it says `N_total`) and `faces()` (it has `GridIndex` instead); the port renames
the first and replaces the second, and the protocol is what the ported consumers are
written against.

---

## 3. The adaptive mesh

`src/core/adaptive_mesh.py` is `Octree` + the physics vocabulary of section 1:

* **construction**: `AdaptiveMesh(tree, physical_size)`, `uniform(...)`,
  `from_bands(...)` (3-D `RefinementBand`s - a region with a target size - refined through
  the octree's own `refine` + 2:1 `balance`), `from_indicator(...)`, and `refine(...)`,
  which carries every per-leaf field to the children (zeroth order) so a solution-driven
  round does not have to repaint the model.
* **the fields** of the protocol, with the `Mesh3D` defaults (air, ambient, insulated
  sides, a ground face at `T_GROUND_DEFAULT`).
* **the films**: `set_convection_bc`, `set_fixed_temperature_bc`, `set_adiabatic`,
  `set_heat_flux_bc`, `set_internal_convection`, plus the environment (`excluded`, `h_out`,
  `t_ambient`) and the contact resistance (`h_contact`).
* **the assembly**: `matrix()` = `Octree.diffusion_matrix` + a correction on the same face
  pattern (contact resistance where two materials meet, no conduction into an excluded
  leaf); `assemble()` adds the film diagonal and the Neumann term and eliminates the
  Dirichlet rows with `src/solver/matrix.apply_dirichlet` - the structured elimination, so
  a fixed leaf is exact and the operator stays symmetric.
* **the solve**: `solve_steady(config)`, through `src.solver.linear.solve_linear` with the
  volume symmetrisation the structured solver uses on a graded mesh.
* **the balance**: `balance()` reports the source/sink power of the free leaves, the
  conduction leaving them into the fixed ones and the films (per box face, environment,
  fluid); the identity `p_source + p_sink == q_fixed + p_film` closes at the round-off of
  the per-leaf closure, which is reported too.
* **the estimator**: `indicator()` is `OctreeSteadySolver.indicator` - the flux jump the
  refinement cycle refines on - so a tree refined here refines as
  `refine_on_objective` would refine it.

### 3.1 Measured equivalence (the numbers `tests/test_adaptive_mesh.py` pins)

| case | mesh | cells | vs analytic | vs `Mesh3D`/`SteadyStateSolver` | balance |
|---|---|---|---|---|---|
| two-layer wall, interface on a face | uniform tree, 4x4x4 | 64 | 1.1e-13 K | 2.3e-13 K | closure 4.8e-14, residual 4.2e-13 W |
| two-layer wall, graded around the interface | 2 levels, hanging nodes | 288 | 3.4e-13 K | (same analytic, `Mesh3D` on the uniform partner) | closure 1.2e-12, interface drop 16.6667 K on a 0.125 m pair |
| slab, uniform source, film on one side | uniform tree | 64 | `q d^2 / 8k` = 7.8125 K, constant | 1.8e-12 K | closure 2.7e-15, film = source |
| slab, uniform source, film on refined side | 3 levels, film leaves 4x finer | 1184 | 0.4883 K at the film (= `q d^2/8k` with the fine edge) | (analytic) | closure 1.8e-15, film 1000.000000 W = source |
| excluded air box + environment film | uniform tree, half excluded | 64 | - | 1.7e-13 K | environment film 1198.142857 W in both, closure 1.8e-16, residual 1.6e-13 W |
| contact resistance between two materials | uniform tree | 64 | - | 2.8e-13 K | closure 4.2e-14 |
| internal (tube) film | uniform tree | 64 | - | < 1e-9 K | closure 6.1e-16 |

The analytic cases are exact where the scheme is exact: a sourceless 1-D profile is
reproduced on *any* mesh (that is why the graded wall is exact), and a uniform source in a
uniform mesh is reproduced up to the film's half-cell truncation, which the adaptive mesh
reproduces **exactly** (`q d^2/(8k)` with the leaf on the film - the same number the
structured solver produces, because it is the same formula).

### 3.2 What the adaptivity buys (the fifth test)

A Gaussian hot spot of 30 K (sigma = 0.08 m) in a 1 m box at 300 K, exact by construction
(`Q = -k grad^2 T`), refined three rounds on the tree's own flux-jump estimator:

| mesh | cells | max error vs analytic | time |
|---|---|---|---|
| adapted tree | **2360 leaves** (1088/888/384 at levels 1/2/3) | **1.311 K** | 1.2 s for the three solves |
| graded `Mesh3D` with the tree's own per-axis resolution, `growth=1.3` | **13 824 cells** (24x24x24) | **1.186 K** | 0.27 s for the solve |

That is **5.9x fewer cells for an error 1.11x larger**: equal accuracy, an order of
magnitude in the cell budget, and the test asserts the claim (`>= 3x fewer cells`, error
within 1.25x) rather than the measurement. The shape of the win is the documented one: the
graded Cartesian mesh pays the tensor product of three 1-D refinements, the tree pays the
*five* faces of the refined region.

---

## 4. What `octree_solver.py` does not have (the bypass list)

Each of these is a capability the adaptive mesh needs and the octree solver does not
provide; each is implemented in `src/core/adaptive_mesh.py` with the *structured* formula,
and each is a candidate for a later move into `src/solver/octree_solver.py`:

1. **Box-face conditions.** `OctreeSteadySolver` holds *leaves* fixed and its box walls
   carry no face at all, so a wall is adiabatic unless its leaves are pinned: there is no
   convection, no Neumann and no way to read the loss through a face back. The adaptive
   mesh adds the diagonal and RHS terms itself (`_films`, `assemble`) with
   `src/core/physics.half_cell_h`, the same half-cell film the structured assembly uses.
2. **The environment film.** `excluded` + `h_out` + `t_ambient` do not exist on the tree:
   implemented here (`_films`, the `"environment"` entry) with the structured rule
   (active side only, `h_out A / V`, plus the pin at `t_ambient`).
3. **The contact resistance.** `Octree.diffusion_matrix` has one harmonic-mean conductance
   per face and no way to pass a series resistance; a correction on the same face pattern
   is added here (`_face_table`, `matrix`).
4. **`Octree.solve` cannot take extra terms.** It builds its own operator from a volumetric
   source and a Dirichlet set, so it cannot express a film (a diagonal + RHS term). The
   solve therefore goes through `src.solver.linear.solve_linear` (the structured linear
   layer, with the volume symmetrisation) after the elimination of
   `src/solver/matrix.apply_dirichlet`.
5. **No public per-face conductance.** `OctreeSteadySolver._face_data` is private and has
   no contact variant, so the adaptive mesh keeps one conductance array of its own
   (`_face_table`); a test pins it to the solver's fluxes when nothing is corrected, so the
   two cannot drift apart silently.
6. **No point location.** `Octree._locate` is private and there is no `cell_size_at`, no
   `find_cell`, no `uniform` and no `on_face`: the wall masks and the "on a box face"
   reduction are derived in `adaptive_mesh` (`wall_indices`, `on_box_face`), and the
   painters of step 6 below need `Octree.locate` to be made public.
7. **No radiation.** `radiation_h` and the linearised Picard loop live in
   `src/solver/matrix.py` + `src/solver/steady.py`; the adaptive assembly has no radiative
   film yet (see section 6).
8. **A defect in a shared helper, worked around.** `src/solver/matrix.apply_dirichlet`
   caches `indptr/indices/data` before its row loop and inserts a missing diagonal with
   `a_csr[row, row] = 1.0`; that insertion *reallocates* the CSR arrays, so every fixed row
   processed after the first insertion is never zeroed and never gets its identity. On a
   structured mesh the fixed rows always have a stored diagonal and the branch is dead; the
   adaptive operator zeroes a whole face pair into an excluded leaf, which can leave a row
   with no stored entries, and the branch fires (measured: 15 of 16 fixed rows silently
   dropped, the linear solve then returns zeros with a `MatrixRankWarning`). The adaptive
   assembly keeps an explicit zero diagonal on those rows (it builds both the corrected
   operator and the film diagonal in one COO triplet list, because a sparse *sum* prunes
   explicit zeros), and the defect itself is reported rather than patched here - `matrix.py`
   is out of this change's scope.

---

## 5. The migration plan

Each step is self-contained: it ports a set of consumers, it names the test that proves the
equivalence, and it leaves the application working (a step never breaks a run that the
previous step made work).

**Step 0 - the foundation (this change).** `src/core/mesh_api.py`,
`src/core/adaptive_mesh.py`, `tests/test_adaptive_mesh.py`, this document. Nothing else
imports them yet.

**Step 1 - the geometry painter.**
Files: `src/core/geometry.py` (`BatteryGeometry.apply_to_mesh`, `_fill`,
`_paint_shell_and_insulation`, `_paint_storage`, `_paint_roof`, `_elements_mask`,
`apply_boundary_conditions`, `apply_environment`, `_widen`), `src/core/materials.py` (no
change), `tests/conftest.py` (the fixtures build uniform boxes: `AdaptiveMesh.uniform`).
Change: every mask built from `mesh.X/Y/Z` becomes a mask over `mesh.centres()`;
`mesh.size_x/size_y/size_z` becomes a leaf-edge query; `mesh.cell_size_at` becomes the
point location of step 6 (until then, the painter can take the size from the leaf it
already picked). The `_fill` helper writes `material_id`, `k`, `rho`, `cp` - four arrays
the protocol names.
Test: a new `tests/test_geometry_on_a_tree.py` that paints the *same* geometry twice - on an
`AdaptiveMesh.uniform` and on the matching uniform `Mesh3D` - and compares `material_id`,
`k`, `rho`, `cp` and `excluded` leaf by leaf (the same comparison `structured_partner` does
in `tests/test_adaptive_mesh.py`, which is the seed of it).
Done when: `apply_to_mesh` takes either mesh and `tests/test_core.py`'s geometry tests run
on both.

**Step 2 - the steady solver.**
Files: `src/solver/matrix.py` (its `face_coefficients` is replaced by
`AdaptiveMesh._face_table` + `matrix`), `src/solver/steady.py` (`SteadyStateSolver` becomes
a thin driver over `AdaptiveMesh.solve_steady`, keeping `SolverResult` and the Picard loop
for radiation), `src/solver/linear.py` (unchanged - already the shared layer).
Test: `tests/test_adaptive_mesh.py` (the five equivalence cases) plus a *battery-model*
steady test: `create_small_test_geometry()` painted on a tree and on `Mesh3D`, same
temperature field to the solver tolerance, same balance.
Done when: `SteadyStateSolver` runs on both meshes and `SolverResult` is unchanged for the
callers (`src/analysis/losses.py:96`, `src/analysis/cycle.py`, `gui/controller.py`).

**Step 3 - the balance and the flux report.**
Files: `src/analysis/balance.py` (`compute_balance`), `src/analysis/fluxes.py`
(`domain_face_flux`, `environment_flux`, `tube_flux`, `stored_energy`, `envelope_fluxes`),
`src/core/grid.py` (`GridIndex` is the structured face list: it is replaced by
`MeshAPI.faces()`).
Change: `q_domain` becomes the sum of the film rates the assembly applied (per box face),
`q_battery` the same restricted to the envelope materials, `environment_flux` the
environment entry of `AdaptiveMesh.balance()`.
Test: a battery model on a tree whose balance closes at the same level as on `Mesh3D`
(`tests/test_environment.py` and `tests/test_graded_mesh.py` ported), plus the *defect of
section 4.8*: the ported `environment_flux` must exclude the rows the elimination pinned,
which is what makes the film short by 0.77 W of 5 kW today.
Done when: `Balance` is filled from either mesh and `tests/test_analysis.py` passes on both.

**Step 4 - the refinement drivers.**
Files: `src/analysis/convergence.py` (`find_mesh`, `ConvergenceLevel`, `observe`),
`src/analysis/mesh_plan.py` (the automatic mesh search), `gui/views/geometry_panel.py`
(`grid_spec`, `build_mesh`).
Change: the search over `GridSpec` levels becomes a search over refinement rounds:
`AdaptiveMesh.from_bands` for the first level, then `refine` on the flux-jump indicator with
the same `ConvergenceTarget`; the panel's bands map 1:1 to `RefinementBand`s.
Test: the root-finding tests of `tests/test_convergence.py` run on the tree (`find_mesh`
with `build_mesh = AdaptiveMesh.from_bands`) and pick a mesh whose objective matches the
graded one within the tolerance.
Done when: the GUI's automatic mesh is a tree.

**Step 5 - the transient march.**
Files: `src/solver/transient.py` (`TransientSolver`, `_source_masks`, `_set_power`,
`_set_extraction`, `_set_fluid_loop`, `apply_initial_condition`),
`src/solver/results.py` (unchanged), `src/analysis/cycle.py` (the charging cycle).
Change: the mass matrix is `rho*cp` per leaf (the protocol has both), the operators come
from `AdaptiveMesh.matrix()` + the film diagonal, `unflatten_field` disappears (the tree's
field *is* the flat vector), `mesh.T = ...` becomes an assignment to the leaf vector.
Test: the 1-D backward-Euler slab of `tests/test_solver.py::test_transient_slab_matches_the_fourier_series`
(and the lumped-capacity exactness test) run on a tree, plus `tests/test_cycle.py`'s energy
ledger on a tree.
Done when: a transient run on a tree closes its ledger exactly as on `Mesh3D`.

**Step 6 - the painters that need a point.**
Files: `src/core/pipes.py` (`_flat_cell`, `rasterize_pipe`), `src/core/pipe_network.py`
(`paint`), `src/core/heaters.py` (`_segment_mask`, `rasterize`, `heater_span`,
`_tube_cells`), `src/core/geometry.py` (`cell_size_at`/`size_x/y/z` users).
Prerequisite, in `src/core/octree.py`: make the point location public
(`Octree.locate(x, y, z) -> Leaf | None`, from the existing `_locate`) and add
`Octree.cell_size_at` - about ten lines, and the only new capability this plan needs from
the octree itself.
Change: a mask over `(i, j, k)` becomes a mask over leaf centres (the segments/specs stay
the same), and the "cells the axis crosses" chains compare against the leaf edge instead of
`dx/dy/dz`.
Test: the painted tube network and the heater masks coincide on `AdaptiveMesh.uniform` and
on the matching `Mesh3D` (`tests/test_pipes.py::test_paint_marks_the_tube_cells`,
`tests/test_heaters.py`), including the power each pattern deposits.
Done when: heaters, tubes and pipes paint a tree.

**Step 7 - the GUI, the view and the IO.**
Files: `gui/views/geometry_panel.py` (the mesh factory and the summary), `gui/controller.py`
(`build(spec_level)`), `gui/main_window.py` (`build_mesh`, the log line), `gui/views/results_panel.py`
(the grid summary), `src/viz/scene.py` (`to_image_data`: `ImageData`/`RectilinearGrid` become
an `UnstructuredGrid` of the leaves), `src/io/state.py` (`save_h5`/`load_h5`: the edges
become the leaf list and the cell count), `scripts/benchmark.py` (the uniform sweep becomes
a refinement sweep).
Change: `mesh.grid_summary()`/`size_label()`/`Nx x Ny x Nz` become `summary()` /
`level_histogram()`; `mesh.N_total` becomes `mesh.n_cells`.
Test: `tests/test_gui.py` builds a mesh and runs a steady analysis through the controller
on a tree; `tests/test_scene.py` renders a tree; the save/load round trip of
`src/io/state.py` keeps a tree field exact (`tests/test_analysis.py`'s state tests).
Done when: the GUI starts, paints, solves, renders and saves with a tree.

**Step 8 - delete the structured mesh.**
The criterion for declaring `Mesh3D` removable:

1. no module outside `src/core/mesh.py`, `src/core/grid.py` and their tests mentions
   `Mesh3D`, `GridIndex`, `N_total`, `ijk_to_linear`, `unflatten_field`, `edges_x`,
   `edges_y`, `edges_z`, `find_cell`, `axis_size` or `size_x/y/z` (a `grep` gate, exactly
   like the census above);
2. every consumer that receives a mesh passes `missing_members(mesh) == []` - i.e. the
   protocol, not the class, is what it was ported against;
3. the equivalence suite of `tests/test_adaptive_mesh.py` (five cases), the ported battery
   tests of steps 1-3, the transient tests of step 5 and the painted-mask tests of step 6
   all pass on the tree, and the GUI run of step 7 reproduces the structured numbers within
   the solver tolerance;
4. the limits `src/solver/octree_solver.py` documents are answered for the models the
   application actually runs: the field is still second order where the cells change size
   (measured in `tests/test_adaptive_mesh.py`), the refinement stops on the objective
   (`refine_on_objective`) and on the cell budget (`ConvergenceTarget.max_cells`), a
   feature the estimator cannot see (a source or an interface *inside* a leaf) is caught by
   the painters of steps 1 and 6 placing leaf faces on it, and the Python face list is not
   the bottleneck for what the GUI builds (about 10^5 faces/s: 1.5 s for 32 768 leaves,
   the cost measured in `src/solver/octree_solver.py`).
5. `src/core/refinement.py` is either ported to the octree's bands (the panel keeps its
   band editor) or kept only for the GUI's band *request*, with `GridSpec.edges` no longer
   feeding a `Mesh3D`.

Until all five hold, `Mesh3D` stays: it is the reference solution for every equivalence
test above, and deleting it early would delete the only oracle the port has.

---

## 6. Limits of this change (what is *not* here)

* **Radiation.** `FaceBC.emissivity` and `radiation_h` are in the structured path only; the
  adaptive assembly applies the plain film. A radiative film is a Picard sweep over the
  linearised coefficient, and it belongs with the port of `SteadyStateSolver` (step 2).
* **The GUI and the view.** Nothing in `gui/` and `src/viz/` is wired to a tree yet: they
  are step 7 (done - section 7 below, and `gui/`, `src/viz/scene.py` and `src/io/state.py`
  take either mesh).
* **`h_char` and the uniform shortcuts** are deliberately absent from the protocol; a
  consumer that needs `V**(1/3)` computes it, and the uniform-only shortcuts (`d`, `V_cell`,
  `A_cell`) are marked for deletion in section 1.5.
* **The cost of the tree is Python.** The face list is a Python loop (about 10^5 faces/s):
  the adaptive mesh is not faster per cell than the structured one, it is cheaper in
  *cells* (5.9x on the hot-spot case). A model that needs 10^6 leaves needs the block
  refinement of Afivo, not a bigger Python loop.
* **The octree remains the only adaptive structure**: this change adds no new mesh type,
  and `Octree` itself was not modified.

---

## 7. Where the migration stands: step 8, evaluated

Steps 1-7 are done: the Mesh tab builds the tree, the window paints it, the controller
solves it, the view renders it and the state file stores it.  Step 8 is the criterion
above, evaluated here point by point on that tree.  **`Mesh3D` stays**: points 1 and 5 do
not hold, and both of them are the structured road itself, which is still selectable and
is the oracle every equivalence test below is written against.

| # | point | holds | evidence |
|---|---|---|---|
| 1 | no module outside `src/core/mesh.py`, `src/core/grid.py` and their tests names the structured vocabulary | **no** | 26 files under `src/`, `gui/`, `scripts/` still do.  Step 7 removed two of them (`gui/views/results_panel.py`, `scripts/benchmark.py`); the rest are the structured road and its branches (see the list below). |
| 2 | every consumer that receives a mesh passes `missing_members(mesh) == []` | **partly** | `missing_members(AdaptiveMesh.uniform(16, 0.5, 0)) == []`, and every ported consumer runs on a tree (the `*_on_tree` suites, `tests/test_scene.py`, `tests/test_analysis.py`, `tests/test_gui.py`: 117 tree tests, exit 0).  `Mesh3D` itself answers `['n_cells', 'faces']`, so *as written* the point holds only once the class is gone. |
| 3 | the equivalence suites pass and the GUI run reproduces the structured numbers | **yes** | 394 tests, exit 0.  The GUI's own tree (30 920 leaves, the Mesh tab's smallest budget) against the structured road at the same 0.1875 m cell: `dT_mean = 2.6e-7 K` (5.6e-10 relative), `dT_max = 5.8e-7 K`, `dQ = 6.0e-5 W` (1.5e-8 relative) - the solver tolerance is 1e-8. |
| 4 | the octree's documented limits are answered | **yes** | Second order across a size jump: `test_the_graded_film_uses_the_local_leaf_size`, `test_the_adapted_tree_beats_a_graded_mesh_on_cells` (both in `tests/test_adaptive_mesh.py`).  The rounds stop on the objective (`AdaptiveMesh.refine_round` is `refine_on_objective`'s rule with a budget check: `test_the_refinement_leaves_a_flat_field_alone`) and on the budget (`test_the_cycle_stops_at_the_cell_budget`, `test_the_cell_budget_stops_the_tree_and_says_so`).  Features a leaf can hide are caught by the painters (`test_the_network_paints_a_refined_tree_and_keeps_its_area`, `test_the_bank_paints_a_refined_tree_and_deposits_its_power`, `test_the_tubes_land_on_the_same_cells_on_a_tree`).  The face list runs at 2.5e5 faces/s on the GUI's tree (90 600 faces in 0.36 s) against the ~1e5/s measured for `octree_solver.py`. |
| 5 | `refinement.py` ported to the bands, or `GridSpec.edges` no longer feeding a `Mesh3D` | **no** | `GridSpec.edges` still feeds one: `gui/views/geometry_panel.py` (`Mesh3D(..., grid=self.grid_spec())`), `gui/controller.py` (`Mesh3D(..., grid=spec_level)`), `src/core/mesh.py` (`_build_graded`), and `gui/views/geometry_panel.py` reads it for the live summary.  The tree's bands live in `AdaptiveMesh`/`RefinementBand`, with `src/analysis/mesh_plan.refinement_bands` as the adapter: `src/core/refinement.py` is neither ported to them nor retired. |

Who still uses `Mesh3D`, and why:

* **builds one**: `gui/views/geometry_panel.py` (the Mesh tab's structured mode),
  `gui/controller.py` (a `GridSpec` level of the automatic search), `scripts/figures.py`
  (a figure of the graded road).
* **dispatches on it**: `src/solver/matrix.py`, `src/solver/steady.py`,
  `src/solver/transient.py`, `src/analysis/fluxes.py`, `src/analysis/balance.py` - the
  structured assembly, solve and flux report, selected by `isinstance`/`hasattr`, with
  `src/core/grid.GridIndex` as their face list.
* **reads a structured member in a branch**: `src/io/state.py` (`edges_*`: the state file
  of a grid), `src/viz/scene.py` (`uniform`, `dx`, `edges_*`: the `ImageData` /
  `RectilinearGrid` road), `src/core/geometry.py`, `src/core/heaters.py`,
  `src/core/pipes.py` (`find_cell`, `axis_size`, `size_x/y/z` in the painters),
  `src/analysis/convergence.py` (`getattr(mesh, "grid_summary", None)`).
* **names it without reading a member**: the annotations of `src/analysis/cycle.py`,
  `src/analysis/losses.py`, `src/solver/fluid.py`, `src/core/pipe_network.py`,
  `gui/main_window.py`; the prose of `src/core/adaptive_mesh.py`, `src/core/octree.py`,
  `src/core/mesh_api.py`, `src/solver/octree_solver.py`; the re-exports of
  `src/__init__.py`, `src/core/__init__.py`, `src/solver/__init__.py`.

Deleting the class therefore means deleting the Mesh tab's structured mode, the graded
`GridSpec` search level and the structured branches of those modules - a change of its own,
with its own equivalence question, and not one this step takes.

## 8. The tree of boxes (2026-09-23)

`src/core/box_tree.py` replaces the cubic octree on the GUI's path: a leaf has a plan
level and a height level, splits into four, two or eight, and is kept 2:1 per direction
[BWG11].  `AdaptiveMesh` runs on either tree - its per-axis `extent` feeds the box-face
films, the Neumann faces, the contact resistance and the flux reports, and the face
table of a box tree carries the overlap areas in metres - so every consumer of this
document works unchanged.  `AdaptivePlan` carries `n_z` and `dz` for it and
`AdaptiveMesh.from_plan` builds either.  The octree and `Mesh3D` stay as the references
of the equivalence tests (`tests/test_box_tree.py` pins a uniform box tree against the
uniform octree to 1e-8 K).  Details in [18](18_SOLVER.md) §1.
