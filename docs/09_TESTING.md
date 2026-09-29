# 9. Testing

```bash
python -m pytest tests/ -q --ignore=tests/test_gui_sweep.py   # 300 tests, the fast suite
python -m pytest tests/ -q                                    # 311 tests incl. the GUI sweep
python -m pytest tests/ --collect-only -q                     # what is collected, per file
python -m ruff check src tests gui --select F,E9,B,SIM,UP      # lint
python scripts/benchmark.py --max-cells 60000                 # timings per mesh size
```

**The counts above are a measurement, not a promise**: 300 / 311 are what
`python -m pytest tests/ --collect-only -q` reported on this working tree on
**2026-09-20**, with `tests/test_gui_sweep.py` contributing 11.  Earlier readings in the
same session said 270 / 259 and then 295 / 284 as the octree solver, the pipe-network
extension and the cycle completion landed.  Test files and counts move while work is in
flight, so re-run the command instead of trusting these numbers.

| file | collected cases | covers |
|---|---|---|
| `tests/test_pipes.py` | 72 | buried pipe networks: layout rules, header plumbing, wetted area, splits, paint, hydraulics, the loop built from the network |
| `tests/test_scene.py` | 60 | the rendering path off-screen: every field x cut x fraction, colormaps, exports |
| `tests/test_core.py` | 26 | units contract, mesh, materials, geometry, profiles |
| `tests/test_fluid.py` | 23 | the gas loop: effectiveness, conservation, pressure, circulation loss |
| `tests/test_solver.py` | 22 | assembly exactness, analytic regressions, linear backends, transient |
| `tests/test_octree.py` | 14 | 2:1 balance, conservative faces, an octree Laplacian against analytic answers |
| `tests/test_octree_solver.py` | 12 | the octree wired to the physics: equivalence with the structured solver, conservation, refinement |
| `tests/test_analysis.py` | 12 | flux closure, balance, losses iteration, HDF5 round trip |
| `tests/test_gui_sweep.py` | 11 | systematic control sweep (head-less); not part of the fast suite |
| `tests/test_cycle.py` | 11 | charge and discharge through the gas loop, end to end (incl. the per-step stop) |
| `tests/test_refinement.py` | 10 | graded-grid generation: exact endpoints, bounded growth, budget |
| `tests/test_heaters.py` | 9 | hairpin bank: rasterisation, power deposit, validations |
| `tests/test_environment.py` | 9 | the outside film, the excluded air mask, the contact resistance |
| `tests/test_convergence.py` | 9 | the automatic mesh search and its GCI report |
| `tests/test_graded_mesh.py` | 6 | the graded grid inside the solver: exactness, mesh independence |
| `tests/test_gui.py` | 5 | head-less GUI smoke test (skipped without PySide6) |

One case is a **wall-clock budget** rather than a physics statement:
`tests/test_octree.py::test_a_tree_of_twenty_thousand_leaves_builds_and_lists_its_faces_in_under_a_second`
asserts under a second for a 32768-leaf tree and fails on this workstation (1.30-1.55 s
with the machine idle, 2.06 s under load) - it is the only failing case of the 311 on
2026-09-20, and it is a machine-speed assertion, not a correctness one.

**Update 2026-09-23** (`--collect-only`): **416** cases, **405** without the GUI sweep.
New files of the anisotropic mesh and the plant physics:

| file | collected cases | covers |
|---|---|---|
| `tests/test_box_tree.py` | 5 | the tree of boxes: octree faces on a uniform tree, the 2:1 rule per direction and faces that tile every leaf after refinement, the same field as the octree, a vertical profile exact on leaves graded in plan, the balance on an anisotropic tree |
| `tests/test_well_model.py` | 4 | one tube in a square of sand against the shape factor (Incropera Table 4.1) on two meshes with the well model, the mesh dependence without it, no correction in a cell smaller than the tube, the whole gas circuit (mass at every node, the bed taking the external power, the headers exchanging) |
| `tests/test_thin_layers_and_soil.py` | 2 | a thin shell blended in series with the insulation, the steel's own mass; the soil layer and its boundary conditions |
| `tests/test_hydraulics.py` | 10 | the network's hydraulics and the header engine: Darcy-Weisbach, parallel pipes, the draught at low and high flow, balancing orifices to 1e-8, the orifice plate, Voronoi targets, the engine (manifold chosen, velocity met, headers lifted, targets delivered, the loop following the hydraulics) |
| `tests/test_packed_bed.py` | 5 | the Zehner-Bauer-Schlünder bed: the gas limit, the bounds without radiation, growth with temperature and grain, dry sand in the measured range, a steady bed at the conductivity of its temperature |

**Update 2026-09-29** (`--collect-only`): **412** cases, **401** without the GUI sweep
(audited later the same day: the cases that could not fail, and the asserts that restated
the production formula, are gone - see the changelog).
The licence release changed the GUI binding (PyQt6 -> PySide6) without touching the
physics: the sweep gained one case - *no slot raised an exception*, which watches
`sys.excepthook` because PySide6 prints a failing handler and carries on where PyQt6
aborted the process - and the rest of the suite is unchanged.

## 1. Strategy

The suite is organised around **observable contracts**, not around source text:

* a numeric result matches an analytic solution or an independently written
  reference implementation;
* an invariant holds (energy balance, mesh independence, symmetry);
* a wrong input is refused with a clear error.

Tests that would only restate the implementation (field copies, defaults,
"convergence is True") are deliberately absent.

## 2. What each file guarantees

**`tests/test_core.py`** — units, mesh, materials, geometry, profiles

* the Celsius/Kelvin guard rejects a field that looks like degC;
* mesh defaults are Kelvin, the domain snaps to whole cells, index round-trips
  and `find_cell` clamping work;
* face conditions are recorded per face and reject invalid faces and degC values;
* the material database is complete and validates the packing fraction;
* the packed-bed model lowers conductivity and density, and the energy density
  scales with the temperature span;
* the geometry marks heater cells for **every** pattern and
  $\sum QV = P_{total}$ exactly; a roof that does not fit, or a battery larger
  than the domain, is refused; inactive tubes create no internal convection;
  active tubes stay inside the storage band and carry no heat source;
* zone volumes/masses are consistent and `estimate_energy_capacity` works;
* profiles validate their input, extrapolate as documented, and initial
  conditions return Kelvin.

**`tests/test_solver.py`** — assembly, linear backends, steady and transient

* the interior stencil is the 7-point scheme with the right diagonal;
* **every row** equals an independently written, unvectorised assembly (this is
  the test that would catch a change in the BC treatment);
* the steady solution reproduces the analytic profile + source parabola *exactly*
  for a uniform medium with Dirichlet faces;
* the Robin flux matches the analytic wall resistance within 2 %;
* an imposed (Neumann) flux carries exactly that power;
* the Dirichlet elimination keeps the matrix symmetric, CG converges on it, and a
  genuinely asymmetric operator triggers the documented switch away from CG;
* direct and iterative solvers agree; AMG is used when PyAMG is available
  (`skip` otherwise) and compared with the Jacobi reference;
* **the transient heating rate is mesh independent** and equals $Qt/\rho c_p$
  (the `1/d^3` regression);
* fixed-temperature faces hold exactly in the transient (the Dirichlet-scaling
  regression);
* the transient matches the analytic Fourier series of a slab within a few
  percent of the driving temperature difference;
* the last step is shortened, `dt > t_final` still runs one step, and a save
  interval finer than `dt` saves every step;
* extraction cannot create energy and is capped by availability; a power profile
  with no source cell raises; a cancellation request stops the run early;
* radiation converges and increases the losses, and stays finite in the transient.

**`tests/test_analysis.py`** — fluxes, balance, losses, persistence

* with convective faces only, the surface integral equals the injected power to
  machine precision (balance closure);
* the Dirichlet ground flux is not identically zero (the old formula was);
* stored energy is referenced to the ambient and exergy is below energy;
* envelope losses do not scale with the air box;
* storage capacity and thermal autonomy follow their definitions;
* the losses analysis reaches its target within tolerance, reports a positive
  required power, and validates its input;
* HDF5 round trip, geometry-hash mismatch refusal, missing file, and upgrade of a
  legacy degC file.

**`tests/test_refinement.py`** — graded-mesh generation

* a uniform request reproduces an exactly uniform grid; grid endpoints are exact;
* band boundaries land on grid lines (masks stay clean); a wide refined band
  really gets the requested size; the neighbour growth stays bounded; no sliver
  cells; the cell budget is respected by scaling the targets;
* overlapping bands are partitioned keeping the finest target;
* the size lookup used to size element masks answers at any coordinate, and
  degenerate requests are rejected or degraded gracefully.

**`tests/test_graded_mesh.py`** — the graded grid inside the solver

* a uniform `GridSpec` reproduces the legacy uniform mesh bit for bit;
* a linear profile is exact on a graded grid and the interface flux matches the
  series resistance of the two half cells;
* halving every target does not move the answer (resolution independence);
* the energy balance still closes on a graded mesh, and every band boundary stays
  on a grid line.

**`tests/test_heaters.py`** — hairpin heater bank

* the power of the bank is deposited exactly ($\sum QV = P$);
* a hairpin occupies two legs and a bend, all inside the storage band, and only
  the active length carries power;
* the surface power is the rating of the element and its window (3-8 W/cm²) is
  reported as a warning, not a failure;
* degenerate layouts (legs or elements sharing cells, an element outside the wall)
  and tube collisions are rejected with a reason;
* the sheath resolution is measured on the mesh, and the geometry build reports
  the surface power of a discrete bank.

**`tests/test_environment.py`** — outside film, excluded air, contact resistance

* the Rayleigh number follows its definition and the Churchill-Chu value for a
  vertical wall lands in the expected range; a hot face up transfers more than a
  hot face down;
* the wind expression matches `4 + 4 v` and adds to the natural share;
* with an excluded (air) region the excluded cells hold the ambient, carry no
  conduction, and the interface loss is the film law $h_{out}A(T-T_\infty)$;
* `h_out = 0` makes that surface adiabatic;
* a contact resistance makes the interface jump and does not touch a
  single-material interface.

These tests build the excluded mask **by hand**: `mesh.excluded` / `mesh.h_out` are a
capability of the mesh and the assembly, and `BatteryGeometry` does not fill them yet
(see [13](13_REDESIGN.md) §4).

**`tests/test_convergence.py`** — the automatic mesh search

* the target rejects nonsense tolerances;
* the search stops when the answer stops moving, a loose tolerance is cheaper than
  a tight one, and a blocked refinement (cell budget, size floor) is never called
  converged;
* an unreachable tolerance is reported instead of hidden, and the reported mesh
  reproduces the converged answer;
* the search jumps to the predicted grid instead of walking one level at a time;
* the report carries the observed order and the GCI, and a noisy probe falls back
  to a conservative order.

**`tests/test_fluid.py`** — the gas loop and the pipe banks

* the rasterised wetted area is the geometric $\pi d L$ and banks split the same
  total area; a pipe outside the domain keeps nothing;
* the film coefficient follows the correlations;
* the effectiveness relation $T_{out} = T_w + (T_{in}-T_w)e^{-NTU}$ holds on a
  uniform wall, and the power leaving the fluid enters the solid;
* a large flow is area limited, a small flow is flow limited, and a lumped film
  model would overstate the extraction;
* a closed loop with no external power settles at the wall temperature; the
  resistors charge and the exchanger discharges; the bank splits the flow and
  keeps the total power; an impossible operating point is refused;
* friction factor against the classical values, pressure drop $\propto \dot m^2$,
  pressure as the cheap lever of a gas loop, the circulation loss against both
  denominators, a pressurised loop using its own density, and a circuit that
  cannot drop its own pressure being flagged;
* the staggered bank follows the published pitches and the header/module height
  limits are enforced.

**`tests/test_octree.py`** — the adaptive octree (`src/core/octree.py`)

* a uniform tree covers the box exactly, splitting replaces a leaf with eight
  children, coarsening puts them back, and the Morton round trip survives a
  mixed-level leaf;
* the **2:1 balance** holds after refining one corner and a coarse face touches at
  most four finer faces;
* the face list is conservative: the listed area equals the interface area and the
  flux balance on it is zero;
* a graded tree loses no interface and reproduces a linear field; 1-D conduction
  and a two-material wall match their analytic solutions;
* the gradient indicator refines the steep region only;
* a tree of twenty thousand leaves builds and lists its faces in under a second.

**`tests/test_cycle.py`** — charge and discharge through the loop

* charging through the loop heats the bed (the resistors heat the gas, the gas
  heats the sand - nothing is deposited in the sand directly);
* discharging through the loop cools the bed;
* the cycle reports what the fan costs (once per phase) and closes the energy
  identity, printing its numbers;
* the charge stops at the target and the discharge stops at the delivery floor
  **inside a chunk** (the per-step stop), a zero discharge power skips the phase
  with a note, and the field stays inside the Kelvin contract on a hard discharge.

**`tests/test_pipes.py`** — buried pipe networks (`src/core/pipe_network.py`)

* every riser sits inside the vessel between the two headers; the published
  pitches, the half-pitch stagger, the ring spacing and the radial files are as
  the design literature prescribes;
* the two-level collection stacks its rings, the central header takes its flow
  along the axis, and the two nozzles cross the wall and stay under the roof;
* an outlet through the roof, or below the inlet, is refused;
* the wetted area is geometric and adds up to the centreline length, the specific
  area is the sizing number, the branch shares sum to one and the reverse return
  equalises them;
* the 3 m header rule and the module height limit are reported;
* unconnected rings, risers that miss their collector and configurations that
  cannot be built are refused with the parameter to change, while warnings do not
  stop the build;
* the same configuration always builds the same network, the vessel can sit
  anywhere in the domain, the voxelisation can move to another grid, and the
  network drives the fluid loop with its branch split;
* the newer cases cover the extension of the network: the sector and ring splits give
  every main the same flow, `paint` marks the pipes inside the vessel while keeping the
  wetted area, a lagged header is painted without a gas film, the hydraulics add the
  risers to the pipes the branches share, a `FluidLoop` can be built from the network
  (on the mesh the solve will use, marching with its split and keeping the enthalpy),
  and the spiral layout winds one pitch per turn.

The network is also still being extended (wall thickness and material, roughness,
insulated headers, junction refinement), so this list is the subset that exists today;
[docs/15](15_PIPE_NETWORKS.md) is the authority for what a network is and how it is
used.

**`tests/test_octree_solver.py`** — the octree wired to the physics
(`src/solver/octree_solver.py`)

* equivalence: on a uniform tree and the matching uniform `Mesh3D` the two solvers
  assemble the same operator, so a two-layer wall (1-D) and a slab with a localised
  source (2-D) agree to the tolerance of the linear solve;
* conservation: the flux report is conservative to machine precision on a graded tree;
* the physics entry points reject what has no solution, the wall helpers pin the leaves
  of a box face, and the mean temperature weights the leaves by volume;
* the a posteriori flux-jump indicator marks the hot spot and leaves the far field
  alone, and the refinement cycle reaches the uniform answer with fewer cells while
  leaving a flat field untouched.

This module is newer than the rest of the suite: it is the first piece that puts the
octree on the physics path, and until it is adopted by `Mesh3D`/`SteadyStateSolver` the
structured grid remains the production mesh ([13](13_REDESIGN.md) §5).

**`tests/test_scene.py`** — rendering path without a display

* every field x cut axis x clip fraction combination renders (4 fields x
  {x, y, z, no cut} x {0.0, 0.5, 1.0} = 48 cases), and an empty cut returns
  `None` instead of raising;
* the material view uses the string colormap PyVista requires
  (`src/viz/scene.py::material_cmap`);
* six colormaps the *scene builder* accepts work (`coolwarm`, `jet`, `viridis`,
  `plasma`, `inferno`, `turbo`); the interactive view itself draws the fields with
  `coolwarm` only;
* legend and geometry preview build, and the preview covers the whole stack;
* a field varying along z lands in the cell whose centre carries that value
  (Fortran/C ordering of the cell data);
* `clip_grid` never leaves the domain;
* VTK/CSV exports are written and readable.

**`tests/test_gui.py`** — head-less smoke test (skipped without PySide6)

* the window builds the mesh with the panel defaults and the sources are marked;
* `RunConfig` reflects the widgets (including the degC→K conversion);
* a steady run fills the statistics/energy/materials tabs with the expected
  sections;
* a transient run produces a monotone time series and fills the transient tab;
* an invalid geometry is refused by `BatteryGeometry.validate` rather than
  clipped.

**`tests/test_gui_sweep.py`** — systematic control sweep (head-less, ~90 s)

* every widget of every panel is driven with its slots connected, with
  `sys.excepthook` watched: PySide6 prints a slot's exception and carries on, so an
  unhandled one is a handler that never worked, and the sweep fails on it;
* every panel getter answers after the sweep; incomplete profiles raise only the
  documented `ValueError`;
* the 3D view is rendered with a head-less PyVista plotter for every field, axis
  and slice position;
* each analysis type runs through the controller's own job factory, and a
  cancellation request stops a transient run early;
* save/load through the window round-trips;
* **no Qt warning or critical message is emitted** during the sweep, and no exception
  reached `sys.excepthook`.

## 3. Fixtures

`tests/conftest.py` provides three models: a 1D slab with Dirichlet ends
(conduction and Robin checks), a fully insulated box (storage, source and
transient-rate checks) and a complete battery built from
`create_small_test_geometry()` (geometry, balance, losses, persistence).

## 4. Not covered here

* **GPU**: none - the solver is CPU-only by design; the CuPy/PyOpenCL backends were
  removed and there is no device path to test (see `CHANGELOG.md`).
* **3D view**: the scene builder is exercised indirectly (the grid, the colour
  table and the exports are pure functions), but a rendered image needs a display;
  the test suite runs with `THERMAL_DISABLE_3D=1` and the widget degrades to a
  placeholder instead of crashing.
* **The default geometry with an excluded air region**: the mesh and the assembly
  support it and `tests/test_environment.py` checks it, but no caller fills
  `mesh.excluded` from `BatteryGeometry` yet, so there is no end-to-end test of it.
* **The octree as the production mesh**: `tests/test_octree.py` and
  `tests/test_octree_solver.py` cover the tree, the conservative faces, the steady solve
  and the refinement, but nothing in `Mesh3D`/`SteadyStateSolver`/`TransientSolver` calls
  them, so no end-to-end adaptive run exists.
* **Very large meshes**: `scripts/benchmark.py` covers up to ~10⁵ cells; beyond that
  the AMG hierarchy is the deciding factor and should be measured on the target
  machine.
