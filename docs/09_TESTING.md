# 9. Testing

```bash
python -m pytest tests/ -q                          # 147 tests, no GPU, no display
python -m pytest tests/ -q -W error::DeprecationWarning   # same, deprecations fatal
python -m ruff check src gui tests scripts --select F,E9,B,SIM,UP
python scripts/benchmark.py                         # timings per mesh size
```

## 1. Strategy

The suite is organised around **observable contracts**, not around source text:

* a numeric result matches an analytic solution or an independently written
  reference implementation;
* an invariant holds (energy balance, mesh independence, symmetry);
* a wrong input is refused with a clear error.

Tests that would only restate the implementation (field copies, defaults,
"convergence is True") are deliberately absent.

## 2. What each file guarantees

**`test_core.py`** — units, mesh, materials, geometry, profiles

* the Celsius/Kelvin guard rejects a field that looks like degC;
* mesh defaults are Kelvin, the domain snaps to whole cells, index round-trips
  and `find_cell` clamping work;
* face conditions are recorded per face and reject invalid faces and degC values;
* the material database is complete and validates the packing fraction;
* the geometry marks heater cells for **every** pattern and
  $\sum QV = P_{total}$ exactly; a roof that does not fit, or a battery larger
  than the domain, is refused; inactive tubes create no internal convection;
  active tubes stay inside the storage band and carry no heat source;
* zone volumes/masses are consistent and `estimate_energy_capacity` works;
* `generate_positions` is pure and element powers sum to the total;
* profiles validate their input, extrapolate as documented, and initial
  conditions return Kelvin.

**`test_solver.py`** — assembly, linear backends, steady and transient

* the interior stencil is the 7-point scheme with the right diagonal;
* **every row** equals an independently written, unvectorised assembly (this is
  the test that would catch a change in the BC treatment);
* the steady solution reproduces the analytic profile + source parabola *exactly*
  for a uniform medium with Dirichlet faces;
* the Robin flux matches the analytic wall resistance within 2 %;
* an imposed (Neumann) flux carries exactly that power;
* the Dirichlet elimination keeps the matrix symmetric, CG converges on it, and a
  genuinely asymmetric operator triggers the documented switch away from CG;
* direct and iterative solvers agree; AMG and the GPU backends are used when
  available (`skip` otherwise) and compared with the CPU reference;
* **the transient heating rate is mesh independent** and equals $Qt/\rho c_p$
  (the `1/d^3` regression);
* fixed-temperature faces hold exactly in the transient (the Dirichlet-scaling
  regression);
* the transient matches the analytic Fourier series of a slab within a few
  percent of the driving temperature difference;
* extraction cannot create energy; a power profile with no source cell raises;
  a cancellation request stops the run early;
* radiation converges and increases the losses.

**`test_analysis.py`** — fluxes, balance, losses, persistence

* with convective faces only, the surface integral equals the injected power to
  machine precision (balance closure);
* the Dirichlet ground flux is not identically zero (the old formula was);
* stored energy is referenced to the ambient and exergy is below energy;
* envelope losses do not scale with the air box;
* the losses analysis reaches its target within tolerance and reports a positive
  required power;
* HDF5 round trip, geometry-hash mismatch refusal, missing file, and upgrade of a
  legacy degC file.

**`test_refinement.py`** — graded-mesh generation

* a uniform request reproduces an exactly uniform grid; grid endpoints are exact;
* band boundaries land on grid lines (masks stay clean); a wide refined band
  really gets the requested size; the neighbour growth stays bounded; no sliver
  cells; the cell budget is respected by scaling the targets;
* overlapping bands are partitioned keeping the finest target.

**`test_scene.py`** — rendering path without a display

* every field × clip axis × clip fraction combination renders (12 × 4 × 3), and an
  empty cut returns `None` instead of raising;
* the material view uses the string colormap PyVista requires;
* every colormap offered by the GUI works; legend and geometry preview build;
* a field varying along z lands in the cell whose centre carries that value
  (Fortran/C ordering of the cell data);
* VTK/CSV exports are written and readable.

**`test_gui_sweep.py`** — systematic control sweep (head-less)

* every combo index, spin endpoint, checkbox state and radio of every panel is
  driven with its slots connected: a PyQt6 process aborts on a slot exception, so
  this is the check that the UI cannot take the application down;
* every panel getter answers after the sweep; incomplete profiles raise only the
  documented `ValueError`;
* the 3D view is rendered with a head-less PyVista plotter for every field, axis
  and slice position;
* each analysis type runs through the controller's own job factory, and a
  cancellation request stops a transient run early;
* save/load through the window round-trips;
* **no Qt warning or critical message is emitted** during the sweep.

**`test_gui.py`** — head-less smoke test (skipped without PyQt6)

* the window builds the mesh with the panel defaults and the sources are marked;
* `RunConfig` reflects the widgets (including the degC→K conversion);
* a steady run fills the statistics/energy/materials tabs with the expected
  sections;
* a transient run produces a monotone time series and fills the transient tab;
* an invalid geometry is refused by `BatteryGeometry.validate` rather than
  clipped.

## 3. Fixtures

`tests/conftest.py` provides three models: a 1D slab with Dirichlet ends
(conduction and Robin checks), a fully insulated box (storage, source and
transient-rate checks) and a complete battery built from
`create_small_test_geometry()` (geometry, balance, losses, persistence).

## 4. Not covered here

* **GPU**: none - the solver is CPU-only by design (see `CHANGELOG.md`).
* **3D view**: the scene builder is exercised indirectly (the grid, the colour
  table and the exports are pure functions), but a rendered image needs a display;
  the test suite runs with `THERMAL_DISABLE_3D=1` and the widget degrades to a
  placeholder instead of crashing.
* **Very large meshes**: the benchmark script covers up to ~10⁵ cells; beyond that
  the AMG hierarchy is the deciding factor and should be measured on the target
  machine.
