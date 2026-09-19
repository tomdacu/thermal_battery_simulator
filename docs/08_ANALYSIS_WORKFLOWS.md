# 8. Analysis workflows

The three runs the application offers, what each one computes, and how the
reported numbers are defined.  Entry points: `src/solver/steady.py`,
`src/analysis/losses.py`, `src/solver/transient.py`.

## 1. Steady state

**Question**: with the heaters running at the rated power, where does the heat
go and what temperature field settles?

1. `BatteryGeometry.apply_to_mesh` paints the mesh and writes the heater power
   into `mesh.Q_source` over `mesh.source_mask`.
2. `SteadyStateSolver.solve()` assembles `A T = b` and solves it
   (`src/solver/matrix.py`, [02](02_FDM_DISCRETIZATION.md)).  With radiation
   enabled the linearised coefficient is refreshed in a Picard sweep until the
   field moves less than `picard_tolerance` (1e-3 K) or `max_picard` sweeps run out.
3. The solver reports `converged`, the relative residual, the sweep count, the
   wall time, the solver notes (method substitutions) and min/max/mean/median
   temperatures.

**Reported**: temperature statistics (whole domain and storage), the envelope
loss breakdown, stored energy and exergy, thermal autonomy
(`ResultsPanel.update_energy`).

## 2. Losses analysis

**Question**: which heater power holds the storage at a target temperature - and
how much of it leaks away?

`solve_losses(mesh, LossesConfig, SolverConfig)`:

1. the storage cell mask is taken from `MaterialID.SAND`; the total volume is the
   number of cells times the cell volume;
2. the ground coupling is applied explicitly: `h_ground > 0` installs a
   convective bottom face, `0` keeps the fixed-temperature ground of the
   geometry (writing `mesh.bc_h` without changing the boundary type - the old
   behaviour - was silently ignored by the assembly);
3. the power is iterated: solve, compare the mean storage temperature with the
   target, update with a secant/Newton step using the measured slope
   $dT/dP$, damped by `relaxation`, with the power floored at 0;
4. the loop stops when $|T_{mean} - T_{target}| \le$ `tolerance` [K], when
   `max_iterations` is reached, when the user cancels, or if the linear solve
   fails.

**Reported**: converged flag, iterations, the required power [W] and its density
[W/m³], the final mean storage temperature, and the full energy balance of the
converged state (losses split by direction, stored energy, autonomy).  The
iteration history (power and temperature per sweep) is in `LossesResult.history`.

## 3. Transient

**Question**: how does the unit charge and discharge over time with a given
power schedule and extraction?

`TransientSolver.run(progress_callback, should_stop)`:

1. `TransientConfig.validate(mesh)` checks the time stepping, the profiles and
   the initial condition; the mesh is validated too;
2. the initial field is applied (`InitialCondition`: uniform, per material, from
   an HDF5 state, or from a steady pre-run driven by the controller);
3. the operators `(M/dt + L)` are built once (rebuilt every step when radiation
   is on), together with the preconditioner and its cache;
4. for every step `t -> t + dt` (the last step is shortened so the loop never
   overshoots `t_final`; the operators are rebuilt for that shortened step,
   otherwise the matrix and the right-hand side would disagree):
   * **power**: `power_at(t)` is spread over `source_mask` as
     $Q = P/(n_{cells}V_{cell})$; a profile that asks for power with no source
     cell raises `ValueError` instead of heating nothing;
   * **extraction**: `off` clears the tube coupling; `flow_rate` installs the
     tube convection with `t_inlet`; `power` imposes a volumetric sink on the tube
     cells **capped** at the available $h A (T_{tube} - T_{inlet})$, so no heat is
     extracted from a body colder than the inlet;
   * the right-hand side is refreshed, the step is solved with a warm start, and
     the field is written back to the mesh;
   * the balance is evaluated and the cumulative energies accumulate
     $P\,dt$;
   * samples are stored every `save_interval` **and always at `t_final`**, so the
     series ends exactly at the requested time (the full field too, if
     `save_full_field`), and the loop checks `should_stop()`;
5. a non-converged linear step is recorded in `solver.notes` and does not stop
   the run; the time step is simply left as solved.

**`TransientResults`** carries one row per sample: mean/max/min storage
temperature, insulation and shell means, heater and extracted power, the four
loss terms, stored energy, the three cumulative energies, exergy, and optionally
the full fields.  `export_csv()` writes the whole time series.

## 4. Definitions of the reported quantities

| quantity | definition | where |
|---|---|---|
| `P_in` | $\sum Q_{source}V$ over the source cells | `analysis/balance.py` |
| `P_extracted` | $-\sum Q_{sink}V$ plus the tube-fluid flux | idem |
| `Q_losses` (envelope) | surface integral over the battery/air interface, split top/side/bottom | `analysis/fluxes.py::envelope_fluxes` |
| `Q_domain` | surface integral over the six box faces - the audit quantity that closes the balance | `analysis/fluxes.py::domain_fluxes` |
| `E_stored` | $\sum \rho c_p (T-T_0)V$ with $T_0$ the ambient | `fluxes.stored_energy` |
| `Ex_stored` | $\sum \rho c_p[(T-T_0)-T_0\ln(T/T_0)]V$ | `fluxes.stored_exergy` |
| `Ex_destroyed` | exergy entering with the heaters (Carnot factor at the storage temperature) minus what is stored | `fluxes.destroyed_exergy` |
| `imbalance` | $P_{in} - P_{extracted} - Q_{domain} - dE/dt$; the self-check, ~0 for a correct solve | `compute_balance` |
| thermal autonomy | `E_stored / Q_losses` expressed in hours | `balance.thermal_autonomy` |
| mean storage T | mean over `MaterialID.SAND` cells | `Balance.t_mean_storage` |

Losses are reported twice on purpose: *envelope* losses describe the battery
(they do not change if you enlarge the surrounding air box) while *domain* losses
describe the simulated box.  The first is the engineering number, the second is
the one that must close the energy balance.

## 5. Saving and restoring a state

`StateManager.save_state(mesh, name, geometry_params)` writes an HDF5 file with
version, creation time, the temperature unit, the geometry hash
(shape, box, spacing, per-material cell counts, source-cell count and the
geometry parameters), the material map, the temperature field and the sources.

`load_state` + `apply(mesh, state)` refuse a file whose grid, geometry hash or
material map does not match the current mesh, and upgrade a legacy degC file to
the Kelvin contract while recording the conversion in `state.notes`.  The GUI
shows those notes in the log; the geometry hash is what stops a field computed
for a different model from being silently mixed with the current one.
