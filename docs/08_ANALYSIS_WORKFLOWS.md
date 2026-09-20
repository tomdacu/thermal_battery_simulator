# 8. Analysis workflows

The runs the application offers, what each one computes, and how the
reported numbers are defined.  Entry points: `src/solver/steady.py`,
`src/analysis/losses.py`, `src/solver/transient.py`, `src/analysis/convergence.py`,
`src/analysis/cycle.py`, `src/solver/fluid.py`.

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

## 4. Automatic mesh search

**Question**: which grid makes the steady answer trustworthy at the tolerances the
user asked for?

Entry point `src/analysis/convergence.py::find_mesh(build, solve, target, ...)`; in the
GUI it is the *Find the mesh* button and the `automesh` job kind (it runs on the same
background thread as a simulation).

1. `main_window._refresh_plan` recomputes the **a priori plan**
   (`src/analysis/mesh_plan.py`): cells across a layer (`thickness/N`) and the
   convective sub-layer (`2k/h`) per region, which *tightens* the manual targets
   (`GeometryPanel.set_plan_targets` / `planned`).
2. The search builds and solves the **steady** case on a sequence of grids, each one
   `refine` finer (default 0.6), up to `max_levels` (default 4).
3. The error model (observed order from the level pairs, Richardson extrapolation, the
   GCI of Roache with $F_s = 1.25$) predicts the grid that should meet
   `delta_temperature` [K] on the storage mean and `delta_power` (relative) on the heat
   leaving the battery; the search **jumps** to that grid rather than walking one level
   at a time.
4. It stops when both changes are within tolerance.  If the cell budget or the minimum
   cell size prevents the refinement, it reports **stopped** with the reason instead of
   calling two identical grids "converged".
5. The adopted `GridSpec` is returned in the report and used by the build; the panel
   shows `converged: <cells> cells, dT .. K, dP ..%`.

**Reported**: `ConvergenceReport` - the levels tried (`ConvergenceLevel`: cells,
dT, dP, wall time), the chosen one, `converged`, the observed order and the GCI, plus a
human-readable `message` when it stopped.

## 5. Cycle: charge, standby, discharge

**Question**: what does a full operating cycle cost and where does the energy go?

`src/analysis/cycle.py::run_cycle(mesh, loop, settings, solver_config, progress)` drives
the three phases with the **transient solver**, in chunks (`settings.chunk`, default
12 h) so the driver can watch the state and stop a phase where the physics says so:

* **charge** until the storage mean reaches `t_target` (default 773.15 K), with the
  resistors as the external power of the loop, limited by `charge_limit`;
* **standby** for `standby_time`;
* **discharge** until the delivery temperature falls below `t_delivery_min`, requesting
  `discharge_power` at the exchanger.

The accounting is the identity that cannot hide a term:

$$E_{in} = \Delta E_{stored} + E_{delivered} + E_{standby} + E_{circulation}
+ E_{unrecovered}$$

with `E_in` what the resistors take from the grid, `E_delivered` what the exchanger
hands to the user, `E_standby` what leaks through the envelope while the store sits,
`E_circulation` the blower work, and `E_unrecovered` the heat still in the bed when the
discharge stops - the number that decides whether a low-temperature tail is worth
chasing.  `CycleReport` also carries the per-phase rows (`CyclePhase`) and `summary()`.

The phase boundaries are inspected per chunk, so a discharge fast compared with the bed
capacity can overshoot the delivery floor inside one chunk; a per-step stopping criterion
inside the transient solver is the clean fix and is the item being completed in
`src/analysis/cycle.py` (its module docstring carries the live status).

## 6. The gas loop as the heat path

The redesign charges and discharges the bed through a **closed gas loop in buried
pipes** ([13](13_REDESIGN.md), [15](15_PIPE_NETWORKS.md)).  The pieces a run uses:

1. `src/core/pipes.py` / `src/core/pipe_network.py` give the geometry: runs (polylines),
   the wetted area per cell, the branches and their flow shares;
2. `src/solver/fluid.py::FluidLoop` marches each run with the exact relation
   $T_{out} = T_w + (T_{in}-T_w)e^{-NTU}$, $NTU = hA/(\dot m c_p)$, deposits
   $\dot m c_p (T_{in}-T_{out})$ in the bed as a volumetric source `q_fluid` [W/m³] - the
   same term charges and discharges, its sign follows the loop - and computes the film
   coefficient from the flow (`pipe_h`) unless the caller fixes it;
3. the loop is closed, so every temperature is affine in the loop inlet temperature and
   the balance $\sum_i \dot m_i c_p (T_{out,i} - T_{in}) = -Q_{ext}$ is solved for
   $T_{in}$ in one step; `Q_ext > 0` are the resistors, `Q_ext < 0` the exchanger;
4. pressure drop, fan power and the circulation loss share come from the same module
   (`friction_factor`, `pressure_drop`, `fan_power`), and an operating point the flow
   cannot carry is refused with a message.

## 7. Definitions of the reported quantities

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
| `Q_environment` | $h_{out}A(T-T_{ambient})$ over the active/excluded interfaces; folded into `Q_domain` so the balance closes when the air is excluded | `fluxes.environment_flux` |
| `E_circulation` | blower work of the gas loop, from the pressure drop and the flow | `solver/fluid.py::fan_power`, `analysis/cycle.py` |
| `E_delivered` / `E_unrecovered` | heat handed to the user during a discharge / heat still in the bed when the delivery temperature falls below its floor | `analysis/cycle.py` |
| thermal autonomy | `E_stored / Q_losses` expressed in hours | `balance.thermal_autonomy` |
| mean storage T | mean over `MaterialID.SAND` cells | `Balance.t_mean_storage` |

Losses are reported twice on purpose: *envelope* losses describe the battery
(they do not change if you enlarge the surrounding air box) while *domain* losses
describe the simulated box.  The first is the engineering number, the second is
the one that must close the energy balance.

## 8. Saving and restoring a state

`StateManager.save_state(mesh, name, geometry_params)` writes an HDF5 file with
version, creation time, the temperature unit, the geometry hash
(shape, box, spacing, per-material cell counts, source-cell count and the
geometry parameters), the material map, the temperature field and the sources.

`load_state` + `apply(mesh, state)` refuse a file whose grid, geometry hash or
material map does not match the current mesh, and upgrade a legacy degC file to
the Kelvin contract while recording the conversion in `state.notes`.  The GUI
shows those notes in the log; the geometry hash is what stops a field computed
for a different model from being silently mixed with the current one.
