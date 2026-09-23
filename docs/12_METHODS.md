# Methods and why

Every numerical and modelling choice of the simulator, with the reason it was made,
the alternative that was rejected and how it is checked.  Read this together with
[02_FDM_DISCRETIZATION.md](02_FDM_DISCRETIZATION.md) (the exact operators) and
[10_MESH_AND_HEATERS.md](10_MESH_AND_HEATERS.md) (the mesher).

The rule of the house: a choice stays only if it can be *stated* (what it assumes),
*justified* (why it beats the alternatives here) and *checked* (which test or
measurement would catch it being wrong).

---

## 1. Discretisation: cell-centred finite volume on a Cartesian grid

**Choice.** The domain is a structured Cartesian grid; the unknown is the cell-average
temperature; the flux between neighbours is computed on the shared face.

**Why.** Finite volume is *conservative by construction*: every face flux appears with
opposite signs in the two cells that share it, so the discrete energy balance closes to
machine precision on **any** grid - uniform, graded, coarse or fine.  That property is
what makes an energy-balance check a useful *verification* tool rather than a symptom of
discretisation error.  A Cartesian grid keeps the geometry masks, the heater rasteriser
and the matrix assembly simple and vectorisable; a body-fitted or cut-cell grid would
represent the cylinder surface better but at the cost of a much larger implementation
surface (and this geometry - a squat cylinder in a box - loses little).

**Rejected.** Finite differences on the same grid (no natural conservation statement),
body-fitted meshes (complexity not justified by the geometry), finite elements (mass
lumping to keep the transient explicit-free, more machinery).

**Checked.** `tests/test_solver.py` compares the assembled operator against an
independently written reference; `tests/test_graded_mesh.py` checks that the energy
balance closes on a graded grid; the analytic slab and cylinder cases in
`tests/test_solver.py` pin the coefficients.

### Conductivity at a material interface: harmonic mean

**Choice.** The face conductivity is the harmonic mean of the two cell values weighted
by the centre-to-centre distances.

**Why.** It is the only average that reproduces the *exact* series resistance of a
one-dimensional two-layer wall when the interface sits on a face.  With conductivity
ratios of 1000:1 (steel 50 vs rock wool 0.04) an arithmetic mean overestimates the
flux through the interface by orders of magnitude; the harmonic mean gives the
interface resistance `t1/(2k1) + t2/(2k2)` exactly.

**Checked.** `tests/test_solver.py::test_two_layer_wall_flux_is_exact` (analytic
composite-wall flux), and the graded/uniform comparison in `tests/test_graded_mesh.py`.

### Symmetrisation for the iterative solver

**Choice.** The system solved is `diag(V)^-1/2 K diag(V)^-1/2 y = diag(V)^-1/2 b`
with `T = diag(V)^-1/2 y`.

**Why.** The physical operator `K` (conductances, W/K) is symmetric, but writing it
per unit volume divides row *i* by `V_i` and destroys symmetry, which costs the
Conjugate Gradient method.  The similarity transform restores symmetry **exactly**
(and is what makes the matrix positive definite), so CG and its AMG preconditioner
stay available on graded grids too.  The alternative - falling back to BiCGSTAB - is
kept for genuinely non-symmetric cases (mixed boundary conditions) but is slower.

**Checked.** `tests/test_solver.py` asserts symmetry to rounding level and that CG
converges on a graded grid; the solver reports a note when it has to fall back.

---

## 2. Boundary conditions

| Condition | Discretisation | Why |
|---|---|---|
| Convective (Robin) | half-cell conduction `h_cell/(2k)` in series with the film `1/h` | keeps the surface resistance physical; a "cell at the film temperature" would need the surface cell to be isothermal |
| Dirichlet | elimination of the pinned unknowns, with the eliminated columns moved to the right-hand side | symmetric, no penalty parameter to tune; the reaction flux is recovered exactly for the balance |
| Adiabatic | zero conductance | - |
| Radiation | linearised `h_rad = eps sigma (T_s^2 + T_inf^2)(T_s + T_inf)`, iterated | the exact `T^4` term makes the system nonlinear; the linearisation converges in a handful of sweeps and reuses the same matrix |

**Consequence to know about.** The Robin condition is **first order** in the cell size
(the half-cell distance is `h_cell/2`), while the interior is second order.  The
observed order of a mesh study therefore sits between 1 and 2 - this is exactly why
the automatic mesh search floors the observed order at 1 (conservative) instead of
trusting a formal order of 2.

**Consequence for the mesh.** The half-cell resistance must not dominate the film
resistance, i.e. `h_cell <= 2k/h`.  For rock wool (k = 0.04) at h = 5 W/(m²·K) that is
16 mm; a 100 mm grid hides the whole surface drop inside the first cell and the losses
cannot converge.  `src/analysis/mesh_plan.py` turns this into the starting point of the
mesh search.

---

## 3. Linear solver

**Choice.** CG on the symmetrised system, AMG (Ruge-Stüben) preconditioner when PyAMG
is installed, Jacobi otherwise; direct sparse LU for small problems.  Since 2026-09-23
this is also what the GUI runs, with no selector: every operator the application
assembles is symmetric once the volumes scale it, and on the default model (143 396
leaves, steady) CG + AMG RS took 1.8 s against 2.8 s for CG + Jacobi, 5.6 s for CG + AMG
smoothed aggregation and 6.3 s for BiCGSTAB + Jacobi (the previous GUI default); all four
agree with a direct solve to 1e-4 K, CG + AMG RS to 1e-5 K in 6 iterations.  The linear
layer still falls back by itself - BiCGSTAB on an operator that is not symmetric, Jacobi
without PyAMG - and says so.

**Why.** The matrix comes from a diffusion operator: CG with a multigrid preconditioner
has iteration counts that grow only weakly with the mesh size, which is what keeps a
1 M-cell solve interactive.  Jacobi is the safe default (no optional dependency, cheap
setup, parallel); ILU is available but is single-threaded in SuperLU and often *slower*
than Jacobi here; direct LU is O(N^1.5) and reserved for small meshes where its
predictability wins.

**Tolerance.** The reported number is the *recomputed* relative residual
`||b - Ax||/||b||`, not the internal estimate of the Krylov method, so two methods can
be compared honestly.  A run that hits `max_iterations` is reported as
`converged=False` and the field is not silently presented as a solution.

**Cache.** The AMG hierarchy is the dominant setup cost, so it is kept while the matrix
*content* is unchanged (a hash of the CSR arrays).  It used to be keyed on the matrix
object as well, and the symmetrised operator of a graded mesh is a new object at every
call, so a transient on the tree rebuilt the hierarchy at every step.

**Checked.** `tests/test_solver.py` (convergence, symmetry, fallback note), the
benchmarks in `scripts/`.

---

## 4. Iterations and their stopping criteria

| Iteration | Criterion | Why this criterion |
|---|---|---|
| Radiation sweeps, gas-loop coupling (steady) | max field change between sweeps <= 1e-3 K, at most 60 sweeps | the linearised coefficient and the gas temperatures are fixed points of the field they drive; the *field* change is what the user sees, and 1e-3 K is far below any physical uncertainty |
| Losses analysis (secant) | change of the storage mean temperature <= tolerance | the target is a temperature set point; the secant method converges in 3-6 iterations because the loss is smooth and monotone in the power |
| Automatic mesh | Richardson estimate of the discretisation error of the *chosen* grid, against the tolerance | with predicted jumps the raw change between two grids is dominated by the coarse one; the Richardson estimate divides it by `r^p - 1` and is what Roache's GCI is built on |

---

## 5. Mesh: graded bands, and how the grid is chosen

**Choice.** The user declares *physical* targets (cells across the storage, the
insulation, the sheath; far-field size; growth ratio); the mesher turns them into a
graded grid.  The automatic search then moves the targets until the steady answer stops
moving.

**Why a graded grid.** The temperature field has three very different length scales
(4 m of sand, 300 mm of insulation, 12 mm of sheath); a uniform grid able to resolve
the sheath would need ~10⁸ cells.  Grading puts the cells where the gradients are.

**Why the ramp is linear** (`h(x) = h_band + (growth-1)·dist`).  The neighbour ratio of
the resulting grid is bounded by `growth` *by construction*: a cell of size `h` grows by
`(growth-1)·h` over its own length, so the next cell is at most `growth` times larger.
An exponential ramp reaches the coarse size in a shorter distance but its per-cell ratio
grows with the cell size, which is exactly what produces the "staircase" error of naive
graded grids.

**Why density equidistribution inside a band.** Cells are placed where the cumulative
density `1/h` crosses an integer: the local size follows the field, a constant field
gives an *exactly* uniform grid, and the remainder is absorbed by the last cell instead
of leaving a sliver.

**Known limits** (measured, not hidden): the ramp needs a distance
`(h_coarse - h_fine)/(growth-1)` to relax, so a very fine band inside a very coarse box
spends cells on the ramp; and a band thinner than its own target must still contain one
cell (a structured grid cannot split a cell across a zone boundary).  The GUI reports
the realised cell-size range and the worst neighbour ratio of every grid it builds.

### The automatic mesh search (Richardson / GCI)

The method is the standard three-grid procedure for discretisation-error estimation
(Richardson extrapolation; Roache's Grid Convergence Index; ASME V&V 20; Celik et al.,
*Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD
Applications*, J. Fluids Eng. 130 (2008) 078001):

1. **Probe** - solve three grids, each one `refine` times finer (refinement factor
   above the 1.3 recommended by ASME), and fit the observed order of accuracy `p` from
   the two changes;
2. **Predict** - calibrate `E = C h^p` on those three grids and invert it for the
   tolerance: the search *jumps* to the grid size it needs instead of walking there one
   level at a time;
3. **Verify** - solve the predicted grid and check the Richardson error of the chosen
   grid against the tolerance; if it still moves, repeat the prediction with the new
   pair.

`p` is floored at 1 (mixed-order scheme, see §2) and the jump is bounded to a few
refinement steps, so a poorly fitted order can only cost an extra solve, never a wrong
answer.  The report carries `p` and the GCI of the chosen grid (with Roache's factor of
safety 1.25), i.e. the discretisation uncertainty is *stated* rather than implied.

**Why this and not "keep refining until two grids agree".** The raw change between two
successive grids is only a valid error estimate when they are adjacent and in the
asymptotic range; with a predicted jump it is dominated by the coarse grid and rejects
grids that are already good.  The Richardson estimate is invariant to the jump size,
which is what makes the jump possible - and the jump is what makes the search cheap.

**Starting point.** `src/analysis/mesh_plan.py` computes, before any solve, the cell
size each region asks for: `min(thickness/N, 2k/h)` with `N = 8` cells across a layer
and the convective sub-layer `2k/h` (the conduction analogue of the "first cell height"
rule of CFD practice).  The search starts from that plan, which is why it does not
waste solves on grids that cannot represent the surface flux.

**Checked.** `tests/test_convergence.py`: the search stops when the answer stops moving,
a loose tolerance is cheaper than a tight one, an unreachable tolerance is reported
instead of hidden, a blocked refinement is never called converged, the reported mesh
reproduces the reported answer, the observed order and the GCI are reported, and a
non-monotone (noisy) probe still converges instead of looping.

---

## 6. Transient: backward Euler with an adaptive step

**Choice.** Implicit (backward) Euler in time, with the step chosen so the field change
per step stays inside a tolerance.

**Why.** Unconditionally stable: the sand battery is stiff (steel and sand differ in
`rho*cp` by a factor 5, the insulation by a factor 100), so an explicit scheme would be
limited by the *smallest* cell and the *largest* diffusivity - a few milliseconds.  The
price is first-order accuracy in time, which is acceptable because the interesting
dynamics (hours) is much slower than any reasonable step (seconds to minutes).  Crank-
Nicolson would be second order but rings on sharp transients (step changes of power),
which is exactly what this tool is used to simulate.

**Checked.** `tests/test_solver.py` (transient decay against the analytic solution), the
energy balance is monitored per step.

---

## 7. Losses analysis

**Choice.** Find the heater power that holds a target storage temperature, by a secant
iteration on the power; the loss itself comes from the envelope flux integral.

**Why the secant method.** The loss is a smooth, monotone function of the power with an
unknown derivative; secant needs no derivative and converges superlinearly (a Newton
method would need `dQ/dP`, which costs two solves per step).  The first step scales the
power by the rise it still has to make over the ambient, which is exact for conduction
and films without radiation, so the default model converges in **2 iterations** (it took
6 with a 2 % first step and a 0.7 under-relaxation); one solver is kept for the whole
iteration, so the AMG hierarchy and the warm start carry over.

**Assumption to know about.** The iteration treats each solve as a steady state: the
loss at the target temperature is computed *as if* the battery had been sitting there.
Storage dynamics (charging/discharging) belong to the transient analysis.  With a painted
network the power is the resistors' and it enters through the pipe walls (the coupled
steady state of §11), so the answer is the plant's holding power; the model is the mesh
as built, the ground condition included (the losses run no longer rewrites it).

---

## 8. Materials and effective properties

**Choice.** A small built-in database (steatite, silica sand, rock wool, glass wool,
carbon steel, concrete, air) plus a packed-bed model that combines the solid and the
fluid conductivity with a geometric mean weighted by the packing fraction.

**Why a geometric mean.** It is the standard interpolation for a two-phase medium when
one phase is dispersed in the other, it reduces to the right limits (`phi -> 0` gives
the solid, `phi -> 1` the fluid) and it is monotone in between.  More elaborate
correlations (Zehner-Bauer, Kunii-Smith, Maxwell) add terms for particle contact,
internal radiation and internal convection that need parameters (particle size,
emissivity, coordination number) this tool does not ask the user for.

**Caveat, stated openly.** The database value for silica sand (k = 0.35 W/(m·K)) is a
*bulk* value for a dry sand bed, i.e. it already contains the porosity.  Feeding it to
the geometric mean as the *solid* phase counts the porosity twice and gives
`k_eff ~ 0.13 W/(m·K)`, below the measured range of dry sand beds (0.15-0.35).  Either
the database value should be the solid-phase value (quartz, 6-12 W/(m·K)) or the
packed-bed model should be skipped for materials whose value is already bulk.  Until
that is decided, treat the sand's effective conductivity as an assumption of the model,
not as data.

**Checked.** `tests/test_core.py` (packing fraction limits, energy density), and the
material database is the single source of truth used by solver, balance and GUI.

---

## 9. Reporting: what the numbers mean

* **Temperatures** are Kelvin inside `src/` and converted only in `gui/units.py`; a field
  that looks like Celsius is rejected (`check_kelvin`).
* **Energy balance** is always reported with the same flux evaluator the solver uses
  (`src/analysis/fluxes.py`), so the balance closes to machine precision; a non-closing
  balance means a bug, not a modelling error.
* **Autonomy** is stored energy divided by the *loss rate of the whole domain* - the
  ground path included.
* **Mesh uncertainty** is reported as the GCI of the chosen grid (see §5), never as an
  unqualified number.

---

## 11. Coupling the gas and the bed

**Choice.** The loop is marched on the current wall temperatures, and each pipe cell
receives the exact exchange of its piece of pipe *written as an implicit film*:

```
q = m_dot c_p (T_in,cell - T_out,cell) = G (T_gas - T_wall)
G = m_dot c_p (1 - e^-NTU)          [W/K]
T_gas = temperature the gas enters the cell at
```

`FluidResult.apply` writes `G` and `T_gas` on the cell as `bc_h` and `bc_T_inf`, so the
solver treats `T_wall` implicitly.  In a transient the loop inlet is solved *with* the
field: every `T_gas` moves with `T_in` by a known slope, the field moves with it by the
solution of one more system with the same operator, and the one scalar that makes the
deposit equal the external power closes the loop's enthalpy balance on the new field (a
one-row Schur complement).  The gas is re-marched at the end of every step, so the state
a caller reads between steps - the delivery temperature a cycle stops on - belongs to the
field it sees.  The steady state iterates march + solve (Picard) with the same
balance held in every sweep, so each sweep deposits the resistors' power and the
iteration only settles how the gas distributes it (11 sweeps, 21 s on the default
model of 143 396 leaves).  A pipe cell keeps the bed's properties: the pipe is a
thin wall inside a cell of sand, and only its exchange is added to the cell.

**Why.** The previous coupling deposited the marched power as a fixed source for the
whole step.  That is an explicit scheme in the wall temperature: a cell whose `G` is
large against its heat capacity overshoots, and on long steps the bed oscillated -
down to below 100 K in the cycle runs of 2026-09-21.  The implicit film is stable at any
step; holding the balance keeps the energy ledger exact (the charge deposits exactly the
resistors' power).  Measured on the test silo, discharge time against the step:
19.0 / 18.5 / 18.0 / 18.0 / 17.9 h for dt = 3600 / 1800 / 900 / 450 / 225 s (the explicit
coupling: 19.0 / 18.5 / 18.25 / 18.1 / 18.0 h, where it did not diverge).

**The steady state is the plant's.**  Before 2026-09-23 the steady and losses runs
spread the power uniformly over the sand and gave the painted pipes a fixed film at a
fixed "gas temperature" of 60 °C - the lumped tube model - so the pipes were a *sink*
(967 W of the 5 kW on the default model, the storage mean 40 K too cold).  Now the power
enters through the pipe walls at the circuit's flow, and the gas temperatures are the
loop's own.

**Limits.** The gas properties are evaluated once, at the prescribed inlet or at 300 K,
not along the loop: `h` and the pressure drop therefore do not follow the temperature of
the gas (at 600 °C the turbulent `h` of air is ~30 % higher, the laminar one ~2.4x).
Making them follow it changes `G`, hence the operator, and needs a rebuild policy; it is
the next refinement of this model.

**Checked.** `tests/test_cycle.py` (the deposit equals the external power; the cycle
identity closes to 1 %; the discharge stop within one step), `tests/test_fluid.py`,
`tests/test_transient_on_tree.py` (the same ledger on the tree and on the grid).

---

## 10. Open modelling questions (decisions for the owner)

1. The sand conductivity caveat of §8.
2. The `2k/h` criterion applies to the outer insulation surface; whether the *inner*
   steel shell (k = 50) needs its own resolution rule is a judgement call - the plan
   currently leaves it to the insulation target.
3. Radiation is linearised and iterated; a Newton linearisation would converge faster
   for large temperature swings (T^4 at 900 K vs 300 K ambient).
