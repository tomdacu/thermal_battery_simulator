# Redesign: the sand battery as it is actually built

This document records the redesign decided after studying the reference machine (Polar
Night Energy, Finland) and the physics of a storage discharge.  It supersedes the parts
of the previous design that assumed a box of air around the battery, hairpin heaters
inside the sand and tubes as lumped convective sinks.

Read with [12_METHODS.md](12_METHODS.md) (why the numerical choices are what they are).

---

## 1. The reference class of machines (published industrial data)

The class of machine this simulator targets is a **high-temperature sensible-heat
storage in a granular medium, charged and discharged by a gas loop through pipes buried
in the bed**.  The numbers below are the published envelope of industrial installations
in operation since 2022; they describe the *class*, not one product, and the design space
is deliberately left open (different geometries, different fluids, different pressures).

| Aspect | Published range | What it means for the model |
|---|---|---|
| Medium | sand, crushed rock, industrial by-products (density matters more than grain size) | the material database stays the source of truth |
| Storage temperature | up to 600 °C (construction materials, not the sand, set the limit) | property temperature dependence is mandatory |
| Output temperature | 60-400 °C, plus steam for process heat | the delivered exergy depends on the *usable* energy above the return temperature |
| **Charging** | electric resistors heat a gas that circulates in a **closed loop through buried pipes**; the resistors are in the gas circuit, never in the bed | charging is an *external power on the fluid loop*, not a volumetric source in the sand |
| **Discharging** | flow reversed: cold gas in, hot gas out to a counter-flow exchanger (or a steam generator) | the same network does both; the loop's cold side is the exchanger return |
| Container | insulated steel vessel, 4-30 m across, 7-15 m tall, **no gas outside the pipes** | the outer wall is a boundary condition, not a domain |
| Module | up to ~100 m³ per module, modular scaling | sizes and powers are per module |
| Efficiency | round-trip 85-90% at scale, ~60-75% at the smallest pilot; **circulation loss ~5% per cycle**, standby loss < 5% | both terms must be *reported*, not assumed |
| Retention | months of useful heat; the bed loses about half of its energy in three months at rest | the standby path decides seasonal viability |
| Measured profile | core ~500 °C, edge 150-200 °C: a strong permanent radial gradient with a cold shell | the mesh must resolve the shell, and the report should give energy above a delivery temperature |

### The gas loop in detail (from a granted patent on the closed-loop design)

* the pipes are **stainless steel** (304L/304H/316L/316H/321H/330), arranged
  **vertically** in the module (horizontal or inclined are claimed alternatives);
* the gas is preferably **inert** at **up to 50 bar**, and the stated reason is that
  *"pressurization may be utilized to increase the heat transfer rate without increasing
  the gas velocity excessively"* - a design lever the model should expose: at constant
  velocity `Re ~ rho`, so `h` rises with pressure while the fan loss stays put;
* the fan is bidirectional, rated for the hot side, with its **motor outside the loop**;
* **charge top->bottom** (aids natural convection, cuts fan power), **discharge
  bottom->top**; the gas leaves the bed close to the bed temperature and goes to a
  **counter-flow** exchanger, leaving it at **40-70 °C** - the loop's cold-side
  temperature, i.e. the `t_in` of a discharge in the model;
* insulation is a low-conductivity granular material (`lambda < 0.3 W/(m K)`);
* an alternative discharge path is claimed as **thermosiphon tubes** (sealed, partial
  liquid fill, dry-up allowed, superheated vapour, inclined, up to 15 m long, example
  9 mm inner diameter at 200 bar / 500 °C): passive, no fan, but a two-phase pressurised
  circuit - listed here, not modelled.

The architecture is therefore **gas inside pipes**, not a distributed flow through the
bed: the distributed-flow family (rock-bed electric-thermal storage, CSP rock beds) is a
different machine and is not what this simulator imitates.

*Sources are deliberately not named: the project explores the design space around this
class of machines rather than reproducing one product.  The numbers above are published
industrial data; the physics references are named in the rest of this document.*

### From the same patent family and the published efficiency notes

* the **resistor sits in the gas circuit** (a heater in an insulated vessel), never in
  the bed; the gas leaves it at 700-1000 °C (up to 2000 °C claimed);
* a module is at most ~100 m³, often underground, insulated with expanded clay (LECA,
  `lambda < 0.3 W/(m K)`);
* bed temperature preferably 300-500 °C, and its implied charge power density is
  ~1.1 kW/m³ of bed;
* **circulation losses (fan + ducts) are ~5 % per cycle**, standby loss < 5 %; the
  published round-trip efficiency (>90 % large, 85-90 % products) is consistent with
  those two terms, which means *the model must report both*;
* the bed **loses about 50 % of its energy in three months at rest** - the number that
  decides whether a seasonal storage is viable at all;
* other architectures for comparison: resistors *inside* a crushed-rock bed (650 °C,
  ~300 kWh/m³, 10 t modules) and Joule heaters in a refractory brick stack at 1500 °C
  where *radiation* does the transfer (>97 % round-trip, <1 %/day).  Radiation is not
  transferable to a 400-600 °C granular bed, which is why the gas loop is the right
  reference for this machine.

---

## 2. What was replaced

| Old model | New model | Why |
|---|---|---|
| 6×6×5.6 m box of air around the battery, air as a conducting solid | **no air**: the battery wall carries a Robin condition with `h = h_natural + h_wind` | the air is not part of the machine; simulating it costs cells and buys a resistance that is 2.6% of the total (see §4) |
| hairpin electric heaters inside the sand, `W/cm²` check | **resistors in the air loop** (`external_power` of the fluid loop) | matches the reference machine; the bed is heated by the pipes.  *Deleted from the code 2026-09-22*: `HeaterConfig` is the plant's power and its band (`src/core/geometry.py`), `src/core/heaters.py` (bank, hairpin elements, rasteriser, `validate_bank`) is gone, and the `W/cm²` rating moved to the **pipe surface** (`src/core/pipes.py::pipe_surface_power_w_cm2`, reported by the Pipes tab) |
| tubes as lumped convective sinks with the inlet temperature everywhere | **1-D fluid network** with the exact effectiveness relation (`src/solver/fluid.py`) | a storage discharges at large NTU, where the lumped model overstates the extraction by `NTU/(1-e^-NTU)` = 10-25×.  *Deleted from the code 2026-09-22*: `TubeConfig`/`TubeElement`/`TubePattern` and the film-on-tube-cells path are gone, and the GUI's lumped-tube tab with them - the exchanger is on the circuit and the loop's `t_in` is the circuit's return temperature |
| uniform or banded Cartesian grid with a global budget | **adaptive octree** with a 2:1 balance (§5) | the scale range (4 m of sand vs 20 mm of shell) cannot be covered by uniform refinement |

Kept as is: the material database and the packed-bed effective properties, the
finite-volume assembly with the harmonic mean, the implicit time integration and
CG+AMG, the energy-balance verification, the GCI-based reporting.

---

### Pipe bundle geometry (design data used by `src/core/pipes.py`)

The full treatment of the buried-pipe networks - the four layouts, the collection
modes, the parameters, the worked examples and what the tests check - is
**[15_PIPE_NETWORKS.md](15_PIPE_NETWORKS.md)**; this subsection keeps only the design
data that motivated the generator.

Pitches are quoted as multiples of the outer tube diameter: **2.0 d** horizontally,
**2.0-2.5 d** vertically, **sqrt(3) d** for an equilateral (triangular) lattice; the
arrangement is **staggered** (alternate rows shifted by half a horizontal pitch), which
is what the design literature recommends for a bundle in a granular bed.  Published
examples use 18-25 mm tubes, 9 m long, prefabricated in a workshop and lifted in, with
collectors made of flat plates and semi-shells at the tube ends.

Two limits come from the same literature and are checked by the generator:

* **header height <= 3 m**: below 1 m the distribution needs no care, between 1 and 3 m
  it does, above 3 m the bundle must be split into identical parallel modules;
* **bed height <= 4 m** per module in the fluidised-bed reference, which is a reasonable
  cap for a static module too.

The sizing number a designer needs is the **specific wetted area** (m^2 of pipe surface
per m^3 of bed); `BankLayout.specific_area` reports it together with the pitch, the
bundle width and the header height.

## 3. The fluid loop (implemented, `src/solver/fluid.py`)

Geometry in `src/core/pipes.py`: a pipe run is a polyline; `rasterize_pipe` returns the
cells it crosses and the **geometric** length inside each cell, so the wetted area is
`pi d L` and never the staircase surface of the voxel mask.  Invariant:
`sum(area) = pi d L_total` to machine precision (tested).

Physics: 1-D advection with distributed wall exchange, solved exactly piece by piece,

```
m_dot c_p dT/ds = h P (T_wall - T)      ->      T_out = T_wall + (T_in - T_wall) e^-NTU
T_mean = T_wall + (T_in - T_wall)(1 - e^-NTU)/NTU,     NTU = h A /(m_dot c_p)
```

* the power into the solid is `m_dot c_p (T_in - T_out)` per cell, distributed as a
  volumetric source (`q_fluid`, W/m³) - **the same term charges and discharges**;
* the film coefficient is computed from the flow (`pipe_h`: `Nu = 3.66` laminar,
  Dittus-Boelter turbulent, blended in between) unless the caller fixes it;
* the fluid properties follow the temperature (Sutherland viscosity, `rho ~ 1/T`);
* the loop is closed: every temperature is affine in the loop inlet temperature, so the
  balance `sum m_dot_i c_p (T_out,i - T_in) = -Q_ext` is solved for `T_in` in one step.
  `Q_ext > 0` are the resistors (charging), `Q_ext < 0` the exchanger (discharging);
* an impossible operating point (a flow that cannot carry the requested power) is
  **refused with a message**, not returned as a negative temperature.

Verified in `tests/test_fluid.py` (23 collected cases on 2026-09-20 - see
[09](09_TESTING.md) for the command and the caveat that counts move): the geometric
area; the effectiveness relation on a uniform wall; conservation (the enthalpy drop of
the air equals the source term integrated over the bed); the two limits (`NTU -> 0`
area-limited, `NTU -> inf` flow-limited); the lumped model's error factor
`NTU/(1-e^-NTU)`; a closed loop with no external power settling at the wall
temperature; charging/discharging signs and the loop balance; the bank splitting the
flow; the friction factor, the pressure drop, the circulation loss and the refusal of an
impossible point.

### What the numbers say for this machine

8 pipes, d = 50 mm, L = 4 m → A = 5.03 m²; h = 500 W/m²K → hA = 2513 W/K.

| operating point | m_dot | NTU | lumped overstates by |
|---|---|---|---|
| GUI default | 0.1 kg/s | 25 | 25× |
| 24 h discharge of the default battery (52-67 kW) | 0.26 kg/s | 9.7 | 9.7× |
| air with a realistic h = 30 W/m²K, same flow | 0.26 kg/s | 0.58 | 1.4× |

A storage discharges with `NTU >> 1` *by design* (that is what "extract the heat"
means); the flow-limited power `m_dot c_p (T_bed - T_in)` is the physical ceiling.

---

## 4. The outside of the silo

Delete the air region; the battery envelope gets a Robin condition with

```
h_out = h_natural + h_wind
h_natural = Nu_cyl(Ra) k_air / L      (Churchill-Chu, vertical cylinder)
h_wind    = 4 + 4 v                   (ISO 6946 external surface, v = wind speed m/s)
```

with `h_wind` optionally modulated by the windward angle (`h(θ) = h_wind (1 + cos θ)/2`
as a cheap directional split, or a cylinder-in-crossflow correlation if the extra
fidelity is wanted).

Why this is defensible: the film resistance is a small share of the envelope resistance
(`R_film = 0.2` vs `R_insulation = 7.5 m²K/W` → 2.6 %), and a 10× change of `h` moved
the losses by 0.9 % in the current model.  What matters is the **insulation** and the
**thermal bridges** (the pipe penetrations through the shell), which the current model
does not represent at all.

### Status: the capability exists, the wiring from the geometry does not

Implemented and tested **in isolation**:

| piece | where |
|---|---|
| the correlations (`AirProperties`, `rayleigh`, `h_natural_vertical`, `h_natural_horizontal`, `h_wind`, `h_out`) | `src/core/environment.py` |
| the mask and the surface data (`mesh.excluded`, `mesh.h_out`, `mesh.t_ambient`, `mesh.h_contact`) | `src/core/mesh.py` |
| no conduction into excluded cells; the film `h_out·A·(T − T_ambient)` on every active/excluded interface; excluded cells pinned at the ambient so the system stays non-singular | `src/solver/matrix.py` (`face_coefficients`, `build_steady_matrix`, `dirichlet_rows`) |
| the same loss in the audit, so the balance still closes | `src/analysis/fluxes.py::environment_flux`, `analysis/balance.py` |
| a real contact resistance between two materials (`h_contact`), in series with the two half cells | `src/solver/matrix.py::face_coefficients` |
| the physics statements | `tests/test_environment.py` |

**Missing**: nothing fills `mesh.excluded` from a run.  `BatteryGeometry` still paints
the air around the vessel as `MaterialID.AIR` cells and applies convection to the six box
faces (`apply_boundary_conditions`), and no caller translates the painted air into the
excluded mask or sets `h_out`.  A normal simulation therefore still conducts through the
air, and the outside film is never applied.  The test builds the mask by hand, which is
why the capability is verified while the machine does not use it.  Closing this gap is
the step that makes the box of air disappear from the default model.

---

## 5. The mesh: the octree and its status

The user's picture - cubes that grow away from the material interfaces, one level
difference at most between neighbours, a coarse face covered by four fine faces - is the
standard **octree AMR with a 2:1 balance constraint**.

Facts from the literature (see the research notes):

* **p4est** stores the octree as a linear array of leaves, identified by a **Morton
  index** (interleaved bits) plus a level; neighbour finding is integer arithmetic on the
  coordinates, and the ghost layer is built from the "half-size" face/edge/corner
  neighbours, which is exactly why the 2:1 balance is required.  Balance and node
  numbering dominate the mesh cost (>90 % on fractal meshes).
* **Dendro** keeps the same linear-octree structure with hanging-node management in a
  single tree traversal, and reports that applying the Laplacian costs "comparable to a
  direct-indexing regular grid with the same number of elements"; geometric multigrid
  needs ~12 iterations against ~7 of BoomerAMG, with a much cheaper setup - but its own
  authors state it "is not robust in the presence of discontinuous coefficients, in
  contrast to AMG".  **Our domain is exactly discontinuous coefficients**, so AMG stays.
* **Afivo** uses a *block* octree (leaves are N^D boxes, N ≥ 8) with a 2-cell buffer and
  a 2:1 rule that includes diagonal neighbours; the block form is what makes the stencil
  vectorised, which matters in NumPy.
* **BoxLib/AMReX/Chombo** use level-based patches with *flux registers* to correct the
  fine-coarse boundary flux - the classical conservative treatment.

### What exists now (`src/core/octree.py`, `tests/test_octree.py`)

A **cell-level** linear octree, in the p4est sense: a leaf is an integer
`(level, x, y, z)` with the coordinates counted in finest cells, `level = log2` of the
edge length in those cells, and a Morton code (21 bits per coordinate, so the round trip
is exact for a mixed-level leaf) giving a canonical order.  The leaf list is indexed per
level by the level-normalised corner, so a neighbour query is a couple of dictionary
lookups.

* the **2:1 balance** is enforced after every refine/coarsen: a coarse face is shared with
  at most four finer faces;
* the **face list is conservative**: a coarse face against `n` finer leaves comes back as
  those `n` sub-faces with the fine area, an equal pair as one entry, and each pair is
  listed once - so a flux appears with opposite signs in the two cells that share it and
  the discrete balance closes to machine precision on any combination of levels;
* the assembly reuses the harmonic mean and the `k A/(d_centers V)` coefficient of the
  structured solver, so an octree mesh and a graded Cartesian mesh agree where their cells
  agree;
* `refine_by_gradient` provides the a posteriori indicator;
* the tests solve 1-D conduction and a two-material wall against their analytic answers,
  check the conservative face list, and pin the cost budget: a uniform tree of 32768
  leaves and its 95232 faces in about half a second.

### What is still open

* **It is not the mesh of the main solver.**  `Mesh3D`, `SteadyStateSolver` and
  `TransientSolver` still run on the structured (uniform or graded) grid.
  `src/solver/octree_solver.py` now puts the octree on the *physics* path (steady
  conduction on the leaf list, equivalence with the structured solver on a matching
  uniform grid, a conservative flux report, an objective-driven refinement cycle), but
  no caller routes a `BatteryGeometry` into it: adopting it as the production mesh is
  still a migration, not a switch.
* **The transient** has no octree driver at all: only the steady problem is solved.
* **Block-based leaves** (Afivo-style) remain the planned optimisation if a *refined*
  tree of that size has to go faster: the balance rounds, not the face list, are what
  cost seconds there, and blocks change no formula in the module.
* The **graded Cartesian path** (`src/core/refinement.py`) stays the production mesh in
  the meantime: it is what the GUI builds, what the automatic mesh search refines
  ([09](09_TESTING.md), [10](10_MESH_AND_HEATERS.md)).

The a priori plan of `src/analysis/mesh_plan.py` (`thickness/N` per layer, `2k/h` at
convective surfaces) plus the GCI reporting of `src/analysis/convergence.py` are the two
other pieces of the refinement story and both exist.  The mesh-loop side is now wired:
the flux-jump indicator of `src/core/octree.py` and `refine_on_objective` in
`src/solver/octree_solver.py` refine a tree on a physical objective, and they reuse
`analysis.convergence`'s `ConvergenceTarget`/`ConvergenceLevel`, so an adaptive run and
the structured search report the same quantities.  What is missing is the *connection*:
`SteadyStateSolver` does not call any of it, and the octree has no transient driver.

---

## 5b. The plant, as it is wired (2026-09-22)

The GUI now shows the architecture of §1 and not the one of the old code:

* **Heaters** is the *gas circuit*: the total electric power, the gas (air / nitrogen /
  steam), the mass flow, the circuit pressure, the temperature the gas comes back to the
  resistors at ("auto" lets the loop solve its own balance, which is the closed circuit
  of a charge), the blower efficiency and the wall roughness.  Its summary is computed on
  the built network: the power over the wetted surface [W/cm²] and the power per riser.
* **Pipes** is the buried network and only that - one tab, because the network is the
  exchange path of *both* phases (gas up the risers and hot collection at the top while
  charging; flow reversed, heat to the exchanger while discharging).
* **`Build mesh` paints the network by default**, so a run started from the Analysis tab
  uses the gas loop on the very cells it solves; the Pipes tab's button rebuilds it after
  a change of the plumbing.
* **The mesh covers the active model only.**  The refinement regions are 3-D boxes
  (`src/analysis/mesh_plan.py::active_regions` → `region_bands`): the sand, the insulation
  ring and the two slabs, the shell, the casing the ambient film sits on and the bundle of
  the pipes.  The air around the vessel is *not* one of them: `apply_environment` excludes
  it and holds it at the ambient temperature, so a leaf out there carries no flux.  The
  graded road keeps per-axis bands and gets one **cap** band at the coarsest active target
  instead of a "far field".  Measured on the default model (tree road, same commands as
  §8): **123 740 leaves before → 143 396 after**, of which the excluded air went from
  **78 584 leaves to 45 232** and the share of the box left at the tree's own coarse level
  went from **3.6 % to 26.8 %**; the build of the a priori tree takes ~11 s against ~13-22 s
  for the old band set, and the paint ~0.7 s against ~0.5 s.  The leaf count is up because
  the bundle is now refined to the tree's floor instead of the air being refined
  everywhere - the cell budget still bounds it, and the flat corners an axis-aligned box
  leaves around a cylinder are what the remainder of the waste is.
* **The 3D preview draws the network** (`src/viz/scene.py::_add_pipe_network`) as one tube
  actor built from the centrelines, so the pipes that carry the heat are visible next to
  the vessel they are buried in.

## 5c. Every analysis runs the plant (2026-09-23)

* **The coupling is implicit and exact.**  The loop hands every pipe cell its exact
  exchange as a film `G (T_gas - T_wall)`, `G = m_dot c_p (1 - e^-NTU)`, and the loop
  inlet is solved with the field, so a step (or a steady sweep) deposits exactly the
  external power ([12](12_METHODS.md) §11).  The explicit coupling it replaces was the
  source of the sub-100 K beds of the earlier cycle runs.
* **Steady and losses couple the loop** (`SteadyStateSolver(fluid_loop=...)`): the power
  enters through the pipe walls.  The fixed pipe film of the paint is gone.
* **The GUI shows the plant only**: octree mesh, Gas circuit, Pipes, an exchanger with a
  set power or a set return temperature; the controls of the old models are removed
  (list in `CHANGELOG.md`).
* **The linear solver is CG + AMG Ruge-Stuben** everywhere in the GUI (measurements in
  [12](12_METHODS.md) §3).

## 6. Migration order

| # | Step | State |
|---|---|---|
| 1 | fluid loop (`src/solver/fluid.py`, `src/core/pipes.py`) | **done**, verified by `tests/test_fluid.py` |
| 2 | wall BC without the air domain (`h_natural + h_wind`) | **capability done, wiring open**: correlations, `mesh.excluded`/`h_out`/`h_contact`, the assembly, the audit and the tests exist (§4); `BatteryGeometry` still paints the air box and fills neither mask nor film |
| 3 | pipe layouts matching the reference (vertical banks, top-down charge, bottom-up discharge) | **done and extended** (spiral layout, wall thickness/material, roughness, insulated headers, `paint`, hydraulics, the `FluidLoop` built from the network): `src/core/pipes.py`, `src/core/pipe_network.py`, `tests/test_pipes.py`; [15](15_PIPE_NETWORKS.md) is the authority |
| 4 | octree mesh with the 2:1 balance and a conservative fine-coarse flux | **core + physics done, not the production mesh**: `src/core/octree.py` and `src/solver/octree_solver.py` solve analytic cases and refine on an objective, but `Mesh3D`/`SteadyStateSolver` still use the structured grid (§5) |
| 5 | a posteriori refinement loop + GCI reporting | **done as pieces**: the GCI report (`src/analysis/convergence.py`) and the flux-jump indicator with `refine_on_objective` (`src/core/octree.py`, `src/solver/octree_solver.py`) exist on separate paths - the *structured* search still chooses from the a priori plan and the Richardson prediction, and nothing connects the indicator to it |
| 6 | losses with the real standby paths (pipe penetrations, foundation) | **open** (§7 item 1) |

## 7. Open decisions

1. **Insulation and bridges**: the reference machine insulates with steel + refractory
   between two shells and the pipes penetrate the top; the standby loss over months is
   dominated by those paths, and the current model has none of them.
2. **Layered operation**: the measured core/edge profile (500/150-200 °C) suggests the
   machine is deliberately run with a strong radial gradient; the model should report
   the *usable* energy above the delivery temperature, not the total stored energy.
3. **Sand properties**: the `silica_sand` database value is a bulk value used as a solid
   phase in the geometric mean (see [12_METHODS.md](12_METHODS.md) §8); with the new
   architecture this must be settled because the bed conductivity now controls the
   charge/discharge time.

## 8. Status of the redesign (2026-09-20)

Every row is a measurement or a file reference, not a plan.  The test counts were read
with `python -m pytest tests/ --collect-only -q` on this working tree on 2026-09-20 and
they move while work is in flight (see [09](09_TESTING.md) §1): treat them as a reading,
not a specification.

| piece | state | evidence |
|---|---|---|
| gas loop: effectiveness relation, closed loop, film coefficient, pressure, blower, circulation loss | done | `src/solver/fluid.py`; `tests/test_fluid.py`, 23 collected cases |
| pipe geometry and networks: runs, geometric wetted area, layouts (incl. spiral), four collection modes, wall material/roughness, insulated headers, `paint`, hydraulics, branch split, the `FluidLoop` built from the network | done | `src/core/pipes.py`, `src/core/pipe_network.py`; `tests/test_pipes.py`, 60 collected cases; [15](15_PIPE_NETWORKS.md) |
| environment: no air domain, natural + wind film, contact resistance | capability done, **not wired to the geometry**: `BatteryGeometry` still paints the air and sets box-face convection | `src/core/environment.py`, `Mesh3D.excluded/h_out/h_contact`, `src/solver/matrix.py`, `src/analysis/fluxes.py`; `tests/test_environment.py`, 9 collected cases |
| adaptive mesh: 2:1 octree with a conservative face list, its Laplacian, the flux-jump indicator and an objective-driven refinement; steady solve on the leaf list | core and physics done, **not the production mesh** (no transient driver, no route from `BatteryGeometry`) | `src/core/octree.py`, `src/solver/octree_solver.py`; `tests/test_octree.py` (14) and `tests/test_octree_solver.py` (12) |
| cycle accounting: charge / standby / discharge through the loop, per-step stops, energy decomposition | done | `src/analysis/cycle.py`; `tests/test_cycle.py`, 10 collected cases |
| graded Cartesian mesh + automatic mesh search | done, and it is the production path | `src/core/refinement.py`, `src/analysis/convergence.py`, `src/analysis/mesh_plan.py`; 10 + 9 collected cases |
| GUI: the circuit tab, the merged Pipes tab, the automatic-mesh button, the region targets | in place | `gui/views/geometry_panel.py`, `gui/main_window.py`; widget defaults measured in [06](06_GUI_CONFIGURATION.md); screenshots `docs/figures/gui_tab_circuit.png`, `gui_tab_pipes.png`, `gui_tab_mesh.png`, `gui_window_after_build.png` |
| plant wiring: the network is painted by `Build mesh` and the circuit's settings drive the loop | in place | `gui/main_window.py::paint_pipe_network`, `gui/controller.py::_fluid_loop`, `RunConfig.pipe_fluid/pipe_pressure/pipe_inlet/pipe_fan_efficiency` |
| mesh plan by active region (sand, insulation, shell, tube wall; no air) | in place | `src/analysis/mesh_plan.py::active_regions/region_bands`; `tests/test_mesh_plan.py` |
| figures: one script regenerates them all | done | `scripts/figures.py`, seven files in `docs/figures/` (list in [README](../README.md#-figures)) |

Measured numbers worth keeping in view (all readings of 2026-09-20, re-run the commands
before quoting them):

* the **suite** collects 311 cases (300 without the GUI sweep); the same command said
  295/284 and 270/259 earlier in the same session, because the octree solver, the pipe
  extension and the cycle completion landed in between ([09](09_TESTING.md) §1);
* the octree builds a uniform tree of **32768 leaves and 95232 faces in about half a
  second** (the budget `tests/test_octree.py` pins);
* the graded mesher reaches the target cell size in a band while keeping the neighbour
  ratio within `growth` (`tests/test_refinement.py`), and a uniform `GridSpec`
  reproduces the legacy uniform mesh bit for bit (`tests/test_graded_mesh.py`);
* the **mesh of the default model**: 143 396 leaves (tree) of which 57.8 % of the volume is
  the excluded air held coarse by the region bands, and 349 496 cells on the graded road -
  the measurements and the before/after are in §5b;
* the **automatic mesh search** on the default geometry does not reach the 2 K default
  tolerance inside the 400 000-cell default budget, and says so instead of pretending
  (`tests/test_convergence.py`, and the GUI reports "stopped" with the limit);
* the **energy balance closes to machine precision** on the battery, on a graded grid,
  on the octree face list and on the network-built loop (separate tests, separate
  implementations of the same identity).

The open items, in the order they should be taken, are: (1) fill `mesh.excluded`/`h_out`
from `BatteryGeometry` so the air box disappears from a normal run (§4); (2) route the
geometry into the octree path (`refine_on_objective` exists; `SteadyStateSolver` does not
call it) and give the octree a transient driver if the adaptive mesh is to become the
production mesh (§5); (3) connect the flux-jump indicator to the structured mesh search
of step 5 in §6; (4) the standby paths of §7 item 1.
