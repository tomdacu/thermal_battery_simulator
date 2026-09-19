# Redesign: the sand battery as it is actually built

This document records the redesign decided after studying the reference machine (Polar
Night Energy, Finland) and the physics of a storage discharge.  It supersedes the parts
of the previous design that assumed a box of air around the battery, hairpin heaters
inside the sand and tubes as lumped convective sinks.

Read with [12_METHODS.md](12_METHODS.md) (why the numerical choices are what they are).

---

## 1. The reference machine (published facts)

| Fact | Value | Source |
|---|---|---|
| Storage medium | sand, or sand-like by-products (crushed soapstone) | polarnightenergy.com/sand-battery |
| Temperature | storage up to **600 °C**; output 60-400 °C | polarnightenergy.com/sand-battery |
| **Charging** | **electric resistors heat air**, the air circulates in a **closed loop through pipes buried in the bed** | polarnightenergy.com/sand-battery (FAQ) |
| **Discharging** | flow reversed: cool air is blown through the pipes, leaves hot and goes to an **air-to-water exchanger or a steam generator** | polarnightenergy.com/sand-battery (FAQ) |
| Direction | patent: **charge top→bottom** (favours natural convection, cuts fan power), **discharge bottom→top** | patent US20220146205A1 |
| Container | insulated steel silo, 10-15 m tall, 4-30 m diameter; **no air domain** - air lives in pipes and ducts only | polarnightenergy.com/sand-battery |
| Products | 2 MW / up to 200 MWh, round-trip **85%**, 15×12 m; 10 MW / up to 1000 MWh, **90%**, 30×12 m | polarnightenergy.com/sand-battery |
| Kankaanpää 2022 | 200 kW, 8 MWh, 4 m diameter × 7 m, ~100 t sand, efficiency 60-75% | polarnightenergy.com/reference/worlds-first-sand-battery |
| Pornainen 2025 | 1 MW, 100 MWh, 13 m × 15 m, 2 000 t crushed soapstone | polarnightenergy.com/news/what-is-a-sand-battery |
| Measured profile | **core ~500 °C, edge 150-200 °C** - a strong, permanent radial gradient with a cold shell | vatajankoski.fi/en/projects/sand-battery |
| Retention | heat for months; 20-200 cycles/year | polarnightenergy.com/sand-battery |

Consequences for the simulator, in one line each:

1. the heat transfer fluid is **air in pipes**, not a fluid in "tubes" with a fixed inlet
   temperature, and the pipes are the *only* exchange surface;
2. the **resistors heat the air**, not the bed: charging and discharging are the same
   loop with opposite external power;
3. there is **no air region** to simulate: the outside of the silo is an insulated wall
   with a natural-convection + wind boundary condition;
4. the bed works with a **strong radial gradient** (core 500 °C, edge 150-200 °C), so the
   mesh must resolve the shell/insulation, not a thin sheath in the middle.

### The gas loop in detail (patent US20220146205A1 / granted US12104855B2)

* the pipes are **stainless steel** (304L/304H/316L/316H/321H/330), arranged
  **vertically** in the module (horizontal or inclined are claimed as alternatives);
* the gas is preferably **inert (nitrogen)** at **up to 50 bar**, and the patent states
  the reason explicitly: *"pressurization may be utilized to increase the heat transfer
  rate without increasing the gas velocity excessively"*.  That is a design lever the
  model should expose: at constant velocity `Re ~ rho`, so `h` rises with pressure while
  the fan loss stays put - the cheap way to cut the circulation loss;
* the fan is bidirectional, rated to 1200 °C, with its **motor outside the loop**;
* **charge top->bottom** (aids natural convection, cuts fan power), **discharge
  bottom->top**; the gas leaves the bed close to the bed temperature and goes to a
  **counter-flow** external exchanger, leaving it at **40-70 °C** - that is the loop's
  cold-side temperature, i.e. the `t_in` of a discharge in the model;
* a module is at most ~100 m³, underground, insulated with expanded clay (LECA,
  `lambda < 0.3 W/(m K)`);
* alternative discharge path claimed: **thermosiphon tubes** (sealed, water fill <= 25%
  of the tube volume, dry-up allowed, superheated steam, 45-90° inclination, up to 15 m
  long, example 9 mm inner diameter at 200 bar / 500 °C).  Passive, no fan, but a
  two-phase pressurised circuit: not modelled here, listed as an alternative.

The architecture is therefore **air inside pipes**, not a distributed flow through the
bed: the distributed-flow family (Siemens Gamesa ETES, CSP rock beds) is a different
machine and is not what this simulator should imitate.

### From the patent (US20220146205A1 / US12104855B2) and the efficiency note

* the resistor sits **in the gas circuit** (a heater in an insulated vessel), never in
  the bed; the gas leaves it at 700-1000 °C (up to 2000 °C claimed);
* **charge top→bottom, discharge bottom→top** with a bidirectional fan; the fan motor is
  *outside* the loop;
* the gas may be an **inert gas (N2) at up to 50 bar**: pressure raises the film
  coefficient without raising the velocity, which is the cheap way to cut the fan loss;
* bed temperature preferably 300-500 °C, insulation with `lambda < 0.3 W/(m K)`
  (expanded clay / LECA is named);
* **circulation losses (fan + ducts) are ~5% per cycle**, standby loss < 5%; the
  published round-trip efficiency (>90% large, 85-90% products) is consistent with those
  two terms, which means *the model must report both*;
* Pornainen's implied charge power density is ~1.1 kW/m³ of bed, and the bed **loses
  about 50% of its energy in three months at rest** - the number that decides whether a
  seasonal storage is viable at all;
* other architectures for comparison: **Brenmiller bGen** puts the resistors *inside* a
  crushed-rock bed (650 °C, ~300 kWh/m³, 10 t modules); **Rondo** uses Joule heaters in
  a refractory brick stack at 1500 °C where *radiation* does the transfer (>97% RTE,
  <1%/day).  Radiation is not transferable to a 400-600 °C granular bed, which is why the
  air loop is the right reference for this machine.

---

## 2. What was replaced

| Old model | New model | Why |
|---|---|---|
| 6×6×5.6 m box of air around the battery, air as a conducting solid | **no air**: the battery wall carries a Robin condition with `h = h_natural + h_wind` | the air is not part of the machine; simulating it costs cells and buys a resistance that is 2.6% of the total (see §4) |
| hairpin electric heaters inside the sand, `W/cm²` check | **resistors in the air loop** (`external_power` of the fluid loop) | matches the reference machine; the bed is heated by the pipes |
| tubes as lumped convective sinks with the inlet temperature everywhere | **1-D fluid network** with the exact effectiveness relation (`src/solver/fluid.py`) | a storage discharges at large NTU, where the lumped model overstates the extraction by `NTU/(1-e^-NTU)` = 10-25× |
| uniform or banded Cartesian grid with a global budget | **adaptive octree** with a 2:1 balance (§5) | the scale range (4 m of sand vs 20 mm of shell) cannot be covered by uniform refinement |

Kept as is: the material database and the packed-bed effective properties, the
finite-volume assembly with the harmonic mean, the implicit time integration and
CG+AMG, the energy-balance verification, the GCI-based reporting.

---

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

Verified in `tests/test_fluid.py` (13 tests): the geometric area; the effectiveness
relation on a uniform wall; conservation (the enthalpy drop of the air equals the
source term integrated over the bed); the two limits (`NTU -> 0` area-limited,
`NTU -> inf` flow-limited); the lumped model's error factor `NTU/(1-e^-NTU)`; a closed
loop with no external power settling at the wall temperature; charging/discharging
signs and the loop balance; the bank splitting the flow; the refusal of an impossible
point.

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

## 4. The outside of the silo (planned)

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
(`R_film = 0.2` vs `R_insulation = 7.5 m²K/W` → 2.6%), and a 10× change of `h` moved the
losses by 0.9% in the current model.  What matters is the **insulation** and the
**thermal bridges** (the pipe penetrations through the shell), which the current model
does not represent at all.

Implementation: the assembly must accept a **Robin face between an active cell and an
inactive cell** (the excluded air), with `h_out` and `T_ambient`; inactive cells are
dropped from the linear system.  This is the same machinery as the current per-face BC,
extended to internal surfaces.

---

## 5. The mesh (planned, research done)

The user's picture - cubes that grow away from the material interfaces, one level
difference at most between neighbours, a coarse face covered by four fine faces - is the
standard **octree AMR with a 2:1 balance constraint**.

Facts from the literature (see the research notes):

* **p4est** stores the octree as a linear array of leaves, 24 bytes per octant,
  identified by a **Morton index** (interleaved bits) plus a level; neighbour finding is
  integer arithmetic on the coordinates, and the ghost layer is built from the
  "half-size" face/edge/corner neighbours, which is exactly why the 2:1 balance is
  required.  Balance and node numbering dominate the mesh cost (>90% on fractal meshes).
* **Dendro** keeps the same linear-octree structure with hanging-node management in a
  single tree traversal, 1 byte per octant, and reports that applying the Laplacian costs
  "comparable to a direct-indexing regular grid with the same number of elements";
  geometric multigrid needs ~12 iterations against ~7 of BoomerAMG, with a much cheaper
  setup - but its own authors state it "is not robust in the presence of discontinuous
  coefficients, in contrast to AMG".  **Our domain is exactly discontinuous
  coefficients**, so AMG stays.
* **Afivo** uses a *block* octree (leaves are N^D boxes, N ≥ 8) with a 2-cell buffer and
  a 2:1 rule that includes diagonal neighbours; the block form is what makes the stencil
  vectorised, which matters in NumPy.
* **BoxLib/AMReX/Chombo** use level-based patches with *flux registers* to correct the
  fine-coarse boundary flux - the classical conservative treatment.

Decision for this codebase:

1. **block-based octree** (Afivo-style: leaves are 8³ boxes, so every leaf is a dense
   NumPy block and the FV stencil stays vectorised);
2. **2:1 balance** with the buffer, enforced after every refine/coarsen;
3. **conservative flux at a fine-coarse face**: the coarse face is decomposed into its
   four fine sub-faces and each carries its own conductance (the "flux register"
   approach); the *hanging nodes* on the coarse side are eliminated by constrained
   interpolation, which keeps the operator symmetric → **CG+AMG keep working**;
4. refinement criterion: **a priori** from the physics (the plan of
   `src/analysis/mesh_plan.py`: `thickness/N` per layer, `2k/h` at convective surfaces,
   the pipe spacing) plus an **a posteriori** local truncation error estimate (apply the
   discrete operator to the interpolated field: `tau = |A T - b|` locally) to catch what
   the a priori rule misses;
5. convergence is reported with the GCI of `src/analysis/convergence.py`, which already
   works on any sequence of grids and therefore on refined/coarsened ones.

Verification planned: an analytic composite wall (exact flux), a spherical source in an
infinite medium (exact radial profile), and the invariant that the energy balance still
closes to machine precision on an unbalanced-then-balanced octree.

---

## 6. Migration order

| # | Step | Verification |
|---|---|---|
| 1 | fluid loop (`src/solver/fluid.py`, `src/core/pipes.py`) | **done** - 13 tests |
| 2 | wall BC without the air domain (`h_natural + h_wind`) | analytic wall with a film; losses insensitive to `h`; balance closes |
| 3 | pipe layouts matching the reference (vertical banks, top-down charge, bottom-up discharge) | area invariant; loop balance |
| 4 | block-octree mesh with 2:1 + conservative fine-coarse flux | composite wall and radial source against the analytic solution |
| 5 | a posteriori refinement loop + GCI reporting | the mesh stops refining when the answer stops moving |
| 6 | losses with the real standby paths (pipe penetrations, foundation) | seasonal loss per day against the published efficiency |

---

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
