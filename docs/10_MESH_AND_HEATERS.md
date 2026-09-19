# 10. Graded mesh and realistic heater elements

Design and migration plan.  **Parts 1-3 are implemented and tested** (163 tests);
part 4 (the optimisation targets) is the next work item.

---

## Part 1 — Mesh refinement core (done)

`src/core/refinement.py`, `tests/test_refinement.py`, `tests/test_graded_mesh.py`.

The user picks **physical targets**, not a cell count:

| target | meaning |
|---|---|
| `Band(start, end, target)` | inside this slice of an axis the cells should be `target` [m] |
| `growth` | largest allowed size change between neighbouring cells (default 1.3) |
| `min_size`, `max_size` | floors/ceilings on the realised cell size (optional) |
| `max_cells` | budget: if the request does not fit, every target is scaled by one factor |

How the grid is built:

1. **size field** - `h(x) = min_k clip(h_k + (growth-1)*dist(x, band_k), min, max)`.
   The ramp is *linear in the distance*, which is what makes the per-cell ratio
   ``≤ growth`` by construction (an exponential ramp reaches the coarse size sooner
   but its per-cell ratio grows with the cell size).  The field is a plain function of
   the coordinate, so a request and its mirror image give mirrored grids.
2. **band partition** - overlapping bands are split so every declared boundary stays
   on a grid line (the geometry masks never split a cell).
3. **density equidistribution** - inside each band the cells are placed where the
   cumulative density ``1/h`` crosses an integer, so the local size follows the field
   and a constant field gives an exactly uniform grid; the remainder is absorbed by the
   last cell of the band (never a sliver).
4. **budget** - the analytic count decides whether the budget is an issue; if it is,
   the targets are scaled by one common factor (coarse first, then walked back towards
   the request while it still fits) and the finest grid that fits is returned.

Verified properties (`tests/test_refinement.py`): uniform request → exactly uniform
grid; exact endpoints; boundaries on grid lines; fine band really fine; bounded growth;
no slivers; budget respected; size lookup for element masks; **symmetric request →
symmetric grid**.

## Part 2 — Graded grid in the solver

### Why

The current mesh is uniform: refining near the heaters refines the whole domain,
and the "cell size" spin box is a blunt instrument (the same mesh is used in the
sand, in the insulation and in the far field).  A graded grid puts cells where the
gradients are — heater sheath, insulation thickness, roof — and leaves the far
field coarse.  That is also what makes realistic heater elements representable at
all (a 12 mm sheath is invisible in a 200 mm cell).

### What changes

| item | now | after |
|---|---|---|
| `Mesh3D.spacing` | scalar | `GridSpec` (bands + growth + budget) or a scalar for the legacy uniform case |
| `mesh.dx/dy/dz` | scalars | arrays `(Nx,)`, `(Ny,)`, `(Nz,)` |
| `mesh.x/y/z` | linspace centres | centres of the graded edges (X, Y, Z unchanged in meaning) |
| `mesh.d` | scalar | removed (call sites use the local size) |
| `mesh.V_cell` | scalar | array `(Nx,Ny,Nz)` |
| `mesh.A_cell` | scalar | `mesh.area("x"/"y"/"z")` per-face areas |
| `mesh.snapped` | Ly/Lz correction | unchanged (the graded grid still ends at the box faces) |

### The discrete coefficients

For cells `P` (size `dx[i]`) and `E` (size `dx[i+1]`) at distance
`d_centers = (dx[i] + dx[i+1])/2`, the face conductance per unit area is the
series of the two half cells, and the per-volume coefficient follows from
`A_x = dy[j]·dz[k]`, `V = dx[i]·dy[j]·dz[k]`:

$$k_{face} = \frac{2\,k_P k_E}{k_P + k_E}, \qquad
a_E = \frac{k_{face}\,A_x}{d_{centers}\,V}, \qquad d_{centers} = \frac{dx_i + dx_{i+1}}{2}$$

(``k_face`` is the harmonic mean of the two *conductivities*, so ``k_face A/d`` is the
conductance of the interface; writing it as ``1/(...)`` — as the first draft of this
document did — is a resistance per unit area and needs no second division by
``d_centers``.)

For a uniform grid this collapses to `2 k_P k_E/(k_P+k_E)/Δ²` — the current
formula, so `tests/test_solver.py` must stay green unchanged on uniform meshes.

Boundary faces keep their half-cell treatment with the *local* first-cell size
(`h_eff = 2kh/(2k + h·dx_0)`), Neumann adds `q''/dx_0`, interior tube convection
uses `h/dx_i`.  Flux integrals use the per-face areas, so the balance closure
identity is preserved.

### Call sites migrated (30)

`src/core/mesh.py` (definition), `src/solver/matrix.py` (coefficients, ~6 sites),
`src/solver/transient.py` (source/sink densities, 3), `src/analysis/fluxes.py`
(6), `src/analysis/balance.py` (2), `src/analysis/losses.py` (1),
`src/core/geometry.py` (element masks and power densities, 5), `src/io/state.py`
(hash: store the spacing arrays), `src/viz/scene.py` (ImageData spacing, 1),
`gui/main_window.py` (log line, 1).

### GUI (done)

*Geometry → Mesh* now has a **Refined mesh** switch and the targets below; the legacy
*Cell size* box stays for the uniform mode and every target is scaled to the cell
budget.  The realised grid is summarised live (cells per axis, size range, worst
neighbour ratio, memory, and whether the budget had to coarsen the targets).

The targets are:

* **Refinement targets**: cells across the storage radius, cells across the
  insulation thickness, cells across the heater sheath (feeds from the heater
  panel), max cell size in the air/far field, growth ratio, cell budget.
* A **read-only summary** computed from those targets: cells per axis, total,
  smallest/largest cell, worst neighbour ratio, estimated memory.

### Tests to add

* uniform `GridSpec` reproduces the current mesh and the analytic results
  bit-for-bit;
* a graded 1-D slab reproduces the analytic profile with the same order of
  accuracy as the uniform grid at the same cell count in the ramp region;
* global refinement (halving every target) converges to the same answer → the
  "results must not depend on the resolution" requirement;
* the balance identity still closes on a graded grid.

---

## Part 3 — Realistic heater elements

### What the reference design is

`photo/heating_elements_3D.png` and the literature (KTH/Aalto packed-bed sand
experiments; NREL's "toaster" concept; commercial sand-battery modules) show the
same solution: a **flanged immersion heater bank** —

* a **mounting flange** on the roof with PG/gland entries;
* **U-shaped (hairpin) tubular elements** welded to the flange in rows, hanging
  down into the sand, supported at their lower end by a **support plate** and
  **support rods**;
* sheathed resistance wire inside a metal tube (Incoloy/stainless), typically
  Ø8–16 mm, with an **active length** in the sand and a cold shank through the
  insulation and the air gap;
* power set by the *surface power density* (3–8 W/cm² for sheathed elements in
  solids), not by an arbitrary volumetric density.

### The model (implemented in `src/core/heaters.py`)

```
HairpinElement (implemented)
  ├─ sheath: outer diameter d_s, wall thickness, material
  ├─ leg spacing: c/c distance between the two legs
  ├─ bend radius at the bottom
  ├─ active length (in the sand) + cold shank length (in the insulation/air)
  └─ rated power [W], surface power = P / (π d_s L_active) [W/cm²]
HeaterBank
  ├─ flange elevation and diameter
  ├─ rows × columns, pitch row/col (or a ring pattern for a cylindrical unit)
  ├─ power per element / total
  └─ support plate elevation
```

Thermal model per element:

* the cells intersected by the *sheath* take the sheath material (steel) —
  resolved because the graded mesh targets ~the sheath diameter in the heater
  region;
* the **rated power is deposited in those cells** (`Q = P/(n_cells V_cell)`), so
  the total is exact whatever the discretisation (`source_mask` unchanged);
* the sheath-to-sand contact resistance is automatic through the half-cell series
  conductance of the fine cells around it;
* inside-wire → sheath conduction is lumped (documented), as is the axial
  conduction along the cold shank.

Discretisation (`discretize_bank(mesh, bank, geometry)`):

1. compute every hairpin's two legs and bend as segments;
2. rasterise each segment: walk it in steps of the local cell size
   (`refinement.size_at`) and mark the containing cells;
3. reject the configuration when it cannot be represented:
   * an element covers no cell,
   * adjacent elements share cells (spacing < sheath size + 2 cells),
   * the surface power density exceeds the material limit,
   * a hairpin collides with the storage wall or with a tube;
4. report per-element power, surface power, cells per element and the minimum
   cells across the sheath.

### GUI

*Heaters* tab becomes: element type (hairpin / straight rod / coil), sheath
diameter and material, active length, leg spacing, rows × columns, flange
elevation, total power with the resulting surface power shown live, plus the
existing offsets.  The preview draws the hairpins (flange, support plate, legs,
bends) instead of simple rods, and the element list shows the surface power per
element with a warning when it exceeds the limit.

### Tests

* power conservation: Σ Q V equals the rated power for every layout;
* a hairpin's cells form two legs and a bend, all inside the storage band;
* minimum spacing and collision validations actually reject bad layouts;
* the surface power limit is enforced;
* results are mesh-independent: refining the refinement targets changes the
  answer by less than the convergence tolerance.

---

## Part 4 — Optimisation targets after the two parts above

| item | measure |
|---|---|
| graded mesh | same accuracy with fewer cells than a uniform grid (report in `scripts/benchmark.py`: cells vs error on the analytic slab) |
| assembly | the graded coefficients must not slow the assembly down by more than ~10 % |
| transient | warm start + cached preconditioner already dominate; keep the operator rebuild tied to a *changed* `dt` only |
| heater rasterisation | O(cells along the elements), done once per build |
