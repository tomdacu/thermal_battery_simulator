# 10. Graded mesh and heater elements

> The GUI's mesh is now the anisotropic tree of boxes ([18](18_SOLVER.md) §1,
> [16](16_ADAPTIVE_MESH_MIGRATION.md) §8).  This document describes the graded Cartesian
> mesher and the heater bank, which stay as the references of the tests.

Status of this work stream, as of 2026-09-20.  **Parts 1-3 are implemented and
tested; part 4 (the optimisation measurements) is open.**  The numbers are
measurements: `python -m pytest tests/ --collect-only -q` for the counts
([09](09_TESTING.md) records the command and the date next to them), and everything
quoted as a property is a test in the file named next to it.

| part | state | where |
|---|---|---|
| 1. refinement core (targets → graded grid) | implemented | `src/core/refinement.py`, `tests/test_refinement.py` |
| 2. graded grid inside the solver | implemented | `src/core/mesh.py`, `src/solver/matrix.py`, `tests/test_graded_mesh.py` |
| 3. hairpin heater bank | implemented | `src/core/heaters.py`, `tests/test_heaters.py` |
| 4. optimisation measurements | **open** | `scripts/benchmark.py` still measures uniform meshes only |

The design path of the machine itself moved during this work: the reference
architecture puts the **resistors in the gas circuit** and heats the bed through
**pipes buried in the granular medium** ([13](13_REDESIGN.md), [15](15_PIPE_NETWORKS.md)).
The graded mesh and the mesher are what both paths need, so they stay central; the
hairpin bank stays as a working, validated option for electric heating inside the
bed, but it is no longer the design this simulator is built around.

---

## Part 1 — Mesh refinement core (done)

`src/core/refinement.py`, `tests/test_refinement.py` (10 collected cases).

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

Verified properties (`tests/test_refinement.py`): a uniform request → exactly uniform
grid; exact endpoints; boundaries on grid lines; a fine band really is fine; bounded
growth; no slivers; the budget is respected by scaling the targets; the size lookup
used to size element masks answers anywhere; degenerate requests are rejected or
degraded gracefully; a symmetric request gives a **symmetric grid** (the old
directional walk refined one side more than the other).

## Part 2 — Graded grid in the solver (done)

`Mesh3D` takes either a scalar `spacing` (uniform) or a `GridSpec` (graded):
`dx/dy/dz` are per-axis arrays, `mesh.V` the cell volumes and `mesh.Ax/Ay/Az` the face
areas.  On a uniform grid `d`, `V_cell` and `A_cell` still exist and raise on a graded
mesh, so a caller cannot silently assume equal cells.

The face coefficient is the same per-volume form for both paths, with the geometry
cached once per mesh in `GridIndex.face_factors` (`= A/(d_centers·V)`):

$$k_{face} = \frac{2\,k_P k_E}{k_P + k_E}, \qquad
a_E = k_{face}\,\frac{A_x}{d_{centers}\,V}, \qquad d_{centers} = \frac{dx_i + dx_{i+1}}{2}$$

Each of these is a measured property of the implementation:

| what | checked by |
|---|---|
| a uniform `GridSpec` reproduces the legacy uniform mesh | `test_uniform_gridspec_reproduces_the_legacy_uniform_mesh` |
| a linear profile is exact on a graded grid, and the interface flux matches the series resistance | `test_a_linear_profile_is_exact_on_a_graded_grid`, `test_the_interface_flux_matches_the_series_resistance` |
| the answer does not move when the grid changes (resolution independence) | `test_a_graded_mesh_reproduces_the_uniform_loss` |
| the energy balance still closes | `test_the_energy_balance_closes_on_a_graded_mesh` |
| band boundaries stay on grid lines | `test_graded_edges_keep_every_band_boundary_on_a_grid_line` |

Consequences that are part of the contract:

* **CG stays available**: per-volume coefficients are `diag(V)^{-1} K` with `K`
  symmetric, so `solve_linear(..., scale=V)` solves the similar symmetric system.  On a
  uniform mesh the transformation is a constant and is not applied.
* **The transient operator** adds the mass diagonal `diag(ρc_p)` *after* the local
  volumes are gone, so the `1/d³` regression stays fixed on a graded grid too.
* **Flux integrals, the balance, the geometry masks, the state hash and the 3D view**
  all use the local sizes (`RectilinearGrid`), so a graded run is reported and saved
  like any other.

## Part 3 — Hairpin heater elements (done, and no longer the design path)

### What the code implements

`src/core/heaters.py`: a **flanged immersion heater bank** of U-shaped (hairpin)
sheathed elements, the shape the literature and
`photo/heating_elements_3D.png` describe:

```
HairpinElement
  ├─ sheath: outer diameter d_s (default 12 mm), material (stainless steel)
  ├─ leg spacing: centre-to-centre distance of the two legs
  ├─ bend radius at the bottom (bend_chords chords in the rasteriser)
  ├─ active length in the sand + cold shank through the insulation and the air
  └─ rated power [W]; surface power = P / (π d_s (L_active + bend)) [W/cm²]
HeaterBank
  ├─ rows × columns (grid) or n_rings (ring layout)
  ├─ power per element and total power
  ├─ offset from the bottom/top of the storage band
  └─ support plate and flange elevations
HeaterConfig.bank(z_storage_start, z_storage_end)  # the config -> bank conversion
rasterize(bank, mesh, cx, cy, r_storage, ...)      # -> RasterResult
validate_bank(bank, mesh, cx, cy, r_storage, ...)  # -> list of problems
```

Thermal model per element:

* the cells crossed by the *sheath* take the sheath material (steel) - which the
  graded mesh makes possible, because the heater band targets the sheath diameter;
* the **rated power is deposited in the active-length cells**
  (`Q = P/Σ V_active`), so the total is exact on any grid (`source_mask` unchanged);
* the sheath-to-sand resistance follows from the half-cell series conductance of the
  fine cells around it;
* wire → sheath conduction is lumped, and so is the axial conduction along the cold
  shank: both are stated assumptions, not modelled physics.

Validations (`validate_bank`): an element covering no cell, legs or elements sharing
cells, an element reaching outside the storage wall, a collision with a heat-exchanger
tube, `leg_spacing` not greater than the sheath diameter, offsets that leave no room.
Warnings (reported, not fatal): legs within one cell, a sheath thinner than a cell,
and a surface power outside **3-8 W/cm²** (`SURFACE_POWER_MIN_W_CM2`,
`SURFACE_POWER_LIMIT_W_CM2`).

### GUI

*Geometry → Heaters* exposes the hairpin design: total power (5 kW by default), pattern,
element count, grid rows/columns, rings, sheath diameter, leg spacing, active length,
cold shank, support plate, flange, offsets, plus the live **power per element** and
**surface power** readouts; *Calculate positions* lists the elements with their rated
power and surface power, and the geometry preview draws the legs, the bends, the
support plate and the flange.  The live readout rejects nothing by itself - the
*warning* appears in the list and in `BatteryGeometry.heater_warnings`.

There is **one** element type: the hairpin.  The "element type (hairpin / straight rod
/ coil)" selector that an earlier draft of this document promised was never built, and
the rod-heater path was deleted in the simplification pass
(`HeaterElement`, `heater_radius`, `heater_length`, `custom_positions` - see
`CHANGELOG.md`).

### Why it is not the design path any more

The reference class of machines charges the bed with **hot gas through buried pipes**
and keeps the resistors in the gas circuit ([13](13_REDESIGN.md) §1-2).  With that
architecture:

* the bed is heated by the pipe surface, so the heater bank is not needed for the
  charge path;
* the pipe network replaced the "tubes as lumped convective sinks" model and brings its
  own sizing rules ([15](15_PIPE_NETWORKS.md));
* what the hairpin bank still needs - a mesh that resolves a 12 mm sheath - is exactly
  what the graded mesh provides, so the two features remain compatible and tested.

The bank is therefore kept, validated and documented, but a study of the default
machine should use the *uniform zone* pattern (the default) or the pipe network, and
treat a discrete hairpin bank as an option for directly heated beds.

## Part 4 — Open: the optimisation measurements

| item | state |
|---|---|
| graded vs uniform accuracy at equal cost | **not measured**: `scripts/benchmark.py` builds uniform meshes only (`build_model(spacing)`) and prints assembly/solve timings per size |
| assembly overhead of the graded coefficients on a graded grid vs the uniform fast path | **not measured** (the uniform special case was removed, so there is one path to measure) |
| transient: warm start + cached preconditioner dominate, and the operator is rebuilt only for a changed `dt` | implemented (`src/solver/transient.py`); no dedicated benchmark |
| heater rasterisation cost | O(cells along the elements), done once per build; measured only by the test that builds a 20 000-leaf octree in under a second (`tests/test_octree.py`), not for the bank |
| the adaptive (octree) alternative to grading: balance, conservative faces, a flux-jump indicator and `refine_on_objective` | implemented in `src/core/octree.py` + `src/solver/octree_solver.py`; **not connected** to `Mesh3D`/`SteadyStateSolver`, so it is not an alternative a run can choose today ([13](13_REDESIGN.md) §5) |

Two further items that this document used to promise and that are still open:

* a **sub-grid heater model** (spread the source over the containing cell with a
  contact resistance) would remove the requirement to resolve the sheath - it is not
  implemented;
* `docs/09` and `docs/11` used to quote stale test counts; they now record the command
  and the date instead ([09](09_TESTING.md) §1).
