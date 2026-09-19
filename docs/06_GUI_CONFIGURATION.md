# 6. GUI configuration

Every control, its default, its range and what it configures.  Defaults are the
**live widget values** (`GeometryPanel()`, `MaterialsPanel()`, `AnalysisPanel()`,
`SolverPanel()`); ranges are the widget limits.

## 1. Geometry — Cylinder

| control | default | range | feeds |
|---|---|---|---|
| Lx / Ly / Lz [m] | 6.0 / 6.0 / 5.6 | 1–50 | `Mesh3D(Lx, Ly, Lz, spacing)` |
| Centre X / Y [m] | 3.0 / 3.0 | 0.1–49 | `CylinderGeometry.center_x/center_y` |
| Base elevation [m] | 0.3 | 0–5 | `base_z` |
| Storage radius [m] | 2.0 | 0.2–20 | `r_storage` |
| Storage height [m] | 4.0 | 0.5–30 | `height` |
| Tubes/heaters phase [deg] | 15 | 0–180 | `phase_offset_deg` |
| Conical roof | on | – | `enable_cone_roof` |
| Roof angle [deg] | 15 | 0–45 | `roof_angle_deg` |
| Steel slab [m] | 0.005 | 0–0.2 | `steel_slab_top` |
| Cone fill (sand) | off | – | `fill_cone_with_sand` |

The roof apex (`base_z + slabs + height + steel slab + r_shell·tan(angle)`) must
stay below `Lz`: the build refuses rather than clipping the roof.

## 2. Geometry — Insulation

| control | default | range | feeds |
|---|---|---|---|
| Radial insulation [m] | 0.3 | 0.02–1 | `insulation_thickness` |
| Steel shell [m] | 0.02 | 0–0.2 | `shell_thickness` |
| Bottom slab [m] | 0.2 | 0–1 | `insulation_slab_bottom` |
| Top slab [m] | 0.2 | 0–1 | `insulation_slab_top` |
| Foundation margin [m] | 0.5 | 0–3 | `foundation_margin` (concrete beyond the shell) |

## 3. Geometry — Heaters

| control | default | range | feeds |
|---|---|---|---|
| Total power [kW] | 50 | 0–10 000 | `HeaterConfig.power_total` |
| Pattern | Uniform zone (volumetric) | 7 patterns | `HeaterConfig.pattern` |
| Elements | 12 | 1–500 | `n_heaters` |
| Element radius [m] | 0.02 | 0.005–0.2 | `heater_radius` |
| Element length [m] | 0 (full band) | 0–30 | `heater_length` |
| Offset bottom / top [m] | 0 / 0 | 0–2 | `offset_bottom` / `offset_top` |
| Grid rows / columns | 4 / 4 | 1–20 | `grid_rows` / `grid_cols` |
| Grid spacing [m] | 0.3 | 0.05–2 | `grid_spacing` |
| Rings | 2 | 1–10 | `n_rings` |
| Sheath diameter [m] | 0.012 | 0.006–0.05 | `sheath_diameter` (hairpin sheath) |
| Leg spacing [m] | 0.08 | 0.02–1 | `leg_spacing` between the two legs |
| Active length [m] | 0 (full band) | 0–30 | `active_length` = heated length in the sand |
| Cold shank [m] | 0.15 | 0–2 | `cold_shank` through insulation and air |
| Support plate [m] | 0.05 | 0–1 | `support_plate_offset` above the storage floor |
| Flange above roof [m] | 0.03 | 0–1 | `flange_offset` |
| Power per element [kW] | computed | – | display only |
| Surface power [W/cm²] | computed | – | `P / (pi d L_active)`, with "in range" / "out of the 3-8 W/cm² range" |
| *Calculate positions* | – | – | preview list (pure, does not change the config) |

Patterns: uniform zone (volumetric, the default), vertical grid, checkerboard,
radial array, spiral, concentric rings, custom positions.  Every discrete pattern is
drawn as a bank of **hairpin elements** (`src/core/heaters.py`) and validated against
the mesh: see `docs/03_GEOMETRY.md` §3.

## 4. Geometry — Tubes

| control | default | range | feeds |
|---|---|---|---|
| Tubes active (discharge) | off | – | `TubeConfig.active` |
| Fluid inlet [°C] | 60 | −20–400 | `t_fluid` (converted to K) |
| Fluid h [W/(m²·K)] | 500 | 10–20 000 | `h_fluid` |
| Pattern | Radial array | 6 patterns | `TubeConfig.pattern` |
| Tubes | 8 | 1–200 | `n_tubes` |
| Diameter [m] | 0.05 | 0.01–0.5 | `diameter` |
| Grid rows / columns | 3 / 3 | 1–20 | `grid_rows` / `grid_cols` |
| Grid spacing [m] | 0.2 | 0.05–2 | `grid_spacing` |
| Rings | 2 | 1–10 | `n_rings` |

## 5. Geometry — Mesh

| control | default | range | feeds |
|---|---|---|---|
| Refined mesh | on | on/off | graded (`GridSpec`) vs uniform (`Mesh3D.spacing`) |
| Cell size (uniform) [m] | 0.2 | 0.02–1 | `spacing` when the refined mode is off |
| Cells across storage | 10 | 2–200 | target = `r_storage / n` inside the storage band |
| Cells across insulation | 3 | 1–50 | target = `insulation_thickness / n` in the shell ring |
| Cells across sheath | 2 | 1–20 | target = `min(sheath, tube diameter) / n` around the **heater bank** |
| Far field [m] | 0.4 | 0.05–2 | largest cell in the air |
| Growth ratio | 1.3 | 1.02–2 | largest size change between neighbouring cells |
| Smallest cell [m] | 0 (no floor) | 0–1 | `GridSpec.min_size`: keeps a deep refinement from exploding |
| Largest cell [m] | 0 (no ceiling) | 0–5 | `GridSpec.max_size` |
| Cell budget | 400 000 | 10 000–20 M | all targets are scaled by a common factor to fit it |
| Grid / Memory | computed | – | cells per axis, size range, worst ratio, field memory, and a note when the budget coarsened the targets |

**Automatic mesh** (same tab): give a temperature tolerance [K] and a power tolerance
[%], press *Find the mesh* and the search runs the steady case on a sequence of grids
(each one `refine factor` finer) until the storage mean temperature and the heat
leaving the battery move by less than the tolerances.  The chosen grid is adopted and
the summary says how many levels and how long it took; when the cell budget or the
minimum cell size stops the refinement the search says so instead of pretending.  The
search runs on the background thread like any other analysis.

The refinement bands are built from the geometry itself (storage band, insulation
slabs and shell ring, heater bank), so the targets are *physical* numbers and do not
depend on the domain size.  The graded grid spans the box exactly (nothing to snap);
the uniform mode still snaps Y and Z to a whole number of cells.

## 6. Materials

| control | default | options | feeds |
|---|---|---|---|
| Storage medium | Steatite (soapstone) | 7 media | `BatteryGeometry.storage_material` |
| Packing [%] | 63 | 20–90 | `packing_fraction` (÷100) |
| Insulation | Rock wool | 5 insulators | `insulation_material` |
| Shell | Carbon steel | 3 structural | `shell_material` |
| Ambient [°C] | 20 | −40–80 | `h_top`/`h_lateral` partner `t_ambient` (K) |
| Ground [°C] | 10 | −20–60 | `t_ground` (K) |
| h top [W/(m²·K)] | 10 | 0–200 | top face convection |
| h lateral [W/(m²·K)] | 5 | 0–200 | lateral face convection |
| Radiation | Model off | off / on (linearised) | `SolverConfig.radiation` |

## 7. Analysis — Type

| control | default | notes |
|---|---|---|
| Steady state | selected | given the geometry and its heater power |
| Losses analysis | – | holds a target storage temperature by iterating the power |
| Transient | – | power and extraction profiles over time |
| Mean storage T [°C] | 400 | losses target (20–1200) |
| Duration | 10 hours | 0.01–100 000 in the selected unit |
| Unit | hours | seconds / minutes / hours / days |
| Time step dt [s] | 60 | 0.1–86 400 |
| Save interval [s] | 600 | 1–86 400 |
| Save T field per sample | off | keeps the full field for every sample |

## 8. Analysis — Initial condition

| control | default | notes |
|---|---|---|
| Uniform temperature | selected | 20 °C, −40–1200 |
| Temperature per material | – | one value per material (sand, insulation, steel, air, concrete, tubes), 20 °C each |
| Load a saved HDF5 state | – | `StateManager.load_state` with geometry-hash check |
| Start from the steady solution | – | the controller solves steady first, then starts the transient |

## 9. Analysis — Power profile

| control | default | notes |
|---|---|---|
| Off | – | no heating |
| Constant power | selected | 10 000 W, unit selector W / kW / MW (default kW) |
| Scheduled profile | – | table of (time [s], power [W]); times must increase |
| From CSV | – | two columns `t,P`; a missing or malformed file raises |

## 10. Analysis — Extraction

| control | default | notes |
|---|---|---|
| No extraction | selected | tubes inactive |
| Target power [W] | 5000 | capped by the available `h A (T_tube − T_inlet)` |
| Fluid flow rate [kg/s] | 0.1 | convective exchange with `t_inlet` |
| Inlet T [°C] | 20 | −20–400 |
| Tube h [W/(m²·K)] | 500 | used by both modes |

## 11. Tools — Solver

| control | default | options |
|---|---|---|
| Method | BiCGSTAB | BiCGSTAB, CG (symmetric only), GMRES, direct LU |
| Preconditioner | Jacobi | Jacobi, none, ILU, AMG Ruge–Stuben, AMG smoothed aggregation |
| Tolerance | 1e-8 | 1e-10 / 1e-8 / 1e-6 / 1e-4 |
| Max iterations | 5000 | 100–200 000 |
| Threads | All − 1 | auto (all cores) / all − 1 / 2 cores / 1 core - BLAS and OpenMP budget |
| Radiation | Off | Off / On |
| *3D view* | – | field, cut axis, position and opacity sliders, reset camera (colormap fixed to coolwarm) |
| Losses: T tolerance [K] | 1.0 | 0.05–20 |
| Losses: max iterations | 20 | 1–200 |
| Losses: under-relaxation | 0.7 | 0.1–1.0 |
| Losses: initial power density [W/m³] | 100 | 1–10 000 |
| Losses: ground h [W/(m²·K)] | 5 | 0–1000 (0 keeps the fixed-temperature ground) |

## 12. Save / Load and exports

| control | notes |
|---|---|
| Name / Description | written into the HDF5 state |
| Save state (HDF5) | `results/states/<name>.h5` with geometry hash, unit tag and version |
| Load state (HDF5) | refused if the geometry hash or the grid does not match |
| Export VTK / CSV | field export from the Tools tab |
| Export time series (CSV) | in the results panel after a transient run |
