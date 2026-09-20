# 6. GUI configuration

Every control, its default, its range and what it configures.  **Defaults are the
live widget values**, not a copy that can drift: they were read by instantiating the
panels off-screen and asking each widget (2026-09-20, PyQt6 + `QT_QPA_PLATFORM=offscreen`):

```python
from gui.views.geometry_panel import GeometryPanel
from gui.views.materials_panel import MaterialsPanel
from gui.views.analysis_panel import AnalysisPanel
from gui.views.solver_panel import SolverPanel
for label, widget in GeometryPanel().tabs.widget(0).fields.items():
    print(label, widget.value())          # .value() / .currentData() / .isChecked()
```

Ranges are the widget limits; "feeds" is the `src` field the getter writes.
`GeometryPanel` has six sub-tabs: Cylinder, Insulation, Heaters, Tubes, Mesh, Pipes.

## 1. Geometry — Cylinder

| control | default | range | feeds |
|---|---|---|---|
| Lx / Ly / Lz [m] | 6.0 / 6.0 / 5.6 | 1–50, step 0.5 | `Mesh3D(Lx, Ly, Lz, ...)` |
| Centre X / Y [m] | 3.0 / 3.0 | 0.1–49, step 0.1 | `CylinderGeometry.center_x/center_y` |
| Base elevation [m] | 0.3 | 0–5 | `base_z` |
| Storage radius [m] | 2.0 | 0.2–20 | `r_storage` |
| Storage height [m] | 4.0 | 0.5–30 | `height` |
| Tubes/heaters phase [deg] | 15 | 0–180 | `phase_offset_deg` |
| Conical roof | on ("enable") | – | `enable_cone_roof` |
| Roof angle [deg] | 15 | 0–45 | `roof_angle_deg` |
| Steel slab [m] | 0.005 | 0–0.2, step 0.005 | `steel_slab_top` |
| Cone fill ("fill with sand") | off | – | `fill_cone_with_sand` |

The roof apex (`base_z + slabs + height + steel slab + r_shell·tan(angle)`) must
stay below `Lz`: the build refuses rather than clipping the roof.

## 2. Geometry — Insulation

| control | default | range | feeds |
|---|---|---|---|
| Radial insulation [m] | 0.3 | 0.02–1 | `insulation_thickness` |
| Steel shell [m] | 0.02 | 0–0.2, step 0.005 | `shell_thickness` |
| Bottom slab [m] | 0.2 | 0–1 | `insulation_slab_bottom` |
| Top slab [m] | 0.2 | 0–1 | `insulation_slab_top` |
| Foundation margin [m] | 0.5 | 0–3 | `foundation_margin` (concrete beyond the shell) |

## 3. Geometry — Heaters

| control | default | range | feeds |
|---|---|---|---|
| Total power [kW] | **5.0** | 0–10 000 | `HeaterConfig.power_total` |
| Pattern | Uniform zone (volumetric) | 6 patterns | `HeaterConfig.pattern` |
| Elements | 12 | 1–500 | `n_heaters` (ring/spiral layouts) |
| Offset from bottom / top [m] | 0.0 / 0.0 | 0–2 | `offset_bottom` / `offset_top` |
| Grid rows / columns | 4 / 4 | 1–20 | `grid_rows` / `grid_cols` |
| Rings | 2 | 1–10 | `n_rings` |
| Sheath diameter [m] | 0.012 | 0.006–0.05, step 0.002 | `sheath_diameter` |
| Leg spacing [m] | 0.08 | 0.02–1 | `leg_spacing` between the two legs |
| Active length [m] | 0.0 (= whole band) | 0–30 | `active_length` (heater `None` when 0) |
| Cold shank [m] | 0.15 | 0–2 | `cold_shank` through insulation and air |
| Support plate [m] | 0.05 | 0–1 | `support_plate_offset` above the storage floor |
| Flange above roof [m] | 0.03 | 0–1 | `flange_offset` |
| Power per element [kW] | computed | – | display: `power_total / n_heaters` |
| Surface power [W/cm²] | computed | – | display: `P / (π d (L_active + bend))`, with "in range" / "out of the 3-8 W/cm² range" |
| *Calculate positions* | – | – | fills the element list (preview only; the config is unchanged) |

Patterns (`HeaterPattern.ALL`): uniform zone (volumetric, the default), vertical
grid, checkerboard, radial array, spiral, concentric rings.  The pattern selects the
*bank geometry* of `src/core/heaters.py`: the two grid names produce a
`grid_rows × grid_cols` bank, the spiral a bank sized from `n_heaters` (about
`sqrt(n) × sqrt(n)`), and the radial/ring names a ring layout whose element count
follows the circumference and the leg spacing.  The validations and the surface
power window are in [03](03_GEOMETRY.md) §3.

## 4. Geometry — Tubes

| control | default | range | feeds |
|---|---|---|---|
| Heat exchanger ("tubes active (discharge)") | off | – | `TubeConfig.active` |
| Fluid inlet [°C] | 60 | −20–400 | `t_fluid` (converted to K) |
| Fluid h [W/(m²·K)] | 500 | 10–20 000 | `h_fluid` |
| Pattern | Radial array | 6 patterns | `TubeConfig.pattern` |
| Tubes | 8 | 1–200 | `n_tubes` |
| Diameter [m] | 0.05 | 0.01–0.5 | `diameter` |
| Grid rows / columns | 3 / 3 | 1–20 | `grid_rows` / `grid_cols` |
| Grid spacing [m] | 0.2 | 0.05–2 | `grid_spacing` |
| Rings | 2 | 1–10 | `n_rings` |
| *Calculate positions* | – | – | fills the tube list (preview only) |

## 5. Geometry — Mesh

| control | default | range | feeds |
|---|---|---|---|
| Refined mesh ("cells placed where the gradients are") | on | on/off | graded (`GridSpec`) vs uniform (`Mesh3D.spacing`) |
| Cell size (uniform) [m] | 0.2 | 0.02–1 | `spacing` when the refined mode is off |
| Cells across storage | 10 | 2–200 | target = `r_storage / n` inside the storage band |
| Cells across insulation | 3 | 1–50 | target = `insulation_thickness / n` in the shell ring |
| Cells across sheath | 2 | 1–20 | target = `min(sheath, tube diameter) / n` around the **heater bank** |
| Far field [m] | 0.4 | 0.05–2 | largest cell in the air |
| Growth ratio | 1.3 | 1.02–2 | largest size change between neighbouring cells |
| Smallest cell [m] | 0 ("auto") | 0–1 | `GridSpec.min_size`: floor on the realised size |
| Largest cell [m] | 0 ("auto") | 0–5 | `GridSpec.max_size`: ceiling on the realised size |
| Cell budget | 400 000 | 10 000–20 000 000 | every target is scaled by a common factor to fit it |
| Physics plan / Grid / Memory | computed | – | read-only summary: the a priori plan, cells per axis, size range, worst ratio, field memory, and whether a rail or the budget is active |

The refined targets are *physical* (they do not depend on the domain size); the
uniform mode uses one cell size everywhere and snaps Y/Z to whole cells.

**Automatic mesh** (same tab):

| control | default | range | feeds |
|---|---|---|---|
| Temperature tolerance [K] | 2.0 | 0.05–100 | `delta_temperature` of the search |
| Power tolerance [%] | 2.0 | 0.05–50 | `delta_power` (stored as a fraction) |
| Levels | 4 | 2–8 | `max_levels` |
| Refine factor | 0.6 | 0.2–0.9 | target scale from one level to the next |
| Automatic mesh ("search before building") | on | on/off | `wants_auto_search()`: *Build mesh* runs the search first |
| *Find the mesh* | – | – | runs `analysis.convergence.find_mesh` on the background thread |
| Search | "not run yet" | – | read-only result: converged or stopped, cells, dT, dP |

When the cell budget or the minimum cell size stops the refinement the search says so
instead of pretending; the adopted spec is stored in the panel and used by the build.

## 6. Geometry — Pipes

| control | default | range | feeds |
|---|---|---|---|
| Layout | Concentric rings (`rings`) | staggered / grid / rings / radial | `PipeNetworkConfig.layout` |
| Collection | Reverse return (balanced) | distributor + collector / reverse return / central header / two-level rings | `PipeNetworkConfig.collection` |
| Rings | 3 | 1–12 | `n_rings` (ring layouts) |
| Radial files | 12 | 3–72 | `n_files` (radial layout) |
| Pipe outer d [m] | 0.05 | 0.01–0.3 | `diameter` (the pitch follows it) |
| Duct d [m] | 0.15 | 0.05–0.6 | `duct_diameter` |
| Inlet azimuth [deg] | 180 | 0–360, step 15 | `azimuth_in` (stored in radians) |
| Outlet azimuth [deg] | 0 | 0–360, step 15 | `azimuth_out` (stored in radians) |
| Flow split | Equal per branch (`equal`) | equal / from path length | `split_mode` |
| *Build network* | – | – | rasterises the network on the current mesh and reports `summary()` plus any validation problem |
| Network | "build the mesh, then the network" | – | read-only: the report of the last build |

The panel derives the band from the vessel: `band_bottom = base_z + max(offset
bottom, 0.1)` and `band_top = base_z + height − 0.4`, so with the defaults the risers
run from 0.4 m to 3.9 m.  Layouts, collection modes and the design rules are in
[docs/15](15_PIPE_NETWORKS.md).

**Caveat (2026-09-20)**: this table is the state of the panel when the defaults were
dumped.  The Pipes tab is being extended - a spiral layout, wall thickness/material/
roughness, insulated headers, junction refinement, a build-and-paint action and a
circuit mass flow that feeds the transient's fluid loop - so re-read the widget dump
(`GeometryPanel().tabs.widget(5)`) and [docs/15](15_PIPE_NETWORKS.md), which is the
authority for the pipe controls, before trusting a row here.

## 7. Materials

| control | default | options | feeds |
|---|---|---|---|
| Storage medium | Steatite (soapstone) | 7 media (silica sand, olivine, steatite, basalt, magnetite, quartzite, granite) | `BatteryGeometry.storage_material` |
| Packing [%] | 63 | 20–90 | `packing_fraction` (÷100) |
| Insulation | Rock wool | 5 insulators (rock wool, glass wool, calcium silicate, ceramic fibre, expanded perlite) | `insulation_material` |
| Shell | Carbon steel | 3 structural (carbon steel, stainless steel 304, concrete) | `shell_material` |
| Ambient [°C] | 20 | −40–80 | `t_ambient` (K) |
| Ground [°C] | 10 | −20–60 | `t_ground` (K) |
| h top [W/(m²·K)] | 10 | 0–200 | top face convection |
| h lateral [W/(m²·K)] | 5 | 0–200 | lateral face convection |
| Radiation | Model off | off / on (linearised) | `SolverConfig.radiation` |

The *Conditions* tab is the single authority for ambient and ground: the hint states
that every analysis reads these values, so a run cannot use a different ambient from
the one displayed.  The *Storage* and *Insulation* tabs also show the properties of
the selected medium (k, ρ, cp, T_max).

## 8. Analysis — Type

| control | default | range / options |
|---|---|---|
| Steady state (fixed heater power) | **selected** | – |
| Losses analysis (hold a target temperature) | – | shows the Losses target group |
| Transient (power and extraction profiles) | – | shows the Time stepping group |
| Mean storage T [°C] (losses) | 400 | 20–1200 |
| Duration | 10 | 0.01–100 000, in the selected unit |
| Unit | hours | seconds / minutes / hours / days |
| Time step dt [s] | 60 | 0.1–86 400 |
| Save interval [s] | 600 | 1–86 400 |
| Full fields ("save T field per sample") | off | – |

The losses and transient groups are hidden unless their radio is selected, and `Run`
follows the type (`Run steady state`, `Run losses analysis`, `Run transient`).

## 9. Analysis — Initial condition

| control | default | notes |
|---|---|---|
| Uniform temperature | selected | 20 °C, −40–1200 |
| Temperature per material | – | one value per material, 20 °C each, −40–1200: sand, insulation, steel, air, concrete, tubes |
| Load a saved HDF5 state | – | read-only path + *Browse*; `StateManager.load_state` with the geometry-hash check |
| Start from the steady solution | – | the controller solves the steady case first and keeps that field (`InitialCondition(mode="keep")`) |

## 10. Analysis — Power profile

| control | default | notes |
|---|---|---|
| Off | – | no heating |
| Constant power | selected | 10 000 in the selected unit; unit default **kW**, options W / kW / MW - so the default is 1 × 10⁷ W |
| Scheduled profile | – | table of (time [s], power [W]) with Add/Remove row; times must increase |
| From CSV (t, P) | – | read-only path + *Browse*; a missing or malformed file raises |

## 11. Analysis — Extraction

| control | default | range | notes |
|---|---|---|---|
| No extraction | selected | – | tubes inactive |
| Target power [W] | 5000 | 0–10 000 000 | capped by the available `h A (T_tube − T_inlet)` |
| Mass flow [kg/s] | 0.1 | 0.001–100 | convective exchange with the inlet temperature |
| Inlet T [°C] | 20 | −20–400 | passed in every mode |
| Tube h [W/(m²·K)] | 500 | 10–20 000 | used by the two extraction modes (not in "off") |

The power widget is enabled only in the target-power mode and the flow/h widgets only
in the flow-rate mode; the inlet temperature stays enabled unless extraction is off.

## 12. Analysis — Save / Load

| control | default | notes |
|---|---|---|
| Name | "Simulation" | written into the HDF5 state |
| Description | empty | written as a note |
| *Save state (HDF5)* | – | `results/states/<name>.h5` with geometry hash, unit tag and version |
| *Load state (HDF5)* | – | refused if the geometry hash or the grid does not match, or while a run is in flight |
| State | "no state loaded" | read-only: result of the last save/load |

## 13. Tools — Solver and Losses

| control | default | options |
|---|---|---|
| Method | BiCGSTAB | BiCGSTAB (robust) / CG (symmetric only) / GMRES / direct LU |
| Preconditioner | Jacobi | Jacobi / none / ILU / AMG Ruge-Stuben / AMG smoothed aggregation |
| Tolerance | 1e-8 | 1e-10 / 1e-8 / 1e-6 / 1e-4 |
| Max iterations | 5000 | 100–200 000, step 500 |
| Threads | All − 1 | Auto (all cores) / All − 1 / 2 cores / 1 core - the BLAS and OpenMP budget |
| Radiation | Off | Off / On |
| Losses: T tolerance [K] | 1.0 | 0.05–20 |
| Losses: max iterations | 20 | 1–200 |
| Losses: under-relaxation | 0.7 | 0.1–1.0 |
| Losses: initial power density [W/m³] | 100 | 1–10 000 |
| Losses: ground h [W/(m²·K)] | 5 | 0–1000 (0 keeps the fixed-temperature ground) |

The default note in the panel: CG is refused on non-symmetric systems, so the solver
reports the switch instead of diverging silently.  The `Tools` tab also carries the
**Help** page with the workflow summary.

## 14. 3D view

| control | default | options |
|---|---|---|
| Field | Temperature | Temperature / Material / Sources / Conductivity |
| Cut | z | x / y / z |
| Position slider | 50 % | 1–99 % of the axis length (clamped to 2–98 % when rendering) |
| Opacity slider | 80 % | 5–100 % |
| *Reset camera* | – | restores the default point of view |
| Colourmap | `coolwarm`, fixed | the selector was removed; the value labels of both sliders are printed next to them |

If no OpenGL context is available the whole group degrades to a placeholder label and
stays disabled (the controls exist but `controls.setEnabled(False)`), which is the mode
the test suite forces with `THERMAL_DISABLE_3D=1`.

## 15. Exports

| control | where | notes |
|---|---|---|
| Export time series to CSV... | results panel (below the tabs) | writes `TransientResults` to CSV; with no transient run in memory it only logs "no transient results to export" |
| Export VTK / CSV of the field | **code only** | `ThermalBatteryGUI.export_vtk` and `export_csv_field` exist as slots but no button is wired to them; scripts call `src.viz.scene.export_vtk` / `export_csv` directly |
