# 6. GUI configuration

Every control, its default, its range and what it configures.  **Defaults are the
live widget values**, not a copy that can drift: they were read by instantiating the
panels off-screen and walking the form rows of every tab (2026-09-23, PyQt6 +
`QT_QPA_PLATFORM=offscreen`):

```python
from gui.views.geometry_panel import GeometryPanel
panel = GeometryPanel()
for i in range(panel.tabs.count()):
    layout = panel.tabs.widget(i).layout()        # a QFormLayout: label + field per row
    ...                                           # .value() / .currentText() / .isChecked()
```

Ranges are the widget limits; "feeds" is the `src` field the getter writes.

**Reference scale.** The defaults are sized on the published pilot of the class (Kankaanpää,
2022: 4 m across, 7 m tall, ~100 t of sand, 8 MWh, 200 kW of charge, 100 kW of
discharge; the 2025 Pornainen unit is 15 m x 13 m, 2000 t of soapstone, 1 MW / 100 MWh):
a 4 m vessel with 5 m of bed (~95-100 t), 200 kW rated, 100 kW out, standby at 500 °C.

**The plant, as the GUI shows it** (2026-09-23): electric resistors heat a gas in a closed
circuit; the gas runs through pipes buried in the sand and hands the heat to the bed
across the pipe walls; on discharge an exchanger on the same circuit takes it back out.
`GeometryPanel` has five sub-tabs: **Cylinder, Insulation, Gas circuit, Pipes, Mesh**.
Every analysis - the standby, the transient and the automatic mesh search - runs the gas
loop through the painted pipes; there is no heater element in the bed, no fixed film on
the pipes and no second exchanger inside the vessel.

## 1. Geometry — Cylinder

| control | default | range | feeds |
|---|---|---|---|
| Lx / Ly / Lz [m] | 6.0 / 6.0 / 6.5 | 1–50 | the box the vessel sits in (`tree_resolution` makes the octree's cube from it) |
| Centre X / Y [m] | 3.0 / 3.0 | 0.1–49 | `CylinderGeometry.center_x/center_y` |
| Base elevation [m] | 0.3 | 0–5 | `base_z` |
| Storage radius [m] | 2.0 | 0.2–20 | `r_storage` |
| Storage height [m] | 5.0 | 0.5–30 | `height` |
| Conical roof | on ("enable") | – | `enable_cone_roof` |
| Roof angle [deg] | 15 | 0–45 | `roof_angle_deg` |
| Steel slab [m] | 0.005 | 0–0.2 | `steel_slab_top` |
| Cone fill ("fill with sand") | off | – | `fill_cone_with_sand` |

The box only has to contain the vessel: the air around it is excluded from the problem
(`BatteryGeometry.apply_environment`), and the roof apex must stay below `Lz` - the build
refuses rather than clipping the roof.

## 2. Geometry — Insulation

| control | default | range | feeds |
|---|---|---|---|
| Radial insulation [m] | 0.3 | 0.02–1 | `insulation_thickness` |
| Steel shell [m] | 0.02 | 0–0.2 | `shell_thickness` |
| Bottom slab [m] | 0.2 | 0–1 | `insulation_slab_bottom` |
| Top slab [m] | 0.2 | 0–1 | `insulation_slab_top` |
| Foundation margin [m] | 0.5 | 0–3 | `foundation_margin` (concrete beyond the shell) |

## 3. Geometry — Gas circuit

| control | default | range | feeds |
|---|---|---|---|
| Rated power [kW] | **200** | 0–100 000 | `HeaterConfig.power_total`: the charging power - the constant power profile of a transient and the rating the tube wall is checked against |
| Gas | Air | air / nitrogen / steam | `RunConfig.pipe_fluid` (`Fluid` of `src/solver/fluid.py`) |
| Mass flow [kg/s] | 1.0 | 0–200 | `RunConfig.pipe_flow`: the loop's total mass flow (the branches split it) |
| Circuit pressure [bar] | 1.01325 ("1 bar") | 0.1–100 | `RunConfig.pipe_pressure`: the density follows it (`Fluid.at_pressure`) |
| Fan efficiency [%] | 70 | 10–100 | `RunConfig.pipe_fan_efficiency`: the blower's electric power |
| Wall roughness [um] | "from the tube material" | 0–2000 | `PipeNetworkConfig.roughness` (tubes, headers, ducts) |
| Circuit | "build the network to size it" | – | read-only: the power over the wetted surface (W/cm²), the power per riser, the flow and the pressure |

The **surface power** is the pipe wall's rating: `pipe_surface_power_w_cm2` puts the
design power over the geometric wetted area of the network and quotes it against the
3–8 W/cm² window.  The film coefficient of every pipe cell is not a setting: the loop
computes it from the flow, the gas and the bore (`pipe_h`).

Removed 2026-09-23, because nothing in the plant corresponds to them any more: the
*Source from bottom/top* offsets (the band of a lumped bed source no analysis uses once
the network is painted) and *Return to the resistors* (a prescribed loop inlet, which
silently ignored the power; the discharge that returns the gas at a set temperature is
now an extraction mode, §10).

## 4. Geometry — Pipes

| control | default | range | feeds |
|---|---|---|---|
| Layout | Concentric rings | staggered / grid / rings / radial / spiral | `PipeNetworkConfig.layout` |
| Collection | Reverse return (balanced) | distributor + collector / reverse return / central header / two-level rings | `collection` |
| Rings / Radial files | 6 / 12 | 1–20 / 3–72 | `n_rings` (spread over the whole radius, one ring every `r / n`, risers on a ring spaced like the rings) / `n_files` |
| Pipe outer d [m] | 0.05 | 0.01–0.3 | `diameter` (the pitch follows it) |
| Wall thickness [mm] | 2.0 | 0–20 | `wall_thickness` (the bore carries the gas) |
| Tube material | Stainless steel (drawn) | stainless / carbon | `material` (and its default roughness) |
| Duct d [m] | 0.15 | 0.05–0.6 | `duct_diameter` |
| Insulated headers | off | – | `insulated_headers` (lagged headers exchange nothing) |
| Inlet / Outlet azimuth [deg] | 180 / 0 | 0–360 | `azimuth_in` / `azimuth_out` |
| Flow split | Equal per branch | equal / path length / per ring main / per sector | `split_mode` |
| Sectors | 4 | 1–16 | `n_sectors` |
| *Rebuild the network on the mesh* | – | – | rebuilds and repaints on the current mesh |
| Network | "build the mesh: it paints the network" | – | read-only: `summary()`, the paint report, the surface power |

Every edit of the Cylinder, Insulation, Gas circuit and Pipes tabs redraws the geometry
preview after 250 ms, with the network as the panels describe it *now* (laid out on a
throw-away tree of eight leaves - `ThermalBatteryGUI.preview_network`), and the status bar
says when the built mesh no longer matches.

`Build mesh` builds **and paints** the network: the pipes are the heat-transfer surface,
so every run uses the loop painted on the very cells it solves.  The paint marks the
exchanging cells (`TUBES`, convective) and writes no film value; the loop writes one per
cell in every analysis.  Removed 2026-09-23: *Gas h / Gas T* (the fixed film of the old
lumped tube model, which turned the pipes into a 60 °C sink in the steady and losses
runs) and *Junction refinement* (a band of the graded grid only).

## 5. Geometry — Mesh

The mesh is the adaptive **octree** (`AdaptiveMesh.from_bands`); the uniform and graded
modes are no longer in the GUI (`Mesh3D` stays in `src/` as the reference road of the
equivalence tests).

| control | default | range | feeds |
|---|---|---|---|
| Cells across storage | 10 | 2–200 | target `r_storage / n` over the sand |
| Cells across insulation | 3 | 1–50 | target `thickness / n` for the insulation ring, the slabs and the shell |
| Cells across the tube | 2 | 1–20 | target `pipe outer d / n` over the pipe bundle |
| Cell budget | 400 000 | 10 000–20 000 000 | `tree_resolution`: the finest leaf the tree may use |
| Active regions / Mesh / Memory | computed | – | read-only: the a priori targets per region, the box and the finest leaf, the leaf count per level after a build, the field memory |

The counts above are the layer rule; the a priori plan of `src/analysis/mesh_plan.py`
adds the surface rule `2k/h` only where a film actually sits (the steel casing, where it
never binds).  Only the **active** model is refined, each region with its own shape: the
sand and the slabs are discs, the insulation and the shell annuli about the vessel axis,
and every riser gets its own column as wide as the pipe - the tree grades the sand
around it by its 2:1 balance.  Every target snaps to the *nearest* leaf size (the leaves
are powers of two of the finest one).  The air outside the vessel is excluded and its
leaves stay coarse.  Measured on the default model (2026-09-23): 90 630 leaves built and
painted in 3.3 s (it was 143 396 leaves in 39 s: a single box over the whole bundle, four
axis-aligned boxes per annulus that refined the air at their corners, a film rule on the
insulation that faces no film, and a 2:1 balance that swept every face of the tree).

**Automatic mesh** (same tab):

| control | default | range | feeds |
|---|---|---|---|
| Temperature tolerance [K] | 2.0 | 0.05–100 | `delta_temperature` |
| Power tolerance [%] | 2.0 | 0.05–50 | `delta_power` (stored as a fraction) |
| Levels | 4 | 2–8 | `max_levels` |
| Refine factor | 0.6 | 0.2–0.9 | target scale from one level to the next |
| *Find the mesh* | – | – | `analysis.convergence.find_mesh` on the background thread |
| Search | "not run yet" | – | read-only: converged or stopped, leaves, dT, dP |

The search builds a tree per level, paints the battery **and the pipe network** on it
and solves the coupled steady state, so it measures the model every run uses.  An
adopted plan is used by the next *Build mesh*; changing a mesh setting drops it.  The
"search before building" switch was removed (2026-09-23): *Build mesh* builds the planned
tree, *Find the mesh* searches.

## 6. Materials

| control | default | options | feeds |
|---|---|---|---|
| Storage medium | Steatite (soapstone) | silica sand, olivine, steatite, basalt, magnetite, quartzite, granite | `storage_material` |
| Packing [%] | 63 | 20–90 | `packing_fraction` (÷100) |
| Insulation | Rock wool | rock wool, glass wool, calcium silicate, ceramic fibre, expanded perlite | `insulation_material` |
| Shell | Carbon steel | carbon steel, stainless steel 304, concrete | `shell_material` |
| Ambient [°C] | 20 | −40–80 | `t_ambient` (K) |
| Ground [°C] | 10 | −20–60 | `t_ground` (K) |
| Wind speed [m/s] | 0 | 0–30 | `BatteryGeometry.wind_speed`: the forced part `4 + 4 v` of the outer film |

The outer film is computed from the correlations (Churchill-Chu natural convection plus
the wind term, `src/core/environment.py`) and logged after every build.  Removed
2026-09-23: *h top / h lateral* (films on the box faces, which only touch excluded air)
and the second *Radiation* switch (never read; radiation is Tools > Solver).

## 7. Analysis — Type

| control | default | range / options |
|---|---|---|
| Steady standby (hold the storage at a temperature) | **selected** | the power that holds the bed at T = the standby losses, with its field |
| Transient (charge / discharge profiles) | – | the profiles drive the loop |
| Mean storage T [°C] (standby) | 500 | 20–1200 |
| Duration | 10 | 0.01–100 000, in the selected unit |
| Unit | hours | seconds / minutes / hours / days |
| Time step dt [s] | 60 | 0.1–86 400 |
| Save interval [s] | 600 | 1–86 400 |
| Full fields ("save T field per sample") | off | – |

**Why no "steady at a power".**  A storage has one steady state worth asking for: the
one where the power in equals the losses.  At a charging power the bed would settle at
whatever temperature makes the losses that large - thousands of degrees for a real
plant - so the steady analysis is asked as a temperature and answered with the holding
power (`solve_losses`, two iterations on a linear problem thanks to a proportional first
step).

The coupling of the gas and the sand is implicit in the wall temperature and holds the
loop balance per step, so the time step is an accuracy choice, not a stability limit
(docs/12 §11).

## 8. Analysis — Initial condition

| control | default | notes |
|---|---|---|
| Uniform temperature | selected | 20 °C, −40–1200 |
| Temperature per material | – | 20 °C each: sand, insulation, steel, concrete, pipes |
| Current field (last result or loaded state) | – | the mesh's own field (`InitialCondition(mode="keep")`): chain a discharge after a charge, or start from a state loaded in Save / Load |
| Start from the standby state | – | the controller solves the standby state at the Type tab's temperature and keeps that field |

Replaced 2026-09-23: *Load a saved HDF5 state* read the `.h5` file with `np.load`, which
cannot read HDF5 - the option could not work.  Loading is on the Save / Load tab and
*Current field* starts from it.

## 9. Analysis — Power profile (the resistors)

| control | default | notes |
|---|---|---|
| Off | – | resistors off |
| Constant: the circuit's rated power | selected | the Gas circuit's rated power (200 kW) - one number for the charge, not two |
| Scheduled profile | – | table of (time [s], power [W]); times must increase |
| From CSV (t, P) | – | a missing or malformed file raises |

## 10. Analysis — Extraction (the exchanger on the circuit)

| control | default | range | notes |
|---|---|---|---|
| No extraction | selected | – | the exchanger is idle |
| Exchanger power [kW] | 100 | 0–100 000 | `ExtractionProfile(mode="power")`: taken out of the gas; the loop solves its inlet temperature, and the run stops (keeping its samples) when the bed can only give that power with gas colder than the return temperature below |
| Exchanger return temperature [°C] | 60 | −20–400 | the coldest the gas comes back (`t_return_min`) in the power mode; in `ExtractionProfile(mode="return_temperature")` the gas comes back at it and the power is whatever the bed gives, with the resistors off |

The loop's external power is `power profile − exchanger power`, so a charge and a
discharge at the same time are a net power into the gas.  Removed 2026-09-23: *Fluid
flow rate* and *Tube h* (the lumped-tube extraction, ignored whenever the loop runs).

## 11. Analysis — Save / Load

| control | default | notes |
|---|---|---|
| Name | "Simulation" | written into the HDF5 state |
| Description | empty | written as a note |
| *Save state (HDF5)* | – | `results/states/<name>.h5` with geometry hash, unit tag and version |
| *Load state (HDF5)* | – | refused if the geometry hash or the tree does not match, or while a run is in flight |
| State file / State | – | read-only: the file and the result of the last save/load |

## 12. Tools — Solver and Losses

| control | default | options |
|---|---|---|
| Tolerance | 1e-8 | 1e-10 / 1e-8 / 1e-6 |
| Max iterations | 2000 | 100–200 000 |
| Threads | All − 1 | Auto (all cores) / All − 1 / 2 cores / 1 core (applied through `threadpoolctl`) |
| Radiation | Off | Off / On |
| Losses: T tolerance [K] | 1.0 | 0.05–20 |
| Losses: max iterations | 20 | 1–200 |
| Losses: under-relaxation | 1.0 | 0.1–1.0 |
| Losses: initial power density [W/m³] | 100 | 1–10 000 |

The linear method is fixed: **conjugate gradients + AMG (Ruge-Stüben)** on the
volume-symmetrised operator (docs/12 §3, the measurements in docs/14).  The linear layer
falls back on its own - BiCGSTAB on a non-symmetric operator, Jacobi without PyAMG - and
reports it in the log.  Removed 2026-09-23: the method and preconditioner selectors and
the losses' *ground h* (it rewrote the ground condition of the mesh and left it changed
for the runs that followed).

## 13. 3D view

| control | default | options |
|---|---|---|
| Field | Temperature | Temperature / Material / Sources / Conductivity |
| Cut | z | x / y / z |
| Position slider | 50 % | 1–99 % of the axis length |
| Opacity slider | 80 % | 5–100 % |
| *Reset camera* | – | restores the default point of view |
| Colourmap | `coolwarm`, fixed | – |

Without an OpenGL context the group degrades to a placeholder (`THERMAL_DISABLE_3D=1`
forces that mode).

## 14. Results and exports

| control | where | notes |
|---|---|---|
| Statistics | results panel | temperatures of the battery (the excluded air is left out), the mesh report |
| Energy balance | results panel | input (gas → bed), extraction, envelope losses by face, all paths, imbalance, stored energy and exergy, and the **gas loop**: inlet/outlet temperature, bed power, NTU, pressure drop, fan power |
| Export time series (CSV)... | results panel | `TransientResults.export_csv` |
| Export field (VTK)... | results panel | `src.viz.scene.export_vtk`: every leaf and every field, `.vtu` for ParaView |
