# 6. GUI configuration

Every control, its default, its range and what it configures.  **Defaults are the
live widget values**, read by instantiating the window off-screen and walking the form
rows of every section (2026-09-23, re-measured on PySide6 2026-09-29, with
`QT_QPA_PLATFORM=offscreen`):

```python
from gui.main_window import ThermalBatteryGUI
window = ThermalBatteryGUI()
for box in window.findChildren(QGroupBox):          # every section
    for form in box.findChildren(QFormLayout):      # label + field per row
        ...                                         # .value() / .currentText() / .isChecked()
```

**Reference scale.** The defaults are sized on the published pilot of the class
(Kankaanpää, 2022: 4 m across, 7 m tall, ~100 t of sand, 8 MWh, 200 kW of charge, 100 kW
of discharge; the 2025 Pornainen unit is 15 m x 13 m, 2000 t of soapstone,
1 MW / 100 MWh): a 4 m vessel with 5 m of bed (~95 t), 200 kW rated, 100 kW out, standby
at 500 °C.

## 0. Layout

The left column has six tabs, one per *thing*, and every page scrolls instead of being
cut by a short screen:

| tab | what it is |
|---|---|
| **Vessel** | the storage bed, the insulation and the shell, the roof - and the materials of those layers, next to the shape they fill |
| **Plant** | the gas circuit the resistors heat, and the buried pipes it runs through |
| **Site** | ambient air, ground, wind: the only ambient values, read by every analysis |
| **Mesh** | the octree and the automatic search |
| **Analysis** | Type (standby or transient), Initial, Charge (the resistors), Discharge (the exchanger), Save / Load |
| **Solver** | accuracy in kelvin, threads, radiation |

Explanations take no room in a form: a control's explanation is its tooltip and its
label carries **ⓘ**; a section's explanation is one "ⓘ about this section" line whose
tooltip holds the text.  Read-outs wrap downwards.  The window opens inside the screen's
free area (95 % of its width, 90 % of its height at most) and the Help menu holds the
workflow.  `tests/test_gui.py::test_the_window_fits_the_screen_and_explanations_take_no_room`
pins all of this.

Removed on 2026-09-23 because they were redundant: **Lx, Ly, Lz, Centre X, Centre Y**.  The
air around the vessel is excluded from the problem, so the box only has to hold the
vessel plus 0.3 m of air (`GeometryPanel.domain`) and the vessel is centred in it - a
centre set by hand is what put the pipe network off the vessel once.

**Geometry and mesh are coupled.**  The window keeps a digest of the model the mesh was
built from (`ThermalBatteryGUI.model_signature`: the vessel, the materials, the site, the
mesh request and the pipe network with its engine design).  An edit on any of those pages
marks the mesh out of date - the 3D view says so in red and the Mesh tab's read-out too -
and *Run* rebuilds the mesh (running the header engine if the plant changed) before it
solves; `tests/test_gui.py::test_the_mesh_follows_the_model_it_was_built_from` pins it.

## 1. Vessel

| control | default | range | feeds |
|---|---|---|---|
| Radius [m] | 2.0 | 0.2–20 | `CylinderGeometry.r_storage` |
| Height [m] | 5.0 | 0.5–30 | `height` (of the bed) |
| Foundation depth [m] ⓘ | 0.3 | 0–5 | the concrete pad under the floor |
| Soil under it [m] ⓘ | 3.0 | 0–20 | `ground_depth`: moist sand (1.5 W/(m K), VDI 4640) down to the ground-temperature face; `base_z` = soil + pad |
| Radial insulation [m] | 0.3 | 0.02–1 | `insulation_thickness` |
| Bottom / Top slab [m] | 0.2 / 0.2 | 0–1 | `insulation_slab_bottom` / `insulation_slab_top` |
| Steel shell [m] ⓘ | 0.02 | 0–0.2 | `shell_thickness` (painted one leaf thick if thinner) |
| Foundation margin [m] ⓘ | 0.5 | 0–3 | `foundation_margin` |
| Conical roof ⓘ / Roof angle [deg] | on / 15 | 0–45 | `enable_cone_roof` / `roof_angle_deg` |
| Steel plate [m] ⓘ | 0.005 | 0–0.2 | `steel_slab_top` |
| Cone fill | off | – | `fill_cone_with_sand` |
| Storage medium | Steatite (soapstone) | 7 media | `storage_material` |
| Packing [%] ⓘ | 63 | 20–90 | `packing_fraction` |
| Grain size [mm] ⓘ | 1.0 | 0.05–50 | `particle_diameter`: the radiation between the grains grows with it |
| Bed | read-out | – | the Zehner-Bauer-Schlünder conductivity at 20 and 500 °C, rho, cp (steatite: 0.30 and 0.57 W/(m K), 1701 kg/m³, 980 J/(kg K)) |
| Insulation / Shell | Rock wool / Carbon steel | 5 / 3 | `insulation_material` / `shell_material` |

## 2. Plant — Gas circuit

| control | default | range | feeds |
|---|---|---|---|
| Rated power [kW] ⓘ | **200** | 0–100 000 | `HeaterConfig.power_total`: the charging power, the constant power of a transient |
| Gas | Air | air / nitrogen / steam | `RunConfig.pipe_fluid` |
| Mass flow [kg/s] ⓘ | 1.0 | 0–200 | `RunConfig.pipe_flow`: a discharge needs at least `P / (cp dT)` |
| Pressure [bar] ⓘ | 1.01325 ("1 bar") | 0.1–100 | `RunConfig.pipe_pressure` |
| Fan efficiency [%] ⓘ | 70 | 10–100 | `RunConfig.pipe_fan_efficiency` |
| Circuit | read-out | – | the rated power over the wetted surface of the network (the **mean wall heat flux**, kW/m²), the power per riser, flow and pressure |

The wall heat flux is reported, not judged: the 3–8 W/cm² window it was once compared with
is the rating of an immersion element's sheath, and a gas-heated tube is limited by the
gas film instead.

## 3. Plant — Buried pipes

| control | default | range | feeds |
|---|---|---|---|
| Layout | Concentric rings | 5 layouts | `PipeNetworkConfig.layout` |
| Rings ⓘ | 6 | 1–20 | `n_rings`, enabled for the ring layout: spread over the whole radius, risers on a ring spaced like the rings (126 risers by default) |
| Radial files | 12 | 3–72 | `n_files`, enabled for the radial layout |
| Collection ⓘ | Radial manifold (rings in parallel) | 5 modes | `collection`; the header engine may pick the manifold instead of a chain and sets the combo to it |
| Pipe outer d [m] ⓘ | 0.05 | 0.01–0.3 | `diameter` |
| Wall thickness [mm] ⓘ | 2.0 | 0–20 | `wall_thickness` |
| Tube material ⓘ / Wall roughness [um] ⓘ | stainless / "from the material" | – / 0–2000 | `material` / `roughness` |
| Insulated headers ⓘ | off ("lagged") | – | `insulated_headers` |
| Inlet / Outlet azimuth [deg] | 180 / 0 | 0–360 | `azimuth_in` / `azimuth_out` |
| Flow split ⓘ | From the network hydraulics | hydraulic + 4 imposed rules | `split_mode`: the flows the pressures give (the plant), or an imposed rule for a comparison |
| Sectors ⓘ | 4 | 1–16 | `n_sectors`, enabled for the sector split |
| *Rebuild the network on the mesh* | – | – | repaints on the current mesh (Build mesh does it too) |
| Network ⓘ | read-out | – | two lines (risers, tube area, flow spread, painted cells); the full `summary()`, paint report, wall heat flux and design notes - each once - are its tooltip |

Every edit of the Vessel and Plant pages redraws the geometry preview after 250 ms, with
the network as the panels describe it now (laid out on a throw-away tree of eight
leaves, `ThermalBatteryGUI.preview_network`); the status bar says when the built mesh no
longer matches.  A pipe cell keeps the bed's properties and carries the pipe's exchange;
the gas loop writes its film in every analysis.

## 3b. Plant — Header engine

| control | default | range | feeds |
|---|---|---|---|
| Flow uniformity [%] ⓘ | 5 | 0.5–50 | the tolerance under which no orifice is added |
| Max header velocity [m/s] ⓘ | 20 | 3–60 | the velocity limit of every header and duct |
| Design gas temperature [°C] ⓘ | 500 | 20–1000 | the gas density the headers are sized at |
| *Size the headers* | – | – | runs the engine now; *Build mesh* runs it when the plant changed |
| Design ⓘ | read-out | – | collection, header sizes, lift, pressure drop, velocity; the tooltip lists every group, the orifices and their plate holes |

The header and duct diameters are the engine's ([19](19_HYDRAULICS.md)); there is no
duct control any more.  Default plant: radial manifold, headers 60–406 mm, 243 mm into
the sand, 751 Pa, 17.1 m/s, orifices on 124 risers (holes 23.7–46 mm).

## 4. Site

| control | default | range | feeds |
|---|---|---|---|
| Ambient [°C] | 20 | −40–80 | `t_ambient` |
| Ground [°C] ⓘ | 10 | −20–60 | `t_ground` (held under the foundation) |
| Wind speed [m/s] ⓘ | 0 | 0–30 | `wind_speed`: `4 + 4 v` added to the natural convection of the vessel |

## 5. Mesh

| control | default | range | feeds |
|---|---|---|---|
| Cells across storage ⓘ | 10 | 2–200 | the bed's plan edge `r / n` = 200 mm → 195 mm leaves |
| Cells across insulation ⓘ | 3 | 1–50 | the insulation ring's plan edge (100 mm → 97 mm), and the height of the slabs (200 mm / 3 → 75 mm) |
| Layers in the bed height ⓘ | 20 | 2–400 | the height of the bed's and the ring's leaves: 250 mm → 301 mm |
| Cell budget ⓘ | 400 000 | 10 000–20 000 000 | the most leaves the build may make |
| Regions / Mesh / Memory | read-outs | – | plan x height of every region, the leaves per size after a build, whether the budget stopped it, a warning when the bed's leaves are too small for the well model (under 2.53 tube diameters) |

The mesh is the anisotropic tree of boxes ([18](18_SOLVER.md) §1): the finest plan edge
and the finest height are the finest regions' targets, snapped to powers of two of the
box; every target snaps to the largest leaf within 1.6 times it (`LEAF_TOLERANCE`); the
budget caps the leaves.  No column follows the tubes any more: a tube is a line in its
cell, coupled by the well model.  Default plant: **36 268 leaves**, built and painted in
~2.3 s.

**Automatic mesh** (same page): tolerances 2 K / 2 %, 4 levels, refine factor 0.6, *Find
the mesh*.  Each level is the **standby** state on a finer tree; a round splits a leaf in
plan, in height or both, following the axis of its flux jump.

## 6. Analysis

**Type**

| control | default | range |
|---|---|---|
| Steady standby (hold the storage at a temperature) | **selected** | the holding power = the standby losses |
| Transient (charge / discharge profiles) | – | the profiles drive the loop |
| Standby: Mean storage T [°C] | 500 | 20–1200 |
| Duration / Unit | 24 / hours | 0.01–100 000 |
| Time step dt [s] ⓘ | 900 | 0.1–86 400 — an accuracy choice, the coupling is implicit |
| Save interval [s] | 3600 | 1–86 400 |
| Full fields | off | save T per sample |

**Initial**: uniform (20 °C), per material (sand, insulation, steel, concrete, pipes),
the current field (the last run or a loaded state), or the standby state at the Type
page's temperature.

**Charge** (the resistors): off, constant (the rated power of the Plant page), a schedule
(t [s], P [W]) or a CSV file.

**Discharge** (the exchanger): off, *Exchanger power* (100 kW; the run stops, keeping its
samples, when the bed can only give it with gas colder than the return temperature), or
*Exchanger return temperature* (60 °C; the power is what the bed gives).  The return
temperature is the floor of the power mode too.

**Save / Load**: HDF5 state with the geometry hash; a loaded field is the start of the next
transient with *Initial > Current field*.

## 7. Solver

| control | default | options |
|---|---|---|
| Temperature tolerance [K] ⓘ | **0.1** | 0.01–5: stops the coupled iterations (gas loop, radiation) and the standby target |
| Linear residual ⓘ | 1e-6 | 1e-5 / 1e-6 / 1e-8 |
| Max linear iterations | 2000 | 100–200 000 |
| Max standby iterations ⓘ | 20 | 1–200 (it needs 2-3) |
| Threads ⓘ | All − 1 | Auto / All − 1 / 2 / 1 (through `threadpoolctl`) |
| Radiation ⓘ | Off | Off / On |

The linear method is fixed: CG + AMG Ruge-Stüben with a V(1,1) Gauss-Seidel cycle
(forward down, backward up), the fastest option measured ([12](12_METHODS.md) §3).

## 8. 3D view and results

The 3D view's field selector lists **Temperature, Material, Sources, Conductivity and
Geometry**: the geometry is one entry among the others and the fields come back from it
(the *Show geometry* button selects it); every pipe is drawn at its own diameter.  The
3D view (field, cut axis, position, opacity, reset camera; `coolwarm`) and the results
tabs (Statistics, Energy balance with the gas loop, Materials, Transient, Log) are
unchanged; the two exports are *Series CSV...* and *Field VTK...* under the results.
