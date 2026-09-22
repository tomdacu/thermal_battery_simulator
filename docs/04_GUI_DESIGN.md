# 4. GUI design

`gui/`.  The window owns no physics: it collects parameters, starts jobs and
displays results.

## 1. Layout

```
┌────────────────────────┬───────────────────────────┬──────────────────┐
│ left column            │  3D view (PyVista)        │ results tabs     │
│  1. Geometry           │  field / cut / position   │  Statistics      │
│     Cylinder           │  opacity / reset camera   │  Energy balance  │
│     Insulation         │  (colormap fixed)         │  Materials       │
│     Heaters (circuit)  │                           │  Transient       │
│     Pipes (network)    │                           │  Log             │
│     Mesh               │                           │                  │
│  2. Materials          │                           │                  │
│     Storage            │                           │                  │
│     Insulation         │                           │                  │
│     Conditions         │                           │                  │
│  3. Analysis           │                           │                  │
│     Type, Initial      │                           │                  │
│     Power, Extraction  │                           │                  │
│     Save / Load        │                           │                  │
│  4. Tools              │                           │                  │
│     Solver, Losses     │                           │                  │
│     Help               │                           │                  │
├────────────────────────┤                           │                  │
│ Build mesh             │                           │                  │
│ Preview geometry       │                           │                  │
│ Run │ Cancel           │                           │                  │
│ progress bar           │                           │                  │
└────────────────────────┴───────────────────────────┴──────────────────┘
```

The three panes sit in a `QSplitter` (initial sizes 520 / 560 / 420; the left column
is capped at 560 px so the 3D view keeps the room).  The window is 1500 x 950 by
default.

* **Geometry** – domain, cylinder, insulation, the gas circuit (`Heaters`), the buried
  pipe network (`Pipes`) and the mesh (`GeometryPanel`, five sub-tabs).  The circuit and
  the network are the plant: the resistors heat the gas, the gas crosses the tube walls,
  and the same network charges and discharges the bed.  There is no lumped-tube tab
  (2026-09-22): the exchanger is on the circuit and the loop's inlet is its return.
* **Materials** – storage medium, insulation and shell, ambient/ground
  conditions and the radiation switch.  This tab is the *only* place where
  ambient and ground values are defined; the analyses read them.
* **Analysis** – analysis type, initial condition, power profile, extraction
  profile, state save/load.
* **Tools** – linear solver settings, losses-iteration controls, help text.

The action row is `Build mesh`, `Preview geometry`, `Run`, `Cancel` plus a progress
bar; `Run` renames itself to `Run steady state` / `Run losses analysis` /
`Run transient` when the analysis type changes (`_on_analysis_changed`).

## 2. Panels and getters

Each panel exposes pure getters that return `src` objects, so the window never
parses strings:

| panel | getters |
|---|---|
| `GeometryPanel` | `domain()`, `cylinder()`, `heaters()`, `apply_geometry(battery)`, `build_mesh()`, `grid_spec()`, `mesh_regions()`, `wants_auto_search()`, `auto_mesh_settings()`, `set_plan_targets(...)`, `planned(name, manual)`, `auto_spec()`, `set_auto_spec(...)`, `pipe_network_config()`, `pipe_paint_settings()`, `pipe_mass_flow()`, `circuit_fluid()`, `inlet_temperature()`, `circuit_pressure_pa()`, `fan_efficiency_fraction()`, `pipe_surface_power()`, `set_pipe_network(...)`, `set_mesh_info(...)` |
| `MaterialsPanel` | `storage_key()`, `insulation_key()`, `shell_key()`, `packing_fraction()`, `conditions()`, `radiation_enabled()`, `refresh_info()` |
| `AnalysisPanel` | `analysis_type()`, `initial_condition()`, `wants_steady_initial_condition()`, `power_profile()`, `extraction_profile()`, `transient_settings()`, `losses_target_kelvin()` |
| `SolverPanel` | `settings()`, `losses_settings()`, `threads()` |
| `ResultsPanel` | `update_statistics()`, `update_energy()`, `update_materials()`, `update_transient()`, `log()` |
| `VizView` | `show_mesh(mesh, field=None)`, `show_geometry(battery, mesh=None)`, `render()`, `enabled` |

Combos store their machine value in `itemData` (`gui/widgets.py::combo`), which
removes the class of bug that silently pinned tolerance/precision/thread
selections to defaults.  There is no `backend()` getter any more: the GPU backends
were removed and the thread budget is the only backend control (`threads()`).

The Pipes-related getters are the ones being extended together with the network itself
(wall data, `paint`, hydraulics, the circuit mass flow): [06](06_GUI_CONFIGURATION.md) §6
carries the reading of the panel and [15](15_PIPE_NETWORKS.md) is the authority for what
a network is.

## 3. The run cycle

```
Build mesh ─► GeometryPanel.build_mesh() ─► BatteryGeometry.apply_to_mesh ─► BuildReport
                 │ (validation errors → dialog, no silent clip)
                 ▼
Run ─► _run_config() ─► RunConfig ─► SimulationController.start(config, mesh)
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
             SimulationJob (QThread)             buttons disabled,
             progress / cancel                   Cancel enabled
                    │
        ┌───────────┼────────────┬──────────────┬──────────────┐
        ▼           ▼            ▼              ▼              ▼
   steady run   losses run   transient    automatic mesh   (results)
        └───────────┴────────────┴──────────────┘
                    ▼
      _on_finished → results tabs + 3D refresh + status
```

* **Build mesh** first recomputes the *a priori* mesh plan
  (`src/analysis/mesh_plan.py`) from the panels, then - when the refined mode and
  *Automatic mesh* are on and no spec has been adopted yet - runs the automatic
  search before building (`find_auto_mesh(build_after=True)`).
* `SimulationController` serialises runs: a second run cannot start while one is
  in flight, and `running_changed` owns button enablement.
* Every job callable receives `progress_callback` and `should_stop`; **Cancel**
  sets the flag and the loops exit at the next checkpoint, leaving the mesh in a
  consistent state.  Closing the window cancels first and waits up to 5 s for the
  worker to unwind (a `QThread` destroyed while running would abort the process).
* Exceptions inside a job are reported through the `failed` signal and shown in a
  dialog with the message; nothing is swallowed.
* `RunConfig` (`gui/controller.py`) is the single object that crosses the thread
  boundary: analysis type, battery, domain, optional `mesh_spec`, the search
  tolerances, solver settings, losses settings, transient settings, the three
  profiles and `start_from_steady`.

## 4. Results

| tab | content |
|---|---|
| Statistics | min/max/mean/std and percentiles in degC, for the whole domain and for the storage region; grid size, cell size, domain |
| Energy balance | `P_in`, `P_extracted`, envelope losses (top/side/bottom) and the box-face audit, stored energy and exergy in kWh, thermal autonomy, the imbalance self-check; after a losses run the required power and density; after a transient the cumulative energies |
| Materials | cell counts per material, packed-bed properties of the storage and insulation |
| Transient | one row per saved sample (T mean/max/min, P heaters/extracted, losses) and the CSV export |
| Log | every message from the controller and the solvers (solver notes, geometry report, the automatic-mesh result, warnings), capped to avoid unbounded growth |

The steady and losses runs also log the solver outcome (`converged=...`,
`residual=...`) and the losses iteration count.

## 5. 3D view

`VizView` builds a PyVista grid from the mesh (`src/viz/scene.py`) with **cell data
in Fortran order** - the same ordering as every field in the code - applies a clip
plane along x/y/z with a position and an opacity, and shows the material legend.
The controls are two compact rows: `Field` (Temperature, Material, Sources,
Conductivity), `Cut` (x/y/z) and **Reset camera** on the first, the *Position*
slider (1-99 %, default 50) and the *Opacity* slider (5-100 %, default 80) with
their value labels on the second.

The field colourmap is **fixed to `coolwarm`** - the colormap selector was removed -
and Temperature is displayed in degC; the conversion happens here and nowhere else.
Sources are shown in W/m³ and Conductivity in W/(m·K) (`scene.FIELD_UNITS`).

`Preview geometry` uses the same controls: the cut and the opacity apply to the
schematic view as well, so the battery can be inspected from the inside before
building the mesh.

The scene itself is built by `src/viz/scene.py` (`add_field`, `add_material_legend`,
`add_geometry_preview`), which is also what the tests exercise off-screen; the
widget only maps its controls onto those calls.  A clip plane that leaves nothing
to draw returns `None` and the view prints "nothing to show" instead of handing an
empty mesh to PyVista (which raises); any other rendering error is caught and
printed in the plot as well, so the view can never take the application down.

If no OpenGL context is available (head-less session, VM, remote desktop) the
widget degrades to a placeholder explaining why, and the rest of the application
keeps working; `THERMAL_DISABLE_3D=1` forces that mode (used by the test suite).

`Preview geometry` draws the schematic configuration (shell, insulation, storage,
roof, foundation, and the pipe network) straight from the geometry object, so what you
see is what `apply_to_mesh` will paint.

## 6. Units at the boundary

`gui/units.py` is the only module that converts:

* every spin box is labelled in degC and converted with `c_to_k` when the config
  object is built;
* every displayed temperature is converted back for the labels, tables and
  exports;
* powers are W, lengths m, time s throughout the GUI.

A `ValueError` from `check_kelvin` therefore means a bug in the GUI layer, not in
the physics.
