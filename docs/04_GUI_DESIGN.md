# 4. GUI design

`gui/`.  The window owns no physics: it collects parameters, starts jobs and
displays results.

## 1. Layout

```
┌───────────────┬──────────────────────────────┬───────────────────────┐
│ left panel    │  3D view (PyVista)           │ results tabs          │
│  1 Geometry   │  field / clip / opacity      │  Statistics           │
│  2 Materials  │  colormap controls           │  Energy balance       │
│  3 Analysis   │                              │  Materials            │
│  4 Tools      │                              │  Transient            │
│               │                              │  Log                  │
├───────────────┤                              │                       │
│ Build mesh    │                              │                       │
│ Preview       │                              │                       │
│ Run │ Cancel  │                              │                       │
│ progress bar  │                              │                       │
└───────────────┴──────────────────────────────┴───────────────────────┘
```

* **Geometry** – domain, cylinder, insulation, heaters, tubes, cell size.
* **Materials** – storage medium, insulation and shell, ambient/ground
  conditions and the radiation switch.  This tab is the *only* place where
  ambient and ground values are defined; the analyses read them.
* **Analysis** – analysis type, initial condition, power profile, extraction
  profile, state save/load.
* **Tools** – linear solver settings, losses-iteration controls, help.

## 2. Panels and getters

Each panel exposes pure getters that return `src` objects, so the window never
parses strings:

| panel | getters |
|---|---|
| `GeometryPanel` | `domain()`, `cylinder()`, `heaters()`, `tubes()`, `apply_geometry(battery)`, `set_mesh_info(...)` |
| `MaterialsPanel` | `storage_key()`, `insulation_key()`, `shell_key()`, `packing_fraction()`, `conditions()`, `radiation_enabled()` |
| `AnalysisPanel` | `analysis_type()`, `initial_condition()`, `wants_steady_initial_condition()`, `power_profile()`, `extraction_profile()`, `transient_settings()`, `losses_target_kelvin()` |
| `SolverPanel` | `settings()`, `losses_settings()`, `backend()` |
| `ResultsPanel` | `update_statistics/energy/materials/transient`, `log` |

Combos store their machine value in `itemData` (`gui/widgets.py::combo`), which
removes the class of bug that silently pinned tolerance/precision/thread
selections to defaults.

## 3. The run cycle

```
Build mesh ─► Mesh3D(spacing) ─► BatteryGeometry.apply_to_mesh ─► BuildReport
                                   │ (validation errors → dialog, no silent clip)
                                   ▼
Run ─► _run_config() ─► RunConfig ─► SimulationController.start(config, mesh)
                                      │
                    ┌─────────────────┴──────────────────┐
                    ▼                                    ▼
             SimulationJob (QThread)             buttons disabled,
             progress / cancel                   Cancel enabled
                    │
        ┌───────────┼────────────┬─────────────────┐
        ▼           ▼            ▼                 ▼
   steady run   losses run   transient run    (results)
        └───────────┴────────────┘
                    ▼
      _on_finished → results tabs + 3D refresh + status
```

* `SimulationController` serialises runs: a second run cannot start while one is
  in flight, and `running_changed` owns button enablement.
* Every job callable receives `progress_callback` and `should_stop`; **Cancel**
  sets the flag and the loops exit at the next checkpoint, leaving the mesh in a
  consistent state.
* Exceptions inside a job are reported through the `failed` signal and shown in a
  dialog with the message; nothing is swallowed.

## 4. Results

| tab | content |
|---|---|
| Statistics | min/max/mean/std and percentiles in degC, for the whole domain and for the storage region; grid size, cell size, domain |
| Energy balance | `P_in`, `P_extracted`, envelope losses (top/side/bottom) and the box-face audit, stored energy and exergy in kWh, thermal autonomy, the imbalance self-check; after a losses run the required power and density; after a transient the cumulative energies |
| Materials | cell counts per material, packed-bed properties of the storage and insulation |
| Transient | one row per saved sample (T mean/max/min, P heaters/extracted, losses) and the CSV export |
| Log | every message from the controller and the solvers (solver notes, geometry report, warnings), capped to avoid unbounded growth |

## 5. 3D view

`VizView` builds a PyVista `ImageData` from the mesh (`src/viz/scene.py`) with
**cell data in Fortran order** - the same ordering as every field in the code -
applies a clip plane along x/y/z with an opacity, and shows the material legend.
The colormap is fixed to `coolwarm` (a single, consistent scale).  Temperature is
displayed in degC; the conversion happens here and nowhere else.

`Preview geometry` uses the same controls: the cut and the opacity apply to the
schematic view as well, so the battery can be inspected from the inside before
building the mesh.

The scene itself is built by `src/viz/scene.py` (`add_field`, `add_material_legend`,
`add_geometry_preview`), which is also what the tests exercise off-screen; the
widget only maps its controls onto those calls.  A clip plane that leaves nothing
to draw returns `None` and the view prints "nothing to show" instead of handing an
empty mesh to PyVista (which raises).

If no OpenGL context is available (head-less session, VM, remote desktop) the
widget degrades to a placeholder explaining why, and the rest of the application
keeps working; `THERMAL_DISABLE_3D=1` forces that mode (used by the test suite).

`Preview geometry` draws the schematic configuration (shell, insulation, storage,
roof, tubes) straight from the geometry object, so what you see is what
`apply_to_mesh` will paint.

## 6. Units at the boundary

`gui/units.py` is the only module that converts:

* every spin box is labelled in degC and converted with `c_to_k` when the config
  object is built;
* every displayed temperature is converted back for the labels, tables and
  exports;
* powers are W, lengths m, time s throughout the GUI.

A `ValueError` from `check_kelvin` therefore means a bug in the GUI layer, not in
the physics.
