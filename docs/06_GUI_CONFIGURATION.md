# GUI Configuration Guide

## 1. Introduction

This document explains how all simulation parameters flow from the GUI to the solver. The system is designed so that **no hardcoded values exist in the simulation engine** - everything is configured through the graphical interface.

---

## 2. Main Entry Points

| File | Purpose |
|------|---------|
| `run_gui.py` | Launches the GUI application |
| `main.py` | CLI script for testing (contains hardcoded test values) |

For production use, always use `run_gui.py`.

---

## 3. GUI Tabs and Their Parameters

### 3.1 Tab: Geometria (Geometry)

**Widget Group: Geometria Batteria**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `lx_spin` | Lx | 1-50 m | 6.0 | `Mesh3D.Lx` |
| `ly_spin` | Ly | 1-50 m | 6.0 | `Mesh3D.Ly` |
| `lz_spin` | Lz | 1-50 m | 5.0 | `Mesh3D.Lz` |
| `radius_spin` | R | 0.5-20 m | 2.3 | `CylinderGeometry.r_*` (scaled) |
| `height_spin` | H | 1-30 m | 4.0 | `CylinderGeometry.height` |

**Widget Group: Mesh**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `spacing_spin` | d | 0.05-1.0 m | 0.2 | `Mesh3D.target_spacing` |

**Widget Group: Materiali**

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `storage_combo` | storage | steatite, silica_sand, ... | steatite | `BatteryGeometry.storage_material` |
| `insulation_combo` | insulation | rock_wool, glass_wool, ... | rock_wool | `BatteryGeometry.insulation_material` |
| `packing_spin` | packing | 50-75% | 63% | `BatteryGeometry.packing_fraction` |

**Widget Group: Condizioni Operative**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `t_amb_spin` | T_ambient | -20 to 50°C | 20°C | Boundary conditions |
| `t_ground_spin` | T_ground | 0-30°C | 10°C | Boundary conditions |

---

### 3.2 Tab: Resistenze (Heaters)

**Widget Group: Potenza**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `power_spin` | P_total | 1-10000 kW | 50 | `HeaterConfig.power_total` |

**Widget Group: Pattern Distribuzione**

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `heater_pattern_combo` | pattern | Uniform, Grid, Radial, ... | Uniform | `HeaterConfig.pattern` |

**Widget Group: Parametri Elementi**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `n_heaters_spin` | n | 1-100 | 12 | `HeaterConfig.n_heaters` |
| `heater_radius_spin` | r | 5-100 mm | 20 mm | `HeaterConfig.heater_radius` |
| `heater_grid_rows` | rows | 1-20 | 4 | `HeaterConfig.grid_rows` |
| `heater_grid_cols` | cols | 1-20 | 4 | `HeaterConfig.grid_cols` |
| `heater_grid_spacing` | spacing | 0.1-2.0 m | 0.3 m | `HeaterConfig.grid_spacing` |
| `heater_n_rings` | rings | 1-10 | 2 | `HeaterConfig.n_rings` |

---

### 3.3 Tab: Tubi (Tubes)

**Widget Group: Stato Scambiatori**

| Widget | Variable | Type | Default | Maps To |
|--------|----------|------|---------|---------|
| `tubes_active_check` | active | bool | False | `TubeConfig.active` |
| `tube_t_fluid_spin` | T_fluid | 10-200°C | 60°C | `TubeConfig.T_fluid` |
| `tube_h_fluid_spin` | h | 10-5000 W/m²K | 500 | `TubeConfig.h_fluid` |

**Widget Group: Pattern Distribuzione**

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `tube_pattern_combo` | pattern | Cluster, Radial, Grid, ... | Radial | `TubeConfig.pattern` |

**Widget Group: Parametri Elementi**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `n_tubes_spin` | n | 1-50 | 8 | `TubeConfig.n_tubes` |
| `tube_diameter_spin` | D | 10-200 mm | 50 mm | `TubeConfig.diameter` |

---

### 3.4 Tab: Solver

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `solver_combo` | method | direct, bicgstab, gmres, cg | direct | `SolverConfig.method` |
| `tolerance_spin` | tol | 1e-12 to 1e-2 | 1e-8 | `SolverConfig.tolerance` |
| `max_iter_spin` | max_iter | 100-100000 | 10000 | `SolverConfig.max_iterations` |

---

## 4. How Parameters Flow to the Solver

### 4.1 The Central Function: `_build_battery_geometry_from_inputs()`

This function reads ALL GUI widgets and creates configuration objects:

```python
def _build_battery_geometry_from_inputs(self):
    """Costruisce BatteryGeometry dai controlli UI (senza creare la mesh)."""
    # 1. Read domain dimensions
    d = self.spacing_spin.value()
    Lx, Ly, Lz = self.lx_spin.value(), self.ly_spin.value(), self.lz_spin.value()

    # 2. Create 4-zone cylinder geometry
    cylinder = CylinderGeometry(
        center_x=Lx / 2,
        center_y=Ly / 2,
        r_storage=self.radius_spin.value(),           # Storage zone radius
        insulation_thickness=self.insul_thick_spin.value(),  # Insulation layer
        shell_thickness=self.shell_thick_spin.value(),       # Steel shell
        height=self.height_spin.value(),
        insulation_slab_bottom=self.slab_bottom_spin.value(),
        insulation_slab_top=self.slab_top_spin.value(),
        enable_cone_roof=self.cone_check.isChecked(),
        roof_angle_deg=self.roof_angle_spin.value(),
        phase_offset_deg=self.phase_offset_spin.value(),
        ...
    )

    # 3. Create heater config from GUI
    heater_config = HeaterConfig(
        power_total=self.power_spin.value(),
        n_heaters=self.n_heaters_spin.value(),
        pattern=self._get_heater_pattern_enum(),
        ...
    )

    # 4. Create tube config from GUI
    tube_config = TubeConfig(
        n_tubes=self.n_tubes_spin.value(),
        active=self.tubes_active_check.isChecked(),
        ...
    )

    # 5. Create final geometry
    geom = BatteryGeometry(
        cylinder=cylinder,
        heaters=heater_config,
        tubes=tube_config,
        storage_material=self.storage_combo.currentText(),
        insulation_material=self.insulation_combo.currentText(),
        packing_fraction=self.packing_spin.value() / 100.0,
    )

    return d, Lx, Ly, Lz, geom
```

### 4.2 Used By Both "Build Mesh" and "Preview Geometry"

```python
def build_mesh(self):
    d, Lx, Ly, Lz, geom = self._build_battery_geometry_from_inputs()
    self.mesh = Mesh3D(Lx=Lx, Ly=Ly, Lz=Lz, target_spacing=d)
    geom.apply_to_mesh(self.mesh, self.mat_manager)
    
def preview_geometry(self):
    _, Lx, Ly, Lz, geom = self._build_battery_geometry_from_inputs()
    # Render analytic geometry without creating mesh
```

---

## 5. Visualization Controls

### 5.1 Visualization Mode

| Mode | Description |
|------|-------------|
| Sezione Clip | Clips the volume at a plane, shows solid behind |
| Multi-Slice | Shows 5 parallel slices |
| Volume 3D | Semi-transparent volume rendering |
| Isosuperficie | Isosurfaces at constant temperature |

### 5.2 Slice Controls

| Widget | Purpose |
|--------|---------|
| `axis_combo` | X, Y, or Z axis |
| `slice_slider` | Position along axis (0-100%) |
| `field_combo` | Field to display (Temperature, Material, k, Q) |

### 5.3 Colormap Controls

| Widget | Purpose |
|--------|---------|
| `cmap_combo` | Color scheme (coolwarm, jet, viridis, ...) |
| `tmin_spin`, `tmax_spin` | Manual color range |
| `auto_range_check` | Auto-compute range from data |

---

## 6. Action Buttons

| Button | Action | Enables |
|--------|--------|---------|
| 👁 Anteprima Geometria | Preview cylinders/tubes/heaters without mesh | - |
| 🗘 Costruisci Mesh | Create mesh + apply geometry | Esegui Simulazione |
| ▶ Esegui Simulazione | Run steady-state solver | Results panels |

---

## 7. Output Panels

### 7.1 Tab: Statistiche
- T_min, T_max, T_mean, T_std
- Mesh dimensions and total nodes

### 7.2 Tab: Bilancio Energetico
- P_input (heaters)
- P_output (tubes, if active)
- P_losses (top, lateral, bottom)
- Energy stored (kWh, MWh)

### 7.3 Tab: Materiali
- Distribution of material types
- Properties of selected storage material

### 7.4 Tab: Log
- Chronological log of operations and solver messages

---

## 8. Export Options

| Menu Item | Format | Content |
|-----------|--------|---------|
| Salva Risultati | CSV | Temperature statistics |
| Esporta VTK | VTI | Full 3D field data (PyVista/ParaView) |
| Screenshot | PNG/JPG | Current 3D view |
