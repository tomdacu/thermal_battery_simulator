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

## 3. GUI Tab Structure

The GUI uses a **4-tab structure** for organized parameter access:

```
[Geometry] [Materials] [Analysis] [Tools]
```

---

## 4. Geometry Tab Parameters

### 4.1 Sub-tab: Cylinder

**Widget Group: Domain**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `lx_spin` | Lx | 1-50 m | 6.0 | `Mesh3D.Lx` |
| `ly_spin` | Ly | 1-50 m | 6.0 | `Mesh3D.Ly` |
| `lz_spin` | Lz | 1-50 m | 5.0 | `Mesh3D.Lz` |
| `radius_spin` | R | 0.5-20 m | 2.3 | `CylinderGeometry.r_storage` |
| `height_spin` | H | 1-30 m | 4.0 | `CylinderGeometry.height` |

**Widget Group: Roof**

| Widget | Variable | Type | Default | Maps To |
|--------|----------|------|---------|---------|
| `cone_check` | enable_cone | bool | False | `CylinderGeometry.enable_cone_roof` |
| `roof_angle_spin` | angle | 0-45° | 15 | `CylinderGeometry.roof_angle_deg` |

### 4.2 Sub-tab: Insulation

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `insul_thick_spin` | thickness | 0.1-1.0 m | 0.3 | `CylinderGeometry.insulation_thickness` |
| `shell_thick_spin` | shell | 0.01-0.1 m | 0.02 | `CylinderGeometry.shell_thickness` |
| `slab_bottom_spin` | bottom | 0.1-1.0 m | 0.3 | `CylinderGeometry.insulation_slab_bottom` |
| `slab_top_spin` | top | 0.1-1.0 m | 0.3 | `CylinderGeometry.insulation_slab_top` |

### 4.3 Sub-tab: Heaters

**Widget Group: Power**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `power_spin` | P_total | 1-10000 kW | 50 | `HeaterConfig.power_total` |

**Widget Group: Pattern**

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `heater_pattern_combo` | pattern | Uniform, Grid, Radial, Spiral | Uniform | `HeaterConfig.pattern` |

**Widget Group: Elements**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `n_heaters_spin` | n | 1-100 | 12 | `HeaterConfig.n_heaters` |
| `heater_radius_spin` | r | 5-100 mm | 20 mm | `HeaterConfig.heater_radius` |
| `heater_grid_rows` | rows | 1-20 | 4 | `HeaterConfig.grid_rows` |
| `heater_grid_cols` | cols | 1-20 | 4 | `HeaterConfig.grid_cols` |

### 4.4 Sub-tab: Tubes

**Widget Group: Status**

| Widget | Variable | Type | Default | Maps To |
|--------|----------|------|---------|---------|
| `tubes_active_check` | active | bool | False | `TubeConfig.active` |
| `tube_t_fluid_spin` | T_fluid | 10-200°C | 60 | `TubeConfig.T_fluid` |
| `tube_h_fluid_spin` | h | 10-5000 W/m²K | 500 | `TubeConfig.h_fluid` |

**Widget Group: Pattern**

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `tube_pattern_combo` | pattern | Cluster, Radial, Grid | Radial | `TubeConfig.pattern` |

**Widget Group: Elements**

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `n_tubes_spin` | n | 1-50 | 8 | `TubeConfig.n_tubes` |
| `tube_diameter_spin` | D | 10-200 mm | 50 mm | `TubeConfig.diameter` |

### 4.5 Sub-tab: Mesh

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `spacing_spin` | d | 0.05-1.0 m | 0.2 | `Mesh3D.target_spacing` |

---

## 5. Materials Tab Parameters

### 5.1 Sub-tab: Storage

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `storage_combo` | storage | steatite, silica_sand, ... | steatite | `BatteryGeometry.storage_material` |
| `packing_spin` | packing | 50-75% | 63% | `BatteryGeometry.packing_fraction` |

### 5.2 Sub-tab: Insulation

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `insulation_combo` | insulation | rock_wool, glass_wool, ... | rock_wool | `BatteryGeometry.insulation_material` |

### 5.3 Sub-tab: Conditions

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `t_amb_spin` | T_ambient | -20 to 50°C | 20 | Boundary conditions |
| `h_ext_spin` | h_ext | 5-50 W/m²K | 10 | External convection |

---

## 6. Analysis Tab Parameters

### 6.1 Sub-tab: Type

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `analysis_type_combo` | type | Steady, Losses, Transient | Steady | Analysis type |
| `target_temp_spin` | T_target | 100-800°C | 400 | Losses analysis target |

### 6.2 Sub-tab: Initial Conditions

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `ic_type_combo` | type | Uniform, By Material, From File | Uniform | Initial condition type |
| `ic_temp_spin` | T_init | 0-100°C | 20 | Initial temperature |

### 6.3 Sub-tab: Power Profile

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `power_profile_combo` | type | Off, Constant, Scheduled, CSV | Off | `PowerProfile.type` |
| `power_value_spin` | power | 0-10000 kW | 0 | `PowerProfile.power` |

### 6.4 Sub-tab: Extraction

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `extraction_type_combo` | type | Off, Imposed, Flow Rate | Off | `ExtractionProfile.type` |
| `extraction_value_spin` | value | 0-10000 kW | 0 | `ExtractionProfile.power` |

---

## 7. Tools Tab Parameters

### 7.1 Sub-tab: Solver

The Solver sub-tab is organized into **three groups**:

#### Common Settings

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `solver_combo` | method | cg, bicgstab, gmres, direct | cg | `SolverConfig.method` |
| `precond_combo` | preconditioner | jacobi, ilu, amg, none | amg | `SolverConfig.preconditioner` |
| `tolerance_spin` | tol | 1e-12 to 1e-2 | 1e-8 | `SolverConfig.tolerance` |
| `max_iter_spin` | max_iter | 100-100000 | 10000 | `SolverConfig.max_iterations` |

#### Performance

| Widget | Variable | Options | Default | Maps To |
|--------|----------|---------|---------|---------|
| `gpu_backend_combo` | backend | CPU, CUDA, OpenCL | CPU | `SolverConfig.gpu_backend` |
| `cpu_threads_spin` | threads | 1-32 | auto | `SolverConfig.n_threads` |
| `precision_combo` | precision | float64, float32, float16 | float64 | `SolverConfig.precision` |

#### Losses Analysis

| Widget | Variable | Range | Default | Maps To |
|--------|----------|-------|---------|---------|
| `losses_tol_spin` | tolerance | 0.1-10 °C | 1.0 | Losses convergence tolerance |
| `losses_maxiter_spin` | max_iter | 5-100 | 20 | Maximum losses iterations |
| `losses_alpha_spin` | α | 0.1-1.0 | 0.5 | Underrelaxation factor |
| `losses_h_conv_spin` | h_conv | 1-100 W/m²K | 10.0 | External convection coefficient |
| `losses_T_ground_spin` | T_ground | -10 to 30 °C | 15.0 | Ground temperature |

---

## 8. How Parameters Flow to the Solver

### 8.1 The Central Function: `_build_battery_geometry_from_inputs()`

This function reads ALL GUI widgets and creates configuration objects:

```python
def _build_battery_geometry_from_inputs(self):
    """Builds BatteryGeometry from UI controls (without creating mesh)."""
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

### 8.2 Used By Both "Build Mesh" and "Preview Geometry"

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

## 9. Visualization Controls

### 9.1 Visualization Mode

| Mode | Description |
|------|-------------|
| Clip Section | Clips the volume at a plane, shows solid behind |
| Multi-Slice | Shows 5 parallel slices |
| Volume 3D | Semi-transparent volume rendering |
| Isosurface | Isosurfaces at constant temperature |

### 9.2 Slice Controls

| Widget | Purpose |
|--------|---------|
| `axis_combo` | X, Y, or Z axis |
| `slice_slider` | Position along axis (0-100%) |
| `field_combo` | Field to display (Temperature, Material, k, Q) |

### 9.3 Colormap Controls

| Widget | Purpose |
|--------|---------|
| `cmap_combo` | Color scheme (coolwarm, jet, viridis, ...) |
| `tmin_spin`, `tmax_spin` | Manual color range |
| `auto_range_check` | Auto-compute range from data |

---

## 10. Action Buttons

| Button | Action | Enables |
|--------|--------|---------|
| 👁 Preview Geometry | Preview cylinders/tubes/heaters without mesh | - |
| 🔧 Build Mesh | Create mesh + apply geometry | Run Simulation |
| ▶ Run Simulation | Run steady-state solver | Results panels |

---

## 11. Output Panels (in Tools Tab)

### 11.1 Sub-tab: Statistics
- T_min, T_max, T_mean, T_std (all in °C)
- Mesh dimensions (Nx × Ny × Nz) and total nodes

### 11.2 Sub-tab: Energy Balance
After Losses Analysis:
- Conditions: T_target, T_final, T_ambient, T_ground, h_conv
- Losses: Total [kW], breakdown by face (top, side, bottom)
- Energy: E_stored [kWh, MWh], Thermal autonomy [hours]
- Convergence: Status, number of iterations

### 11.3 Sub-tab: Materials Info
- Distribution of material types by volume
- Properties of selected storage material

### 11.4 Sub-tab: Export
- CSV export of temperature statistics
- VTK export of full 3D field data
- Screenshot of current 3D view

---

## 12. Losses Analysis Parameter Tuning

The losses analysis uses an iterative secant method. Here's how to tune the parameters:

### 12.1 Underrelaxation Factor (α)

| α Value | Behavior | Use Case |
|---------|----------|----------|
| 0.1-0.3 | Very stable, slow convergence | Large temperature differences |
| 0.4-0.6 | Balanced (recommended) | Most cases |
| 0.7-0.9 | Fast but may oscillate | Small refinements |
| 1.0 | No damping (pure Newton) | Very small changes only |

### 12.2 Tolerance

| Tolerance | Accuracy | Iterations |
|-----------|----------|------------|
| 0.1 °C | High precision | 15-25 |
| 1.0 °C | Standard (default) | 8-15 |
| 5.0 °C | Quick estimate | 3-6 |

### 12.3 Convection Coefficient (h_conv)

| h_conv | Condition |
|--------|-----------|
| 5-10 W/m²K | Natural convection, indoor |
| 10-25 W/m²K | Light wind, outdoor |
| 25-50 W/m²K | Strong wind |
| 50-100 W/m²K | Forced convection |

### 12.4 Ground Temperature (T_ground)

| T_ground | Location/Season |
|----------|-----------------|
| 5-10 °C | Northern Europe winter |
| 10-15 °C | Temperate regions |
| 15-20 °C | Southern Europe |
| 20-25 °C | Warm climates |
