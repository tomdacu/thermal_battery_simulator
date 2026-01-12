# User Interface (GUI)

## 1. Introduction

The graphical interface is developed in **PyQt6** and provides an integrated environment for configuration, execution, and analysis of simulations. The main file is `gui/main_window.py`.

The GUI uses a clean, streamlined design with:
- **No menu bar** - all functions accessible through tabs
- **No toolbar** - actions via buttons in the interface
- **4 main tabs** for organized parameter access

---

## 2. Window Structure

The GUI is divided into three main areas:

### 2.1 Side Panel (Input) - 4-Tab Structure

The side panel uses a **4-tab structure** for organized parameter access:

```
[Geometry] [Materials] [Analysis] [Tools]
```

Each main tab contains sub-tabs for detailed configuration:

| Main Tab | Sub-tabs |
|----------|----------|
| **Geometry** | Cylinder, Insulation, Heaters, Tubes, Mesh |
| **Materials** | Storage, Insulation, Operating Conditions |
| **Analysis** | Type, Initial Conditions, Power Profile, Extraction, Save/Load |
| **Tools** | Solver, Statistics, Energy Balance, Materials Info, Export, Help |

### 2.2 Central Area (3D Visualization)
Uses `PyVistaQt` to integrate an interactive 3D rendering engine:
- Temperature field visualization (in Celsius)
- Material distribution visualization
- **Slicing** tools (X, Y, Z section planes) to inspect the battery interior
- Single vertical colorbar for clean display

### 2.3 Bottom Panel (Results)
Shows data derived from the simulation:
- Solver log (computation time, residual, iterations)
- Progress bar during computation
- Status messages

---

## 3. Simulation Management

### 3.1 Threading
To avoid blocking the interface during intensive calculations, the simulation runs in a separate thread (`SimulationThread`). This allows:
- Keeping the 3D visualization responsive
- Updating a progress bar in real time
- Processing events during iterative analysis

### 3.2 User Workflow
1. **Configure Geometry** (Geometry tab): Define domain, cylinder, insulation, heaters, tubes, mesh
2. **Set Materials** (Materials tab): Select storage/insulation materials, operating conditions
3. **Configure Analysis** (Analysis tab): Choose analysis type, set profiles
4. **Configure Solver** (Tools > Solver): Select method, tolerance, CPU/GPU, losses parameters
5. **Build Mesh**: Click "Build Mesh" button
6. **Run Simulation**: Click "Run Simulation" button
7. **View Results** (Tools tab): Analyze statistics, energy balance, export data

---

## 4. Tab Organization

### 4.1 GEOMETRY Tab

#### Sub-tab: Cylinder
| Widget Group | Contents |
|--------------|----------|
| Domain | Lx, Ly, Lz domain dimensions [m] |
| Storage Cylinder | Radius, height [m] |
| Roof | Enable cone, angle, steel slab, fill with sand |

#### Sub-tab: Insulation
| Widget Group | Contents |
|--------------|----------|
| Radial Insulation | Insulation thickness, shell thickness [m] |
| Vertical Insulation | Bottom slab, top slab thickness [m] |

#### Sub-tab: Heaters
| Widget Group | Contents |
|--------------|----------|
| Power | Total power [kW] |
| Pattern | Distribution pattern (Uniform, Grid, Radial, Spiral) |
| Elements | Number, radius, spacing |
| Preview | Visual preview of positions |

#### Sub-tab: Tubes
| Widget Group | Contents |
|--------------|----------|
| Status | Active/inactive toggle |
| Fluid | Temperature, convection coefficient |
| Pattern | Distribution pattern |
| Elements | Number, diameter |

#### Sub-tab: Mesh
| Widget Group | Contents |
|--------------|----------|
| Spacing | Target cell spacing [m] |
| Info | Resulting cell count, memory estimate |

### 4.2 MATERIALS Tab

#### Sub-tab: Storage
| Widget Group | Contents |
|--------------|----------|
| Material | Material selection (Steatite, Sand, etc.) |
| Packing | Packing fraction [%] |
| Properties | Display of k, ρ, cp values |

#### Sub-tab: Insulation
| Widget Group | Contents |
|--------------|----------|
| Material | Insulation material selection |
| Properties | Display of k, ρ, cp values |

#### Sub-tab: Conditions
| Widget Group | Contents |
|--------------|----------|
| Environment | T_ambient [°C] |
| Convection | External convection coefficient h_ext [W/(m²·K)] |

### 4.3 ANALYSIS Tab

#### Sub-tab: Type
| Widget Group | Contents |
|--------------|----------|
| Analysis Type | Steady-state, Losses analysis, Transient |
| Steady | Heater power configuration |
| Losses | Target temperature, ambient temperature |
| Transient | Duration, time step, save interval |

#### Sub-tab: Initial Conditions
| Widget Group | Contents |
|--------------|----------|
| Type | Uniform, By Material, From File, From Steady |
| Temperature | Initial temperature settings per mode |

#### Sub-tab: Power
| Widget Group | Contents |
|--------------|----------|
| Profile Type | Off, Constant, Scheduled, From CSV |
| Parameters | Power values, timing |

#### Sub-tab: Extraction
| Widget Group | Contents |
|--------------|----------|
| Profile Type | Off, Imposed Power, Flow Rate, Target Outlet T |
| Parameters | Flow rate, fluid type, temperature |

#### Sub-tab: Save/Load
| Widget Group | Contents |
|--------------|----------|
| Save State | Save current simulation to HDF5 |
| Load State | Load simulation from HDF5 |

### 4.4 TOOLS Tab

#### Sub-tab: Solver
| Widget Group | Contents |
|--------------|----------|
| **Common Settings** | Method (cg, bicgstab, gmres, direct), Preconditioner, Tolerance, Max iterations |
| **Performance** | CPU threads / GPU selection (CUDA, OpenCL), Precision (float64/32/16) |
| **Losses Analysis** | Temperature tolerance, Max iterations, Underrelaxation α, h_conv, T_ground |
| **Tips** | Performance optimization suggestions |

#### Sub-tab: Statistics
| Widget Group | Contents |
|--------------|----------|
| Temperature | T_min, T_max, T_mean, T_std (all in °C) |
| Mesh | Dimensions (Nx × Ny × Nz), total nodes |

#### Sub-tab: Energy Balance
| Widget Group | Contents |
|--------------|----------|
| Conditions | T_target, T_final, T_ambient, T_ground, h_conv |
| Losses | Total losses (kW), breakdown by face (top, side, bottom) |
| Energy | E_stored (kWh, MWh), Thermal autonomy (hours) |
| Convergence | Status, number of iterations |

#### Sub-tab: Materials Info
| Widget Group | Contents |
|--------------|----------|
| Distribution | Volume fractions by material type |
| Properties | Selected material thermal properties |

#### Sub-tab: Export
| Widget Group | Contents |
|--------------|----------|
| Formats | CSV, VTK, HDF5 options |
| Screenshot | Save current 3D view |

#### Sub-tab: Help
| Widget Group | Contents |
|--------------|----------|
| Quick Guide | Usage instructions |
| Performance | Optimization tips |
| Troubleshooting | Common issues and solutions |

---

## 5. Visualization Controls

### 5.1 Visualization Mode
| Mode | Description |
|------|-------------|
| Clip Section | Clips the volume at a plane, shows solid behind |
| Multi-Slice | Shows 5 parallel slices |
| Volume 3D | Semi-transparent volume rendering |
| Isosurface | Isosurfaces at constant temperature |

### 5.2 Slice Controls
| Widget | Purpose |
|--------|---------|
| Axis selector | X, Y, or Z axis |
| Position slider | Position along axis (0-100%) |
| Field selector | Field to display (Temperature, Material, k, Q) |

### 5.3 Colorbar
- **Single vertical colorbar** on the right side
- Temperature displayed in **Celsius** (converted from internal Kelvin)
- Auto-ranging or manual T_min/T_max

---

## 6. Action Buttons

| Button | Action | Enables |
|--------|--------|---------|
| 👁 Preview Geometry | Preview cylinders/tubes/heaters without mesh | - |
| 🔧 Build Mesh | Create mesh + apply geometry | Run Simulation |
| ▶ Run Simulation | Run selected analysis type | Results panels |

---

## 7. Analysis Types

### 7.1 Steady-State Analysis
- Solves equilibrium temperature distribution with constant heater power
- Single solver call
- Results: temperature field, power balance

### 7.2 Losses Analysis (Iterative)
Uses an **iterative secant method** to find thermal losses:

1. **Input**: Target sand temperature (T_target), Ambient temperature (T_amb)
2. **Algorithm**:
   - Initialize temperatures (sand=T_target, insulation=interpolated, external=T_amb)
   - Set convection BC on external faces
   - Iterate:
     - Apply heat source Q to sand cells
     - Solve steady-state
     - Compute T_mean of sand
     - Adjust Q using secant method with underrelaxation
     - Repeat until |T_mean - T_target| < tolerance
3. **Output**: 
   - Q_total = thermal losses [kW]
   - Breakdown by face (top, side, bottom)
   - Thermal autonomy estimate
   - Physically consistent temperature profile

**Configurable Parameters** (in Solver sub-tab):
| Parameter | Description | Default |
|-----------|-------------|---------|
| Tolerance (°C) | Acceptable error on T_mean | 1.0 |
| Max iterations | Limit for Q-T iterations | 20 |
| α (underrelaxation) | Damping factor (0.1=stable, 1.0=fast) | 0.5 |
| h_conv | External convection coefficient | 10.0 W/(m²·K) |
| T_ground | Ground temperature | 15.0 °C |

### 7.3 Transient Analysis
- Time-dependent simulation with power/extraction profiles
- Backward Euler implicit scheme
- State save/load capability

---

## 8. Energy Balance Panel

After Losses Analysis, the Energy Balance panel shows:

```
╔══════════════════════════════════════════════╗
║      ENERGY BALANCE - LOSSES ANALYSIS        ║
╠══════════════════════════════════════════════╣
║ CONDITIONS                                   ║
╠──────────────────────────────────────────────╣
║ T target sand:    600.0 °C                   ║
║ T final mean:     599.8 °C                   ║
║ T ambient:         20.0 °C                   ║
║ T ground:          15.0 °C                   ║
║ h convection:      10.0 W/(m²·K)             ║
╠══════════════════════════════════════════════╣
║ THERMAL LOSSES                               ║
╠──────────────────────────────────────────────╣
║ TOTAL:            12.50 kW                   ║
║   - Top:           3.20 kW                   ║
║   - Side:          7.80 kW                   ║
║   - Bottom:        1.50 kW                   ║
╠──────────────────────────────────────────────╣
║ Loss density:     125.0 W/m³                 ║
╠══════════════════════════════════════════════╣
║ STORED ENERGY                                ║
╠──────────────────────────────────────────────╣
║ E thermal:       1250 kWh                    ║
║                   1.25 MWh                   ║
║ Autonomy:        100.0 hours                 ║
╠══════════════════════════════════════════════╣
║ CONVERGENCE: ✓ CONVERGED (8 iter)            ║
╚══════════════════════════════════════════════╝
```

---

## 9. GUI Requirements

For proper GUI functionality, the following are required:
- `PyQt6`: Window framework
- `pyvista`: Rendering engine
- `pyvistaqt`: Integration between PyVista and Qt

---

## 10. Future Developments
- 2D plots of temporal evolution
- Result export in additional formats
- Editable material database directly from interface
- Multi-physics coupling (flow + heat)
