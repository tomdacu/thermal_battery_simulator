# User Interface (GUI)

## 1. Introduction

The graphical interface is developed in **PyQt6** and provides an integrated environment for configuration, execution, and analysis of simulations. The main file is `gui/main_window.py`.

---

## 2. Window Structure

The GUI is divided into three main areas:

### 2.1 Side Panel (Input)
Allows configuration of all simulation parameters:
- **Geometry**: Domain dimensions, storage radius, insulation thickness.
- **Mesh**: Grid resolution and target spacing.
- **Materials**: Selection of storage and insulation materials.
- **Heaters**: Total power, pattern, number of elements.
- **Tubes**: Fluid temperature, convection coefficient, pattern.
- **Solver**: Method selection (Direct, CG, BiCGStab, GMRES), tolerance, preconditioner.

### 2.2 Central Area (3D Visualization)
Uses `PyVistaQt` to integrate an interactive 3D rendering engine:
- Temperature field visualization.
- Material distribution visualization.
- **Slicing** tools (X, Y, Z section planes) to inspect the battery interior.
- Temperature isosurfaces.

### 2.3 Bottom Panel (Results)
Shows data derived from the simulation:
- Solver log (computation time, residual, iterations).
- Power balance (P_in, P_out, Losses).
- Total stored energy [kWh, MWh].

---

## 3. Simulation Management

### 3.1 Threading
To avoid blocking the interface during intensive calculations, the simulation runs in a separate thread (`SimulationThread`). This allows:
- Keeping the 3D visualization responsive.
- Updating a progress bar in real time.
- Stopping the simulation if needed.

### 3.2 User Workflow
1.  **Configuration**: User sets parameters in the widgets.
2.  **Preview Geometry**: (Optional) Visualization of tubes/heaters without mesh.
3.  **Build Mesh**: Create the 3D mesh and apply geometry.
4.  **Run Simulation**: Execute the steady-state solver.
5.  **Analysis**: Explore results using section planes and charts.

---

## 4. Tab Organization

### 4.1 Geometry Tab
| Widget Group | Contents |
|--------------|----------|
| Domain | Lx, Ly, Lz domain dimensions |
| Battery | Radius, height, center position |
| Insulation | Lateral thickness, top/bottom slabs |
| Roof | Enable cone, angle |
| Mesh | Target spacing, resulting cell count |

### 4.2 Heaters Tab
| Widget Group | Contents |
|--------------|----------|
| Power | Total power [kW] |
| Pattern | Distribution pattern (Uniform, Grid, Radial, Spiral) |
| Elements | Number, radius, spacing |
| Preview | Visual preview of positions |

### 4.3 Tubes Tab
| Widget Group | Contents |
|--------------|----------|
| Status | Active/inactive toggle |
| Fluid | Temperature, convection coefficient |
| Pattern | Distribution pattern |
| Elements | Number, diameter |

### 4.4 Solver Tab
| Widget Group | Contents |
|--------------|----------|
| Method | Direct, CG, BiCGStab, GMRES |
| Preconditioner | None, Jacobi, ILU, AMG |
| Tolerance | Convergence tolerance (1e-4 to 1e-12) |
| Threads | Number of CPU threads |
| Tips | Usage recommendations |

### 4.5 Help/Guide Tab
Comprehensive guide with:
- Usage instructions
- Performance optimization tips
- Solver recommendations
- Troubleshooting

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

### 5.3 Colormap Controls
| Widget | Purpose |
|--------|---------|
| Colormap | Color scheme (coolwarm, jet, viridis, plasma...) |
| T_min, T_max | Manual color range |
| Auto range | Auto-compute range from data |

---

## 6. Action Buttons

| Button | Action | Enables |
|--------|--------|---------|
| 👁 Preview Geometry | Preview cylinders/tubes/heaters without mesh | - |
| 🔧 Build Mesh | Create mesh + apply geometry | Run Simulation |
| ▶ Run Simulation | Run steady-state solver | Results panels |
| 📊 Export Results | Export to CSV/VTK | - |

---

## 7. Output Panels

### 7.1 Statistics Tab
- T_min, T_max, T_mean, T_std
- Mesh dimensions and total nodes
- Computation time

### 7.2 Energy Balance Tab
- P_input (heaters)
- P_output (tubes, if active)
- P_losses (top, lateral, bottom)
- Energy stored (kWh, MWh)

### 7.3 Materials Tab
- Distribution of material types
- Properties of selected storage material
- Volume fractions

### 7.4 Log Tab
- Chronological log of operations
- Solver messages and convergence info

---

## 8. Export Options

| Menu Item | Format | Content |
|-----------|--------|---------|
| Save Results | CSV | Temperature statistics |
| Export VTK | VTI | Full 3D field data (PyVista/ParaView) |
| Screenshot | PNG/JPG | Current 3D view |

---

## 9. GUI Requirements

For proper GUI functionality, the following are required:
- `PyQt6`: Window framework
- `pyvista`: Rendering engine
- `pyvistaqt`: Integration between PyVista and Qt

---

## 10. Future Developments
- 2D plots of temporal evolution (for transient simulations).
- Result export in additional formats.
- Editable material database directly from interface.
- Parameter presets and project saving.
