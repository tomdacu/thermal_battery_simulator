# User Interface (GUI)

## 1. Introduction

The graphical interface is developed in **PyQt6** and provides an integrated environment for configuration, execution, and analysis of simulations. The main file is `gui/main_window.py`.

---

## 2. Window Structure

The GUI is divided into three main areas:

### 2.1 Side Panel (Input) - 2-Level Tab Structure

The side panel uses a **2-level tab structure** for organized parameter access:

**Level 1 (Main Tabs):**
```
[1. Geometria] [2. Materiali] [3. Analisi] [4. Risultati]
```

**Level 2 (Sub-tabs for each main tab):**

| Main Tab | Sub-tabs |
|----------|----------|
| **1. Geometria** | Cilindro, Isolamento, Resistenze, Tubi, Mesh |
| **2. Materiali** | Storage, Isolamento, Condizioni |
| **3. Analisi** | Tipo, Condizioni Iniziali, Potenza, Estrazione, Solver, Salvataggio |
| **4. Risultati** | Statistiche, Bilancio, Materiali, Esporta, 📖 Guida |

### 2.2 Central Area (3D Visualization)
Uses `PyVistaQt` to integrate an interactive 3D rendering engine:
- Temperature field visualization
- Material distribution visualization
- **Slicing** tools (X, Y, Z section planes) to inspect the battery interior
- Temperature isosurfaces

### 2.3 Bottom Panel (Results)
Shows data derived from the simulation:
- Solver log (computation time, residual, iterations)
- Power balance (P_in, P_out, Losses)
- Total stored energy [kWh, MWh]

---

## 3. Simulation Management

### 3.1 Threading
To avoid blocking the interface during intensive calculations, the simulation runs in a separate thread (`SimulationThread`). This allows:
- Keeping the 3D visualization responsive
- Updating a progress bar in real time
- Stopping the simulation if needed

### 3.2 User Workflow
1.  **Configure Geometry** (1. Geometria): Define cylinder, insulation, heaters, tubes, mesh
2.  **Set Materials** (2. Materiali): Select storage/insulation materials, operating conditions
3.  **Configure Analysis** (3. Analisi): Choose analysis type, set profiles, configure solver
4.  **Build Mesh**: Click "Costruisci Mesh" button
5.  **Run Simulation**: Click "Esegui Simulazione" button
6.  **View Results** (4. Risultati): Analyze statistics, energy balance, export data

---

## 4. Tab Organization (2-Level Structure)

### 4.1 GEOMETRIA Tab (Level 1)

#### Sub-tab: Cilindro
| Widget Group | Contents |
|--------------|----------|
| Dominio | Lx, Ly, Lz domain dimensions [m] |
| Cilindro Storage | Radius, height [m] |
| Tetto | Enable cone, angle, steel slab, fill with sand |

#### Sub-tab: Isolamento
| Widget Group | Contents |
|--------------|----------|
| Isolamento Radiale | Insulation thickness, shell thickness [m] |
| Isolamento Verticale | Bottom slab, top slab thickness [m] |

#### Sub-tab: Resistenze
| Widget Group | Contents |
|--------------|----------|
| Potenza | Total power [kW] |
| Pattern | Distribution pattern (Uniform, Grid, Radial, Spiral) |
| Elementi | Number, radius, spacing |
| Preview | Visual preview of positions |

#### Sub-tab: Tubi
| Widget Group | Contents |
|--------------|----------|
| Stato | Active/inactive toggle |
| Fluido | Temperature, convection coefficient |
| Pattern | Distribution pattern |
| Elementi | Number, diameter |

#### Sub-tab: Mesh
| Widget Group | Contents |
|--------------|----------|
| Spaziatura | Target cell spacing [m] |
| Info | Resulting cell count, memory estimate |

### 4.2 MATERIALI Tab (Level 1)

#### Sub-tab: Storage
| Widget Group | Contents |
|--------------|----------|
| Materiale | Material selection (Steatite, Sand, etc.) |
| Packing | Packing fraction [%] |
| Proprietà | Display of k, ρ, cp values |

#### Sub-tab: Isolamento
| Widget Group | Contents |
|--------------|----------|
| Materiale | Insulation material selection |
| Proprietà | Display of k, ρ, cp values |

#### Sub-tab: Condizioni
| Widget Group | Contents |
|--------------|----------|
| Ambiente | T_ambient [°C] |
| Convezione | External convection coefficient h_ext [W/(m²·K)] |

### 4.3 ANALISI Tab (Level 1)

#### Sub-tab: Tipo
| Widget Group | Contents |
|--------------|----------|
| Tipo Analisi | Steady-state, Losses analysis, Transient |
| Parametri | Duration, time step (for transient) |

#### Sub-tab: Condizioni Iniziali
| Widget Group | Contents |
|--------------|----------|
| Tipo | Uniform, Custom, From file |
| Temperatura | Initial temperature [°C] |

#### Sub-tab: Potenza
| Widget Group | Contents |
|--------------|----------|
| Tipo Profilo | Constant, Step, Ramp, Sinusoidal |
| Parametri | Base power, amplitude, frequency, timing |

#### Sub-tab: Estrazione
| Widget Group | Contents |
|--------------|----------|
| Tipo Profilo | Constant, Modulated, Temperature-controlled |
| Parametri | Flow rate, target temperature |

#### Sub-tab: Solver
| Widget Group | Contents |
|--------------|----------|
| Metodo | Direct, CG, BiCGStab, GMRES |
| Preconditioner | None, Jacobi, ILU, AMG |
| Tolleranza | Convergence tolerance (1e-4 to 1e-12) |
| Performance | CPU threads / GPU selection |
| Precisione | Float16/32/64 precision toggle |

#### Sub-tab: Salvataggio
| Widget Group | Contents |
|--------------|----------|
| Salva Stato | Save current simulation to HDF5 |
| Carica Stato | Load simulation from HDF5 |
| File Recenti | List of recent save files |

### 4.4 RISULTATI Tab (Level 1)

#### Sub-tab: Statistiche
| Widget Group | Contents |
|--------------|----------|
| Temperature | T_min, T_max, T_mean, T_std |
| Mesh | Dimensions, total nodes |
| Tempo | Computation time |

#### Sub-tab: Bilancio
| Widget Group | Contents |
|--------------|----------|
| Potenza | P_input (heaters), P_output (tubes) |
| Perdite | P_losses (top, lateral, bottom) |
| Energia | E stored [kWh, MWh] |
| Exergia | Exergy analysis (optional) |

#### Sub-tab: Materiali
| Widget Group | Contents |
|--------------|----------|
| Distribuzione | Volume fractions by material |
| Proprietà | Selected material properties |

#### Sub-tab: Esporta
| Widget Group | Contents |
|--------------|----------|
| Formati | CSV, VTK, HDF5 |
| Screenshot | Save current 3D view |

#### Sub-tab: 📖 Guida
| Widget Group | Contents |
|--------------|----------|
| Istruzioni | Usage guide |
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
- 2D plots of temporal evolution
- Result export in additional formats
- Editable material database directly from interface
- Real-time collaboration features
