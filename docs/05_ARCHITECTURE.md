# System Architecture - Thermal Battery Simulation

## 1. Overview

This document describes the software architecture of the Thermal Battery simulation system. The project simulates 3D thermal behavior of a sand-based thermal energy storage system using the Finite Difference Method (FDM).

---

## 2. Design Philosophy: GUI-Driven Configuration

**KEY PRINCIPLE**: All simulation parameters are defined through the GUI. The code contains **no hardcoded simulation values** - everything flows from user input widgets.

### 2.1 Configuration Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        GUI (main_window.py)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Geometry    │  │  Materials   │  │   Solver     │           │
│  │  - Lx,Ly,Lz  │  │  - storage   │  │  - method    │           │
│  │  - radius    │  │  - insulation│  │  - tolerance │           │
│  │  - height    │  │  - packing % │  │  - max_iter  │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │         _build_battery_geometry_from_inputs()               │ │
│  │    Reads ALL widget values and creates configuration        │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     BatteryGeometry           │
              │   (dataclass configuration)   │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     apply_to_mesh()           │
              │   Maps geometry → mesh fields │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     Mesh3D                    │
              │   3D arrays: T, k, ρ, cp, Q   │
              └───────────────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │     SteadyStateSolver         │
              │   Solves the heat equation    │
              └───────────────────────────────┘
```

### 2.2 Widget → Parameter Mapping

| GUI Widget | Stored In | Used By |
|------------|-----------|---------|
| `lx_spin`, `ly_spin`, `lz_spin` | `Mesh3D(Lx, Ly, Lz)` | Domain size |
| `spacing_spin` | `Mesh3D(target_spacing)` | Mesh resolution |
| `radius_spin`, `height_spin` | `CylinderGeometry` | Battery dimensions |
| `storage_combo` | `BatteryGeometry.storage_material` | Thermal properties |
| `power_spin` | `HeaterConfig.power_total` | Heat source Q |
| `tubes_active_check` | `TubeConfig.active` | Enable/disable tubes |
| `solver_combo` | `SolverConfig.method` | Direct/iterative solver |

---

## 3. Module Structure

```
battery_simulation/
├── gui/                     # User Interface Layer
│   └── main_window.py       # PyQt6 GUI - ALL user inputs here
│
├── src/core/                # Domain Model Layer
│   ├── mesh.py              # Mesh3D: 3D grid + field storage
│   ├── geometry.py          # BatteryGeometry: analytic geometry → mesh
│   └── materials.py         # MaterialManager: thermal property database
│
├── src/solver/              # Numerical Engine Layer
│   ├── matrix_builder.py    # Assembles A matrix and b vector
│   └── steady_state.py      # Solves A·T = b
│
├── src/analysis/            # Post-Processing Layer
│   └── power_balance.py     # Computes energy/power balance
│
├── src/visualization/       # Rendering Layer
│   └── renderer.py          # PyVista 3D visualization (standalone)
│
├── main.py                  # CLI entry point (for testing/scripting)
└── run_gui.py               # GUI entry point
```

---

## 4. Data Flow During Simulation

### 4.1 Mesh Build Phase (No Simulation)

```python
# 1. User clicks "Costruisci Mesh" button
def build_mesh(self):
    # 2. Extract ALL parameters from GUI widgets
    d, Lx, Ly, Lz, geom = self._build_battery_geometry_from_inputs()
    
    # 3. Create empty uniform mesh
    self.mesh = Mesh3D(Lx=Lx, Ly=Ly, Lz=Lz, target_spacing=d)
    
    # 4. Apply geometry to mesh (sets k, ρ, cp, Q, boundary_type, etc.)
    geom.apply_to_mesh(self.mesh, self.mat_manager)
    
    # 5. Visualize materials (simulation NOT run yet)
    self.update_visualization()
```

### 4.2 Simulation Phase

```python
# 1. User clicks "Esegui Simulazione" button
def run_simulation(self):
    # 2. Solver config from GUI
    config = SolverConfig(method=self.solver_combo.currentText(), ...)
    
    # 3. Launch in separate thread (keeps GUI responsive)
    self.simulation_thread = SimulationThread(self.mesh, config)
    self.simulation_thread.start()

# 4. Inside SimulationThread.run():
#    - build_steady_state_matrix(mesh) → A, b
#    - spsolve(A, b) → T
#    - mesh.T = unflatten(T)
```

---

## 5. Key Classes

### 5.1 Mesh3D (src/core/mesh.py)

Holds the computational grid and all field data:

```python
@dataclass
class Mesh3D:
    Lx, Ly, Lz: float           # Domain dimensions [m]
    Nx, Ny, Nz: int             # Number of cells per axis
    dx, dy, dz: float           # Cell spacing [m] (uniform: dx=dy=dz=d)
    
    # 3D field arrays (shape: Nx × Ny × Nz)
    T: np.ndarray               # Temperature [°C]
    k: np.ndarray               # Thermal conductivity [W/(m·K)]
    rho: np.ndarray             # Density [kg/m³]
    cp: np.ndarray              # Specific heat [J/(kg·K)]
    Q: np.ndarray               # Heat source [W/m³]
    material_id: np.ndarray     # Material identifier (enum)
    boundary_type: np.ndarray   # Boundary condition type (enum)
    bc_h: np.ndarray            # Convection coefficient [W/(m²·K)]
    bc_T_inf: np.ndarray        # External temperature [°C]
```

### 5.2 BatteryGeometry (src/core/geometry.py)

Translates analytic geometry into mesh fields:

```python
@dataclass
class BatteryGeometry:
    cylinder: CylinderGeometry  # 4 radial zones (STORAGE, INSULATION, STEEL, AIR)
    heaters: HeaterConfig       # Heating elements configuration
    tubes: TubeConfig           # Heat exchange tubes configuration
    storage_material: str       # Material name (from GUI combo)
    insulation_material: str    # Material name (from GUI combo)
    packing_fraction: float     # From GUI spin (0.63 = 63%)
    
    def apply_to_mesh(self, mesh: Mesh3D, mat_manager: MaterialManager):
        """Iterate all cells, determine zone, set properties"""
        for i, j, k in mesh_cells:
            r = distance_from_center(i, j)
            z = mesh.z[k]
            
            # Determine radial zone (4-zone model)
            if r > r_storage + insulation_thickness + shell_thickness:
                set_air_properties(mesh, i, j, k)
            elif r > r_storage + insulation_thickness:
                set_steel_properties(mesh, i, j, k)
            elif r > r_storage:
                set_insulation_properties(mesh, i, j, k)
            else:
                # Inside storage: check for discrete elements
                if is_inside_tube(x, y):
                    set_tube_properties(mesh, i, j, k)
                elif is_inside_heater(x, y):
                    set_heater_properties(mesh, i, j, k)
                else:
                    set_storage_properties(mesh, i, j, k)
```

### 5.3 SteadyStateSolver (src/solver/steady_state.py)

Solves the discretized heat equation:

```python
class SteadyStateSolver:
    def solve(self):
        # 1. Build linear system
        A, b = build_steady_state_matrix(self.mesh)
        
        # 2. Solve (direct or iterative based on config)
        if self.config.method == "direct":
            T = spsolve(A, b)
        else:
            T, info = bicgstab(A, b, M=preconditioner)
        
        # 3. Store result back in mesh
        self.mesh.T = unflatten(T)
```

---

## 6. Indexing Convention

The entire system uses **Fortran-order (column-major)** indexing for flattening 3D arrays to 1D vectors:

```python
# Linear index from 3D indices
p = i + j * Nx + k * Nx * Ny

# 3D indices from linear index
k = p // (Nx * Ny)
j = (p % (Nx * Ny)) // Nx
i = p % Nx

# When flattening arrays for solver
T_flat = mesh.T.ravel(order='F')  # Fortran order
T_3d = T_flat.reshape((Nx, Ny, Nz), order='F')
```

---

## 7. Boundary Conditions

### 7.1 Types Supported

| Type | Enum | Equation | Use Case |
|------|------|----------|----------|
| Internal | `INTERNAL` | Standard 7-point stencil | Interior cells |
| Dirichlet | `DIRICHLET` | T = T_prescribed | Ground (z=0) |
| Convection | `CONVECTION` | -k∂T/∂n = h(T - T∞) | External surfaces, tubes |

### 7.2 Internal Convection (Tube Cells)

When tubes are active, cells inside tube regions become internal convection sinks:
- Matrix: adds `h/d` to diagonal
- RHS: adds `h/d * T_fluid`

This models heat extraction without requiring actual fluid flow simulation.

---

## 8. Files That Should NOT Contain Hardcoded Values

| File | What it should contain | What it should NOT contain |
|------|------------------------|---------------------------|
| `main_window.py` | Widget defaults | Simulation logic |
| `geometry.py` | Geometry algorithms | Specific dimensions |
| `matrix_builder.py` | FDM discretization | Physical parameters |
| `power_balance.py` | Balance calculations | Fixed temperatures |

The CLI script `main.py` contains hardcoded values for **testing/scripting** purposes, but the GUI should always be the primary user interface.

---

## 9. Extension Points

### 9.1 Adding a New Material
1. Add entry to `STORAGE_MATERIALS` or similar dict in `materials.py`
2. Add to GUI combo box in `_create_geometry_tab()`

### 9.2 Adding a New Heater Pattern
1. Add pattern constant to `HeaterPattern` class
2. Implement `_generate_*_positions()` method in `HeaterConfig`
3. Add to GUI combo box in `_create_heater_tab()`

### 9.3 Adding a New Solver Method
1. Add case in `SteadyStateSolver._solve_iterative()`
2. Add to GUI combo box in `_create_solver_tab()`

---

## 10. Thread Safety

The simulation runs in a background thread (`SimulationThread`) to keep the GUI responsive:

```python
class SimulationThread(QThread):
    progress = pyqtSignal(int, str)  # Progress updates
    finished = pyqtSignal(object)     # Result when done
    error = pyqtSignal(str)           # Error message
    
    def run(self):
        # Mesh is passed by reference - solver modifies mesh.T in place
        solver = SteadyStateSolver(self.mesh, self.config)
        result = solver.solve()
        self.finished.emit(result)
```

**Important**: Do not modify mesh fields from the main thread while simulation is running.
