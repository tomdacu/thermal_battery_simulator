# Code Structure - Detailed Module Documentation

## 1. Introduction

This document provides a deep dive into the code structure of the Thermal Battery simulation project. It explains what each module does, its key functions, and how they interact.

**IMPORTANT**: This project follows a **GUI-driven design**. All simulation parameters originate from GUI widgets - there are no hardcoded values in the core modules.

---

## 2. Entry Points

### 2.1 `run_gui.py` - Production Entry Point

**Purpose**: Launches the PyQt6 GUI application.

```python
# Usage
python run_gui.py
```

This is the **recommended** way to run the application. All parameters are configured through the graphical interface.

---

## 3. Core Modules (`src/core/`)

### 3.1 `mesh.py` - 3D Mesh Data Structure

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `Mesh3D` | Main mesh data class holding all 3D arrays |
| `MaterialID` | Enum for material identification |
| `BoundaryType` | Enum for boundary condition types |
| `NodeProperties` | Properties for a single node |

**Mesh3D Arrays** (all in Fortran order):

| Array | Shape | Type | Description |
|-------|-------|------|-------------|
| `T` | (Nx, Ny, Nz) | float64 | Temperature field [°C] |
| `k` | (Nx, Ny, Nz) | float64 | Thermal conductivity [W/(m·K)] |
| `rho` | (Nx, Ny, Nz) | float64 | Density [kg/m³] |
| `cp` | (Nx, Ny, Nz) | float64 | Specific heat [J/(kg·K)] |
| `Q` | (Nx, Ny, Nz) | float64 | Volumetric heat source [W/m³] |
| `material_id` | (Nx, Ny, Nz) | int32 | Material identifier |
| `boundary_type` | (Nx, Ny, Nz) | int32 | Boundary condition type |
| `bc_h` | (Nx, Ny, Nz) | float64 | Convection coefficient [W/(m²·K)] |
| `bc_T_inf` | (Nx, Ny, Nz) | float64 | Reference temperature [°C] |

**Key Methods**:

```python
# Index conversion (Fortran order)
p = mesh.ijk_to_linear(i, j, k)  # p = i + j*Nx + k*Nx*Ny
i, j, k = mesh.linear_to_ijk(p)

# Boundary condition setters
mesh.set_fixed_temperature_bc('z_min', T=20.0)
mesh.set_convection_bc('z_max', h=10.0, T_inf=20.0)
mesh.set_symmetry_bc('x_min')
```

### 3.2 `geometry.py` - Battery Geometry Definition

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `CylinderGeometry` | 4-zone radial structure (r_storage, insulation_thickness, shell_thickness) |
| `HeaterConfig` | Heater power, count, pattern configuration |
| `TubeConfig` | Tube fluid properties and pattern configuration |
| `HeaterElement` | Single heater element properties |
| `TubeElement` | Single tube element properties |
| `BatteryGeometry` | Combines all above + materials |
| `HeaterPattern` | Enum-like class for heater patterns |
| `TubePattern` | Enum-like class for tube patterns |

**Central Method - `BatteryGeometry.apply_to_mesh()`**:

This is the key function that maps the analytic geometry to mesh fields:

```python
def apply_to_mesh(self, mesh: Mesh3D, mat_manager: MaterialManager):
    """
    Applies the battery geometry to a mesh.
    
    Steps:
    1. Assign material_id based on radial position
    2. Set thermal properties (k, rho, cp) for each material
    3. Generate heater/tube element positions
    4. Apply heat sources Q for heaters
    5. Apply internal convection for tubes (if active)
    6. Set boundary conditions on all faces
    """
```

### 3.3 `materials.py` - Material Database

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `MaterialManager` | Loads and provides material properties |
| `MaterialType` | Enum for material categories |
| `ThermalProperties` | Dataclass for k, rho, cp values |

**Material Database Location**: `materials_database.py` (project root)

**Usage**:

```python
mat_manager = MaterialManager()

# Get material by name
steatite = mat_manager.get_material("steatite")
print(steatite.k)  # 3.5 W/(m·K)

# List available materials
mat_manager.list_materials()
```

---

## 4. Solver Modules (`src/solver/`)

### 4.1 `matrix_builder.py` - FDM Matrix Assembly

**Main Function**: `build_steady_state_matrix(mesh)`

Constructs the sparse linear system A·T = b for the steady-state heat equation.

**Algorithm**:

```
For each node (i, j, k):
    If DIRICHLET:
        a_P = 1, b_P = T_fixed
    
    If INTERNAL:
        Use 7-point stencil with harmonic mean for k at faces
        a_P = Σ(k_face/d²) for all 6 neighbors
        a_neighbor = -k_face/d²
        b_P = Q_P * d³  (volumetric source)
    
    If CONVECTION (boundary):
        Add Robin term: a_P += h/d, b_P += h*T_inf/d
```

**Helper Functions**:

| Function | Purpose |
|----------|---------|
| `_get_internal_coefficients_v2()` | Coefficients for internal nodes |
| `_get_boundary_coefficients()` | Coefficients for boundary nodes |
| `_harmonic_mean_k()` | Computes k_face from k_P and k_neighbor |

### 4.2 `steady_state.py` - Linear System Solver

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `SolverConfig` | Solver method, tolerance, preconditioner |
| `SolverResult` | Solution T, convergence info, timing |
| `SteadyStateSolver` | Main solver class |

**Solver Methods**:

| Method | Description | Best For |
|--------|-------------|----------|
| `direct` | LU factorization (spsolve) | Small/medium systems |
| `cg` | Conjugate Gradient | SPD matrices |
| `gmres` | GMRES | General systems |
| `bicgstab` | BiCGSTAB | General systems |

**Usage**:

```python
config = SolverConfig(method="direct", verbose=True)
solver = SteadyStateSolver(mesh, config)
result = solver.solve()

if result.converged:
    T = result.T  # 1D array in Fortran order
    T_3d = T.reshape((mesh.Nx, mesh.Ny, mesh.Nz), order='F')
```

---

## 5. Analysis Module (`src/analysis/`)

### 5.1 `power_balance.py` - Energy Balance Calculations

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `PowerBalance` | Dataclass for power balance results |
| `PowerBalanceAnalyzer` | Computes power flows and energy storage |

**Key Methods**:

```python
analyzer = PowerBalanceAnalyzer(mesh)

# Power balance
balance = analyzer.compute_power_balance()
print(f"P_input: {balance.P_input} W")
print(f"P_loss: {balance.P_loss_total} W")

# Stored energy
energy = analyzer.compute_stored_energy(T_ref=20.0)
print(f"E = {energy['E_kWh']} kWh")

# Radial profile
r, T = analyzer.compute_radial_temperature_profile(x_c, y_c, z)
```

---

## 6. Visualization Module (`src/visualization/`)

### 6.1 `renderer.py` - 3D Visualization

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `VisualizationConfig` | Colormap, range, styling |
| `BatteryRenderer` | Standalone PyVista rendering |

**Note**: The GUI uses its own embedded PyVistaQt plotter (`QtInteractor`), not this standalone renderer. This module is primarily for CLI/scripting use.

---

## 7. GUI Module (`gui/`)

### 7.1 `main_window.py` - Main Application Window

**This is the CENTRAL module** - all simulation parameters originate here.

**Key Classes**:

| Class | Purpose |
|-------|---------|
| `ThermalBatteryGUI` | Main window with all widgets |
| `SimulationThread` | Background thread for solver |

**Critical Method - `_build_battery_geometry_from_inputs()`**:

This function reads ALL GUI widgets and creates configuration objects:

```python
def _build_battery_geometry_from_inputs(self):
    """
    Reads every GUI widget and creates:
    - CylinderGeometry
    - HeaterConfig
    - TubeConfig
    - BatteryGeometry
    
    Returns: (spacing, Lx, Ly, Lz, BatteryGeometry)
    """
    # Step 1: Read domain parameters
    d = self.spacing_spin.value()
    Lx = self.lx_spin.value()
    ...
    
    # Step 2: Create geometry configs
    cylinder = CylinderGeometry(...)
    heater_config = HeaterConfig(...)
    tube_config = TubeConfig(...)
    
    # Step 3: Combine into BatteryGeometry
    geom = BatteryGeometry(
        cylinder=cylinder,
        heaters=heater_config,
        tubes=tube_config,
        ...
    )
    
    return d, Lx, Ly, Lz, geom
```

**Widget Organization**:

| Tab | Contains |
|-----|----------|
| Geometria | Domain size, mesh spacing, materials |
| Resistenze | Heater power, pattern, count |
| Tubi | Tube h_fluid, T_fluid, pattern |
| Solver | Solution method, tolerance |

**Action Flow**:

```
[Costruisci Mesh] → build_mesh()
    → _build_battery_geometry_from_inputs()
    → Mesh3D()
    → BatteryGeometry.apply_to_mesh()
    → update_visualization()

[Anteprima Geometria] → preview_geometry()
    → _build_battery_geometry_from_inputs()
    → Render cylinders/elements (no mesh)

[Esegui Simulazione] → run_simulation()
    → SteadyStateSolver.solve()
    → update_visualization()
    → update_energy_balance()
```

---

## 8. Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              GUI WIDGETS                                 │
│  lx_spin, power_spin, h_fluid_spin, solver_combo, etc.                  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                _build_battery_geometry_from_inputs()                     │
│  Reads ALL widgets, creates CylinderGeometry, HeaterConfig, TubeConfig  │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        BatteryGeometry                                   │
│  Combines geometry + materials + heater/tube configs                    │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    BatteryGeometry.apply_to_mesh()                       │
│  Maps analytic geometry → mesh arrays (k, rho, cp, Q, boundaries)       │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                             Mesh3D                                       │
│  3D arrays: T, k, rho, cp, Q, material_id, boundary_type                │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    build_steady_state_matrix()                           │
│  Assembles sparse A matrix and b vector from mesh fields                │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      SteadyStateSolver.solve()                           │
│  Solves A·T = b using direct or iterative method                        │
└──────────────────────────────────┬──────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    Temperature Field (mesh.T)                            │
│  3D solution used for visualization and analysis                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Indexing Convention

The entire codebase uses **Fortran-order** (column-major) indexing:

```python
# 3D index (i, j, k) → Linear index p
p = i + j * Nx + k * Nx * Ny

# Linear index p → 3D index (i, j, k)
i = p % Nx
j = (p // Nx) % Ny
k = p // (Nx * Ny)
```

**Why Fortran order?**
- Matches NumPy's default for `order='F'`
- Efficient for z-first iteration (common in FDM)
- Consistent across all modules

---

## 10. New Modules (v2.0)

### 10.1 Transient Solver (`src/solver/transient.py`)

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `TransientSolverConfig` | Time stepping and solver configuration |
| `TransientSolver` | Backward Euler time-stepping solver |

**Key Methods:**
```python
class TransientSolver:
    def step(self, Q_current, extraction_power) -> SolverResult:
        """Advances one time step"""
    
    def run(self, power_profile, extraction_profile) -> TransientResults:
        """Runs full transient simulation"""
```

### 10.2 Profiles (`src/core/profiles.py`)

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `PowerProfile` | Time-dependent heater power (constant, step, ramp, sinusoidal) |
| `ExtractionProfile` | Time-dependent extraction (constant, modulated, T-controlled) |
| `InitialCondition` | Starting temperature distribution |
| `TransientConfig` | Time discretization parameters (t_end, dt) |

### 10.3 State Manager (`src/io/state_manager.py`)

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `SimulationState` | Complete simulation state (T, mesh, config) |
| `StateManager` | HDF5 save/load with geometry hash verification |
| `TransientResults` | Time series of transient simulation results |

### 10.4 Energy Balance (`src/analysis/energy_balance.py`)

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `EnergyBalanceResult` | Dataclass with energy/exergy terms |
| `EnergyBalanceAnalyzer` | Computes stored energy, losses, exergy |

### 10.5 Analysis Tab Widgets (`gui/analysis_tab.py`)

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `AnalysisTypeWidget` | Radio buttons for Steady/Losses/Transient |
| `InitialConditionWidget` | Initial temperature configuration |
| `PowerProfileWidget` | Heater power profile editor with preview |
| `ExtractionProfileWidget` | Extraction profile editor |
| `SaveLoadWidget` | HDF5 save/load interface |

---

## 11. Extension Points

To add new features:

| Feature | Files to Modify |
|---------|----------------|
| New material | `materials_database.py` |
| New heater pattern | `geometry.py` (HeaterConfig) |
| New tube pattern | `geometry.py` (TubeConfig) |
| New solver method | `steady_state.py` |
| New boundary type | `mesh.py`, `matrix_builder.py` |
| New analysis metric | `power_balance.py`, `energy_balance.py` |
| New GUI widget | `main_window.py` |
| New power profile | `profiles.py` (PowerProfile) |
| New state format | `state_manager.py` |

---

## 12. Testing

Tests are in the `tests/` directory:

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src
```

Key test files:
- `test_core.py` - Tests for mesh, geometry, materials
- `test_solver.py` - Tests for matrix builder and solver

---

## 13. Configuration Files

| File | Purpose |
|------|---------|
| `config/default_config.yaml` | Default GUI values (not yet used) |
| `requirements.txt` | Python dependencies |

