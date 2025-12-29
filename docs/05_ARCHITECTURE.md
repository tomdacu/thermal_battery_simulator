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
│   ├── main_window.py       # PyQt6 GUI - 2-level tabs, ALL user inputs
│   ├── analysis_tab.py      # Analysis widgets (type, profiles, save/load)
│   └── transient_results_widget.py  # Transient visualization
│
├── src/core/                # Domain Model Layer
│   ├── mesh.py              # Mesh3D: 3D grid + field storage
│   ├── geometry.py          # BatteryGeometry: analytic geometry → mesh
│   ├── materials.py         # MaterialManager: thermal property database
│   └── profiles.py          # PowerProfile, ExtractionProfile, TransientConfig
│
├── src/solver/              # Numerical Engine Layer
│   ├── matrix_builder.py    # Assembles A matrix and b vector (Numba JIT)
│   ├── steady_state.py      # Solves A·T = b (CPU + GPU)
│   └── transient.py         # Backward Euler transient solver
│
├── src/analysis/            # Post-Processing Layer
│   ├── power_balance.py     # Computes power flows
│   └── energy_balance.py    # Energy and exergy balance analyzer
│
├── src/io/                  # Input/Output Layer
│   └── state_manager.py     # HDF5 state save/load manager
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
---

## 11. Performance Optimizations

This section documents the performance optimizations implemented in the solver to achieve
fast simulation times even on large meshes.

### 11.1 Overview of Optimizations

| Optimization | Location | Speedup | Description |
|--------------|----------|---------|-------------|
| `fastmath=True` | matrix_builder.py | 20-40% | Enables aggressive floating-point optimizations in Numba |
| AMG Ruge-Stuben | steady_state.py | 1.5-2x | Uses classic Ruge-Stuben instead of Smoothed Aggregation |
| AMG Cache | steady_state.py | 50-80% on repeat | Reuses multigrid hierarchy between solves |
| Warm Start | steady_state.py | 3-10x on repeat | Uses previous solution as initial guess |
| **Unified Numba Kernel** | matrix_builder.py | 1.2-1.5x | Single kernel for harmonic means + coefficients |
| **Vectorized Elements** | geometry.py | 5-20x | NumPy broadcasting for heaters/tubes |
| **Pre-allocated COO** | matrix_builder.py | 1.2-1.5x | Pre-allocated arrays instead of lists |

### 11.2 Numba JIT Compilation with `fastmath`

**File**: `src/solver/matrix_builder.py`

The `@njit(parallel=True, cache=True, fastmath=True)` decorator enables:

1. **Parallel execution**: Loops are automatically parallelized across CPU cores
2. **Caching**: Compiled code is cached to disk, eliminating recompilation overhead
3. **Fast math**: Allows reordering of floating-point operations for SIMD vectorization

```python
@njit(parallel=True, cache=True, fastmath=True)
def _compute_fdm_coefficients_unified_numba(...):
    """
    Unified kernel: computes harmonic means + FDM coefficients in one pass.
    fastmath=True allows aggressive optimizations for ~30% speedup.
    """
```

**Note**: `fastmath=True` can change results slightly due to reordering, but for
heat equation simulations, the differences are negligible compared to discretization errors.

### 11.3 AMG Ruge-Stuben Solver

**File**: `src/solver/steady_state.py`

For the heat equation (symmetric positive definite matrix), Ruge-Stuben (RS) AMG
is faster than Smoothed Aggregation (SA):

| Algorithm | Best For | Speed |
|-----------|----------|-------|
| Ruge-Stuben | Scalar PDEs (heat, diffusion) | Faster |
| Smoothed Aggregation | Vector PDEs (elasticity) | More robust |

```python
ml = pyamg.ruge_stuben_solver(
    A,
    max_coarse=500,       # Coarse grid size for direct solve
    max_levels=10,        # Hierarchy depth for meshes up to ~10M cells
    strength='symmetric'  # Exploits matrix symmetry
)
```

### 11.4 AMG Hierarchy Caching

**File**: `src/solver/steady_state.py`

The multigrid hierarchy construction is O(N) but has significant constant overhead.
For repeated solves with the same geometry, we cache the hierarchy:

```python
# In __init__:
self._amg_hierarchy = None   # Cache for AMG hierarchy
self._matrix_hash = None     # Hash to detect matrix changes

# In _get_preconditioner():
if self._amg_hierarchy is not None:
    # Reuse cached hierarchy (50-80% speedup on setup)
    ml = self._amg_hierarchy
else:
    ml = pyamg.ruge_stuben_solver(A, ...)
    self._amg_hierarchy = ml  # Save for next solve
```

The cache is automatically invalidated when the matrix changes (detected via hash).

### 11.5 Warm Start (Initial Guess)

**File**: `src/solver/steady_state.py`

For parametric studies (varying power, boundary conditions), the previous solution
is an excellent starting point:

```python
# In _solve_iterative():
if self._last_solution is not None:
    x0 = self._last_solution  # Warm start: ~80% fewer iterations
else:
    x0 = mesh.T.flatten()     # Cold start: use current temperature

# In solve():
self._last_solution = result.T.copy()  # Save for next solve
```

**Typical speedup**: 3-10x for repeated solves when only changing power or BC temperatures.

### 11.6 Unified Numba Kernel (Phase 2)

**File**: `src/solver/matrix_builder.py`

The original implementation used two separate Numba kernels:
1. `_compute_harmonic_means_numba()` - computes interface conductivities
2. `_compute_coefficients_numba()` - computes FDM stencil weights

**Problem**: This requires:
- Two function call overheads
- Intermediate arrays `k_w, k_e, ...` (6 × N × 8 bytes = 48N bytes memory)
- Data read twice (once per kernel)

**Solution**: Unified kernel `_compute_fdm_coefficients_unified_numba()`:
```python
@njit(parallel=True, cache=True, fastmath=True)
def _compute_fdm_coefficients_unified_numba(
        k_P, k_W, k_E, k_S, k_N, k_D, k_U,
        d2, eps,
        is_x_min, is_x_max, is_y_min, is_y_max, is_z_min, is_z_max):
    """
    Single kernel that:
    1. Computes harmonic mean k_face = 2*k1*k2/(k1+k2) inline
    2. Computes coefficient a_face = k_face/d² with boundary check
    
    Benefits:
    - No intermediate arrays (k_w, k_e, ... are scalars in the loop)
    - Single parallel loop (one prange instead of two)
    - Better cache locality (data stays in L1/L2)
    """
```

**Speedup**: 1.2-1.5x on matrix construction for meshes > 50k cells.

### 11.7 Vectorized Discrete Elements (Phase 2)

**File**: `src/core/geometry.py`

The original implementation used Python loops over heaters/tubes:
```python
# SLOW: O(N_elements * mesh_size) with Python overhead
for heater in heater_elements:
    dist_xy = np.sqrt((X - heater.x)**2 + (Y - heater.y)**2)
    mask = (dist_xy <= heater.radius) & ...
```

**Problem**: For 100 heaters on a 100³ mesh, this is 100M operations in slow Python.

**Solution**: 4D NumPy broadcasting:
```python
# FAST: Single vectorized operation using 4D broadcasting
heater_x = np.array([h.x for h in heater_elements])  # (n_heaters,)
dx_h = X[:,:,:,np.newaxis] - heater_x[np.newaxis,np.newaxis,np.newaxis,:]  # (Nx,Ny,Nz,n_heaters)
dist_xy_h = np.sqrt(dx_h**2 + dy_h**2)
mask_any_heater = np.any(in_radius & in_z_range, axis=-1)  # (Nx,Ny,Nz)
```

**Benefits**:
- Single vectorized computation for all elements
- Automatic SIMD optimization by NumPy
- Parallel execution on multi-core via NumPy's underlying BLAS

**Speedup**: 5-20x on geometry initialization with many discrete elements.

**Trade-off**: Requires O(mesh_size × n_elements) temporary memory. For very large
meshes with many elements, consider batching.

### 11.8 Pre-allocated COO Arrays (Phase 2)

**File**: `src/solver/matrix_builder.py`

The original implementation used Python lists for COO construction:
```python
# SLOW: Dynamic allocation and concatenation
row_list = [p_all]
col_list = [p_all]
data_list = [a_P]
# ... append 6 more times ...
rows = np.concatenate(row_list)  # Extra copy
```

**Problem**: 
- `list.append()` has O(1) amortized but with allocation spikes
- `np.concatenate()` allocates new array and copies all data

**Solution**: Pre-allocate arrays with known size:
```python
# FAST: Pre-allocation with known nnz
nnz = N + 2*(N - Nx*Ny) + 2*(N - Nx*Nz) + 2*(N - Ny*Nz)
rows = np.empty(nnz, dtype=np.int32)
cols = np.empty(nnz, dtype=np.int32)
data = np.empty(nnz, dtype=np.float64)

offset = 0
rows[offset:offset+N] = p_all
cols[offset:offset+N] = p_all
data[offset:offset+N] = a_P
offset += N
# ... slice assignment for neighbors ...
```

**Benefits**:
- Single allocation upfront (no realloc/copy)
- No concatenation overhead
- `int32` indices save 50% memory vs `int64` (sufficient for meshes up to 2B cells)

**Speedup**: 1.2-1.5x on matrix construction.

### 11.9 Performance Recommendations

For best performance, use these settings:

| Mesh Size | Recommended Config | Notes |
|-----------|-------------------|-------|
| < 50k cells | `method=direct` | Fast and exact |
| 50k - 500k | `method=cg, preconditioner=jacobi` | Good balance |
| 500k - 2M | `method=bicgstab, preconditioner=amg` | AMG shines here |
| > 2M cells | **GPU (CUDA)** with CuPy | 10-50x faster than CPU |

---

## 12. GPU Solver with CuPy

The solver supports GPU acceleration via CuPy for NVIDIA GPUs with CUDA.

### 12.1 Supported Backends

The solver automatically selects the best available backend:

| Backend | GPU Support | Installation | Speedup |
|---------|-------------|--------------|---------|
| **CUDA (CuPy)** | NVIDIA only | `pip install cupy-cuda11x` | 5-50x |
| **OpenCL** | AMD, Intel, NVIDIA | `pip install pyopencl` | 2-10x |
| **CPU** | Always available | Built-in | 1x (baseline) |

Priority: CUDA > OpenCL > CPU

### 12.2 Installation

```bash
# Per GPU NVIDIA (CUDA) - più veloce
pip install cupy-cuda11x  # oppure cupy-cuda12x

# Per GPU AMD/Intel/NVIDIA (OpenCL) - universale
pip install pyopencl

# Verifica driver OpenCL (Windows)
# I driver OpenCL sono solitamente inclusi nei driver GPU AMD/Intel/NVIDIA
```

### 12.3 Usage

In the GUI, select **"🎮 GPU (Auto)"** from the "Calcolo" dropdown in the Solver panel.

Or programmatically:
```python
from src.solver.steady_state import SolverConfig, SteadyStateSolver, is_gpu_available, get_gpu_info

# Check available backends
info = get_gpu_info()
print(f"Backend: {info['backend']}, Device: {info['name']}")

# Use GPU acceleration
if is_gpu_available():
    config = SolverConfig(
        method="cg",
        use_gpu=True,  # Auto-selects best backend (CUDA > OpenCL)
        tolerance=1e-8
    )
    solver = SteadyStateSolver(mesh, config)
    result = solver.solve()
```

### 12.4 How It Works

**CUDA Backend (CuPy):**
1. Transfers sparse CSR matrix and RHS to GPU memory
2. Uses cuSPARSE for matrix-vector operations
3. CuPy's CG/GMRES solvers run entirely on GPU

**OpenCL Backend:**
1. Creates OpenCL context (auto-detects best GPU or CPU)
2. Implements Conjugate Gradient with Jacobi preconditioner
3. Matrix-vector product on CPU (sparse), vector ops parallelized

```
CPU Memory                    GPU Memory
┌─────────────┐              ┌─────────────┐
│ A (CSR)     │ ──transfer→  │ A_gpu (CSR) │
│ b (dense)   │ ──transfer→  │ b_gpu       │
└─────────────┘              └─────────────┘
                                   │
                             ┌─────▼─────┐
                             │ CG Solver │
                             │ (cuSPARSE)│
                             └─────┬─────┘
                                   │
┌─────────────┐              ┌─────▼─────┐
│ T (solution)│ ←─transfer── │ T_gpu     │
└─────────────┘              └───────────┘
```

### 12.4 Supported Methods on GPU

| Method | GPU Support | Notes |
|--------|-------------|-------|
| `cg` | ✅ Full | Best for SPD matrices (heat equation) |
| `gmres` | ✅ Full | General matrices |
| `bicgstab` | ⚠️ Fallback to CG | Not in cupy.sparse.linalg |
| `direct` | ⚠️ Fallback to CPU | cuSolver has limited sparse support |

### 12.5 Preconditioners on GPU

| Preconditioner | GPU Support | Notes |
|----------------|-------------|-------|
| `jacobi` | ✅ Full | Very fast on GPU |
| `none` | ✅ Full | No preconditioning |
| `ilu` | ⚠️ Fallback to Jacobi | ILU not efficient on GPU |
| `amg` | ⚠️ Fallback to Jacobi | PyAMG is CPU-only |

### 12.6 Performance Guidelines

| Mesh Size | Recommended Backend | Expected Speedup |
|-----------|---------------------|------------------|
| < 50k cells | CPU | GPU overhead too high |
| 50k - 200k | GPU or CPU | 2-5x with GPU |
| 200k - 1M | GPU | 5-15x with GPU |
| > 1M cells | GPU | 20-50x with GPU |

### 12.7 Memory Requirements

GPU memory usage: approximately `12 * nnz + 24 * N` bytes, where:
- `nnz` = number of non-zeros in matrix (~7*N for 3D FDM)
- `N` = number of mesh cells

Example: 1M cell mesh uses ~100 MB GPU memory.

---

## 13. Transient Solver

The transient solver implements time-dependent heat transfer simulation using the Backward Euler (implicit) scheme.

### 13.1 Mathematical Formulation

The transient heat equation:
$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$$

Discretized with Backward Euler:
$$\frac{\rho c_p}{\Delta t} (T^{n+1} - T^n) = \nabla \cdot (k \nabla T^{n+1}) + Q^{n+1}$$

This is **unconditionally stable** (no CFL constraint on time step).

### 13.2 Implementation

**File**: `src/solver/transient.py`

```python
class TransientSolver:
    def step(self, Q_current: float, extraction_power: float = 0) -> SolverResult:
        """Advances one time step with given power input/output"""
        # 1. Update heat sources
        self._update_source_term(Q_current, extraction_power)
        
        # 2. Build transient matrix: A_trans = A_steady + M/dt
        A_trans = self._build_transient_matrix()
        
        # 3. Build transient RHS: b_trans = b_steady + M*T^n/dt
        b_trans = self._build_transient_rhs()
        
        # 4. Solve linear system
        T_new = solve(A_trans, b_trans)
        return result
    
    def run(self, power_profile, extraction_profile) -> TransientResults:
        """Runs full transient simulation with time-dependent profiles"""
        for t in time_steps:
            Q = power_profile.get_power(t)
            Q_ext = extraction_profile.get_power(t, T_mean)
            result = self.step(Q, Q_ext)
            save_result(t, result)
```

### 13.3 Profiles

**File**: `src/core/profiles.py`

| Profile Type | Description | Parameters |
|--------------|-------------|------------|
| **PowerProfile** | Time-dependent heater power | constant, step, ramp, sinusoidal |
| **ExtractionProfile** | Time-dependent extraction | constant, modulated, T-controlled |
| **InitialCondition** | Starting temperature | uniform, from file, custom |
| **TransientConfig** | Time discretization | t_end, dt, save_interval |

### 13.4 Key Features

- **AMG hierarchy reuse**: Multigrid setup is cached across time steps
- **Warm start**: Previous solution as initial guess
- **Adaptive time step** (planned): dt adjustment based on temperature change rate
- **Checkpoint saving**: Periodic save to HDF5 for restart capability

---

## 14. State Persistence (HDF5)

The simulation state can be saved to and loaded from HDF5 files.

### 14.1 State Manager

**File**: `src/io/state_manager.py`

```python
class StateManager:
    def save_state(self, filepath: Path, state: SimulationState) -> bool:
        """Saves complete simulation state to HDF5"""
        with h5py.File(filepath, 'w') as f:
            f.attrs['geometry_hash'] = state.geometry_hash
            f.create_dataset('T', data=state.T, compression='gzip')
            f.create_dataset('k', data=state.k)
            # ... all mesh fields
    
    def load_state(self, filepath: Path) -> SimulationState:
        """Loads simulation state from HDF5"""
        with h5py.File(filepath, 'r') as f:
            state = SimulationState(
                T=f['T'][:],
                geometry_hash=f.attrs['geometry_hash'],
                # ...
            )
        return state
```

### 14.2 Geometry Hash Verification

Each saved state includes a hash of the geometry configuration. When loading, the hash is verified to ensure the state is compatible with the current geometry.

### 14.3 Transient Results

For transient simulations, results include:
- Time series of temperatures (sampled at save_interval)
- Energy input/output over time
- T_mean, T_max, T_min evolution

---

## 15. Energy and Exergy Balance

**File**: `src/analysis/energy_balance.py`

### 15.1 Energy Balance

| Term | Description | Formula |
|------|-------------|---------|
| E_stored | Thermal energy in storage | $\int \rho c_p (T - T_{ref}) \, dV$ |
| P_input | Heater power | $\int Q \, dV$ |
| P_extraction | Tube extraction | $\int h(T - T_{fluid}) \, dA$ |
| P_losses | Heat losses | $\int h_{ext}(T - T_{amb}) \, dA$ |

### 15.2 Exergy Balance

Exergy (available work) considers the reference temperature:

$$Ex = \int \rho c_p \left[ (T - T_0) - T_0 \ln\frac{T}{T_0} \right] dV$$

where $T_0$ is the ambient (dead state) temperature.

---

## 16. Future Optimizations (Planned)

The following optimizations are planned for future releases:

1. **CSR Direct Construction**: Build sparse matrix directly in CSR format
2. **KDTree for Elements**: O(log N) lookup for heater/tube positions
3. **Visualization LOD**: Level-of-detail for interactive 3D rendering
4. **Multi-GPU Support**: Distribute large meshes across multiple GPUs
5. **Adaptive Time Stepping**: Variable dt for transient simulations