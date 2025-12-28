# Bug Report - Sand Battery Simulation
**Model:** Gemini 3 Flash (Preview)
**Date:** December 26, 2025

## 1. Critical Bugs

### 1.1 Internal Convection Ignored (Solver)
**File:** [src/solver/matrix_builder.py](src/solver/matrix_builder.py)
**Description:** The function `_get_boundary_coefficients` only applies convection terms (`h_bc`) if the node is on the domain boundaries (e.g., `i == 0` or `i == Nx - 1`). However, the geometry module marks nodes inside heat exchanger tubes as `BoundaryType.CONVECTION`. Since these nodes are internal to the domain, the solver fails to add the convection heat exchange term to the matrix and RHS.
**Impact:** Heat extraction via tubes is completely ignored in the simulation results.

### 1.2 Numba Normalization Inconsistency
**File:** [src/solver/matrix_builder.py](src/solver/matrix_builder.py)
**Description:** The Numba-optimized function `_compute_coefficients_numba` normalizes coefficients by dividing by `k_P`. The standard `build_steady_state_matrix` does not do this. If Numba is enabled, the RHS (source terms and BCs) must also be normalized by `k_P`, which is currently not handled.
**Impact:** Potential for incorrect results or solver divergence if Numba is used.

## 2. Logic & Consistency Issues

### 2.1 Tube Material Assignment
**File:** [src/core/geometry.py](src/core/geometry.py)
**Description:** In `apply_to_mesh`, nodes identified as being inside a tube are assigned `sand_props` (thermal properties of the storage material) instead of fluid properties (e.g., air or water).
**Impact:** The thermal inertia and conductivity inside the tubes are physically incorrect.

### 2.2 Power Balance Incomplete
**File:** [src/analysis/power_balance.py](src/analysis/power_balance.py)
**Description:** The `P_output` (power extracted by tubes) is hardcoded to `0.0` with a `TODO` comment. 
**Impact:** The energy balance analysis will always show a large imbalance when the battery is in discharge mode (tubes active).

### 2.3 Mesh Uniformity Assertion
**File:** [src/solver/matrix_builder.py](src/solver/matrix_builder.py)
**Description:** The solver contains `assert` statements requiring `dx == dy == dz`. While `Mesh3D` supports non-uniform meshes (`uniform=False`), the solver is not yet implemented to handle them.
**Impact:** The program crashes if a user attempts to use a non-uniform mesh.

## 3. GUI & UX Issues

### 3.1 Simulation Progress Feedback
**File:** [gui/main_window.py](gui/main_window.py)
**Description:** The `SimulationThread` only emits progress at 10%, 50%, and 90%. For large meshes, the "Risoluzione sistema..." step (50%) can take a long time without any intermediate feedback.
**Impact:** The UI may appear frozen during long solves.

### 3.2 Missing Validation for Geometry Overlap
**File:** [src/core/geometry.py](src/core/geometry.py)
**Description:** There is no check to ensure that heaters and tubes do not overlap in space.
**Impact:** If they overlap, the last one applied in the loop "wins," which might lead to unexpected physical configurations.

## 4. Recommendations
1.  **Fix Solver**: Update `_get_boundary_coefficients` to check `mesh.boundary_type[i, j, k]` and apply convection terms regardless of whether the node is on the domain boundary.
2.  **Implement Tube Power**: Update `PowerBalanceAnalyzer` to iterate over `CONVECTION` nodes and calculate $Q = \sum h \cdot A \cdot (T_{surf} - T_{inf})$.
3.  **Correct Tube Properties**: Use fluid properties for nodes marked as `MaterialID.TUBES`.
