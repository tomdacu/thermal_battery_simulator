# Bug Report - Sand Battery Simulation
**Model:** GPT-5.2
**Date:** December 26, 2025

This report re-validates and expands the findings from [report/flash_3.md](report/flash_3.md).

## 1. Critical / Functional Bugs

### 1.1 Internal Tube Convection Is Not Applied By The Solver
**File:** [src/solver/matrix_builder.py](src/solver/matrix_builder.py)
**File:** [src/core/geometry.py](src/core/geometry.py)

**What happens**
- The geometry marks cells inside tubes as `BoundaryType.CONVECTION` and sets `bc_h` / `bc_T_inf` for those cells.
- The matrix builder treats convection as a *domain-boundary-only* effect. `_get_boundary_coefficients()` only adds Robin terms when the node is on the outer boundary (e.g., `i==0`, `i==Nx-1`, etc.).

**Impact**
- Any “internal convection” (tubes inside the domain) contributes **zero** to `A` and `b`.
- In practice, discharge/heat extraction through tubes is **ignored** by the PDE solve.

**Why it’s a bug**
- The codebase uses `BoundaryType.CONVECTION` for both external boundaries and internal tube cells, but the solver only implements the external-boundary case.

---

### 1.2 Transient Mass Matrix Uses Wrong Flattening Order
**File:** [src/solver/matrix_builder.py](src/solver/matrix_builder.py)

**What happens**
- `build_transient_matrix()` builds `L` using the mesh linear indexing `p = i + j*Nx + k*Nx*Ny` (Fortran-like ordering).
- It builds the diagonal mass matrix using:
  - `rho_cp = (mesh.rho * mesh.cp).ravel()` (default C-order)

**Impact**
- The diagonal entries of the mass matrix do not match the same node ordering as `L`.
- Any transient solve built on `(A, B)` from this function will be physically wrong (node capacities will be permuted).

---

## 2. Numerical / Consistency Issues

### 2.1 Iterative Solver Initial Guess Uses Wrong Flattening Order
**File:** [src/solver/steady_state.py](src/solver/steady_state.py)

**What happens**
- The matrix is built in the mesh’s linear indexing (Fortran-like).
- The iterative solver uses `x0 = self.mesh.T.ravel()` (default C-order), while elsewhere the code explicitly documents Fortran ordering and even provides `mesh.flatten_field(field)`.

**Impact**
- The initial guess is a permutation of the intended initial state.
- Usually not fatal (iterative solvers may still converge), but it can:
  - slow convergence,
  - destabilize convergence for tougher problems,
  - make residual/debugging misleading.

Related: `compute_residual()` also uses `self.mesh.T.ravel()` which can report an incorrect residual for the *current* mesh field.

---

### 2.2 Numba Coefficient Routine Is Not Consistent With Main Formulation
**File:** [src/solver/matrix_builder.py](src/solver/matrix_builder.py)

**What happens**
- `_compute_coefficients_numba()` divides neighbor coefficients by `k_P` (normalization), which is not what the main assembly does.

**Impact**
- If this code path is ever used to assemble/solve without a corresponding RHS normalization, it can yield incorrect results.

---

## 3. Physics / Modeling Issues

### 3.1 Tube Cells Use Storage (Sand) Thermal Properties
**File:** [src/core/geometry.py](src/core/geometry.py)

**What happens**
- Cells inside tubes are assigned `sand_props` (via `_set_node_properties(..., MaterialID.TUBES, sand_props)`), not fluid properties.

**Impact**
- Tube regions have unrealistic thermal inertia and conductivity.
- Even after fixing internal convection, discharge behavior may still be physically off.

---

### 3.2 Power Balance: Tube Output Not Implemented
**File:** [src/analysis/power_balance.py](src/analysis/power_balance.py)

**What happens**
- `P_output` is hardcoded to `0.0` with a TODO.

**Impact**
- Power balance will be incomplete for discharge cases.

---

## 4. Documentation / Repo Consistency Bugs

### 4.1 README Links Point To Non-Existing Docs
**File:** [README.md](README.md)

**What happens**
- README references:
  - `docs/01_THEORY_AND_NUMERICS.md`
  - `docs/02_USER_AND_GUI_GUIDE.md`
  - `docs/03_DEVELOPER_GUIDE.md`
- But the repo contains:
  - `docs/01_THEORY.md`
  - `docs/02_FDM_DISCRETIZATION.md`
  - `docs/03_GEOMETRY.md`
  - `docs/04_GUI_DESIGN.md`

**Impact**
- Broken links / confusion for users.

---

## 5. Lower-Severity UX / Robustness Issues

### 5.1 GUI Progress Updates Are Too Sparse
**File:** [gui/main_window.py](gui/main_window.py)

**Impact**
- On large meshes, the UI can appear stuck during solve.

### 5.2 No Validation For Heater/Tube Overlap
**File:** [src/core/geometry.py](src/core/geometry.py)

**Impact**
- Overlapping elements resolve by “last assignment wins,” which can silently create inconsistent setups.

---

## 6. Recommended Fix Order (Highest Value First)
1. Implement internal convection terms in the solver for tube cells.
2. Fix all flattening/reshaping order mismatches (steady-state iterative x0, transient mass matrix, residual checks) to consistently use Fortran order.
3. Decide a clear model for “tube cells” (fluid region vs solid wall + Robin BC) and make properties consistent.
4. Implement tube output power in `PowerBalanceAnalyzer` so the reported balance matches the physics.
5. Update README doc links to match the actual files under `docs/`.
