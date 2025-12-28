# FDM Discretization - Finite Difference Method

## 1. Introduction to Finite Difference Method

The finite difference method (FDM) approximates derivatives with difference quotients,
transforming PDEs into systems of algebraic equations.

---

## 2. Derivative Approximation

### 2.1 First Derivative

**Forward difference:**
$$\frac{\partial T}{\partial x}\bigg|_i \approx \frac{T_{i+1} - T_i}{\Delta x} + O(\Delta x)$$

**Backward difference:**
$$\frac{\partial T}{\partial x}\bigg|_i \approx \frac{T_i - T_{i-1}}{\Delta x} + O(\Delta x)$$

**Centered difference:**
$$\frac{\partial T}{\partial x}\bigg|_i \approx \frac{T_{i+1} - T_{i-1}}{2\Delta x} + O(\Delta x^2)$$

### 2.2 Second Derivative

**Centered difference:**
$$\frac{\partial^2 T}{\partial x^2}\bigg|_i \approx \frac{T_{i+1} - 2T_i + T_{i-1}}{\Delta x^2} + O(\Delta x^2)$$

---

## 3. 3D Laplacian (Uniform Cartesian Mesh)

For a mesh with uniform spacing $\Delta = \Delta x = \Delta y = \Delta z$:

$$\nabla^2 T_{i,j,k} \approx \frac{1}{\Delta^2}[T_{i+1,j,k} + T_{i-1,j,k} + T_{i,j+1,k} + T_{i,j-1,k} + T_{i,j,k+1} + T_{i,j,k-1} - 6T_{i,j,k}]$$

### 3.1 Matrix Form

For each internal node, the steady-state equation $\nabla^2 T = -Q/k$ becomes:

$$\frac{1}{\Delta^2}[T_{E} + T_{W} + T_{N} + T_{S} + T_{U} + T_{D} - 6T_{P}] = -\frac{Q_P}{k}$$

Where:
- P = Principal node (i,j,k)
- E = East (i+1,j,k), W = West (i-1,j,k)
- N = North (i,j+1,k), S = South (i,j-1,k)  
- U = Up (i,j,k+1), D = Down (i,j,k-1)

Rearranging:

$$-6T_P + T_E + T_W + T_N + T_S + T_U + T_D = -\frac{Q_P \Delta^2}{k}$$

---

## 4. Materials with Variable Conductivity

When $k$ is not constant (interfaces between different materials), the **harmonic mean**
is used for the interface conductivity:

$$k_{i+1/2} = \frac{2 k_i k_{i+1}}{k_i + k_{i+1}}$$

### 4.1 Conservative Scheme

To ensure energy conservation:

$$\nabla \cdot (k \nabla T) \approx \frac{1}{\Delta^2}\left[k_{E}(T_E - T_P) + k_W(T_W - T_P) + ...\right]$$

Where $k_E = k_{i+1/2,j,k}$, etc.

The equation for node P becomes:

$$(k_E + k_W + k_N + k_S + k_U + k_D) T_P - k_E T_E - k_W T_W - k_N T_N - k_S T_S - k_U T_U - k_D T_D = Q_P \Delta^2$$

---

## 5. Boundary Conditions

### 5.1 Dirichlet (Fixed Temperature)

If the boundary node has $T_{boundary} = T_{prescribed}$:

**Option 1:** Insert directly into the system
- Corresponding row: $1 \cdot T_{boundary} = T_{prescribed}$
- Contribution moved to RHS vector of neighboring equations

**Option 2:** Ghost node
- A virtual node is created beyond the boundary
- The condition is imposed and the ghost node is eliminated

### 5.2 Neumann (Zero Flux - Adiabatic)

For an adiabatic boundary on $x = x_{max}$:

$$\frac{\partial T}{\partial x}\bigg|_{boundary} = 0$$

Approximation: $T_{i+1} = T_i$ (reflected ghost node)

The boundary node equation uses the reflected value.

### 5.3 Robin (Convection)

For convection at the boundary, the energy balance on the halved control volume at the boundary leads to an effective exchange coefficient that accounts for both conductive resistance (half cell) and convective resistance:

$$R_{tot} = R_{cond} + R_{conv} = \frac{\Delta/2}{k} + \frac{1}{h}$$

The heat flux is therefore:
$$q = \frac{T_{internal} - T_\infty}{R_{tot}} = \frac{T_{internal} - T_\infty}{\frac{\Delta}{2k} + \frac{1}{h}}$$

In coefficient form for the matrix:
$$a_{bc} = \frac{1}{R_{tot} \cdot \Delta} = \frac{2kh}{2k + h\Delta} \cdot \frac{1}{\Delta}$$

This approach provides greater accuracy than simply approximating the first derivative at the boundary.

---

## 6. Implicit Scheme for Transient (Future Implementation)

### 6.1 Backward Euler Method (Implicit)

$$\rho c_p \frac{T^{n+1} - T^n}{\Delta t} = \nabla \cdot (k \nabla T^{n+1}) + Q$$

Rearranging in matrix form:

$$(\mathbf{M} + \Delta t \mathbf{L}) \mathbf{T}^{n+1} = \mathbf{M} \mathbf{T}^n + \Delta t \mathbf{Q}$$

Where $\mathbf{M}$ is the mass matrix (thermal capacity) and $\mathbf{L}$ is the negative Laplacian.

> **Note**: Transient simulation is planned for future implementation. The current version supports steady-state only.

---

## 7. Matrix Structure

### 7.1 Linear System

The FDM system can be written as:

$$\mathbf{A} \cdot \mathbf{T} = \mathbf{b}$$

Where:
- $\mathbf{A}$ = coefficient matrix (sparse)
- $\mathbf{T}$ = unknown temperature vector
- $\mathbf{b}$ = known terms vector

### 7.2 Structure of Matrix A

For a 3D grid with $N_x \times N_y \times N_z$ nodes, the matrix is constructed in **Fortran order (column-major)**, where the $i$ index (X) is the fastest, followed by $j$ (Y) and finally $k$ (Z).

- Dimension: $N \times N$ where $N = N_x \cdot N_y \cdot N_z$
- Banded structure with 7 diagonals (3D)
- Sparse matrix (many zeros)

**Diagonal pattern:**
- Main diagonal: coefficient of node P
- Diagonals ±1: coefficients E/W (X direction)
- Diagonals ±$N_x$: coefficients N/S (Y direction)
- Diagonals ±$N_x N_y$: coefficients U/D (Z direction)

### 7.3 Example Matrix Row

For an internal node with linear index $p$:

```
A[p, p-Nx*Ny] = -k_d / Δ²     (Down contribution)
A[p, p-Nx]    = -k_s / Δ²     (South contribution)
A[p, p-1]     = -k_w / Δ²     (West contribution)
A[p, p]       = (k_e+k_w+k_n+k_s+k_u+k_d) / Δ²  (diagonal)
A[p, p+1]     = -k_e / Δ²     (East contribution)
A[p, p+Nx]    = -k_n / Δ²     (North contribution)
A[p, p+Nx*Ny] = -k_u / Δ²     (Up contribution)

b[p] = Q_p
```

Where $k_e, k_w, ...$ are the harmonic mean conductivities at the interfaces.

---

## 8. Index Linearization

### 8.1 From 3D to 1D (Fortran Order)

To convert from indices $(i, j, k)$ to linear index $p$:

$$p = i + j \cdot N_x + k \cdot N_x \cdot N_y$$

### 8.2 From 1D to 3D

For the inverse conversion:

$$k = p // (N_x \cdot N_y)$$
$$j = (p \% (N_x \cdot N_y)) // N_x$$
$$i = p \% N_x$$

---

## 9. Efficient Implementation

### 9.1 Sparse Matrices

Use `scipy.sparse` for sparse matrices:
- `csr_matrix`: Compressed Sparse Row - efficient for multiplication
- `csc_matrix`: Compressed Sparse Column - efficient for slicing
- `lil_matrix`: List of Lists - efficient for construction

### 9.2 Solvers

**Direct:**
- `scipy.sparse.linalg.spsolve`: Sparse LU factorization
- Robust but memory O(N^1.5) for 3D

**Iterative:**
- `scipy.sparse.linalg.cg`: Conjugate Gradient (symmetric positive definite matrix)
- `scipy.sparse.linalg.gmres`: General Minimal Residual
- `scipy.sparse.linalg.bicgstab`: BiCGStab

**Preconditioners:**
- ILU (Incomplete LU)
- Jacobi
- SSOR
- AMG (Algebraic Multigrid via PyAMG)

### 9.3 Vectorized Implementation

The matrix builder uses vectorized NumPy operations for 10-50x speedup over loop-based approaches:

```python
# Vectorized coefficient calculation
k_e = 2 * k[1:, :, :] * k[:-1, :, :] / (k[1:, :, :] + k[:-1, :, :])
# Apply to all nodes simultaneously
```

---

## 10. Validation

### 10.1 Analytical Solutions

To validate the code, compare with known solutions:

**1D steady-state case:**
$$T(x) = T_0 + \frac{q''}{k} x - \frac{Q}{2k} x^2$$

**3D case with uniform source:**
Sphere in infinite domain, analytical solution for comparison.

### 10.2 Energy Balance

Verify that:
$$P_{in} = P_{out} + \Delta E_{stored}$$

With tolerance < 1% for sufficiently fine mesh.

---

## 11. Mesh Convergence

### 11.1 Convergence Study

Run the simulation with progressively finer meshes:
- Mesh 1: N = 20
- Mesh 2: N = 40
- Mesh 3: N = 80

Verify that $||T_{N} - T_{N/2}||$ decreases with expected order (second order for centered scheme).

### 11.2 Stopping Criterion

Convergence achieved when:
$$\frac{||T_{N} - T_{N/2}||_\infty}{||T_{N}||_\infty} < \epsilon$$

With $\epsilon$ typically 0.01 (1%).
