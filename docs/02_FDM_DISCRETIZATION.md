# 2. FDM discretization

Implemented in `src/solver/matrix.py` (assembly), `src/core/physics.py` (face
coefficients) and `src/solver/steady.py` / `transient.py` (the drivers).  The
scheme is pinned by `tests/test_solver.py`, including a comparison against an
independently written reference assembly.

## 1. Derivatives

Centred differences, second order:

$$T'_i \approx \frac{T_{i+1}-T_{i-1}}{2\Delta} + O(\Delta^2), \qquad
T''_i \approx \frac{T_{i+1}-2T_i+T_{i-1}}{\Delta^2} + O(\Delta^2)$$

On the uniform Cartesian grid ($\Delta = \Delta x = \Delta y = \Delta z$, see
`Mesh3D.spacing`):

$$\nabla^2 T_P \approx \frac{1}{\Delta^2}
\left[T_E+T_W+T_N+T_S+T_U+T_D-6T_P\right]$$

The production grid is usually **graded** (`Mesh3D.grid`, a `GridSpec`): `dx/dy/dz`
are per-axis arrays and the face distances differ from cell to cell.  The scheme is
then written in *finite-volume* form (§2), which reduces to the stencil above when
every cell is the same size; `tests/test_graded_mesh.py` checks that a uniform
`GridSpec` reproduces the uniform mesh bit for bit.

## 2. Cell balance and coefficient units

For cell $P$ the faces are the six cell sides; the balance
$\sum_{faces} k_{face}(T_{nb}-T_P)A/d_{centers} + QV = 0$ divided by $V$ is what the
code assembles:

$$a_P T_P - \sum_{nb} a_{nb} T_{nb} = Q, \qquad
a_{nb} = \frac{k_{face}}{d_{centers}}\frac{A}{V}
\ \left[\frac{\mathrm{W}}{\mathrm{m^3 K}}\right]$$

On a uniform grid $A/V = \Delta^2/\Delta^3 = 1/\Delta$ and $d_{centers} = \Delta$, so
this is the familiar $k_{face}/\Delta^2$; on a graded grid the factor
`A/(d_centers·V)` is exactly the `GridIndex.face_factors` the solver caches per mesh
(built once, lazily, on first use).

**All coefficients are per unit volume, the source is volumetric [W/m³].**  This
single convention is what makes the steady and the transient operators
compatible; §6 shows what breaks without it.

## 3. Variable conductivity

At an interface the two half cells are in series, so the face conductivity is
their harmonic mean (`src/core/physics.py::harmonic_mean`):

$$k_{face} = \frac{2 k_1 k_2}{k_1 + k_2 + \epsilon}$$

Because the medium is voxelised, the interface always lies on a face between two
cells and the two half cells have the same thickness, which is exactly the
assumption the harmonic mean encodes.  On a graded grid the two half cells have
*different* thicknesses, so `face_coefficients` builds the series resistance from the
local distances:

$$k_{eff} = \frac{0.5\,(d_P + d_{nb})}
{\dfrac{0.5\,d_P}{k_P} + \dfrac{1}{h_{contact}} + \dfrac{0.5\,d_{nb}}{k_{nb}}}
 \qquad\text{when } mesh.h\_contact > 0$$

`h_contact = 0` reproduces the harmonic mean, so a perfect interface is the default
and a real contact resistance is one number away (`mesh.h_contact`, applied only between
two *different* materials).  `tests/test_environment.py` checks that the interface step
grows with the resistance and that a single-material interface is untouched.

## 4. Assembled row

For an interior node $P$ with linear index $p$ (`GridIndex.flat`, Fortran order) and
the cached factor $f_{nb} = A/(d_{centers}V)$ of the face towards $nb$:

```
A[p, p-Nx*Ny] = -k_D*f_D      A[p, p]       =  k_E*f_E + k_W*f_W + k_N*f_N
A[p, p-Nx]    = -k_S*f_S                          + k_S*f_S + k_U*f_U + k_D*f_D
A[p, p-1]     = -k_W*f_W      A[p, p+1]     = -k_E*f_E
A[p, p+Nx]    = -k_N*f_N      A[p, p+Nx*Ny] = -k_U*f_U
b[p] = Q_source[p] + Q_sink[p]
```

(Faces on the box boundary get no off-diagonal entry - their coefficient is zeroed in
`face_coefficients` - so a node on an edge sees only the neighbours it really has.)

## 5. Boundary conditions

### 5.1 Dirichlet - symmetric elimination

A fixed-temperature face pins its nodes: `dirichlet_rows()` returns the mask and
the values, and `apply_dirichlet()` (a) moves the known contribution of each
pinned node into the right-hand side of the neighbouring equations, (b) zeroes
that **column**, and (c) replaces the pinned row with the identity.

Zeroing the column is what keeps the matrix **symmetric**.  Replacing only the
row (the previous behaviour) left it asymmetric, which made BiCGSTAB break down
(`info=-10`) and made CG inapplicable in principle.  `tests/test_solver.py`
asserts `is_symmetric(matrix)` and that CG converges on it.

The pinned node holds the *face* temperature: the surface is treated as if it
passed through the first cell centre, a first-order boundary artefact of
$O(q''\,dx_0/2k)$ that is visible in the first cell next to the face ($dx_0$ is the
local size normal to that face, so a graded mesh has a smaller artefact where the
grid is fine).

### 5.2 Neumann - imposed flux

A flux $q''$ [W/m²] entering the domain adds $q''/dx_0$ to the right-hand side
of the face nodes (per unit volume: $q''A/V = q''/dx_0$, with $dx_0$ the local cell
size normal to the face).  $\varepsilon = 0$ is the adiabatic case, which is also the
default of the `FaceBC` dataclass (`BoundaryType.INTERNAL`).

### 5.3 Robin - convection and radiation

The surface sits $dx_0/2$ away from the node centre, so the film resistance is
in series with the half cell:

$$R_{tot} = \frac{dx_0}{2k} + \frac{1}{h} \quad\Longrightarrow\quad
h_{eff} = \frac{2kh}{2k + h\,dx_0}, \qquad a_{bc} = \frac{h_{eff}\,A}{V}
= \frac{h_{eff}}{dx_0}$$

(previous documentation labelled the coefficient "h/d": the half-cell series
form above is the one the code uses, `half_cell_h`, evaluated with the local first-cell
size).  With radiation enabled the same expression is used with $h \to h + h_r$.

Each of the six faces contributes with **its own** $h$ and $T_\infty$
(`Mesh3D.face_bc`), and a node on an edge or corner accumulates one term per
exposed face - the correct treatment, and the reason a corner of the top face no
longer borrows the top face's coefficient for its lateral faces.

### 5.4 Internal convection (heat-exchanger tubes)

Tube cells are interior cells with `boundary_type == CONVECTION`.  They are the
tube wall, so the fluid exchange is applied to the cell volume directly:

$$a_{tube} = \frac{h}{d_{cell}}, \qquad b_{tube} = a_{tube} T_{fluid}$$

i.e. a cell-sized surface area with the local cell size as the conduction length
($h/d_{cell}$ per unit volume).  The wall conduction resistance is not modelled
separately: the cell holds the tube-wall material (steel) and the film coefficient is
the tube-side one.  A buried pipe network uses the same cell coupling with the wetted
area of the pipe crossing the cell ([15](15_PIPE_NETWORKS.md)).

### 5.5 An internal film: the excluded air and the outer surface

The redesign removes the air around the vessel from the problem
([13](13_REDESIGN.md) §4).  A cell can be marked **excluded** (`mesh.excluded`): it
carries no conduction (`face_coefficients` zeroes any coefficient towards it) and it is
pinned at `mesh.t_ambient` by the Dirichlet table, so the system stays non-singular and
the report can still show the whole box.

Every active cell that faces an excluded one then exchanges with the environment
through the film `mesh.h_out`:

$$a_{film} = h_{out}\frac{A}{V}, \qquad b_{film} = a_{film}\,T_{ambient}
\qquad\text{(added to the row of the active cell)}$$

which is the same half-cell-free form as the tube convection because the "surface" is
the cell face, not a boundary of the domain.  The energy audit adds the same integral
(`analysis/fluxes.py::environment_flux`) and `compute_balance` folds it into
`q_domain`, otherwise the balance identity would miss the loss entirely.
`h_out = 0` makes that surface adiabatic - the documented way to switch the film off.

**Status**: the mechanism and its tests exist, but `BatteryGeometry` does not fill
`mesh.excluded`/`mesh.h_out` yet, so the default geometry still paints the air as a
conducting solid and uses the box-face convection of §5.3.

## 6. Transient formulation

Backward Euler (unconditionally stable, first order in time):

$$\rho c_p \frac{T^{n+1}-T^n}{\Delta t} = \nabla\cdot(k\nabla T^{n+1}) + Q$$

assembled per unit volume as

$$\left(\underbrace{\mathrm{diag}(\rho c_p)/\Delta t + L}_{A}\right) T^{n+1}
= \mathrm{diag}(\rho c_p)/\Delta t \cdot T^n + b$$

* `M = diag(rho*cp)` [J/(m³·K)] - **no cell volume**.  `L` is per unit volume, so
  a $V$-factor would scale every time constant by $1/\Delta^3$ and make the answer
  depend on the mesh size (that defect is the subject of the first regression
  test: the adiabatic heating rate must be $\Delta T = Qt/\rho c_p$ for any grid).
* The Dirichlet columns are eliminated *after* the mass diagonal is added, so the
  fixed temperatures hold exactly (`test_transient_enforces_dirichlet_faces_exactly`).
* The operators are built once per `dt`; the right-hand side is refreshed every
  step with the current sources (`transient_rhs`).  Warm starts (`x0 = T^n`) and
  the cached preconditioner make a step cheap.

## 7. Matrix structure and indexing

**Fortran order**, $p = i + jN_x + kN_xN_y$ (`Mesh3D.ijk_to_linear`), matching
`field.ravel(order="F")` used by every other module.  The matrix is sparse with
at most seven non-zeros per row, assembled as COO in one shot (pre-computed
`nnz`) and converted to CSR.

| offset | neighbour |
|---|---|
| $\pm 1$ | W/E (x) |
| $\pm N_x$ | S/N (y) |
| $\pm N_xN_y$ | D/U (z) |

## 8. Solvers

`src/solver/linear.py` selects the method and reports every substitution it had
to make:

| method | when |
|---|---|
| `direct` (sparse LU) | the `LinearConfig` default: reference answers and small meshes; memory and time grow fast, so measure with `scripts/benchmark.py` |
| `cg` | symmetric systems only - refused with a note on anything else; the graded operator is symmetrised with `solve_linear(..., scale=V)` |
| `bicgstab` | general, the GUI default |
| `gmres` | alternative for difficult systems |

Preconditioners: `none`, `jacobi`, `ilu`, `amg_rs`, `amg_sa` (PyAMG), with the
expensive AMG hierarchy cached while the matrix content is unchanged
(`fingerprint`).  There is **no GPU path**: the solver is CPU-only through SciPy, and
the thread budget is the only backend control (`set_num_threads`).

The relative residual $\|Ax-b\|/\|b\|$ is always recomputed on the host after the
solve, so the reported number does not depend on the solver's own bookkeeping.

## 9. Validation

* **Exactness**: with uniform $k$ and two Dirichlet faces the discrete solution
  is exactly the analytic parabola, including the source term
  (`test_steady_solution_is_exact_for_dirichlet_and_source`).
* **Reference assembly**: every row of the production matrix equals an
  independently written, unvectorised assembly
  (`test_matrix_matches_an_independent_assembly`).
* **Flux consistency**: the surface integrals of §5 satisfy
  $P_{in} = P_{out}$ to <1 % on convective-only configurations, and the
  half-cell Robin flux matches the analytic wall resistance within 2 %.
* **Grids**: a uniform `GridSpec` reproduces the legacy uniform mesh bit for bit, a
  linear profile is exact on a graded grid, the interface flux matches the series
  resistance, the answer does not move when the grid changes and the balance still
  closes (`tests/test_graded_mesh.py`).
* **Octree**: the 2:1 balance, the conservative face list and 1-D / two-material
  analytic solutions (`tests/test_octree.py`, [13](13_REDESIGN.md) §5).
* **Mesh convergence**: refine $\Delta$, the interior error must fall with the
  expected order; the transient rate must not move at all (§6).

## 10. Performance notes

Assembly is fully vectorised (`face_coefficients` builds the six coefficient
arrays at once) plus one COO construction; no Python loop over cells.  The
geometric factor `A/(d_centers·V)` of every face is built once per mesh and cached in
`GridIndex.face_factors`, warm starts (`x0 = T^n`) and the preconditioner cache keep
repeated solves (losses iteration, transient stepping) cheap, and the transient
operators are rebuilt only when `dt` or the radiation linearisation changes.
`scripts/benchmark.py` prints the numbers for the mesh sizes you care about.
