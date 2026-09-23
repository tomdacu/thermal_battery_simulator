# 18. The solver, end to end

What happens between *Build mesh* and a result, in the order the code does it, with the
equation of every step and the source it comes from (keys of
[17_REFERENCES.md](17_REFERENCES.md)).  The rationale of each choice against its
alternatives is in [12_METHODS.md](12_METHODS.md); the discrete operators in
[02_FDM_DISCRETIZATION.md](02_FDM_DISCRETIZATION.md).  Numbers are measured on the
default plant of the GUI (2026-09-23).

## 1. The mesh: a tree of boxes (`src/core/box_tree.py`)

The vessel is a vertical cylinder with vertical tubes: across a riser, a wall or the
insulation the temperature changes over centimetres, along them over metres.  A leaf of
the tree therefore has **two levels**: `lxy` for its edge in plan (x and y together) and
`lz` for its height, in finest cells of `dx` and `dz`.  It splits into four (plan), two
(height) or eight (both) children, so its aspect ratio is free
(`2^lz dz / 2^lxy dx`: 1:4, 4:1, 1:256 ...), the layered "2.5-D" arrangement of
reservoir and ground-heat grids [AS79] with the layers allowed to change from column to
column.

* **Balance.**  Two face neighbours differ by at most one level in plan and one in
  height - the 2:1 rule of [BWG11] per direction.  It bounds the size jump across a face
  and guarantees that a face is tiled by neighbours whose tangential extents are at
  least half its own.
* **Faces.**  The positive face of every leaf is probed just outside it at
  `corner + {0, half} x {0, half}` of its tangential extents; the unique hits are its
  neighbours.  Each pair carries the *overlap* area of the two faces and the centre
  distance along the normal `(e_i + e_j)/2`, once: the flux is conservative on any
  combination of levels (the balance closes to 1e-9, `tests/test_box_tree.py`).
* **Arrays only.**  The leaves are five integer arrays; locating a point is one
  `searchsorted` per (plan, height) level pair on sorted location codes, the linear
  tree of [BWG11].
* **What is refined** (`src/analysis/mesh_plan.py::active_regions`, the Mesh tab): the
  bed and the radial insulation to `r / cells` in plan and `height / layers` in height;
  the slabs, the roof and the foundation to their thickness over the insulation count
  in height; the soil coarse.  The air around the vessel is excluded (§4), so its leaves
  stay coarse.  No refinement follows the tubes: a tube is a line in its cell (§6).
  Every target snaps to the largest leaf within 1.6 of it (`LEAF_TOLERANCE`), and the
  cell budget caps the leaves.

Default plant: **36 268 leaves** (bed 195 x 301 mm, insulation 97 x 301 mm, slabs
195 x 75 mm), built and painted in **2.3 s**.  The cubic octree of the previous version
needed 214 089 leaves and 12 s for a coarser description of the same model.

## 2. Painting the model (`src/core/geometry.py`)

Cell centres decide the material: soil, concrete pad, insulation ring and slabs, bed,
shell, plate and roof.  Two corrections keep the thin layers physical on cells wider
than they are:

* **A shell thinner than its cell** (20 mm of steel on ~100 mm) is a *blend*: the outer
  cell is steel and insulation in series across it,
  `k = 1 / (f / k_steel + (1 - f) / k_insulation)` with `f = t / cell`, and their
  capacities by volume.  The insulation keeps its resistance and the shell its own mass;
  painting the whole cell as steel took up to 100 mm of insulation away and multiplied
  the steel by five.
* **A roof cone or a plate widened to a cell** keeps the steel's conduction along the
  surface and the steel's own mass (`_scale_capacity`).
* **The soil** under the pad (moist sand, 1.5 W/(m K) [VDI4640]) carries the
  deep-ground temperature a few metres down; its top outside the pad faces the ambient
  film and the box sides are a symmetry.

The pipe cells keep the bed's properties: a tube is a thin wall inside a cell of sand
and adds only its exchange (§6).

## 3. The bed's conductivity (`src/core/materials.py::PackedBed`)

The bed is a packed bed of grains with air in the voids.  Its conductivity is the
Zehner-Bauer-Schlünder model with the radiation between the grains [VDI-D6.3], [ZS70],
[BB80]; with porosity `psi`, `kappa = k_s / k_f`, grain diameter `d` and emissivity `eps`:

```
k_rad = 4 sigma T^3 d / ((2/eps - 1) k_f)
B     = C_f ((1 - psi) / psi)^(10/9),          C_f = 1.4 (broken grains)
N     = 1 + (k_rad - B) / kappa
k_c   = 2/N [ B (kappa + k_rad - 1) / (N^2 kappa) ln((kappa + k_rad) / B)
              + (B + 1)/(2B) (k_rad - B) - (B - 1)/N ]
k_bed = k_f [ (1 - sqrt(1 - psi)) (1 + psi k_rad)
              + sqrt(1 - psi) (phi kappa + (1 - phi) k_c) ],      phi = 0.0077
```

with `k_f` the air of [Incropera] Table A.4 at the cell's temperature.  Steatite at 63 %
packing, 1 mm grains: **0.30 W/(m K) at 20 °C, 0.57 at 500 °C, 0.68 at 700 °C** (the
constant geometric mean used before gave 0.52).  The law rides on the mesh
(`mesh.bed_conductivity`) and every solver re-evaluates it on the field it solves
(`update_bed_conductivity`): per sweep in the steady state, per step in the transient,
a cell updated only when its value moved by more than 2 % so the operator survives.

## 4. The envelope

The air outside the shell and above the roof is **excluded**: its cells are pinned at
the ambient and the active cells facing them carry the outer film
`h = h_natural + h_wind` (Churchill-Chu on the vessel [CC75], `4 + 4 v` [ISO6946];
radiation linearised and iterated when switched on).  The ground face under the soil is
fixed at the ground temperature.  An excluded cell is an identity row after the
symmetric Dirichlet elimination, and the linear layer solves it directly (§8).

## 5. The gas circuit (`src/solver/fluid.py`, `src/core/pipe_network.py`)

The resistors heat the gas of a closed circuit; the gas crosses the inlet duct, the
distributor, one riser, the collector and the outlet duct, and hands its heat to the
bed through every wall it touches [PNE].  `PipeNetwork.gas_graph` builds that circuit
as **straight segments between nodes**, each with the share of the flow it carries: the
branch split on the risers, the flow still to be delivered along the distributor (it
falls tap by tap) and the flow already collected along the collector.  A ring is fed at
its entry tap and divides both ways round to its exit tap; mass is conserved at every
node (`GasGraph.check`).

On each segment of wetted area `A` at wall temperature `T_w` the gas follows the
effectiveness relation [Kays]:

```
T_out = T_w + (T_in - T_w) exp(-NTU),     NTU = UA / (m_dot c_p)
q     = m_dot c_p (T_in - T_out) = G (T_in - T_w),     G = m_dot c_p (1 - exp(-NTU))
```

The segments are marched in topological order; at a node the streams mix by enthalpy.
Every temperature stays **affine in the loop inlet** `T_in`, so the balance

```
sum over the cells of q = Q_ext        (+ resistors, - exchanger)
```

is solved for `T_in` in one step, and the bed receives exactly the external power.

**Film.**  `h` from Dittus-Boelter (turbulent), `Nu = 3.66` (laminar), blended in
between [Incropera]; the friction factor from Haaland [Haaland] and the roughness of the
tube material [Idelchik].

**Gas properties** follow the gas: every segment is marched with `cp`, `mu`, `k` and
`Pr` at its own mean temperature (air and nitrogen from [Incropera] Table A.4, steam
from Table A.6), lagged by one march and refreshed when the gas has moved by more than
10 K (cp changes by ~0.2 % in 10 K; the hysteresis keeps the operator).

## 6. The tube in its cell: the well model

A tube is a line inside a cell of the bed, so the cell temperature is not the wall
temperature.  For a line source in a cell of cross-section `a x b` the discrete
solution equals the radial one at the **equivalent radius** [Pea78], [Pea83]

```
r_eq = 0.14 sqrt(a^2 + b^2)        (0.198 h for a square cell of edge h)
```

so the wall-to-cell resistance per unit length is `ln(r_eq / r_o) / (2 pi k_bed)`, in
series with the tube wall and the gas film:

```
1 / UA' = 1 / (h pi d_i) + ln(r_o / r_i) / (2 pi k_wall) + ln(r_eq / r_o) / (2 pi k_bed)
```

`a, b` are the two edges of the cell across the segment (plan for a riser, plan and
height for a header).  This makes the exchange independent of the cell: one tube in a
2.4 m square of sand held at 300 K matches the shape factor of [Incropera] Table 4.1
within 3.5 % on 300 mm cells and 1.5 % on 150 mm cells; without the model the same
exchange was off by +26 % and +5 %, and by -9 % on 75 mm cells
(`tests/test_well_model.py`).  The model needs `r_eq > r_o`, i.e. cells at least
`2.53 d` wide: the Mesh tab warns below that, and a result lists the tube area that sat
in smaller cells.  It is quasi-steady: valid once the thermal wave has crossed the cell,
`t > r_eq^2 / alpha` (about an hour for 40 mm of sand).

## 7. Coupling the gas and the bed

The march gives every crossed cell its `G` and the temperature `T_gas` the gas enters it
at; `FluidResult.apply` writes them as an **implicit film** (`bc_h`, `bc_T_inf`), so the
wall temperature is solved implicitly and the coupling is stable at any step.  The loop
inlet is solved *with* the field: every `T_gas` moves with `T_in` by a known slope, the
field moves by the solution `y` of one more system with the same operator, and the
scalar that makes the deposit equal `Q_ext` closes the balance on the new field (a
one-row Schur complement, `hold_loop_balance`; `y` is solved once per operator).

## 8. The linear systems (`src/solver/linear.py`)

Per unit volume the operator is `diag(V)^-1 K` with `K` symmetric; it is symmetrised by
`diag(V)^(1/2)` and solved by **CG with an AMG preconditioner** (Ruge-Stüben [RS87]
through [PyAMG], V(1,1) with forward/backward Gauss-Seidel) [Saad].

* **Decoupled rows** - the excluded air and the Dirichlet walls, which the elimination
  leaves as a bare diagonal - are solved directly; CG sees the coupled rows only (a
  quarter fewer unknowns on the default plant).
* **Hierarchy reuse.**  The AMG setup is the dominant cost.  It is kept for the same
  matrix, and reused for a matrix with the same pattern whose entries all moved by less
  than 20 %: a preconditioner only has to be close to the operator, and CG still
  converges to the tolerance asked [Saad].
* **Tolerance.**  Relative residual 1e-6 (a few hundredths of a kelvin on the field);
  the coupled iterations stop at 0.1 K.

## 9. The analyses

* **Standby** (`src/analysis/losses.py`): the bed held at a temperature; the holding
  power - the losses - by a secant iteration on the power, each step a coupled steady
  solve (Picard sweeps: loop march, bed conductivity, linear solve).  Default plant at
  500 °C: **6.4 kW in 10.7 s**, two secant steps.
* **Transient** (`src/solver/transient.py`): backward Euler; the loop is marched at the
  end of every step and its film is implicit in the next; the bed conductivity and the
  gas properties are lagged by one step.  Default plant, 200 kW charge, dt = 900 s:
  **6 h of operation in 18 s**.
* **Automatic mesh** (`src/analysis/convergence.py`): the standby on finer and finer
  trees, Richardson extrapolation and the GCI [Roache], [Celik08]; on the box tree the
  flux-jump rounds split a leaf in plan, in height or both, following the axis its error
  lies on.

## 10. What the model does not do (stated, not hidden)

* The gas is 1-D and incompressible (valid while the circuit loses less than 10 % of
  its pressure; the result says when it does not).
* Several tubes in one cell share it (the well model assumes one): cells of the bed are
  kept wider than a tube, not wider than the pitch.
* The flow split is imposed (equal, by path, by ring, by sector), not solved from the
  hydraulics of the network.
* The Smoluchowski reduction of the gas conductivity near the grain contacts is left
  out (it matters below ~0.1 mm or under vacuum).
* No natural convection inside the roof air or the pores.
