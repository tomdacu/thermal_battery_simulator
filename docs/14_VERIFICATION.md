# 14. Verification campaign

Date: 2026-09-20

This campaign was run without changing source code; this report is the requested
documentation output. Temperatures in the implementation are Kelvin; Celsius
values below are used only to make the checks readable.

## Test suite

The requested command was run exactly as follows:

```text
python -m pytest tests/ -q --ignore=tests/test_gui_sweep.py
```

| Result | Wall time |
|---|---:|
| **195 passed, 0 failed** | **46.72 s** |

The run used Python 3.13.5, NumPy 2.4.3, SciPy 1.17.1 and PyAMG 5.3.0 on an
8-CPU Windows machine with 16.9 GB of physical memory.

## Steady-solver timing

Each case used `Mesh3D(Lx=8, Ly=8, Lz=7, spacing=...)`, followed by
`create_small_test_geometry().apply_to_mesh(mesh)`, and then
`SteadyStateSolver` with:

```text
method="cg", preconditioner="amg_rs", tolerance=1e-8,
max_iterations=5000, n_threads=-1
```

The solve time is measured with `time.perf_counter()` immediately around
`solver.solve()`. It includes matrix assembly, AMG setup and the linear solve;
imports, mesh construction and `SteadyStateSolver` construction are outside the
timed interval. The mesh-build column is included to show that separation.

| Target | Requested spacing | Realised grid | Cells | Mesh build [s] | Solve wall time [s] | Converged | Relative residual |
|---|---:|---:|---:|---:|---:|---|---:|
| ~10k | 0.35 m | 23 x 23 x 20 | 10,580 | 0.0047 | **0.1163** | yes | 5.66e-9 |
| ~100k | 0.16 m | 50 x 50 x 44 | 110,000 | 0.0123 | **1.4358** | yes | 2.98e-9 |
| ~400k | 0.10 m | 80 x 80 x 70 | 448,000 | 0.0777 | **7.5782** | yes | 2.11e-9 |

The `SolverResult.iterations` value is the number of steady/Picard sweeps. With
radiation disabled there is one sweep; it is not a count of CG Krylov steps.
The existing timing script uses the lower-level matrix and `solve_linear` calls
(`scripts/benchmark.py:45-62`), so the values above are a separate measurement
of the requested public `Mesh3D` to `SteadyStateSolver` path.

The timed geometry is the existing 50 kW small-test case, not a physical 5 kW
design. With radiation off its maximum temperatures were 4,209.6 K, 5,006.9 K
and 6,514.9 K for the three meshes. These fields are useful for stressing the
solver but are outside material service limits and must not be interpreted as a
real operating prediction.

### Memory estimate

`Mesh3D._estimate_memory()` counts 12 float64 arrays and two one-byte arrays per
cell, although the mesh actually allocates four one-byte arrays. A 7-point CSR matrix needs approximately 88 bytes per cell when values
are float64 and indices are int32. The face-coefficient workspace adds another
48 bytes per cell. The following is therefore a lower-bound estimate before the
AMG hierarchy and temporary assembly arrays:

| Cells | Mesh estimate reported by code [MB] | Mesh including both boolean masks [MB] | 7-point CSR [MB] | Mesh + CSR + face workspace [MB] |
|---:|---:|---:|---:|---:|
| 10,580 | 1.04 | 1.06 | 0.93 | 2.50 |
| 110,000 | 10.78 | 11.00 | 9.68 | 25.96 |
| 448,000 | 43.90 | 44.80 | 39.42 | 105.73 |

For the 448,000-cell run, process RSS was 69.4 MB before the case and peaked at
493.3 MB during AMG setup and the solve, an observed increase of 423.9 MB. The
peak is the useful capacity-planning number; the table is the component estimate
and does not include the AMG hierarchy or allocator overhead.

## Hand checks

### 50 m3 sand bed heated with 5 kW

The primary calculation uses the repository's `silica_sand` entry with the
default 0.63 solid packing fraction, 0.37 air porosity, and no losses. The
temperature interval used for a finite storage capacity is 20 to 500 degC,
so `Delta T = 480 K`.

The packed-bed volumetric heat capacity is

```text
rho cp = 0.63(1500)(800) + 0.37(1.2)(1005)
       = 756,446 J/(m3 K)
```

| Property | Solid database value used directly | Default packed-bed calculation |
|---|---:|---:|
| Volumetric heat capacity | 1,200,000 J/(m3 K) | 756,446 J/(m3 K) |
| Capacity of 50 m3 | 16.667 kWh/K | **10.506 kWh/K** |
| Temperature rise at 5 kW | 0.300 K/h | **0.476 K/h** |
| Thermal capacity, 20 to 500 degC | 8,000 kWh | **5,042.97 kWh** |
| 5 kW heating time for 480 K | 1,600 h | **1,008.6 h** |
| 87% reporting efficiency | 6,960 kWh | **4,387.39 kWh usable** |

The packed result is internally consistent with the implemented effective
properties. It also shows the scale of the scenario: 5 kW raises a 50 m3 packed
bed by only about half a degree per hour before losses.

### 30 cm rock wool around a 4 m diameter cylinder

The request gives no cylinder height, so the side-wall result is reported per
metre of height and an illustrative 4 m-high cylinder is shown in the last
column. The assumed inner and outer radii are 2.0 m and 2.3 m, the hot inner
surface is 500 degC, and the ambient is 20 degC.

| Calculation | Heat loss per metre height | For 4 m height |
|---|---:|---:|
| Planar approximation, inner area | 804 W/m | 3.217 kW |
| Exact cylindrical rock-wool conduction | **863 W/m** | **3.453 kW** |
| Cylindrical conduction + 20 mm steel (`k=50`) + `h=5` film | 842 W/m | 3.369 kW |

The equations are

```text
q'' = k DeltaT / L = 64 W/m2
q'_cyl = 2 pi k DeltaT / ln(r2/r1) = 863 W/m
```

Thus a 4 m-high side wall in this condition should lose roughly 3.2 to 3.5 kW
before top, bottom, penetrations and thermal bridges. The 30 cm insulation
dominates the resistance; the illustrative default 5 W/(m2 K) outer film and
20 mm steel change the exact cylindrical result only modestly.

### Blower power: eight 50 mm x 4 m vertical pipes

This check assumes that 0.1 kg/s is the **total loop flow**, split evenly between
eight parallel pipes, with the default `Fluid()` at 300 K and 1 atm, 45 um wall
roughness, no fittings, and 70% blower efficiency. The pipe-only result is a
lower bound because headers, bends and valves are omitted.

For each pipe, `m_dot = 0.1/8 = 0.0125 kg/s`, velocity is 5.305 m/s,
`Re = 17,684`, the Haaland Darcy factor is 0.02814, and the pressure drop is
38.0 Pa. Therefore:

```text
P_fan = (0.1 / 1.2) (38.0) / 0.70 = 4.53 W
```

| Case | Flow used per pipe | One-pipe pressure drop | Loop pressure drop | Blower power |
|---|---:|---:|---:|---:|
| Hand calculation, total 0.1 kg/s split eight ways | 0.0125 kg/s | **38.0 Pa** | 38.0 Pa | **4.53 W** |
| Current `FluidLoop(runs=8, mass_flow=0.1)` | 0.0125 kg/s for heat exchange, 0.1 kg/s for hydraulics | 1,814 Pa in each hydraulic call | **14,514 Pa** | **1.728 kW** |

The current result is 382 times the pipe-only hand estimate and is 34.6% of a
5 kW heater input. If the gas itself is at 500 degC, applying
`Fluid().at(773.15)` to the same correct split-flow calculation gives about
34.9 W before fittings; the hot-gas value is still far below 1.728 kW.

## Discrepancies found

The following are the discrepancies exposed by this campaign. They are recorded
with file and line references; no fixes were applied because this task is
documentation-only.

| Finding | Evidence | Consequence |
|---|---|---|
| Parallel-pipe hydraulics use the wrong flow. | `src/solver/fluid.py:276-278` correctly splits total flow for the thermal march, but `src/solver/fluid.py:341-349` passes `self.mass_flow` to `pressure_drop` for every run. `src/core/pipes.py:107-110` describes the bank as parallel runs. | The requested eight-pipe, 0.1 kg/s case reports 14.5 kPa and 1.728 kW instead of about 38 Pa and 4.53 W, before fittings. |
| Fluid temperature dependence is not used automatically at ambient pressure. | The module documents temperature-dependent properties at `src/solver/fluid.py:80-88`, but `src/solver/fluid.py:266-268` keeps `self.fluid` unchanged whenever pressure is 1 atm. | A loop using the default fluid is evaluated at 300 K even when the wall or gas is near 500 degC; the 34.9 W hot-gas estimate is not produced unless the caller supplies a temperature-adjusted `Fluid`. |
| The default battery does not apply the planned outer film directly at the cylinder envelope. | `src/core/geometry.py:463-465` paints the surrounding air as active cells and `src/core/geometry.py:479-488` applies convection to the six box faces. `src/core/mesh.py:207-213` leaves `excluded` false and `h_out` zero, while `src/solver/matrix.py:142-155` applies `h_out` only at active/excluded interfaces. The direct envelope Robin treatment is still described as planned in `docs/13_REDESIGN.md:192-215`. | The 30 cm rock-wool hand resistance cannot be compared directly with the default 3-D geometry: heat can conduct through the simulated air box before reaching its box-face film. |
| Silica sand is packed twice in the effective-property model. | `src/core/materials.py:50-52` labels `k=0.35 W/(m K)` as a bulk sand-bed value, while `src/core/materials.py:123-140` applies the solid/void mixing model; the geometry calls it through `src/core/geometry.py:448-450`. The same issue is acknowledged in `docs/12_METHODS.md:241-248`. | The default packed calculation gives `k_eff` about 0.13 W/(m K), and its density/capacity differ from treating the database entry as an already-packed bed. The capacity table therefore reports both interpretations. |
| The default storage material is steatite, not silica sand. | `src/core/geometry.py:360-366` sets `storage_material="steatite"`; the sand check must explicitly select `silica_sand`. | A default GUI or geometry run does not represent the sand-bed hand calculation. |
| The timed benchmark is numerically converged but physically out of range. | `src/core/geometry.py:767-776` uses a 50 kW source; `src/solver/steady.py:21-27` leaves radiation off by default. The measured peak temperatures reach 6,515 K, above the material limits in `src/core/materials.py:50-70`. | The timing result is a solver-performance result only, not evidence that the test geometry is a valid 500 degC operating point. |
| The mesh memory report is incomplete for a solve. | `src/core/mesh.py:433-436` counts two small arrays, although `src/core/mesh.py:207-213` and `src/core/mesh.py:217` allocate four (`material_id`, `boundary_type`, `excluded`, and `source_mask`); it also reports mesh arrays only. Matrix face work is allocated at `src/solver/matrix.py:50-60`, and AMG hierarchy storage is retained by `src/solver/linear.py:127-134`. | `Mesh3D.get_info()['memory_MB']` is a mesh estimate, not a peak solver requirement. The 448k run measured a 493.3 MB RSS peak versus 43.9 MB for the code's mesh estimate. |
| The documented test counts are stale. | `docs/09_TESTING.md:4` and `docs/11_HANDOFF.md:104` say 147 tests; `docs/11_HANDOFF.md:12` says 169. | The exact requested command currently reports 195 passed tests. |
| The README overstates the default solver configuration. | `README.md:38-40` says CG + AMG is the default, while `src/solver/linear.py:33-40` defaults to direct + Jacobi and `gui/controller.py:39-44` defaults to BiCGSTAB + Jacobi. The GUI labels Jacobi as default at `gui/views/solver_panel.py:16-21`. | The benchmark explicitly selected CG + AMG; ordinary solver runs do not select that pair by default. |
| The README still recommends GPU use after the GPU path was removed. | `README.md:347` recommends a GPU above 100k cells, while `docs/09_TESTING.md:126` and the CPU-only solver description at `src/solver/linear.py:1-4` say the supported solver is CPU-only. | The recommendation is not actionable on the current code path. |
| The numerical method is called FDM in some public labels but finite volume in the implementation description. | `README.md:35` and `src/core/mesh.py:1` say FDM, whereas `docs/12_METHODS.md:14-20` describes the cell-centred finite-volume flux balance. | This is terminology rather than a numerical failure, but it can mislead readers about the discretisation being verified. |

The sand capacity and cylindrical insulation checks are consistent at the level
of the stated assumptions. The blower result and the default envelope treatment
are the material realism failures that require attention before those outputs are
used as design predictions.
