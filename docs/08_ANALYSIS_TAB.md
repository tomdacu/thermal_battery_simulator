# Analysis Tab Documentation

## Overview

The **Analysis** tab allows configuration of advanced thermal simulations:
- **Steady-State**: Thermal equilibrium with constant power
- **Losses Analysis**: Calculates thermal losses given target storage temperature
- **Transient**: Time-dependent evolution (charge/discharge cycles)

## Sub-tab Structure

```
Analysis
├── Type          → Analysis type selection + specific parameters
├── Conditions    → Initial conditions for transient
├── Power         → Heater power profile
├── Extraction    → Heat extraction from tubes
└── Save/Load     → Save/load simulation state
```

---

## 1. Analysis Type

### Steady-State
Calculates steady-state temperature distribution:
- **Input**: Total heater power [kW]
- **Output**: Temperature field T, thermal losses, efficiency

### Losses Analysis (Iterative)
Sets target mean temperature in storage and calculates thermal losses using an **iterative secant method**:

- **Input**: Target T storage [°C], Ambient T [°C]
- **Output**: Total losses [kW], breakdown by face (top, side, bottom), thermal autonomy

#### Algorithm Description

The losses analysis uses a **secant method with underrelaxation** to find the heat source Q that produces the target mean temperature:

```
1. INITIALIZE
   - T_target = user-specified target temperature
   - T_amb = ambient temperature
   - T_ground = ground temperature (configurable)
   - h_conv = external convection coefficient (configurable)
   
2. SET BOUNDARY CONDITIONS
   - External faces: convection BC with h_conv, T_inf = T_amb or T_ground
   - Internal cells: set initial temperature profile
   
3. INITIAL GUESSES
   - Q₀ = small value (e.g., 0.1 W/m³)
   - Solve steady-state → T_mean₀
   - Q₁ = slightly larger value (e.g., 1.0 W/m³)
   - Solve steady-state → T_mean₁

4. ITERATE (secant method)
   FOR iteration = 1 to max_iter:
       error = T_mean - T_target
       
       IF |error| < tolerance:
           CONVERGED → EXIT
       
       # Secant update
       dQ_dT = (Q_current - Q_prev) / (T_mean_current - T_mean_prev)
       Q_newton = Q_current - error * dQ_dT
       
       # Underrelaxation for stability
       Q_new = α * Q_newton + (1 - α) * Q_current
       
       # Solve with new Q
       Apply Q_new to all sand cells
       Solve steady-state → T_mean_new
       
       # Update for next iteration
       Q_prev, T_mean_prev = Q_current, T_mean_current
       Q_current, T_mean_current = Q_new, T_mean_new
   
5. CALCULATE LOSSES
   Q_total = Q_new × V_sand [kW]
   Q_top, Q_side, Q_bottom = surface flux integrals
```

#### Configurable Parameters

| Parameter | Widget | Default | Description |
|-----------|--------|---------|-------------|
| Tolerance | `losses_tol_spin` | 1.0 °C | Acceptable error on T_mean |
| Max iterations | `losses_maxiter_spin` | 20 | Maximum Q-T iterations |
| α (underrelaxation) | `losses_alpha_spin` | 0.5 | Damping factor (0.1=stable, 1.0=fast) |
| h_conv | `losses_h_conv_spin` | 10.0 W/(m²·K) | External convection coefficient |
| T_ground | `losses_T_ground_spin` | 15.0 °C | Ground temperature |

#### Key Features
- **Physical Consistency**: Produces temperature profiles that satisfy the heat equation
- **Convergence**: Secant method converges quadratically near the solution
- **Stability**: Underrelaxation (α < 1) prevents oscillations
- **GPU Support**: Each steady-state solve uses configured GPU/CPU backend

### Transient Analysis
Simulates time-dependent evolution:
- **Duration**: Total simulation time [s]
- **dt**: Time step [s] (recommended: 60s)
- **Save interval**: How often to save results
- **Numerical scheme**: Backward Euler (implicit, unconditionally stable)

---

## 2. Initial Conditions

### Uniform Temperature
Entire domain at constant temperature.

### By Material
Different temperatures for each material:
- Sand (storage)
- Insulation
- Steel
- Air
- Ground
- Concrete

### From File
Load temperature field from previous simulation (.h5)

### From Steady-State
First calculate steady-state, then use as initial condition for transient.

---

## 3. Power Profile

### Off
Heaters disabled (discharge/losses only)

### Constant
Fixed power throughout simulation.

### Scheduled
Table of (time, power):
```
t [s]    P [W]
0        10000
3600     5000
7200     0
```

### From CSV
CSV file with columns `time,power`:
```csv
time,power
0,10000
3600,5000
7200,0
```

---

## 4. Heat Extraction

### Off
No extraction from tubes

### Imposed Power
Specify power to extract [W]

### Flow Rate
Specify fluid flow rate [kg/s]:
- Per tube or total
- Fluid: water, thermal oil, air, glycol

### Target Outlet Temperature
Impose desired outlet temperature [°C]

---

## 5. State Save/Load

### Save
- Simulation name
- Optional description
- Format: compressed HDF5 (.h5)

### Load
- Browse for .h5 file
- Geometry compatibility check (hash)
- Loads temperature field into mesh

---

## 6. Transient Numerical Scheme

**Backward Euler** (implicit):

$$
\frac{\rho c_p}{\Delta t}(T^{n+1} - T^n) = \nabla \cdot (k \nabla T^{n+1}) + Q^{n+1}
$$

In matrix form:

$$
\left(\frac{M}{\Delta t} + A\right) T^{n+1} = \frac{M}{\Delta t} T^n + b
$$

**Advantages**:
- Unconditionally stable (any dt)
- One linear system per timestep
- Reuses AMG preconditioner

---

## 7. State File Format (.h5)

```
/metadata
    version, timestamp, name, analysis_type
/geometry
    mesh_shape, geometry_hash, parameters
/state
    T (compressed)
    materials (compressed)
/results
    computed quantities
```

---

## 8. Energy Balance Output

After Losses Analysis completes, the Energy Balance panel displays:

```
╔══════════════════════════════════════════════╗
║      ENERGY BALANCE - LOSSES ANALYSIS        ║
╠══════════════════════════════════════════════╣
║ CONDITIONS                                   ║
╠──────────────────────────────────────────────╣
║ T target sand:    600.0 °C                   ║
║ T final mean:     599.8 °C                   ║
║ T ambient:         20.0 °C                   ║
║ T ground:          15.0 °C                   ║
║ h convection:      10.0 W/(m²·K)             ║
╠══════════════════════════════════════════════╣
║ THERMAL LOSSES                               ║
╠──────────────────────────────────────────────╣
║ TOTAL:            12.50 kW                   ║
║   - Top:           3.20 kW                   ║
║   - Side:          7.80 kW                   ║
║   - Bottom:        1.50 kW                   ║
╠──────────────────────────────────────────────╣
║ Loss density:     125.0 W/m³                 ║
╠══════════════════════════════════════════════╣
║ STORED ENERGY                                ║
╠──────────────────────────────────────────────╣
║ E thermal:       1250 kWh                    ║
║                   1.25 MWh                   ║
║ Autonomy:        100.0 hours                 ║
╠══════════════════════════════════════════════╣
║ CONVERGENCE: ✓ CONVERGED (8 iter)            ║
╚══════════════════════════════════════════════╝
```

### Autonomy Calculation

The thermal autonomy represents how long the battery can supply heat at the current loss rate:

$$
\text{Autonomy} = \frac{E_{stored}}{Q_{losses}}
$$

Where:
- $E_{stored} = \sum_i m_i \cdot c_{p,i} \cdot (T_i - T_{amb})$ [kWh]
- $Q_{losses}$ = total thermal losses [kW]

---

## 9. Typical Workflows

### Sizing (Steady-State)
1. Configure geometry
2. Set heater power
3. Run steady-state
4. Check T_max, thermal losses

### Cycle Analysis (Transient)
1. Configure geometry
2. Initial condition: uniform 20°C
3. Power: charge/discharge schedule
4. Run transient (e.g., 24h)
5. Visualize T vs time plots
6. Export results to CSV

### Loss Verification (Losses Analysis)
1. Configure complete geometry
2. Type: Losses Analysis
3. Set T storage = 600°C (or target)
4. Configure parameters (α=0.5, h_conv=10, T_ground=15)
5. Run → get required power to maintain temperature
6. Review energy balance for autonomy estimate

---

## 10. Performance Considerations

### For Losses Analysis
- Each iteration requires a full steady-state solve
- GPU acceleration (CUDA/OpenCL) significantly speeds up convergence
- Lower α (0.3-0.5) is more stable but slower
- Higher α (0.7-1.0) is faster but may oscillate

### Recommended Settings
| Scenario | α | Tolerance | GPU |
|----------|---|-----------|-----|
| Large mesh (>1M cells) | 0.3 | 2.0°C | CUDA |
| Medium mesh (100k-1M) | 0.5 | 1.0°C | OpenCL |
| Small mesh (<100k) | 0.7 | 0.5°C | CPU |
