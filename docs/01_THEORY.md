# Heat Transfer Theory - Thermal Battery Simulation

## 1. Simulation Objectives

The simulation aims to predict the spatial temperature distribution inside a thermal energy storage system. This enables:
- Identification of thermal stagnation zones or excessive heat losses.
- Calculation of actual energy capacity based on non-uniform temperature distribution.
- Evaluation of the effectiveness of heat extraction (tubes) and heating (resistors) systems.

---

## 2. Fundamental Equations

### 2.1 General Heat Equation

The equation governing temperature distribution in a solid is:

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$$

Where:
- $\rho$ = material density [kg/m³]
- $c_p$ = specific heat at constant pressure [J/(kg·K)]
- $T$ = temperature [K or °C]
- $t$ = time [s]
- $k$ = thermal conductivity [W/(m·K)]
- $Q$ = volumetric heat source/sink [W/m³]

### 2.2 Steady-State Case

When $\frac{\partial T}{\partial t} = 0$:

$$\nabla \cdot (k \nabla T) + Q = 0$$

For constant conductivity:

$$k \nabla^2 T + Q = 0$$

---

## 3. Heat Transfer Mechanisms

### 3.1 Conduction (Fourier's Law)

The heat flux by conduction is:

$$\vec{q} = -k \nabla T$$

In one-dimensional scalar form:

$$q = -k \frac{dT}{dx}$$

### 3.2 Convection (Newton's Law)

The heat flux by convection between a surface and a fluid:

$$q = h (T_s - T_\infty)$$

Where:
- $h$ = convective heat transfer coefficient [W/(m²·K)]
- $T_s$ = surface temperature
- $T_\infty$ = fluid temperature far from the surface

**Typical values of h:**
| Condition | h [W/(m²·K)] |
|-----------|--------------|
| Natural convection in air | 5-25 |
| Forced convection in air | 25-250 |
| Natural convection in water | 100-900 |
| Forced convection in water | 50-20,000 |

### 3.3 Radiation (Stefan-Boltzmann Law)

$$q = \epsilon \sigma (T_s^4 - T_{surr}^4)$$

Where:
- $\epsilon$ = surface emissivity (0-1)
- $\sigma$ = 5.67 × 10⁻⁸ W/(m²·K⁴)

**Note:** For typical Thermal Battery temperatures (< 600°C), radiation 
is significant but often linearized or included in an effective h coefficient.

---

## 4. Thermal Resistances

### 4.1 Electrical Analogy

Like in an electrical circuit:
- Temperature ↔ Voltage
- Heat flux ↔ Current
- Thermal resistance ↔ Electrical resistance

$$q = \frac{\Delta T}{R_{th}}$$

### 4.2 Resistances in Series

$$R_{tot} = R_1 + R_2 + R_3 + ...$$

### 4.3 Types of Thermal Resistance

**Conduction (flat wall):**
$$R_{cond} = \frac{L}{k \cdot A}$$

**Conduction (cylinder):**
$$R_{cond,cyl} = \frac{\ln(r_2/r_1)}{2\pi k L}$$

**Convection:**
$$R_{conv} = \frac{1}{h \cdot A}$$

---

## 5. Heat Transfer in Porous Materials

### 5.1 Effective Thermal Conductivity

For sand with air in the pores, the effective conductivity can be estimated:

**Parallel model (upper bound):**
$$k_{eff,\parallel} = \phi \cdot k_{fluid} + (1-\phi) \cdot k_{solid}$$

**Series model (lower bound):**
$$\frac{1}{k_{eff,series}} = \frac{\phi}{k_{fluid}} + \frac{1-\phi}{k_{solid}}$$

**Geometric mean (good approximation):**
$$k_{eff} = k_{solid}^{(1-\phi)} \cdot k_{fluid}^{\phi}$$

Where $\phi$ = porosity (void fraction)

### 5.2 Effective Thermal Capacity

$$(\rho c_p)_{eff} = \phi \cdot (\rho c_p)_{fluid} + (1-\phi) \cdot (\rho c_p)_{solid}$$

---

## 6. Boundary Conditions

### 6.1 Dirichlet (Prescribed Temperature)

$$T|_{\Gamma} = T_{prescribed}$$

Example: Battery base in contact with ground at constant temperature.

### 6.2 Neumann (Prescribed Flux)

$$-k \frac{\partial T}{\partial n}\bigg|_{\Gamma} = q_{prescribed}$$

Example: Adiabatic surface (q = 0) for symmetry.

### 6.3 Robin (Convection)

$$-k \frac{\partial T}{\partial n}\bigg|_{\Gamma} = h(T_s - T_\infty)$$

Example: External surface in contact with ambient air.

---

## 7. Dimensionless Numbers

### 7.1 Biot Number

$$Bi = \frac{h \cdot L_c}{k}$$

- $Bi << 1$: Uniform temperature in the solid (lumped capacitance)
- $Bi >> 1$: Significant gradients in the solid

### 7.2 Fourier Number

$$Fo = \frac{\alpha \cdot t}{L_c^2}$$

Where $\alpha = k/(\rho c_p)$ = thermal diffusivity [m²/s]

- Indicates how "close" the system is to thermal equilibrium

---

## 8. Application to the Thermal Battery

### 8.1 Battery Architecture

The battery is modeled as a vertical cylinder composed of different concentric radial zones:

1.  **STORAGE Zone**: Central area containing the thermal storage material with embedded tubes and heaters.
2.  **INSULATION Zone**: Layer of low-conductivity material (e.g., rock wool) to minimize losses.
3.  **STEEL Zone**: External steel shell for structural protection.
4.  **AIR Zone**: External air (outside the shell).

Additionally, the geometry includes:
- **Insulation slabs** at top and bottom for vertical thermal protection.
- **Optional conical roof** for realistic geometry modeling.
- **Phase offset** between tubes and heaters to avoid overlapping.

### 8.2 Global Energy Balance

**Charging state:**
$$\dot{E}_{in} = \dot{E}_{stored} + \dot{E}_{losses}$$

$$P_{heaters} = \frac{d}{dt}(m \cdot c_p \cdot \bar{T}) + P_{dispersion}$$

**Discharging state:**
$$\dot{E}_{stored} = \dot{E}_{out} + \dot{E}_{losses}$$

$$\frac{d}{dt}(m \cdot c_p \cdot \bar{T}) = P_{tubes} + P_{dispersion}$$

### 8.3 Heater Power

The thermal power generated per unit volume ($Q$) is distributed in the heater zone:

$$P_{heaters} = \sum_i Q_i \cdot V_i$$

Where $V_i$ is the volume of cell i belonging to the heating zone.

### 8.4 Power Extracted by Tubes

Heat exchange with the fluid in tubes is modeled through an internal convection condition:

$$q_{tubes} = h_{fluid} (T_{wall} - T_{fluid})$$

### 8.5 Heat Losses

Dispersion to the environment occurs by convection on external surfaces:

$$P_{disp} = \oint h(T_s - T_\infty) \, dA$$

And by conduction toward the ground at the base:

$$q_{base} = -k \frac{\partial T}{\partial z}\bigg|_{z=0}$$

---

## 9. Exergy Analysis (Overview)

### 9.1 Thermal Exergy

The exergy associated with a heat flow Q at temperature T:

$$\dot{Ex} = \dot{Q} \cdot \left(1 - \frac{T_0}{T}\right)$$

Where $T_0$ = reference temperature (ambient)

### 9.2 Stored Exergy

$$Ex_{stored} = m \cdot c_p \cdot \left[(T - T_0) - T_0 \cdot \ln\left(\frac{T}{T_0}\right)\right]$$

### 9.3 Exergy Destruction

For a system with temperatures $T_1$ and $T_2$ exchanging heat Q:

$$\dot{Ex}_{destr} = T_0 \cdot \dot{Q} \cdot \left(\frac{1}{T_2} - \frac{1}{T_1}\right)$$

---

## 10. References

1. Incropera, F.P., DeWitt, D.P. - "Fundamentals of Heat and Mass Transfer"
2. Çengel, Y.A. - "Heat Transfer: A Practical Approach"
3. Bejan, A. - "Advanced Engineering Thermodynamics" (for exergy)
4. Kaviany, M. - "Principles of Heat Transfer in Porous Media"
