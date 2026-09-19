# 1. Heat transfer theory

## 1.1 What the simulation predicts

The spatial temperature distribution inside a thermal energy storage unit (a
"sand battery"): where heat stagnates, how much energy is really stored, how
much leaks to the environment, and how the unit responds to a heating or
extraction schedule.

## 1.2 Governing equation

$$\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k \nabla T) + Q$$

with $\rho$ [kg/m³], $c_p$ [J/(kg·K)], $k$ [W/(m·K)] and the volumetric source
$Q$ [W/m³].  Steady state drops the time derivative:

$$\nabla \cdot (k \nabla T) + Q = 0$$

**Unit contract.**  Inside `src/` every temperature is an absolute temperature in
**Kelvin**; the GUI works in degC and converts in `gui/units.py`
(`src/constants.py`, `src/units.py`).  Conduction and convection only involve
temperature *differences*, but radiation, exergy and the stored energy reference
do not, which is why the convention is enforced (`check_kelvin`).

## 1.3 Mechanisms

**Conduction** (Fourier): $q'' = -k \nabla T$.

**Convection** (Newton): $q'' = h (T_s - T_\infty)$, $h$ in W/(m²·K).

| Situation | typical $h$ |
|---|---|
| natural convection, air | 5–25 |
| forced convection, air | 25–250 |
| natural convection, water | 100–900 |
| forced convection, water | 50–20 000 |

**Radiation** (Stefan–Boltzmann): $q'' = \varepsilon \sigma (T_s^4 - T_{surr}^4)$
with $\sigma = 5.670\cdot10^{-8}$ W/(m²·K⁴).

The solver treats radiation **optionally and linearised** (off by default):
$h_r = \varepsilon \sigma (T_s + T_\infty)(T_s^2 + T_\infty^2)$ so that
$q'' \approx h_r (T_s - T_\infty)$ (`src/core/physics.py::radiation_h`).  In the
steady solver $h_r$ is refreshed in a Picard sweep
(`src/solver/steady.py`, `SolverConfig.radiation`, `max_picard`); in the
transient the operators are rebuilt at every step when radiation is on.  At the
temperatures this unit reaches (500–600 °C on the shell) radiation is of the same
order as the convective loss, so leaving it off is a deliberate, visible choice -
the GUI exposes it in *Materials → Conditions* and *Tools → Solver*.

## 1.4 Thermal resistances

Series resistances add: $q = \Delta T / R_{tot}$ with

$$R_{cond,wall} = \frac{L}{kA}, \qquad
R_{cond,cyl} = \frac{\ln(r_2/r_1)}{2\pi k L}, \qquad
R_{conv} = \frac{1}{hA}.$$

The discretisation uses exactly this: the conductance of a face between two
cells is the series of two half cells, $k_{face} = 2 k_1 k_2/(k_1+k_2)$, and a
convective surface adds the film in series with the half cell it sits on
(`half_cell_h = 2kh/(2k+hd)`, see [02](02_FDM_DISCRETIZATION.md) §5).

## 1.5 Porous storage medium

The storage region is a packed bed: solid particles with air in the voids.
With packing fraction $\phi_x$ (solid fraction) and porosity $\phi = 1-\phi_x$,
`src/core/materials.py::MaterialManager.compute_effective_properties` uses

$$k_{eff} = k_{solid}^{1-\phi}\, k_{fluid}^{\phi}, \qquad
\rho_{eff} = (1-\phi)\rho_s + \phi \rho_f, \qquad
c_{p,eff} = \frac{(1-\phi)\rho_s c_{p,s} + \phi \rho_f c_{p,f}}{\rho_{eff}}$$

(the geometric mean is the standard interpolation for a random two-phase
medium; the density is arithmetic and the heat capacity is mass-weighted, so
$\rho_{eff} c_{p,eff}$ is the arithmetic mean of the volumetric capacities).
The default packing fraction is 0.63; the GUI allows 0.20–0.90.

Properties are **constant with temperature**: the previous temperature-dependent
hooks were never called by any solver and were removed rather than pretending.
Over a 20–600 °C operating range a real $k(T)$ can vary by tens of percent, so
this is an assumption to state in a report, not a hidden one.

## 1.6 Boundary conditions

* **Dirichlet** (prescribed temperature): $T|_\Gamma = T_{prescribed}$; used for
  the ground under the foundation.
* **Neumann** (prescribed flux): $-k\,\partial T/\partial n|_\Gamma = q''$;
  $q''=0$ is the adiabatic (symmetry) case.
* **Robin** (convection/radiation): $-k\,\partial T/\partial n|_\Gamma = h(T_s-T_\infty)$;
  used on every air-exposed face.

Each of the six domain faces carries its own condition
(`Mesh3D.face_bc`, `FaceBC`), so a face can be convective while its neighbour is
adiabatic and a corner node receives one contribution per exposed face.

## 1.7 Dimensionless numbers

$$Bi = \frac{h L_c}{k}, \qquad Fo = \frac{\alpha t}{L_c^2}, \qquad
\alpha = \frac{k}{\rho c_p}$$

With $Bi \ll 1$ the solid is nearly isothermal (lumped capacitance); $Fo$
measures how far the transient is from equilibrium.  Both are useful to sanity
check a result before trusting it.

## 1.8 Energy bookkeeping used by the code

Charging: $P_{in} = \dfrac{dE_{stored}}{dt} + P_{extracted} + P_{losses}$;
discharging: $-\dfrac{dE_{stored}}{dt} = P_{extracted} + P_{losses} - P_{in}$.

The implementations are:

$$E_{stored} = \sum_{cells} \rho c_p (T - T_0) V, \qquad
P_{in} = \sum_{cells} Q_{source} V, \qquad
P_{extracted} = -\sum_{cells} Q_{sink} V + P_{tube\ fluid}$$

with $T_0$ the ambient temperature.  Losses are the surface integrals of §5 of
[02](02_FDM_DISCRETIZATION.md), evaluated on the battery envelope (insulation,
shell, foundation against air) *and* on the six box faces for auditing
(`src/analysis/fluxes.py`).  The identity
$P_{in} - P_{extracted} - P_{losses} - dE/dt \approx 0$ is the self-check that
`Balance.imbalance` reports, and `tests/test_analysis.py` asserts it.

**Exergy.**  Heat available at temperature $T$: $\dot{Ex} = \dot Q (1 - T_0/T)$;
stored exergy

$$Ex_{stored} = \sum_{cells} \rho c_p
\left[(T-T_0) - T_0 \ln\frac{T}{T_0}\right] V$$

and destroyed exergy is the difference between the exergy entering with the
heaters (Carnot factor evaluated at the storage temperature) and what is stored
(`src/analysis/balance.py`, `fluxes.destroyed_exergy`).

## 1.9 Extracting heat

The tube cells exchange $h_{fluid}(T_{wall} - T_{fluid})$ with the fluid.  In
*flow-rate* mode the fluid temperature is the inlet temperature and the removed
power is whatever that exchange produces; in *target-power* mode a volumetric
sink is imposed on the tube cells, **capped** by the available
$h A (T_{tube} - T_{inlet})$ - so the model can never extract heat from a body
colder than the inlet (`src/solver/transient.py`).

## 1.10 References

1. Incropera, DeWitt, Bergman, Lavine - *Fundamentals of Heat and Mass Transfer*
2. Çengel - *Heat Transfer: A Practical Approach*
3. Bejan - *Advanced Engineering Thermodynamics* (exergy)
4. Kaviany - *Principles of Heat Transfer in Porous Media*
