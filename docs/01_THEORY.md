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

The storage region is a packed bed: solid grains with air in the voids.  With packing
fraction $\phi_x$ (solid fraction) and porosity $\psi = 1-\phi_x$ the capacity is the
volume mean,

$$\rho_{eff} = (1-\psi)\rho_s + \psi \rho_f, \qquad
c_{p,eff} = \frac{(1-\psi)\rho_s c_{p,s} + \psi \rho_f c_{p,f}}{\rho_{eff}},$$

and the conductivity is the **Zehner-Bauer-Schlünder** model with the radiation between
the grains (VDI Heat Atlas D6.3 [VDI-D6.3]; Zehner and Schlünder [ZS70]; Breitbach and
Barthels [BB80]; `src/core/materials.py::PackedBed`).  Three paths in parallel and in
series: the gas in the voids, the conduction through the grains and their contacts, and
the radiation across the voids, whose conductivity grows as $4\sigma T^3 d$:

$$k_{rad} = \frac{4 \sigma T^3 d}{(2/\varepsilon - 1)\, k_f}, \qquad
\frac{k_{bed}}{k_f} = (1-\sqrt{1-\psi})(1 + \psi k_{rad})
+ \sqrt{1-\psi}\,\big(\varphi \kappa + (1-\varphi) k_c(\kappa, k_{rad}, B)\big)$$

($\kappa = k_s/k_f$, $B = C_f ((1-\psi)/\psi)^{10/9}$, $k_c$ the unit-cell core term;
the full expressions are in [18](18_SOLVER.md) §3).  For steatite at 63 % packing and 1 mm
grains the bed conducts **0.30 W/(m K) at 20 °C, 0.57 at 500 °C and 0.68 at 700 °C**:
the conductivity is a law of the temperature, and the solvers evaluate it on the field.
The default packing fraction is 0.63 (the GUI allows 0.20–0.90) and the grain 1 mm.
The constant geometric mean $k_s^{1-\psi} k_f^{\psi}$ used before stays available as the
`"geometric"` model; it gave 0.52 W/(m K) at every temperature.

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

The same Robin law also applies to an **internal** surface: when the air around the
vessel is excluded from the problem, the outer surface of the insulation/shell is where
the film acts, with `h_out` the sum of the natural and wind shares
(`src/core/environment.py`, [13](13_REDESIGN.md) §4).  A third resistance can sit
*between* two materials: a contact conductance `mesh.h_contact` in series with the two
half cells.

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

That lumped model puts the *inlet* temperature everywhere.  The redesigned heat path is
a **1-D network** whose gas temperature falls along the run
(`src/solver/fluid.py`), which for a storage that discharges at large NTU is the
difference between a plausible and an overstated extraction
([13](13_REDESIGN.md) §3, [15](15_PIPE_NETWORKS.md)).

## 1.10 Heat exchanged with a flowing gas

For one pipe segment with wetted area $A$, wall temperature $T_w$ and mass flow
$\dot m$:

$$NTU = \frac{h A}{\dot m c_p}, \qquad
T_{out} = T_w + (T_{in}-T_w)e^{-NTU}, \qquad
T_{mean} = T_w + (T_{in}-T_w)\frac{1-e^{-NTU}}{NTU}$$

so the power delivered to the solid is $q = \dot m c_p (T_{in} - T_{out})$, and the two
limits are the *area-limited* regime ($NTU \ll 1$, $q \approx hA\,\Delta T$) and the
*flow-limited* one ($NTU \gg 1$, $q \approx \dot m c_p \Delta T$).  A storage discharges
in the second regime by design.

With the tube buried in a cell of the bed, $UA$ is not $hA$: between the wall and the
cell centre lies the bed itself.  The well model of Peaceman [Pea78, Pea83] gives the
radius at which the discrete cell temperature equals the radial solution,
$r_{eq} = 0.14\sqrt{a^2 + b^2}$ for a cell of cross-section $a \times b$, so

$$\frac{1}{UA'} = \frac{1}{h \pi d_i} + \frac{\ln(r_o/r_i)}{2\pi k_{wall}}
+ \frac{\ln(r_{eq}/r_o)}{2\pi k_{bed}}$$

per unit length ([18](18_SOLVER.md) §6).  The gas properties are those of the gas at
its own temperature along the loop (Incropera Table A.4).

The film coefficient follows the usual correlations for internal flow (laminar
$Nu = 3.66$, turbulent Dittus-Boelter, blended in between, `solver/fluid.py::pipe_h`),
the friction factor comes from the classical correlations, and the **fan work** is

$$P_{fan} = \frac{\dot m}{\rho}\Delta p \qquad\text{(per unit flow, the circulation
loss of the loop)}$$

which the cycle accounting reports as `E_circulation` next to the standby loss - the two
terms that decide the round-trip efficiency of a real machine.

## 1.11 References

The keys are those of [17_REFERENCES.md](17_REFERENCES.md): [Incropera] for the
properties, the shape factors and the film correlations; [VDI-D6.3], [ZS70] and [BB80]
for the packed bed; [Pea78], [Pea83] for the tube in its cell; [Kays] for the
effectiveness relation; [CC75] and [ISO6946] for the outer film.  Background:
Çengel, *Heat Transfer: A Practical Approach*; Bejan, *Advanced Engineering
Thermodynamics* (exergy); Kaviany, *Principles of Heat Transfer in Porous Media*.
