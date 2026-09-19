"""The heat-transfer fluid as a 1-D network: the heart of the sand-battery model.

Polar Night Energy's published architecture is a **closed air loop through pipes
buried in the sand**: the electric resistors heat the air (not the bed), the air
charges the bed through the pipe walls, and on discharge the same loop delivers the
heat to an exchanger.  The fluid is therefore not a boundary condition with a fixed
temperature: its temperature evolves along the path, and every cell of the bed sees
the *local* fluid temperature.

The model is the classical 1-D advection with distributed wall exchange,

    m_dot c_p dT/ds = h P (T_wall(s) - T(s))

whose exact solution over a piece of pipe of wetted area ``A`` is the effectiveness
relation

    T_out = T_wall + (T_in - T_wall) exp(-NTU),      NTU = h A / (m_dot c_p)

with the mean fluid temperature over that piece

    T_mean = T_wall + (T_in - T_wall) (1 - exp(-NTU)) / NTU

and the power exchanged ``m_dot c_p (T_in - T_out)``.  Two limits come out for free and
they are the two regimes of a storage:

* ``NTU -> 0``  (high flow, small area): ``Q -> h A (T_wall - T_in)``, **area-limited**;
* ``NTU -> inf`` (low flow, large area): ``Q -> m_dot c_p (T_wall - T_in)``, **flow-limited**.

A storage discharges in the second regime (that is what makes it a storage), which is
exactly the regime a lumped film model with a fixed inlet temperature gets wrong by a
factor ``NTU / (1 - exp(-NTU))``.

Every temperature is affine in the loop inlet temperature, so the marching solution
propagates ``T(s) = a T_in + b`` and the loop energy balance

    sum_i m_dot_i c_p (T_out,i - T_in) = Q_ext

is solved exactly for ``T_in`` in one step.  ``Q_ext > 0`` are the resistors
(charging), ``Q_ext < 0`` is the exchanger (discharging): the same loop does both.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import Mesh3D
from ..core.pipes import PipeRun


@dataclass(frozen=True)
class Fluid:
    """Heat-transfer fluid properties (defaults: air at 300 K)."""

    name: str = "air"
    cp: float = 1005.0          # [J/(kg K)]
    rho: float = 1.2            # [kg/m^3]
    mu: float = 1.8e-5          # [Pa s]
    k: float = 0.026            # [W/(m K)]
    pr: float = 0.71            # [-]
    molar_mass: float = 0.02896  # [kg/mol] air; N2 0.02801, steam 0.01802

    def at_pressure(self, pressure: float, temperature: float) -> Fluid:
        """The same fluid at ``pressure`` [Pa] and ``temperature`` [K] (ideal gas).

        Pressure is the cheap design lever of a gas loop: at a fixed mass flow the
        velocity falls as ``1/rho`` and the Reynolds number rises as ``rho``, so the
        film coefficient grows and the pressure drop *falls* - the reason the published
        designs go to tens of bar instead of pushing the velocity.
        """
        base = self.at(temperature)
        rho = base.rho * pressure / 101325.0        # the reference is 1 atm
        return Fluid(name=self.name, cp=base.cp, rho=rho, mu=base.mu, k=base.k,
                     pr=base.pr, molar_mass=self.molar_mass)

    def at(self, temperature: float) -> Fluid:
        """The same fluid at another temperature (Sutherland + power laws).

        The loop of a sand battery spans 20-600 C, so the properties cannot be held
        constant: density falls as 1/T, viscosity rises, conductivity rises.
        """
        t = max(float(temperature), 100.0)
        t_ref, sutherland = 300.0, 110.4
        return Fluid(name=self.name, cp=self.cp, rho=self.rho * t_ref / t,
                     mu=self.mu * (t / t_ref) ** 1.5 * (t_ref + sutherland)
                     / (t + sutherland),
                     k=self.k * (t / t_ref) ** 0.8, pr=self.pr)


def pipe_h(mass_flow: float, diameter: float, fluid: Fluid) -> float:
    """Film coefficient of the fluid inside a pipe [W/(m^2 K)].

    Laminar (Re < 2300): ``Nu = 3.66`` (constant wall temperature, fully developed).
    Turbulent: Dittus-Boelter ``Nu = 0.023 Re^0.8 Pr^0.4``.  In between the two are
    blended, so a design sweep does not step.
    """
    if mass_flow <= 0 or diameter <= 0:
        return 0.0
    area = 0.25 * np.pi * diameter ** 2
    velocity = mass_flow / (fluid.rho * area)
    re = fluid.rho * velocity * diameter / fluid.mu
    if re <= 0:
        return 0.0
    nu_lam = 3.66
    nu_turb = 0.023 * re ** 0.8 * fluid.pr ** 0.4
    if re < 2300.0:
        nu = nu_lam
    elif re < 4000.0:
        weight = (re - 2300.0) / 1700.0
        nu = (1.0 - weight) * nu_lam + weight * nu_turb
    else:
        nu = nu_turb
    return float(nu * fluid.k / diameter)


def friction_factor(reynolds: float, relative_roughness: float = 0.0) -> float:
    """Darcy friction factor: laminar ``64/Re``, turbulent Haaland (explicit Colebrook).

    Haaland is within 1.5% of Colebrook over the whole turbulent range and needs no
    iteration, which keeps the whole loop solvable in closed form.
    """
    if reynolds <= 0:
        return 0.0
    if reynolds < 2300.0:
        return 64.0 / reynolds
    return float(1.0 / (-1.8 * np.log10((relative_roughness / 3.7) ** 1.11
                                        + 6.9 / reynolds)) ** 2)


def pressure_drop(mass_flow: float, diameter: float, length: float, fluid: Fluid,
                  roughness: float = 4.5e-5, fittings_k: float = 0.0) -> float:
    """Pressure drop of one pipe run [Pa]: Darcy-Weisbach plus local losses.

    ``roughness`` defaults to 45 um (stainless steel); ``fittings_k`` sums the local
    loss coefficients of the bends, headers and valves of that run.
    """
    if mass_flow <= 0 or diameter <= 0 or length <= 0:
        return 0.0
    area = 0.25 * np.pi * diameter ** 2
    velocity = mass_flow / (fluid.rho * area)
    re = fluid.rho * velocity * diameter / fluid.mu
    factor = friction_factor(re, roughness / diameter)
    return float((factor * length / diameter + fittings_k)
                 * 0.5 * fluid.rho * velocity ** 2)


def fan_power(mass_flow: float, delta_p: float, fluid: Fluid,
              efficiency: float = 0.7) -> float:
    """Shaft power of the blower [W]: ``V_dot dp / eta``."""
    if mass_flow <= 0 or delta_p <= 0 or efficiency <= 0:
        return 0.0
    return float(mass_flow / fluid.rho * delta_p / efficiency)


@dataclass
class RunResult:
    """What one pipe run did."""

    name: str
    cells: np.ndarray
    t_fluid: np.ndarray          # [K] mean fluid temperature inside each cell
    q_solid: np.ndarray          # [W] power into the solid per cell (+ = charging)
    t_out: float = float("nan")
    ntu: float = 0.0
    power: float = 0.0           # [W] into the solid (+ = the bed is being charged)
    mass_flow: float = 0.0


@dataclass
class FluidResult:
    """Outcome of the loop: per-cell fields and the loop temperatures."""

    runs: list[RunResult] = field(default_factory=list)
    t_in: float = float("nan")
    t_out: float = float("nan")
    power: float = 0.0                  # [W] into the solid
    external_power: float = 0.0         # [W] from the resistors (+) / to the exchanger (-)
    q_fluid: np.ndarray | None = None   # [W/m^3] source term for the solid, + = in
    delta_p: float = 0.0                # [Pa] pressure drop of the circuit
    fan_power: float = 0.0              # [W] shaft power of the blower

    @property
    def circulation_loss(self) -> float:
        """Fan power as a fraction of the power exchanged with the bed [-]."""
        if abs(self.power) <= 0.0:
            return 0.0
        return self.fan_power / abs(self.power)

    @property
    def ntu(self) -> float:
        return float(np.mean([run.ntu for run in self.runs])) if self.runs else 0.0

    def summary(self) -> str:
        return (f"loop: T_in {self.t_in:.1f} K -> T_out {self.t_out:.1f} K, "
                f"NTU {self.ntu:.2f}, bed {self.power / 1000:+.2f} kW, "
                f"external {self.external_power / 1000:+.2f} kW, "
                f"dp {self.delta_p / 1000:.2f} kPa, fan {self.fan_power / 1000:.2f} kW "
                f"({100 * self.circulation_loss:.1f}% of the bed power)")


@dataclass
class FluidLoop:
    """A closed loop of pipe runs with a total mass flow and an external power."""

    runs: Sequence[PipeRun]
    mass_flow: float                       # [kg/s] total
    fluid: Fluid = Fluid()
    h_fluid: float | None = None           # [W/(m^2 K)]; None = computed from the flow
    external_power: float = 0.0            # [W] + resistors, - exchanger
    t_in: float | None = None              # [K] prescribed inlet (open loop)
    split: Sequence[float] | None = None   # fraction of the flow per run
    #: absolute roughness of the pipe wall [m] (45 um: stainless steel)
    roughness: float = 4.5e-5
    #: local loss coefficients of the fittings of the *whole* circuit [-]
    fittings_k: float = 0.0
    #: blower efficiency (electric power = shaft power / efficiency)
    fan_efficiency: float = 0.7

    # ------------------------------------------------------------------ helpers
    def _flow_split(self) -> np.ndarray:
        if not self.runs:
            raise ValueError("the loop has no pipe runs")
        if self.split is None:
            return np.full(len(self.runs), 1.0 / len(self.runs))
        split = np.asarray(self.split, dtype=float)
        if split.size != len(self.runs) or split.sum() <= 0:
            raise ValueError("split must give one positive fraction per run")
        return split / split.sum()

    def _h(self, mass_flow_run: float) -> float:
        if self.h_fluid is not None:
            return float(self.h_fluid)
        diameters = [run.diameter for run in self.runs if run.diameter > 0]
        diameter = float(np.mean(diameters)) if diameters else 0.05
        return pipe_h(mass_flow_run, diameter, self.fluid)

    # -------------------------------------------------------------------- solve
    def solve(self, mesh: Mesh3D, wall: np.ndarray | None = None) -> FluidResult:
        """March the loop once; returns the fields and the loop temperatures.

        ``wall`` is the solid temperature field [K] (defaults to ``mesh.T``).  The
        coupling with the solid is explicit: solve the solid, march the fluid, solve
        the solid again.  Each cell stores the affine coefficients of its inlet,
        outlet and mean fluid temperature, so the loop balance and the per-cell power
        are evaluated in one vectorised pass after the march.
        """
        wall_flat = np.asarray(mesh.T if wall is None else wall).ravel(order="F")
        volume = mesh.V.ravel(order="F")
        split = self._flow_split()
        result = FluidResult(external_power=float(self.external_power))

        marches = []
        slope = offset = 0.0
        for run, fraction in zip(self.runs, split, strict=True):
            m_dot = float(self.mass_flow * fraction)
            h = self._h(m_dot)
            if m_dot <= 0 or h <= 0 or run.cells.size == 0:
                marches.append(None)
                continue
            mc = m_dot * self.fluid.cp
            count = run.cells.size
            # columns: in_a, in_b, out_a, out_b, mean_a, mean_b
            coeff = np.empty((count, 6), dtype=float)
            a, b = 1.0, 0.0
            for index in range(count):
                t_wall = float(wall_flat[run.cells[index]])
                ntu = h * float(run.area[index]) / mc
                decay = float(np.exp(-ntu))
                mean_factor = (1.0 - decay) / ntu if ntu > 0.0 else 1.0
                coeff[index] = (a, b, decay * a, t_wall + (b - t_wall) * decay,
                                a * mean_factor, t_wall + (b - t_wall) * mean_factor)
                a, b = decay * a, t_wall + (b - t_wall) * decay
            marches.append((m_dot, h, coeff, a, b))
            slope += mc * (a - 1.0)
            offset += mc * b

        if self.t_in is not None:
            t_in = float(self.t_in)
        elif not marches or all(entry is None for entry in marches):
            t_in = float("nan")
        else:
            # the loop is steady: what the bed gives the air leaves through the
            # external device, so sum_i m c (T_out,i - T_in) = -Q_ext
            if slope >= 0.0:
                raise ValueError("the loop balance has no solution: check the flow")
            t_in = (-self.external_power - offset) / slope
            if t_in <= 0.0:
                raise ValueError(
                    f"the loop cannot carry {self.external_power / 1000:+.1f} kW at "
                    f"{self.mass_flow:.3f} kg/s through a bed at "
                    f"{np.mean(wall_flat):.0f} K: the air would have to be at "
                    f"{t_in:.0f} K.  Increase the mass flow or reduce the power")
        result.t_in = t_in

        q_fluid = np.zeros(wall_flat.shape, dtype=float)
        total_power = 0.0
        weighted_out = 0.0
        weight = 0.0
        for run, entry in zip(self.runs, marches, strict=True):
            if entry is None:
                result.runs.append(RunResult(name=run.name, cells=run.cells,
                                             t_fluid=np.empty(0), q_solid=np.empty(0)))
                continue
            m_dot, h, coeff, a, b = entry
            t_in_cell = coeff[:, 0] * t_in + coeff[:, 1]
            t_out_cell = coeff[:, 2] * t_in + coeff[:, 3]
            t_mean = coeff[:, 4] * t_in + coeff[:, 5]
            q_cell = m_dot * self.fluid.cp * (t_in_cell - t_out_cell)     # [W]
            q_fluid[run.cells] += q_cell / volume[run.cells]              # [W/m^3]
            run_power = float(np.sum(q_cell))
            total_power += run_power
            result.runs.append(RunResult(
                name=run.name, cells=run.cells, t_fluid=t_mean, q_solid=q_cell,
                t_out=float(a * t_in + b), mass_flow=m_dot, power=run_power,
                ntu=float(np.sum(h * run.area / (m_dot * self.fluid.cp)))))
            weighted_out += m_dot * float(a * t_in + b)
            weight += m_dot

        # hydraulics: the pressure drop of the circuit and the blower power
        delta_p = 0.0
        for run in self.runs:
            delta_p += pressure_drop(self.mass_flow, run.diameter, run.total_length,
                                     self.fluid, self.roughness, self.fittings_k)
        result.delta_p = delta_p
        result.fan_power = fan_power(self.mass_flow, delta_p, self.fluid,
                                     self.fan_efficiency)
        result.q_fluid = q_fluid
        result.power = total_power
        result.t_out = weighted_out / weight if weight > 0 else float("nan")
        return result
