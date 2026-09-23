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

**The circuit as a network.**  A plant is not a set of parallel tubes: the gas crosses
the inlet duct, the distributor, one riser, the collector and the outlet duct, and the
headers are shared - their flow falls tap by tap along the distributor and grows along
the collector.  :class:`GasGraph` is that circuit: straight segments between nodes, each
with its own share of the flow, marched in topological order with the enthalpy of the
streams mixed at every node.  Temperatures stay affine in ``T_in``, so the balance is
still solved in one step, and the headers exchange with the bed like the risers.  The
balance is imposed on the exchange itself, ``sum_cells q = Q_ext``, so the energy the
bed receives is the energy the external device gives whatever the gas properties do.

**Well model.**  A tube is a line inside a cell of the bed, so the cell temperature is
not the wall temperature: for a line source in a square cell of edge ``h`` the discrete
solution equals the radial one at the *equivalent radius* ``r_eq = 0.198 h``
(Peaceman, SPE J. 18 (1978) 183-194; ``0.14 sqrt(a^2 + b^2)`` for an ``a x b`` cell,
SPE J. 23 (1983) 531-543).  The wall-to-cell resistance per unit length is therefore
``ln(r_eq / r_o) / (2 pi k)``, in series with the tube wall ``ln(r_o / r_i) /
(2 pi k_wall)`` and the gas film ``1 / (h pi d_i)``:

    1 / UA' = 1 / (h pi d_i) + ln(r_o / r_i) / (2 pi k_wall) + ln(r_eq / r_o) / (2 pi k)

This is what makes the exchange independent of the mesh: without it the bed between
the wall and the cell centre is a spurious resistance that grows as the cell shrinks
towards the tube (on 52 mm leaves around a 50 mm tube it was as large as the gas film).
The model needs ``r_eq > r_o`` (cells at least ``2.53 d`` wide); a smaller cell gets no
correction (the resistance is then overstated) and the result says how much surface
was in that case.  The correction is quasi-steady: it holds once the thermal wave has
crossed the cell, ``t > r_eq^2 / alpha`` (about an hour for 40 mm in sand).

**Gas properties.**  Air between 20 and 700 degC changes its cp by 12 %, its viscosity
by a factor 2.2 and its conductivity by 2.4; with ``variable_properties`` every segment
is marched with the properties of the gas at its own mean temperature (Incropera et
al., *Fundamentals of Heat and Mass Transfer*, 7th ed., Table A.4), lagged by one march:
the previous march - or a first pass at the wall temperature - gives the temperatures.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np

from ..core.mesh import BoundaryType, MaterialID, Mesh3D
from ..core.pipes import PipeRun


#: air at 1 atm (Incropera et al., 7th ed., Table A.4): T [K], cp [J/(kg K)],
#: mu [1e-7 Pa s], k [1e-3 W/(m K)], Pr [-]
_AIR_TABLE = np.array([
    (250.0, 1006.0, 159.6, 22.3, 0.720), (300.0, 1007.0, 184.6, 26.3, 0.707),
    (350.0, 1009.0, 208.2, 30.0, 0.700), (400.0, 1014.0, 230.1, 33.8, 0.690),
    (450.0, 1021.0, 250.7, 37.3, 0.686), (500.0, 1030.0, 270.1, 40.7, 0.684),
    (550.0, 1040.0, 288.4, 43.9, 0.683), (600.0, 1051.0, 305.8, 46.9, 0.685),
    (650.0, 1063.0, 322.5, 49.7, 0.690), (700.0, 1075.0, 338.8, 52.4, 0.695),
    (750.0, 1087.0, 354.6, 54.9, 0.702), (800.0, 1099.0, 369.8, 57.3, 0.709),
    (850.0, 1110.0, 384.3, 59.6, 0.716), (900.0, 1121.0, 398.1, 62.0, 0.720),
    (950.0, 1131.0, 411.3, 64.3, 0.723), (1000.0, 1141.0, 424.4, 66.7, 0.726),
    (1100.0, 1159.0, 449.0, 71.5, 0.728), (1200.0, 1175.0, 473.0, 76.3, 0.728),
])
#: water vapour at 1 atm (Incropera et al., Table A.6, superheated)
_STEAM_TABLE = np.array([
    (380.0, 2060.0, 127.1, 24.6, 1.06), (400.0, 2014.0, 134.4, 26.1, 1.04),
    (450.0, 1980.0, 152.5, 29.9, 1.01), (500.0, 1985.0, 170.4, 33.9, 1.00),
    (550.0, 1997.0, 188.4, 37.9, 0.99), (600.0, 2026.0, 206.7, 42.2, 0.99),
    (650.0, 2056.0, 224.7, 46.4, 0.99), (700.0, 2085.0, 242.6, 50.5, 1.00),
    (750.0, 2119.0, 260.4, 54.9, 1.00), (800.0, 2152.0, 278.6, 59.2, 1.01),
    (850.0, 2186.0, 296.9, 63.7, 1.02),
])


def property_shape(name: str, temperature: float) -> np.ndarray | None:
    """``(cp, mu, k, pr)`` at ``temperature`` [K] over the same at 300 K.

    From the tables of Incropera et al.; nitrogen follows the air table (a diatomic gas
    of nearly the same molar mass and temperature dependence).  A fluid without a table
    returns ``None`` and keeps its power laws.  Outside the table the end value holds.
    """
    table = {"air": _AIR_TABLE, "nitrogen": _AIR_TABLE, "steam": _STEAM_TABLE}.get(name)
    if table is None:
        return None
    here = np.array([np.interp(float(temperature), table[:, 0], table[:, c])
                     for c in range(1, 5)])
    ref = np.array([np.interp(300.0, table[:, 0], table[:, c]) for c in range(1, 5)])
    return here / ref


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
        """The same fluid at another temperature.

        The loop of a sand battery spans 20-700 C, so the properties cannot be held
        constant: density falls as 1/T (ideal gas), and cp, viscosity, conductivity and
        Prandtl follow the property table of the gas (:func:`property_shape`), scaled so
        that the fluid's own values hold at 300 K.  A gas without a table keeps cp and
        takes Sutherland's viscosity and ``k ~ T^0.8``.
        """
        t = max(float(temperature), 100.0)
        shape = property_shape(self.name, t)
        if shape is not None:
            return Fluid(name=self.name, cp=self.cp * shape[0], rho=self.rho * 300.0 / t,
                         mu=self.mu * shape[1], k=self.k * shape[2], pr=self.pr * shape[3],
                         molar_mass=self.molar_mass)
        t_ref, sutherland = 300.0, 110.4
        return Fluid(name=self.name, cp=self.cp, rho=self.rho * t_ref / t,
                     mu=self.mu * (t / t_ref) ** 1.5 * (t_ref + sutherland)
                     / (t + sutherland),
                     k=self.k * (t / t_ref) ** 0.8, pr=self.pr,
                     molar_mass=self.molar_mass)


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
    #: per cell, the exchange as the solid solver takes it: ``q = G (T_gas - T_wall)``
    #: with ``G = m_dot c_p (1 - e^-NTU)`` [W/K] summed over the runs that cross the cell
    #: and ``T_gas`` the conductance-weighted temperature the gas *enters* the cell at
    conductance: np.ndarray | None = None
    t_gas: np.ndarray | None = None
    #: ``d T_gas / d T_in`` per cell (conductance-weighted): how the gas entering a cell
    #: follows the loop inlet, which is what lets a solver hold the loop balance exactly
    gas_slope: np.ndarray | None = None
    #: True when ``t_in`` was solved from the loop balance (not prescribed)
    balanced: bool = False
    delta_p: float = 0.0                # [Pa] pressure drop of the circuit
    fan_power: float = 0.0              # [W] shaft power of the blower
    pressure: float = 101325.0          # [Pa] loop pressure
    notes: list[str] = field(default_factory=list)

    @property
    def circulation_loss(self) -> float:
        """Fan power over the power exchanged with the bed [-].

        The published circulation figure of a storage cycle (~5%) is an *aggregate*:
        blower, controls and the heating of the hot ducts.  The blower alone is quoted
        at 1-2% of the plant power in the packed-bed literature, so both denominators
        are reported: this one (thermal) and ``circulation_loss_electric``.
        """
        if abs(self.power) <= 0.0:
            return 0.0
        return self.fan_power / abs(self.power)

    @property
    def circulation_loss_electric(self) -> float:
        """Fan power over the external power of the loop (the electricity in)."""
        if abs(self.external_power) <= 0.0:
            return 0.0
        return self.fan_power / abs(self.external_power)

    @property
    def ntu(self) -> float:
        return float(np.mean([run.ntu for run in self.runs])) if self.runs else 0.0

    def apply(self, mesh) -> np.ndarray:
        """Write the exchange onto the mesh as an implicit film on the pipe cells.

        The march gives every crossed cell the exact exchange
        ``q = G (T_gas - T_wall)``: ``T_gas`` depends on the cells *upstream*, the local
        term on the cell's own wall.  Handing the solid solver ``G`` and ``T_gas`` as a
        convective film - ``bc_h = G V^(1/3) / V`` so that the assembled coefficient is
        ``G / V`` - keeps the local exchange implicit in the wall temperature, which is
        what makes the coupling stable at any time step; depositing the marched ``q`` as
        a fixed source (the explicit coupling it replaces) let a cell with a large
        ``G`` over its heat capacity overshoot and oscillate, down to below 100 K on a
        long step.  At a converged coupling (the steady Picard loop) the two are the same
        number.

        The pipe cells the march does not cross (the headers) keep no film, and the lumped
        bed source is cleared: with the loop, the gas is the only heat path.  Returns the
        mask of the cells that carry the film.
        """
        if self.conductance is None:
            raise ValueError("the loop result carries no per-cell exchange")
        shape = np.shape(mesh.T)
        volume = np.asarray(mesh.V, dtype=float).ravel(order="F")
        h_char = np.asarray(mesh.h_char, dtype=float).ravel(order="F")
        bc_h = np.asarray(mesh.bc_h, dtype=float).ravel(order="F").copy()
        bc_t = np.asarray(mesh.bc_T_inf, dtype=float).ravel(order="F").copy()
        kind = np.asarray(mesh.boundary_type).ravel(order="F").copy()
        tube = np.asarray(mesh.material_id).ravel(order="F") == int(MaterialID.TUBES)
        active = self.conductance > 0.0
        bc_h[tube] = 0.0
        bc_h[active] = self.conductance[active] * h_char[active] / volume[active]
        bc_t[active] = self.t_gas[active]
        kind[active] = int(BoundaryType.CONVECTION)
        mesh.bc_h = bc_h.reshape(shape, order="F")
        mesh.bc_T_inf = bc_t.reshape(shape, order="F")
        mesh.boundary_type = kind.reshape(shape, order="F").astype(
            np.asarray(mesh.boundary_type).dtype)
        mesh.Q_source = np.zeros(shape)
        mesh.Q_sink = np.zeros(shape)
        return active.reshape(shape, order="F")

    def summary(self) -> str:
        return (f"loop: T_in {self.t_in:.1f} K -> T_out {self.t_out:.1f} K, "
                f"NTU {self.ntu:.2f}, bed {self.power / 1000:+.2f} kW, "
                f"external {self.external_power / 1000:+.2f} kW, "
                f"dp {self.delta_p / 1000:.2f} kPa, fan {self.fan_power / 1000:.2f} kW "
                f"({100 * self.circulation_loss:.1f}% of the bed power)")


def hold_loop_balance(result: FluidResult, mesh, x: np.ndarray,
                      solve: Callable[[np.ndarray], np.ndarray],
                      pinned: np.ndarray, memo: dict | None = None,
                      operator: object = None) -> np.ndarray:
    """The field that makes the implicit film deposit exactly the loop's external power.

    The march fixes the gas entering every cell from the walls it was marched on, so the
    implicit film alone deposits ``sum G (T_gas - T_new)``, which drifts from the power
    the resistors put in as the walls move.  The loop inlet is therefore solved *with*
    the field: every ``T_gas`` moves with ``T_in`` by its slope ``alpha``, the field by
    ``delta * y`` with ``A y = G alpha / V``, and the one scalar ``delta`` that makes the
    deposit equal the external power closes the loop's enthalpy balance on the new field
    - a Schur complement of one row, at the cost of one more solve with the same
    operator (``solve``).  ``x`` is the field the film alone gave, flat; ``pinned`` the
    cells whose rows the elimination fixed.  ``result`` and the mesh's film temperatures
    are updated to the new inlet; the new field is returned.  A prescribed inlet
    (``result.balanced`` False) leaves everything as it is.

    ``memo`` (a dict the caller keeps) stores ``y`` for ``operator``: ``G`` and ``alpha``
    do not depend on the wall temperatures, so with the same operator the second solve
    is the same solve, and a transient or a Picard loop pays it once.
    """
    if not result.balanced or result.conductance is None:
        return x
    g = result.conductance
    cells = (g > 0.0) & ~np.asarray(pinned, dtype=bool)
    if not cells.any():
        return x
    volume = np.asarray(mesh.V, dtype=float).ravel(order="F")
    alpha = result.gas_slope
    e = np.zeros(g.size)
    e[cells] = g[cells] * alpha[cells] / volume[cells]
    key = hash(e.tobytes())
    if memo is not None and memo.get("operator") is operator and memo.get("key") == key:
        y = memo["y"]
    else:
        y = np.asarray(solve(e), dtype=float)
        if memo is not None:
            memo.update(operator=operator, key=key, y=y)
    x = np.asarray(x, dtype=float)
    deposit = float(np.sum(g[cells] * (result.t_gas[cells] - x[cells])))
    slope = float(np.sum(g[cells] * (alpha[cells] - y[cells])))
    if slope <= 0.0:
        return x
    delta = (float(result.external_power) - deposit) / slope
    result.t_in += delta
    result.t_gas = result.t_gas + np.where(cells, alpha * delta, 0.0)
    t_inf = np.asarray(mesh.bc_T_inf, dtype=float).ravel(order="F").copy()
    t_inf[cells] = result.t_gas[cells]
    mesh.bc_T_inf = t_inf.reshape(np.shape(mesh.T), order="F")
    return x + delta * y


@dataclass
class GasSegment:
    """One straight piece of the circuit between two nodes, and the cells it crosses.

    ``cells`` are in the order the gas meets them and ``length`` is the centreline
    length inside each [m]; ``flow`` is the segment's share of the loop's mass flow.
    ``bore`` is the diameter the gas sees (film, friction), ``outer`` the one the bed
    sees (the well model), ``axis`` the axis the segment runs along (0, 1 or 2: what the
    well model reads the cross-section of the cell from).  A lagged segment (``exchanges``
    False) carries the gas and exchanges nothing.
    """

    name: str
    source: int
    target: int
    flow: float
    bore: float
    outer: float
    axis: int = 2
    cells: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    length: np.ndarray = field(default_factory=lambda: np.empty(0))
    exchanges: bool = True


@dataclass
class GasGraph:
    """The circuit: segments between nodes; the external device joins ``outlet`` to
    ``inlet`` (the resistors heat, the exchanger cools, the gas between them)."""

    segments: list[GasSegment]
    inlet: int = 0
    outlet: int = 1

    def order(self) -> list[int]:
        """Segments in an order that meets every node after all of its feeds.

        Kahn's topological sort on the nodes; a cycle (a circuit that feeds itself
        without passing the external device) is refused.
        """
        pending: dict[int, int] = {}
        for segment in self.segments:
            pending[segment.target] = pending.get(segment.target, 0) + 1
        ready = [self.inlet]
        done: list[int] = []
        leaving: dict[int, list[int]] = {}
        for index, segment in enumerate(self.segments):
            leaving.setdefault(segment.source, []).append(index)
        seen_nodes = set()
        while ready:
            node = ready.pop()
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            for index in leaving.get(node, []):
                done.append(index)
                target = self.segments[index].target
                pending[target] -= 1
                if pending[target] == 0:
                    ready.append(target)
        if len(done) != len(self.segments):
            raise ValueError("the gas circuit has a segment the inlet does not reach, or "
                             "a loop that bypasses the external device")
        return done

    def check(self, tolerance: float = 1e-9) -> None:
        """Mass is conserved at every node (the inlet gives 1, the outlet takes 1)."""
        net: dict[int, float] = {}
        for segment in self.segments:
            net[segment.source] = net.get(segment.source, 0.0) - segment.flow
            net[segment.target] = net.get(segment.target, 0.0) + segment.flow
        net[self.inlet] = net.get(self.inlet, 0.0) + 1.0
        net[self.outlet] = net.get(self.outlet, 0.0) - 1.0
        worst = max((abs(v) for v in net.values()), default=0.0)
        if worst > tolerance:
            raise ValueError(f"the gas circuit does not conserve mass (worst node "
                             f"{worst:.3g} of the flow)")


def _run_axis(run: PipeRun) -> int:
    """The axis a run mostly runs along: what the well model's cross-section is."""
    step = np.abs(np.asarray(run.points[-1], dtype=float)
                  - np.asarray(run.points[0], dtype=float))
    return int(np.argmax(step)) if step.any() else 2


def _cell_extents(mesh) -> np.ndarray:
    """``(n, 3)`` cell edges [m], flat, on either mesh."""
    extent = getattr(mesh, "extent", None)
    if extent is not None:
        return np.asarray(extent, dtype=float)
    return np.column_stack([np.asarray(mesh.axis_size(axis)).ravel(order="F")
                            for axis in range(3)])


#: Peaceman's equivalent radius of a cell of cross-section ``a x b``: 0.14 sqrt(a^2+b^2)
PEACEMAN = 0.14

#: the gas moves this far [K] before a segment's properties are re-evaluated
PROPERTY_STEP = 10.0


@dataclass
class FluidLoop:
    """A closed loop of pipe runs with a total mass flow and an external power.

    Two ways to describe the circuit: ``runs`` with a ``split`` (parallel tubes between
    the inlet and the outlet, the reference model the unit tests pin) or a ``graph``
    (the whole circuit, headers included, :class:`GasGraph`).  ``runs`` always carry the
    hydraulics.  ``well_model`` adds the bed-side resistance of the tube in its cell and
    ``variable_properties`` evaluates the gas at its own temperature, segment by segment.
    """

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
    #: absolute pressure of the loop [Pa]: sets the density and the validity of the
    #: incompressible pressure-drop treatment (valid while dp/p < 10%)
    pressure: float = 101325.0
    #: the coldest the gas can come back into the bed [K]: an exchanger cannot return
    #: it colder than its own cold side, so a balance that asks for less is refused
    t_in_min: float = 0.0
    #: the whole circuit (None: the runs in parallel)
    graph: GasGraph | None = None
    #: the bed-side resistance of the tube in its cell (Peaceman)
    well_model: bool = False
    #: tube wall: thickness [m] (the outer diameter of a run is ``d + 2 t``) and
    #: conductivity [W/(m K)] (stainless steel at 500 degC ~ 21)
    wall_thickness: float = 0.0
    k_wall: float = 21.0
    #: the gas at its own temperature, segment by segment (lagged by one march)
    variable_properties: bool = False
    #: per segment, the gas temperature the properties were taken at [K] (the lag)
    segment_temperatures: np.ndarray | None = field(default=None, repr=False)
    #: wetted area [m^2] whose cell was too small for the well model, last march
    well_area_uncorrected: float = field(default=0.0, repr=False)

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

    def _h(self, mass_flow_run: float, fluid: Fluid, diameter: float | None = None
           ) -> float:
        if self.h_fluid is not None:
            return float(self.h_fluid)
        if diameter is None:
            diameters = [run.diameter for run in self.runs if run.diameter > 0]
            diameter = float(np.mean(diameters)) if diameters else 0.05
        return pipe_h(mass_flow_run, diameter, fluid)

    def _base_fluid(self) -> Fluid:
        """The gas of a march at constant properties (a pressurised loop's density)."""
        if abs(self.pressure - 101325.0) <= 1.0:
            return self.fluid
        return self.fluid.at_pressure(self.pressure, self.t_in or 300.0)

    def circuit(self) -> GasGraph:
        """The circuit the march walks: the graph, or the runs in parallel."""
        if self.graph is not None:
            return self.graph
        split = self._flow_split()
        return GasGraph([GasSegment(name=run.name, source=0, target=1,
                                    flow=float(fraction), bore=float(run.diameter),
                                    outer=float(run.diameter + 2.0 * self.wall_thickness),
                                    axis=_run_axis(run), cells=run.cells,
                                    length=np.asarray(run.length, dtype=float))
                         for run, fraction in zip(self.runs, split, strict=True)])

    # -------------------------------------------------------------------- march
    def _march(self, graph: GasGraph, order: list[int], temperatures: np.ndarray | None,
               wall_flat: np.ndarray, extents: np.ndarray | None,
               k_bed: np.ndarray | None) -> tuple[list, dict]:
        """One march of the circuit; temperatures affine in the loop inlet.

        Returns, per segment, ``(m_dot, cp, h, coeff, (a_out, b_out))`` - ``coeff``
        the per-cell columns ``in_a, in_b, out_a, out_b, mean_a, mean_b, decay`` - and
        the affine outlet of every node.
        """
        base = self._base_fluid()
        nodes: dict[int, tuple[float, float]] = {graph.inlet: (1.0, 0.0)}
        mixing: dict[int, list[float]] = {}
        marches: list = [None] * len(graph.segments)
        uncorrected = 0.0
        for index in order:
            segment = graph.segments[index]
            if segment.source not in nodes:
                acc = mixing.get(segment.source)
                if acc is None or acc[2] <= 0.0:
                    nodes[segment.source] = (0.0, 0.0)
                else:
                    nodes[segment.source] = (acc[0] / acc[2], acc[1] / acc[2])
            a0, b0 = nodes[segment.source]
            m_dot = float(self.mass_flow * segment.flow)
            if self.variable_properties and temperatures is not None:
                fluid = self.fluid.at_pressure(self.pressure, float(temperatures[index]))
            else:
                fluid = base
            cp = fluid.cp
            count = segment.cells.size
            if m_dot <= 0.0:
                marches[index] = None
                continue
            h = self._h(m_dot, fluid, segment.bore) if segment.exchanges else 0.0
            walls = wall_flat[segment.cells].astype(float) if count else np.empty(0)
            if count and segment.exchanges and h > 0.0:
                length = np.asarray(segment.length, dtype=float)
                resistance = 1.0 / (h * np.pi * segment.bore)             # [m K/W]
                if self.wall_thickness > 0.0 and segment.outer > segment.bore:
                    resistance += (np.log(segment.outer / segment.bore)
                                   / (2.0 * np.pi * self.k_wall))
                resistance = np.full(count, resistance)
                if self.well_model and extents is not None and k_bed is not None:
                    cross = np.delete(extents[segment.cells], segment.axis, axis=1)
                    r_eq = PEACEMAN * np.hypot(cross[:, 0], cross[:, 1])
                    r_o = 0.5 * segment.outer
                    valid = r_eq > r_o
                    well = np.where(valid, np.log(np.maximum(r_eq, r_o) / r_o), 0.0)
                    resistance = resistance + well / (2.0 * np.pi * k_bed[segment.cells])
                    uncorrected += float(np.sum(np.pi * segment.outer * length[~valid]))
                ntu = length / resistance / (m_dot * cp)
            else:
                ntu = np.zeros(count)
            decay = np.exp(-ntu)
            mean_factor = np.divide(1.0 - decay, ntu, out=np.ones(count), where=ntu > 0.0)
            # the inlet of every piece is affine in the loop inlet, T = a T_in + b: the
            # slope is the product of the decays upstream, the offset follows the one
            # recurrence the march cannot vectorise (a plain float loop)
            in_a = a0 * np.concatenate(([1.0], np.cumprod(decay)[:-1])) if count else \
                np.empty(0)
            in_b = np.empty(count)
            b = b0
            for position, (t_wall, d) in enumerate(zip(walls.tolist(), decay.tolist(),
                                                       strict=True)):
                in_b[position] = b
                b = t_wall + (b - t_wall) * d
            a = float(in_a[-1] * decay[-1]) if count else a0
            coeff = np.column_stack((in_a, in_b, decay * in_a,
                                     walls + (in_b - walls) * decay,
                                     in_a * mean_factor,
                                     walls + (in_b - walls) * mean_factor, decay))
            marches[index] = (m_dot, cp, h, coeff, (a, b))
            acc = mixing.setdefault(segment.target, [0.0, 0.0, 0.0])
            acc[0] += m_dot * cp * a
            acc[1] += m_dot * cp * b
            acc[2] += m_dot * cp
        for node, acc in mixing.items():
            if node not in nodes and acc[2] > 0.0:
                nodes[node] = (acc[0] / acc[2], acc[1] / acc[2])
        self.well_area_uncorrected = uncorrected
        return marches, nodes

    # -------------------------------------------------------------------- solve
    def solve(self, mesh: Mesh3D, wall: np.ndarray | None = None) -> FluidResult:
        """March the loop once; returns the fields and the loop temperatures.

        ``wall`` is the solid temperature field [K] (defaults to ``mesh.T``).  Each cell
        stores the affine coefficients of its inlet, outlet and mean gas temperature, so
        the loop balance and the per-cell power are evaluated in one vectorised pass after
        the march.  The balance is ``sum_cells q = Q_ext``: the bed receives what the
        external device gives, exactly.
        """
        graph = self.circuit()
        order = graph.order()
        wall_flat = np.asarray(mesh.T if wall is None else wall, dtype=float).ravel(
            order="F")
        volume = np.asarray(mesh.V, dtype=float).ravel(order="F")
        extents = _cell_extents(mesh) if self.well_model else None
        k_bed = (np.asarray(mesh.k, dtype=float).ravel(order="F")
                 if self.well_model else None)
        temperatures = self.segment_temperatures
        if temperatures is not None and temperatures.size != len(graph.segments):
            temperatures = None
        passes = 1
        if self.variable_properties and temperatures is None:
            # no march yet to lag on: a first pass with the gas at the mean wall
            # temperature of each segment, then the march at the temperatures it gives
            temperatures = np.array([
                float(np.mean(wall_flat[s.cells])) if s.cells.size else
                float(self.t_in or np.mean(wall_flat)) for s in graph.segments])
            passes = 2
        for step in range(passes):
            marches, nodes = self._march(graph, order, temperatures, wall_flat, extents,
                                         k_bed)
            t_in = self._inlet(marches, wall_flat, graph)
            if self.variable_properties:
                fresh = np.array([
                    (float(np.mean(entry[3][:, 4] * t_in + entry[3][:, 5]))
                     if entry is not None and entry[3].shape[0] else
                     float(temperatures[index]))
                    for index, entry in enumerate(marches)])
                if step + 1 < passes:
                    temperatures = fresh
                else:
                    # a segment keeps the temperature its properties were taken at until
                    # the gas has moved by PROPERTY_STEP: cp changes by ~0.2 % in 10 K,
                    # and a film that stays put keeps the operator (and its AMG
                    # hierarchy) of the previous step instead of rebuilding it every step
                    kept = temperatures if temperatures is not None else fresh
                    moved = np.abs(fresh - kept) > PROPERTY_STEP
                    self.segment_temperatures = np.where(moved, fresh, kept)
        return self._result(graph, marches, nodes, t_in, wall_flat, volume)

    def _inlet(self, marches: list, wall_flat: np.ndarray, graph: GasGraph) -> float:
        """The loop inlet: prescribed, or the one that makes the bed take ``Q_ext``."""
        if self.t_in is not None:
            return float(self.t_in)
        if all(entry is None for entry in marches):
            return float("nan")
        slope = offset = 0.0
        for segment, entry in zip(graph.segments, marches, strict=True):
            if entry is None:
                continue
            m_dot, cp, _h, coeff, _out = entry
            g = m_dot * cp * (1.0 - coeff[:, 6])
            slope += float(np.sum(g * coeff[:, 0]))
            offset += float(np.sum(g * (coeff[:, 1] - wall_flat[segment.cells])))
        if slope <= 0.0:
            raise ValueError("the loop balance has no solution: check the flow")
        t_in = (float(self.external_power) - offset) / slope
        if t_in <= max(self.t_in_min, 0.0):
            raise ValueError(
                f"the loop cannot carry {self.external_power / 1000:+.1f} kW at "
                f"{self.mass_flow:.3f} kg/s: the gas would have to come back at "
                f"{t_in - 273.15:.0f} degC, below the exchanger's "
                f"{self.t_in_min - 273.15:.0f} degC.  The bed around the pipes cannot "
                f"give that power any more: reduce it, or add flow or pipe surface")
        return t_in

    def _result(self, graph: GasGraph, marches: list, nodes: dict, t_in: float,
                wall_flat: np.ndarray, volume: np.ndarray) -> FluidResult:
        """The per-cell fields, the runs and the hydraulics of one march."""
        result = FluidResult(external_power=float(self.external_power))
        result.t_in = t_in
        q_fluid = np.zeros(wall_flat.shape, dtype=float)
        conductance = np.zeros(wall_flat.shape, dtype=float)
        weighted_gas = np.zeros(wall_flat.shape, dtype=float)
        weighted_slope = np.zeros(wall_flat.shape, dtype=float)
        total_power = 0.0
        for segment, entry in zip(graph.segments, marches, strict=True):
            if entry is None:
                result.runs.append(RunResult(name=segment.name, cells=segment.cells,
                                             t_fluid=np.empty(0), q_solid=np.empty(0)))
                continue
            m_dot, cp, h, coeff, (a, b) = entry
            t_in_cell = coeff[:, 0] * t_in + coeff[:, 1]
            t_out_cell = coeff[:, 2] * t_in + coeff[:, 3]
            t_mean = coeff[:, 4] * t_in + coeff[:, 5]
            q_cell = m_dot * cp * (t_in_cell - t_out_cell)                # [W]
            np.add.at(q_fluid, segment.cells, q_cell / volume[segment.cells])  # [W/m^3]
            # q = G (T_gas,in - T_wall) exactly: G = m c (1 - e^-NTU) of the piece
            g_cell = m_dot * cp * (1.0 - coeff[:, 6])
            np.add.at(conductance, segment.cells, g_cell)
            np.add.at(weighted_gas, segment.cells, g_cell * t_in_cell)
            np.add.at(weighted_slope, segment.cells, g_cell * coeff[:, 0])
            run_power = float(np.sum(q_cell))
            total_power += run_power
            result.runs.append(RunResult(
                name=segment.name, cells=segment.cells, t_fluid=t_mean, q_solid=q_cell,
                t_out=float(a * t_in + b), mass_flow=m_dot, power=run_power,
                ntu=float(np.sum(-np.log(np.maximum(coeff[:, 6], 1e-300))))))
        a_out, b_out = nodes.get(graph.outlet, (float("nan"), float("nan")))
        result.t_out = float(a_out * t_in + b_out)

        # hydraulics: the runs are in *parallel*, so they share the pressure drop and
        # each one carries only its share of the flow.  Adding the drops of the runs
        # (as if they were in series) would overstate the blower by orders of magnitude.
        fluid = self._base_fluid()
        split = self._flow_split()
        drops = [pressure_drop(self.mass_flow * fraction, run.diameter, run.total_length,
                               fluid, self.roughness, self.fittings_k)
                 for run, fraction in zip(self.runs, split, strict=True)]
        delta_p = float(np.mean(drops)) if drops else 0.0
        result.delta_p = delta_p
        result.pressure = float(self.pressure)
        result.fan_power = fan_power(self.mass_flow, delta_p, fluid, self.fan_efficiency)
        if delta_p > 0.1 * self.pressure:
            result.notes.append(
                f"the circuit drops {delta_p / self.pressure * 100:.0f}% of its absolute "
                f"pressure: the gas is no longer incompressible, the density varies "
                f"along the loop and this pressure-drop model loses validity")
        if self.well_model and self.well_area_uncorrected > 0.0:
            result.notes.append(
                f"{self.well_area_uncorrected:.2f} m2 of tube sit in cells too small for "
                f"the well model (plan edge under 2.53 d): their bed-side resistance is "
                f"the mesh's, overstated")
        result.q_fluid = q_fluid
        result.conductance = conductance
        result.t_gas = np.divide(weighted_gas, conductance,
                                 out=np.zeros_like(weighted_gas), where=conductance > 0.0)
        result.gas_slope = np.divide(weighted_slope, conductance,
                                     out=np.zeros_like(weighted_slope),
                                     where=conductance > 0.0)
        result.balanced = self.t_in is None
        result.power = total_power
        if self.t_in is not None:
            # a prescribed inlet fixes the gas, and the power the external device has to
            # supply (+) or takes (-) is what the bed takes from it
            result.external_power = total_power
        return result
