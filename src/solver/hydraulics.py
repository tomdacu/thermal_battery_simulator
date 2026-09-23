"""The hydraulics of the gas circuit: flows from pressures, and headers sized for them.

A buried network is a looped pipe network: a ring main is two paths in parallel between
its entry and its exit tap, the rings of a chain are in series, every riser is a branch
between the distributor and the collector.  How the flow divides between the risers is
not a choice: it is what the pressures of that network give, and it decides where the
heat goes.  This module solves it and sizes the headers so that it comes out as wanted.

**The network** (:class:`HydraulicNetwork`) is a set of straight pipes between nodes.  In
each pipe of bore ``D``, area ``A``, length ``L`` and local losses ``K``, carrying the
mass flow ``m`` from node ``a`` to node ``b`` at a gas density ``rho``:

    p_a - p_b - rho g (z_b - z_a) = (f L / D + K) m |m| / (2 rho A^2)

Darcy-Weisbach with the friction factor of Haaland (turbulent, explicit Colebrook;
Haaland, J. Fluids Eng. 105 (1983) 89) or ``64 / Re`` (laminar), the local losses of the
tees and bends from Idelchik's handbook, and the hydrostatic head of the gas - a hot
riser draws like a chimney.  The node balances ``sum m = injection`` (the whole flow in at
the inlet, out at the outlet, whose pressure is the reference) are solved by Newton's
method on the node pressures: every pipe is linearised as ``m = w (dp - dp_0)`` with
``w = dm/d(dp) = 1 / (2 R |m|)``, the resulting weighted Laplacian is solved, and the step
is damped until the balance closes to 1e-9 of the flow (the classical nodal method of
network analysis; Todini and Pilati, 1988, for the water-network form of it).

**The sizing engine** (:func:`size_headers`) picks the diameter of every header group
(each ring main, each jumper, the ducts; the sections of a ladder header) from a
catalogue of nominal pipe sizes so that

* every riser carries its **target** share of the flow within ``tolerance``: the target
  is the share of the bed the riser serves (its Voronoi area in plan), so that the heat
  goes to the sand in proportion to the sand;
* no header exceeds the **velocity** limit (pressure drop, noise and erosion; hot-air
  ducts are designed at 15-25 m/s);
* a ring main or a jumper fits between the rings;

and among the designs that do, it prefers the lighter one: a greedy step-up on the group
whose next size reduces the maldistribution the most per unit of added pipe volume, then
a step-down pass on every group that can lose a size without breaking a constraint.  The
riser tubes are the heat-transfer surface and stay the user's.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from .fluid import Fluid

G = 9.80665


def darcy_friction(reynolds: np.ndarray, relative_roughness: np.ndarray) -> np.ndarray:
    """Darcy friction factor, continuous: ``max(64 / Re, Haaland)``.

    The laminar law below and Haaland's explicit Colebrook above, joined where they meet
    (Re ~ 1200-2300 depending on the roughness) instead of jumping at Re = 2300: a jump
    makes a network whose pipes sit near the transition cycle between two solutions.
    """
    re = np.asarray(reynolds, dtype=float)
    turbulent = 1.0 / (-1.8 * np.log10((np.asarray(relative_roughness) / 3.7) ** 1.11
                                       + 6.9 / re)) ** 2
    return np.maximum(64.0 / re, turbulent)

#: outer diameters of the nominal pipe sizes a header is chosen from [m]: DN 50 to DN 600
#: (ISO 1127 / EN 10220 outer diameters)
CATALOGUE = (0.0603, 0.0761, 0.0889, 0.1143, 0.1397, 0.1683, 0.2191, 0.2730, 0.3239,
             0.3556, 0.4064, 0.4570, 0.5080, 0.6100)

#: local losses [-] (Idelchik): a tee's branch in and out of a riser, a jumper's two
#: tees, a duct's nozzle and bends
K_RISER = 2.0
K_JUMPER = 1.0
K_DUCT = 1.5

#: groups tried per step of the sizing: the ones with the steepest pressure gradient
CANDIDATES = 6
#: the largest header the engine may choose [m] (DN 400): a larger one no longer fits a
#: bed of a few metres, and the design should change instead (pressure, modules)
MAX_HEADER = 0.4064


@dataclass
class Pipe:
    """One straight pipe between two nodes (nominal direction ``a -> b``)."""

    name: str
    a: int
    b: int
    start: np.ndarray                 # (3,) [m] centreline at node a
    end: np.ndarray                   # (3,) [m] centreline at node b
    bore: float                       # [m]
    outer: float                      # [m]
    k_local: float = 0.0
    group: str = ""                   # the sizing group ("" = not sized: a riser)
    kind: str = "header"              # riser, header, jumper, duct
    riser: int = -1                   # branch index of a riser

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.end - self.start))

    @property
    def rise(self) -> float:
        return float(self.end[2] - self.start[2])

    @property
    def area(self) -> float:
        return 0.25 * np.pi * self.bore ** 2


@dataclass
class HydraulicState:
    """A solved network: flows, pressures and what they cost."""

    flow: np.ndarray                  # [kg/s] per pipe, + in the nominal direction
    pressure: np.ndarray              # [Pa] per node, the outlet at 0
    velocity: np.ndarray              # [m/s] per pipe
    delta_p: float                    # [Pa] inlet - outlet
    iterations: int
    converged: bool
    riser_flow: np.ndarray            # [kg/s] per branch, in branch order


@dataclass
class Balancing:
    """The riser orifices that put every riser on its target (see ``balance``)."""

    k_orifice: np.ndarray            # [-] per riser, referred to the tube velocity
    orifice_dp: np.ndarray           # [Pa] per riser
    delta_p: float                   # [Pa] the circuit, inlet to outlet
    state: HydraulicState            # the headers around the fixed riser flows
    max_velocity: float              # [m/s] in the headers


@dataclass
class HydraulicNetwork:
    """Pipes between nodes, the inlet fed with the whole flow, the outlet the reference."""

    pipes: list[Pipe]
    n_nodes: int
    inlet: int = 0
    outlet: int = 1
    roughness: float = 1.5e-5

    @property
    def n_risers(self) -> int:
        return sum(1 for pipe in self.pipes if pipe.riser >= 0)

    def groups(self) -> list[str]:
        """The sizing groups, in the order their pipes first appear."""
        seen: dict[str, None] = {}
        for pipe in self.pipes:
            if pipe.group:
                seen.setdefault(pipe.group, None)
        return list(seen)

    def _properties(self, fluid: Fluid, temperature, pressure: float
                    ) -> tuple[np.ndarray, np.ndarray]:
        """Density and viscosity of the gas in every pipe (ideal gas at ``pressure``)."""
        n_pipes = len(self.pipes)
        temps = np.broadcast_to(np.asarray(temperature, dtype=float), (n_pipes,))
        states = {}
        rho = np.empty(n_pipes)
        mu = np.empty(n_pipes)
        for index, t in enumerate(temps):
            key = round(float(t), 1)
            if key not in states:
                states[key] = fluid.at_pressure(pressure, float(t))
            rho[index] = states[key].rho
            mu[index] = states[key].mu
        return rho, mu

    def drops(self, flow: np.ndarray, rho: np.ndarray, mu: np.ndarray) -> np.ndarray:
        """Friction and local loss of every pipe at ``flow`` [Pa], signed with the flow."""
        bore = np.array([p.bore for p in self.pipes])
        area = 0.25 * np.pi * bore ** 2
        length = np.array([p.length for p in self.pipes])
        k_local = np.array([p.k_local for p in self.pipes])
        reynolds = np.maximum(np.abs(flow) * bore / (area * mu), 1.0)
        friction = darcy_friction(reynolds, self.roughness / bore)
        return ((friction * length / bore + k_local) * flow * np.abs(flow)
                / (2.0 * rho * area ** 2))

    def solve(self, mass_flow: float, fluid: Fluid, temperature=773.15,
              pressure: float = 101325.0, initial: np.ndarray | None = None,
              tolerance: float = 1e-9, max_iterations: int = 200,
              fixed: dict[int, float] | None = None) -> HydraulicState:
        """Flows and pressures for ``mass_flow`` [kg/s] of ``fluid`` at ``temperature``.

        ``temperature`` is one value or one per pipe [K]: the density and the viscosity
        of every pipe are the gas's there (ideal gas at ``pressure``), so a hot riser is
        lighter and draws.  ``fixed`` gives some pipes a flow [kg/s] instead of a law:
        they become draws on their two nodes, and a part of the network they cut off
        from the outlet takes its own reference pressure (its lowest node) - the pressures
        of such a part are relative to it.
        """
        n_pipes = len(self.pipes)
        fixed = fixed or {}
        rho, mu = self._properties(fluid, temperature, pressure)
        a = np.array([p.a for p in self.pipes])
        b = np.array([p.b for p in self.pipes])
        length = np.array([p.length for p in self.pipes])
        bore = np.array([p.bore for p in self.pipes])
        area = 0.25 * np.pi * bore ** 2
        k_local = np.array([p.k_local for p in self.pipes])
        head = rho * G * np.array([p.rise for p in self.pipes])       # [Pa]
        relative = self.roughness / bore
        injection = np.zeros(self.n_nodes)
        injection[self.inlet] = mass_flow
        injection[self.outlet] -= mass_flow
        solved = np.ones(n_pipes, dtype=bool)
        for position, value in fixed.items():
            solved[position] = False
            injection[a[position]] -= value
            injection[b[position]] += value
        references = self._references(solved)
        free = np.ones(self.n_nodes, dtype=bool)
        free[references] = False
        sa, sb = a[solved], b[solved]

        flow = (np.full(n_pipes, mass_flow / max(self.n_risers, 1)) if initial is None
                else np.asarray(initial, dtype=float).copy())
        for position, value in fixed.items():
            flow[position] = value
        p = np.zeros(self.n_nodes)
        converged = False
        iteration = 0
        for step in range(1, max_iterations + 1):
            iteration = step
            reynolds = np.maximum(np.abs(flow) * bore / (area * mu), 1.0)
            friction = darcy_friction(reynolds, relative)
            coeff = (friction * length / bore + k_local) / (2.0 * rho * area ** 2)
            # laminar floor: Hagen-Poiseuille, so a pipe with no flow still conducts
            laminar = 32.0 * mu * length / (rho * area * bore ** 2)
            resistance = np.maximum(2.0 * coeff * np.abs(flow), laminar)   # d(dp)/dm
            w = 1.0 / resistance
            # Newton: m = m0 + w (dp - dp(m0)),  dp(m0) = coeff m0 |m0| (laminar floor)
            drop0 = np.where(2.0 * coeff * np.abs(flow) > laminar,
                             coeff * flow * np.abs(flow), laminar * flow)
            offset = flow - w * (drop0 + head)       # m = w (p_a - p_b) + offset
            ws, os_ = w[solved], offset[solved]
            rows = np.concatenate((sa, sb, sa, sb))
            cols = np.concatenate((sa, sb, sb, sa))
            vals = np.concatenate((ws, ws, -ws, -ws))
            lap = sparse.coo_matrix((vals, (rows, cols)),
                                    shape=(self.n_nodes, self.n_nodes)).tocsr()
            rhs = injection.copy()
            np.subtract.at(rhs, sa, os_)
            np.add.at(rhs, sb, os_)
            reduced = lap[free][:, free]
            p_new = np.zeros(self.n_nodes)
            p_new[free] = spsolve(reduced.tocsc(), rhs[free])
            new_flow = flow.copy()
            new_flow[solved] = ws * (p_new[sa] - p_new[sb]) + os_
            change = float(np.max(np.abs(new_flow - flow))) / max(mass_flow, 1e-30)
            # a damped step: the linearisation of m|m| overshoots when a pipe's flow
            # changes sign, and a looped network (a ring main) can then cycle
            damping = 1.0 if change < 0.05 else 0.6
            flow = flow + damping * (new_flow - flow)
            p = p_new
            if change < tolerance:
                converged = True
                break
        velocity = flow / (rho * area)
        riser_flow = np.zeros(self.n_risers)
        for position, pipe in enumerate(self.pipes):
            if pipe.riser >= 0:
                riser_flow[pipe.riser] = flow[position]
        return HydraulicState(flow=flow, pressure=p, velocity=velocity,
                              delta_p=float(p[self.inlet] - p[self.outlet]),
                              iterations=iteration, converged=converged,
                              riser_flow=riser_flow)

    def _references(self, solved: np.ndarray) -> np.ndarray:
        """One reference node per connected part of the solved pipes: the outlet for its
        own part, the lowest node for every other part."""
        n = self.n_nodes
        a = np.array([p.a for p in self.pipes])[solved]
        b = np.array([p.b for p in self.pipes])[solved]
        graph = sparse.coo_matrix((np.ones(a.size), (a, b)), shape=(n, n))
        from scipy.sparse.csgraph import connected_components

        _count, labels = connected_components(graph, directed=False)
        references = [self.outlet]
        seen = {labels[self.outlet]}
        for node in range(n):
            if labels[node] not in seen:
                seen.add(labels[node])
                references.append(node)
        return np.asarray(references, dtype=int)

    def balance(self, mass_flow: float, fluid: Fluid, targets: np.ndarray,
                temperature=773.15, pressure: float = 101325.0) -> Balancing:
        """The orifices that give every riser exactly its target flow.

        The riser flows are fixed at ``mass_flow * targets`` and the headers solved
        around them: the distributor and the collector are then two separate networks,
        each with its own pressure field.  A riser needs
        ``dp_orifice = C + p_dist - p_coll - rho g H - dp_friction`` across an orifice
        at its inlet, where ``C`` is the one pressure offset between the two networks;
        the smallest ``C`` that leaves no orifice negative puts the worst-placed riser at
        zero, and ``C`` is then the circuit's pressure drop.  The loss coefficient of an
        orifice, referred to the tube velocity, is ``K = 2 rho A^2 dp / m^2``.
        """
        targets = np.asarray(targets, dtype=float)
        risers = [position for position, pipe in enumerate(self.pipes)
                  if pipe.riser >= 0]
        wanted = {position: mass_flow * float(targets[self.pipes[position].riser])
                  for position in risers}
        state = self.solve(mass_flow, fluid, temperature, pressure, fixed=wanted)
        rho, mu = self._properties(fluid, temperature, pressure)
        friction = self.drops(state.flow, rho, mu)
        need = np.zeros(self.n_risers)
        head = np.zeros(self.n_risers)
        for position in risers:
            pipe = self.pipes[position]
            head[pipe.riser] = rho[position] * G * pipe.rise
            need[pipe.riser] = (state.pressure[pipe.a] - state.pressure[pipe.b]
                                - head[pipe.riser] - friction[position])
        offset = float(-np.min(need))            # the worst riser gets no orifice
        orifice_dp = need + offset
        k_orifice = np.zeros(self.n_risers)
        for position in risers:
            pipe = self.pipes[position]
            m = wanted[position]
            k_orifice[pipe.riser] = (2.0 * rho[position] * pipe.area ** 2
                                     * orifice_dp[pipe.riser] / max(m * m, 1e-30))
        # the circuit's drop: the collector's inlet-to-outlet drop plus the distributor's
        # from the inlet to the reference riser, i.e. p_inlet - p_outlet with the offset
        # the distributor's pressures are relative to its reference node, which the
        # offset puts at ``offset`` above the outlet
        delta_p = float(state.pressure[self.inlet] + offset)
        header = np.array([pipe.riser < 0 for pipe in self.pipes])
        return Balancing(k_orifice=k_orifice, orifice_dp=orifice_dp, delta_p=delta_p,
                         state=state,
                         max_velocity=float(np.max(np.abs(state.velocity[header]))))

    def with_sizes(self, sizes: dict[str, float], wall: float) -> HydraulicNetwork:
        """The same network with the header groups at ``sizes`` (outer diameters)."""
        pipes = []
        for pipe in self.pipes:
            if pipe.group and pipe.group in sizes:
                outer = float(sizes[pipe.group])
                pipes.append(Pipe(**{**pipe.__dict__, "outer": outer,
                                     "bore": outer - 2.0 * wall}))
            else:
                pipes.append(pipe)
        return HydraulicNetwork(pipes, self.n_nodes, self.inlet, self.outlet,
                                self.roughness)


    def with_orifices(self, k_orifice: np.ndarray) -> HydraulicNetwork:
        """The same network with an orifice of loss ``k_orifice[b]`` on every riser."""
        pipes = []
        for pipe in self.pipes:
            if pipe.riser >= 0:
                pipes.append(Pipe(**{**pipe.__dict__, "k_local": pipe.k_local
                                     + float(k_orifice[pipe.riser])}))
            else:
                pipes.append(pipe)
        return HydraulicNetwork(pipes, self.n_nodes, self.inlet, self.outlet,
                                self.roughness)


# ------------------------------------------------------------------------ sizing
@dataclass
class SizingResult:
    """What the engine chose, and what it achieves."""

    sizes: dict[str, float]                   # [m] outer diameter per header group
    k_orifice: np.ndarray                     # [-] orifice loss per riser (tube velocity)
    maldistribution: float                    # max |m / target - 1| with the orifices
    natural_maldistribution: float            # the same without them
    max_velocity: float                       # [m/s] in the headers
    delta_p: float                            # [Pa] across the circuit
    orifice_dp: np.ndarray                    # [Pa] across each orifice
    feasible: bool
    steps: int
    targets: np.ndarray                       # target share per riser
    shares: np.ndarray                        # achieved share per riser
    notes: list[str] = field(default_factory=list)

    @property
    def orifices_needed(self) -> bool:
        return bool(np.any(self.k_orifice > 0.0))

    def summary(self) -> str:
        sizes = sorted(set(round(v * 1000.0, 1) for v in self.sizes.values()))
        balance = (f"orifices on {int(np.count_nonzero(self.k_orifice > 1e-9))} risers "
                   f"(K up to {float(np.max(self.k_orifice)):.1f}) balance the flow to "
                   f"{100 * self.maldistribution:.2f} %, "
                   f"{100 * self.natural_maldistribution:.0f} % without them"
                   if self.orifices_needed else
                   f"the flow is within {100 * self.maldistribution:.1f} % of the target "
                   f"with no orifice")
        return (f"headers {sizes[0]:.0f}-{sizes[-1]:.0f} mm in {len(self.sizes)} groups; "
                f"{balance}; header velocity up to {self.max_velocity:.1f} m/s; circuit "
                f"drop {self.delta_p:.0f} Pa")


def maldistribution(state: HydraulicState, mass_flow: float,
                    targets: np.ndarray) -> float:
    """``max |m_riser / (M target) - 1|``: 0 is the flow exactly where it is wanted."""
    wanted = mass_flow * np.asarray(targets, dtype=float)
    return float(np.max(np.abs(state.riser_flow / wanted - 1.0)))


def size_headers(network: HydraulicNetwork, mass_flow: float, fluid: Fluid,
                 targets: np.ndarray, *, temperature: float = 773.15,
                 pressure: float = 101325.0, tolerance: float = 0.05,
                 max_velocity: float = 20.0, wall: float = 0.003,
                 limits: dict[str, float] | None = None, minimum: float = 0.0,
                 maximum: float = MAX_HEADER, catalogue=CATALOGUE,
                 max_steps: int = 60, gain: float = 0.02) -> SizingResult:
    """Choose the header diameters and the riser orifices (see the module docstring).

    1. every group starts at the smallest catalogue size that carries its flow (the flow
       of the balanced network) under ``max_velocity``;
    2. the orifices that put every riser on its target are computed
       (:meth:`HydraulicNetwork.balance`) and the circuit's pressure drop with them;
    3. a group is stepped up while that lowers the drop by at least ``gain`` of it per
       step - the one with the best gain per unit of added pipe volume, among the groups
       with the steepest pressure gradient - so the headers stop growing where a larger
       pipe no longer pays for itself in fan power;
    4. if the network is within ``tolerance`` of the targets with no orifice at all, the
       orifices are dropped.

    ``limits`` caps a group's diameter (the room between two rings), ``minimum`` is the
    smallest header allowed (a tee on the riser needs a header at least as wide).
    """
    groups = network.groups()
    limits = limits or {}
    sizes_all = np.asarray(catalogue, dtype=float)
    allowed = {g: [float(d) for d in sizes_all if d >= minimum - 1e-12
                   and d <= min(limits.get(g, np.inf), maximum) + 1e-12
                   and d - 2 * wall > 0]
               for g in groups}
    for g in groups:
        if not allowed[g]:
            allowed[g] = [float(sizes_all[0])]
    targets = np.asarray(targets, dtype=float)
    notes: list[str] = []

    def sized(choice: dict[str, int]) -> HydraulicNetwork:
        return network.with_sizes({g: allowed[g][choice[g]] for g in groups}, wall)

    def evaluate(choice: dict[str, int]) -> Balancing:
        return sized(choice).balance(mass_flow, fluid, targets, temperature, pressure)

    def volume(choice: dict[str, int]) -> float:
        return float(sum(0.25 * np.pi * allowed[p.group][choice[p.group]] ** 2 * p.length
                         for p in network.pipes if p.group))

    def group_velocity(balancing: Balancing) -> dict[str, float]:
        out = {g: 0.0 for g in groups}
        for position, pipe in enumerate(network.pipes):
            if pipe.group:
                out[pipe.group] = max(out[pipe.group],
                                      abs(float(balancing.state.velocity[position])))
        return out

    # 1. the flows of the balanced network, every header at its widest
    choice = {g: len(allowed[g]) - 1 for g in groups}
    widest = evaluate(choice)
    if widest.max_velocity > max_velocity * (1.0 + 1e-6):
        # even the largest headers that fit are too fast: no size makes this layout
        # work, and the caller is told so at once instead of after a search
        free = sized(choice).solve(mass_flow, fluid, temperature, pressure)
        header = np.array([pipe.group != "" for pipe in network.pipes])
        return SizingResult(
            sizes={g: allowed[g][choice[g]] for g in groups},
            k_orifice=widest.k_orifice, maldistribution=0.0,
            natural_maldistribution=maldistribution(free, mass_flow, targets),
            max_velocity=float(np.max(np.abs(free.velocity[header]))),
            delta_p=widest.delta_p, orifice_dp=widest.orifice_dp, feasible=False,
            steps=0, targets=targets, shares=free.riser_flow / mass_flow,
            notes=[f"even the largest headers that fit run at "
                   f"{widest.max_velocity:.1f} m/s (limit {max_velocity:.1f})"])
    base = fluid.at_pressure(pressure, temperature)
    for g in groups:
        flow = max(abs(float(widest.state.flow[i])) for i, pipe in enumerate(network.pipes)
                   if pipe.group == g)
        for position, d in enumerate(allowed[g]):
            if flow / (base.rho * 0.25 * np.pi * (d - 2 * wall) ** 2) <= max_velocity:
                choice[g] = position
                break
    current = evaluate(choice)
    steps = 0
    # the velocity limit first (the density along the network is not uniform)
    while current.max_velocity > max_velocity and steps < max_steps:
        fast = [g for g, v in group_velocity(current).items()
                if v > max_velocity and choice[g] + 1 < len(allowed[g])]
        if not fast:
            break
        for g in fast:
            choice[g] += 1
        current = evaluate(choice)
        steps += 1
    # 2-3. larger headers while they pay for themselves in pressure drop
    while steps < max_steps:
        drop = {g: 0.0 for g in groups}
        for pipe in network.pipes:
            if pipe.group:
                dp = abs(current.state.pressure[pipe.a] - current.state.pressure[pipe.b])
                drop[pipe.group] = max(drop[pipe.group], dp / max(pipe.length, 1e-6))
        ranked = sorted((g for g in groups if choice[g] + 1 < len(allowed[g])),
                        key=lambda name: -drop[name])[:CANDIDATES]
        best, best_score, best_eval = None, 0.0, None
        here = volume(choice)
        for g in ranked:
            trial = dict(choice)
            trial[g] += 1
            result = evaluate(trial)
            saved = current.delta_p - result.delta_p
            if saved < max(gain * current.delta_p, 1.0):
                continue
            score = saved / max(volume(trial) - here, 1e-12)
            if score > best_score:
                best, best_score, best_eval = g, score, result
        if best is None:
            break
        choice[best] += 1
        current = best_eval
        steps += 1
    # the velocity of the free network (no orifice) too: its flows are not the balanced
    # ones, and the pipe that carries more there must still respect the limit
    # (a ring fed at both of its trunk taps is also a bypass of the trunk between them,
    # so the balanced and the free flows of a header can differ a lot)
    free = sized(choice).solve(mass_flow, fluid, temperature, pressure)
    while steps < max_steps:
        fast = set()
        for position, pipe in enumerate(network.pipes):
            speed = max(abs(float(free.velocity[position])),
                        abs(float(current.state.velocity[position])))
            if (pipe.group and speed > max_velocity
                    and choice[pipe.group] + 1 < len(allowed[pipe.group])):
                fast.add(pipe.group)
        if not fast:
            break
        for g in fast:
            choice[g] += 1
        free = sized(choice).solve(mass_flow, fluid, temperature, pressure,
                                   initial=free.flow)
        current = evaluate(choice)
        steps += 1
    sizes = {g: allowed[g][choice[g]] for g in groups}
    final = sized(choice)
    # 4. the orifices, unless the headers alone are good enough
    natural = maldistribution(free, mass_flow, targets)
    if natural <= tolerance:
        k_orifice = np.zeros(network.n_risers)
        state, delta_p = free, free.delta_p
        mal = natural
        orifice_dp = np.zeros(network.n_risers)
    else:
        k_orifice = current.k_orifice
        orifice_dp = current.orifice_dp
        state = final.with_orifices(k_orifice).solve(mass_flow, fluid, temperature,
                                                     pressure, initial=current.state.flow)
        delta_p = state.delta_p
        mal = maldistribution(state, mass_flow, targets)
    header = np.array([pipe.group != "" for pipe in network.pipes])
    vmax = float(np.max(np.abs(state.velocity[header]))) if header.any() else 0.0
    feasible = vmax <= max_velocity * (1.0 + 1e-6) and mal <= tolerance
    if vmax > max_velocity * (1.0 + 1e-6):
        notes.append(f"a header runs at {vmax:.1f} m/s with the largest size that fits "
                     f"(limit {max_velocity:.1f}): raise the loop pressure, lower the flow "
                     f"or give the headers more room")
    return SizingResult(sizes=sizes, k_orifice=k_orifice, maldistribution=mal,
                        natural_maldistribution=natural, max_velocity=vmax,
                        delta_p=float(delta_p), orifice_dp=orifice_dp,
                        feasible=feasible, steps=steps, targets=targets,
                        shares=state.riser_flow / mass_flow, notes=notes)


def orifice_loss(area_ratio: np.ndarray) -> np.ndarray:
    """Loss coefficient of a thin sharp-edged orifice plate in a pipe [-].

    Idelchik (Handbook of Hydraulic Resistance, diagram 4-15, thin-walled orifice):
    ``K = (1 + 0.707 sqrt(1 - f) - f)^2 / f^2`` with ``f = (d_orifice / d_pipe)^2``,
    referred to the velocity in the pipe.
    """
    f = np.clip(np.asarray(area_ratio, dtype=float), 1e-6, 1.0)
    return (1.0 + 0.707 * np.sqrt(1.0 - f) - f) ** 2 / f ** 2


def orifice_bore(k: np.ndarray, bore: float) -> np.ndarray:
    """The hole of the plate that gives each loss ``k`` in a pipe of ``bore`` [m].

    The inverse of :func:`orifice_loss` by bisection on the area ratio (the loss falls
    monotonically as the hole opens); ``k = 0`` is no plate (the full bore).
    """
    k = np.atleast_1d(np.asarray(k, dtype=float))
    low = np.full(k.shape, 1e-6)
    high = np.ones(k.shape)
    for _ in range(60):
        middle = 0.5 * (low + high)
        too_open = orifice_loss(middle) < k
        high = np.where(too_open, middle, high)
        low = np.where(too_open, low, middle)
    ratio = np.where(k <= 1e-9, 1.0, 0.5 * (low + high))
    return bore * np.sqrt(ratio)


def voronoi_shares(points: np.ndarray, radius: float, samples: int = 240) -> np.ndarray:
    """Share of a disc of ``radius`` closest to each of ``points`` (plan, about its centre).

    A raster of ``samples^2`` points over the disc, each given to its nearest riser: the
    bed each riser heats, if the heat goes to the nearest tube.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    grid = np.linspace(-radius, radius, samples)
    x, y = np.meshgrid(grid, grid)
    inside = x ** 2 + y ** 2 <= radius ** 2
    px, py = x[inside], y[inside]
    nearest = np.empty(px.size, dtype=int)
    chunk = 20000
    for start in range(0, px.size, chunk):
        dx = px[start:start + chunk, None] - points[None, :, 0]
        dy = py[start:start + chunk, None] - points[None, :, 1]
        nearest[start:start + chunk] = np.argmin(dx * dx + dy * dy, axis=1)
    counts = np.bincount(nearest, minlength=points.shape[0]).astype(float)
    return counts / counts.sum()
