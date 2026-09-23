# 19. The hydraulics of the circuit, and the header engine

The flow each riser carries decides where the heat goes.  It is not a setting: it is
what the pressures of the network give it.  This document describes how the simulator
solves those pressures (`src/solver/hydraulics.py`), how the **header engine** sizes the
distributor, the collector and the ducts so that every riser gets the flow its share of
the bed asks for (`src/core/pipe_network.py::design_headers`), and what it found on the
default plant.  Keys in brackets refer to [17_REFERENCES.md](17_REFERENCES.md).

## 1. The network

`PipeNetwork.hydraulic_network()` turns the built network into straight pipes between
nodes:

| pipes | sizing group | local losses |
|---|---|---|
| the inlet and outlet ducts (their polylines) | `inlet_duct`, `outlet_duct` | K = 1.5 (nozzle, bends) |
| a ring main: its chords between consecutive taps, closed | `dist_ring_g`, `coll_ring_g` | - |
| the jumpers of a ring chain | `dist_jumper_g`, `coll_jumper_g` | K = 1.0 (two tees) |
| the trunk of a radial manifold, between consecutive crossings | `dist_trunk_k`, `coll_trunk_k` | K = 1.0 |
| the sections of a ladder header (up to 8) | `dist_section_k`, `coll_section_k` | - |
| the risers (the user's tubes, never sized) | - | K = 2.0 (tee in and out) + the orifice |

A ring main is two paths in parallel between its taps, a chain is rings in series, a
manifold feeds every ring at the two points where the trunk crosses it: the network is
*looped*, and only a network solve says how the flow divides.

## 2. Solving it

In each pipe of bore `D`, area `A`, length `L`, local losses `K`, from node `a` to node
`b`, at the gas density `rho` of that pipe:

```
p_a - p_b - rho g (z_b - z_a) = (f L / D + K) m |m| / (2 rho A^2)
f = max(64 / Re, Haaland(Re, eps / D))
```

Darcy-Weisbach with Haaland's explicit Colebrook [Haaland] joined to the laminar law
where they meet (a jump at Re = 2300 makes a looped network cycle between two
solutions); the hydrostatic term is the **draught of the gas** - a hot riser draws like a
chimney.  The node balances (the whole flow in at the inlet, out at the outlet whose
pressure is the reference) are solved by Newton's method on the node pressures: each
pipe is linearised as `m = w (dp - dp_0)` with `w = dm/d(dp) = 1 / (2 R |m|)` (a
Hagen-Poiseuille floor keeps a stagnant pipe conducting), the weighted Laplacian is
solved, and the step is damped when a flow changes a lot - the nodal form of network
analysis, as in the global gradient algorithm of Todini and Pilati [TP88].  The default
network (126 risers, ~480 pipes) converges in 15-25 iterations, ~0.1 s.

**The draught matters in both directions** (`tests/test_hydraulics.py`): two parallel
5 m risers at 300 and 900 K share the flow in favour of the hot one at a low flow
(the 39 Pa of draught wins) and against it at a high flow (the hot gas is three times
lighter, runs three times faster at the same mass flow, and loses three times more to
friction) - the hot-channel starvation of forced gas flow.  The loop therefore re-solves
the hydraulics at the gas temperatures of its own march (§6).

## 3. What each riser should carry

The target of a riser is the **share of the bed it serves**: its Voronoi area in plan
(`voronoi_shares`, a raster of the bed's disc given to the nearest riser).  If the gas
leaves the same fraction of its heat in every riser, the heat then goes to the sand in
proportion to the sand.  On the default rings the targets range from 0.65 to 0.91 % of
the flow per riser.

## 4. Balancing orifices

A riser loses ~100 Pa at its design flow, which is small next to the pressure changes
along a header: headers alone would have to be very large to keep every riser on target.
Real distributors are balanced with **calibrated orifice plates** at the riser inlets,
and so is this one.  `HydraulicNetwork.balance` fixes every riser at its target flow and
solves the headers around them; the distributor and the collector are then two
separate networks, each with its own pressure field.  A riser needs

```
dp_orifice = C + p_distributor - p_collector - rho g H - dp_friction
```

across its orifice, with `C` the one pressure offset between the two networks.  The
smallest `C` that leaves no orifice negative puts the worst-placed riser at zero, and `C`
is the circuit's pressure drop.  The loss coefficient, referred to the tube velocity, is
`K = 2 rho A^2 dp / m^2`, and the plate that gives it is a thin sharp-edged orifice of
Idelchik [Idelchik, diagram 4-15]:

```
K = (1 + 0.707 sqrt(1 - f) - f)^2 / f^2,    f = (d_hole / d_bore)^2
```

inverted by bisection (`orifice_bore`).  With the orifices the design reproduces the
targets to round-off (`tests/test_hydraulics.py`: a rebuilt network with the designed
plates delivers every target to 1e-10), and because the orifices dominate a riser's
resistance the distribution is robust off the design point.

## 5. The engine (`design_headers`)

For every candidate collection - the chosen one, and the **radial manifold** for a ring
layout:

1. **Start** each header group at the smallest nominal size (EN 10220 outer diameters,
   DN 50 to DN 400 [EN10220]) that carries its flow under the velocity limit (20 m/s by
   default: hot-air ducts are designed at 15-25 m/s), a ring main never wider than the
   room between two rings, no header narrower than the tube.  A layout that is too fast
   even with its largest headers is rejected at once.
2. **Balance** it (§4) and read the circuit's pressure drop.
3. **Grow** the group whose next size lowers the drop the most per unit of added pipe
   volume, among the six with the steepest pressure gradient, while a step saves at
   least 2 % of the drop - the headers stop growing where a larger pipe no longer pays
   for itself in fan power.
4. **Check the velocity** of the balanced and of the free network (a ring fed at both
   trunk taps is also a bypass of the trunk, so the two can differ) and grow what is too
   fast.
5. **Drop the orifices** if the headers alone keep every riser within the tolerance
   (5 % by default).
6. **Lift the headers into the sand**: the centreline one radius plus a cover of a
   quarter diameter (at least 50 mm) above the bottom slab and below the top one, so a
   header lies in the bed it charges instead of on the insulation.  The lift shortens the
   risers and moves the pressures, so steps 1-5 are repeated until it stops moving.

The candidate that meets the velocity limit with the lowest pressure drop wins.

## 6. In the runs

`PipeNetwork.gas_graph` orients every pipe along its solved flow and gives it its share
of the flow; the thermal march ([18](18_SOLVER.md) §5) walks it.  When the gas properties
are refreshed (the gas moved by more than 10 K), the loop re-solves the hydraulics at the
gas temperatures of every segment and moves the flows - turning a segment round if its
flow reversed (a ring fed at both ends moves its stagnation point).  The fan works
against the network's own pressure drop.  A split imposed on purpose (equal, path, ring,
sector) is still available for comparisons: the risers take it and the headers carry the
flows that conserve mass with it.

## 7. The default plant

4 m bed, 6 rings, 126 risers of 50 mm, 1 kg/s of air at 1 atm, design gas at 500 °C:

| collection | largest headers | header velocity | circuit drop | verdict |
|---|---|---|---|---|
| reverse-return ring chain | DN 200 between the rings (the room there), ducts DN 400 | **51.6 m/s** | 3169 Pa | rejected: every ring's flow crosses the rings before it |
| radial manifold | trunk tapering 324 → 60 → 219 mm, rings 60-219 mm, ducts DN 400 | **17.1 m/s** | **751 Pa** | chosen |

With the manifold the headers alone leave the risers 61 % off target; orifice plates with
holes from 23.7 to 46 mm in the 46 mm bore (K up to 25, 413 Pa across the most throttled)
put them on target, two risers need none.  The headers sit 243 mm into the sand.  The
engine takes ~17 s on this plant and runs when the plant changed (*Build mesh*, or *Size
the headers*); a standby then takes 15 s (5.9 kW at 500 °C) and 6 h of charge 13 s.

**What the chain teaches.**  The ring chain looked connected but hydraulically it is
rings in series: the outer ring's two halves carry the flow of every ring inside it,
and its jumper nearly all of it.  With headers of the default 150 mm the riser flows
ranged from 0.01 to 5 times the mean; no header that fits between the rings fixes it.
Feeding the rings in parallel is the change that does.

## 8. Limits

* The design point is one gas temperature (the density the headers are sized at); the
  runs re-solve at the real temperatures, and the orifices keep the distribution close.
* The gas is incompressible per pipe (valid while the circuit loses a few percent of its
  absolute pressure: 750 Pa of 101 kPa here).
* The tees' losses are constant coefficients; the branch/run split of a real tee
  depends on the flow ratio (Idelchik gives the full diagrams).
* The engine is greedy: it finds a good design, not the proven optimum.
