# 15. Pipe networks of a cylindrical vessel

Date: 2026-09-20.  Code: `src/core/pipe_network.py` (builds on `src/core/pipes.py`),
tests: `tests/test_pipes.py`.

## 1. What this is

`src/core/pipes.py` generates *a bundle of tubes*.  A vessel large enough to store
something useful cannot be fed by one header: the flow has to be distributed between
hundreds of risers, collected again, and taken out through the wall.  That is a
**network**, and it is what `src/core/pipe_network.py` builds:

```
        outlet duct ___________
            |                  |          <- hot out, through the WALL (never the roof)
    ==============================  collector  z_top   (+ ring jumpers, + take-offs)
      | | | | | | | | | | | | | | |   risers     (they span the active band exactly)
    ==============================  distributor z_bottom
            |                  |          <- cold in, through the wall
        inlet duct ------------
```

* **risers** - one `PipeRun` each, from the distributor to the collector;
* **distributor / collector** - the headers: a ladder run per elevation (lattice
  layouts) or one closed ring run per ring, linked by radial jumpers (ring layout);
* **ducts** - the two runs that cross the vessel wall, plus the vertical duct on the
  axis of the `central_header` collection;
* **connectors** - the jumpers between the rings.

Every one of those is a `PipeRun`, so it carries its own wetted area, its own length
and, through `src/solver/fluid.py`, its own pressure drop.  A collector is *not*
"losses we ignore": it is a pipe like any other.

## 2. Invariants

Both invariants of `src/core/pipes.py` are preserved and are the reason the module
exists in this shape:

1. **The wetted area is geometric.**  For a cell `c`,
   `A_c = pi d L_c`, with `L_c` the length of centreline inside that cell - never the
   surface of the voxel mask.  The mask is a staircase and its surface is a mesh
   artefact, not a heat-transfer surface.
2. **Nothing is created or destroyed by the rasterisation.**  `sum_c A_c = pi d L_total`
   to machine precision, per run and for the whole network, which is what makes
   `FluidLoop` and `PipeNetwork.voxelize` agree.
3. **The branch flows add up to one.**  `PipeNetwork.split()` returns one fraction per
   riser, summing to 1, either equal (`equal`, the design target) or proportional to the
   branch path (`path`, the coarse rule a header is sized against).

## 3. Layouts

The plan inside the circle of radius `radius - wall_clearance =: r_eff`.  Two pitches
describe every layout: `p_h` between neighbouring risers *inside* a row, a ring or a
radial file, and `p_v` *between* rows, rings or files.

| layout | plan | risers |
|---|---|---|
| `staggered` | equilateral lattice on one origin, alternate rows shifted by `p_h/2` | the bundle the literature recommends |
| `grid` | square lattice on the same rows, all columns lined up | the easy one to fabricate |
| `rings` | concentric rings at `r = k p_v`, each carrying its own risers, linked by radial jumpers | a ring main per circle; the centre stays free for a duct |
| `radial` | radial files of risers, `p_h` apart along the radius, files pitched at the mean radius | a manifold-style bundle |

Rules the generators enforce:

* a pitch below the tube diameter is refused (`validate()`, and `build_pipe_network`
  raises): the tubes would touch;
* a ring takes an **even** number of taps, `2 floor(pi r / p_h)`, so that the tap
  opposite the feed point exists and the ring is symmetric about the diameter that
  joins its feed to its exit.  The floor keeps the arc spacing between neighbouring
  taps at or above `p_h`;
* the entry azimuth of a ring alternates with the ring index, so consecutive rings -
  which the chain visits one after the other - are joined by a short radial hop
  instead of a chord across the vessel;
* a radial file starts at the *pitch circle* `r = p_h / (2 sin(pi / n_files))`, where
  two neighbouring files are exactly `p_h` apart, so no pair of risers of the bundle is
  tighter than the pitch whatever the file count;
* the ring radii are `k p_v`, so the innermost ring is one pitch off the axis.

## 4. Collection modes

The flow is collected in one of four ways.  Each branch is described by the arc it
travels in the distributor (`feed_length`), the riser itself, and the arc it travels in
the collector (`return_length`); `Branch.path_length` is the sum.

| collection | distributor | collector | what it buys |
|---|---|---|---|
| `distributor_collector` | fed at one end of the ladder | discharged at the **same** end | the shortest pipe work; *direct return*, so the branch nearest the nozzles is favoured |
| `reverse_return` | fed at one end | discharged at the **far** end (Tichelmann) | every branch travels the same header length, so the split is uniform |
| `central_header` | taken off a vertical duct on the axis | returned to the same duct, then out through a high nozzle | the headers become short take-offs (*derivazioni*); the return trunk crosses the bundle |
| `two_level_rings` | ring chain at `z_bottom` | ring chain at `z_top`, rings alternating `2 d_duct` in elevation | adjacent rings clear each other, the droppers have a defined length, and the chain stays balanced |

Why the reverse return balances exactly.  Walk the branch order of the distributor.
For the ladder layouts, the branch at position `i` travels `A(i)` along the distributor
and `A_total - A(i)` along the collector when the discharge sits at the far end, so its
header travel is `A_total` for **every** branch.  For the rings the same argument runs
around each ring: a ring fed at one point divides its flow both ways round, so the
branch takes the shorter arc in and the rest of its half out - exactly `C/2` per ring,
whichever tap it is - and the chain then contributes the same jumpers to all of them.
`PipeNetwork.path_spread()` is 1.000 to machine precision in the balanced modes and is
what `note:` reports when it is not.

The `path` split then has a useful consequence: in a balanced network the
path-weighted split *is* the equal split, which is what makes Tichelmann plumbing the
answer to header maldistribution.

## 5. Design rules and sources

| rule | value | why |
|---|---|---|
| horizontal pitch | `2.0 d` (grid/rings/radial), `sqrt(3) d` (staggered) | published practice for a bundle in a granular bed, `docs/13_REDESIGN.md` §2 |
| vertical pitch (rows, rings, files) | `2.5 d` (or `sqrt(3) d` staggered) | idem |
| wall clearance | `2 d` | keeps the outermost riser off the shell, where the sand is a bypass |
| header span (distributor to collector) | `<= 1 m` no care, `1-3 m` check the distribution, `> 3 m` split into parallel modules | the header limit of `src/core/pipes.py` (`HEADER_SAFE`, `HEADER_LIMIT`) |
| bed height per module | `< 4 m` | published practice, `docs/13_REDESIGN.md` §2 |
| nozzles | through the lateral wall, inlet below the distributor, outlet at or above the collector | the roof carries the insulation and the cone: an outlet through it is refused |
| reverse return | outlet nozzle opposite the inlet | the return has to leave at the far end of the chain |

Sources:

1. Polar Night Energy's published description of the closed air loop through pipes
   buried in sand (the architecture this project implements, collected in
   `docs/13_REDESIGN.md`).
2. Incropera, DeWitt, Bergman, Lavine - *Fundamentals of Heat and Mass Transfer*
   (tube bundles: pitches, staggered arrangement).
3. Kaviany - *Principles of Heat Transfer in Porous Media* (the sand side of the same
   surface).
4. ASHRAE Handbook - *HVAC Systems and Equipment*, hydronic piping (direct versus
   reverse return, i.e. Tichelmann balancing; the reason a reverse-return system has
   equal pressure drops per terminal).
5. The 1 m / 3 m header rule and the 4 m module height are the design limits already
   recorded in `docs/13_REDESIGN.md` §2; this module implements exactly those numbers
   and does not invent new ones.  Where a rule is a convention of *this* code - the
   `2 d_duct` level gap, the alternating ring entrances, the even tap count - the text
   above says so instead of quoting a source.

## 6. Parameters

`PipeNetworkConfig` (metres, degrees, absolute elevations from the domain floor):

| field | default | unit | meaning |
|---|---|---|---|
| `radius` | 2.0 | m | inner radius of the vessel |
| `height` | 7.0 | m | wall height of the vessel (the roof is at `base_z + height`) |
| `base_z` | 0.0 | m | elevation of the vessel floor |
| `band_bottom` | 0.3 | m | bottom of the active band, above the floor |
| `band_top` | `height - band_bottom` | m | top of the active band, above the floor |
| `diameter` | 0.05 | m | outer diameter of the tubes |
| `horizontal_pitch` | `2 d` / `sqrt(3) d` | m | spacing inside a row, a ring or a file |
| `vertical_pitch` | `2.5 d` / `sqrt(3) d` | m | spacing between rows, rings or files |
| `layout` | `staggered` | - | `staggered`, `grid`, `rings`, `radial` |
| `collection` | `distributor_collector` | - | `distributor_collector`, `reverse_return`, `central_header`, `two_level_rings` |
| `n_rings` | `None` | - | rings to build; `None` = as many as `p_v` fits |
| `n_files` | `None` | - | radial files; `None` = as many as `pi r_eff / p_v` |
| `duct_diameter` | `diameter` | m | headers, ducts and jumpers (go one size up in a real design) |
| `wall_clearance` | `2 d` | m | distance kept between the outermost riser and the wall |
| `azimuth_in` | 0 | deg | azimuth of the inlet nozzle |
| `azimuth_out` | mode dependent | deg | azimuth of the outlet nozzle |
| `elevation_in` | midway floor to distributor | m | inlet nozzle elevation (cold side) |
| `elevation_out` | collector (midway to the roof for `central_header`) | m | outlet nozzle elevation (hot side) |
| `split_mode` | `equal` | - | `equal` or `path` |

`validate()` returns the list of problems: blocking ones (no prefix) stop
`build_pipe_network` with a `ValueError`, and the ones prefixed `warning: ` build but
break a rule above (for example an inlet duct that crosses the active band).  Every
message names the parameter to change, not the symptom.

## 7. Two worked examples

Both run in the project's reference vessel (`radius = 2.0 m`, `base_z = 0.5 m`, wall
up to `7.9 m`, active band `0.7 .. 7.7 m`, i.e. the storage band of
`CylinderGeometry`), with 50 mm tubes:

```python
import numpy as np
from src.core.mesh import Mesh3D
from src.core.pipe_network import (COLLECTION_REVERSE, COLLECTION_TWO_LEVEL,
                                   LAYOUT_RINGS, LAYOUT_STAGGERED, PipeNetworkConfig,
                                   build_pipe_network)

mesh = Mesh3D(8.0, 8.0, 10.0, spacing=0.5)
vessel = dict(radius=2.0, height=7.4, base_z=0.5, band_bottom=0.2, band_top=7.2,
              diameter=0.05)
network = build_pipe_network(mesh, PipeNetworkConfig(
    layout=LAYOUT_STAGGERED, collection=COLLECTION_REVERSE, **vessel))
print(network.summary())
```

### 7.1 A staggered bundle with a reverse return

```text
network: staggered layout, reverse_return, 1512 risers in 1 distributor and 1 collector runs
pitch: 87 mm horizontal, 87 mm vertical (1.73 d / 1.73 d), d = 50.0 mm
bundle: 3.72 x 3.72 m plan inside r = 1.90 m of a 2.00 m vessel, risers 7.00 m long
tube surface: risers 1662.5 m2 over 10584 m, whole network 1705.1 m2 over 10855 m
specific area: 19.38 m2/m3 of active band (88.0 m3)
headers: distributor 0.70 m, collector 7.70 m, span 7.00 m - the 3 m rule: split into parallel modules above 3 m
flow: 1512 branches, split equal (0.0661%-0.0661%), path 145.46-145.46 m (spread 1.000x)
note: the two headers are 7.0 m apart: above 3 m the bundle must be split into identical parallel modules to keep the flow distribution uniform
note: the risers are 7.0 m long: the published practice keeps a module below 4 m of bed, so consider splitting the height into modules
```

Reading it: 1512 tubes of 7.00 m give 1662.5 m2 of wetted surface in 88.0 m3 of sand,
i.e. **19.4 m2/m3** - the sizing number of the design, and it comes straight from the
pitch (`1 / (p_h p_v)` metres of tube per m3).  The headers add 42.6 m2 and 271 m of
pipe, which is why they are runs of their own and not a foot-note in the model.
The sizing number comes straight from the pitch: one tube per `p_h x p_v` cell of the
lattice, so a full cross-section holds `1 / (p_h p_v)` metres of tube per m3 (133 m/m3
for this pitch; 120 m/m3 of *active band*, because the bundle fills 90% of the
cross-section).  The last two lines are the
design verdict: a 7 m band exceeds both the 3 m header rule and the 4 m module height,
so this bundle has to be built as two 3.5 m modules in parallel, each with its own
pair of headers.

The same bundle with the *direct* return shows what the collection mode is worth:

```text
flow: 1512 branches, split equal (0.0661%-0.0661%), path 13.39-278.47 m (spread 20.793x)
note: direct return: the branches run 13.39-278.47 m (1979.3% spread), so the shortest branch takes more flow; the reverse_return collection equalises the paths
```

The branch next to the nozzles travels 13.4 m and the last one 278.5 m - a factor
**20.8** in the path, so at equal diameters the far risers see far less flow.  The
reverse return takes the path spread to 1.000 and the `path` split to exactly
`1 / 1512` each.

### 7.2 Concentric rings with the two-level chain

```text
network: rings layout, two_level_rings, 926 risers in 15 distributor and 15 collector runs
pitch: 100 mm horizontal, 125 mm vertical (2.00 d / 2.50 d), d = 50.0 mm
bundle: 3.75 x 3.75 m plan inside r = 1.88 m of a 2.00 m vessel, risers 7.00 m long
tube surface: risers 1018.2 m2 over 6482 m, whole network 1048.9 m2 over 6677 m
specific area: 11.92 m2/m3 of active band (88.0 m3)
headers: distributor 0.70 m, collector 7.70 m, span 7.10 m - the 3 m rule: split into parallel modules above 3 m
flow: 926 branches, split equal (0.1080%-0.1080%), path 58.80-58.80 m (spread 1.000x)
```

Reading it: 15 rings, 926 risers, 1018.2 m2 of tube surface and therefore
**11.9 m2/m3** - 39% less surface than the lattice at the same vessel, because the
rings leave the axis (and the gaps between them) empty.  What the rings buy instead is
plumbing: 15 closed ring mains of one run each, 28 short radial jumpers, and a branch
path of 58.8 m against the 145.5 m of the staggered ladder.  The header span is 7.10 m
rather than 7.00 m because the raised level sits two duct diameters (0.1 m) above the
collector.  `PipeNetwork.validate()` returns `[]` for both examples, and
`path_spread()` is 1.000 in both - the ring chain is balanced by construction, tap by
tap.

## 8. Using a network

```python
from src.solver.fluid import FluidLoop

# the 1-D march of the gas: the risers, one branch each, with the network's split
result = FluidLoop(runs=network.risers, mass_flow=1.5, t_in=800.0,
                   split=network.split()).solve(mesh)

# the wetted area per cell, the same interface as rasterize_pipe
cells, length, area = network.voxelize()
assert np.isclose(area.sum(), network.total_area)   # the network lies inside the domain
```

The headers and the ducts are deliberately *not* part of that march: a header carries
the flow of many branches and its fluid temperature varies along it, which the
parallel-runs march cannot represent.  They are runs all the same, so a hydraulic check
can add their pressure drop
(`src/solver/fluid.py::pressure_drop` on each run, in parallel) and a geometry paint can
voxelise them.

## 9. What the tests check (`tests/test_pipes.py`)

* every riser sits inside `r_eff` and spans the active band exactly, for all four
  layouts and all four collections;
* the lattice is one lattice: rows `p_v` apart, risers `p_h` apart, the stagger exactly
  `p_h / 2`, and the closest pair of the bundle never tighter than the pitch;
* every ring carries risers, the rings are `k p_v` apart, and `validate()` refuses a
  ring that carries no riser or that is not connected to the duct;
* the wetted area is geometric and `sum(area) = pi d L_total` to machine precision,
  per run and over the network, on the mesh it was built on and on a finer one;
* the split sums to 1, is equal in `equal` mode and proportional to the paths in
  `path` mode - and the two coincide in the balanced collections;
* the 3 m rule is reported with the collector elevations, and the module height is
  flagged above 4 m;
* the nozzles cross the wall and stay under the roof; an outlet above the roof is
  refused with the parameter to change in the message;
* the same configuration always builds the same network (bit for bit), and the design
  translates with the vessel centre;
* the risers drive `FluidLoop` with the branch split, and the enthalpy of the air is
  the power into the bed.
