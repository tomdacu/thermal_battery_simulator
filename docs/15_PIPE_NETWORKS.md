# 15. Pipe networks of a cylindrical vessel

Date: 2026-09-20.  Code: `src/core/pipe_network.py` (builds on `src/core/pipes.py`),
tests: `tests/test_pipes.py`, GUI: the *Pipes* tab of `gui/views/geometry_panel.py`
wired through `gui/controller.py` and `gui/main_window.py`.

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
   riser, summing to 1, either equal (`equal`, the design target), proportional to the
   branch path (`path`, the coarse rule a header is sized against), or grouped
   (`ring`, `sector` - see section 8).
4. **The paint never measures anything.**  `PipeNetwork.paint()` marks the cells the
   centrelines cross, but the wetted area it reports stays the geometric `pi d L` of
   section 2: what falls outside the vessel (the nozzle stubs) is *subtracted*, not
   ignored, so `painted + dropped = total_area` to machine precision.

## 3. Layouts

The plan inside the circle of radius `radius - wall_clearance =: r_eff`.  Two pitches
describe every layout: `p_h` between neighbouring risers *inside* a row, a ring or a
radial file, and `p_v` *between* rows, rings or files.

| layout | plan | risers |
|---|---|---|
| `staggered` | equilateral lattice on one origin, alternate rows shifted by `p_h/2` | the bundle the literature recommends |
| `grid` | square lattice on the same rows, all columns lined up | the easy one to fabricate |
| `rings` | concentric rings at `r = k p_v`, each carrying its own risers, linked by radial jumpers | a ring main per circle; the centre stays free for a duct |
| `spiral` | one Archimedean spiral `r(phi) = p_v phi / 2 pi`, walked from the wall inwards | one feeder that doubles as the distributor; no jumper and no ring main at all |
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
* the ring radii are `k p_v`, so the innermost ring is one pitch off the axis;
* the spiral steps `p_h` of *arc* at a time and gains `p_v` of radius per full turn,
  starting on the clearance circle (where the inlet duct arrives, so the first branch
  is the one the nozzle sees) and stopping half a horizontal pitch short of the axis,
  which leaves the centre free for the return duct.  The turns are exactly `p_v` apart
  radially, and the diagonal between neighbouring turns comes no closer than
  `0.5 min(p_h, p_v)`: the tubes of a spiral never touch, but the packing is a few per
  cent tighter than a lattice with the same two pitches.

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
| nozzles | through the lateral wall, inlet below the distributor, outlet at or above the collector | the roof carries the insulation and the cone: an outlet through it is refused.  The defaults are the midway points, clamped so that the duct fits over the floor and under the roof |
| reverse return | outlet nozzle opposite the inlet | the return has to leave at the far end of the chain |
| tube-header junction | the mesh band of one tube diameter either side of the two header elevations is refined to `junction_refinement` | the gas turns 90 deg there and the surface is singular: it is the one place in the bundle where the cell size decides the local wall temperature |
| lagged headers | `insulated_headers` when the ducts are wide (`d_duct >= 2 d`) and the bed taller than 3 m | a bare header pair that wide exchanges with the bed along the whole span, which is surface outside the bundle |

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
| `diameter` | 0.05 | m | **outer** diameter of the tubes: the pitches, the clearance and the wetted area follow it |
| `wall_thickness` | 0.002 | m | wall of the tube: the bore `diameter - 2 t` is what the gas flows through (section 7) |
| `material` | `stainless_steel` | - | tube wall as a label and a roughness: `stainless_steel` (drawn, 15 um) or `carbon_steel` (commercial, 46 um) |
| `roughness` | `None` | m | absolute roughness of the wall; `None` = the material's own value |
| `horizontal_pitch` | `2 d` / `sqrt(3) d` | m | spacing inside a row, a ring, a file or along the spiral |
| `vertical_pitch` | `2.5 d` / `sqrt(3) d` | m | spacing between rows, rings, files or spiral turns |
| `layout` | `staggered` | - | `staggered`, `grid`, `rings`, `spiral`, `radial` |
| `collection` | `distributor_collector` | - | `distributor_collector`, `reverse_return`, `central_header`, `two_level_rings` |
| `n_rings` | `None` | - | rings to build; `None` = as many as `p_v` fits |
| `n_files` | `None` | - | radial files; `None` = as many as `pi r_eff / p_v` |
| `duct_diameter` | `diameter` | m | headers, ducts and jumpers (go one size up in a real design) |
| `wall_clearance` | `2 d` | m | distance kept between the outermost riser and the wall |
| `azimuth_in` | 0 | deg | azimuth of the inlet nozzle (and the start of the spiral, the sector fan and the radial file) |
| `azimuth_out` | mode dependent | deg | azimuth of the outlet nozzle |
| `elevation_in` | midway floor to distributor | m | inlet nozzle elevation (cold side) |
| `elevation_out` | collector (midway to the roof for `central_header`) | m | outlet nozzle elevation (hot side) |
| `split_mode` | `equal` | - | `equal`, `path`, `ring` or `sector` (section 8) |
| `n_sectors` | 4 | - | angular sectors of the `sector` distribution |
| `insulated_headers` | `False` | - | `True`: the distributor and the collector are lagged and exchange nothing with the bed (section 9) |
| `junction_refinement` | `None` | m | cell size asked for the band around the two header elevations; `None` = no band (section 10) |

`validate()` returns the list of problems: blocking ones (no prefix) stop
`build_pipe_network` with a `ValueError`, and the ones prefixed `warning: ` build but
break a rule above (for example an inlet duct that crosses the active band, a relative
roughness outside the Moody chart, a junction refinement coarser than the tube, or a
sector nobody feeds).  Every message names the parameter to change, not the symptom,
and every option below is reported by `summary()` and checked by `validate()`.

## 7. The tube, the wall and the material

A pipe is not a line, and the difference shows up in exactly one place: the gas side.

* `diameter` stays the **outer** diameter.  The pitches are multiples of it, the wall
  clearance, the bundle radius and the wetted area `pi d L` are made of it, and
  `total_area` / `specific_area` are the numbers the bed sees;
* `wall_thickness` gives the **bore** `d_in = diameter - 2 t`, and the same wall on the
  headers (`d_duct - 2 t`).  The gas flows in the bore, so the bore carries the
  velocity, the Reynolds number, the friction factor and the film coefficient
  (`pipe_h`), and `PipeNetwork.fluid_loop` hands the march the risers *as the gas sees
  them*: bore diameter, inner wetted area `pi d_in L`;
* `material` is a label **and** a roughness: `stainless_steel` is a drawn tube
  (eps = 15 um), `carbon_steel` a commercial one (eps = 46 um, the value
  `src/solver/fluid.py` uses by default).  The keys are the keys of
  `src.core.materials.STRUCTURAL_MATERIALS`, so `paint()` writes the thermal properties
  of the material the label names;
* `roughness` overrides the wall roughness in metres when the tubes are old, fouled or
  lined.  `relative_roughness = eps / d_in` is what `friction_factor` is read with, and
  `validate()` warns above 0.05 (the edge of the Moody chart).

```python
from src.core.pipe_network import PIPE_CARBON, PipeNetworkConfig

config = PipeNetworkConfig(diameter=0.05, wall_thickness=0.0025, material=PIPE_CARBON)
config.inner_diameter          # 0.045 m - what the gas flows through
config.absolute_roughness      # 4.6e-05 m, from the material
config.relative_roughness      # 0.00102 = eps / d_in
config.pipe_material           # "carbon steel, commercial (eps 46 um)"
```

On the reference bundle (50 mm tubes, 2 mm wall) that is a bore of 46 mm: 8% less area
than the outer `pi d L`, and 21% more pressure drop per metre than a 50 mm bore, which
is why the wall is a parameter and not a detail.

## 8. Distributions

`split()` returns one fraction of the total mass flow per branch, summing to 1, and it
is what `FluidLoop(split=...)` marches.  Four rules:

| mode | what it divides | when |
|---|---|---|
| `equal` | the flow, equally between the branches | the design target: every riser sees the same flow |
| `path` | proportionally to the branch path | the coarse hydraulic rule a ladder header is sized against: *if the branches really saw the same pressure drop*, the longest one would carry the most |
| `ring` | equally between the **ring mains**, then equally inside each ring | a manifold per ring: each ring main is a take-off with its own share, and a ring with few taps gives each of its tubes more.  Needs `layout=rings` (the per-ring split has no meaning on a ladder, and `validate()` says so) |
| `sector` | equally between `n_sectors` angular sectors about `azimuth_in`, then equally inside the sector | a valve (or a duct) per quadrant: the bed is charged the same in every direction, whatever the lattice does at its edges.  A sector with no riser is refused with the parameter to change |

The two grouped modes are plumbing choices: they say *which pipe gets the flow*, not
what the pressure drop does.  The two hydraulic modes (`equal`, `path`) are the ones a
header is sized with, and in a balanced (`reverse_return`, `two_level_rings`)
collection the path-weighted split *is* the equal one - see section 4.

## 9. Painting the mesh

`PipeNetwork.paint(mesh, h_fluid=..., t_fluid=...)` turns the drawing into cells:

* the centrelines are re-rasterised on `mesh` (the same accounting as `voxelize`) and
  every cell whose **centre** is inside the vessel - `r <= radius`,
  `base_z <= z <= roof_z` - becomes `MaterialID.TUBES` and **keeps the properties it
  had** (the bed's): a pipe of a few centimetres is a thin wall inside a cell of sand.
  Until 2026-09-23 the whole cell was painted with the tube material, which put ~30 times
  the real steel of the default network into the bed (1.6 m³ for 0.05 m³ of wall);
* a tube cell is never a heater: its volumetric source is cleared and its
  `source_mask` bit dropped;
* every cell that exchanges is marked convective, and the gas loop writes its film per
  cell in every analysis ([12](12_METHODS.md) §11): the risers always, the headers and
  the ducts only when they are not lagged, so `insulated_headers=True` leaves the header
  cells with no film.  The paint writes no film value by default (`h_fluid=0`); a fixed
  `h_fluid`/`t_fluid` is the lumped tube model and is kept only for a caller that asks;
* the report says what happened and keeps the area honest:

```text
painted: 18,795 cells as tubes (18,312 of them on a riser), 506.33 m2 of wetted area,
film 300 W/(m2 K) at 26.9 degC
material: carbon_steel; outside the vessel: 0.126 m2; lagged: 42.54 m2
```

`report.area + report.dropped == network.total_area` to machine precision: what fell
outside the wall (the nozzle stubs, which the voxel mask is *supposed* to show crossing
the wall) is subtracted and named, never silently dropped.  The mask itself is never
measured: `total_area` and `riser_area` stay the geometric `pi d L` of the centrelines.

The film the paint writes is what a **steady** or **losses** run sees (a lumped
exchanger where the pipes are).  A **transient** driven by a `FluidLoop` does not use
it: the loop marches the gas itself and deposits `q_fluid` in exactly those cells, and
the solver rebuilds its operators when the film changes - so the two models never
speak at the same time.

## 10. The junction band and the mesh

`PipeNetworkConfig.junction_bands()` returns, for `junction_refinement = t`:

```text
[ (z_bottom - d, z_bottom + d, t), (z_top - d, z_top + d, t) ]
```

clipped to the vessel, where `d` is the tube diameter.  That is one band per header
elevation and the reason is local: the tube-header junction is where the gas turns 90
degrees and where the wetted surface is singular, so it is the one place in the bundle
where the cell size decides the wall temperature the bed sees.  A graded mesh adds
these bands as `Band(low, high, t)` of `GridSpec.z`; on a uniform mesh the option is a
promise the grid cannot keep, and `validate()` warns when `t` is coarser than the tube
it is meant to resolve.

## 11. From the GUI to the solver

The *Pipes* tab of the Geometry panel owns the network; the window owns the mesh, and
the mesh is what a run solves.  The wiring is therefore:

1. **Build mesh** (`gui/main_window.py`) builds the geometry and the grid; changing it
   forgets the painted network, because its cells are gone.
2. **Build network and paint it on the mesh** (Pipes tab) asks the window - through the
   `pipe_network_requested` signal - to build the network from the tab's widgets and
   paint it on *that* mesh, and prints the summary, the paint report and the validation
   problems back into the tab.  A successful paint switches the lumped tube bank of the
   *Tubes* tab off: the network is the heat exchanger now, and two exchangers would
   count the surface twice.
3. **Circuit flow [kg/s]** and **Gas h / Gas T** live in the same tab: the flow is the
   operating point the loop is marched at, the film and the gas temperature are what
   the paint writes for the runs that do not use a loop.
4. **Run** assembles a `RunConfig` carrying the network and the flow.  For a transient
   the controller builds the loop on the run mesh
   (`network.fluid_loop(flow, mesh=mesh)`) and hands it to
   `TransientConfig(fluid_loop=...)`:

   * the resistors of the power profile become the *external power* of the loop, so the
     gas is what heats the bed (the published architecture), and the exchanger of the
     extraction profile is the negative external power;
   * the painted cells receive `q_fluid` and their film is cleared by the solver, which
     rebuilds its operators when the film changes;
   * the controller logs the circuit (`hydraulics(flow).summary()`) next to the run.
5. **Steady** and **losses** runs do not march a loop: they see the painted cells with
   the film of the tab, which is the lumped exchanger model the project has always had.

`Junction refinement [m]` in the Pipes tab feeds the *Mesh* tab: the band of section 10
is added to the `GridSpec` of a graded mesh, so the junction refinement is in place
before the network is built (the network is built *on* the mesh).

## 12. Two worked examples

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

### 12.1 A staggered bundle with a reverse return

```text
network: staggered layout, reverse_return, 1512 risers in 1 distributor and 1 collector runs
pitch: 87 mm horizontal, 87 mm vertical (1.73 d / 1.73 d), d = 50.0 mm
tube: 50.0 x 2.0 mm wall, bore 46.0 mm, stainless steel, drawn (eps 15 um), relative roughness 0.00033
bundle: 3.72 x 3.72 m plan inside r = 1.90 m of a 2.00 m vessel, risers 7.00 m long
tube surface: risers 1662.5 m2 over 10584 m, whole network 1705.1 m2 over 10855 m
specific area: 19.38 m2/m3 of active band (88.0 m3)
headers: distributor 0.70 m, collector 7.70 m, span 7.00 m - the 3 m rule: split into parallel modules above 3 m
flow: 1512 branches, split equal (0.0661%-0.0661%), path 145.46-145.46 m (spread 1.000x)
bare headers: no lagging, so 2.4% of the wetted area sits in the 2 distributor and collector runs and reaches the sand unlagged
note: the two headers are 7.0 m apart: above 3 m the bundle must be split into identical parallel modules to keep the flow distribution uniform
note: the risers are 7.0 m long: the published practice keeps a module below 4 m of bed, so consider splitting the height into modules
```

Reading it: 1512 tubes of 7.00 m give 1662.5 m2 of wetted surface in 88.0 m3 of sand,
i.e. **19.4 m2/m3** - the sizing number of the design, and it comes straight from the
pitch (`1 / (p_h p_v)` metres of tube per m3).  The headers add 42.6 m2 and 271 m of
pipe, which is why they are runs of their own and not a foot-note in the model - and
why the summary says out loud that 2.4% of that wetted area reaches the sand through a
bare header rather than through the bundle (section 9).
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

### 12.2 Concentric rings with the two-level chain

```text
network: rings layout, two_level_rings, 926 risers in 15 distributor and 15 collector runs
pitch: 100 mm horizontal, 125 mm vertical (2.00 d / 2.50 d), d = 50.0 mm
tube: 50.0 x 2.0 mm wall, bore 46.0 mm, stainless steel, drawn (eps 15 um), relative roughness 0.00033
bundle: 3.75 x 3.75 m plan inside r = 1.88 m of a 2.00 m vessel, risers 7.00 m long
tube surface: risers 1018.2 m2 over 6482 m, whole network 1048.9 m2 over 6677 m
specific area: 11.92 m2/m3 of active band (88.0 m3)
headers: distributor 0.70 m, collector 7.70 m, span 7.10 m - the 3 m rule: split into parallel modules above 3 m
flow: 926 branches, split equal (0.1080%-0.1080%), path 58.80-58.80 m (spread 1.000x)
bare headers: no lagging, so 2.8% of the wetted area sits in the 30 distributor and collector runs and reaches the sand unlagged
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

## 13. Using a network

The gas circuit is one call:

```python
# the loop the solver marches: one run per branch, the branch split, and the
# pressure drop of the headers, the connectors and the ducts folded in
loop = network.fluid_loop(mass_flow=1.5, t_in=800.0)
result = loop.solve(mesh)
assert np.isclose(result.power, -loop.fluid.cp * 1.5 * (result.t_out - result.t_in))

print(network.hydraulics(1.5).summary())
# circuit at 1.5000 kg/s: mean branch drop 19.7 Pa (95% of it in the headers, the
# connectors and the 5.9 m of duct), spread 1.000x, equivalent fittings K = 126.27
```

* **the runs are the branches, not the pipes.**  A header is traversed by every branch
  and its fluid temperature varies along it, so it cannot be one *parallel* run without
  taking flow away from the tubes: the parallel march of
  `src/solver/fluid.py` has one run per branch, and the branch's tube is what it
  marches.  The runs carry the bore and the inner wetted area (section 7), so the film
  coefficient, the friction and the exchange surface are all on the gas side;
* **the rest of the circuit is in the loop as hydraulics.**  `hydraulics(mass_flow)`
  gives the pressure drop of every branch - its tube *plus* its arcs in the two headers
  and the two ducts every branch crosses - and the equivalent fitting coefficient
  `K = 2 dp_shared / (rho v_bore^2)` that `FluidLoop(fittings_k=...)` adds to each run.
  The loop's `delta_p`, `fan_power` and circulation figures are therefore the whole
  circuit's, not the tubes' alone, and `result.delta_p == hydraulics(flow).mean_drop` to
  machine precision.  Pass `fittings_k=` to add the local losses of the plant (bends,
  valves) on top;
* `mesh=` re-rasterises the risers on the grid the solve will use (the network may have
  been built on another one - the GUI always passes its run mesh);
* **wetted area per cell**, the same interface as `rasterize_pipe`:

```python
cells, length, area = network.voxelize()
assert np.isclose(area.sum(), network.total_area)
```

The headers and the ducts are runs all the same: the hydraulic report above prices
them, `voxelize()` rasterises them and `paint()` marks them on the mesh.

## 14. Worked example: a spiral, a sector split, and an hour of charging

The reference case: the storage band of `create_small_test_geometry()` (2 m radius,
0.2 m bottom slab, 4 m of sand, 4.5 .. 4.7 m of top slab), 50 mm carbon-steel tubes
with a 2.5 mm wall, a 120 mm duct, a spiral bundle with the balanced collection, lagged
headers, four sectors and a 50 mm junction band.  The script below is exactly what was
run; the numbers are its output.

```python
from src.core.geometry import create_small_test_geometry
from src.core.mesh import Mesh3D
from src.core.pipe_network import (COLLECTION_REVERSE, LAYOUT_SPIRAL, PIPE_CARBON,
                                   SPLIT_SECTOR, PipeNetworkConfig, build_pipe_network)
from src.core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from src.solver.steady import SolverConfig
from src.solver.transient import TransientConfig, TransientSolver

mesh = Mesh3D(6.0, 6.0, 5.6, spacing=0.1)
battery = create_small_test_geometry()
battery.apply_to_mesh(mesh)
cyl = battery.cylinder
config = PipeNetworkConfig(
    radius=cyl.r_storage, height=cyl.z_cone_base - cyl.base_z, base_z=cyl.base_z,
    band_bottom=cyl.z_storage_start - cyl.base_z, band_top=cyl.z_storage_end - cyl.base_z,
    diameter=0.05, wall_thickness=0.0025, material=PIPE_CARBON, duct_diameter=0.12,
    layout=LAYOUT_SPIRAL, collection=COLLECTION_REVERSE, insulated_headers=True,
    junction_refinement=0.05, split_mode=SPLIT_SECTOR, n_sectors=4)

network = build_pipe_network(mesh, config)
print(network.summary())
report = network.paint(mesh, h_fluid=300.0, t_fluid=300.0)
print(report.summary())
print(network.hydraulics(0.4).summary())

# a closed, insulated box: the example is about the loop, not about the envelope
for face in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
    mesh.set_adiabatic(face)
mesh.T[:] = 293.15
solver = TransientSolver(mesh, TransientConfig(
    t_final=3600.0, dt=900.0, save_interval=900.0,
    initial_condition=InitialCondition(mode="uniform", t_uniform=293.15),
    power_profile=PowerProfile(mode="constant", constant_power=20000.0),
    extraction_profile=ExtractionProfile(mode="off"),
    fluid_loop=network.fluid_loop(0.4, mesh=mesh),
), SolverConfig(method="cg", tolerance=1e-8))
results = solver.run()
```

```text
network: spiral layout, reverse_return, 909 risers in 1 distributor and 1 collector runs
pitch: 100 mm horizontal, 125 mm vertical (2.00 d / 2.50 d), d = 50.0 mm
tube: 50.0 x 2.5 mm wall, bore 45.0 mm, carbon steel, commercial (eps 46 um), relative roughness 0.00102
bundle: 3.69 x 3.65 m plan inside r = 1.90 m of a 2.00 m vessel, risers 4.00 m long
tube surface: risers 571.1 m2 over 3636 m, whole network 641.2 m2 over 3822 m
specific area: 12.76 m2/m3 of active band (50.3 m3)
headers: distributor 0.50 m, collector 4.50 m, span 4.00 m - the 3 m rule: split into parallel modules above 3 m
flow: 909 branches, split sector (0.1068%-0.1147%), path 99.23-99.23 m (spread 1.000x)
distribution: 4 sectors about the inlet azimuth 0 deg, each fed the same share of the flow
insulated headers: the 2 header runs carry the gas and exchange nothing, so the heat transfer surface is 572.9 m2 of 641.2 m2
junctions: the mesh band around the two header elevations asks for cells of 50.0 mm (one tube diameter either side of 0.50 and 4.50 m)
note: the two headers are 4.0 m apart: above 3 m the bundle must be split into identical parallel modules to keep the flow distribution uniform
painted: 36,219 cells as tubes (34,960 of them on a riser), 641.05 m2 of wetted area, film 300 W/(m2 K) at 26.9 degC
material: carbon_steel; outside the vessel: 0.151 m2; lagged: 68.31 m2
note: 0.151 m2 of pipe fell outside the vessel (the nozzle stubs): it is not painted, because a cell outside the wall is not part of the bed
note: 68.31 m2 of lagged header was painted as a pipe but carries no gas film: it only conducts in the sand
circuit at 0.4000 kg/s: mean branch drop 0.4 Pa (36% of it in the headers, the connectors and the 4.6 m of duct), spread 1.073x, equivalent fittings K = 4.59
```

(the paint report repeats the network's notes; one of them is shown once here)

and the transient, printed from `results`:

```text
time               15      30      45      60   min
bed degC       20.016  20.044  20.083  20.131
pipes degC   23.8 leaving the loop, 21.4 at the hottest pipe cell
energy       20.00 kWh in, 15.00 kWh kept by the bed, 0.0002 kWh of fan work
```

Reading it:

* **the design numbers.**  909 risers of 4.00 m give 571.1 m2 of tube surface in the
  50.3 m3 of sand - `riser_area / bed_volume` is 11.4 m2/m3 - and the whole network
  641.2 m2 over the same band is the `specific area` of the summary, **12.76 m2/m3**.
  The two lagged headers hold the other 68.3 m2, i.e. 11% of the wetted area of the
  network that the bed never sees (section 9).  The spiral packs a little tighter than
  a lattice at the same two pitches (section 3), which is where the extra square metres
  come from;
* **the flow.**  `sector` gives every one of the four sectors 25% of the 0.4 kg/s, and
  each sector's 227-228 risers divide that: hence the 0.1068%-0.1147% band.  The fan of
  tubes is *not* fed equally, it is fed equally **per sector** - a plumbing decision,
  which is why the distribution is a parameter and not a constant.  The paths are equal
  to machine precision (spread 1.000x) because the collection is the balanced one: on
  this geometry the two knobs are independent;
* **the circuit.**  At 0.4 kg/s a branch drops 0.4 Pa, 36% of it in its arcs and in the
  4.6 m of duct every branch crosses.  Dividing 0.4 Pa of gas friction between 909
  laminar tubes is why the whole hour costs a fraction of a watt-hour of fan work
  against 20 kWh of heat: the bore is chosen for the *plant*, not for the solver;
* **the charge.**  20 kW through the gas for an hour lifts the bed mean by 0.13 K.  The
  loop balance fixes the gas inlet: `t_in = t_out + Q / (m cp)`, i.e. about 50 K hotter
  than the leaving gas, so the resistors heat the air to ~74 degC, the air gives it
  back through 573 m2 of tube and leaves at 23.8 degC - still 3.7 K above the sand it
  just passed, which is what NTU ~ 1.4 looks like: neither transparent
  (area-limited) nor fully equilibrated (flow-limited);
* **the energy closes.**  The samples run from `t = 15 min` to `t = 60 min`, and the
  stored energy rises by 15.00 kWh over exactly that window - 20 kW for three quarters
  of an hour - against 20.00 kWh of electricity for the whole hour and 0.0002 kWh of
  blower work.  Nothing is lost, hidden or created: the profile's power goes into the
  gas, the gas puts it in the pipes, the pipes put it in the sand;
* **the local picture.**  The hottest pipe cell sits 1.4 K above the bed mean: that is
  the local wall temperature the junction band of section 10 exists to resolve, and the
  reason a 0.1 m cell is the coarsest grid on which this design means anything.

## 15. What the tests check (`tests/test_pipes.py`)

* every riser sits inside `r_eff` and spans the active band exactly, for all five
  layouts and all four collections;
* the lattice is one lattice: rows `p_v` apart, risers `p_h` apart, the stagger exactly
  `p_h / 2`, and the closest pair of the bundle never tighter than the pitch;
* the spiral is an Archimedean one: the walk starts on the clearance circle, winds in
  one way only, `dr` and `dphi` keep the relation `dphi = 2 pi dr / p_v`, the steps are
  one horizontal pitch of arc and the turns never come closer than half the pitch;
* every ring carries risers, the rings are `k p_v` apart, and `validate()` refuses a
  ring that carries no riser or that is not connected to the duct;
* the wetted area is geometric and `sum(area) = pi d L_total` to machine precision,
  per run and over the network, on the mesh it was built on and on a finer one;
* the wall thickness gives the bore the gas flows in (`d - 2 t`, identical to the
  duct's), the geometry keeps the outer diameter, and a thicker wall is a smaller,
  relatively rougher pipe;
* the material switch moves the absolute and the relative roughness and the summary
  reports both; a roughness beyond the Moody chart is a warning, not a silent number;
* the split sums to 1 in all four modes; the `sector` mode gives every sector
  `1 / n_sectors` of the flow and the `ring` mode every ring main
  `1 / n_groups`, with the taps of a ring dividing that share - and an empty sector is
  refused with the parameter to change;
* `paint()` marks only cells whose centre is inside the vessel, writes `TUBES` and keeps
  the bed's properties, clears the sources (rescaling the lumped source over the rest),
  marks the riser cells convective and the lagged header cells not, and keeps
  `painted + dropped = total_area`;
* `hydraulics()` adds the tube to the arcs and the ducts, prices the shared pipes with
  the bore, and its `mean_drop` is exactly the loop's `delta_p` (fittings included);
* `fluid_loop()` marches one run per branch with the network's split and the bore's
  film, and the enthalpy of the gas is the power into the bed;
* the junction bands sit on the two header elevations with the requested target;
* the 3 m rule is reported with the collector elevations, and the module height is
  flagged above 4 m;
* the nozzles cross the wall and stay under the roof; an outlet above the roof is
  refused with the parameter to change in the message;
* the same configuration always builds the same network (bit for bit), and the design
  translates with the vessel centre.
