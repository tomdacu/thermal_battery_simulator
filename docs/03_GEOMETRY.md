# 3. Geometry model

`src/core/geometry.py`.  Three configuration dataclasses (`CylinderGeometry`,
`HeaterConfig`, `TubeConfig`) are combined by `BatteryGeometry`, which converts
them into the voxel arrays of a `Mesh3D` (`apply_to_mesh`).

## 1. Reference design

```
                     cone roof (steel shell, optional sand fill)
        ┌───────────────▲───────────────┐
        │        top insulation slab    │   z_slab_top_start .. z_slab_top_end
        ├───────────────────────────────┤
        │                               │
        │        STORAGE (sand,         │   z_storage_start .. z_storage_end
        │        packed bed, heaters,   │   r < r_storage
        │        tubes)                 │
        ├───────────────────────────────┤
        │      bottom insulation slab   │   base_z .. z_storage_start
        └───────────────────────────────┘
        radial insulation  r_storage .. r_insulation
        steel shell        r_insulation .. r_shell
        concrete foundation under the footprint
```

Elevations (all measured from the domain floor, `CylinderGeometry` properties):

```
base_z
  └─ z_storage_start      = base_z + insulation_slab_bottom
     └─ z_storage_end     = z_storage_start + height
        └─ z_slab_top_end = z_storage_end + insulation_slab_top
           └─ z_cone_base = z_slab_top_end + steel_slab_top
              └─ z_cone_apex = z_cone_base + r_shell·tan(roof_angle)
```

The conical roof radius at elevation $z$ is
$r_{cone}(z) = r_{shell}\left(1 - \frac{z - z_{cone\ base}}{h_{roof}}\right)$
with $h_{roof} = r_{shell}\tan\theta$; the steel shell of the roof has the same
radial thickness as the lateral shell.

## 2. Paint order (`apply_to_mesh`)

Later steps win, but the overlaps are explicit now:

| # | what | condition |
|---|---|---|
| 1 | air everywhere | reset of `material_id/k/rho/cp` |
| 2 | concrete foundation | `z < base_z` and `r ≤ r_shell + foundation_margin` |
| 3 | steel lateral shell | `base_z ≤ z < z_shell_top`, `r_insulation ≤ r < r_shell` |
| 4 | radial insulation | same band, `r_storage ≤ r < r_insulation` |
| 5 | bottom slab | `base_z ≤ z < z_storage_start`, `r < r_storage` |
| 6 | storage sand | `z_storage_start ≤ z < z_storage_end`, `r < r_storage` |
| 7 | top slab | `z_slab_top_start ≤ z < z_slab_top_end`, `r < r_storage` |
| 8 | steel plate under the roof | only if `steel_slab_top > 0` |
| 9 | cone: sand fill (optional) then steel shell | `z_cone_base ≤ z ≤ z_cone_apex` |
| 10 | discrete heaters | element mask ∩ storage band → `material_id = HEATERS` + `source_mask` |
| 11 | tubes | element mask ∩ storage band → `material_id = TUBES` (steel), sources cleared |
| 12 | domain boundary conditions | see §5 |

Bands are half-open `[start, end)`, so no cell is painted twice by two adjacent
bands.  **A zone thinner than a cell is widened to one cell** (growing away from
the storage): the 20 mm shell, the 5 mm steel plate and the conical roof would
otherwise contain no cell centre and disappear from the model entirely - the roof
used to vanish at every realistic cell size.  The **concrete foundation is painted over the footprint only** (it used
to cover the whole X–Y plane, silently adding a lateral conduction path across
the entire domain floor).

## 3. Heaters: hairpin (U-shaped) sheathed elements

A discrete heater pattern is a **bank of flanged hairpin elements** built by
`HeaterConfig.bank(...)` and rasterised by `src/core/heaters.py`.  One element is two
vertical legs `leg_spacing` apart (Ø12 mm stainless sheath by default) joined by a
180 deg bend at the bottom, hanging from a flange above the roof and resting on a
support plate; the *active length* is inside the sand, the rest of the leg (the cold
shank) crosses the insulation.

**Rating.**  Power is set by the surface power density
$P_s = P / (\pi d (L_{active} + \tfrac{\pi}{2} b))$ in W/cm², the way sheathed
elements are actually rated: 3-8 W/cm² for a solid medium.  Outside that window the
build continues but a warning is reported (`BatteryGeometry.heater_warnings`).

**Rasterisation.**  Every segment of every element (two legs plus `bend_chords`
chords for the bend) marks the cells it crosses *and* the cells within half a local
cell of its axis, so an element thinner than a cell still forms an unbroken chain
along its length.  Power is deposited on the cells of the active length only:
$Q = P / \sum V_{active}$, hence $\sum Q V = P$ exactly on any grid.

**Rejections** (`validate_bank`, errors unless marked as warnings): legs sharing
cells, two elements sharing cells, an element reaching outside the storage wall, a
collision with a heat-exchanger tube, no cell covered at all, `leg_spacing` not
greater than the sheath diameter, offsets leaving no room in the storage band.
Warnings: the legs are within one cell (the mesh sees a rod, not a U), the sheath is
thinner than a cell, surface power outside 3-8 W/cm².

| pattern | layout |
|---|---|
| `uniform_zone` (default) | no discrete element: the power is spread over the whole storage band, honouring `offset_bottom/offset_top` |
| `grid_vertical`, `chess_pattern` | `grid_rows × grid_cols` hairpins |
| `radial_array`, `spiral`, `concentric_rings` | hairpins on `n_rings` concentric rings |
| `custom` | a square arrangement sized on `custom_positions` |

**Power normalisation.**  Whatever the pattern, the injected power is exactly
`power_total`: the source density is $Q = P/\sum V$ over the cells that were painted,
so $\sum Q V = P$ (`tests/test_heaters.py`).

## 4. Tube patterns

`TubeConfig.generate_positions(...)`, also pure: `central_cluster`, `radial_array`
(default), `grid`, `hexagonal`, `single_central`, `custom`.

* Tube cells are painted **inside the storage band only**: the tubes no longer
  replace insulation-slab cells, which used to short-circuit the insulation
  exactly where they crossed it.
* Tube cells take the **shell material** (steel) rather than the packed-bed
  properties, so the conduction path along the tube is metal.
* Any volumetric source on a tube cell is cleared: a tube is not a heater.
* The fluid coupling is the internal convection of
  [02](02_FDM_DISCRETIZATION.md) §5.4, set from `h_fluid`/`t_fluid` (Kelvin).

## 5. Boundary conditions applied by the geometry

`BatteryGeometry.apply_boundary_conditions`:

* lateral faces (`x_min/x_max/y_min/y_max`): convection with `h_lateral`,
  `t_ambient`;
* top face (`z_max`): convection with `h_top`, `t_ambient`;
* bottom face (`z_min`): fixed temperature `t_ground`;
* the emissivity of the shell material is recorded on the air-exposed faces for
  the optional radiation model.

Temperatures are Kelvin (`BatteryGeometry.t_ambient = 293.15`,
`t_ground = 283.15` by default).

## 6. Validation

`BatteryGeometry.validate(mesh)` returns a list of problems and `apply_to_mesh`
raises `ValueError` on a non-empty list.  It checks:

* positive radius, height and insulation thickness, non-negative shell and slabs;
* packing fraction in (0, 1);
* **the battery fits in X and Y** (`center ± r_shell` inside the domain);
* **the roof apex fits below `Lz`** - the old code silently cut the roof and left
  a hole over the storage, losing heat straight through the top insulation.

The mesh itself validates its fields (`Mesh3D.validate`): positive $k,\rho,c_p$,
non-negative sources, sinks ≤ 0, and `check_kelvin` on the temperature field and
the face conditions.

## 7. Reporting

`BatteryGeometry.zone_volumes()` / `zone_masses()` return a consistent key set
(`storage`, `slab_bottom`, `slab_top`, `insulation_radial`, `insulation`,
`shell`, `cone_shell`, `steel_slab`, `foundation`, `total`) computed from the
same formulas as the paint masks - the previous key mismatch (`sand_total` /
`insulation`) raised `KeyError` in `estimate_energy_capacity`.
`estimate_energy_capacity(t_high, t_low, efficiency)` returns the stored and
usable energy of the storage region in joules and kWh plus the sand mass.

`apply_to_mesh` returns a `BuildReport` with the zone volumes/masses, the number
of source and tube cells, the number of elements, and notes (grid snapping, ...)
that the GUI writes to its log.
