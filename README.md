# Thermal Battery Simulator

[![CI](https://github.com/tomdacu/thermal_battery_simulator/actions/workflows/ci.yml/badge.svg)](https://github.com/tomdacu/thermal_battery_simulator/actions/workflows/ci.yml)
![licence: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/licence-PolyForm%20Noncommercial%201.0.0-blue)
![python 3.10 | 3.12](https://img.shields.io/badge/python-3.10%20%7C%203.12-blue)

![Thermal Battery Simulator](assets/banner.png)

**A 3D simulator of sand batteries: the bed, the envelope and the gas circuit that
charges and discharges it, solved together.**

A sand battery stores electricity as heat in a vessel of sand or crushed rock: electric
resistors heat the gas of a closed circuit, the gas runs through pipes buried in the bed
and leaves its heat there, and on discharge an exchanger on the same circuit takes it
back out - the architecture of the Polar Night Energy plants.  This code answers the
questions a designer asks of such a plant: what it loses in standby, how the bed charges
and discharges, what the gas temperatures are, how the flow divides between the tubes,
how the headers must be sized, what the fan costs and where the heat goes.

Version **2.1.0** (2026-09-29): the interface runs on **PySide6** (LGPL) and the project
is licensed under the **PolyForm Noncommercial License 1.0.0** - see
[Licence](#licence).  Version **2.0.0** (2026-09-24) was: an anisotropic mesh, the tube
coupled to its cell by a well model, the whole gas circuit with its hydraulics, a
header-sizing engine, a packed bed that conducts like a hot one, and the mesh coupled to
the model it is built from.

## Contents

- [What it models](#what-it-models)
- [How it solves](#how-it-solves)
- [Installation](#installation)
- [Quick start](#quick-start)
- [The default plant and what it takes to run](#the-default-plant-and-what-it-takes-to-run)
- [Documentation](#documentation)
- [Project structure](#project-structure)
- [Figures](#figures)
- [Development](#development)
- [Licence](#licence)

---

## What it models

**The vessel.**  A cylindrical bed of sand or crushed rock (steatite, silica sand,
olivine, basalt, magnetite, quartzite, granite) inside a radial insulation, a steel
shell, insulation slabs under and over the bed, an optional steel plate and conical
roof, on a concrete pad over a layer of soil.

**The bed.**  A packed bed of grains with air in the voids.  Its conductivity is the
**Zehner-Bauer-Schlünder** model of the VDI Heat Atlas with the **radiation between the
grains**, evaluated at the temperature of every cell: 0.30 W/(m K) at 20 °C, 0.57 at
500 °C for 1 mm steatite at 63 % packing.  The capacity is the grains' and the air's by
volume.

**The plant.**  The resistors heat the gas of a closed circuit (air, nitrogen or steam,
at a set mass flow and pressure); the gas runs through the **whole buried network** -
inlet duct, distributor, risers, collector, outlet duct - and gives its heat to the bed
through every wall it touches.  Five layouts (staggered bundle, square grid, concentric
rings, radial files, spiral) and five collection modes (direct, reverse return, central
header, two-level rings, radial manifold).  The same circuit discharges the bed through
an exchanger at a set power or a set return temperature.  Pressure drop and fan power are
part of the ledger.

**The hydraulics, and the header engine.**  How the flow divides between the risers is
what the network's pressures give: Darcy-Weisbach in every pipe, the tees' losses and the
draught of the hot gas, solved by Newton on the node pressures.  The headers are not set
by hand: an **engine** sizes the ring mains, trunks and ducts from the nominal pipe sizes
so every riser carries the share of the flow equal to the share of the bed it serves,
under a velocity limit, growing them while they pay for themselves in pressure drop and
balancing the rest with calibrated orifice plates; it lifts the headers into the sand and
tries a radial manifold (rings fed in parallel) against the ring chain.

**The tube in the bed.**  A tube is a line inside a cell of the bed, coupled to it by
**Peaceman's well model**: the bed between the tube wall and the cell centre is counted
exactly, so the exchange does not depend on the mesh.

**The gas.**  Its properties (cp, viscosity, conductivity, Prandtl) follow its own
temperature along the loop.

**The envelope.**  No air domain: the air around the vessel is excluded and the outer
surface carries the natural + wind film; the soil carries the deep-ground temperature a
few metres down.  A shell thinner than its cell keeps its own mass and leaves the
insulation its resistance.

**The analyses.**
- **Standby**: the bed held at a temperature, and the power that holds it - the losses.
- **Transient**: charge and discharge profiles (constant, schedule, CSV), backward
  Euler, the energy ledger per step, HDF5 save/load of the state.
- **Automatic mesh**: the standby on finer and finer meshes until the answer stops
  moving (Richardson extrapolation, Grid Convergence Index).
- **Cycle accounting** (scripts): charge, standby, discharge, where the energy goes.

---

## How it solves

The full account - every equation and its source - is in
**[docs/18_SOLVER.md](docs/18_SOLVER.md)**; the reasons behind each choice in
[docs/12_METHODS.md](docs/12_METHODS.md); the bibliography in
[docs/17_REFERENCES.md](docs/17_REFERENCES.md).

| Piece | What | Source |
|---|---|---|
| Discretisation | cell-centred finite volume, harmonic mean at interfaces, half-cell films | Patankar (1980) |
| Mesh | a **tree of boxes**: every leaf has its own width in plan and its own height (any aspect ratio), 2:1 balanced per direction, arrays only | p4est, Burstedde et al. (2011); layered grids, Aziz & Settari (1979) |
| Bed conductivity | Zehner-Bauer-Schlünder + radiation, per cell, re-evaluated on the field | VDI Heat Atlas D6.3; Zehner & Schlünder (1970); Breitbach & Barthels (1980) |
| Gas circuit | 1-D segments between nodes, effectiveness-NTU, enthalpy mixing, balance imposed on the exchange | Kays & London (1984) |
| Hydraulics | looped pipe network, Darcy-Weisbach + tees + draught, Newton on the node pressures; headers sized on EN 10220 sizes, riser orifices (Idelchik) | Todini & Pilati (1988); Idelchik (2007) |
| Tube ↔ bed | well model, `r_eq = 0.14 sqrt(a^2 + b^2)` | Peaceman (1978, 1983) |
| Gas properties | air, nitrogen, steam at their temperature | Incropera et al., Tables A.4, A.6 |
| Film, friction | Dittus-Boelter / laminar Nu = 3.66; Haaland | Incropera; Haaland (1983) |
| Outer film | Churchill-Chu natural convection + `4 + 4 v` wind | Churchill & Chu (1975); ISO 6946 |
| Coupling | the gas film implicit in the wall temperature, the loop inlet solved with the field (one-row Schur complement) | - |
| Linear systems | CG on the volume-symmetrised operator + AMG Ruge-Stüben V(1,1); pinned rows solved directly; the AMG hierarchy reused for nearby operators | Ruge & Stüben (1987); PyAMG; Saad (2003) |
| Time | backward Euler, the nonlinearities lagged by one step | - |
| Mesh uncertainty | Richardson extrapolation, GCI | Roache (1998); Celik et al. (2008) |

Accuracy is asked in kelvin: the coupled iterations stop at **0.1 K**, the linear solves
at a relative residual of 1e-6.

---

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/tomdacu/thermal_battery_simulator.git
cd thermal_battery_simulator
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt   # numpy, scipy, h5py, PySide6, pyvista, pyvistaqt, pyamg, matplotlib
```

Without PyAMG the linear layer falls back to CG + Jacobi (slower, same answer).

---

## Quick start

```bash
python run_gui.py
```

The left column has six tabs; every explanation is behind an **ⓘ** tooltip.

1. **Vessel** - radius and height of the bed, foundation pad and the soil under it,
   insulation, shell, slabs, roof; the medium, its packing and grain size (the *Bed*
   read-out gives its conductivity at 20 and 500 °C), the insulation and the shell.
2. **Plant** - rated power, gas, mass flow, pressure, fan; the buried network (layout,
   rings or files, collection, tube and wall, lagged headers, azimuths, flow split); the
   **Header engine** (flow uniformity, velocity limit, design gas temperature, *Size the
   headers*).  The preview redraws as you edit.  In the 3D view, *Geometry* is one entry
   of the field selector.
3. **Site** - ambient, ground and wind.
4. **Mesh** - cells across the bed and the insulation, layers in the bed height, the
   cell budget; *Find the mesh* runs the automatic search.
5. **Analysis** - standby or transient; initial condition; *Charge* (the resistors) and
   *Discharge* (the exchanger); save and load a state.
6. **Solver** - temperature tolerance, linear residual, threads, radiation.

Then **Build mesh** and **Run**.  The mesh is coupled to the model it is built from:
change the vessel, a material, the site, the mesh settings or the plant and the mesh is
marked out of date (in the 3D view and on the Mesh tab), and **Run rebuilds it** - with
the header engine when the plant changed - before it solves, so a result always belongs
to the model on screen.  The right-hand panel shows the statistics, the energy
balance with the gas loop, the materials, the time series and the log; the time series
exports to CSV and the field to VTK (ParaView).

---

## The default plant and what it takes to run

The defaults are sized on the published pilot of the class (Kankaanpää, 2022: 4 m
across, 7 m tall, ~100 t of sand, 8 MWh, 200 kW of charge, 100 kW of discharge):

- 4 m bed, 5 m tall, steatite at 63 % packing, 1 mm grains (~95 t);
- 0.3 m of rock wool, 20 mm carbon-steel shell, 0.2 m slabs, 15° roof;
- 0.3 m concrete pad on 3 m of soil;
- 200 kW, 1 kg/s of air at 1 atm, 6 concentric rings, 126 risers of 50 mm; the engine
  chooses the radial manifold, headers 60-406 mm, 243 mm into the sand, 751 Pa at
  17 m/s, orifice plates balancing the risers (the ring chain needed 52 m/s).

Measured on the development machine (2026-09-23):

| Step | Time | Result |
|---|---|---|
| Size the headers (once per plant) | ~17 s | radial manifold, 751 Pa |
| Build and paint the mesh | 2-6 s | 36 268 leaves |
| Standby at 500 °C | 15 s | 5.9 kW of losses |
| Transient, 6 h of charge at 200 kW, dt = 900 s | 13 s | ~0.5 s per step |

---

## Documentation

| Document | Read it for |
|---|---|
| [00 Index](docs/00_INDEX.md) | the map, and how the documents are kept honest |
| [01 Theory](docs/01_THEORY.md) | heat equation, the packed bed, the gas, energy and exergy |
| [02 Discretisation](docs/02_FDM_DISCRETIZATION.md) | the discrete operators and boundary treatments |
| [03 Geometry](docs/03_GEOMETRY.md) | zones and paint order |
| [04 GUI design](docs/04_GUI_DESIGN.md) / [06 GUI configuration](docs/06_GUI_CONFIGURATION.md) | the panels, and every control with its default |
| [05 Architecture](docs/05_ARCHITECTURE.md) / [07 Code structure](docs/07_CODE_STRUCTURE.md) | layers and module map |
| [08 Analysis workflows](docs/08_ANALYSIS_WORKFLOWS.md) | the runs step by step |
| [09 Testing](docs/09_TESTING.md) | what each test file guarantees |
| [11 Handoff](docs/11_HANDOFF.md) | the state of the work |
| [12 Methods](docs/12_METHODS.md) | every choice, the rejected alternative, how it is checked |
| [13 Redesign](docs/13_REDESIGN.md) | the gas-loop architecture and its migration |
| [14 Verification](docs/14_VERIFICATION.md) | the verification campaign |
| [15 Pipe networks](docs/15_PIPE_NETWORKS.md) | layouts, collection modes, design rules, worked examples |
| [16 Adaptive mesh](docs/16_ADAPTIVE_MESH_MIGRATION.md) | the mesh protocol, the octree and the tree of boxes |
| [17 References](docs/17_REFERENCES.md) | every source, with the module that uses it |
| [18 The solver, end to end](docs/18_SOLVER.md) | the whole solution path with its equations |
| [19 Hydraulics and the header engine](docs/19_HYDRAULICS.md) | the network's flows, the orifices, the engine, the default design |

---

## Project structure

```
battery_simulation/
├── run_gui.py                 # entry point
├── src/                       # domain code - runs without Qt
│   ├── constants.py, units.py # physical constants, the Kelvin contract
│   ├── core/
│   │   ├── box_tree.py        # the anisotropic tree of boxes (the GUI's mesh)
│   │   ├── adaptive_mesh.py   # the physics on a tree: fields, films, assembly, balance
│   │   ├── octree.py          # the cubic octree (reference)
│   │   ├── mesh.py, grid.py   # the structured Mesh3D (reference), MaterialID
│   │   ├── materials.py       # grain database, Zehner-Bauer-Schlünder packed bed
│   │   ├── geometry.py        # vessel, painting, thin layers, soil, outer film
│   │   ├── pipes.py           # pipe runs and their rasterisation
│   │   ├── pipe_network.py    # layouts, collection modes, the gas graph
│   │   ├── environment.py     # natural + wind film
│   │   ├── physics.py         # harmonic mean, half-cell film, radiation
│   │   ├── profiles.py        # power / extraction / initial-condition profiles
│   │   └── refinement.py      # graded grid of Mesh3D (reference)
│   ├── solver/
│   │   ├── fluid.py           # the gas circuit: graph march, well model, gas properties
│   │   ├── hydraulics.py      # the network's pressures and flows, the header engine
│   │   ├── linear.py          # CG + AMG, decoupled rows, hierarchy reuse
│   │   ├── matrix.py          # structured assembly (reference)
│   │   ├── steady.py          # Picard sweeps: loop, bed conductivity, radiation
│   │   ├── transient.py       # backward Euler with the loop
│   │   └── octree_solver.py   # the octree's own solver (reference)
│   ├── analysis/              # fluxes, balance, losses (standby), mesh plan and search, cycle
│   ├── io/state.py            # HDF5 state with a geometry hash
│   └── viz/scene.py           # PyVista grids and exports
├── gui/                       # Qt layer - no physics
├── scripts/                   # benchmark.py, figures.py
├── tests/                     # pytest suite, analytic and reference checks
└── docs/                      # 00-18 and figures/
```

---

## Figures

Every figure in `docs/figures/` is generated offscreen by `python scripts/figures.py`:
the vessel with the buried bundle, a temperature slice, the graded mesh, the 2:1
balance, the closed gas loop, the effectiveness and blower design curves.

---

## Development

```bash
python -m pytest tests/ -q                                   # 412 cases, ~6 minutes, head-less
python -m pytest tests/ -q --ignore=tests/test_gui_sweep.py  # 401 without the GUI sweep
python -m ruff check src tests gui --select F,E9,B,SIM,UP     # lint
```

Counts measured on 2026-09-23 with `--collect-only`; re-run the command instead of
trusting the number.  `tests/test_octree.py` holds one wall-clock assertion (a
32 768-leaf tree in under a second) that a loaded machine can miss.

The same three jobs run on every push (`.github/workflows/ci.yml`): ruff, the suite on
Python 3.10 and 3.12 - head-less, without the GUI sweep and without that wall-clock
case - and the sweep on its own, where the wall-clock case runs as an informational
step.

**Stated limits** ([docs/18](docs/18_SOLVER.md) §10, [docs/19](docs/19_HYDRAULICS.md)
§8): 1-D incompressible gas, one tube per cell in the well model, headers sized at one
design gas temperature, no natural convection in the roof air or in the pores.

---

## Licence

**Code** — [PolyForm Noncommercial License 1.0.0](LICENSE): free to use, study, modify
and share for any non-commercial purpose — personal use (research, experiment, testing,
study, hobby) and any non-commercial organisation (charitable, educational, public
research, public safety or health, environmental protection, government), regardless of
their funding — with the attribution the licence carries
(`Required Notice: Copyright 2026 Tommaso D'Acunzio`).  This is a **source-available**
licence, not an open-source one.

**Commercial use** — inside a company, in a paid product or service, or as a hosted
service — needs a separate licence: write to <tommaso@dacunzio.it>.  Small and
academic cases are usually granted free of charge.

**Documents, figures and images** (`docs/`, `assets/`, `photo/`) are under
[CC BY-NC 4.0](LICENSE-CONTENT).

Author: Tommaso D'Acunzio — [GitHub](https://github.com/tomdacu) —
[LinkedIn](https://www.linkedin.com/in/tommaso-d-acunzio-344876185/) —
<tommaso@dacunzio.it>.  If you use this project, or you are interested in a
collaboration, write to me.
