# 🔋 Thermal Battery Simulator

![Banner](photo/Banner%20thermal%20battery%20simulator.png)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: PolyForm-Noncommercial-1.0.0](https://img.shields.io/badge/License-PolyForm_Noncommercial_1.0.0-blue.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

A comprehensive 3D thermal simulation tool for designing and analyzing **thermal energy storage systems** (also known as "Sand Batteries"). This software enables engineers and researchers to visualize temperature distributions, optimize insulation design, and evaluate energy storage performance.

![Thermal Battery Visualization](photo/heating_elements_3D.png)

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Goals](#-project-goals)
- [Architecture Overview](#-architecture-overview)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [User Guide](#-user-guide)
- [Performance Optimization](#-performance-optimization)
- [Documentation](#-documentation)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Simulation
- **3D Finite Difference Method (FDM)** solver for heat equation
- **Steady-state and Transient analysis** with Backward Euler implicit scheme
- **Iterative Losses Analysis** with secant method convergence
- **Solver methods**: Direct (LU), CG, BiCGSTAB, GMRES, with Jacobi/ILU/AMG
  preconditioning (CG + AMG by default)
- **Preconditioners**: Jacobi, ILU, AMG (PyAMG)
- **Vectorised NumPy matrix assembly** (no JIT dependency, reference-tested)

### Geometry Modeling
- **Cartesian 3D mesh** with flexible dimensions (Lx, Ly, Lz)
- **Cylindrical storage region** centered in domain
- **Multi-layer insulation**: radial, top, and bottom slabs
- **Optional conical roof** for realistic industrial designs
- **Flexible heater patterns**: Uniform, Grid, Radial, Spiral, Custom
- **Heat exchanger tubes**: Various patterns with internal convection BC

### Materials & Physics
- **Built-in material database**: Steatite, silica sand, rock wool, glass wool, etc.
- **Packing fraction adjustment** for porous media
- **Convection, conduction, and Dirichlet boundary conditions**
- **Energy and exergy balance calculations** with detailed loss analysis
- **Thermal autonomy estimation** based on stored energy and losses

### Analysis Types
- **Steady-State**: Equilibrium temperature distribution with constant power
- **Losses Analysis**: Iterative solver to find thermal losses at target temperature
- **Transient**: Time-dependent simulation with power and extraction profiles

### Transient Analysis
- **Power profiles**: off, constant, a time schedule, or a CSV file
- **Extraction profiles**: off, a fixed power, or a fluid flow rate
- **State save/load**: HDF5 format with geometry hash verification
- **Time-series results** with CSV export (full fields can be saved per sample for ParaView)

### Visualization
- **Interactive 3D visualization** with PyVista
- **Slice planes** (X, Y, Z) for internal inspection, with adjustable transparency
- **Real-time updates** during parameter changes
- **Temperature display in Celsius** with single vertical colorbar

### User Interface
- **Clean PyQt6 GUI** with 4-tab structure (Geometry, Materials, Analysis, Tools)
- **Threaded simulation** - responsive UI during computation
- **Detailed energy balance panel** with losses breakdown
- **Save/Load simulation states** in HDF5 format
- **Export options**: CSV, VTK for ParaView, HDF5 for state persistence

---

## 🎯 Project Goals

The **Thermal Battery Simulator** is designed to:

1. **Configure Complex Geometries**: Define dimensions, insulation layers, and placement of heat exchangers and heaters.

2. **Simulate Operating Scenarios**: analyse the steady state, the losses set point, or
   the time evolution with power and extraction profiles.

3. **Optimize Design**: Evaluate the impact of different materials and configurations on energy efficiency and thermal losses.

4. **Accessibility**: Make complex numerical simulation accessible through an intuitive graphical interface, eliminating the need to modify code for each test.

---

## 🏗️ Architecture Overview

The system follows a **GUI-driven design** where all simulation parameters originate from the user interface:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GUI (PyQt6)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Geometry    │  │  Materials   │  │   Analysis   │           │
│  │  - Lx,Ly,Lz  │  │  - storage   │  │  - type      │           │
│  │  - cylinder  │  │  - insulation│  │  - profiles  │           │
│  │  - heaters   │  │  - packing % │  │  - solver    │           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
└─────────┼──────────────────┼──────────────────┼─────────────────┘
          │                  │                  │
          ▼                  ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BatteryGeometry                               │
│         (Dataclass combining all configuration)                  │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Mesh3D                                   │
│            3D arrays: T, k, ρ, cp, Q, material_id                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 SteadyStateSolver / TransientSolver              │
│                    Solves A·T = b  (iterative)                   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               3D Temperature Field + Energy Balance              │
└─────────────────────────────────────────────────────────────────┘
```

---


## 🌡️ Units and conventions

* Inside `src/` every temperature is an **absolute temperature in Kelvin**;
  ambient/ground defaults are 293.15 K / 283.15 K.  The GUI works in degC and
  converts only in `gui/units.py`, so a unit mistake cannot hide in the physics.
* Powers are watts, energies joules, lengths metres, time seconds.
* Volumetric sources (`mesh.Q_source`) are `W/m^3`, sinks (`mesh.Q_sink`) are
  `W/m^3` and always negative.  The matrix coefficients are per unit volume.
* `check_kelvin()` rejects a field that looks like degC, and the mesh refuses to
  build a geometry whose roof or shell does not fit the domain.

## 💻 Installation

### Prerequisites
- Python 3.10 or higher
- Git (optional, for cloning)

### Step-by-step Installation

```bash
# 1. Clone the repository
git clone https://github.com/PhyTom/Thermal_battery_simulator.git
cd Thermal_battery_simulator

# 2. Create virtual environment
python -m venv .venv

# 3. Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On Linux/Mac:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Optional: PyAMG for AMG Preconditioner
```bash
pip install pyamg
```

---

## 🚀 Quick Start

### Launch the GUI
```bash
python run_gui.py
```

### Basic Workflow

1. **Configure Geometry** (Geometry tab)
   - Set domain dimensions (Lx, Ly, Lz)
   - Configure cylinder dimensions (radius, height)
   - Set up insulation layers (radial, top, bottom slabs)
   - Configure heaters pattern and power
   - Define mesh spacing

2. **Set Materials** (Materials tab)
   - Select storage material (Steatite, Sand, etc.)
   - Choose insulation material
   - Set packing fraction
   - Configure operating conditions (T_ambient, h_external)

3. **Configure Analysis** (Analysis tab)
   - Choose analysis type: Steady, Losses, or Transient
   - For Losses: set target temperature and ambient conditions
   - For Transient: set power and extraction profiles
   - Configure initial conditions

4. **Configure Solver** (Tools > Solver sub-tab)
   - Select solver method and preconditioner
   - Set tolerance and max iterations
   - Choose the number of threads
   - For Losses: adjust iteration parameters (α, h_conv, T_ground)

5. **Build & Run**
   - Click "Build Mesh" button
   - Click "Run Simulation"
   - View results in 3D visualization

6. **Analyze Results** (Tools > Results sub-tabs)
   - View temperature statistics
   - Analyze energy balance with losses breakdown
   - Check thermal autonomy
   - Export results (CSV, VTK, HDF5)

---

## 📖 User Guide

### Geometry Configuration

The battery uses a **4-zone concentric structure**:

| Zone | Description | Typical Material |
|------|-------------|------------------|
| **STORAGE** | Central thermal mass | Steatite, Sand |
| **INSULATION** | Thermal barrier | Rock wool |
| **STEEL** | Structural shell | Carbon steel |
| **AIR** | External environment | Air |

**Key Parameters:**
- `r_storage`: Radius of storage zone [m]
- `insulation_thickness`: Insulation layer [m]
- `shell_thickness`: Steel shell [m]
- `height`: Total battery height [m]

### Heater Patterns

| Pattern | Description | Best For |
|---------|-------------|----------|
| **Uniform** | Distributed throughout volume | Simple analysis |
| **Grid** | Rectangular array | Regular layouts |
| **Radial** | Concentric rings | Cylindrical symmetry |
| **Spiral** | Spiral from center | Uniform coverage |

### Tube Patterns

| Pattern | Description | Best For |
|---------|-------------|----------|
| **Central Cluster** | Group at center | Small systems |
| **Radial Array** | Rings around center | Large systems |
| **Hexagonal** | Maximum density | High extraction |

---

## 🧭 Methods and why

The full rationale - each choice, the alternative that was rejected, and how it is
checked - is in **[docs/12_METHODS.md](docs/12_METHODS.md)**.  The short version:

| Question | Answer | Why not the alternative |
|---|---|---|
| Discretisation | cell-centred **finite volume** on a Cartesian grid | conservation holds on *any* grid, so the energy balance is a verification tool; finite differences lose that, body-fitted meshes cost far more complexity for this squat-cylinder geometry |
| Conductivity at a material interface | **harmonic mean** of the two cells | reproduces the exact series resistance of a two-layer wall; the arithmetic mean overestimates the flux by orders of magnitude when steel (50) meets rock wool (0.04) |
| Solver | **CG on the symmetrised system** + AMG | the physical operator is symmetric; writing it per unit volume destroys that, so the similarity transform `diag(V)^-1/2 K diag(V)^-1/2` restores it and keeps CG (and AMG) available on graded grids |
| Convective surfaces | half-cell conduction in series with the film | a "cell at the film temperature" would need the surface cell to be isothermal; the price is first-order accuracy at the surface, which is why the mesh search floors the observed order at 1 |
| Transient | **backward Euler**, adaptive step | unconditionally stable for a stiff sand/steel/insulation stack; Crank-Nicolson rings on the power steps this tool exists to simulate |
| Losses | **secant** iteration on the power | smooth monotone loss, no derivative needed, 3-6 iterations; bisection needs a bracket and is slower |
| Mesh | graded bands, **linear ramp** + density equidistribution | the linear ramp bounds the neighbour ratio by `growth` by construction; an exponential ramp looks shorter but its per-cell ratio grows with the cell size |
| Mesh size | **a priori plan** (`thickness/N`, `2k/h`) + **Richardson/GCI search** | the error model predicts the grid that meets the tolerance, so the search jumps there; "refine until two grids agree" is only valid for adjacent grids and rejects good ones after a jump |
| Reported uncertainty | **GCI** of the chosen grid (Roache, Fs = 1.25) | an unqualified number hides the discretisation error; ASME V&V 20 and Celik et al. (2008) prescribe exactly this reporting |

## ⚡ Performance Optimization

### Why is simulation slow?

Computation time depends on:
- **Number of cells**: $N = N_x \times N_y \times N_z$ (100×100×100 = 1 million cells!)
- **Solver method**: Direct methods are O(N^1.5), iterative are O(N)
- **Tolerance**: Tighter tolerances require more iterations

### Recommended Configuration by Scenario

| Scenario | Method | Precond. | Tolerance | Est. Time |
|----------|--------|----------|-----------|-----------|
| Quick test | cg | none | 1e-4 | ~1 sec |
| Visualization | cg | jacobi | 1e-6 | ~5 sec |
| Standard precision | cg | jacobi | 1e-8 | ~15 sec |
| High precision | bicgstab | jacobi | 1e-10 | ~30 sec |

### Solver Methods

| Method | Description | When to Use |
|--------|-------------|-------------|
| **bicgstab** | BiCGSTAB | ⭐ **RECOMMENDED**. Robust, always works |
| **cg** | Conjugate Gradient | Fast but may not converge with mixed BC |
| **gmres** | GMRES | Excellent convergence, uses more memory |
| **direct** | Direct LU | Only for small meshes (<30k cells) |

> ⚠️ **Note on CG**: CG requires symmetric positive definite matrix. With mixed boundary conditions (tube convection + Dirichlet), the matrix may lose symmetry → use BiCGSTAB.

### Preconditioners

| Precond. | Description | Performance |
|----------|-------------|-------------|
| **jacobi** | Diagonal | ⭐ **RECOMMENDED**. Multi-threaded, fast |
| **none** | None | Pure CG, surprisingly fast! |
| **ilu** | Incomplete LU | ⚠️ Single-threaded, can be SLOW |
| **amg** | Algebraic Multigrid | Best for very large systems (requires PyAMG) |

> ⚠️ **Important**: ILU uses SuperLU which is single-threaded. For large meshes, Jacobi or no preconditioner is often faster!

### Built-in Performance Optimizations

### Tolerance Guide

| Value | Use Case | Notes |
|-------|----------|-------|
| 1e-10 | High precision | For validation and detailed analysis |
| 1e-8 | Default | Good speed/precision balance |
| 1e-6 | Fast | Sufficient for visualization |
| 1e-4 | Very fast | Only for quick tests |

### Multi-Threading

- **Auto**: Uses all CPU cores → maximum speed, may slow system
- **All - 1**: ⭐ **Recommended**. Leaves one core free for GUI
- **N cores**: Limits to N specific cores

### Practical Tips

1. **Start with small meshes** (30-40 points) for quick tests
2. **Use BiCGSTAB + Jacobi** for most cases (or GPU for large meshes)
3. **Increase mesh** only for final results
4. **Tolerance 1e-6** is sufficient for visualization
5. **Check energy balance** to validate results
6. **Use GPU** for meshes >100k cells if available

---

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [00_INDEX.md](docs/00_INDEX.md) | Index and how the documents are kept in sync |
| [01_THEORY.md](docs/01_THEORY.md) | Heat transfer fundamentals, packed bed, energy and exergy |
| [02_FDM_DISCRETIZATION.md](docs/02_FDM_DISCRETIZATION.md) | Discrete operators, boundary treatments, validation |
| [03_GEOMETRY.md](docs/03_GEOMETRY.md) | Zones, paint order, heater/tube patterns, validation rules |
| [04_GUI_DESIGN.md](docs/04_GUI_DESIGN.md) | Panels, run state machine, threading, results |
| [05_ARCHITECTURE.md](docs/05_ARCHITECTURE.md) | Layers, contracts, error handling, extension points |
| [06_GUI_CONFIGURATION.md](docs/06_GUI_CONFIGURATION.md) | Every control with default, range and unit |
| [07_CODE_STRUCTURE.md](docs/07_CODE_STRUCTURE.md) | Module map and public API |
| [08_ANALYSIS_WORKFLOWS.md](docs/08_ANALYSIS_WORKFLOWS.md) | The three runs step by step and the reported quantities |
| [09_TESTING.md](docs/09_TESTING.md) | What is verified, by which test, and how to run it |
| [10_MESH_AND_HEATERS.md](docs/10_MESH_AND_HEATERS.md) | Graded mesh, hairpin heaters, automatic mesh search |
| [11_HANDOFF.md](docs/11_HANDOFF.md) | State of the work: read this first after a context reset |
| [12_METHODS.md](docs/12_METHODS.md) | **Every method choice, why it was made, how it is checked** |

---

## 📁 Project Structure

```
battery_simulation/
├── run_gui.py                 # entry point
├── requirements.txt
├── config/                    # (removed: defaults live in src/, one source of truth)
├── src/                       # domain code - runs without Qt
│   ├── constants.py           # physical constants, Kelvin contract
│   ├── units.py               # degC <-> K helpers and check_kelvin()
│   ├── core/
│   │   ├── mesh.py            # Mesh3D, MaterialID, FaceBC (per-face BC)
│   │   ├── grid.py            # index tables shared by solver and analysis
│   │   ├── physics.py         # half-cell / harmonic-mean / radiation coefficients
│   │   ├── materials.py       # material database + packed-bed model
│   │   ├── geometry.py        # cylinder/heater/tube config + voxel painting
│   │   ├── heaters.py         # hairpin bank, rasteriser, surface-power check
│   │   ├── refinement.py      # bands -> graded grid (ramp + density equidistribution)
│   │   └── profiles.py        # power / extraction / initial-condition profiles
│   ├── solver/
│   │   ├── matrix.py          # FDM assembly (steady and transient operators)
│   │   ├── linear.py          # direct / CG / BiCGSTAB / GMRES + preconditioners
│   │   ├── steady.py          # steady solver (with optional radiation sweeps)
│   │   ├── transient.py       # backward-Euler transient solver
│   │   └── results.py         # TransientResults container
│   ├── analysis/
│   │   ├── fluxes.py          # single flux evaluator (solver-consistent)
│   │   ├── balance.py         # energy / exergy balance of a mesh state
│   │   ├── losses.py          # losses analysis (power needed at a set point)
│   │   ├── convergence.py     # automatic mesh: Richardson/GCI error model
│   │   └── mesh_plan.py       # a priori cell size per region (layer/N, 2k/h)
│   ├── io/state.py            # HDF5 state with geometry hash and unit tag
│   └── viz/scene.py           # shared PyVista grid, material colours, exports
├── gui/                       # Qt layer - no physics
│   ├── main_window.py         # wiring only
│   ├── controller.py          # RunConfig + worker threads + run state machine
│   ├── widgets.py, units.py, safe.py
│   └── views/                 # geometry, materials, analysis, solver, results, 3D
├── scripts/                   # benchmarks and diagnostics (not part of the package)
├── tests/                     # pytest suite incl. analytic regressions
└── docs/                      # theory and design notes
```

## 📦 Requirements

### Core Dependencies
- **Python** 3.10+
- **NumPy** - Numerical computations
- **SciPy** - Sparse matrices and solvers
- **PyQt6** - GUI framework
- **PyVista** - 3D visualization
- **PyVistaQt** - PyVista-Qt integration
- **h5py** - HDF5 state persistence

### Optional Dependencies
- **PyAMG** - Algebraic Multigrid preconditioner (recommended: it is what keeps
  large meshes converging in a handful of iterations)
- **Matplotlib** - the time-series plots of the transient results

### Installation
```bash
pip install -r requirements.txt

```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
# Clone and install in development mode
git clone https://github.com/PhyTom/Thermal_battery_simulator.git
cd Thermal_battery_simulator
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Run tests
pytest tests/
```

### Future Enhancements
- [ ] Additional export formats
- [ ] Editable material database in GUI
- [ ] 2D temporal evolution plots
- [ ] Multi-physics coupling (flow + heat)

---

## 📄 License

This project is licensed under the PolyForm Noncommercial License 1.0.0 - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**PhyTom**

---

## 🙏 Acknowledgments

- Heat transfer theory based on Incropera & DeWitt
- PyVista for excellent 3D visualization
- SciPy for robust numerical solvers
