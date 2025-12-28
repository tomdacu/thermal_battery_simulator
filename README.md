# 🔋 Thermal Battery Simulator 3D

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://www.riverbankcomputing.com/software/pyqt/)

A comprehensive 3D thermal simulation tool for designing and analyzing **thermal energy storage systems** (also known as "Sand Batteries"). This software enables engineers and researchers to visualize temperature distributions, optimize insulation design, and evaluate energy storage performance.

![Thermal Battery Visualization](photo/screenshot.png)

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
- **Steady-state analysis** with future transient support planned
- **Multiple solver methods**: Direct (LU), CG, BiCGSTAB, GMRES
- **Preconditioners**: Jacobi, ILU, AMG (PyAMG)
- **Vectorized matrix builder** for 10-50x faster assembly

### Geometry Modeling
- **4-zone concentric cylinder**: Storage, Insulation, Steel Shell, Air
- **Vertical insulation slabs**: Top and bottom thermal protection
- **Optional conical roof** for realistic industrial designs
- **Flexible heater patterns**: Uniform, Grid, Radial, Spiral, Custom
- **Heat exchanger tubes**: Various patterns with internal convection BC

### Materials & Physics
- **Built-in material database**: Steatite, silica sand, rock wool, glass wool, etc.
- **Packing fraction adjustment** for porous media
- **Convection, conduction, and Dirichlet boundary conditions**
- **Energy balance calculations** with loss analysis

### Visualization
- **Interactive 3D visualization** with PyVista
- **Slice planes** (X, Y, Z) for internal inspection
- **Volume rendering** and isosurfaces
- **Real-time updates** during parameter changes

### User Interface
- **Intuitive PyQt6 GUI** - no code modification needed
- **Threaded simulation** - responsive UI during computation
- **Export options**: CSV, VTK for ParaView

---

## 🎯 Project Goals

The **Thermal Battery Simulator** is designed to:

1. **Configure Complex Geometries**: Define dimensions, insulation layers, and placement of heat exchangers and heaters.

2. **Simulate Operating Scenarios**: Analyze thermal behavior in steady-state (and future transient) conditions by varying power and temperatures.

3. **Optimize Design**: Evaluate the impact of different materials and configurations on energy efficiency and thermal losses.

4. **Accessibility**: Make complex numerical simulation accessible through an intuitive graphical interface, eliminating the need to modify code for each test.

---

## 🏗️ Architecture Overview

The system follows a **GUI-driven design** where all simulation parameters originate from the user interface:

```
┌─────────────────────────────────────────────────────────────────┐
│                        GUI (PyQt6)                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│  │  Geometry    │  │  Materials   │  │   Solver     │           │
│  │  - radius    │  │  - storage   │  │  - method    │           │
│  │  - height    │  │  - insulation│  │  - tolerance │           │
│  │  - slabs     │  │  - packing % │  │  - threads   │           │
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
│            3D arrays: T, k, ρ, cp, Q, boundaries                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SteadyStateSolver                             │
│                    Solves A·T = b                                │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               3D Temperature Field + Analysis                    │
└─────────────────────────────────────────────────────────────────┘
```

---

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
   - Define battery radius and height
   - Configure insulation thickness

2. **Set Heaters** (Heaters tab)
   - Total power [kW]
   - Distribution pattern
   - Number of elements

3. **Configure Tubes** (Tubes tab) - Optional
   - Enable/disable heat extraction
   - Fluid temperature and convection coefficient

4. **Build Mesh**
   - Click "Build Mesh" button
   - Mesh preview appears in 3D view

5. **Run Simulation**
   - Click "Run Simulation"
   - View temperature distribution
   - Analyze energy balance

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
2. **Use BiCGSTAB + Jacobi** for most cases
3. **Increase mesh** only for final results
4. **Tolerance 1e-6** is sufficient for visualization
5. **Check energy balance** to validate results

---

## 📚 Documentation

Detailed documentation is available in the `docs/` folder:

| Document | Description |
|----------|-------------|
| [01_THEORY.md](docs/01_THEORY.md) | Heat transfer fundamentals and equations |
| [02_FDM_DISCRETIZATION.md](docs/02_FDM_DISCRETIZATION.md) | Finite Difference Method details |
| [03_GEOMETRY.md](docs/03_GEOMETRY.md) | Geometry model and mesh mapping |
| [04_GUI_DESIGN.md](docs/04_GUI_DESIGN.md) | GUI structure and usage |
| [05_ARCHITECTURE.md](docs/05_ARCHITECTURE.md) | Software architecture |
| [06_GUI_CONFIGURATION.md](docs/06_GUI_CONFIGURATION.md) | Parameter configuration guide |
| [07_CODE_STRUCTURE.md](docs/07_CODE_STRUCTURE.md) | Detailed code documentation |

---

## 📁 Project Structure

```
battery_simulation/
├── run_gui.py              # 🚀 Main entry point - launches GUI
├── materials_database.py   # Material properties database
├── requirements.txt        # Python dependencies
│
├── gui/                    # User Interface
│   └── main_window.py      # PyQt6 main window
│
├── src/                    # Source code
│   ├── core/               # Domain model
│   │   ├── mesh.py         # 3D mesh data structure
│   │   ├── geometry.py     # Battery geometry definition
│   │   └── materials.py    # Material manager
│   │
│   ├── solver/             # Numerical engine
│   │   ├── matrix_builder.py  # FDM matrix assembly
│   │   └── steady_state.py    # Linear system solver
│   │
│   ├── analysis/           # Post-processing
│   │   └── power_balance.py   # Energy balance calculations
│   │
│   └── visualization/      # Rendering
│       └── renderer.py     # Standalone PyVista renderer
│
├── tests/                  # Unit tests
│   ├── test_core.py
│   └── test_solver.py
│
├── docs/                   # Documentation
├── config/                 # Configuration files
└── photo/                  # Screenshots and images
```

---

## 📦 Requirements

### Core Dependencies
- **Python** 3.10+
- **NumPy** - Numerical computations
- **SciPy** - Sparse matrices and solvers
- **PyQt6** - GUI framework
- **PyVista** - 3D visualization
- **PyVistaQt** - PyVista-Qt integration
- **PyYAML** - Configuration files

### Optional Dependencies
- **Numba** - JIT acceleration (optional)
- **PyAMG** - Algebraic Multigrid preconditioner

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
- [ ] Transient simulation support
- [ ] Parameter presets and project saving
- [ ] Additional export formats
- [ ] Editable material database in GUI
- [ ] 2D temporal evolution plots

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**PhyTom**

---

## 🙏 Acknowledgments

- Heat transfer theory based on Incropera & DeWitt
- PyVista for excellent 3D visualization
- SciPy for robust numerical solvers
