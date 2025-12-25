# Battery Simulation Project

## Descrizione

Sistema di simulazione termica 3D per Sand Battery basato sul metodo delle differenze finite (FDM).

## Struttura del Progetto

```
battery_simulation/
├── src/
│   ├── core/           # Motore di calcolo
│   │   ├── mesh.py     # Classe Mesh3D
│   │   ├── materials.py # Gestione materiali
│   │   └── geometry.py # Definizione geometria batteria
│   ├── solver/         # Solutori numerici
│   │   ├── matrix_builder.py
│   │   └── steady_state.py
│   ├── analysis/       # Post-processing
│   │   └── power_balance.py
│   └── visualization/  # Visualizzazione PyVista
│       └── renderer.py
├── gui/                # Interfaccia PyQt6
│   └── main_window.py
├── config/             # Configurazioni
│   └── default_config.yaml
├── docs/               # Documentazione
│   ├── 01_THEORY.md
│   ├── 02_FDM_DISCRETIZATION.md
│   ├── 03_GEOMETRY.md
│   └── 04_GUI_DESIGN.md
├── tests/              # Unit tests
├── materials_database.py
└── PROJECT_ARCHITECTURE.py
```

## Installazione

```bash
# Creare ambiente virtuale
python -m venv .venv
.venv\Scripts\activate  # Windows

# Installare dipendenze
pip install -r requirements.txt
```

## Uso Rapido

```python
from src.core.mesh import Mesh3D
from src.core.materials import MaterialManager
from src.core.geometry import create_small_test_geometry
from src.solver.steady_state import SteadyStateSolver

# 1. Creare mesh uniforme
mesh = Mesh3D(Lx=6, Ly=6, Lz=5, target_spacing=0.2)

# 2. Definire geometria e materiali
geom = create_small_test_geometry()
materials = MaterialManager()
geom.apply_to_mesh(mesh, materials)

# 3. Risolvere il caso stazionario
solver = SteadyStateSolver(mesh)
result = solver.solve()

# 4. Visualizzare (richiede PyVista)
from src.visualization.renderer import BatteryRenderer
renderer = BatteryRenderer()
renderer.plot_3d(mesh)
```

## Requisiti

- Python 3.10+
- NumPy
- SciPy
- PyVista
- PyQt6
- Numba (opzionale, per accelerazione)

## Licenza

MIT License
