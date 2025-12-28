"""
main_window.py - Finestra principale della GUI

=============================================================================
ARCHITECTURE NOTE: GUI-DRIVEN CONFIGURATION
=============================================================================

This file is the SINGLE SOURCE OF TRUTH for all simulation parameters.
The simulation engine (solver, analysis) receives ALL its parameters from
the GUI widgets defined here. There are NO hardcoded simulation values in
the core modules.

KEY DESIGN PRINCIPLE:
    Widget values → _build_battery_geometry_from_inputs() → BatteryGeometry
    BatteryGeometry → apply_to_mesh() → Mesh3D fields
    Mesh3D → SteadyStateSolver → Temperature solution

PARAMETER FLOW:
    1. User configures widgets in tabs (Geometria, Resistenze, Tubi, Solver)
    2. _build_battery_geometry_from_inputs() reads ALL widgets
    3. Creates BatteryGeometry, HeaterConfig, TubeConfig objects
    4. apply_to_mesh() maps analytic geometry to mesh fields
    5. Solver uses mesh fields (k, rho, cp, Q, boundary_type, etc.)

See docs/05_ARCHITECTURE.md and docs/06_GUI_CONFIGURATION.md for details.
=============================================================================

Interfaccia grafica completa per la simulazione Sand Battery:
- Pannello sinistro: Input parametri e controlli
- Pannello centrale: Visualizzazione 3D PyVista
- Pannello inferiore: Risultati e statistiche
- Controlli slicing interattivi
"""

import sys
import numpy as np
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox, QSlider,
    QSpinBox, QDoubleSpinBox, QTabWidget, QTextEdit, QProgressBar,
    QSplitter, QFrame, QGridLayout, QCheckBox, QMessageBox,
    QFileDialog, QStatusBar, QToolBar, QMenuBar, QMenu,
    QScrollArea, QListWidget, QListWidgetItem, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QAction, QFont, QIcon

import pyvista as pv
from pyvistaqt import QtInteractor

# Aggiungi src al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.mesh import Mesh3D, MaterialID
from src.core.materials import MaterialManager, MaterialType
from src.core.geometry import (
    BatteryGeometry, CylinderGeometry, 
    HeaterConfig, TubeConfig,
    HeaterPattern, TubePattern
)
from src.solver.steady_state import SteadyStateSolver, SolverConfig
from src.analysis.power_balance import PowerBalanceAnalyzer


class SimulationThread(QThread):
    """Thread per eseguire la simulazione senza bloccare la GUI"""
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, mesh, config):
        super().__init__()
        self.mesh = mesh
        self.config = config
    
    def run(self):
        try:
            self.progress.emit(10, "Costruzione matrice...")
            solver = SteadyStateSolver(self.mesh, self.config)
            solver.build_system()
            
            self.progress.emit(50, "Risoluzione sistema...")
            result = solver.solve(rebuild=False)
            
            self.progress.emit(90, "Analisi risultati...")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class SandBatteryGUI(QMainWindow):
    """Finestra principale dell'applicazione"""
    
    def __init__(self):
        super().__init__()
        
        self.mesh = None
        self.battery_geometry = None
        self.mat_manager = MaterialManager()
        self.simulation_thread = None
        self._simulation_completed = False  # Flag per tracciare se simulazione eseguita
        
        self.init_ui()
        self.init_default_values()
        
    def init_ui(self):
        """Inizializza l'interfaccia utente"""
        self.setWindowTitle("Sand Battery Thermal Simulation")
        self.setGeometry(100, 100, 1600, 900)
        
        # Menu bar
        self.create_menu_bar()
        
        # Toolbar
        self.create_toolbar()
        
        # Widget centrale
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principale con splitter
        main_layout = QHBoxLayout(central_widget)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Pannello sinistro - Controlli
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)
        
        # Pannello centrale - Visualizzazione 3D
        center_panel = self.create_center_panel()
        splitter.addWidget(center_panel)
        
        # Pannello destro - Risultati
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)
        
        # Proporzioni splitter
        splitter.setSizes([300, 900, 400])
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Pronto")
        
    def create_menu_bar(self):
        """Crea la barra dei menu"""
        menubar = self.menuBar()
        
        # Menu File
        file_menu = menubar.addMenu("File")
        
        new_action = QAction("Nuova Simulazione", self)
        new_action.setShortcut("Ctrl+N")
        new_action.triggered.connect(self.new_simulation)
        file_menu.addAction(new_action)
        
        save_action = QAction("Salva Risultati...", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        export_vtk = QAction("Esporta VTK...", self)
        export_vtk.triggered.connect(self.export_vtk)
        file_menu.addAction(export_vtk)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Esci", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Menu Visualizzazione
        view_menu = menubar.addMenu("Visualizzazione")
        
        reset_view = QAction("Reset Vista", self)
        reset_view.triggered.connect(self.reset_view)
        view_menu.addAction(reset_view)
        
        screenshot = QAction("Screenshot...", self)
        screenshot.triggered.connect(self.take_screenshot)
        view_menu.addAction(screenshot)
        
    def create_toolbar(self):
        """Crea la toolbar"""
        toolbar = QToolBar("Toolbar principale")
        self.addToolBar(toolbar)
        
        run_btn = QAction("▶ Simula", self)
        run_btn.triggered.connect(self.run_simulation)
        toolbar.addAction(run_btn)
        
        toolbar.addSeparator()
        
        slice_x = QAction("Sezione X", self)
        slice_x.triggered.connect(lambda: self.show_slice('x'))
        toolbar.addAction(slice_x)
        
        slice_y = QAction("Sezione Y", self)
        slice_y.triggered.connect(lambda: self.show_slice('y'))
        toolbar.addAction(slice_y)
        
        slice_z = QAction("Sezione Z", self)
        slice_z.triggered.connect(lambda: self.show_slice('z'))
        toolbar.addAction(slice_z)
        
        toolbar.addSeparator()
        
        view_3d = QAction("Vista 3D", self)
        view_3d.triggered.connect(self.show_3d_view)
        toolbar.addAction(view_3d)
        
    def create_left_panel(self) -> QWidget:
        """Crea il pannello sinistro con i controlli input usando tab scrollabili"""
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        
        # Tab widget per organizzare le configurazioni
        tabs = QTabWidget()
        
        # === TAB 1: GEOMETRIA E MESH ===
        geom_tab = self._create_geometry_tab()
        tabs.addTab(geom_tab, "Geometria")
        
        # === TAB 2: RESISTENZE (HEATERS) ===
        heater_tab = self._create_heater_tab()
        tabs.addTab(heater_tab, "Resistenze")
        
        # === TAB 3: TUBI (TUBES) ===
        tube_tab = self._create_tube_tab()
        tabs.addTab(tube_tab, "Tubi")
        
        # === TAB 4: SOLVER ===
        solver_tab = self._create_solver_tab()
        tabs.addTab(solver_tab, "Solver")
        
        main_layout.addWidget(tabs)
        
        # === PULSANTI (sempre visibili) ===
        btn_layout = QVBoxLayout()
        
        # Pulsante Costruisci Mesh
        self.build_mesh_btn = QPushButton("🗘 Costruisci Mesh")
        self.build_mesh_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.build_mesh_btn.clicked.connect(self.build_mesh)
        btn_layout.addWidget(self.build_mesh_btn)

        # Pulsante Anteprima Geometria (senza mesh)
        self.preview_geom_btn = QPushButton("👁 Anteprima Geometria")
        self.preview_geom_btn.clicked.connect(self.preview_geometry)
        btn_layout.addWidget(self.preview_geom_btn)
        
        # Pulsante Esegui Simulazione
        self.run_btn = QPushButton("▶ Esegui Simulazione")
        self.run_btn.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        self.run_btn.clicked.connect(self.run_simulation)
        self.run_btn.setEnabled(False)  # Disabilitato finché non c'è la mesh
        btn_layout.addWidget(self.run_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Stato: Pronto")
        btn_layout.addWidget(self.status_label)
        
        main_layout.addLayout(btn_layout)
        
        return panel
    
    def _create_scrollable_widget(self, content_widget: QWidget) -> QScrollArea:
        """Crea un widget scrollabile contenente il content_widget"""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll
    
    def _create_geometry_tab(self) -> QWidget:
        """Crea il tab per geometria e mesh"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # === GEOMETRIA ===
        geom_group = QGroupBox("Geometria Batteria")
        geom_layout = QGridLayout()
        
        # Dimensioni dominio
        geom_layout.addWidget(QLabel("Lx [m]:"), 0, 0)
        self.lx_spin = QDoubleSpinBox()
        self.lx_spin.setRange(1, 50)
        self.lx_spin.setValue(6.0)
        self.lx_spin.setDecimals(1)
        geom_layout.addWidget(self.lx_spin, 0, 1)
        
        geom_layout.addWidget(QLabel("Ly [m]:"), 1, 0)
        self.ly_spin = QDoubleSpinBox()
        self.ly_spin.setRange(1, 50)
        self.ly_spin.setValue(6.0)
        self.ly_spin.setDecimals(1)
        geom_layout.addWidget(self.ly_spin, 1, 1)
        
        geom_layout.addWidget(QLabel("Lz [m]:"), 2, 0)
        self.lz_spin = QDoubleSpinBox()
        self.lz_spin.setRange(1, 50)
        self.lz_spin.setValue(5.0)
        self.lz_spin.setDecimals(1)
        geom_layout.addWidget(self.lz_spin, 2, 1)
        
        # Raggio cilindro
        geom_layout.addWidget(QLabel("Raggio [m]:"), 3, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.5, 20)
        self.radius_spin.setValue(2.3)
        self.radius_spin.setDecimals(2)
        geom_layout.addWidget(self.radius_spin, 3, 1)
        
        # Altezza
        geom_layout.addWidget(QLabel("Altezza [m]:"), 4, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 30)
        self.height_spin.setValue(4.0)
        self.height_spin.setDecimals(1)
        geom_layout.addWidget(self.height_spin, 4, 1)
        
        geom_group.setLayout(geom_layout)
        layout.addWidget(geom_group)
        
        # === MESH ===
        mesh_group = QGroupBox("Mesh")
        mesh_layout = QGridLayout()
        
        mesh_layout.addWidget(QLabel("Spaziatura [m]:"), 0, 0)
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.05, 1.0)
        self.spacing_spin.setValue(0.2)
        self.spacing_spin.setDecimals(2)
        self.spacing_spin.setSingleStep(0.05)
        mesh_layout.addWidget(self.spacing_spin, 0, 1)
        
        self.mesh_info_label = QLabel("Celle: -")
        mesh_layout.addWidget(self.mesh_info_label, 1, 0, 1, 2)
        
        self.memory_label = QLabel("Memoria: -")
        self.memory_label.setStyleSheet("color: gray; font-size: 9px;")
        mesh_layout.addWidget(self.memory_label, 2, 0, 1, 2)
        
        self.spacing_spin.valueChanged.connect(self.update_mesh_info)
        self.lx_spin.valueChanged.connect(self.update_mesh_info)
        self.ly_spin.valueChanged.connect(self.update_mesh_info)
        self.lz_spin.valueChanged.connect(self.update_mesh_info)
        
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
        # === MATERIALI ===
        mat_group = QGroupBox("Materiali")
        mat_layout = QGridLayout()
        
        mat_layout.addWidget(QLabel("Stoccaggio:"), 0, 0)
        self.storage_combo = QComboBox()
        self.storage_combo.addItems([
            "steatite", "silica_sand", "olivine", 
            "basalt", "magnetite", "quartzite", "granite"
        ])
        mat_layout.addWidget(self.storage_combo, 0, 1)
        
        mat_layout.addWidget(QLabel("Isolamento:"), 1, 0)
        self.insulation_combo = QComboBox()
        self.insulation_combo.addItems([
            "rock_wool", "glass_wool", "calcium_silicate",
            "ceramic_fiber", "perlite"
        ])
        mat_layout.addWidget(self.insulation_combo, 1, 1)
        
        mat_layout.addWidget(QLabel("Packing [%]:"), 2, 0)
        self.packing_spin = QSpinBox()
        self.packing_spin.setRange(50, 75)
        self.packing_spin.setValue(63)
        mat_layout.addWidget(self.packing_spin, 2, 1)
        
        mat_group.setLayout(mat_layout)
        layout.addWidget(mat_group)
        
        # === CONDIZIONI OPERATIVE ===
        op_group = QGroupBox("Condizioni Operative")
        op_layout = QGridLayout()
        
        op_layout.addWidget(QLabel("T ambiente [°C]:"), 0, 0)
        self.t_amb_spin = QDoubleSpinBox()
        self.t_amb_spin.setRange(-20, 50)
        self.t_amb_spin.setValue(20)
        op_layout.addWidget(self.t_amb_spin, 0, 1)
        
        op_layout.addWidget(QLabel("T terreno [°C]:"), 1, 0)
        self.t_ground_spin = QDoubleSpinBox()
        self.t_ground_spin.setRange(0, 30)
        self.t_ground_spin.setValue(10)
        op_layout.addWidget(self.t_ground_spin, 1, 1)
        
        op_group.setLayout(op_layout)
        layout.addWidget(op_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _create_heater_tab(self) -> QWidget:
        """Crea il tab per configurazione resistenze"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # === POTENZA ===
        power_group = QGroupBox("Potenza")
        power_layout = QGridLayout()
        
        power_layout.addWidget(QLabel("Potenza totale [kW]:"), 0, 0)
        self.power_spin = QDoubleSpinBox()
        self.power_spin.setRange(1, 10000)
        self.power_spin.setValue(50)
        self.power_spin.setDecimals(1)
        power_layout.addWidget(self.power_spin, 0, 1)
        
        power_group.setLayout(power_layout)
        layout.addWidget(power_group)
        
        # === PATTERN RESISTENZE ===
        pattern_group = QGroupBox("Pattern Distribuzione")
        pattern_layout = QVBoxLayout()
        
        self.heater_pattern_combo = QComboBox()
        self.heater_pattern_combo.addItems([
            "Zona Uniforme (default)",
            "Griglia Verticale",
            "Pattern a Scacchiera",
            "Array Radiale (come foto)",
            "Spirale",
            "Anelli Concentrici",
            "Personalizzato"
        ])
        self.heater_pattern_combo.currentIndexChanged.connect(self._on_heater_pattern_changed)
        pattern_layout.addWidget(self.heater_pattern_combo)
        
        # Descrizione pattern
        self.heater_pattern_desc = QLabel(
            "La zona resistenze viene trattata come sorgente\n"
            "di calore uniforme nell'anello radiale."
        )
        self.heater_pattern_desc.setStyleSheet("color: gray; font-size: 9px;")
        self.heater_pattern_desc.setWordWrap(True)
        pattern_layout.addWidget(self.heater_pattern_desc)
        
        pattern_group.setLayout(pattern_layout)
        layout.addWidget(pattern_group)
        
        # === PARAMETRI ELEMENTI ===
        elem_group = QGroupBox("Parametri Elementi")
        elem_layout = QGridLayout()
        
        elem_layout.addWidget(QLabel("N° resistenze:"), 0, 0)
        self.n_heaters_spin = QSpinBox()
        self.n_heaters_spin.setRange(1, 100)
        self.n_heaters_spin.setValue(12)
        elem_layout.addWidget(self.n_heaters_spin, 0, 1)
        
        elem_layout.addWidget(QLabel("Raggio elemento [m]:"), 1, 0)
        self.heater_radius_spin = QDoubleSpinBox()
        self.heater_radius_spin.setRange(0.005, 0.1)
        self.heater_radius_spin.setValue(0.02)
        self.heater_radius_spin.setDecimals(3)
        self.heater_radius_spin.setSingleStep(0.005)
        elem_layout.addWidget(self.heater_radius_spin, 1, 1)
        
        # Potenza per elemento (calcolata)
        elem_layout.addWidget(QLabel("Potenza/elemento:"), 2, 0)
        self.power_per_heater_label = QLabel("-")
        elem_layout.addWidget(self.power_per_heater_label, 2, 1)
        
        self.power_spin.valueChanged.connect(self._update_power_per_heater)
        self.n_heaters_spin.valueChanged.connect(self._update_power_per_heater)
        
        elem_group.setLayout(elem_layout)
        layout.addWidget(elem_group)
        
        # === PARAMETRI GRIGLIA (per pattern a griglia) ===
        grid_group = QGroupBox("Parametri Griglia/Anelli")
        grid_layout = QGridLayout()
        
        grid_layout.addWidget(QLabel("Righe griglia:"), 0, 0)
        self.heater_grid_rows = QSpinBox()
        self.heater_grid_rows.setRange(1, 20)
        self.heater_grid_rows.setValue(4)
        grid_layout.addWidget(self.heater_grid_rows, 0, 1)
        
        grid_layout.addWidget(QLabel("Colonne griglia:"), 1, 0)
        self.heater_grid_cols = QSpinBox()
        self.heater_grid_cols.setRange(1, 20)
        self.heater_grid_cols.setValue(4)
        grid_layout.addWidget(self.heater_grid_cols, 1, 1)
        
        grid_layout.addWidget(QLabel("Spaziatura [m]:"), 2, 0)
        self.heater_grid_spacing = QDoubleSpinBox()
        self.heater_grid_spacing.setRange(0.1, 2.0)
        self.heater_grid_spacing.setValue(0.3)
        self.heater_grid_spacing.setDecimals(2)
        grid_layout.addWidget(self.heater_grid_spacing, 2, 1)
        
        grid_layout.addWidget(QLabel("N° anelli:"), 3, 0)
        self.heater_n_rings = QSpinBox()
        self.heater_n_rings.setRange(1, 10)
        self.heater_n_rings.setValue(2)
        grid_layout.addWidget(self.heater_n_rings, 3, 1)
        
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)
        
        # === PREVIEW POSIZIONI ===
        preview_group = QGroupBox("Anteprima Posizioni")
        preview_layout = QVBoxLayout()
        
        self.heater_preview_btn = QPushButton("Calcola posizioni")
        self.heater_preview_btn.clicked.connect(self._preview_heater_positions)
        preview_layout.addWidget(self.heater_preview_btn)
        
        self.heater_positions_list = QListWidget()
        self.heater_positions_list.setMaximumHeight(120)
        preview_layout.addWidget(self.heater_positions_list)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _create_tube_tab(self) -> QWidget:
        """Crea il tab per configurazione tubi scambiatori"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # === STATO TUBI ===
        state_group = QGroupBox("Stato Scambiatori")
        state_layout = QGridLayout()
        
        self.tubes_active_check = QCheckBox("Tubi attivi (scarica)")
        state_layout.addWidget(self.tubes_active_check, 0, 0, 1, 2)
        
        state_layout.addWidget(QLabel("T fluido [°C]:"), 1, 0)
        self.tube_t_fluid_spin = QDoubleSpinBox()
        self.tube_t_fluid_spin.setRange(10, 200)
        self.tube_t_fluid_spin.setValue(60)
        state_layout.addWidget(self.tube_t_fluid_spin, 1, 1)
        
        state_layout.addWidget(QLabel("h convettivo [W/m²K]:"), 2, 0)
        self.tube_h_fluid_spin = QDoubleSpinBox()
        self.tube_h_fluid_spin.setRange(10, 5000)
        self.tube_h_fluid_spin.setValue(500)
        state_layout.addWidget(self.tube_h_fluid_spin, 2, 1)
        
        state_group.setLayout(state_layout)
        layout.addWidget(state_group)
        
        # === PATTERN TUBI ===
        pattern_group = QGroupBox("Pattern Distribuzione")
        pattern_layout = QVBoxLayout()
        
        self.tube_pattern_combo = QComboBox()
        self.tube_pattern_combo.addItems([
            "Cluster Centrale",
            "Array Radiale",
            "Griglia",
            "Esagonale (alta densità)",
            "Singolo Centrale",
            "Personalizzato"
        ])
        self.tube_pattern_combo.setCurrentIndex(1)  # Default: Array Radiale
        self.tube_pattern_combo.currentIndexChanged.connect(self._on_tube_pattern_changed)
        pattern_layout.addWidget(self.tube_pattern_combo)
        
        # Descrizione pattern
        self.tube_pattern_desc = QLabel(
            "I tubi sono disposti in array radiale\n"
            "attorno al centro della batteria."
        )
        self.tube_pattern_desc.setStyleSheet("color: gray; font-size: 9px;")
        self.tube_pattern_desc.setWordWrap(True)
        pattern_layout.addWidget(self.tube_pattern_desc)
        
        pattern_group.setLayout(pattern_layout)
        layout.addWidget(pattern_group)
        
        # === PARAMETRI ELEMENTI ===
        elem_group = QGroupBox("Parametri Elementi")
        elem_layout = QGridLayout()
        
        elem_layout.addWidget(QLabel("N° tubi:"), 0, 0)
        self.n_tubes_spin = QSpinBox()
        self.n_tubes_spin.setRange(1, 50)
        self.n_tubes_spin.setValue(8)
        elem_layout.addWidget(self.n_tubes_spin, 0, 1)
        
        elem_layout.addWidget(QLabel("Diametro tubo [m]:"), 1, 0)
        self.tube_diameter_spin = QDoubleSpinBox()
        self.tube_diameter_spin.setRange(0.01, 0.2)
        self.tube_diameter_spin.setValue(0.05)
        self.tube_diameter_spin.setDecimals(3)
        self.tube_diameter_spin.setSingleStep(0.01)
        elem_layout.addWidget(self.tube_diameter_spin, 1, 1)
        
        elem_group.setLayout(elem_layout)
        layout.addWidget(elem_group)
        
        # === PARAMETRI GRIGLIA/ANELLI ===
        grid_group = QGroupBox("Parametri Griglia/Anelli")
        grid_layout = QGridLayout()
        
        grid_layout.addWidget(QLabel("Righe griglia:"), 0, 0)
        self.tube_grid_rows = QSpinBox()
        self.tube_grid_rows.setRange(1, 10)
        self.tube_grid_rows.setValue(3)
        grid_layout.addWidget(self.tube_grid_rows, 0, 1)
        
        grid_layout.addWidget(QLabel("Colonne griglia:"), 1, 0)
        self.tube_grid_cols = QSpinBox()
        self.tube_grid_cols.setRange(1, 10)
        self.tube_grid_cols.setValue(3)
        grid_layout.addWidget(self.tube_grid_cols, 1, 1)
        
        grid_layout.addWidget(QLabel("Spaziatura [m]:"), 2, 0)
        self.tube_grid_spacing = QDoubleSpinBox()
        self.tube_grid_spacing.setRange(0.05, 1.0)
        self.tube_grid_spacing.setValue(0.2)
        self.tube_grid_spacing.setDecimals(2)
        grid_layout.addWidget(self.tube_grid_spacing, 2, 1)
        
        grid_layout.addWidget(QLabel("N° anelli:"), 3, 0)
        self.tube_n_rings = QSpinBox()
        self.tube_n_rings.setRange(1, 5)
        self.tube_n_rings.setValue(2)
        grid_layout.addWidget(self.tube_n_rings, 3, 1)
        
        grid_group.setLayout(grid_layout)
        layout.addWidget(grid_group)
        
        # === PREVIEW POSIZIONI ===
        preview_group = QGroupBox("Anteprima Posizioni")
        preview_layout = QVBoxLayout()
        
        self.tube_preview_btn = QPushButton("Calcola posizioni")
        self.tube_preview_btn.clicked.connect(self._preview_tube_positions)
        preview_layout.addWidget(self.tube_preview_btn)
        
        self.tube_positions_list = QListWidget()
        self.tube_positions_list.setMaximumHeight(120)
        preview_layout.addWidget(self.tube_positions_list)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _create_solver_tab(self) -> QWidget:
        """Crea il tab per configurazione solver"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # === SOLVER ===
        solver_group = QGroupBox("Metodo di Soluzione")
        solver_layout = QGridLayout()
        
        solver_layout.addWidget(QLabel("Metodo:"), 0, 0)
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["direct", "bicgstab", "gmres", "cg"])
        solver_layout.addWidget(self.solver_combo, 0, 1)
        
        solver_layout.addWidget(QLabel("Tolleranza:"), 1, 0)
        self.tolerance_spin = QDoubleSpinBox()
        self.tolerance_spin.setRange(1e-12, 1e-2)
        self.tolerance_spin.setValue(1e-8)
        self.tolerance_spin.setDecimals(10)
        self.tolerance_spin.setSingleStep(1e-9)
        solver_layout.addWidget(self.tolerance_spin, 1, 1)
        
        solver_layout.addWidget(QLabel("Max iterazioni:"), 2, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 100000)
        self.max_iter_spin.setValue(10000)
        solver_layout.addWidget(self.max_iter_spin, 2, 1)
        
        solver_group.setLayout(solver_layout)
        layout.addWidget(solver_group)
        
        # === INFO SOLVER ===
        info_group = QGroupBox("Info Metodi")
        info_layout = QVBoxLayout()
        
        info_text = QLabel(
            "<b>direct:</b> Soluzione diretta (LU). Robusto, più lento per mesh grandi.<br>"
            "<b>bicgstab:</b> BiCGSTAB iterativo. Veloce per mesh grandi.<br>"
            "<b>gmres:</b> GMRES iterativo. Buona convergenza.<br>"
            "<b>cg:</b> Gradiente Coniugato. Solo per matrici simmetriche."
        )
        info_text.setWordWrap(True)
        info_text.setStyleSheet("color: gray; font-size: 9px;")
        info_layout.addWidget(info_text)
        
        info_group.setLayout(info_layout)
        layout.addWidget(info_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _on_heater_pattern_changed(self, index: int):
        """Callback quando cambia il pattern resistenze"""
        descriptions = [
            "La zona resistenze viene trattata come sorgente\ndi calore uniforme nell'anello radiale.",
            "Resistenze verticali disposte in una griglia\nregolare. Usa parametri righe/colonne.",
            "Resistenze in pattern a scacchiera.\nMaggiore uniformità di riscaldamento.",
            "Resistenze disposte su anelli concentrici\ncome elementi tubolari (vedi foto).",
            "Resistenze disposte a spirale dal centro\nverso l'esterno della zona.",
            "Resistenze su anelli concentrici uniformi\ncon offset alternato.",
            "Posizioni definite manualmente.\nEsporta le coordinate calcolate."
        ]
        self.heater_pattern_desc.setText(descriptions[index])
    
    def _on_tube_pattern_changed(self, index: int):
        """Callback quando cambia il pattern tubi"""
        descriptions = [
            "Cluster di tubi raggruppati al centro\ndella batteria.",
            "Tubi disposti su anelli concentrici\nattorno al centro.",
            "Tubi in griglia regolare quadrata.",
            "Pattern esagonale per massima densità\ndi impaccamento.",
            "Singolo tubo centrale per configurazioni\nsemplici.",
            "Posizioni definite manualmente."
        ]
        self.tube_pattern_desc.setText(descriptions[index])
    
    def _update_power_per_heater(self):
        """Aggiorna la potenza per resistenza"""
        power = self.power_spin.value()
        n = self.n_heaters_spin.value()
        power_per = power / max(n, 1)
        self.power_per_heater_label.setText(f"{power_per:.2f} kW")
    
    def _get_heater_pattern_enum(self) -> str:
        """Converte l'indice combo in pattern enum"""
        patterns = [
            HeaterPattern.UNIFORM_ZONE,
            HeaterPattern.GRID_VERTICAL,
            HeaterPattern.CHESS_PATTERN,
            HeaterPattern.RADIAL_ARRAY,
            HeaterPattern.SPIRAL,
            HeaterPattern.CONCENTRIC_RINGS,
            HeaterPattern.CUSTOM
        ]
        return patterns[self.heater_pattern_combo.currentIndex()]
    
    def _get_tube_pattern_enum(self) -> str:
        """Converte l'indice combo in pattern enum"""
        patterns = [
            TubePattern.CENTRAL_CLUSTER,
            TubePattern.RADIAL_ARRAY,
            TubePattern.GRID,
            TubePattern.HEXAGONAL,
            TubePattern.SINGLE_CENTRAL,
            TubePattern.CUSTOM
        ]
        return patterns[self.tube_pattern_combo.currentIndex()]
    
    def _preview_heater_positions(self):
        """Mostra anteprima posizioni resistenze"""
        self.heater_positions_list.clear()
        
        # Crea config temporanea
        config = HeaterConfig(
            power_total=self.power_spin.value(),
            n_heaters=self.n_heaters_spin.value(),
            pattern=self._get_heater_pattern_enum(),
            heater_radius=self.heater_radius_spin.value(),
            grid_rows=self.heater_grid_rows.value(),
            grid_cols=self.heater_grid_cols.value(),
            grid_spacing=self.heater_grid_spacing.value(),
            n_rings=self.heater_n_rings.value()
        )
        
        # Parametri geometrici approssimati
        Lx, Ly = self.lx_spin.value(), self.ly_spin.value()
        center_x, center_y = Lx / 2, Ly / 2
        radius = self.radius_spin.value()
        r_inner = radius * 0.4
        r_outer = radius * 0.5
        z_bottom = 0.3
        z_top = 0.3 + self.height_spin.value()
        
        elements = config.generate_positions(
            center_x, center_y,
            r_inner, r_outer,
            z_bottom, z_top
        )
        
        if not elements:
            self.heater_positions_list.addItem("Pattern uniforme (nessun elemento discreto)")
        else:
            for i, elem in enumerate(elements):
                item = QListWidgetItem(f"{i+1}: ({elem.x:.2f}, {elem.y:.2f}) - {elem.power:.1f} kW")
                self.heater_positions_list.addItem(item)
            self.heater_positions_list.addItem(f"--- Totale: {len(elements)} elementi ---")
    
    def _preview_tube_positions(self):
        """Mostra anteprima posizioni tubi"""
        self.tube_positions_list.clear()
        
        # Crea config temporanea
        config = TubeConfig(
            n_tubes=self.n_tubes_spin.value(),
            diameter=self.tube_diameter_spin.value(),
            h_fluid=self.tube_h_fluid_spin.value(),
            T_fluid=self.tube_t_fluid_spin.value(),
            active=self.tubes_active_check.isChecked(),
            pattern=self._get_tube_pattern_enum(),
            grid_rows=self.tube_grid_rows.value(),
            grid_cols=self.tube_grid_cols.value(),
            grid_spacing=self.tube_grid_spacing.value(),
            n_rings=self.tube_n_rings.value()
        )
        
        # Parametri geometrici approssimati
        Lx, Ly = self.lx_spin.value(), self.ly_spin.value()
        center_x, center_y = Lx / 2, Ly / 2
        radius = self.radius_spin.value()
        r_max = radius * 0.3  # Zona tubi centrale
        z_bottom = 0.3
        z_top = 0.3 + self.height_spin.value()
        
        elements = config.generate_positions(
            center_x, center_y,
            r_max,
            z_bottom, z_top
        )
        
        for i, elem in enumerate(elements):
            item = QListWidgetItem(f"{i+1}: ({elem.x:.2f}, {elem.y:.2f}) - Ø{elem.radius*2*1000:.0f}mm")
            self.tube_positions_list.addItem(item)
        self.tube_positions_list.addItem(f"--- Totale: {len(elements)} tubi ---")
    
    def create_center_panel(self) -> QWidget:
        """Crea il pannello centrale con la visualizzazione 3D"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Frame per PyVista
        self.plotter_frame = QFrame()
        self.plotter_frame.setMinimumSize(600, 500)
        
        plotter_layout = QVBoxLayout(self.plotter_frame)
        
        # Inizializza PyVista Plotter embedded
        self.plotter = QtInteractor(self.plotter_frame)
        plotter_layout.addWidget(self.plotter.interactor)
        
        layout.addWidget(self.plotter_frame, stretch=1)
        
        # Controlli visualizzazione
        viz_group = QGroupBox("Controlli Visualizzazione")
        viz_layout = QGridLayout()
        
        # Riga 1: Modalità visualizzazione
        viz_layout.addWidget(QLabel("Modalità:"), 0, 0)
        self.viz_mode_combo = QComboBox()
        self.viz_mode_combo.addItems([
            "Sezione Clip",      # Taglia e mostra solido dietro
            "Multi-Slice",       # Più fette parallele
            "Volume 3D",         # Volume rendering trasparente
            "Isosuperficie"      # Superfici a temperatura costante
        ])
        self.viz_mode_combo.currentTextChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.viz_mode_combo, 0, 1, 1, 2)
        
        # Campo da visualizzare
        viz_layout.addWidget(QLabel("Campo:"), 0, 3)
        self.field_combo = QComboBox()
        self.field_combo.addItems(["Temperature", "Material", "k", "Q"])
        self.field_combo.currentTextChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.field_combo, 0, 4)
        
        # Riga 2: Controlli slice/clip
        viz_layout.addWidget(QLabel("Asse:"), 1, 0)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        self.axis_combo.setCurrentIndex(2)
        self.axis_combo.currentTextChanged.connect(self.update_slice_range)
        viz_layout.addWidget(self.axis_combo, 1, 1)
        
        viz_layout.addWidget(QLabel("Posizione:"), 1, 2)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, 100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self.update_slice_position)
        viz_layout.addWidget(self.slice_slider, 1, 3)
        
        self.slice_pos_label = QLabel("0.00 m")
        viz_layout.addWidget(self.slice_pos_label, 1, 4)
        
        # Riga 3: Opzioni avanzate per Volume 3D e Isosuperficie
        viz_layout.addWidget(QLabel("Opacità:"), 2, 0)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(30)
        self.opacity_slider.valueChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.opacity_slider, 2, 1, 1, 2)
        
        self.opacity_label = QLabel("30%")
        viz_layout.addWidget(self.opacity_label, 2, 3)
        
        # Numero isosuperfici
        viz_layout.addWidget(QLabel("N. iso:"), 2, 4)
        self.n_iso_spin = QSpinBox()
        self.n_iso_spin.setRange(1, 10)
        self.n_iso_spin.setValue(5)
        self.n_iso_spin.valueChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.n_iso_spin, 2, 5)
        
        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)
        
        # Colormap controls
        cmap_group = QGroupBox("Colormap")
        cmap_layout = QHBoxLayout()
        
        cmap_layout.addWidget(QLabel("Colormap:"))
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems(["coolwarm", "jet", "viridis", "plasma", "inferno", "turbo"])
        self.cmap_combo.currentTextChanged.connect(self.update_visualization)
        cmap_layout.addWidget(self.cmap_combo)
        
        cmap_layout.addWidget(QLabel("  T min:"))
        self.tmin_spin = QDoubleSpinBox()
        self.tmin_spin.setRange(-100, 1000)
        self.tmin_spin.setValue(0)
        self.tmin_spin.valueChanged.connect(self.update_visualization)
        cmap_layout.addWidget(self.tmin_spin)
        
        cmap_layout.addWidget(QLabel("  T max:"))
        self.tmax_spin = QDoubleSpinBox()
        self.tmax_spin.setRange(0, 2000)
        self.tmax_spin.setValue(500)
        self.tmax_spin.valueChanged.connect(self.update_visualization)
        cmap_layout.addWidget(self.tmax_spin)
        
        self.auto_range_check = QCheckBox("Auto")
        self.auto_range_check.setChecked(True)
        cmap_layout.addWidget(self.auto_range_check)
        
        cmap_group.setLayout(cmap_layout)
        layout.addWidget(cmap_group)
        
        return panel
    
    def create_right_panel(self) -> QWidget:
        """Crea il pannello destro con i risultati"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Tab per diversi risultati
        tabs = QTabWidget()
        
        # Tab Statistiche
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 10))
        stats_layout.addWidget(self.stats_text)
        
        tabs.addTab(stats_tab, "Statistiche")
        
        # Tab Bilancio Energetico
        energy_tab = QWidget()
        energy_layout = QVBoxLayout(energy_tab)
        
        self.energy_text = QTextEdit()
        self.energy_text.setReadOnly(True)
        self.energy_text.setFont(QFont("Consolas", 10))
        energy_layout.addWidget(self.energy_text)
        
        tabs.addTab(energy_tab, "Bilancio Energetico")
        
        # Tab Materiali
        mat_tab = QWidget()
        mat_layout = QVBoxLayout(mat_tab)
        
        self.mat_text = QTextEdit()
        self.mat_text.setReadOnly(True)
        self.mat_text.setFont(QFont("Consolas", 10))
        mat_layout.addWidget(self.mat_text)
        
        tabs.addTab(mat_tab, "Materiali")
        
        # Tab Log
        log_tab = QWidget()
        log_layout = QVBoxLayout(log_tab)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        
        tabs.addTab(log_tab, "Log")
        
        layout.addWidget(tabs)
        
        return panel
    
    def init_default_values(self):
        """Inizializza i valori di default"""
        self.update_mesh_info()
        self._update_power_per_heater()
        self._on_heater_pattern_changed(0)
        self._on_tube_pattern_changed(1)  # Default: Array Radiale
        self.show_initial_message()
        
    def show_initial_message(self):
        """Mostra messaggio iniziale nel plotter"""
        self.plotter.add_text(
            "Sand Battery Thermal Simulation\n\n"
            "1. Configura i parametri nel pannello sinistro\n"
            "2. Premi 'Costruisci Mesh' per visualizzare\n"
            "3. Premi 'Esegui Simulazione' per calcolare",
            position='upper_left',
            font_size=12,
            color='gray'
        )
    
    def update_mesh_info(self):
        """Aggiorna le informazioni sulla mesh"""
        d = self.spacing_spin.value()
        Lx, Ly, Lz = self.lx_spin.value(), self.ly_spin.value(), self.lz_spin.value()
        
        Nx = int(np.ceil(Lx / d))
        Ny = int(np.ceil(Ly / d))
        Nz = int(np.ceil(Lz / d))
        N_total = Nx * Ny * Nz
        
        self.mesh_info_label.setText(f"Celle: {Nx}×{Ny}×{Nz} = {N_total:,}")
        
        # Stima memoria: T, k, rho, cp, Q, material_id = 5*float64 + 1*int32
        # Plus matrice sparsa stimata ~10 nonzero per riga
        bytes_arrays = N_total * (5 * 8 + 4)  # 5 float64 + 1 int32
        bytes_matrix = N_total * 10 * 12  # ~10 nonzero per riga, 12 bytes ciascuno
        total_bytes = bytes_arrays + bytes_matrix
        
        if total_bytes < 1024:
            mem_str = f"{total_bytes} B"
        elif total_bytes < 1024**2:
            mem_str = f"{total_bytes/1024:.1f} KB"
        elif total_bytes < 1024**3:
            mem_str = f"{total_bytes/1024**2:.1f} MB"
        else:
            mem_str = f"{total_bytes/1024**3:.2f} GB"
        
        self.memory_label.setText(f"Memoria stimata: ~{mem_str}")
        
    def log(self, message: str):
        """Aggiunge messaggio al log"""
        self.log_text.append(message)
    
    def build_mesh(self):
        """Costruisce la mesh e applica la geometria (senza eseguire la simulazione)"""
        self.log("=" * 50)
        self.log("Costruzione mesh...")
        self.progress_bar.setValue(0)
        self.status_label.setText("Stato: Creazione mesh...")
        
        try:
            d, Lx, Ly, Lz, geom = self._build_battery_geometry_from_inputs()
            self.battery_geometry = geom
            self.log(f"Dominio: {Lx}×{Ly}×{Lz} m, spaziatura: {d} m")

            # Crea mesh uniforme
            self.mesh = Mesh3D(Lx=Lx, Ly=Ly, Lz=Lz, target_spacing=d)
            self.log(f"Mesh creata: {self.mesh.Nx}×{self.mesh.Ny}×{self.mesh.Nz} celle")
            
            self.progress_bar.setValue(50)
            self.status_label.setText("Stato: Applicazione geometria...")
            
            # Applica geometria
            self.battery_geometry.apply_to_mesh(self.mesh, self.mat_manager)
            
            self.log("Geometria applicata")
            
            # Log numero elementi discreti (accedi tramite l'oggetto geom)
            if geom.heaters.elements:
                self.log(f"  Resistenze discrete: {len(geom.heaters.elements)}")
            if geom.tubes.elements:
                self.log(f"  Tubi discreti: {len(geom.tubes.elements)}")
            
            self.update_material_info()
            
            self.progress_bar.setValue(100)
            self.status_label.setText("Stato: Mesh pronta - puoi visualizzare o eseguire simulazione")
            self.log("Mesh costruita con successo!")
            
            # Abilita il pulsante simulazione
            self.run_btn.setEnabled(True)
            
            # Mostra la mesh (materiali)
            self.field_combo.setCurrentText("Material")
            self.update_visualization()
            
        except Exception as e:
            self.log(f"ERRORE: {e}")
            self.status_label.setText(f"Errore: {e}")
            QMessageBox.critical(self, "Errore", str(e))
            self.run_btn.setEnabled(False)

    def _build_battery_geometry_from_inputs(self):
        """
        CENTRAL CONFIGURATION FUNCTION - Reads ALL GUI widgets and creates config objects.
        
        This is the single function that bridges GUI widgets to simulation parameters.
        It reads every relevant widget and creates:
        - CylinderGeometry: radial zone definitions
        - HeaterConfig: heater pattern, power, positions
        - TubeConfig: tube pattern, h_fluid, T_fluid, positions
        - BatteryGeometry: combines all above with material selections
        
        Returns:
            tuple: (spacing, Lx, Ly, Lz, BatteryGeometry)
        
        NOTE: This function does NOT create the mesh - it only builds the configuration.
        The actual mesh creation happens in build_mesh() or is skipped in preview_geometry().
        """
        # ==========================================================================
        # STEP 1: Read domain and mesh parameters from GUI
        # ==========================================================================
        d = self.spacing_spin.value()
        Lx, Ly, Lz = self.lx_spin.value(), self.ly_spin.value(), self.lz_spin.value()

        center_x, center_y = Lx / 2, Ly / 2
        radius = self.radius_spin.value()
        height = self.height_spin.value()

        cylinder = CylinderGeometry(
            center_x=center_x,
            center_y=center_y,
            base_z=0.3,
            height=height,
            r_tubes=0.3,
            r_sand_inner=radius * 0.4,
            r_heaters=radius * 0.5,
            r_sand_outer=radius * 0.85,
            r_insulation=radius,
            r_shell=radius + 0.02,
        )

        heater_config = HeaterConfig(
            power_total=self.power_spin.value(),
            n_heaters=self.n_heaters_spin.value(),
            pattern=self._get_heater_pattern_enum(),
            heater_radius=self.heater_radius_spin.value(),
            grid_rows=self.heater_grid_rows.value(),
            grid_cols=self.heater_grid_cols.value(),
            grid_spacing=self.heater_grid_spacing.value(),
            n_rings=self.heater_n_rings.value()
        )

        tube_config = TubeConfig(
            n_tubes=self.n_tubes_spin.value(),
            diameter=self.tube_diameter_spin.value(),
            h_fluid=self.tube_h_fluid_spin.value(),
            T_fluid=self.tube_t_fluid_spin.value(),
            active=self.tubes_active_check.isChecked(),
            pattern=self._get_tube_pattern_enum(),
            grid_rows=self.tube_grid_rows.value(),
            grid_cols=self.tube_grid_cols.value(),
            grid_spacing=self.tube_grid_spacing.value(),
            n_rings=self.tube_n_rings.value()
        )

        self.log(f"Heater pattern: {heater_config.pattern}")
        self.log(f"Tube pattern: {tube_config.pattern}")

        geom = BatteryGeometry(
            cylinder=cylinder,
            heaters=heater_config,
            tubes=tube_config,
            storage_material=self.storage_combo.currentText(),
            insulation_material=self.insulation_combo.currentText(),
            packing_fraction=self.packing_spin.value() / 100.0,
        )

        return d, Lx, Ly, Lz, geom

    def preview_geometry(self):
        """Visualizza la geometria analitica (senza costruire la mesh)."""
        self.log("=" * 50)
        self.log("Anteprima geometria (senza mesh)...")

        try:
            _, Lx, Ly, Lz, geom = self._build_battery_geometry_from_inputs()
            self.battery_geometry = geom

            cyl = geom.cylinder

            self.plotter.clear()
            self.plotter.add_text(
                "Anteprima Geometria (no mesh)\n"
                "- Strati cilindrici\n"
                "- Tubi / resistenze (se presenti)",
                position='upper_left',
                font_size=12,
                color='gray'
            )

            # Outline del dominio
            domain = pv.Cube(
                center=(Lx / 2, Ly / 2, Lz / 2),
                x_length=Lx,
                y_length=Ly,
                z_length=Lz
            )
            self.plotter.add_mesh(domain.outline(), color='gray', line_width=1)

            # Strati: mostra solo i contorni dei raggi principali
            z_center = cyl.base_z + cyl.height / 2
            radii = [
                cyl.r_tubes,
                cyl.r_sand_inner,
                cyl.r_heaters,
                cyl.r_sand_outer,
                cyl.r_insulation,
                cyl.r_shell,
            ]
            for r in radii:
                c = pv.Cylinder(
                    center=(cyl.center_x, cyl.center_y, z_center),
                    direction=(0, 0, 1),
                    radius=float(r),
                    height=float(cyl.height),
                    resolution=72
                )
                self.plotter.add_mesh(c, color='gray', style='wireframe', line_width=1)

            # Elementi discreti: heaters
            heater_elems = []
            if geom.heaters.pattern != HeaterPattern.UNIFORM_ZONE:
                heater_elems = geom.heaters.generate_positions(
                    cyl.center_x, cyl.center_y,
                    cyl.r_sand_inner, cyl.r_heaters,
                    cyl.base_z, cyl.top_z
                )
            for htr in heater_elems:
                h_center = (htr.x, htr.y, (htr.z_bottom + htr.z_top) / 2)
                h_cyl = pv.Cylinder(
                    center=h_center,
                    direction=(0, 0, 1),
                    radius=float(htr.radius),
                    height=float(htr.z_top - htr.z_bottom),
                    resolution=36
                )
                self.plotter.add_mesh(h_cyl, color='gray', opacity=0.6)

            # Elementi discreti: tubes
            tube_elems = geom.tubes.generate_positions(
                cyl.center_x, cyl.center_y,
                cyl.r_tubes,
                cyl.base_z, cyl.top_z
            )
            for tube in tube_elems:
                t_center = (tube.x, tube.y, (tube.z_bottom + tube.z_top) / 2)
                t_cyl = pv.Cylinder(
                    center=t_center,
                    direction=(0, 0, 1),
                    radius=float(tube.radius),
                    height=float(tube.z_top - tube.z_bottom),
                    resolution=36
                )
                self.plotter.add_mesh(t_cyl, color='gray', opacity=0.3)

            self.plotter.reset_camera()
            self.plotter.render()
            self.status_label.setText("Stato: Anteprima geometria pronta")

        except Exception as e:
            self.log(f"ERRORE anteprima geometria: {e}")
            self.status_label.setText(f"Errore: {e}")
            QMessageBox.critical(self, "Errore", str(e))
        
    def run_simulation(self):
        """Esegue la simulazione termica sulla mesh esistente"""
        if self.mesh is None:
            QMessageBox.warning(self, "Attenzione", "Prima costruisci la mesh!")
            return
        
        self.log("=" * 50)
        self.log("Avvio simulazione termica...")
        self.progress_bar.setValue(0)
        self.status_label.setText("Stato: Configurazione solver...")
        
        try:
            # Configura solver
            config = SolverConfig(
                method=self.solver_combo.currentText(),
                tolerance=1e-8,
                verbose=False
            )
            
            self.progress_bar.setValue(10)
            
            # Esegui simulazione in thread separato
            self.simulation_thread = SimulationThread(self.mesh, config)
            self.simulation_thread.progress.connect(self.on_simulation_progress)
            self.simulation_thread.finished.connect(self.on_simulation_finished)
            self.simulation_thread.error.connect(self.on_simulation_error)
            self.simulation_thread.start()
            
            self.run_btn.setEnabled(False)
            self.build_mesh_btn.setEnabled(False)
            
        except Exception as e:
            self.log(f"ERRORE: {e}")
            self.status_label.setText(f"Errore: {e}")
            QMessageBox.critical(self, "Errore", str(e))
    
    def on_simulation_progress(self, value: int, message: str):
        """Callback per il progresso della simulazione"""
        self.progress_bar.setValue(value)
        self.status_label.setText(f"Stato: {message}")
        self.log(message)
    
    def on_simulation_finished(self, result):
        """Callback per il completamento della simulazione"""
        self.progress_bar.setValue(100)
        self.status_label.setText("Stato: Completato!")
        self.run_btn.setEnabled(True)
        self.build_mesh_btn.setEnabled(True)
        
        # Segna che la simulazione è stata completata
        self._simulation_completed = True
        
        self.log(f"Simulazione completata!")
        self.log(f"  Convergenza: {result.converged}")
        self.log(f"  Tempo: {result.solve_time:.2f} s")
        self.log(f"  Residuo: {result.residual:.2e}")
        
        # Aggiorna statistiche
        self.update_statistics()
        
        # Aggiorna visualizzazione
        self.update_visualization()
        
        # Aggiorna bilancio energetico
        self.update_energy_balance()
    
    def on_simulation_error(self, error_msg: str):
        """Callback per errori durante la simulazione"""
        self.progress_bar.setValue(0)
        self.status_label.setText("Stato: Errore")
        self.run_btn.setEnabled(True)
        self.build_mesh_btn.setEnabled(True)
        self.log(f"ERRORE: {error_msg}")
        QMessageBox.critical(self, "Errore Simulazione", error_msg)
    
    def update_statistics(self):
        """Aggiorna il pannello statistiche"""
        if self.mesh is None:
            return
        
        T = self.mesh.T
        
        text = f"""
╔══════════════════════════════════════╗
║     STATISTICHE TEMPERATURA          ║
╠══════════════════════════════════════╣
║ T minima:     {T.min():8.1f} °C         ║
║ T massima:    {T.max():8.1f} °C         ║
║ T media:      {T.mean():8.1f} °C         ║
║ Deviazione:   {T.std():8.1f} °C         ║
╠══════════════════════════════════════╣
║ Mediana:      {np.median(T):8.1f} °C         ║
║ 10° perc:     {np.percentile(T, 10):8.1f} °C         ║
║ 90° perc:     {np.percentile(T, 90):8.1f} °C         ║
╚══════════════════════════════════════╝

MESH:
  Dimensioni: {self.mesh.Nx} × {self.mesh.Ny} × {self.mesh.Nz}
  Nodi totali: {self.mesh.N_total:,}
  Spaziatura: {self.mesh.dx:.3f} m
"""
        self.stats_text.setText(text)
    
    def update_energy_balance(self):
        """Aggiorna il pannello bilancio energetico"""
        if self.mesh is None:
            return
        
        analyzer = PowerBalanceAnalyzer(self.mesh)
        balance = analyzer.compute_power_balance()
        stored = analyzer.compute_stored_energy()
        
        text = f"""
╔══════════════════════════════════════╗
║     BILANCIO ENERGETICO              ║
╠══════════════════════════════════════╣
║ POTENZE [kW]                         ║
╠──────────────────────────────────────╣
║ P ingresso:   {balance.P_input/1000:8.2f} kW         ║
║ P uscita:     {balance.P_output/1000:8.2f} kW         ║
║ P perdite:    {balance.P_loss_total/1000:8.2f} kW         ║
║   - Superiore:{balance.P_loss_top/1000:8.3f} kW         ║
║   - Laterale: {balance.P_loss_lateral/1000:8.3f} kW         ║
║   - Inferiore:{balance.P_loss_bottom/1000:8.3f} kW         ║
╠──────────────────────────────────────╣
║ Sbilanciamento: {balance.imbalance_pct:5.1f} %             ║
╠══════════════════════════════════════╣
║ ENERGIA IMMAGAZZINATA                ║
╠──────────────────────────────────────╣
║ E termica:    {stored['E_kWh']:8.0f} kWh         ║
║               {stored['E_MWh']:8.2f} MWh         ║
╚══════════════════════════════════════╝
"""
        self.energy_text.setText(text)
    
    def update_material_info(self):
        """Aggiorna le informazioni sui materiali"""
        if self.mesh is None:
            return
        
        unique, counts = np.unique(self.mesh.material_id, return_counts=True)
        
        text = "╔══════════════════════════════════════╗\n"
        text += "║     DISTRIBUZIONE MATERIALI          ║\n"
        text += "╠══════════════════════════════════════╣\n"
        
        for mat_id, count in zip(unique, counts):
            pct = 100 * count / self.mesh.N_total
            try:
                name = MaterialID(mat_id).name
            except:
                name = f"ID={mat_id}"
            text += f"║ {name:12s}: {count:6,} ({pct:5.1f}%)    ║\n"
        
        text += "╚══════════════════════════════════════╝\n"
        
        # Aggiungi proprietà materiali
        if self.battery_geometry:
            props = self.mat_manager.compute_packed_bed_properties(
                self.battery_geometry.storage_material,
                self.battery_geometry.packing_fraction
            )
            text += f"\n{props.name}:\n"
            text += f"  k = {props.k:.3f} W/(m·K)\n"
            text += f"  ρ = {props.rho:.0f} kg/m³\n"
            text += f"  cp = {props.cp:.0f} J/(kg·K)\n"
        
        self.mat_text.setText(text)
    
    def update_visualization(self):
        """Aggiorna la visualizzazione 3D con diverse modalità"""
        if self.mesh is None:
            return
        
        self.plotter.clear()
        
        field = self.field_combo.currentText()
        cmap = self.cmap_combo.currentText()
        viz_mode = self.viz_mode_combo.currentText()
        
        # Aggiorna label opacità
        opacity_pct = self.opacity_slider.value()
        self.opacity_label.setText(f"{opacity_pct}%")
        opacity = opacity_pct / 100.0
        
        # Crea grid PyVista
        grid = pv.ImageData(
            dimensions=(self.mesh.Nx + 1, self.mesh.Ny + 1, self.mesh.Nz + 1),
            spacing=(self.mesh.dx, self.mesh.dy, self.mesh.dz),
            origin=(0, 0, 0)
        )
        
        # Aggiungi dati
        grid.cell_data["Temperature"] = self.mesh.T.ravel(order='F')
        grid.cell_data["Material"] = self.mesh.material_id.ravel(order='F').astype(float)
        grid.cell_data["k"] = self.mesh.k.ravel(order='F')
        grid.cell_data["Q"] = self.mesh.Q.ravel(order='F')
        
        # Range colori
        if self.auto_range_check.isChecked():
            data = grid.cell_data[field]
            clim = [data.min(), data.max()]
            self.tmin_spin.setValue(clim[0])
            self.tmax_spin.setValue(clim[1])
        else:
            clim = [self.tmin_spin.value(), self.tmax_spin.value()]
        
        # Determina posizione e normale per slice/clip
        axis = self.axis_combo.currentText()
        slider_pct = self.slice_slider.value() / 100.0
        eps = 0.01
        slider_pct = max(eps, min(1.0 - eps, slider_pct))
        
        if axis == 'x':
            pos = slider_pct * self.mesh.Lx
            normal = (1, 0, 0)
            origin = (pos, self.mesh.Ly/2, self.mesh.Lz/2)
        elif axis == 'y':
            pos = slider_pct * self.mesh.Ly
            normal = (0, 1, 0)
            origin = (self.mesh.Lx/2, pos, self.mesh.Lz/2)
        else:
            pos = slider_pct * self.mesh.Lz
            normal = (0, 0, 1)
            origin = (self.mesh.Lx/2, self.mesh.Ly/2, pos)
        
        self.slice_pos_label.setText(f"{pos:.2f} m")
        
        # === MODALITÀ VISUALIZZAZIONE ===
        
        if viz_mode == "Sezione Clip":
            # CLIP: taglia il solido e mostra tutto ciò che c'è dietro il piano
            clipped = grid.clip(normal=normal, origin=origin, invert=False)
            
            # Rimuovi l'aria (Material ID = 0) per rendere trasparente
            clipped_no_air = clipped.threshold(value=0.5, scalars="Material")
            
            if field == "Material":
                self.plotter.add_mesh(clipped_no_air, scalars=field, cmap=self._get_material_cmap(), 
                                      clim=[0, 7], show_scalar_bar=False)
            else:
                self.plotter.add_mesh(clipped_no_air, scalars=field, cmap=cmap, clim=clim)
            
            # Aggiungi outline per riferimento
            self.plotter.add_mesh(grid.outline(), color='gray', line_width=1)
            
            info_text = f"Clip {axis.upper()} = {pos:.2f} m"
            
        elif viz_mode == "Multi-Slice":
            # MULTI-SLICE: più fette parallele
            n_slices = 5
            for i in range(n_slices):
                frac = (i + 1) / (n_slices + 1)
                if axis == 'x':
                    orig = (frac * self.mesh.Lx, self.mesh.Ly/2, self.mesh.Lz/2)
                elif axis == 'y':
                    orig = (self.mesh.Lx/2, frac * self.mesh.Ly, self.mesh.Lz/2)
                else:
                    orig = (self.mesh.Lx/2, self.mesh.Ly/2, frac * self.mesh.Lz)
                
                sliced = grid.slice(normal=normal, origin=orig)
                if field == "Material":
                    self.plotter.add_mesh(sliced, scalars=field, cmap=self._get_material_cmap(), 
                                          clim=[0, 7], opacity=0.9, show_scalar_bar=False)
                else:
                    self.plotter.add_mesh(sliced, scalars=field, cmap=cmap, clim=clim, opacity=0.9)
            
            info_text = f"Multi-Slice {axis.upper()} (5 sezioni)"
            
        elif viz_mode == "Volume 3D":
            # VOLUME RENDERING: solido semi-trasparente che permette di vedere l'interno
            # Escludi sempre l'aria (Material ID = 0)
            threshed = grid.threshold(value=0.5, scalars="Material")
            
            if field == "Material":
                # Per materiali, mostra solo celle non-aria con trasparenza
                self.plotter.add_mesh(threshed, scalars=field, cmap=self._get_material_cmap(),
                                      clim=[0, 7], opacity=opacity, show_scalar_bar=False)
            else:
                # Per temperatura, clip e mostra senza aria
                clipped = threshed.clip(normal='y', origin=(self.mesh.Lx/2, self.mesh.Ly/2, 0))
                self.plotter.add_mesh(clipped, scalars=field, cmap=cmap, clim=clim, 
                                      opacity=opacity, show_edges=False)
            
            # Aggiungi outline per riferimento
            self.plotter.add_mesh(grid.outline(), color='gray', line_width=1)
            
            info_text = f"Volume 3D (opacità {opacity_pct}%)"
            
        elif viz_mode == "Isosuperficie":
            # ISOSURFACE: superfici a valore costante (per temperatura)
            if field == "Material":
                # Per materiali non ha senso l'isosuperficie, usa threshold
                threshed = grid.threshold(value=0.5, scalars="Material")
                self.plotter.add_mesh(threshed, scalars=field, cmap=self._get_material_cmap(),
                                      clim=[0, 7], opacity=opacity, show_scalar_bar=False)
                info_text = "Isosuperficie (mostra materiali)"
            else:
                # Per campi scalari, crea isosuperfici
                n_iso = self.n_iso_spin.value()
                data = grid.cell_data[field]
                data_range = data.max() - data.min()
                
                if data_range > 1e-6:  # Evita divisione per zero
                    iso_values = np.linspace(data.min() + data_range*0.1, 
                                            data.max() - data_range*0.1, n_iso)
                    
                    # Converti in point data per contour
                    grid_point = grid.cell_data_to_point_data()
                    
                    for i, iso_val in enumerate(iso_values):
                        try:
                            iso = grid_point.contour([iso_val], scalars=field)
                            if iso.n_points > 0:
                                self.plotter.add_mesh(iso, scalars=field, cmap=cmap, clim=clim,
                                                      opacity=opacity, show_scalar_bar=False)
                        except Exception:
                            pass  # Ignora isosuperfici vuote
                
                # Imposta info_text anche se data_range è piccolo
                info_text = f"Isosuperfici ({n_iso} livelli)"
        
        # === ELEMENTI COMUNI ===
        self.plotter.add_axes()
        
        # Legenda/Scalar bar
        if field == "Material":
            # Nomi coerenti con MaterialID enum in mesh.py
            material_names = {
                0: "AIR", 1: "SAND", 2: "INSULATION", 3: "STEEL",
                4: "TUBES", 5: "HEATERS", 6: "GROUND", 7: "CONCRETE"
            }
            legend_entries = []
            unique_mats = np.unique(self.mesh.material_id)
            for mat_id in sorted(unique_mats):
                name = material_names.get(int(mat_id), f"ID={mat_id}")
                legend_entries.append([name, self._get_material_color(int(mat_id))])
            
            if legend_entries:
                self.plotter.add_legend(legend_entries, bcolor='white', 
                                        face='rectangle', loc='upper right')
        else:
            self.plotter.add_scalar_bar(
                title=f"{field} [°C]" if field == "Temperature" else field,
                vertical=True
            )
        
        # Info text
        self.plotter.add_text(info_text, position='upper_left', font_size=10)
        
        self.plotter.reset_camera()
    
    def _get_material_color(self, mat_id: int):
        """Restituisce il colore per un materiale dato il suo ID"""
        # Colori coerenti con MaterialID enum in mesh.py:
        # AIR=0, SAND=1, INSULATION=2, STEEL=3, TUBES=4, HEATERS=5, GROUND=6, CONCRETE=7
        colors = {
            0: [0.53, 0.81, 0.98],  # AIR - azzurro cielo
            1: [0.85, 0.65, 0.45],  # SAND - sabbia/beige
            2: [0.95, 0.85, 0.55],  # INSULATION - giallo chiaro
            3: [0.40, 0.40, 0.45],  # STEEL - grigio scuro
            4: [0.60, 0.45, 0.35],  # TUBES - marrone
            5: [0.95, 0.35, 0.25],  # HEATERS - rosso
            6: [0.45, 0.35, 0.25],  # GROUND - marrone scuro
            7: [0.55, 0.55, 0.55],  # CONCRETE - grigio
        }
        return colors.get(mat_id, [0.5, 0.5, 0.5])
    
    def _get_material_cmap(self):
        """Crea una colormap discreta per i materiali"""
        from matplotlib.colors import ListedColormap
        # Colori coerenti con MaterialID enum in mesh.py:
        # AIR=0, SAND=1, INSULATION=2, STEEL=3, TUBES=4, HEATERS=5, GROUND=6, CONCRETE=7
        colors = [
            [0.53, 0.81, 0.98, 1.0],  # 0 AIR - azzurro cielo
            [0.85, 0.65, 0.45, 1.0],  # 1 SAND - sabbia/beige
            [0.95, 0.85, 0.55, 1.0],  # 2 INSULATION - giallo chiaro
            [0.40, 0.40, 0.45, 1.0],  # 3 STEEL - grigio scuro
            [0.60, 0.45, 0.35, 1.0],  # 4 TUBES - marrone
            [0.95, 0.35, 0.25, 1.0],  # 5 HEATERS - rosso
            [0.45, 0.35, 0.25, 1.0],  # 6 GROUND - marrone scuro
            [0.55, 0.55, 0.55, 1.0],  # 7 CONCRETE - grigio
        ]
        return ListedColormap(colors, name='materials')
    
    def update_slice_range(self):
        """Aggiorna il range dello slider quando cambia l'asse"""
        self.update_slice_position()
    
    def update_slice_position(self):
        """Aggiorna la posizione della sezione"""
        if self.mesh is None:
            return
        
        axis = self.axis_combo.currentText()
        slider_pct = self.slice_slider.value() / 100.0
        
        if axis == 'x':
            pos = slider_pct * self.mesh.Lx
        elif axis == 'y':
            pos = slider_pct * self.mesh.Ly
        else:
            pos = slider_pct * self.mesh.Lz
        
        self.slice_pos_label.setText(f"{pos:.2f} m")
        
        # Aggiorna la visualizzazione (anche prima della simulazione per vedere i materiali)
        self.update_visualization()
    
    def show_slice(self, axis: str):
        """Mostra una sezione specifica"""
        self.axis_combo.setCurrentText(axis)
        self.slice_slider.setValue(50)
        self.update_visualization()
    
    def show_3d_view(self):
        """Mostra la vista 3D completa"""
        if self.mesh is None:
            return
        
        self.plotter.clear()
        
        field = self.field_combo.currentText()
        cmap = self.cmap_combo.currentText()
        clim = [self.tmin_spin.value(), self.tmax_spin.value()]
        
        grid = pv.ImageData(
            dimensions=(self.mesh.Nx + 1, self.mesh.Ny + 1, self.mesh.Nz + 1),
            spacing=(self.mesh.dx, self.mesh.dy, self.mesh.dz),
            origin=(0, 0, 0)
        )
        
        grid.cell_data["Temperature"] = self.mesh.T.ravel(order='F')
        grid.cell_data["Material"] = self.mesh.material_id.ravel(order='F').astype(float)
        
        # Clip per mostrare metà
        clipped = grid.clip(normal='y', origin=(self.mesh.Lx/2, self.mesh.Ly/2, 0))
        
        self.plotter.add_mesh(clipped, scalars=field, cmap=cmap, clim=clim, opacity=0.8)
        self.plotter.add_axes()
        self.plotter.add_scalar_bar(title=f"{field}", vertical=True)
        self.plotter.reset_camera()
    
    def reset_view(self):
        """Reset della vista"""
        self.plotter.reset_camera()
    
    def take_screenshot(self):
        """Salva uno screenshot"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Salva Screenshot", "", "PNG (*.png);;JPEG (*.jpg)"
        )
        if filename:
            self.plotter.screenshot(filename)
            self.log(f"Screenshot salvato: {filename}")
    
    def save_results(self):
        """Salva i risultati"""
        if self.mesh is None:
            QMessageBox.warning(self, "Attenzione", "Nessun risultato da salvare")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Salva Risultati", "", "CSV (*.csv)"
        )
        if filename:
            # Salva statistiche base
            with open(filename, 'w') as f:
                f.write("Statistica,Valore,Unità\n")
                f.write(f"T_min,{self.mesh.T.min():.2f},°C\n")
                f.write(f"T_max,{self.mesh.T.max():.2f},°C\n")
                f.write(f"T_mean,{self.mesh.T.mean():.2f},°C\n")
                f.write(f"T_std,{self.mesh.T.std():.2f},°C\n")
            self.log(f"Risultati salvati: {filename}")
    
    def export_vtk(self):
        """Esporta in formato VTK"""
        if self.mesh is None:
            QMessageBox.warning(self, "Attenzione", "Nessun dato da esportare")
            return
        
        # Avvisa se la simulazione non è stata eseguita
        if not hasattr(self, '_simulation_completed') or not self._simulation_completed:
            reply = QMessageBox.question(
                self, "Attenzione",
                "La simulazione non è stata eseguita.\n"
                "Il campo temperatura conterrà solo i valori iniziali.\n"
                "Continuare comunque?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Esporta VTK", "", "VTK ImageData (*.vti)"
        )
        if filename:
            grid = pv.ImageData(
                dimensions=(self.mesh.Nx + 1, self.mesh.Ny + 1, self.mesh.Nz + 1),
                spacing=(self.mesh.dx, self.mesh.dy, self.mesh.dz),
                origin=(0, 0, 0)
            )
            grid.cell_data["Temperature"] = self.mesh.T.ravel(order='F')
            grid.cell_data["Material"] = self.mesh.material_id.ravel(order='F').astype(float)
            grid.cell_data["k"] = self.mesh.k.ravel(order='F')
            grid.cell_data["Q"] = self.mesh.Q.ravel(order='F')
            grid.save(filename)
            self.log(f"VTK esportato: {filename}")
    
    def new_simulation(self):
        """Reset per nuova simulazione"""
        self.mesh = None
        self.battery_geometry = None
        self._simulation_completed = False  # Reset flag simulazione
        self.plotter.clear()
        self.show_initial_message()
        self.stats_text.clear()
        self.energy_text.clear()
        self.mat_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("Stato: Pronto")
        self.run_btn.setEnabled(False)
        self.build_mesh_btn.setEnabled(True)
    
    def closeEvent(self, event):
        """Gestisce la chiusura della finestra"""
        self.plotter.close()
        event.accept()


def main():
    """Entry point dell'applicazione"""
    app = QApplication(sys.argv)
    
    # Stile
    app.setStyle("Fusion")
    
    window = SandBatteryGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
