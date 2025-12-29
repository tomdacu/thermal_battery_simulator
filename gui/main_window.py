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

Interfaccia grafica completa per la simulazione Thermal Battery:
- Pannello sinistro: Input parametri e controlli
- Pannello centrale: Visualizzazione 3D PyVista
- Pannello inferiore: Risultati e statistiche
- Controlli visualizzazione 3D interattivi
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
from src.solver.transient import TransientSolver, run_transient_simulation
from src.analysis.power_balance import PowerBalanceAnalyzer
from src.analysis.energy_balance import EnergyBalanceAnalyzer
from src.core.profiles import PowerProfile, ExtractionProfile, InitialCondition, TransientConfig
from src.io.state_manager import StateManager, SimulationState, TransientResults
# NOTA: AnalysisTab importato localmente in create_left_panel() per evitare import circolari


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


class ThermalBatteryGUI(QMainWindow):
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
        self.setWindowTitle("Thermal Battery Simulation")
        self.setGeometry(100, 100, 1600, 900)
        
        # Imposta icona della finestra
        icon_path = Path(__file__).parent.parent / "photo" / "Icona Thermal Battery.png"
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        
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
        
    def create_left_panel(self) -> QWidget:
        """
        Crea il pannello sinistro con struttura a 2 livelli di tabs.
        
        LIVELLO 1: [Geometria] [Materiali] [Analisi] [Risultati]
        LIVELLO 2: Sub-tabs specifiche per ogni sezione
        """
        panel = QWidget()
        main_layout = QVBoxLayout(panel)
        
        # === LIVELLO 1: TABS PRINCIPALI ===
        self.main_tabs = QTabWidget()
        self.main_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #ccc;
                background: white;
            }
            QTabBar::tab {
                background: #e0e0e0;
                padding: 8px 12px;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: #4CAF50;
                color: white;
            }
        """)
        
        # ============================================================
        # TAB 1: GEOMETRIA (con sub-tabs)
        # ============================================================
        self.geom_tabs = QTabWidget()
        self.geom_tabs.setStyleSheet("QTabBar::tab { padding: 5px 10px; font-size: 10px; }")
        
        # Sub-tab: Cilindro
        cylinder_tab = self._create_cylinder_subtab()
        self.geom_tabs.addTab(cylinder_tab, "Cilindro")
        
        # Sub-tab: Isolamento
        insulation_tab = self._create_insulation_subtab()
        self.geom_tabs.addTab(insulation_tab, "Isolamento")
        
        # Sub-tab: Resistenze
        heaters_tab = self._create_heater_tab()
        self.geom_tabs.addTab(heaters_tab, "Resistenze")
        
        # Sub-tab: Tubi
        tubes_tab = self._create_tube_tab()
        self.geom_tabs.addTab(tubes_tab, "Tubi")
        
        # Sub-tab: Mesh
        mesh_tab = self._create_mesh_subtab()
        self.geom_tabs.addTab(mesh_tab, "Mesh")
        
        self.main_tabs.addTab(self.geom_tabs, "1. Geometria")
        
        # ============================================================
        # TAB 2: MATERIALI (con sub-tabs)
        # ============================================================
        self.mat_tabs = QTabWidget()
        self.mat_tabs.setStyleSheet("QTabBar::tab { padding: 5px 10px; font-size: 10px; }")
        
        # Sub-tab: Storage
        storage_mat_tab = self._create_storage_material_subtab()
        self.mat_tabs.addTab(storage_mat_tab, "Storage")
        
        # Sub-tab: Isolamento
        insul_mat_tab = self._create_insulation_material_subtab()
        self.mat_tabs.addTab(insul_mat_tab, "Isolamento")
        
        # Sub-tab: Condizioni operative
        conditions_tab = self._create_conditions_subtab()
        self.mat_tabs.addTab(conditions_tab, "Condizioni")
        
        self.main_tabs.addTab(self.mat_tabs, "2. Materiali")
        
        # ============================================================
        # TAB 3: ANALISI (con sub-tabs)
        # ============================================================
        self.analysis_tabs = QTabWidget()
        self.analysis_tabs.setStyleSheet("QTabBar::tab { padding: 5px 10px; font-size: 10px; }")
        
        # Importa i widget dall'analysis_tab
        from gui.analysis_tab import (
            AnalysisTypeWidget, InitialConditionWidget,
            PowerProfileWidget, ExtractionProfileWidget, SaveLoadWidget
        )
        
        # Sub-tab: Tipo Analisi
        self.analysis_type_widget = AnalysisTypeWidget()
        self.analysis_type_widget.analysis_changed.connect(self._on_analysis_type_changed)
        self.analysis_tabs.addTab(self.analysis_type_widget, "Tipo")
        
        # Sub-tab: Condizioni Iniziali
        self.initial_cond_widget = InitialConditionWidget()
        self.analysis_tabs.addTab(self.initial_cond_widget, "Condizioni Iniziali")
        
        # Sub-tab: Potenza
        self.power_widget = PowerProfileWidget()
        self.analysis_tabs.addTab(self.power_widget, "Potenza")
        
        # Sub-tab: Estrazione
        self.extraction_widget = ExtractionProfileWidget()
        self.analysis_tabs.addTab(self.extraction_widget, "Estrazione")
        
        # Sub-tab: Solver
        solver_tab = self._create_solver_tab()
        self.analysis_tabs.addTab(solver_tab, "Solver")
        
        # Sub-tab: Salvataggio
        self.save_load_widget = SaveLoadWidget()
        self.save_load_widget.state_loaded.connect(self._on_state_loaded)
        self.save_load_widget.save_btn.clicked.connect(self._save_current_state)
        self.analysis_tabs.addTab(self.save_load_widget, "Salvataggio")
        
        self.main_tabs.addTab(self.analysis_tabs, "3. Analisi")
        
        # ============================================================
        # TAB 4: RISULTATI (con sub-tabs)
        # ============================================================
        self.results_tabs = QTabWidget()
        self.results_tabs.setStyleSheet("QTabBar::tab { padding: 5px 10px; font-size: 10px; }")
        
        # Sub-tab: Statistiche
        stats_tab = self._create_statistics_subtab()
        self.results_tabs.addTab(stats_tab, "Statistiche")
        
        # Sub-tab: Bilancio Energetico
        balance_tab = self._create_energy_balance_subtab()
        self.results_tabs.addTab(balance_tab, "Bilancio")
        
        # Sub-tab: Materiali (distribuzione)
        mat_info_tab = self._create_materials_info_subtab()
        self.results_tabs.addTab(mat_info_tab, "Materiali")
        
        # Sub-tab: Esporta
        export_tab = self._create_export_subtab()
        self.results_tabs.addTab(export_tab, "Esporta")
        
        # Sub-tab: Guida
        help_tab = self._create_help_tab()
        self.results_tabs.addTab(help_tab, "📖 Guida")
        
        self.main_tabs.addTab(self.results_tabs, "4. Risultati")
        
        main_layout.addWidget(self.main_tabs)
        
        # === PULSANTI (sempre visibili) ===
        btn_layout = QVBoxLayout()
        
        # Pulsante Costruisci Mesh
        self.build_mesh_btn = QPushButton("🗘 Costruisci Mesh")
        self.build_mesh_btn.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; padding: 8px;")
        self.build_mesh_btn.clicked.connect(self.build_mesh)
        btn_layout.addWidget(self.build_mesh_btn)

        # Pulsante Costruisci Geometria (senza mesh)
        self.preview_geom_btn = QPushButton("🔧 Preview Geometria")
        self.preview_geom_btn.clicked.connect(self.build_geometry)
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
    
    # ================================================================
    # SUB-TABS GEOMETRIA
    # ================================================================
    
    def _create_cylinder_subtab(self) -> QWidget:
        """Sub-tab per configurazione cilindro e dominio"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Dominio
        domain_group = QGroupBox("Dominio Simulazione")
        domain_layout = QGridLayout()
        
        domain_layout.addWidget(QLabel("Lx [m]:"), 0, 0)
        self.lx_spin = QDoubleSpinBox()
        self.lx_spin.setRange(1, 50)
        self.lx_spin.setValue(6.0)
        self.lx_spin.setDecimals(1)
        domain_layout.addWidget(self.lx_spin, 0, 1)
        
        domain_layout.addWidget(QLabel("Ly [m]:"), 1, 0)
        self.ly_spin = QDoubleSpinBox()
        self.ly_spin.setRange(1, 50)
        self.ly_spin.setValue(6.0)
        self.ly_spin.setDecimals(1)
        domain_layout.addWidget(self.ly_spin, 1, 1)
        
        domain_layout.addWidget(QLabel("Lz [m]:"), 2, 0)
        self.lz_spin = QDoubleSpinBox()
        self.lz_spin.setRange(1, 50)
        self.lz_spin.setValue(5.0)
        self.lz_spin.setDecimals(1)
        domain_layout.addWidget(self.lz_spin, 2, 1)
        
        domain_group.setLayout(domain_layout)
        layout.addWidget(domain_group)
        
        # Cilindro Storage
        cyl_group = QGroupBox("Cilindro Storage")
        cyl_layout = QGridLayout()
        
        cyl_layout.addWidget(QLabel("Raggio [m]:"), 0, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.5, 20)
        self.radius_spin.setValue(2.0)
        self.radius_spin.setDecimals(2)
        cyl_layout.addWidget(self.radius_spin, 0, 1)
        
        cyl_layout.addWidget(QLabel("Altezza [m]:"), 1, 0)
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(1, 30)
        self.height_spin.setValue(4.0)
        self.height_spin.setDecimals(1)
        cyl_layout.addWidget(self.height_spin, 1, 1)
        
        cyl_layout.addWidget(QLabel("Sfasamento Tubi/Heaters [°]:"), 2, 0)
        self.phase_offset_spin = QDoubleSpinBox()
        self.phase_offset_spin.setRange(0, 180)
        self.phase_offset_spin.setValue(15)
        self.phase_offset_spin.setDecimals(1)
        cyl_layout.addWidget(self.phase_offset_spin, 2, 1)
        
        cyl_group.setLayout(cyl_layout)
        layout.addWidget(cyl_group)
        
        # Tetto
        roof_group = QGroupBox("Tetto")
        roof_layout = QGridLayout()
        
        self.enable_cone_roof_check = QCheckBox("Abilita tetto conico")
        self.enable_cone_roof_check.setChecked(True)
        roof_layout.addWidget(self.enable_cone_roof_check, 0, 0, 1, 2)
        
        roof_layout.addWidget(QLabel("Angolo [°]:"), 1, 0)
        self.roof_angle_spin = QDoubleSpinBox()
        self.roof_angle_spin.setRange(0, 45)
        self.roof_angle_spin.setValue(15)
        self.roof_angle_spin.setDecimals(1)
        roof_layout.addWidget(self.roof_angle_spin, 1, 1)
        
        roof_layout.addWidget(QLabel("Slab acciaio [m]:"), 2, 0)
        self.steel_slab_spin = QDoubleSpinBox()
        self.steel_slab_spin.setRange(0.0, 0.1)
        self.steel_slab_spin.setValue(0.005)
        self.steel_slab_spin.setDecimals(3)
        roof_layout.addWidget(self.steel_slab_spin, 2, 1)
        
        self.fill_cone_check = QCheckBox("Riempi cono con sabbia")
        roof_layout.addWidget(self.fill_cone_check, 3, 0, 1, 2)
        
        roof_group.setLayout(roof_layout)
        layout.addWidget(roof_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _create_insulation_subtab(self) -> QWidget:
        """Sub-tab per configurazione isolamento"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # Isolamento radiale
        radial_group = QGroupBox("Isolamento Radiale")
        radial_layout = QGridLayout()
        
        radial_layout.addWidget(QLabel("Spessore isolamento [m]:"), 0, 0)
        self.insulation_thickness_spin = QDoubleSpinBox()
        self.insulation_thickness_spin.setRange(0.05, 1.0)
        self.insulation_thickness_spin.setValue(0.3)
        self.insulation_thickness_spin.setDecimals(2)
        self.insulation_thickness_spin.setSingleStep(0.05)
        radial_layout.addWidget(self.insulation_thickness_spin, 0, 1)
        
        radial_layout.addWidget(QLabel("Spessore acciaio [m]:"), 1, 0)
        self.shell_thickness_spin = QDoubleSpinBox()
        self.shell_thickness_spin.setRange(0.005, 0.1)
        self.shell_thickness_spin.setValue(0.02)
        self.shell_thickness_spin.setDecimals(3)
        self.shell_thickness_spin.setSingleStep(0.005)
        radial_layout.addWidget(self.shell_thickness_spin, 1, 1)
        
        radial_group.setLayout(radial_layout)
        layout.addWidget(radial_group)
        
        # Isolamento verticale
        vert_group = QGroupBox("Isolamento Verticale (Slab)")
        vert_layout = QGridLayout()
        
        vert_layout.addWidget(QLabel("Slab inferiore [m]:"), 0, 0)
        self.slab_bottom_spin = QDoubleSpinBox()
        self.slab_bottom_spin.setRange(0.0, 1.0)
        self.slab_bottom_spin.setValue(0.2)
        self.slab_bottom_spin.setDecimals(2)
        self.slab_bottom_spin.setSingleStep(0.05)
        vert_layout.addWidget(self.slab_bottom_spin, 0, 1)
        
        vert_layout.addWidget(QLabel("Slab superiore [m]:"), 1, 0)
        self.slab_top_spin = QDoubleSpinBox()
        self.slab_top_spin.setRange(0.0, 1.0)
        self.slab_top_spin.setValue(0.2)
        self.slab_top_spin.setDecimals(2)
        self.slab_top_spin.setSingleStep(0.05)
        vert_layout.addWidget(self.slab_top_spin, 1, 1)
        
        vert_group.setLayout(vert_layout)
        layout.addWidget(vert_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _create_mesh_subtab(self) -> QWidget:
        """Sub-tab per configurazione mesh"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        mesh_group = QGroupBox("Parametri Mesh")
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
        
        self.memory_label = QLabel("Memoria stimata: -")
        self.memory_label.setStyleSheet("color: gray; font-size: 9px;")
        mesh_layout.addWidget(self.memory_label, 2, 0, 1, 2)
        
        self.spacing_spin.valueChanged.connect(self.update_mesh_info)
        self.lx_spin.valueChanged.connect(self.update_mesh_info)
        self.ly_spin.valueChanged.connect(self.update_mesh_info)
        self.lz_spin.valueChanged.connect(self.update_mesh_info)
        
        mesh_group.setLayout(mesh_layout)
        layout.addWidget(mesh_group)
        
        # Suggerimenti
        tips_group = QGroupBox("⚡ Suggerimenti")
        tips_layout = QVBoxLayout()
        tips_text = QLabel(
            "<b>0.3-0.5m</b>: Preview veloce (~1-5k celle)<br>"
            "<b>0.1-0.2m</b>: Simulazione accurata (~20-100k)<br>"
            "<b>0.05m</b>: Alta risoluzione (>500k, lento)"
        )
        tips_text.setWordWrap(True)
        tips_text.setStyleSheet("padding: 5px; background: #e8f5e9;")
        tips_layout.addWidget(tips_text)
        tips_group.setLayout(tips_layout)
        layout.addWidget(tips_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    # ================================================================
    # SUB-TABS MATERIALI
    # ================================================================
    
    def _create_storage_material_subtab(self) -> QWidget:
        """Sub-tab per materiale storage"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        storage_group = QGroupBox("Materiale di Accumulo")
        storage_layout = QGridLayout()
        
        storage_layout.addWidget(QLabel("Tipo:"), 0, 0)
        self.storage_combo = QComboBox()
        self.storage_combo.addItems([
            "steatite", "silica_sand", "olivine", 
            "basalt", "magnetite", "quartzite", "granite"
        ])
        storage_layout.addWidget(self.storage_combo, 0, 1)
        
        storage_layout.addWidget(QLabel("Packing [%]:"), 1, 0)
        self.packing_spin = QSpinBox()
        self.packing_spin.setRange(50, 75)
        self.packing_spin.setValue(63)
        storage_layout.addWidget(self.packing_spin, 1, 1)
        
        # Info materiale
        info_label = QLabel(
            "<b>Proprietà tipiche:</b><br>"
            "• ρ ≈ 2500-3000 kg/m³<br>"
            "• cp ≈ 800-1000 J/kg·K<br>"
            "• k ≈ 1-3 W/m·K"
        )
        info_label.setStyleSheet("color: #666; font-size: 9px; padding: 8px;")
        storage_layout.addWidget(info_label, 2, 0, 1, 2)
        
        storage_group.setLayout(storage_layout)
        layout.addWidget(storage_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _create_insulation_material_subtab(self) -> QWidget:
        """Sub-tab per materiale isolamento"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        insulation_group = QGroupBox("Materiale Isolante")
        insulation_layout = QGridLayout()
        
        insulation_layout.addWidget(QLabel("Tipo:"), 0, 0)
        self.insulation_combo = QComboBox()
        self.insulation_combo.addItems([
            "rock_wool", "glass_wool", "calcium_silicate",
            "ceramic_fiber", "perlite"
        ])
        insulation_layout.addWidget(self.insulation_combo, 0, 1)
        
        # Info materiale
        info_label = QLabel(
            "<b>Proprietà tipiche:</b><br>"
            "• k ≈ 0.03-0.1 W/m·K<br>"
            "• T max ≈ 600-1200°C"
        )
        info_label.setStyleSheet("color: #666; font-size: 9px; padding: 8px;")
        insulation_layout.addWidget(info_label, 1, 0, 1, 2)
        
        insulation_group.setLayout(insulation_layout)
        layout.addWidget(insulation_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    def _create_conditions_subtab(self) -> QWidget:
        """Sub-tab per condizioni operative"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        cond_group = QGroupBox("Condizioni Ambiente")
        cond_layout = QGridLayout()
        
        cond_layout.addWidget(QLabel("T ambiente [°C]:"), 0, 0)
        self.t_amb_spin = QDoubleSpinBox()
        self.t_amb_spin.setRange(-20, 50)
        self.t_amb_spin.setValue(20)
        cond_layout.addWidget(self.t_amb_spin, 0, 1)
        
        cond_layout.addWidget(QLabel("T terreno [°C]:"), 1, 0)
        self.t_ground_spin = QDoubleSpinBox()
        self.t_ground_spin.setRange(0, 30)
        self.t_ground_spin.setValue(10)
        cond_layout.addWidget(self.t_ground_spin, 1, 1)
        
        cond_group.setLayout(cond_layout)
        layout.addWidget(cond_group)
        
        layout.addStretch()
        return self._create_scrollable_widget(content)
    
    # ================================================================
    # SUB-TABS RISULTATI
    # ================================================================
    
    def _create_statistics_subtab(self) -> QWidget:
        """Sub-tab per statistiche temperatura"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        self.stats_text = QTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.stats_text)
        
        return content
    
    def _create_energy_balance_subtab(self) -> QWidget:
        """Sub-tab per bilancio energetico"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        self.energy_text = QTextEdit()
        self.energy_text.setReadOnly(True)
        self.energy_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.energy_text)
        
        return content
    
    def _create_materials_info_subtab(self) -> QWidget:
        """Sub-tab per distribuzione materiali"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        self.mat_text = QTextEdit()
        self.mat_text.setReadOnly(True)
        self.mat_text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.mat_text)
        
        return content
    
    def _create_export_subtab(self) -> QWidget:
        """Sub-tab per esportazione risultati"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        export_group = QGroupBox("Esporta Risultati")
        export_layout = QVBoxLayout()
        
        csv_btn = QPushButton("📊 Esporta CSV...")
        csv_btn.clicked.connect(self.save_results)
        export_layout.addWidget(csv_btn)
        
        vtk_btn = QPushButton("🔲 Esporta VTK...")
        vtk_btn.clicked.connect(self.export_vtk)
        export_layout.addWidget(vtk_btn)
        
        screenshot_btn = QPushButton("📷 Screenshot...")
        screenshot_btn.clicked.connect(self.take_screenshot)
        export_layout.addWidget(screenshot_btn)
        
        export_group.setLayout(export_layout)
        layout.addWidget(export_group)
        
        layout.addStretch()
        return content
    
    def _create_scrollable_widget(self, content_widget: QWidget) -> QScrollArea:
        """Crea un widget scrollabile contenente il content_widget"""
        scroll = QScrollArea()
        scroll.setWidget(content_widget)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll
    
    # NOTA: La vecchia _create_geometry_tab è stata rimossa.
    # Le sue funzionalità sono ora divise in:
    # - _create_cylinder_subtab()
    # - _create_insulation_subtab()
    # - _create_mesh_subtab()
    # - _create_storage_material_subtab()
    # - _create_insulation_material_subtab()
    # - _create_conditions_subtab()
    
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
        
        # === OFFSET VERTICALI RESISTENZE ===
        elem_layout.addWidget(QLabel("<b>Offset Verticali</b>"), 3, 0, 1, 2)
        
        elem_layout.addWidget(QLabel("Offset dal basso [m]:"), 4, 0)
        self.heater_offset_bottom_spin = QDoubleSpinBox()
        self.heater_offset_bottom_spin.setRange(0.0, 2.0)
        self.heater_offset_bottom_spin.setValue(0.0)
        self.heater_offset_bottom_spin.setDecimals(2)
        self.heater_offset_bottom_spin.setSingleStep(0.1)
        self.heater_offset_bottom_spin.setToolTip("Distanza dalla fine dello slab isolante inferiore")
        elem_layout.addWidget(self.heater_offset_bottom_spin, 4, 1)
        
        elem_layout.addWidget(QLabel("Offset dall'alto [m]:"), 5, 0)
        self.heater_offset_top_spin = QDoubleSpinBox()
        self.heater_offset_top_spin.setRange(0.0, 2.0)
        self.heater_offset_top_spin.setValue(0.0)
        self.heater_offset_top_spin.setDecimals(2)
        self.heater_offset_top_spin.setSingleStep(0.1)
        self.heater_offset_top_spin.setToolTip("Distanza prima dell'inizio dello slab isolante superiore")
        elem_layout.addWidget(self.heater_offset_top_spin, 5, 1)
        
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
        
        # === METODO DI SOLUZIONE ===
        solver_group = QGroupBox("Metodo di Soluzione")
        solver_layout = QGridLayout()
        
        row = 0
        solver_layout.addWidget(QLabel("Metodo:"), row, 0)
        self.solver_combo = QComboBox()
        self.solver_combo.addItems(["bicgstab", "cg", "gmres", "direct"])
        self.solver_combo.setCurrentText("bicgstab")  # Default: BiCGSTAB è più robusto
        self.solver_combo.currentTextChanged.connect(self._on_solver_method_changed)
        solver_layout.addWidget(self.solver_combo, row, 1)
        
        row += 1
        self.solver_method_desc = QLabel("")
        self.solver_method_desc.setWordWrap(True)
        self.solver_method_desc.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
        solver_layout.addWidget(self.solver_method_desc, row, 0, 1, 2)
        
        solver_group.setLayout(solver_layout)
        layout.addWidget(solver_group)
        
        # === PRECONDIZIONATORE ===
        prec_group = QGroupBox("Precondizionatore")
        prec_layout = QGridLayout()
        
        row = 0
        prec_layout.addWidget(QLabel("Tipo:"), row, 0)
        self.preconditioner_combo = QComboBox()
        self.preconditioner_combo.addItems([
            "jacobi", 
            "none", 
            "ilu", 
            "amg (Ruge-Stuben)", 
            "amg (Smoothed Aggregation)"
        ])
        self.preconditioner_combo.setCurrentText("jacobi")  # Jacobi è più veloce per eq. calore
        self.preconditioner_combo.currentTextChanged.connect(self._on_preconditioner_changed)
        prec_layout.addWidget(self.preconditioner_combo, row, 1)
        
        row += 1
        self.precond_desc = QLabel("")
        self.precond_desc.setWordWrap(True)
        self.precond_desc.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
        prec_layout.addWidget(self.precond_desc, row, 0, 1, 2)
        
        row += 1
        self.amg_warning = QLabel("")
        self.amg_warning.setWordWrap(True)
        self.amg_warning.setStyleSheet("color: #E65100; font-size: 9px; padding: 4px; background-color: #FFF3E0;")
        self.amg_warning.hide()  # Nascosto di default
        prec_layout.addWidget(self.amg_warning, row, 0, 1, 2)
        
        prec_group.setLayout(prec_layout)
        layout.addWidget(prec_group)
        
        # === CONVERGENZA ===
        conv_group = QGroupBox("Parametri di Convergenza")
        conv_layout = QGridLayout()
        
        row = 0
        conv_layout.addWidget(QLabel("Tolleranza:"), row, 0)
        self.tolerance_combo = QComboBox()
        self.tolerance_combo.addItems(["1e-10 (Alta precisione)", "1e-8 (Default)", 
                                        "1e-6 (Veloce)", "1e-4 (Molto veloce)"])
        self.tolerance_combo.setCurrentIndex(1)  # Default 1e-8
        self.tolerance_combo.currentTextChanged.connect(self._on_tolerance_changed)
        conv_layout.addWidget(self.tolerance_combo, row, 1)
        
        row += 1
        self.tolerance_desc = QLabel("")
        self.tolerance_desc.setWordWrap(True)
        self.tolerance_desc.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
        conv_layout.addWidget(self.tolerance_desc, row, 0, 1, 2)
        
        row += 1
        conv_layout.addWidget(QLabel("Max iterazioni:"), row, 0)
        self.max_iter_spin = QSpinBox()
        self.max_iter_spin.setRange(100, 100000)
        self.max_iter_spin.setValue(5000)
        self.max_iter_spin.setSingleStep(1000)
        conv_layout.addWidget(self.max_iter_spin, row, 1)
        
        row += 1
        conv_layout.addWidget(QLabel("Precisione:"), row, 0)
        self.precision_combo = QComboBox()
        self.precision_combo.addItems([
            "float64 (doppia)",
            "float32 (singola)",
            "float16 (mezza)"
        ])
        self.precision_combo.setCurrentIndex(0)  # Default float64
        self.precision_combo.currentTextChanged.connect(self._on_precision_changed)
        conv_layout.addWidget(self.precision_combo, row, 1)
        
        row += 1
        self.precision_desc = QLabel(
            "<b>float64</b>: massima precisione, standard.<br>"
            "<b>float32</b>: 2x meno memoria, leggermente più veloce.<br>"
            "<b>float16</b>: 4x meno memoria, può avere problemi di convergenza."
        )
        self.precision_desc.setWordWrap(True)
        self.precision_desc.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
        conv_layout.addWidget(self.precision_desc, row, 0, 1, 2)
        
        conv_group.setLayout(conv_layout)
        layout.addWidget(conv_group)
        
        # === PERFORMANCE / THREAD ===
        perf_group = QGroupBox("Performance (CPU/GPU)")
        perf_layout = QGridLayout()
        
        row = 0
        perf_layout.addWidget(QLabel("Calcolo:"), row, 0)
        self.threads_combo = QComboBox()
        self.threads_combo.addItems([
            "CPU: Auto (tutti)", 
            "CPU: Tutti - 1", 
            "CPU: 4 core", 
            "CPU: 2 core", 
            "CPU: 1 core",
            "GPU: CUDA (NVIDIA)",
            "GPU: OpenCL (AMD/Intel/NVIDIA)"
        ])
        self.threads_combo.setCurrentIndex(1)  # Default: tutti - 1
        self.threads_combo.currentTextChanged.connect(self._on_compute_mode_changed)
        perf_layout.addWidget(self.threads_combo, row, 1)
        
        row += 1
        self.threads_desc = QLabel(
            "<b>Auto</b>: massima velocità, può rallentare il sistema.<br>"
            "<b>Tutti - 1</b>: raccomandato, lascia un core libero per la GUI."
        )
        self.threads_desc.setWordWrap(True)
        self.threads_desc.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
        perf_layout.addWidget(self.threads_desc, row, 0, 1, 2)
        
        perf_group.setLayout(perf_layout)
        layout.addWidget(perf_group)
        
        # === SUGGERIMENTI RAPIDI ===
        tips_group = QGroupBox("⚡ Suggerimenti per Velocizzare")
        tips_layout = QVBoxLayout()
        
        tips_text = QLabel(
            "<b>Per mesh grandi (>50k celle):</b><br>"
            "• Usa metodo <b>cg</b> + precondizionatore <b>jacobi</b> o <b>none</b><br>"
            "• Tolleranza <b>1e-6</b> è sufficiente per visualizzazione<br>"
            "• ⚠️ Evita <b>ilu</b>: è single-threaded e lento!<br><br>"
            "<b>Per risultati precisi:</b><br>"
            "• Metodo <b>cg</b> con tolleranza <b>1e-10</b><br>"
            "• Metodo <b>direct</b> solo per mesh piccole (<20k celle)"
        )
        tips_text.setWordWrap(True)
        tips_text.setStyleSheet("background-color: #e8f5e9; padding: 8px; border-radius: 4px;")
        tips_layout.addWidget(tips_text)
        
        tips_group.setLayout(tips_layout)
        layout.addWidget(tips_group)
        
        layout.addStretch()
        
        # Inizializza descrizioni
        self._on_solver_method_changed(self.solver_combo.currentText())
        self._on_preconditioner_changed(self.preconditioner_combo.currentText())
        self._on_tolerance_changed(self.tolerance_combo.currentText())
        
        return self._create_scrollable_widget(content)
    
    def _create_help_tab(self) -> QWidget:
        """Crea la scheda di aiuto/guida all'uso"""
        content = QWidget()
        layout = QVBoxLayout(content)
        
        # === GUIDA RAPIDA ===
        guide_group = QGroupBox("📖 Guida Rapida")
        guide_layout = QVBoxLayout()
        
        guide_text = QTextEdit()
        guide_text.setReadOnly(True)
        guide_text.setMinimumHeight(400)
        guide_text.setHtml("""
        <style>
            body { font-family: Arial, sans-serif; font-size: 10px; }
            h2 { color: #1976D2; margin-top: 12px; }
            h3 { color: #388E3C; margin-top: 8px; }
            .tip { background-color: #FFF9C4; padding: 8px; margin: 4px 0; border-left: 3px solid #FFC107; }
            .warning { background-color: #FFEBEE; padding: 8px; margin: 4px 0; border-left: 3px solid #F44336; }
            code { background-color: #ECEFF1; padding: 2px 4px; }
        </style>
        
        <h2>🚀 Come Usare il Simulatore</h2>
        
        <h3>1. Configura la Geometria</h3>
        <p>Nella scheda <b>Geometria</b>, imposta:</p>
        <ul>
            <li><b>Dimensioni dominio</b>: Lx, Ly, Lz del volume di simulazione</li>
            <li><b>Punti mesh</b>: Risoluzione della griglia 3D</li>
            <li><b>Raggio batteria</b>: Dimensione del cilindro di accumulo</li>
        </ul>
        
        <div class="tip">
        💡 <b>Suggerimento</b>: Inizia con pochi punti mesh (30-40) per test rapidi, 
        poi aumenta (60-80) per risultati finali.
        </div>
        
        <h3>2. Configura Resistenze e Tubi</h3>
        <p>Nelle schede <b>Resistenze</b> e <b>Tubi</b>:</p>
        <ul>
            <li>Scegli il pattern di disposizione</li>
            <li>Imposta potenza totale e numero elementi</li>
            <li>Usa "Anteprima" per vedere le posizioni</li>
        </ul>
        
        <h3>3. Configura il Solver</h3>
        <p>Nella scheda <b>Solver</b>, ottimizza le prestazioni:</p>
        
        <table border="1" cellpadding="4" style="border-collapse: collapse;">
            <tr style="background-color: #E3F2FD;">
                <th>Scenario</th>
                <th>Metodo</th>
                <th>Tolleranza</th>
                <th>Tempo stimato</th>
            </tr>
            <tr>
                <td>Test rapido</td>
                <td>cg + ilu</td>
                <td>1e-4</td>
                <td>~5 sec</td>
            </tr>
            <tr>
                <td>Visualizzazione</td>
                <td>cg + ilu</td>
                <td>1e-6</td>
                <td>~15 sec</td>
            </tr>
            <tr>
                <td>Precisione standard</td>
                <td>cg + ilu</td>
                <td>1e-8</td>
                <td>~30 sec</td>
            </tr>
            <tr>
                <td>Alta precisione</td>
                <td>direct</td>
                <td>N/A</td>
                <td>~2 min</td>
            </tr>
        </table>
        
        <h3>4. Esegui Simulazione</h3>
        <ol>
            <li>Clicca <b>"Costruisci Mesh"</b> per creare la griglia</li>
            <li>Clicca <b>"Esegui Simulazione"</b> per risolvere</li>
            <li>Usa i controlli di visualizzazione per esplorare i risultati</li>
        </ol>
        
        <h2>⚡ Ottimizzazione Performance</h2>
        
        <h3>Perché è lento?</h3>
        <ul>
            <li><b>Troppi punti mesh</b>: N³ celle = tempo esponenziale. 100³ = 1 milione di celle!</li>
            <li><b>Metodo diretto</b>: Usa O(N²) memoria, lentissimo per mesh grandi</li>
            <li><b>Tolleranza troppo stretta</b>: Più iterazioni = più tempo</li>
        </ul>
        
        <h3>Come velocizzare</h3>
        <div class="tip">
        <b>1. Usa solver iterativo</b>: <code>cg</code> o <code>bicgstab</code> invece di <code>direct</code><br>
        <b>2. Precondizionatore ILU</b>: Riduce iterazioni 5-10x<br>
        <b>3. Tolleranza rilassata</b>: <code>1e-6</code> è sufficiente per visualizzazione<br>
        <b>4. Meno punti mesh</b>: 50³ invece di 100³ è 8x più veloce
        </div>
        
        <h3>Uso CPU/GPU</h3>
        <ul>
            <li><b>Multi-core</b>: Seleziona "Auto" o "Tutti-1" nel tab Solver</li>
            <li><b>GPU</b>: Non supportata nativamente. Richiede CuPy (avanzato)</li>
        </ul>
        
        <h2>📊 Metodi di Soluzione</h2>
        
        <h3>Metodi Iterativi (consigliati per mesh grandi)</h3>
        <ul>
            <li><b>CG (Conjugate Gradient)</b>: Il più veloce per matrici simmetriche (il nostro caso!). 
            Converge rapidamente con buon precondizionatore.</li>
            <li><b>BiCGSTAB</b>: Più robusto di CG, funziona anche con matrici non simmetriche.</li>
            <li><b>GMRES</b>: Ottima convergenza ma usa più memoria.</li>
        </ul>
        
        <h3>Metodo Diretto</h3>
        <ul>
            <li><b>Direct (LU)</b>: Soluzione esatta senza iterazioni. 
            Molto lento per mesh >50k celle. Usa molta RAM.</li>
        </ul>
        
        <div class="warning">
        ⚠️ <b>Attenzione</b>: Con 100³ celle (1 milione), il metodo diretto può usare 
        >10 GB di RAM e impiegare ore!
        </div>
        
        <h2>🔧 Risoluzione Problemi</h2>
        
        <h3>Simulazione non converge</h3>
        <ul>
            <li>Aumenta max iterazioni</li>
            <li>Prova precondizionatore diverso (jacobi invece di ilu)</li>
            <li>Verifica condizioni al contorno (temperature ragionevoli)</li>
        </ul>
        
        <h3>Out of Memory</h3>
        <ul>
            <li>Riduci punti mesh</li>
            <li>Usa metodo iterativo invece di diretto</li>
            <li>Chiudi altre applicazioni</li>
        </ul>
        """)
        
        guide_layout.addWidget(guide_text)
        guide_group.setLayout(guide_layout)
        layout.addWidget(guide_group)
        
        return self._create_scrollable_widget(content)
    
    def _on_solver_method_changed(self, method: str):
        """Aggiorna descrizione metodo solver"""
        descriptions = {
            "bicgstab": "⭐ <b>BiCGSTAB</b>: CONSIGLIATO! Robusto e veloce. "
                        "Funziona con qualsiasi matrice, anche non simmetrica. "
                        "Ideale per condizioni al contorno complesse.",
            "cg": "⚡ <b>Gradiente Coniugato</b>: Molto veloce ma richiede matrice simmetrica. "
                  "Può non convergere con condizioni al contorno miste (convezione + Dirichlet). "
                  "Usa se BiCGSTAB è troppo lento.",
            "gmres": "📈 <b>GMRES</b>: Eccellente convergenza ma usa più memoria. "
                     "Utile per problemi difficili dove altri metodi falliscono.",
            "direct": "🎯 <b>Soluzione Diretta (LU)</b>: Sempre converge, nessuna iterazione. "
                      "⚠️ MOLTO LENTO per mesh >50k celle. Usa molta RAM. "
                      "Solo per mesh piccole o debug."
        }
        self.solver_method_desc.setText(descriptions.get(method, ""))
    
    def _on_preconditioner_changed(self, prec: str):
        """Aggiorna descrizione precondizionatore"""
        descriptions = {
            "jacobi": "⭐ <b>Jacobi (Diagonale)</b>: CONSIGLIATO per mesh < 200k! Velocissimo e multi-thread. "
                      "Per l'equazione del calore è spesso il migliore.",
            "none": "⚡ <b>Nessuno</b>: CG puro. Sorprendentemente veloce per matrici ben condizionate. "
                    "Prova questa opzione se Jacobi è lento.",
            "ilu": "⚠️ <b>ILU (Incomplete LU)</b>: Single-threaded, può essere LENTO! "
                   "Riduce iterazioni ma ogni iterazione è più costosa. Usa solo per problemi difficili.",
            "amg (Ruge-Stuben)": "🚀 <b>AMG Ruge-Stuben</b>: OTTIMO per mesh grandi (>200k). "
                   "Veloce setup, ottimo per eq. calore. Riduce iterazioni a ~5-10.",
            "amg (Smoothed Aggregation)": "🔧 <b>AMG Smoothed Aggregation</b>: Più robusto di RS. "
                   "Meglio per problemi difficili. Setup leggermente più lento."
        }
        self.precond_desc.setText(descriptions.get(prec, ""))
        
        # Mostra warning/info per AMG
        is_amg = "amg" in prec.lower()
        if is_amg:
            if self.mesh is not None and self.mesh.N_total < 200000:
                self.amg_warning.setText(
                    f"⚠️ La mesh attuale ha solo {self.mesh.N_total:,} celle. "
                    "AMG è consigliato per mesh >200k celle. Il tempo di setup può superare il risparmio."
                )
                self.amg_warning.setStyleSheet("color: #E65100; font-size: 9px; padding: 4px; background-color: #FFF3E0;")
                self.amg_warning.show()
            elif self.mesh is None:
                self.amg_warning.setText(
                    "ℹ️ AMG è consigliato per mesh >200k celle. "
                    "Per mesh più piccole, Jacobi è generalmente più veloce."
                )
                self.amg_warning.setStyleSheet("color: #E65100; font-size: 9px; padding: 4px; background-color: #FFF3E0;")
                self.amg_warning.show()
            else:
                self.amg_warning.setText(
                    f"✅ Mesh con {self.mesh.N_total:,} celle. AMG è una buona scelta!"
                )
                self.amg_warning.setStyleSheet("color: #2E7D32; font-size: 9px; padding: 4px; background-color: #E8F5E9;")
                self.amg_warning.show()
        else:
            self.amg_warning.hide()
    
    def _on_tolerance_changed(self, tol_str: str):
        """Aggiorna descrizione tolleranza"""
        descriptions = {
            "1e-10 (Alta precisione)": "🔬 Massima precisione. Molte iterazioni. "
                                        "Per analisi dettagliate e validazione.",
            "1e-8 (Default)": "⚖️ Buon compromesso tra velocità e precisione. "
                              "Adatto per la maggior parte degli usi.",
            "1e-6 (Veloce)": "⚡ Sufficiente per visualizzazione e analisi qualitativa. "
                             "Circa 2x più veloce del default.",
            "1e-4 (Molto veloce)": "🚀 Solo per test rapidi e debugging. "
                                    "Precisione ridotta ma molto veloce."
        }
        self.tolerance_desc.setText(descriptions.get(tol_str, ""))
    
    def _get_tolerance_value(self) -> float:
        """Estrae il valore numerico della tolleranza dalla combo"""
        tol_map = {
            "1e-10 (Alta precisione)": 1e-10,
            "1e-8 (Default)": 1e-8,
            "1e-6 (Veloce)": 1e-6,
            "1e-4 (Molto veloce)": 1e-4
        }
        return tol_map.get(self.tolerance_combo.currentText(), 1e-8)
    
    def _on_precision_changed(self, prec_str: str):
        """Aggiorna descrizione precisione"""
        if "float16" in prec_str:
            self.precision_desc.setText(
                "<b>float16</b>: 4x meno memoria, ma precisione limitata (~3 cifre).<br>"
                "Potrebbe non convergere per problemi difficili."
            )
            self.precision_desc.setStyleSheet(
                "color: #E65100; font-size: 9px; padding: 4px; background-color: #FFF3E0;"
            )
        elif "float32" in prec_str:
            self.precision_desc.setText(
                "<b>float32</b>: 2x meno memoria, ~7 cifre di precisione.<br>"
                "Buon compromesso per mesh grandi. Speedup memory-bound."
            )
            self.precision_desc.setStyleSheet(
                "color: #1565C0; font-size: 9px; padding: 4px; background-color: #E3F2FD;"
            )
        else:
            self.precision_desc.setText(
                "<b>float64</b>: massima precisione (~15 cifre).<br>"
                "Standard per calcoli scientifici."
            )
            self.precision_desc.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
    
    def _get_precision(self) -> str:
        """Estrae il tipo di precisione dalla combo"""
        prec_map = {
            "float64 (doppia)": "float64",
            "float32 (singola)": "float32",
            "float16 (mezza)": "float16"
        }
        return prec_map.get(self.precision_combo.currentText(), "float64")
    
    def _get_preconditioner_value(self) -> str:
        """Converte il testo della combo nel nome del precondizionatore per il solver"""
        prec_map = {
            "jacobi": "jacobi",
            "none": "none",
            "ilu": "ilu",
            "amg (Ruge-Stuben)": "amg_rs",
            "amg (Smoothed Aggregation)": "amg_sa"
        }
        return prec_map.get(self.preconditioner_combo.currentText(), "jacobi")
    
    def _get_n_threads(self) -> int:
        """Estrae il numero di thread dalla combo (solo per CPU)"""
        thread_map = {
            "CPU: Auto (tutti)": 0,
            "CPU: Tutti - 1": -1,
            "CPU: 4 core": 4,
            "CPU: 2 core": 2,
            "CPU: 1 core": 1,
            "GPU: CUDA (NVIDIA)": -1,
            "GPU: OpenCL (AMD/Intel/NVIDIA)": -1
        }
        return thread_map.get(self.threads_combo.currentText(), -1)
    
    def _is_gpu_selected(self) -> bool:
        """Verifica se è selezionata la modalità GPU"""
        return "GPU:" in self.threads_combo.currentText()
    
    def _get_gpu_backend(self) -> str:
        """Restituisce il backend GPU selezionato: 'cuda', 'opencl' o None"""
        text = self.threads_combo.currentText()
        if "CUDA" in text:
            return "cuda"
        elif "OpenCL" in text:
            return "opencl"
        return None
    
    def _on_compute_mode_changed(self, text: str):
        """Callback quando cambia la modalità di calcolo (CPU/GPU)"""
        from src.solver.steady_state import HAS_CUPY, HAS_OPENCL, get_gpu_info
        
        if "CUDA" in text:
            # Verifica disponibilità CUDA
            if HAS_CUPY:
                info = get_gpu_info()
                name = info.get('name', 'Unknown')
                mem = info.get('memory_gb', 0)
                desc = f"<b>NVIDIA CUDA disponibile</b><br>GPU: {name}<br>Memoria: {mem:.1f} GB - Speedup 5-50x"
                color = "#1565c0"
                bg = "#e3f2fd"
            else:
                desc = (
                    "<b>CUDA NON disponibile!</b><br>"
                    "Installa con: pip install cupy-cuda11x<br>"
                    "(oppure cupy-cuda12x per CUDA 12)"
                )
                color = "#d84315"
                bg = "#fff3e0"
            
            self.threads_desc.setText(desc)
            self.threads_desc.setStyleSheet(
                f"color: {color}; font-size: 9px; padding: 4px; "
                f"background-color: {bg}; border-radius: 4px;"
            )
        
        elif "OpenCL" in text:
            # Verifica disponibilità OpenCL
            if HAS_OPENCL:
                info = get_gpu_info()
                name = info.get('name', 'Unknown')
                mem = info.get('memory_gb', 0)
                backend = info.get('backend', 'opencl')
                
                if backend == 'opencl-cpu':
                    desc = f"<b>OpenCL CPU disponibile</b><br>Device: {name}<br>Accelerazione parallela"
                else:
                    desc = f"<b>OpenCL GPU disponibile</b><br>GPU: {name}<br>Memoria: {mem:.1f} GB - Speedup 2-10x"
                color = "#c62828"
                bg = "#ffebee"
            else:
                desc = (
                    "<b>OpenCL NON disponibile!</b><br>"
                    "Installa con: pip install pyopencl<br>"
                    "Richiede driver OpenCL (inclusi nei driver GPU)"
                )
                color = "#d84315"
                bg = "#fff3e0"
            
            self.threads_desc.setText(desc)
            self.threads_desc.setStyleSheet(
                f"color: {color}; font-size: 9px; padding: 4px; "
                f"background-color: {bg}; border-radius: 4px;"
            )
        
        else:
            # CPU mode
            self.threads_desc.setText(
                "<b>Auto</b>: massima velocità, può rallentare il sistema.<br>"
                "<b>Tutti - 1</b>: raccomandato, lascia un core libero per la GUI."
            )
            self.threads_desc.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
    
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
        
        # Crea config temporanea con offset
        config = HeaterConfig(
            power_total=self.power_spin.value(),
            n_heaters=self.n_heaters_spin.value(),
            pattern=self._get_heater_pattern_enum(),
            heater_radius=self.heater_radius_spin.value(),
            offset_bottom=self.heater_offset_bottom_spin.value(),
            offset_top=self.heater_offset_top_spin.value(),
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
        
        # Calcola quote z per le resistenze (dentro lo storage con offset)
        base_z = 0.3
        slab_bottom = self.slab_bottom_spin.value()
        height = self.height_spin.value()
        slab_top = self.slab_top_spin.value()
        
        z_storage_start = base_z + slab_bottom
        z_storage_end = z_storage_start + height
        z_bottom = z_storage_start + config.offset_bottom
        z_top = z_storage_end - config.offset_top
        
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
            self.heater_positions_list.addItem(f"Z: {z_bottom:.2f}m - {z_top:.2f}m")
    
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
        
        # I tubi vanno da slab_bottom_start a steel_slab_end
        base_z = 0.3
        slab_bottom = self.slab_bottom_spin.value()
        height = self.height_spin.value()
        slab_top = self.slab_top_spin.value()
        steel_slab = self.steel_slab_spin.value()
        
        z_bottom = base_z  # Da inizio slab isolante inferiore
        z_top = base_z + slab_bottom + height + slab_top + steel_slab
        
        elements = config.generate_positions(
            center_x, center_y,
            r_max,
            z_bottom, z_top
        )
        
        for i, elem in enumerate(elements):
            item = QListWidgetItem(f"{i+1}: ({elem.x:.2f}, {elem.y:.2f}) - Ø{elem.radius*2*1000:.0f}mm")
            self.tube_positions_list.addItem(item)
        self.tube_positions_list.addItem(f"--- Totale: {len(elements)} tubi ---")
        self.tube_positions_list.addItem(f"Z: {z_bottom:.2f}m - {z_top:.2f}m")
    
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
        
        # Riga 1: Campo da visualizzare (includes Geometry option)
        viz_layout.addWidget(QLabel("Campo:"), 0, 0)
        self.field_combo = QComboBox()
        self.field_combo.addItems(["Temperature", "Material", "Geometry", "k", "Q"])
        self.field_combo.currentTextChanged.connect(self.update_visualization)
        viz_layout.addWidget(self.field_combo, 0, 1, 1, 2)
        
        # Riga 2: Controlli clip position
        viz_layout.addWidget(QLabel("Asse taglio:"), 1, 0)
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(["x", "y", "z"])
        self.axis_combo.setCurrentIndex(2)
        self.axis_combo.currentTextChanged.connect(self._on_axis_changed)
        viz_layout.addWidget(self.axis_combo, 1, 1)
        
        viz_layout.addWidget(QLabel("Posizione:"), 1, 2)
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(1, 99)  # Avoid 0 and 100 to prevent crashes
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self._on_slider_changed)
        viz_layout.addWidget(self.slice_slider, 1, 3, 1, 2)
        
        self.slice_pos_label = QLabel("0.00 m")
        viz_layout.addWidget(self.slice_pos_label, 1, 5)
        
        # Riga 3: Opacità
        viz_layout.addWidget(QLabel("Opacità:"), 2, 0)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        viz_layout.addWidget(self.opacity_slider, 2, 1, 1, 4)
        
        self.opacity_label = QLabel("80%")
        viz_layout.addWidget(self.opacity_label, 2, 5)
        
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
            "Thermal Battery Simulation\n\n"
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
        - CylinderGeometry: Multi-zone structure with slabs and conical roof
        - HeaterConfig: heater pattern, power, positions (inside storage)
        - TubeConfig: tube pattern, h_fluid, T_fluid, positions (inside storage)
        - BatteryGeometry: combines all above with material selections
        
        ZONE STRUCTURE (from bottom to top):
        1. CONCRETE: Base foundation layer (base_z thickness)
        2. SHELL: Steel shell extending full height
        3. RADIAL INSULATION: Annular insulation layer between shell and storage
        4. SLAB BOTTOM: Bottom insulation slab (under storage)
        5. STORAGE: Central zone with storage material containing tubes/heaters
        6. SLAB TOP: Top insulation slab (above storage)
        7. STEEL SLAB TOP (optional): Steel slab under roof
        8. CONE: Conical steel roof with optional sand fill
        9. AIR: External air
        
        Returns:
            tuple: (spacing, Lx, Ly, Lz, BatteryGeometry)
        
        NOTE: This function does NOT create the mesh - it only builds the configuration.
        """
        # ==========================================================================
        # STEP 1: Read domain and mesh parameters from GUI
        # ==========================================================================
        d = self.spacing_spin.value()
        Lx, Ly, Lz = self.lx_spin.value(), self.ly_spin.value(), self.lz_spin.value()

        center_x, center_y = Lx / 2, Ly / 2
        radius = self.radius_spin.value()
        height = self.height_spin.value()

        # ==========================================================================
        # STEP 2: Create cylinder geometry with slabs and roof
        # ==========================================================================
        cylinder = CylinderGeometry(
            center_x=center_x,
            center_y=center_y,
            base_z=0.3,
            height=height,
            r_storage=radius,  # Main storage zone radius
            insulation_thickness=self.insulation_thickness_spin.value(),
            shell_thickness=self.shell_thickness_spin.value(),
            # Nuovi parametri: slab isolanti
            insulation_slab_bottom=self.slab_bottom_spin.value(),
            insulation_slab_top=self.slab_top_spin.value(),
            # Nuovi parametri: tetto
            roof_angle_deg=self.roof_angle_spin.value(),
            steel_slab_top=self.steel_slab_spin.value(),
            fill_cone_with_sand=self.fill_cone_check.isChecked(),
            enable_cone_roof=self.enable_cone_roof_check.isChecked(),
            # Sfasamento angolare
            phase_offset_deg=self.phase_offset_spin.value(),
        )

        heater_config = HeaterConfig(
            power_total=self.power_spin.value(),
            n_heaters=self.n_heaters_spin.value(),
            pattern=self._get_heater_pattern_enum(),
            heater_radius=self.heater_radius_spin.value(),
            # Offset verticali resistenze
            offset_bottom=self.heater_offset_bottom_spin.value(),
            offset_top=self.heater_offset_top_spin.value(),
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
        self.log(f"Phase offset: {cylinder.phase_offset_deg}°")
        
        # Log nuovi parametri geometrici
        self.log(f"  Struttura: base_z={cylinder.base_z}m, storage_height={height}m")
        self.log(f"  Slab isolanti: bottom={cylinder.insulation_slab_bottom}m, top={cylinder.insulation_slab_top}m")
        self.log(f"  Tetto conico: angle={cylinder.roof_angle_deg}°, roof_height={cylinder.roof_height:.3f}m")
        if cylinder.steel_slab_top > 0:
            self.log(f"  Steel slab sotto tetto: {cylinder.steel_slab_top}m")
        if cylinder.fill_cone_with_sand:
            self.log(f"  Cono riempito con sabbia")
        self.log(f"  Altezza totale struttura: {cylinder.total_height:.3f}m")
        
        # Verifica che Lz sia sufficiente
        min_Lz_required = cylinder.total_height + 0.5  # Margine aria sopra
        if Lz < min_Lz_required:
            self.log(f"⚠️ ATTENZIONE: Lz={Lz}m potrebbe essere insufficiente!")
            self.log(f"    Altezza minima consigliata: {min_Lz_required:.2f}m")

        geom = BatteryGeometry(
            cylinder=cylinder,
            heaters=heater_config,
            tubes=tube_config,
            storage_material=self.storage_combo.currentText(),
            insulation_material=self.insulation_combo.currentText(),
            packing_fraction=self.packing_spin.value() / 100.0,
        )

        return d, Lx, Ly, Lz, geom

    def build_geometry(self):
        """
        Costruisce e visualizza la geometria analitica usando la vista Geometry.
        
        Questo metodo:
        1. Legge tutti i parametri dalla GUI
        2. Crea l'oggetto BatteryGeometry
        3. Mostra la visualizzazione colorata con zone e legenda
        
        Non crea la mesh - per quello usare build_mesh().
        """
        self.log("=" * 50)
        self.log("Costruzione geometria...")

        try:
            _, Lx, Ly, Lz, geom = self._build_battery_geometry_from_inputs()
            self.battery_geometry = geom

            cyl = geom.cylinder
            self.log(f"  R_storage: {cyl.r_storage:.2f} m")
            self.log(f"  R_insulation: {cyl.r_insulation:.2f} m")
            self.log(f"  R_shell: {cyl.r_shell:.2f} m")
            self.log(f"  Altezza: {cyl.height:.2f} m")
            self.log(f"  Sfasamento angolare: {cyl.phase_offset_deg:.1f}°")
            
            # Usa la nuova visualizzazione Geometry colorata
            self.plotter.clear()
            self._render_geometry_view_standalone(Lx, Ly, Lz)
            
            self.plotter.reset_camera()
            self.plotter.render()
            self.status_label.setText("Stato: Geometria costruita")
            self.log("Geometria costruita con successo")

        except Exception as e:
            self.log(f"ERRORE costruzione geometria: {e}")
            self.status_label.setText(f"Errore: {e}")
            QMessageBox.critical(self, "Errore", str(e))
    
    def _render_geometry_view_standalone(self, Lx: float, Ly: float, Lz: float):
        """
        Render della geometria analitica con zone colorate (versione standalone).
        
        Visualizzazione completa della struttura:
        - FONDAZIONE (grigio scuro)
        - SLAB ISOLANTE INFERIORE (giallo)
        - STORAGE (beige/sabbia)
        - SLAB ISOLANTE SUPERIORE (giallo)
        - STEEL SLAB (grigio metallico)
        - TETTO CONICO (grigio metallico) con eventuale riempimento sabbia
        - SHELL laterale (grigio)
        - ISOLAMENTO RADIALE (giallo)
        - HEATERS (rosso) e TUBES (blu)
        """
        geom = self.battery_geometry
        cyl = geom.cylinder
        
        # Domain outline
        domain = pv.Cube(
            center=(Lx / 2, Ly / 2, Lz / 2),
            x_length=Lx,
            y_length=Ly,
            z_length=Lz
        )
        self.plotter.add_mesh(domain.outline(), color='gray', line_width=1)
        
        legend_entries = []
        
        # Colori per le zone
        COLOR_CONCRETE = [0.35, 0.35, 0.35]
        COLOR_INSULATION = [0.95, 0.85, 0.55]
        COLOR_STORAGE = [0.85, 0.65, 0.45]
        COLOR_STEEL = [0.50, 0.50, 0.55]
        COLOR_HEATER = [0.95, 0.2, 0.1]
        COLOR_TUBE = [0.3, 0.3, 0.8]
        
        # =====================================================================
        # 1. FONDAZIONE (cilindro sotto la struttura)
        # =====================================================================
        if cyl.base_z > 0:
            z_foundation = cyl.base_z / 2
            foundation = pv.Cylinder(
                center=(cyl.center_x, cyl.center_y, z_foundation),
                direction=(0, 0, 1),
                radius=float(cyl.r_shell + 0.5),
                height=float(cyl.base_z),
                resolution=72
            )
            self.plotter.add_mesh(foundation, color=COLOR_CONCRETE, opacity=0.5)
            legend_entries.append(["CONCRETE", COLOR_CONCRETE])
        
        # =====================================================================
        # 2. SHELL LATERALE (copre tutta l'altezza della struttura)
        # =====================================================================
        shell_z_center = (cyl.base_z + cyl.z_shell_top) / 2
        shell_height = cyl.z_shell_top - cyl.base_z
        
        # Shell esterna
        shell_outer = pv.Cylinder(
            center=(cyl.center_x, cyl.center_y, shell_z_center),
            direction=(0, 0, 1),
            radius=float(cyl.r_shell),
            height=float(shell_height),
            resolution=72
        )
        self.plotter.add_mesh(shell_outer, color=COLOR_STEEL, opacity=0.3)
        legend_entries.append(["STEEL SHELL", COLOR_STEEL])
        
        # =====================================================================
        # 3. ISOLAMENTO RADIALE (tra storage e shell)
        # =====================================================================
        insul_z_center = (cyl.base_z + cyl.z_slab_top_end) / 2
        insul_height = cyl.z_slab_top_end - cyl.base_z
        
        insul_cyl = pv.Cylinder(
            center=(cyl.center_x, cyl.center_y, insul_z_center),
            direction=(0, 0, 1),
            radius=float(cyl.r_insulation),
            height=float(insul_height),
            resolution=72
        )
        self.plotter.add_mesh(insul_cyl, color=COLOR_INSULATION, opacity=0.4)
        legend_entries.append(["INSULATION", COLOR_INSULATION])
        
        # =====================================================================
        # 4. SLAB ISOLANTE INFERIORE
        # =====================================================================
        if cyl.insulation_slab_bottom > 0:
            slab_bot_z = (cyl.z_slab_bottom_start + cyl.z_slab_bottom_end) / 2
            slab_bot = pv.Cylinder(
                center=(cyl.center_x, cyl.center_y, slab_bot_z),
                direction=(0, 0, 1),
                radius=float(cyl.r_storage),
                height=float(cyl.insulation_slab_bottom),
                resolution=72
            )
            self.plotter.add_mesh(slab_bot, color=COLOR_INSULATION, opacity=0.6)
        
        # =====================================================================
        # 5. STORAGE (zona principale sabbia)
        # =====================================================================
        storage_z_center = (cyl.z_storage_start + cyl.z_storage_end) / 2
        storage = pv.Cylinder(
            center=(cyl.center_x, cyl.center_y, storage_z_center),
            direction=(0, 0, 1),
            radius=float(cyl.r_storage),
            height=float(cyl.height),
            resolution=72
        )
        self.plotter.add_mesh(storage, color=COLOR_STORAGE, opacity=0.6)
        legend_entries.append(["STORAGE", COLOR_STORAGE])
        
        # =====================================================================
        # 6. SLAB ISOLANTE SUPERIORE
        # =====================================================================
        if cyl.insulation_slab_top > 0:
            slab_top_z = (cyl.z_slab_top_start + cyl.z_slab_top_end) / 2
            slab_top = pv.Cylinder(
                center=(cyl.center_x, cyl.center_y, slab_top_z),
                direction=(0, 0, 1),
                radius=float(cyl.r_storage),
                height=float(cyl.insulation_slab_top),
                resolution=72
            )
            self.plotter.add_mesh(slab_top, color=COLOR_INSULATION, opacity=0.6)
        
        # =====================================================================
        # 7. STEEL SLAB TOP (opzionale)
        # =====================================================================
        if cyl.steel_slab_top > 0:
            steel_slab_z = (cyl.z_slab_top_end + cyl.z_steel_slab_end) / 2
            steel_slab = pv.Cylinder(
                center=(cyl.center_x, cyl.center_y, steel_slab_z),
                direction=(0, 0, 1),
                radius=float(cyl.r_insulation),
                height=float(cyl.steel_slab_top),
                resolution=72
            )
            self.plotter.add_mesh(steel_slab, color=COLOR_STEEL, opacity=0.7)
        
        # =====================================================================
        # 8. TETTO CONICO (se abilitato)
        # =====================================================================
        if cyl.enable_cone_roof and cyl.roof_height > 0:
            cone = pv.Cone(
                center=(cyl.center_x, cyl.center_y, cyl.z_cone_base + cyl.roof_height/2),
                direction=(0, 0, 1),
                height=float(cyl.roof_height),
                radius=float(cyl.r_shell),
                resolution=72
            )
            self.plotter.add_mesh(cone, color=COLOR_STEEL, opacity=0.5)
            legend_entries.append(["CONE ROOF", COLOR_STEEL])
            
            # Riempimento sabbia sotto il cono (opzionale)
            if cyl.fill_cone_with_sand:
                cone_fill = pv.Cone(
                    center=(cyl.center_x, cyl.center_y, cyl.z_cone_base + cyl.roof_height/2),
                    direction=(0, 0, 1),
                    height=float(cyl.roof_height * 0.95),
                    radius=float(cyl.r_shell * 0.95),
                    resolution=72
                )
                self.plotter.add_mesh(cone_fill, color=COLOR_STORAGE, opacity=0.4)
        
        # =====================================================================
        # 9. RESISTENZE
        # =====================================================================
        heater_z_start = cyl.z_storage_start + geom.heaters.offset_bottom
        heater_z_end = cyl.z_storage_end - geom.heaters.offset_top
        
        heater_elems = []
        if geom.heaters.pattern != HeaterPattern.UNIFORM_ZONE:
            heater_elems = geom.heaters.generate_positions(
                cyl.center_x, cyl.center_y,
                0, cyl.r_storage * 0.9,
                heater_z_start, heater_z_end,
                phase_offset=cyl.phase_offset_rad
            )
            for htr in heater_elems:
                h_center = (htr.x, htr.y, (htr.z_bottom + htr.z_top) / 2)
                h_cyl = pv.Cylinder(
                    center=h_center,
                    direction=(0, 0, 1),
                    radius=float(htr.radius),
                    height=float(htr.z_top - htr.z_bottom),
                    resolution=24
                )
                self.plotter.add_mesh(h_cyl, color=COLOR_HEATER, opacity=0.9)
            if heater_elems:
                legend_entries.append(["HEATERS", COLOR_HEATER])
        
        # =====================================================================
        # 10. TUBI (attraversano tutta la struttura)
        # =====================================================================
        tube_z_start = cyl.z_slab_bottom_start
        tube_z_end = cyl.z_steel_slab_end
        
        tube_elems = geom.tubes.generate_positions(
            cyl.center_x, cyl.center_y,
            cyl.r_storage * 0.9,
            tube_z_start, tube_z_end
        )
        for tube in tube_elems:
            t_center = (tube.x, tube.y, (tube.z_bottom + tube.z_top) / 2)
            t_cyl = pv.Cylinder(
                center=t_center,
                direction=(0, 0, 1),
                radius=float(tube.radius),
                height=float(tube.z_top - tube.z_bottom),
                resolution=24
            )
            self.plotter.add_mesh(t_cyl, color=COLOR_TUBE, opacity=0.8)
        if tube_elems:
            legend_entries.append(["TUBES", COLOR_TUBE])
        
        # =====================================================================
        # LEGENDA E INFO
        # =====================================================================
        if legend_entries:
            self.plotter.add_legend(legend_entries, bcolor='white', 
                                    face='rectangle', loc='upper right')
        
        self.plotter.add_axes()
        
        # Info text con dettagli struttura
        roof_text = f"Cone: {cyl.roof_angle_deg:.0f}°" if cyl.enable_cone_roof else "Flat roof"
        self.plotter.add_text(
            f"Geometria Completa\n"
            f"R_storage={cyl.r_storage:.2f}m, H_storage={cyl.height:.2f}m\n"
            f"Slab: {cyl.insulation_slab_bottom:.2f}m / {cyl.insulation_slab_top:.2f}m\n"
            f"Insulation: {cyl.insulation_thickness:.2f}m, Shell: {cyl.shell_thickness:.3f}m\n"
            f"{roof_text}, Total H: {cyl.total_height:.2f}m",
            position='upper_left', font_size=9
        )
    
    # NOTA: La funzione run_simulation() è stata spostata nella sezione
    #       "NUOVI METODI PER SCHEDA ANALISI" in fondo alla classe
        
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
    
    # =========================================================================
    # VISUALIZATION HELPER METHODS (for seamless updates)
    # =========================================================================
    
    def _on_axis_changed(self):
        """Called when axis combo changes - update visualization"""
        self._update_position_label()
        self.update_visualization()
    
    def _on_slider_changed(self):
        """Called when position slider changes - update seamlessly"""
        self._update_position_label()
        self._update_visualization_seamless()
    
    def _on_opacity_changed(self):
        """Called when opacity slider changes - update seamlessly"""
        opacity_pct = self.opacity_slider.value()
        self.opacity_label.setText(f"{opacity_pct}%")
        self._update_visualization_seamless()
    
    def _update_position_label(self):
        """Update the position label without rebuilding visualization"""
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
    
    def _update_visualization_seamless(self):
        """Update visualization without flickering using render() instead of full rebuild"""
        if self.mesh is None:
            return
        # For seamless updates during slider movement, rebuild the scene
        # PyVista requires rebuilding meshes, but we skip camera reset
        self._do_visualization(reset_camera=False)
    
    def update_visualization(self):
        """Full visualization update with camera reset"""
        if self.mesh is None:
            return
        self._do_visualization(reset_camera=True)
    
    def _do_visualization(self, reset_camera: bool = True):
        """
        Core visualization routine - renders the 3D view.
        
        Unified Volume 3D view with:
        - Clip plane controlled by position slider
        - Opacity controlled by opacity slider
        - Field selected from dropdown (Temperature, Material, Geometry, k, Q)
        """
        if self.mesh is None:
            return
        
        self.plotter.clear()
        
        field = self.field_combo.currentText()
        cmap = self.cmap_combo.currentText()
        
        # Opacity
        opacity_pct = self.opacity_slider.value()
        self.opacity_label.setText(f"{opacity_pct}%")
        opacity = opacity_pct / 100.0
        
        # Handle special "Geometry" field - shows real analytic geometry
        if field == "Geometry":
            self._render_geometry_view(opacity)
            if reset_camera:
                self.plotter.reset_camera()
            return
        
        # Create PyVista grid
        grid = pv.ImageData(
            dimensions=(self.mesh.Nx + 1, self.mesh.Ny + 1, self.mesh.Nz + 1),
            spacing=(self.mesh.dx, self.mesh.dy, self.mesh.dz),
            origin=(0, 0, 0)
        )
        
        # Add data
        grid.cell_data["Temperature"] = self.mesh.T.ravel(order='F')
        grid.cell_data["Material"] = self.mesh.material_id.ravel(order='F').astype(float)
        grid.cell_data["k"] = self.mesh.k.ravel(order='F')
        grid.cell_data["Q"] = self.mesh.Q.ravel(order='F')
        
        # Color range
        if self.auto_range_check.isChecked():
            data = grid.cell_data[field]
            clim = [float(data.min()), float(data.max())]
            self.tmin_spin.setValue(clim[0])
            self.tmax_spin.setValue(clim[1])
        else:
            clim = [self.tmin_spin.value(), self.tmax_spin.value()]
        
        # Clip plane position and normal
        axis = self.axis_combo.currentText()
        slider_pct = self.slice_slider.value() / 100.0
        # Clamp to avoid edge cases
        slider_pct = max(0.02, min(0.98, slider_pct))
        
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
        
        # === UNIFIED VOLUME 3D VISUALIZATION ===
        # Clip the grid and remove air (Material ID = 0)
        clipped = grid.clip(normal=normal, origin=origin, invert=False)
        
        # Remove air cells for cleaner visualization
        if clipped.n_cells > 0:
            clipped_no_air = clipped.threshold(value=0.5, scalars="Material")
        else:
            clipped_no_air = clipped
        
        if clipped_no_air.n_cells > 0:
            if field == "Material":
                self.plotter.add_mesh(
                    clipped_no_air, scalars=field, 
                    cmap=self._get_material_cmap(),
                    clim=[0, 7], opacity=opacity, show_scalar_bar=False
                )
            else:
                self.plotter.add_mesh(
                    clipped_no_air, scalars=field, 
                    cmap=cmap, clim=clim, opacity=opacity
                )
        
        # Add domain outline for reference
        self.plotter.add_mesh(grid.outline(), color='gray', line_width=1)
        
        # === LEGEND / SCALAR BAR ===
        if field == "Material":
            self._add_material_legend()
        else:
            unit = "°C" if field == "Temperature" else ("W/m³" if field == "Q" else "W/(m·K)")
            self.plotter.add_scalar_bar(
                title=f"{field} [{unit}]",
                vertical=True
            )
        
        # Axes
        self.plotter.add_axes()
        
        # Info text
        info_text = f"Taglio {axis.upper()} = {pos:.2f} m | Opacità {opacity_pct}%"
        self.plotter.add_text(info_text, position='upper_left', font_size=10)
        
        if reset_camera:
            self.plotter.reset_camera()
        else:
            self.plotter.render()
    
    def _render_geometry_view(self, opacity: float):
        """
        Render the analytic geometry view with proper colors and legend.
        
        Shows the 4 concentric zones:
        - STORAGE: central zone with storage material (beige)
        - INSULATION: thermal insulation (yellow)  
        - STEEL: external shell (dark gray)
        - TUBES and HEATERS: discrete elements inside storage
        """
        if self.battery_geometry is None:
            self.plotter.add_text("Nessuna geometria - Costruisci prima la geometria", 
                                  position='upper_left', font_size=12, color='red')
            return
        
        geom = self.battery_geometry
        cyl = geom.cylinder
        
        # Domain outline
        domain = pv.Cube(
            center=(self.mesh.Lx / 2, self.mesh.Ly / 2, self.mesh.Lz / 2),
            x_length=self.mesh.Lx,
            y_length=self.mesh.Ly,
            z_length=self.mesh.Lz
        )
        self.plotter.add_mesh(domain.outline(), color='gray', line_width=1)
        
        z_center = cyl.base_z + cyl.height / 2
        
        # Zone colors - 4 zone semplificate
        zone_info = [
            (cyl.r_shell, cyl.r_insulation, "STEEL", [0.40, 0.40, 0.45]),
            (cyl.r_insulation, cyl.r_storage, "INSULATION", [0.95, 0.85, 0.55]),
            (cyl.r_storage, 0, "STORAGE", [0.85, 0.65, 0.45]),
        ]
        
        legend_entries = []
        
        for r_outer, r_inner, name, color in zone_info:
            if r_outer <= 0:
                continue
            cyl_mesh = pv.Cylinder(
                center=(cyl.center_x, cyl.center_y, z_center),
                direction=(0, 0, 1),
                radius=float(r_outer),
                height=float(cyl.height),
                resolution=72
            )
            self.plotter.add_mesh(cyl_mesh, color=color, opacity=opacity * 0.7, 
                                  show_edges=False)
            legend_entries.append([name, color])
        
        # Add individual heater elements if discrete pattern (inside storage)
        heater_elems = []
        if geom.heaters.pattern != HeaterPattern.UNIFORM_ZONE:
            heater_elems = geom.heaters.generate_positions(
                cyl.center_x, cyl.center_y,
                0, cyl.r_storage * 0.9,
                cyl.base_z, cyl.top_z,
                phase_offset=cyl.phase_offset_rad
            )
            for htr in heater_elems:
                h_center = (htr.x, htr.y, (htr.z_bottom + htr.z_top) / 2)
                h_cyl = pv.Cylinder(
                    center=h_center,
                    direction=(0, 0, 1),
                    radius=float(htr.radius),
                    height=float(htr.z_top - htr.z_bottom),
                    resolution=24
                )
                self.plotter.add_mesh(h_cyl, color=[0.95, 0.2, 0.1], opacity=0.9)
            if heater_elems:
                legend_entries.append(["HEATERS", [0.95, 0.2, 0.1]])
        
        # Add individual tube elements (inside storage)
        tube_elems = geom.tubes.generate_positions(
            cyl.center_x, cyl.center_y,
            cyl.r_storage * 0.9,
            cyl.base_z, cyl.top_z
        )
        for tube in tube_elems:
            t_center = (tube.x, tube.y, (tube.z_bottom + tube.z_top) / 2)
            t_cyl = pv.Cylinder(
                center=t_center,
                direction=(0, 0, 1),
                radius=float(tube.radius),
                height=float(tube.z_top - tube.z_bottom),
                resolution=24
            )
            self.plotter.add_mesh(t_cyl, color=[0.3, 0.3, 0.8], opacity=0.8)
        if tube_elems:
            legend_entries.append(["TUBES", [0.3, 0.3, 0.8]])
        
        # Legend
        if legend_entries:
            self.plotter.add_legend(legend_entries, bcolor='white', 
                                    face='rectangle', loc='upper right')
        
        # Axes
        self.plotter.add_axes()
        
        # Info text
        self.plotter.add_text(
            f"Geometria Analitica (4 Zone)\n"
            f"R_storage={cyl.r_storage:.2f}m, H={cyl.height:.2f}m\n"
            f"Insulation: {cyl.insulation_thickness:.2f}m, Steel: {cyl.shell_thickness:.3f}m",
            position='upper_left', font_size=10
        )
    
    def _add_material_legend(self):
        """Add legend for material visualization"""
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
    
    # =========================================================================
    # NUOVI METODI PER SCHEDA ANALISI
    # =========================================================================
    
    def _on_analysis_type_changed(self, analysis_type: str):
        """Callback quando cambia il tipo di analisi"""
        self.log(f"[ANALISI] Tipo cambiato: {analysis_type}")
        
        # Aggiorna testo del pulsante in base al tipo
        if analysis_type == "steady":
            self.run_btn.setText("▶ Calcola Stazionario")
        elif analysis_type == "losses":
            self.run_btn.setText("▶ Calcola Perdite")
        elif analysis_type == "transient":
            self.run_btn.setText("▶ Simula Transitorio")
    
    def _on_state_loaded(self, state: SimulationState):
        """Callback quando viene caricato uno stato salvato"""
        self.log(f"[STATE] Caricato: {state.name}")
        
        # Verifica compatibilità geometria
        if self.mesh is not None:
            from src.io.state_manager import compute_geometry_hash
            current_hash = compute_geometry_hash(self.mesh)
            if current_hash != state.geometry_hash:
                reply = QMessageBox.question(
                    self, "Geometria diversa",
                    "Lo stato salvato ha una geometria diversa dalla mesh attuale.\n"
                    "Vuoi comunque caricare la temperatura (potrebbe non essere corretto)?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
            
            # Carica temperatura
            if state.T.shape == self.mesh.T.shape:
                self.mesh.T[:] = state.T
                self._simulation_completed = True
                self.update_visualization()
                self.update_statistics()
                self.update_energy_balance()
                self.log(f"[STATE] Temperatura caricata: T_mean = {state.T.mean():.1f} °C")
            else:
                QMessageBox.warning(
                    self, "Errore", 
                    f"Dimensioni incompatibili:\nMesh: {self.mesh.T.shape}\nFile: {state.T.shape}"
                )
        else:
            QMessageBox.warning(self, "Attenzione", "Prima costruisci una mesh!")
    
    def _save_current_state(self):
        """Salva lo stato corrente in un file HDF5"""
        if self.mesh is None:
            QMessageBox.warning(self, "Attenzione", "Prima costruisci la mesh!")
            return
        
        # Prendi nome e descrizione dalla GUI
        name = self.analysis_tab.save_load_widget.save_name_edit.text()
        if not name:
            name = "Simulation"
        description = self.analysis_tab.save_load_widget.save_desc_edit.text()
        
        # Crea oggetto stato
        analysis_type = self.analysis_tab.get_analysis_type()
        state = SimulationState.from_mesh(
            mesh=self.mesh,
            name=name,
            description=description,
            analysis_type=analysis_type
        )
        
        # Seleziona file
        filename, _ = QFileDialog.getSaveFileName(
            self, "Salva Stato", f"{name}.h5",
            "File stato (*.h5)"
        )
        if filename:
            StateManager.save_state(state, filename)
            self.log(f"[STATE] Stato salvato: {filename}")
            QMessageBox.information(self, "Salvato", f"Stato salvato in:\n{filename}")
    
    def run_simulation(self):
        """Esegue la simulazione in base al tipo selezionato"""
        if self.mesh is None:
            QMessageBox.warning(self, "Attenzione", "Prima costruisci la mesh!")
            return
        
        analysis_type = self.analysis_tab.get_analysis_type()
        
        if analysis_type == "steady":
            self._run_steady_simulation()
        elif analysis_type == "losses":
            self._run_losses_analysis()
        elif analysis_type == "transient":
            self._run_transient_simulation()
    
    def _run_steady_simulation(self):
        """Esegue simulazione stazionaria (come prima)"""
        self.log("=" * 50)
        self.log("[ANALISI] Simulazione Stazionaria...")
        self.progress_bar.setValue(0)
        self.status_label.setText("Stato: Configurazione solver...")
        
        try:
            # Configura solver con parametri dalla GUI
            gpu_backend = self._get_gpu_backend()
            precision = self._get_precision()
            config = SolverConfig(
                method=self.solver_combo.currentText(),
                tolerance=self._get_tolerance_value(),
                max_iterations=self.max_iter_spin.value(),
                preconditioner=self._get_preconditioner_value(),
                n_threads=self._get_n_threads(),
                gpu_backend=gpu_backend,
                precision=precision,
                verbose=True
            )
            
            self.log(f"[SOLVER] Metodo: {config.method}, Tolleranza: {config.tolerance:.0e}")
            if gpu_backend == "cuda":
                self.log("[GPU] Backend: CUDA (NVIDIA)")
            elif gpu_backend == "opencl":
                self.log("[GPU] Backend: OpenCL")
            else:
                self.log(f"[CPU] Precondizionatore: {config.preconditioner}, Thread: {config.n_threads}")
            
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
    
    def _run_losses_analysis(self):
        """Esegue analisi perdite impostando T media nello storage"""
        self.log("=" * 50)
        self.log("[ANALISI] Calcolo Perdite...")
        self.progress_bar.setValue(0)
        
        try:
            # Ottieni parametri da scheda analisi
            T_storage = self.analysis_tab.analysis_type_widget.losses_T_spin.value() + 273.15
            T_amb = self.analysis_tab.analysis_type_widget.losses_T_amb_spin.value() + 273.15
            
            self.log(f"  T storage: {T_storage - 273.15:.1f} °C")
            self.log(f"  T ambiente: {T_amb - 273.15:.1f} °C")
            
            # Imposta temperatura nello storage
            from src.core.mesh import MaterialID
            storage_mask = (self.mesh.material_id == MaterialID.SAND.value)
            self.mesh.T[storage_mask] = T_storage
            
            # Imposta condizione al contorno di Dirichlet per lo storage
            # e calcola il flusso necessario per mantenerla
            self.progress_bar.setValue(20)
            
            # Usa il solver con Q=0 e T fissata nello storage
            # Per fare questo dovremmo fissare le celle storage come Dirichlet
            # Approccio semplificato: calcoliamo direttamente le perdite
            
            self.progress_bar.setValue(50)
            
            # Usa EnergyBalanceAnalyzer
            analyzer = EnergyBalanceAnalyzer(self.mesh, T_ambient=T_amb)
            
            # Per calcolare le perdite, impostiamo tutto lo storage a T_storage
            # e calcoliamo il gradiente attraverso l'isolamento
            self.mesh.T[storage_mask] = T_storage
            
            # Imposta condizioni al contorno
            self.mesh.T[self.mesh.boundary_type == 1] = T_amb  # Pareti esterne
            
            # Calcola perdite
            result = analyzer.compute_full_balance()
            
            self.progress_bar.setValue(90)
            
            # Log risultati
            self.log(f"\n[RISULTATO] Perdite Termiche:")
            self.log(f"  Superiore: {result.Q_loss_top/1000:.2f} kW")
            self.log(f"  Laterale: {result.Q_loss_lateral/1000:.2f} kW")
            self.log(f"  Inferiore: {result.Q_loss_bottom/1000:.2f} kW")
            self.log(f"  TOTALE: {result.Q_loss_total/1000:.2f} kW")
            self.log(f"\n  Efficienza exergetica: {result.exergy_efficiency*100:.1f}%")
            
            self.progress_bar.setValue(100)
            self._simulation_completed = True
            self.update_visualization()
            self.update_statistics()
            
            # Mostra messaggio
            QMessageBox.information(
                self, "Analisi Perdite Completata",
                f"Perdite termiche totali: {result.Q_loss_total/1000:.2f} kW\n\n"
                f"  - Superiore: {result.Q_loss_top/1000:.2f} kW\n"
                f"  - Laterale: {result.Q_loss_lateral/1000:.2f} kW\n"
                f"  - Inferiore: {result.Q_loss_bottom/1000:.2f} kW"
            )
            
        except Exception as e:
            self.log(f"ERRORE: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Errore", str(e))
    
    def _run_transient_simulation(self):
        """Esegue simulazione transitoria"""
        self.log("=" * 50)
        self.log("[ANALISI] Simulazione Transitoria...")
        self.progress_bar.setValue(0)
        self.status_label.setText("Stato: Configurazione transitorio...")
        
        try:
            # Ottieni configurazione transitoria
            trans_config = self.analysis_tab.get_transient_config()
            ic = self.analysis_tab.get_initial_condition()
            power_profile = self.analysis_tab.get_power_profile()
            
            self.log(f"  Durata: {trans_config.t_final:.0f} s ({trans_config.t_final/3600:.1f} ore)")
            self.log(f"  dt: {trans_config.dt:.1f} s")
            self.log(f"  Condizione iniziale: {ic.mode}")
            self.log(f"  Potenza: {power_profile.mode}")
            
            self.progress_bar.setValue(5)
            
            # Applica condizione iniziale
            self._apply_initial_condition(ic)
            
            self.progress_bar.setValue(10)
            
            # Configura solver
            gpu_backend = self._get_gpu_backend()
            solver_config = SolverConfig(
                method=self.solver_combo.currentText(),
                tolerance=self._get_tolerance_value(),
                max_iterations=self.max_iter_spin.value(),
                preconditioner=self._get_preconditioner_value(),
                n_threads=self._get_n_threads(),
                gpu_backend=gpu_backend,
                precision=self._get_precision(),
                verbose=False  # Meno verbose per transitorio
            )
            
            self.status_label.setText("Stato: Simulazione transitoria in corso...")
            
            # Avvia thread transitorio
            self.transient_thread = TransientSimulationThread(
                mesh=self.mesh,
                solver_config=solver_config,
                transient_config=trans_config,
                power_profile=power_profile
            )
            self.transient_thread.progress.connect(self._on_transient_progress)
            self.transient_thread.step_completed.connect(self._on_transient_step)
            self.transient_thread.finished.connect(self._on_transient_finished)
            self.transient_thread.error.connect(self.on_simulation_error)
            self.transient_thread.start()
            
            self.run_btn.setEnabled(False)
            self.build_mesh_btn.setEnabled(False)
            
        except Exception as e:
            self.log(f"ERRORE: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Errore", str(e))
    
    def _apply_initial_condition(self, ic: InitialCondition):
        """Applica la condizione iniziale alla mesh"""
        from src.core.mesh import MaterialID
        
        if ic.mode == "uniform":
            T_init = ic.T_uniform + 273.15
            self.mesh.T[:] = T_init
            self.log(f"  T iniziale uniforme: {ic.T_uniform:.1f} °C")
            
        elif ic.mode == "by_material":
            self.mesh.T[self.mesh.material_id == MaterialID.SAND.value] = ic.T_sand + 273.15
            self.mesh.T[self.mesh.material_id == MaterialID.INSULATION.value] = ic.T_insulation + 273.15
            self.mesh.T[self.mesh.material_id == MaterialID.STEEL.value] = ic.T_steel + 273.15
            self.mesh.T[self.mesh.material_id == MaterialID.AIR.value] = ic.T_air + 273.15
            self.mesh.T[self.mesh.material_id == MaterialID.GROUND.value] = ic.T_ground + 273.15
            self.mesh.T[self.mesh.material_id == MaterialID.CONCRETE.value] = ic.T_concrete + 273.15
            self.log(f"  T per materiale applicata")
            
        elif ic.mode == "from_file":
            if ic.file_path:
                state = StateManager.load_state(ic.file_path)
                if state and state.T.shape == self.mesh.T.shape:
                    self.mesh.T[:] = state.T
                    self.log(f"  T caricata da file: {ic.file_path}")
                else:
                    raise ValueError("File stato incompatibile con mesh attuale!")
            else:
                raise ValueError("Nessun file selezionato per condizione iniziale!")
                
        elif ic.mode == "from_steady":
            # Calcola prima lo stazionario
            self.log("  Calcolo stazionario preliminare...")
            solver_config = SolverConfig(
                method=self.solver_combo.currentText(),
                tolerance=self._get_tolerance_value(),
                max_iterations=self.max_iter_spin.value(),
                preconditioner=self._get_preconditioner_value()
            )
            solver = SteadyStateSolver(self.mesh, solver_config)
            solver.build_system()
            result = solver.solve(rebuild=False)
            if result.converged:
                self.log(f"  Stazionario converguto: T_mean = {self.mesh.T.mean() - 273.15:.1f} °C")
            else:
                raise ValueError("Stazionario non converguto!")
    
    def _on_transient_progress(self, value: int, message: str):
        """Callback per progresso transitorio"""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)
    
    def _on_transient_step(self, step: int, t: float, T_mean: float):
        """Callback per ogni step temporale completato"""
        if step % 10 == 0:  # Log ogni 10 step
            self.log(f"  t = {t:.0f} s, T_mean = {T_mean - 273.15:.1f} °C")
        
        # Aggiorna visualizzazione ogni N step
        if step % 20 == 0:
            self.update_visualization()
    
    def _on_transient_finished(self, results: TransientResults):
        """Callback quando la simulazione transitoria è completata"""
        self.progress_bar.setValue(100)
        self.status_label.setText("Stato: Transitorio completato!")
        self.run_btn.setEnabled(True)
        self.build_mesh_btn.setEnabled(True)
        self._simulation_completed = True
        
        self.log(f"\n[RISULTATO] Transitorio completato!")
        self.log(f"  Durata simulata: {results.times[-1]:.0f} s")
        self.log(f"  Step totali: {len(results.times)}")
        self.log(f"  T finale media: {results.T_mean[-1] - 273.15:.1f} °C")
        self.log(f"  T finale min: {results.T_min[-1] - 273.15:.1f} °C")
        self.log(f"  T finale max: {results.T_max[-1] - 273.15:.1f} °C")
        
        # Salva riferimento ai risultati
        self._transient_results = results
        
        # Aggiorna visualizzazione
        self.update_visualization()
        self.update_statistics()
        self.update_energy_balance()
        
        # Chiedi se esportare
        reply = QMessageBox.question(
            self, "Transitorio Completato",
            f"Simulazione completata!\n\n"
            f"Durata: {results.times[-1]/3600:.1f} ore\n"
            f"T finale: {results.T_mean[-1] - 273.15:.1f} °C\n\n"
            f"Vuoi esportare i risultati in CSV?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._export_transient_results(results)
    
    def _export_transient_results(self, results: TransientResults):
        """Esporta risultati transitorio in CSV"""
        filename, _ = QFileDialog.getSaveFileName(
            self, "Esporta Risultati Transitorio", "transient_results.csv",
            "File CSV (*.csv)"
        )
        if filename:
            results.export_csv(filename)
            self.log(f"[EXPORT] Risultati esportati: {filename}")
            QMessageBox.information(self, "Esportato", f"Risultati salvati in:\n{filename}")


class TransientSimulationThread(QThread):
    """Thread per eseguire simulazione transitoria"""
    
    progress = pyqtSignal(int, str)
    step_completed = pyqtSignal(int, float, float)  # step, t, T_mean
    finished = pyqtSignal(object)  # TransientResults
    error = pyqtSignal(str)
    
    def __init__(self, mesh, solver_config, transient_config, power_profile):
        super().__init__()
        self.mesh = mesh
        self.solver_config = solver_config
        self.transient_config = transient_config
        self.power_profile = power_profile
    
    def run(self):
        try:
            results = run_transient_simulation(
                mesh=self.mesh,
                solver_config=self.solver_config,
                t_final=self.transient_config.t_final,
                dt=self.transient_config.dt,
                save_interval=self.transient_config.save_interval,
                power_profile=self.power_profile,
                callback=self._step_callback
            )
            self.finished.emit(results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))
    
    def _step_callback(self, step, t, T_mean):
        """Callback per ogni step temporale"""
        n_steps = int(self.transient_config.t_final / self.transient_config.dt)
        pct = int(10 + 80 * step / max(n_steps, 1))
        self.progress.emit(pct, f"t = {t:.0f} s / {self.transient_config.t_final:.0f} s")
        self.step_completed.emit(step, t, T_mean)


def main():
    """Entry point dell'applicazione"""
    app = QApplication(sys.argv)
    
    # Stile
    app.setStyle("Fusion")
    
    window = ThermalBatteryGUI()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
