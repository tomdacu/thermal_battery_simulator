"""
analysis_tab.py - Scheda Analisi per la GUI

=============================================================================
TAB ANALISI
=============================================================================

Questo modulo contiene i widget per la scheda Analisi, che include:
- Tipo di analisi (Stazionaria, Perdite, Transitoria)
- Condizioni iniziali
- Profilo potenza resistenze
- Profilo estrazione tubi
- Salvataggio/caricamento stato

=============================================================================
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QComboBox,
    QSpinBox, QDoubleSpinBox, QTabWidget, QTextEdit,
    QFileDialog, QCheckBox, QRadioButton, QButtonGroup,
    QTableWidget, QTableWidgetItem, QHeaderView,
    QScrollArea, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from pathlib import Path

from src.core.profiles import (
    PowerProfile, ExtractionProfile, 
    InitialCondition, TransientConfig
)
from src.io.state_manager import StateManager, SimulationState


class AnalysisTypeWidget(QWidget):
    """Widget per la selezione del tipo di analisi"""
    
    analysis_changed = pyqtSignal(str)  # Emette "steady", "losses", "transient"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Gruppo tipo analisi
        type_group = QGroupBox("Tipo di Analisi")
        type_layout = QVBoxLayout()
        
        self.btn_group = QButtonGroup(self)
        
        # Stazionaria
        self.radio_steady = QRadioButton("Stazionaria")
        self.radio_steady.setChecked(True)
        self.btn_group.addButton(self.radio_steady)
        type_layout.addWidget(self.radio_steady)
        
        steady_desc = QLabel(
            "   Calcola l'equilibrio termico con potenza costante.\n"
            "   Utile per: dimensionamento, verifica design."
        )
        steady_desc.setStyleSheet("color: #666; font-size: 9px;")
        type_layout.addWidget(steady_desc)
        
        # Perdite
        self.radio_losses = QRadioButton("Analisi Perdite")
        self.btn_group.addButton(self.radio_losses)
        type_layout.addWidget(self.radio_losses)
        
        losses_desc = QLabel(
            "   Imposta T media nello storage, calcola dispersioni.\n"
            "   Utile per: stima perdite a regime, efficienza isolamento."
        )
        losses_desc.setStyleSheet("color: #666; font-size: 9px;")
        type_layout.addWidget(losses_desc)
        
        # Transitoria
        self.radio_transient = QRadioButton("Transitoria (dinamica)")
        self.btn_group.addButton(self.radio_transient)
        type_layout.addWidget(self.radio_transient)
        
        trans_desc = QLabel(
            "   Simula l'evoluzione temporale: carica, scarica, cicli.\n"
            "   Utile per: analisi operativa, ottimizzazione cicli."
        )
        trans_desc.setStyleSheet("color: #666; font-size: 9px;")
        type_layout.addWidget(trans_desc)
        
        type_group.setLayout(type_layout)
        layout.addWidget(type_group)
        
        # Connetti segnali
        self.btn_group.buttonClicked.connect(self._on_type_changed)
        
        # Parametri specifici per tipo
        self._create_steady_params(layout)
        self._create_losses_params(layout)
        self._create_transient_params(layout)
        
        # Mostra solo parametri stazionaria inizialmente
        self._update_visibility()
        
        layout.addStretch()
    
    def _create_steady_params(self, parent_layout):
        """Crea parametri per analisi stazionaria"""
        self.steady_group = QGroupBox("Parametri Stazionaria")
        layout = QGridLayout()
        
        row = 0
        layout.addWidget(QLabel("Potenza totale resistenze:"), row, 0)
        self.steady_power_spin = QDoubleSpinBox()
        self.steady_power_spin.setRange(0, 1000000)
        self.steady_power_spin.setValue(10000)
        self.steady_power_spin.setSuffix(" W")
        self.steady_power_spin.setSingleStep(1000)
        layout.addWidget(self.steady_power_spin, row, 1)
        
        self.steady_group.setLayout(layout)
        parent_layout.addWidget(self.steady_group)
    
    def _create_losses_params(self, parent_layout):
        """Crea parametri per analisi perdite"""
        self.losses_group = QGroupBox("Parametri Analisi Perdite")
        layout = QGridLayout()
        
        row = 0
        layout.addWidget(QLabel("Temperatura media storage:"), row, 0)
        self.losses_T_spin = QDoubleSpinBox()
        self.losses_T_spin.setRange(20, 800)
        self.losses_T_spin.setValue(400)
        self.losses_T_spin.setSuffix(" °C")
        layout.addWidget(self.losses_T_spin, row, 1)
        
        row += 1
        layout.addWidget(QLabel("Temperatura ambiente:"), row, 0)
        self.losses_T_amb_spin = QDoubleSpinBox()
        self.losses_T_amb_spin.setRange(-40, 50)
        self.losses_T_amb_spin.setValue(20)
        self.losses_T_amb_spin.setSuffix(" °C")
        layout.addWidget(self.losses_T_amb_spin, row, 1)
        
        self.losses_group.setLayout(layout)
        parent_layout.addWidget(self.losses_group)
    
    def _create_transient_params(self, parent_layout):
        """Crea parametri per analisi transitoria"""
        self.transient_group = QGroupBox("Parametri Transitoria")
        layout = QGridLayout()
        
        row = 0
        layout.addWidget(QLabel("Durata simulazione:"), row, 0)
        self.trans_duration_spin = QDoubleSpinBox()
        self.trans_duration_spin.setRange(60, 86400*30)  # 1 min - 30 giorni
        self.trans_duration_spin.setValue(3600)
        self.trans_duration_spin.setSuffix(" s")
        self.trans_duration_spin.setSingleStep(3600)
        layout.addWidget(self.trans_duration_spin, row, 1)
        
        # Combo per unità tempo
        self.trans_duration_unit = QComboBox()
        self.trans_duration_unit.addItems(["secondi", "minuti", "ore", "giorni"])
        self.trans_duration_unit.setCurrentIndex(2)  # ore
        self.trans_duration_unit.currentIndexChanged.connect(self._on_duration_unit_changed)
        layout.addWidget(self.trans_duration_unit, row, 2)
        
        row += 1
        layout.addWidget(QLabel("Passo temporale (dt):"), row, 0)
        self.trans_dt_spin = QDoubleSpinBox()
        self.trans_dt_spin.setRange(0.1, 3600)
        self.trans_dt_spin.setValue(60)
        self.trans_dt_spin.setSuffix(" s")
        layout.addWidget(self.trans_dt_spin, row, 1)
        
        row += 1
        layout.addWidget(QLabel("Intervallo salvataggio:"), row, 0)
        self.trans_save_spin = QDoubleSpinBox()
        self.trans_save_spin.setRange(1, 3600)
        self.trans_save_spin.setValue(60)
        self.trans_save_spin.setSuffix(" s")
        layout.addWidget(self.trans_save_spin, row, 1)
        
        row += 1
        self.trans_save_field_check = QCheckBox("Salva campo T completo (usa più memoria)")
        layout.addWidget(self.trans_save_field_check, row, 0, 1, 2)
        
        self.transient_group.setLayout(layout)
        parent_layout.addWidget(self.transient_group)
    
    def _on_type_changed(self, button):
        self._update_visibility()
        
        if button == self.radio_steady:
            self.analysis_changed.emit("steady")
        elif button == self.radio_losses:
            self.analysis_changed.emit("losses")
        else:
            self.analysis_changed.emit("transient")
    
    def _on_duration_unit_changed(self, index):
        """Aggiorna il suffisso quando cambia l'unità"""
        units = ["s", "min", "h", "giorni"]
        multipliers = [1, 60, 3600, 86400]
        # Potremmo convertire il valore qui se necessario
    
    def _update_visibility(self):
        """Mostra/nasconde i gruppi parametri in base al tipo selezionato"""
        self.steady_group.setVisible(self.radio_steady.isChecked())
        self.losses_group.setVisible(self.radio_losses.isChecked())
        self.transient_group.setVisible(self.radio_transient.isChecked())
    
    def get_analysis_type(self) -> str:
        if self.radio_steady.isChecked():
            return "steady"
        elif self.radio_losses.isChecked():
            return "losses"
        else:
            return "transient"
    
    def get_transient_config(self) -> TransientConfig:
        """Restituisce la configurazione transitoria"""
        # Converti durata in secondi
        unit_idx = self.trans_duration_unit.currentIndex()
        multipliers = [1, 60, 3600, 86400]
        t_final = self.trans_duration_spin.value() * multipliers[unit_idx]
        
        return TransientConfig(
            t_final=t_final,
            dt=self.trans_dt_spin.value(),
            save_interval=self.trans_save_spin.value(),
            save_full_field=self.trans_save_field_check.isChecked()
        )


class InitialConditionWidget(QWidget):
    """Widget per la configurazione delle condizioni iniziali"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Gruppo selezione modalità
        mode_group = QGroupBox("Condizione Iniziale")
        mode_layout = QVBoxLayout()
        
        self.btn_group = QButtonGroup(self)
        
        # T uniforme nella matrice
        self.radio_uniform = QRadioButton("Temperatura uniforme nella matrice")
        self.radio_uniform.setChecked(True)
        self.btn_group.addButton(self.radio_uniform)
        mode_layout.addWidget(self.radio_uniform)
        
        uniform_layout = QHBoxLayout()
        uniform_layout.addSpacing(20)
        uniform_layout.addWidget(QLabel("T media:"))
        self.T_uniform_spin = QDoubleSpinBox()
        self.T_uniform_spin.setRange(-40, 800)
        self.T_uniform_spin.setValue(20)
        self.T_uniform_spin.setSuffix(" °C")
        uniform_layout.addWidget(self.T_uniform_spin)
        uniform_layout.addStretch()
        mode_layout.addLayout(uniform_layout)
        
        # T per materiale
        self.radio_by_material = QRadioButton("Temperatura per materiale")
        self.btn_group.addButton(self.radio_by_material)
        mode_layout.addWidget(self.radio_by_material)
        
        self.material_temps_widget = self._create_material_temps_widget()
        mode_layout.addWidget(self.material_temps_widget)
        
        # Da file
        self.radio_from_file = QRadioButton("Carica da file salvato")
        self.btn_group.addButton(self.radio_from_file)
        mode_layout.addWidget(self.radio_from_file)
        
        file_layout = QHBoxLayout()
        file_layout.addSpacing(20)
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Seleziona file .h5...")
        self.file_path_edit.setReadOnly(True)
        file_layout.addWidget(self.file_path_edit)
        self.browse_btn = QPushButton("Sfoglia...")
        self.browse_btn.clicked.connect(self._browse_file)
        file_layout.addWidget(self.browse_btn)
        mode_layout.addLayout(file_layout)
        
        # Da stazionario
        self.radio_from_steady = QRadioButton("Calcola da stazionario")
        self.btn_group.addButton(self.radio_from_steady)
        mode_layout.addWidget(self.radio_from_steady)
        
        steady_desc = QLabel("   Prima calcola lo stazionario, poi usa come condizione iniziale")
        steady_desc.setStyleSheet("color: #666; font-size: 9px;")
        mode_layout.addWidget(steady_desc)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Connetti segnali
        self.btn_group.buttonClicked.connect(self._update_visibility)
        self._update_visibility()
        
        layout.addStretch()
    
    def _create_material_temps_widget(self) -> QWidget:
        """Crea widget per temperature per materiale"""
        widget = QWidget()
        layout = QGridLayout(widget)
        layout.setContentsMargins(20, 5, 5, 5)
        
        materials = [
            ("Sabbia (storage):", "T_sand", 20),
            ("Isolamento:", "T_insulation", 20),
            ("Acciaio shell:", "T_steel", 20),
            ("Aria:", "T_air", 20),
            ("Terreno:", "T_ground", 15),
            ("Calcestruzzo:", "T_concrete", 20),
        ]
        
        self.material_spins = {}
        
        for row, (label, key, default) in enumerate(materials):
            layout.addWidget(QLabel(label), row, 0)
            spin = QDoubleSpinBox()
            spin.setRange(-40, 800)
            spin.setValue(default)
            spin.setSuffix(" °C")
            layout.addWidget(spin, row, 1)
            self.material_spins[key] = spin
        
        return widget
    
    def _browse_file(self):
        """Apre dialogo per selezionare file stato"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Carica Stato", "", 
            "File stato (*.h5);;Tutti i file (*.*)"
        )
        if filepath:
            self.file_path_edit.setText(filepath)
    
    def _update_visibility(self):
        """Aggiorna visibilità widget in base alla selezione"""
        self.T_uniform_spin.setEnabled(self.radio_uniform.isChecked())
        self.material_temps_widget.setEnabled(self.radio_by_material.isChecked())
        self.file_path_edit.setEnabled(self.radio_from_file.isChecked())
        self.browse_btn.setEnabled(self.radio_from_file.isChecked())
    
    def get_initial_condition(self) -> InitialCondition:
        """Restituisce la condizione iniziale configurata"""
        ic = InitialCondition()
        
        if self.radio_uniform.isChecked():
            ic.mode = "uniform"
            ic.T_uniform = self.T_uniform_spin.value()
        
        elif self.radio_by_material.isChecked():
            ic.mode = "by_material"
            ic.T_sand = self.material_spins["T_sand"].value()
            ic.T_insulation = self.material_spins["T_insulation"].value()
            ic.T_steel = self.material_spins["T_steel"].value()
            ic.T_air = self.material_spins["T_air"].value()
            ic.T_ground = self.material_spins["T_ground"].value()
            ic.T_concrete = self.material_spins["T_concrete"].value()
        
        elif self.radio_from_file.isChecked():
            ic.mode = "from_file"
            ic.file_path = self.file_path_edit.text()
        
        elif self.radio_from_steady.isChecked():
            ic.mode = "from_steady"
        
        return ic


class PowerProfileWidget(QWidget):
    """Widget per la configurazione del profilo potenza resistenze"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Modalità potenza
        mode_group = QGroupBox("Potenza Resistenze")
        mode_layout = QVBoxLayout()
        
        self.btn_group = QButtonGroup(self)
        
        # Spente
        self.radio_off = QRadioButton("Spente")
        self.btn_group.addButton(self.radio_off)
        mode_layout.addWidget(self.radio_off)
        
        # Costante
        self.radio_constant = QRadioButton("Potenza costante")
        self.radio_constant.setChecked(True)
        self.btn_group.addButton(self.radio_constant)
        mode_layout.addWidget(self.radio_constant)
        
        const_layout = QHBoxLayout()
        const_layout.addSpacing(20)
        const_layout.addWidget(QLabel("Potenza:"))
        self.power_spin = QDoubleSpinBox()
        self.power_spin.setRange(0, 1000000)
        self.power_spin.setValue(10000)
        self.power_spin.setSuffix(" W")
        self.power_spin.setSingleStep(1000)
        const_layout.addWidget(self.power_spin)
        
        self.power_unit_combo = QComboBox()
        self.power_unit_combo.addItems(["W", "kW", "MW"])
        self.power_unit_combo.setCurrentIndex(1)
        const_layout.addWidget(self.power_unit_combo)
        const_layout.addStretch()
        mode_layout.addLayout(const_layout)
        
        # Schedulata
        self.radio_schedule = QRadioButton("Profilo schedulato")
        self.btn_group.addButton(self.radio_schedule)
        mode_layout.addWidget(self.radio_schedule)
        
        # Tabella schedule
        self.schedule_table = QTableWidget(5, 2)
        self.schedule_table.setHorizontalHeaderLabels(["Tempo [s]", "Potenza [W]"])
        self.schedule_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.schedule_table.setMaximumHeight(150)
        mode_layout.addWidget(self.schedule_table)
        
        schedule_btn_layout = QHBoxLayout()
        self.add_row_btn = QPushButton("+ Aggiungi riga")
        self.add_row_btn.clicked.connect(lambda: self.schedule_table.insertRow(self.schedule_table.rowCount()))
        schedule_btn_layout.addWidget(self.add_row_btn)
        self.remove_row_btn = QPushButton("- Rimuovi riga")
        self.remove_row_btn.clicked.connect(lambda: self.schedule_table.removeRow(self.schedule_table.currentRow()))
        schedule_btn_layout.addWidget(self.remove_row_btn)
        schedule_btn_layout.addStretch()
        mode_layout.addLayout(schedule_btn_layout)
        
        # Da CSV
        self.radio_csv = QRadioButton("Da file CSV")
        self.btn_group.addButton(self.radio_csv)
        mode_layout.addWidget(self.radio_csv)
        
        csv_layout = QHBoxLayout()
        csv_layout.addSpacing(20)
        self.csv_path_edit = QLineEdit()
        self.csv_path_edit.setPlaceholderText("File CSV (colonne: time, power)")
        csv_layout.addWidget(self.csv_path_edit)
        self.csv_browse_btn = QPushButton("Sfoglia...")
        self.csv_browse_btn.clicked.connect(self._browse_csv)
        csv_layout.addWidget(self.csv_browse_btn)
        mode_layout.addLayout(csv_layout)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Connetti segnali
        self.btn_group.buttonClicked.connect(self._update_visibility)
        self._update_visibility()
        
        layout.addStretch()
    
    def _browse_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Carica Profilo Potenza", "", 
            "File CSV (*.csv);;Tutti i file (*.*)"
        )
        if filepath:
            self.csv_path_edit.setText(filepath)
    
    def _update_visibility(self):
        self.power_spin.setEnabled(self.radio_constant.isChecked())
        self.power_unit_combo.setEnabled(self.radio_constant.isChecked())
        self.schedule_table.setEnabled(self.radio_schedule.isChecked())
        self.add_row_btn.setEnabled(self.radio_schedule.isChecked())
        self.remove_row_btn.setEnabled(self.radio_schedule.isChecked())
        self.csv_path_edit.setEnabled(self.radio_csv.isChecked())
        self.csv_browse_btn.setEnabled(self.radio_csv.isChecked())
    
    def get_power_profile(self) -> PowerProfile:
        """Restituisce il profilo potenza configurato"""
        profile = PowerProfile()
        
        if self.radio_off.isChecked():
            profile.mode = "off"
        
        elif self.radio_constant.isChecked():
            profile.mode = "constant"
            power = self.power_spin.value()
            unit_idx = self.power_unit_combo.currentIndex()
            multipliers = [1, 1000, 1000000]
            profile.constant_power = power * multipliers[unit_idx]
        
        elif self.radio_schedule.isChecked():
            profile.mode = "schedule"
            schedule = []
            for row in range(self.schedule_table.rowCount()):
                t_item = self.schedule_table.item(row, 0)
                p_item = self.schedule_table.item(row, 1)
                if t_item and p_item:
                    try:
                        t = float(t_item.text())
                        p = float(p_item.text())
                        schedule.append((t, p))
                    except ValueError:
                        pass
            profile.schedule = sorted(schedule, key=lambda x: x[0])
            profile._build_cache()
        
        elif self.radio_csv.isChecked():
            profile.mode = "csv"
            profile.csv_path = self.csv_path_edit.text()
            profile._build_cache()
        
        return profile


class ExtractionProfileWidget(QWidget):
    """Widget per la configurazione dell'estrazione dai tubi"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Modalità estrazione
        mode_group = QGroupBox("Estrazione Energia (Tubi)")
        mode_layout = QVBoxLayout()
        
        self.btn_group = QButtonGroup(self)
        
        # Off
        self.radio_off = QRadioButton("Nessuna estrazione")
        self.radio_off.setChecked(True)
        self.btn_group.addButton(self.radio_off)
        mode_layout.addWidget(self.radio_off)
        
        # Potenza imposta
        self.radio_power = QRadioButton("Imponi potenza estratta")
        self.btn_group.addButton(self.radio_power)
        mode_layout.addWidget(self.radio_power)
        
        power_layout = QHBoxLayout()
        power_layout.addSpacing(20)
        power_layout.addWidget(QLabel("Potenza:"))
        self.extraction_power_spin = QDoubleSpinBox()
        self.extraction_power_spin.setRange(0, 1000000)
        self.extraction_power_spin.setValue(5000)
        self.extraction_power_spin.setSuffix(" W")
        power_layout.addWidget(self.extraction_power_spin)
        power_layout.addStretch()
        mode_layout.addLayout(power_layout)
        
        # Flow rate imposto
        self.radio_flow = QRadioButton("Imponi flow rate")
        self.btn_group.addButton(self.radio_flow)
        mode_layout.addWidget(self.radio_flow)
        
        flow_layout = QHBoxLayout()
        flow_layout.addSpacing(20)
        flow_layout.addWidget(QLabel("Portata:"))
        self.flow_rate_spin = QDoubleSpinBox()
        self.flow_rate_spin.setRange(0, 100)
        self.flow_rate_spin.setValue(0.1)
        self.flow_rate_spin.setSuffix(" kg/s")
        self.flow_rate_spin.setDecimals(3)
        flow_layout.addWidget(self.flow_rate_spin)
        
        self.flow_per_tube_check = QCheckBox("per tubo")
        self.flow_per_tube_check.setChecked(True)
        flow_layout.addWidget(self.flow_per_tube_check)
        flow_layout.addStretch()
        mode_layout.addLayout(flow_layout)
        
        # T outlet imposta
        self.radio_temp = QRadioButton("Imponi temperatura uscita")
        self.btn_group.addButton(self.radio_temp)
        mode_layout.addWidget(self.radio_temp)
        
        temp_layout = QHBoxLayout()
        temp_layout.addSpacing(20)
        temp_layout.addWidget(QLabel("T uscita:"))
        self.T_outlet_spin = QDoubleSpinBox()
        self.T_outlet_spin.setRange(20, 200)
        self.T_outlet_spin.setValue(80)
        self.T_outlet_spin.setSuffix(" °C")
        temp_layout.addWidget(self.T_outlet_spin)
        temp_layout.addStretch()
        mode_layout.addLayout(temp_layout)
        
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)
        
        # Parametri fluido
        fluid_group = QGroupBox("Fluido")
        fluid_layout = QGridLayout()
        
        fluid_layout.addWidget(QLabel("Tipo fluido:"), 0, 0)
        self.fluid_combo = QComboBox()
        self.fluid_combo.addItems(["Acqua", "Olio termico", "Aria", "Glicole 30%"])
        fluid_layout.addWidget(self.fluid_combo, 0, 1)
        
        fluid_layout.addWidget(QLabel("T ingresso:"), 1, 0)
        self.T_inlet_spin = QDoubleSpinBox()
        self.T_inlet_spin.setRange(0, 100)
        self.T_inlet_spin.setValue(20)
        self.T_inlet_spin.setSuffix(" °C")
        fluid_layout.addWidget(self.T_inlet_spin, 1, 1)
        
        fluid_group.setLayout(fluid_layout)
        layout.addWidget(fluid_group)
        
        # Connetti segnali
        self.btn_group.buttonClicked.connect(self._update_visibility)
        self._update_visibility()
        
        layout.addStretch()
    
    def _update_visibility(self):
        self.extraction_power_spin.setEnabled(self.radio_power.isChecked())
        self.flow_rate_spin.setEnabled(self.radio_flow.isChecked())
        self.flow_per_tube_check.setEnabled(self.radio_flow.isChecked())
        self.T_outlet_spin.setEnabled(self.radio_temp.isChecked())
    
    def get_extraction_profile(self) -> ExtractionProfile:
        """Restituisce il profilo estrazione configurato"""
        profile = ExtractionProfile()
        
        fluid_map = {
            0: "water", 1: "oil", 2: "air", 3: "glycol_30"
        }
        profile.fluid = fluid_map.get(self.fluid_combo.currentIndex(), "water")
        profile.T_inlet = self.T_inlet_spin.value()
        
        if self.radio_off.isChecked():
            profile.mode = "off"
        
        elif self.radio_power.isChecked():
            profile.mode = "power"
            profile.power = self.extraction_power_spin.value()
        
        elif self.radio_flow.isChecked():
            profile.mode = "flow_rate"
            if self.flow_per_tube_check.isChecked():
                profile.flow_rate = self.flow_rate_spin.value()
            else:
                profile.flow_rate_total = self.flow_rate_spin.value()
        
        elif self.radio_temp.isChecked():
            profile.mode = "temperature"
            profile.T_outlet_target = self.T_outlet_spin.value()
        
        return profile


class SaveLoadWidget(QWidget):
    """Widget per salvataggio e caricamento stato"""
    
    state_loaded = pyqtSignal(object)  # Emette SimulationState
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Salvataggio
        save_group = QGroupBox("Salva Stato Corrente")
        save_layout = QVBoxLayout()
        
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Nome:"))
        self.save_name_edit = QLineEdit()
        self.save_name_edit.setPlaceholderText("Nome simulazione...")
        name_layout.addWidget(self.save_name_edit)
        save_layout.addLayout(name_layout)
        
        desc_layout = QHBoxLayout()
        desc_layout.addWidget(QLabel("Descrizione:"))
        self.save_desc_edit = QLineEdit()
        self.save_desc_edit.setPlaceholderText("Descrizione opzionale...")
        desc_layout.addWidget(self.save_desc_edit)
        save_layout.addLayout(desc_layout)
        
        self.save_btn = QPushButton("Salva Stato...")
        self.save_btn.clicked.connect(self._on_save)
        save_layout.addWidget(self.save_btn)
        
        save_group.setLayout(save_layout)
        layout.addWidget(save_group)
        
        # Caricamento
        load_group = QGroupBox("Carica Stato Salvato")
        load_layout = QVBoxLayout()
        
        self.load_btn = QPushButton("Carica Stato...")
        self.load_btn.clicked.connect(self._on_load)
        load_layout.addWidget(self.load_btn)
        
        self.state_info_label = QLabel("Nessuno stato caricato")
        self.state_info_label.setStyleSheet("color: #666; font-size: 9px; padding: 4px;")
        self.state_info_label.setWordWrap(True)
        load_layout.addWidget(self.state_info_label)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        layout.addStretch()
    
    def _on_save(self):
        """Salva lo stato corrente"""
        # Questo sarà chiamato dal main window che ha accesso alla mesh
        pass  # Implementato nel main_window
    
    def _on_load(self):
        """Carica uno stato salvato"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Carica Stato", "",
            "File stato (*.h5);;Tutti i file (*.*)"
        )
        if filepath:
            state = StateManager.load_state(filepath)
            if state:
                self.state_info_label.setText(
                    f"<b>{state.name}</b><br>"
                    f"Tipo: {state.analysis_type}<br>"
                    f"Data: {state.timestamp[:19]}<br>"
                    f"Mesh: {state.mesh_shape}"
                )
                self.state_loaded.emit(state)
            else:
                QMessageBox.warning(self, "Errore", "Impossibile caricare il file!")


class AnalysisTab(QWidget):
    """
    Tab principale per la configurazione dell'analisi.
    
    Contiene sub-tabs per:
    - Tipo Analisi
    - Condizioni Iniziali
    - Potenza Resistenze
    - Estrazione Tubi
    - Salvataggio
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Sub-tabs
        self.sub_tabs = QTabWidget()
        
        # Tab Tipo Analisi
        self.analysis_type_widget = AnalysisTypeWidget()
        self.sub_tabs.addTab(self.analysis_type_widget, "Tipo")
        
        # Tab Condizioni Iniziali
        self.initial_cond_widget = InitialConditionWidget()
        self.sub_tabs.addTab(self.initial_cond_widget, "Condizioni Iniziali")
        
        # Tab Potenza
        self.power_widget = PowerProfileWidget()
        self.sub_tabs.addTab(self.power_widget, "Potenza")
        
        # Tab Estrazione
        self.extraction_widget = ExtractionProfileWidget()
        self.sub_tabs.addTab(self.extraction_widget, "Estrazione")
        
        # Tab Salvataggio
        self.save_load_widget = SaveLoadWidget()
        self.sub_tabs.addTab(self.save_load_widget, "Salvataggio")
        
        layout.addWidget(self.sub_tabs)
    
    def get_analysis_type(self) -> str:
        return self.analysis_type_widget.get_analysis_type()
    
    def get_transient_config(self) -> TransientConfig:
        tc = self.analysis_type_widget.get_transient_config()
        tc.initial_condition = self.initial_cond_widget.get_initial_condition()
        tc.power_profile = self.power_widget.get_power_profile()
        tc.extraction_profile = self.extraction_widget.get_extraction_profile()
        return tc
    
    def get_power_profile(self) -> PowerProfile:
        return self.power_widget.get_power_profile()
    
    def get_initial_condition(self) -> InitialCondition:
        return self.initial_cond_widget.get_initial_condition()
