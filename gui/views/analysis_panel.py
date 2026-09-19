"""Analysis panel: analysis type, initial condition, power and extraction profiles."""
from __future__ import annotations

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QRadioButton,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from src.core.mesh import MaterialID
from src.core.profiles import ExtractionProfile, InitialCondition, PowerProfile
from src.units import c_to_k

from ..safe import safe_slot
from ..widgets import FormPanel, button, check, combo, double_spin, hint

_IC_MATERIALS = (
    ("Sand", int(MaterialID.SAND), 20.0),
    ("Insulation", int(MaterialID.INSULATION), 20.0),
    ("Steel", int(MaterialID.STEEL), 20.0),
    ("Air", int(MaterialID.AIR), 20.0),
    ("Concrete", int(MaterialID.CONCRETE), 20.0),
    ("Tubes", int(MaterialID.TUBES), 20.0),
)
_POWER_UNITS = (("W", 1.0), ("kW", 1e3), ("MW", 1e6))
_DURATION_UNITS = (("seconds", 1.0), ("minutes", 60.0), ("hours", 3600.0), ("days", 86400.0))


class AnalysisPanel(QWidget):
    """Everything that drives the run: type, initial state, power and extraction."""

    analysis_changed = pyqtSignal(str)
    state_loaded = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from PyQt6.QtWidgets import QTabWidget

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        self._build_type_tab()
        self._build_initial_tab()
        self._build_power_tab()
        self._build_extraction_tab()
        self._build_save_tab()
        self._update_visibility()

    # ------------------------------------------------------------------ type
    def _build_type_tab(self) -> None:
        panel = FormPanel()
        self.radio_steady = QRadioButton("Steady state (fixed heater power)")
        self.radio_losses = QRadioButton("Losses analysis (hold a target temperature)")
        self.radio_transient = QRadioButton("Transient (power and extraction profiles)")
        self.type_group = QButtonGroup(self)
        for radio in (self.radio_steady, self.radio_losses, self.radio_transient):
            self.type_group.addButton(radio)
            panel.add_row(radio)
        self.losses_group = QGroupBox("Losses target")
        losses = FormPanel(self.losses_group)
        self.losses_target = losses.add("Mean storage T [°C]",
                                        double_spin(400.0, 20.0, 1200.0, 10.0, 1))
        self.transient_group = QGroupBox("Time stepping")
        transient = FormPanel(self.transient_group)
        self.duration = transient.add("Duration", double_spin(10.0, 0.01, 100000.0, 1.0, 2))
        self.duration_unit = transient.add("Unit", combo(_DURATION_UNITS, 2,
                                                         self._on_duration_unit))
        self.dt = transient.add("Time step dt [s]", double_spin(60.0, 0.1, 86400.0, 10.0, 2))
        self.save_interval = transient.add("Save interval [s]",
                                           double_spin(600.0, 1.0, 86400.0, 60.0, 1))
        self.save_field = transient.add("Full fields", check("save T field per sample", False))
        panel.add_row(self.losses_group)
        panel.add_row(self.transient_group)
        for radio in (self.radio_steady, self.radio_losses, self.radio_transient):
            radio.toggled.connect(self._update_visibility)
        self.radio_steady.setChecked(True)
        panel.add_hint("Ambient and ground conditions come from Materials > Conditions; "
                       "the heater power from the Heaters tab.")
        self.tabs.addTab(panel, "Type")

    @safe_slot
    def _on_duration_unit(self) -> None:
        self.duration.setSuffix(" " + self.duration_unit.currentText())

    @safe_slot
    def _update_visibility(self) -> None:
        state = self.analysis_type()
        self.losses_group.setVisible(state == "losses")
        self.transient_group.setVisible(state == "transient")
        self.analysis_changed.emit(state)

    def analysis_type(self) -> str:
        if self.radio_losses.isChecked():
            return "losses"
        if self.radio_transient.isChecked():
            return "transient"
        return "steady"

    # ------------------------------------------------------- initial condition
    def _build_initial_tab(self) -> None:
        panel = FormPanel()
        self.radio_uniform = QRadioButton("Uniform temperature")
        self.radio_by_material = QRadioButton("Temperature per material")
        self.radio_from_file = QRadioButton("Load a saved HDF5 state")
        self.radio_from_steady = QRadioButton("Start from the steady solution")
        self.ic_group = QButtonGroup(self)
        for radio in (self.radio_uniform, self.radio_by_material, self.radio_from_file,
                      self.radio_from_steady):
            self.ic_group.addButton(radio)
            panel.add_row(radio)
        self.t_uniform = panel.add("Uniform T [°C]", double_spin(20.0, -40.0, 1200.0, 5.0, 1))
        self.material_spins = {}
        for label, material_id, default in _IC_MATERIALS:
            self.material_spins[material_id] = panel.add(f"{label} [°C]",
                                                         double_spin(default, -40.0, 1200.0, 5.0, 1))
        self.file_path = panel.add("State file", QLineEdit())
        self.file_path.setReadOnly(True)
        panel.add_row(button("Browse...", self._browse_state))
        for radio in (self.radio_uniform, self.radio_by_material, self.radio_from_file,
                      self.radio_from_steady):
            radio.toggled.connect(self._update_ic_visibility)
        self.radio_uniform.setChecked(True)
        self._update_ic_visibility()
        self.tabs.addTab(panel, "Initial condition")

    @safe_slot
    def _update_ic_visibility(self) -> None:
        uniform = self.radio_uniform.isChecked()
        by_material = self.radio_by_material.isChecked()
        self.t_uniform.setEnabled(uniform)
        for spin in self.material_spins.values():
            spin.setEnabled(by_material)
        self.file_path.setEnabled(self.radio_from_file.isChecked())

    def _browse_state(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open state", "results/states", "HDF5 (*.h5)")
        if path:
            self.file_path.setText(path)

    def initial_condition(self) -> InitialCondition:
        if self.radio_by_material.isChecked():
            return InitialCondition(
                mode="by_material",
                t_by_material={mid: c_to_k(spin.value())
                               for mid, spin in self.material_spins.items()})
        if self.radio_from_file.isChecked():
            return InitialCondition(mode="from_file", file_path=self.file_path.text())
        if self.radio_from_steady.isChecked():
            # the controller solves the steady case first: keep that field
            return InitialCondition(mode="keep")
        return InitialCondition(mode="uniform", t_uniform=c_to_k(self.t_uniform.value()))

    def wants_steady_initial_condition(self) -> bool:
        return self.radio_from_steady.isChecked()

    # ------------------------------------------------------------- power
    def _build_power_tab(self) -> None:
        panel = FormPanel()
        self.power_off = QRadioButton("Off")
        self.power_constant = QRadioButton("Constant power")
        self.power_schedule = QRadioButton("Scheduled profile")
        self.power_csv = QRadioButton("From CSV (t, P)")
        self.power_group = QButtonGroup(self)
        for radio in (self.power_off, self.power_constant, self.power_schedule, self.power_csv):
            self.power_group.addButton(radio)
            panel.add_row(radio)
        self.power_value = panel.add("Power", double_spin(10000.0, 0.0, 1e6, 1000.0, 0))
        self.power_unit = panel.add("Unit", combo(_POWER_UNITS, 1))
        self.schedule = QTableWidget(3, 2)
        self.schedule.setHorizontalHeaderLabels(["Time [s]", "Power [W]"])
        self.schedule.setMaximumHeight(150)
        panel.add_row(self.schedule)
        buttons = QWidget()
        row = QHBoxLayout(buttons)
        row.addWidget(button("Add row", lambda: self.schedule.insertRow(self.schedule.rowCount())))
        row.addWidget(button("Remove row", lambda: self.schedule.removeRow(self.schedule.currentRow())))
        panel.add_row(buttons)
        self.csv_path = panel.add("CSV file", QLineEdit())
        self.csv_path.setReadOnly(True)
        panel.add_row(button("Browse...", self._browse_csv))
        for radio in (self.power_off, self.power_constant, self.power_schedule, self.power_csv):
            radio.toggled.connect(self._update_power_visibility)
        self.power_constant.setChecked(True)
        self._update_power_visibility()
        self.tabs.addTab(panel, "Power")

    @safe_slot
    def _update_power_visibility(self) -> None:
        self.power_value.setEnabled(self.power_constant.isChecked())
        self.power_unit.setEnabled(self.power_constant.isChecked())
        self.schedule.setEnabled(self.power_schedule.isChecked())
        self.csv_path.setEnabled(self.power_csv.isChecked())

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open power profile", "", "CSV (*.csv)")
        if path:
            self.csv_path.setText(path)

    def power_profile(self) -> PowerProfile:
        if self.power_off.isChecked():
            return PowerProfile(mode="off")
        if self.power_schedule.isChecked():
            times, powers = [], []
            for row in range(self.schedule.rowCount()):
                t_item, p_item = self.schedule.item(row, 0), self.schedule.item(row, 1)
                if t_item and p_item and t_item.text() and p_item.text():
                    times.append(float(t_item.text()))
                    powers.append(float(p_item.text()))
            return PowerProfile(mode="schedule", times=times, powers=powers)
        if self.power_csv.isChecked():
            return PowerProfile(mode="csv", csv_path=self.csv_path.text())
        return PowerProfile(mode="constant",
                            constant_power=self.power_value.value() * self.power_unit.currentData())

    # --------------------------------------------------------- extraction
    def _build_extraction_tab(self) -> None:
        panel = FormPanel()
        self.extract_off = QRadioButton("No extraction")
        self.extract_power = QRadioButton("Target power")
        self.extract_flow = QRadioButton("Fluid flow rate")
        self.extract_group = QButtonGroup(self)
        for radio in (self.extract_off, self.extract_power, self.extract_flow):
            self.extract_group.addButton(radio)
            panel.add_row(radio)
        self.extract_power_value = panel.add("Power [W]", double_spin(5000.0, 0.0, 1e7, 500.0, 0))
        self.extract_flow_value = panel.add("Mass flow [kg/s]",
                                            double_spin(0.1, 0.001, 100.0, 0.05, 3))
        self.extract_t_inlet = panel.add("Inlet T [°C]", double_spin(20.0, -20.0, 400.0, 5.0, 1))
        self.extract_h = panel.add("Tube h [W/(m²·K)]", double_spin(500.0, 10.0, 20000.0, 50.0, 0))
        panel.add_hint("Target power is capped by what the tube surface can deliver "
                       "(h·A·ΔT): the solver never extracts heat from a colder body.")
        for radio in (self.extract_off, self.extract_power, self.extract_flow):
            radio.toggled.connect(self._update_extraction_visibility)
        self.extract_off.setChecked(True)
        self._update_extraction_visibility()
        self.tabs.addTab(panel, "Extraction")

    @safe_slot
    def _update_extraction_visibility(self) -> None:
        self.extract_power_value.setEnabled(self.extract_power.isChecked())
        for widget in (self.extract_flow_value, self.extract_h):
            widget.setEnabled(self.extract_flow.isChecked())
        self.extract_t_inlet.setEnabled(not self.extract_off.isChecked())

    def extraction_profile(self) -> ExtractionProfile:
        if self.extract_power.isChecked():
            return ExtractionProfile(mode="power", power=self.extract_power_value.value(),
                                     t_inlet=c_to_k(self.extract_t_inlet.value()),
                                     h_fluid=self.extract_h.value())
        if self.extract_flow.isChecked():
            return ExtractionProfile(mode="flow_rate",
                                     mass_flow=self.extract_flow_value.value(),
                                     t_inlet=c_to_k(self.extract_t_inlet.value()),
                                     h_fluid=self.extract_h.value())
        return ExtractionProfile(mode="off", t_inlet=c_to_k(self.extract_t_inlet.value()))

    # ------------------------------------------------------------ save/load
    def _build_save_tab(self) -> None:
        panel = FormPanel()
        self.state_name = panel.add("Name", QLineEdit("Simulation"))
        self.state_description = panel.add("Description", QLineEdit())
        panel.add_row(button("Save state (HDF5)", self._save_clicked))
        panel.add_row(button("Load state (HDF5)", self._load_clicked))
        self.state_info = panel.add("State", hint("no state loaded"))
        panel.add_hint("A saved state records the geometry hash: loading it into a "
                       "different model is refused.")
        self.tabs.addTab(panel, "Save / Load")

    save_requested = pyqtSignal(str, str)
    load_requested = pyqtSignal()

    def _save_clicked(self) -> None:
        self.save_requested.emit(self.state_name.text() or "Simulation",
                                 self.state_description.text())

    def _load_clicked(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load state", "results/states", "HDF5 (*.h5)")
        if not path:
            return
        self.file_path.setText(path)
        self.load_requested.emit()

    # ---------------------------------------------------------------- values
    def transient_settings(self) -> dict:
        return {
            "t_final": self.duration.value() * self.duration_unit.currentData(),
            "dt": self.dt.value(),
            "save_interval": self.save_interval.value(),
            "save_full_field": self.save_field.isChecked(),
        }

    def losses_target_kelvin(self) -> float:
        return c_to_k(self.losses_target.value())
