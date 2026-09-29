"""Analysis panel: analysis type, initial condition, power and extraction profiles."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
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
from ..widgets import FormPanel, button, check, combo, double_spin, hint, scrollable

_IC_MATERIALS = (
    ("Sand", int(MaterialID.SAND), 20.0),
    ("Insulation", int(MaterialID.INSULATION), 20.0),
    ("Steel", int(MaterialID.STEEL), 20.0),
    ("Concrete", int(MaterialID.CONCRETE), 20.0),
    ("Pipes", int(MaterialID.TUBES), 20.0),
)
_DURATION_UNITS = (("seconds", 1.0), ("minutes", 60.0), ("hours", 3600.0), ("days", 86400.0))


class AnalysisPanel(QWidget):
    """Everything that drives the run: type, initial state, power and extraction."""

    analysis_changed = Signal(str)
    state_loaded = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        from PySide6.QtWidgets import QTabWidget

        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
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
        self.radio_standby = QRadioButton("Steady standby (hold the storage at a temperature)")
        self.radio_transient = QRadioButton("Transient (charge / discharge profiles)")
        self.type_group = QButtonGroup(self)
        for radio in (self.radio_standby, self.radio_transient):
            self.type_group.addButton(radio)
            panel.add_row(radio)
        panel.add_hint("Steady standby: the only steady state a storage has is the one "
                       "where the power in equals the losses, so it is asked as a "
                       "temperature - the bed held at T - and the answer is the power "
                       "that holds it, i.e. the standby losses, with the field that goes "
                       "with them.  (A steady state at a charging power would sit at "
                       "whatever temperature makes the losses that large: thousands of "
                       "degrees for a real plant.)  Transient: the resistors (Charge) and "
                       "the exchanger (Discharge) drive the gas loop.")
        losses = panel.section("Standby")
        self.losses_group = losses.parentWidget()
        self.losses_target = losses.add("Mean storage T [°C]",
                                        double_spin(500.0, 20.0, 1200.0, 10.0, 1))
        transient = panel.section("Time stepping")
        self.transient_group = transient.parentWidget()
        self.duration = transient.add("Duration", double_spin(24.0, 0.01, 100000.0, 1.0, 2))
        self.duration_unit = transient.add("Unit", combo(_DURATION_UNITS, 2,
                                                         self._on_duration_unit))
        self.dt = transient.add("Time step dt [s]", double_spin(
            900.0, 0.1, 86400.0, 60.0, 1,
            tooltip="An accuracy choice, not a stability limit: the gas-bed coupling is "
                    "implicit.  900 s is within minutes of the converged discharge time "
                    "on the test silo; halve it to see whether an answer moves"))
        self.save_interval = transient.add("Save interval [s]",
                                           double_spin(3600.0, 1.0, 86400.0, 60.0, 1))
        self.save_field = transient.add("Full fields", check("save T field per sample", False))
        for radio in (self.radio_standby, self.radio_transient):
            radio.toggled.connect(self._update_visibility)
        self.radio_standby.setChecked(True)
        self.tabs.addTab(scrollable(panel), "Type")

    @safe_slot
    def _on_duration_unit(self) -> None:
        self.duration.setSuffix(" " + self.duration_unit.currentText())

    @safe_slot
    def _update_visibility(self) -> None:
        state = self.analysis_type()
        self.losses_group.setVisible(state == "standby")
        self.transient_group.setVisible(state == "transient")
        self.analysis_changed.emit(state)

    def analysis_type(self) -> str:
        return "transient" if self.radio_transient.isChecked() else "standby"

    # ------------------------------------------------------- initial condition
    def _build_initial_tab(self) -> None:
        panel = FormPanel()
        self.radio_uniform = QRadioButton("Uniform temperature")
        self.radio_by_material = QRadioButton("Temperature per material")
        self.radio_current = QRadioButton("Current field (last result or loaded state)")
        self.radio_from_steady = QRadioButton("Start from the standby state (Type > Standby T)")
        self.ic_group = QButtonGroup(self)
        for radio in (self.radio_uniform, self.radio_by_material, self.radio_current,
                      self.radio_from_steady):
            self.ic_group.addButton(radio)
            panel.add_row(radio)
        self.t_uniform = panel.add("Uniform T [°C]", double_spin(20.0, -40.0, 1200.0, 5.0, 1))
        self.material_spins = {}
        for label, material_id, default in _IC_MATERIALS:
            self.material_spins[material_id] = panel.add(f"{label} [°C]",
                                                         double_spin(default, -40.0, 1200.0, 5.0, 1))
        panel.add_hint("Current field: the transient starts from what the mesh holds - "
                       "the field of the last run, or a state loaded from Save / Load.")
        for radio in (self.radio_uniform, self.radio_by_material, self.radio_current,
                      self.radio_from_steady):
            radio.toggled.connect(self._update_ic_visibility)
        self.radio_uniform.setChecked(True)
        self._update_ic_visibility()
        self.tabs.addTab(scrollable(panel), "Initial")

    @safe_slot
    def _update_ic_visibility(self) -> None:
        uniform = self.radio_uniform.isChecked()
        by_material = self.radio_by_material.isChecked()
        self.t_uniform.setEnabled(uniform)
        for spin in self.material_spins.values():
            spin.setEnabled(by_material)

    def initial_condition(self) -> InitialCondition:
        if self.radio_by_material.isChecked():
            return InitialCondition(
                mode="by_material",
                t_by_material={mid: c_to_k(spin.value())
                               for mid, spin in self.material_spins.items()})
        if self.radio_current.isChecked() or self.radio_from_steady.isChecked():
            # the field the mesh carries (a loaded state, the last run), or the one the
            # controller's steady pre-run leaves in it
            return InitialCondition(mode="keep")
        return InitialCondition(mode="uniform", t_uniform=c_to_k(self.t_uniform.value()))

    def wants_steady_initial_condition(self) -> bool:
        return self.radio_from_steady.isChecked()

    # ------------------------------------------------------------- power
    def _build_power_tab(self) -> None:
        panel = FormPanel()
        self.power_off = QRadioButton("Off")
        self.power_constant = QRadioButton("Constant (the rated power)")
        self.power_constant.setToolTip("The rated power of Plant > Gas circuit")
        self.power_schedule = QRadioButton("Scheduled profile")
        self.power_csv = QRadioButton("From CSV (t, P)")
        self.power_group = QButtonGroup(self)
        for radio in (self.power_off, self.power_constant, self.power_schedule, self.power_csv):
            self.power_group.addButton(radio)
            panel.add_row(radio)
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
        self.tabs.addTab(scrollable(panel), "Charge")

    @safe_slot
    def _update_power_visibility(self) -> None:
        self.schedule.setEnabled(self.power_schedule.isChecked())
        self.csv_path.setEnabled(self.power_csv.isChecked())

    def _browse_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open power profile", "", "CSV (*.csv)")
        if path:
            self.csv_path.setText(path)

    def power_profile(self, rated_power: float = 0.0) -> PowerProfile:
        """The resistors' power over time [W]; ``rated_power`` is the circuit's rating."""
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
        return PowerProfile(mode="constant", constant_power=float(rated_power))

    # --------------------------------------------------------- extraction
    def _build_extraction_tab(self) -> None:
        """The exchanger on the gas circuit: what the discharge takes out of the gas."""
        panel = FormPanel()
        self.extract_off = QRadioButton("No extraction")
        self.extract_power = QRadioButton("Exchanger power")
        self.extract_return = QRadioButton("Exchanger return temperature")
        self.extract_group = QButtonGroup(self)
        for radio in (self.extract_off, self.extract_power, self.extract_return):
            self.extract_group.addButton(radio)
            panel.add_row(radio)
        self.extract_power_value = panel.add("Power [kW]", double_spin(
            100.0, 0.0, 100000.0, 10.0, 1,
            tooltip="Heat the exchanger takes out of the gas: the loop solves the "
                    "temperature the gas re-enters the bed at, and the run stops when "
                    "the bed can no longer give that power"))
        self.extract_return_t = panel.add("Return T [°C]", double_spin(
            60.0, -20.0, 400.0, 5.0, 1,
            tooltip="Temperature the exchanger returns the gas at (40-70 degC in the "
                    "reference plants).  Return-temperature mode: the power is whatever "
                    "the bed gives the gas.  Power mode: the coldest the gas may come "
                    "back - a power the bed can only give with colder gas ends the run"))
        panel.add_hint("The exchanger sits on the gas circuit of Geometry > Gas circuit: "
                       "the power profile heats the gas, the extraction cools it, and the "
                       "loop carries the difference through the pipe walls.")
        for radio in (self.extract_off, self.extract_power, self.extract_return):
            radio.toggled.connect(self._update_extraction_visibility)
        self.extract_off.setChecked(True)
        self._update_extraction_visibility()
        self.tabs.addTab(scrollable(panel), "Discharge")

    @safe_slot
    def _update_extraction_visibility(self) -> None:
        self.extract_power_value.setEnabled(self.extract_power.isChecked())
        self.extract_return_t.setEnabled(not self.extract_off.isChecked())

    def extraction_profile(self) -> ExtractionProfile:
        t_return = c_to_k(self.extract_return_t.value())
        if self.extract_power.isChecked():
            # the exchanger cannot return the gas colder than its return temperature: a
            # power the bed can only give with colder gas ends the discharge
            return ExtractionProfile(mode="power",
                                     power=self.extract_power_value.value() * 1000.0,
                                     t_inlet=t_return, t_return_min=t_return)
        if self.extract_return.isChecked():
            return ExtractionProfile(mode="return_temperature", t_inlet=t_return)
        return ExtractionProfile(mode="off", t_inlet=t_return)

    # ------------------------------------------------------------ save/load
    def _build_save_tab(self) -> None:
        panel = FormPanel()
        self.state_name = panel.add("Name", QLineEdit("Simulation"))
        self.state_description = panel.add("Description", QLineEdit())
        panel.add_row(button("Save state (HDF5)", self._save_clicked))
        panel.add_row(button("Load state (HDF5)", self._load_clicked))
        self.file_path = panel.add("State file", QLineEdit())
        self.file_path.setReadOnly(True)
        self.state_info = panel.add("State", hint("no state loaded"))
        panel.add_hint("A saved state records the geometry hash: loading it into a "
                       "different model is refused.  A loaded field is the start of the "
                       "next transient with Initial condition > Current field.")
        self.tabs.addTab(scrollable(panel), "Save / Load")

    save_requested = Signal(str, str)
    load_requested = Signal()

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
