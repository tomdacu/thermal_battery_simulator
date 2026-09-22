"""Materials and operating-condition panel."""
from __future__ import annotations


from PyQt6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from src.core.materials import INSULATION_MATERIALS, MATERIALS, STORAGE_MATERIALS
from src.units import c_to_k

from ..widgets import FormPanel, combo, double_spin, hint, int_spin

_STORAGE = tuple((props.name, key) for key, props in STORAGE_MATERIALS.items())
_INSULATION = tuple((props.name, key) for key, props in INSULATION_MATERIALS.items())
_STRUCTURAL = tuple((props.name, key) for key, props in MATERIALS.items()
                    if key in ("carbon_steel", "stainless_steel", "concrete"))


def _index_of(items, key: str) -> int:
    """Index of an entry by its key (default of a combo must not depend on dict order)."""
    for index, (_, value) in enumerate(items):
        if value == key:
            return index
    return 0


class MaterialsPanel(QWidget):
    """Medium selection plus the single source of truth for the ambient conditions."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        storage = FormPanel()
        self.storage_material = storage.add("Storage medium",
                                            combo(_STORAGE, _index_of(_STORAGE, "steatite")))
        self.packing = storage.add("Packing [%]", int_spin(63, 20, 90, 1))
        self.storage_info = storage.add("Properties", hint("-"))
        self.tabs.addTab(storage, "Storage")

        insulation = FormPanel()
        self.insulation_material = insulation.add("Insulation", combo(_INSULATION))
        self.shell_material = insulation.add("Shell", combo(_STRUCTURAL))
        self.insulation_info = insulation.add("Properties", hint("-"))
        self.tabs.addTab(insulation, "Insulation")

        conditions = FormPanel()
        self.t_ambient = conditions.add("Ambient [°C]", double_spin(20.0, -40.0, 80.0, 1.0, 1))
        self.t_ground = conditions.add("Ground [°C]", double_spin(10.0, -20.0, 60.0, 1.0, 1))
        self.wind = conditions.add("Wind speed [m/s]", double_spin(
            0.0, 0.0, 30.0, 0.5, 1,
            tooltip="Forced part of the outer film, 4 + 4 v (ISO 6946), added to the "
                    "natural convection of the vessel (Churchill-Chu)"))
        conditions.add_hint("These are the only ambient/ground values used: they feed "
                            "BatteryGeometry and every analysis.  The outer surface of the "
                            "vessel exchanges with the ambient through a film computed "
                            "from the correlations (the log reports it after a build); "
                            "the radiative share is switched on in Tools > Solver.")
        self.tabs.addTab(conditions, "Conditions")
        self.refresh_info()

    def refresh_info(self) -> None:
        storage = MATERIALS[self.storage_material.currentData()]
        insulation = MATERIALS[self.insulation_material.currentData()]
        self.storage_info.setText(
            f"k {storage.k:.3f} W/(m·K) · rho {storage.rho:.0f} kg/m³ · "
            f"cp {storage.cp:.0f} J/(kg·K) · T_max {storage.t_max - 273.15:.0f} °C")
        self.insulation_info.setText(
            f"k {insulation.k:.3f} W/(m·K) · rho {insulation.rho:.0f} kg/m³ · "
            f"cp {insulation.cp:.0f} J/(kg·K) · T_max {insulation.t_max - 273.15:.0f} °C")

    # -------------------------------------------------------------- accessors
    def storage_key(self) -> str:
        return self.storage_material.currentData()

    def insulation_key(self) -> str:
        return self.insulation_material.currentData()

    def shell_key(self) -> str:
        return self.shell_material.currentData()

    def packing_fraction(self) -> float:
        return self.packing.value() / 100.0

    def conditions(self) -> dict[str, float]:
        return {
            "t_ambient": c_to_k(self.t_ambient.value()),
            "t_ground": c_to_k(self.t_ground.value()),
            "wind_speed": self.wind.value(),
        }
