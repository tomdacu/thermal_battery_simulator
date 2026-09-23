"""Materials of the vessel's layers, and the site the vessel stands in.

Two pages: the **materials** of the storage bed, the insulation and the shell - laid out
by the window on the Vessel page, next to the shape they fill - and the **site**: the
ambient air, the ground and the wind the outer surface exchanges with.  The site is not
a material: it is the one place the ambient values are defined, and every analysis
reads them from here.
"""
from __future__ import annotations

from PyQt6.QtWidgets import QWidget

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
    """Layer materials plus the single source of truth for the ambient conditions."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.materials_page = FormPanel()
        bed = self.materials_page.section(
            "Materials",
            "The bed is a packed bed of the medium: its conductivity, density and heat "
            "capacity are those of the grains at the packing fraction, with the voids "
            "(compute_packed_bed_properties).  The insulation and the shell are solid.")
        self.storage_material = bed.add("Storage medium", combo(
            _STORAGE, _index_of(_STORAGE, "steatite"), self.refresh_info))
        self.packing = bed.add("Packing [%]", int_spin(
            63, 20, 90, 1, on_change=self.refresh_info,
            tooltip="Solid fraction of the bed: 60-65 % for poured sand or crushed rock"))
        self.storage_info = bed.add("Bed", hint("-"))
        self.insulation_material = bed.add("Insulation", combo(
            _INSULATION, 0, self.refresh_info))
        self.shell_material = bed.add("Shell", combo(_STRUCTURAL))
        self.insulation_info = bed.add("Insulation k", hint("-"))

        self.site_page = FormPanel()
        site = self.site_page.section(
            "Site",
            "The only ambient values the model uses.  The outer surface exchanges with the "
            "air through a film computed from the correlations - natural convection on the "
            "vessel plus the wind (ISO 6946, 4 + 4 v) - which the log reports after every "
            "build; the ground is held at its temperature under the foundation.")
        self.t_ambient = site.add("Ambient [°C]", double_spin(20.0, -40.0, 80.0, 1.0, 1))
        self.t_ground = site.add("Ground [°C]", double_spin(
            10.0, -20.0, 60.0, 1.0, 1,
            tooltip="Held under the foundation: the ground below a plant is close to the "
                    "annual mean air temperature"))
        self.wind = site.add("Wind speed [m/s]", double_spin(
            0.0, 0.0, 30.0, 0.5, 1,
            tooltip="Forced part of the outer film, 4 + 4 v (ISO 6946), added to the "
                    "natural convection of the vessel (Churchill-Chu)"))
        self.refresh_info()

    def refresh_info(self, *_args) -> None:
        from src.core.materials import MaterialManager

        bed = MaterialManager().compute_packed_bed_properties(
            self.storage_material.currentData(), self.packing_fraction())
        insulation = MATERIALS[self.insulation_material.currentData()]
        self.storage_info.setText(
            f"k {bed.k:.3f} W/(m·K) · rho {bed.rho:.0f} kg/m³ · cp {bed.cp:.0f} J/(kg·K)")
        self.insulation_info.setText(
            f"{insulation.k:.3f} W/(m·K) · up to {insulation.t_max - 273.15:.0f} °C")

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
