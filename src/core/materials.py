"""Material database and porous-media models.  SI units, temperatures in KELVIN.

Public API (frozen - the rest of the package is written against it)::

    MATERIALS: dict[str, ThermalProperties]
    MaterialManager.get(name) / list_materials(category) / add_custom_material(props)
    MaterialManager.compute_packed_bed_properties(solid, packing_fraction)
    MaterialManager.compute_effective_properties(solid, porosity, fluid=None)
    MaterialManager.get_energy_density(name, t_high, t_low, packing_fraction=1.0)
    MaterialManager.compare_materials(names, t_high, t_low, packing_fraction=1.0)

Packed bed (the model actually used for the storage region): geometric mean for
the conductivity, arithmetic mean for the density, mass-weighted mean for the
specific heat::

    k_eff = k_solid^(1-phi) * k_fluid^phi
    rho_eff = (1-phi) rho_solid + phi rho_fluid
    cp_eff = [(1-phi) rho_s rho... ] / rho_eff

Properties are constant with temperature: the previous temperature-dependent
hooks (``k_func``/``cp_func``) were never called by any solver, so they were
removed instead of pretending the model is temperature aware.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..constants import PACKING_FRACTION_DEFAULT
from ..units import c_to_k


@dataclass(frozen=True)
class ThermalProperties:
    """Constant thermal properties of a material."""

    name: str
    k: float                    # [W/(m*K)]
    rho: float                  # [kg/m^3]
    cp: float                   # [J/(kg*K)]
    t_max: float                # [K] maximum service temperature
    emissivity: float = 0.9     # [-] total hemispherical emissivity
    description: str = ""


def _p(name: str, k: float, rho: float, cp: float, t_max_c: float, eps: float = 0.9,
       description: str = "") -> ThermalProperties:
    return ThermalProperties(name, k, rho, cp, c_to_k(t_max_c), eps, description)


STORAGE_MATERIALS: dict[str, ThermalProperties] = {
    "silica_sand": _p("Silica sand", 0.35, 1500, 800, 1200, 0.9, "cheap bulk medium"),
    "olivine": _p("Olivine", 3.5, 3300, 900, 1400, 0.85, "high conductivity sand"),
    "steatite": _p("Steatite (soapstone)", 3.0, 2700, 980, 1200, 0.9, "reference medium"),
    "basalt": _p("Basalt", 1.7, 2900, 850, 1100, 0.9),
    "magnetite": _p("Magnetite", 4.5, 5150, 650, 600, 0.95, "very dense"),
    "quartzite": _p("Quartzite", 5.0, 2650, 900, 1400, 0.85),
    "granite": _p("Granite", 2.5, 2700, 800, 800, 0.9),
}

INSULATION_MATERIALS: dict[str, ThermalProperties] = {
    "rock_wool": _p("Rock wool", 0.04, 100, 840, 700),
    "glass_wool": _p("Glass wool", 0.035, 30, 840, 400),
    "calcium_silicate": _p("Calcium silicate", 0.07, 200, 840, 1000),
    "ceramic_fiber": _p("Ceramic fibre", 0.12, 130, 1130, 1260),
    "perlite": _p("Expanded perlite", 0.05, 100, 900, 900),
}

STRUCTURAL_MATERIALS: dict[str, ThermalProperties] = {
    "carbon_steel": _p("Carbon steel", 50.0, 7850, 490, 400, 0.8),
    "stainless_steel": _p("Stainless steel 304", 16.0, 8000, 500, 800, 0.6),
    "concrete": _p("Concrete", 1.4, 2400, 880, 300),
    # moist sandy soil, the design value of ground-heat practice (VDI 4640-1, table 1:
    # 1.2-1.9 W/(m K) for moist sand; volumetric capacity ~1.9 MJ/(m3 K))
    "soil": _p("Soil (moist sand)", 1.5, 1900, 1000, 300),
}

FLUID_MATERIALS: dict[str, ThermalProperties] = {
    "air": _p("Air", 0.026, 1.2, 1005, 2000, 0.0),
    "water": _p("Water", 0.6, 1000, 4186, 100, 0.95),
    "thermal_oil": _p("Thermal oil", 0.12, 900, 2100, 350),
}

MATERIALS: dict[str, ThermalProperties] = {
    **STORAGE_MATERIALS, **INSULATION_MATERIALS, **STRUCTURAL_MATERIALS, **FLUID_MATERIALS
}

CATEGORIES = {
    "storage": STORAGE_MATERIALS,
    "insulation": INSULATION_MATERIALS,
    "structural": STRUCTURAL_MATERIALS,
    "fluid": FLUID_MATERIALS,
}


class MaterialManager:
    """Lookup and effective-property models for the material database."""

    def __init__(self, materials: dict[str, ThermalProperties] | None = None,
                 pore_fluid: str = "air") -> None:
        self.materials = dict(materials if materials is not None else MATERIALS)
        self.pore_fluid = pore_fluid

    # ------------------------------------------------------------- accessors
    def get(self, name: str) -> ThermalProperties:
        if name not in self.materials:
            raise KeyError(f"unknown material {name!r}; available: {sorted(self.materials)}")
        return self.materials[name]

    def list_materials(self, category: str | None = None) -> list[str]:
        if category is None:
            return list(self.materials)
        if category not in CATEGORIES:
            raise ValueError(f"unknown category {category!r}; expected {sorted(CATEGORIES)}")
        return [n for n in CATEGORIES[category] if n in self.materials]

    def add_custom_material(self, key: str, props: ThermalProperties) -> None:
        if not key:
            raise ValueError("material key must be a non-empty string")
        if props.k <= 0 or props.rho <= 0 or props.cp <= 0:
            raise ValueError(f"{key}: k, rho and cp must all be > 0")
        if props.t_max <= 0:
            raise ValueError(f"{key}: t_max must be > 0 K")
        self.materials[key] = replace(props)

    # ------------------------------------------------------ porous media
    def compute_effective_properties(self, solid: str, porosity: float,
                                     fluid: str | None = None) -> ThermalProperties:
        """Effective properties of a porous medium (geometric-mean conductivity)."""
        if not 0.0 <= porosity < 1.0:
            raise ValueError(f"porosity must be in [0, 1), got {porosity}")
        solid_props = self.get(solid)
        fluid_props = self.get(fluid or self.pore_fluid)
        phi = float(porosity)
        k_eff = solid_props.k ** (1.0 - phi) * fluid_props.k ** phi
        rho_eff = (1.0 - phi) * solid_props.rho + phi * fluid_props.rho
        cp_eff = ((1.0 - phi) * solid_props.rho * solid_props.cp
                  + phi * fluid_props.rho * fluid_props.cp) / rho_eff
        return ThermalProperties(
            name=f"{solid_props.name} (porous, phi={phi:.2f})",
            k=k_eff, rho=rho_eff, cp=cp_eff,
            t_max=min(solid_props.t_max, fluid_props.t_max),
            emissivity=solid_props.emissivity,
            description=f"packed bed, porosity {phi:.2f}",
        )

    def compute_packed_bed_properties(self, solid: str,
                                      packing_fraction: float = PACKING_FRACTION_DEFAULT
                                      ) -> ThermalProperties:
        """Effective properties of a packed bed of solid particles."""
        if not 0.0 < packing_fraction < 1.0:
            raise ValueError(f"packing_fraction must be in (0, 1), got {packing_fraction}")
        return self.compute_effective_properties(solid, 1.0 - float(packing_fraction))

    # --------------------------------------------------------------- energy
    def get_energy_density(self, material: str, t_high: float, t_low: float,
                           packing_fraction: float = 1.0) -> float:
        """Usable volumetric energy between two temperatures [J/m^3]."""
        if t_high < t_low:
            raise ValueError(f"t_high ({t_high}) must be >= t_low ({t_low})")
        props = (self.get(material) if packing_fraction == 1.0
                 else self.compute_packed_bed_properties(material, packing_fraction))
        return props.rho * props.cp * (t_high - t_low)

    def compare_materials(self, names: list[str], t_high: float, t_low: float,
                          packing_fraction: float = 1.0) -> list[dict]:
        """Energy density and conductivity table for a set of materials."""
        rows = []
        for name in names:
            props = (self.get(name) if packing_fraction == 1.0
                     else self.compute_packed_bed_properties(name, packing_fraction))
            rows.append({
                "material": name,
                "k": props.k,
                "rho": props.rho,
                "cp": props.cp,
                "t_max": props.t_max,
                "energy_density": props.rho * props.cp * (t_high - t_low),
            })
        return sorted(rows, key=lambda r: r["energy_density"], reverse=True)
