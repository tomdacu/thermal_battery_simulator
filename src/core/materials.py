"""Material database and porous-media models.  SI units, temperatures in KELVIN.

Public API (frozen - the rest of the package is written against it)::

    MATERIALS: dict[str, ThermalProperties]
    MaterialManager.get(name) / list_materials(category) / add_custom_material(props)
    MaterialManager.compute_packed_bed_properties(solid, packing_fraction)
    MaterialManager.compute_effective_properties(solid, porosity, fluid=None)
    MaterialManager.get_energy_density(name, t_high, t_low, packing_fraction=1.0)
    MaterialManager.compare_materials(names, t_high, t_low, packing_fraction=1.0)

Packed bed.  The density is the arithmetic mean and the specific heat the mass-weighted
mean of the grains and the pore gas.  The conductivity is the **Zehner-Bauer-Schlunder**
model of the VDI Heat Atlas (2nd ed., 2010, chapter D6.3, E. Tsotsas; Zehner and
Schlunder, Chem. Ing. Tech. 42 (1970) 933), with the radiation between the grains
(Breitbach and Barthels, Nucl. Technol. 49 (1980) 392): the gas in the voids, the
contact through the grain cores and the radiation across the voids - which grows as
``T^3 d`` and at 500 degC carries a third of the heat of a millimetre sand.  In the
VDI's notation (``psi`` porosity, ``kappa = k_s / k_f``, ``B`` the deformation
parameter of the unit cell, ``k_rad`` the radiative conductivity over ``k_f``)::

    k_rad = 4 sigma T^3 d / ((2 / eps - 1) k_f)
    B     = C_f ((1 - psi) / psi)^(10/9)            C_f = 1.25 spheres, 1.4 broken grains
    N     = 1 + (k_rad - B) / kappa
    k_c   = 2/N [ B (kappa + k_rad - 1) / (N^2 kappa) ln((kappa + k_rad) / B)
                  + (B + 1) / (2 B) (k_rad - B) - (B - 1) / N ]
    k_bed / k_f = (1 - sqrt(1 - psi)) (1 + psi k_rad)
                  + sqrt(1 - psi) (phi kappa + (1 - phi) k_c)

with ``phi = 0.0077`` the flattening of the contacts (VDI) and the pore gas at its own
temperature (the air of Incropera's Table A.4).  The Smoluchowski reduction of the gas
conductivity near the contacts is left out: it matters below ~0.1 mm or under vacuum,
not for millimetre grains at one atmosphere.  The older model - the geometric mean
``k_s^(1-psi) k_f^psi``, constant in temperature - stays available as ``"geometric"``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..constants import PACKING_FRACTION_DEFAULT
from ..units import c_to_k

#: Stefan-Boltzmann constant [W/(m^2 K^4)]
SIGMA = 5.670374419e-8
#: grain diameter of the bed by default [m]: a millimetre, crushed rock or coarse sand
PARTICLE_DIAMETER_DEFAULT = 1.0e-3
#: bed conductivity models
BED_MODELS = ("zbs", "geometric")


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
    # the *grain*: quartz (Incropera, Table A.3: 6.21 W/(m K) across the axis, 2650
    # kg/m3); the bed values follow from the packing.  The entry used to carry the
    # bed's own 0.35 W/(m K) and 1500 kg/m3, and the packing then diluted them again
    "silica_sand": _p("Silica sand", 6.2, 2650, 800, 1200, 0.9, "cheap bulk medium"),
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


@dataclass(frozen=True)
class PackedBed:
    """The conductivity of a packed bed as a function of temperature (see the module).

    ``solid_k`` is the grain's conductivity, ``porosity`` the void fraction, ``diameter``
    the grain size [m], ``emissivity`` the grain surface's.  ``model`` is ``"zbs"`` or
    ``"geometric"`` (constant).  Calling it with temperatures [K] returns the bed
    conductivity [W/(m K)], elementwise.
    """

    solid_k: float
    porosity: float
    diameter: float = PARTICLE_DIAMETER_DEFAULT
    emissivity: float = 0.9
    shape: float = 1.4
    flattening: float = 0.0077
    fluid_k: float = 0.0263          # the pore gas at 300 K
    fluid: str = "air"
    model: str = "zbs"

    def gas_k(self, temperature: np.ndarray) -> np.ndarray:
        """The pore gas conductivity at ``temperature`` [W/(m K)]."""
        from ..solver.fluid import property_shape

        t = np.atleast_1d(np.asarray(temperature, dtype=float))
        shapes = [property_shape(self.fluid, value) for value in np.unique(t)]
        if shapes[0] is None:
            return np.full(t.shape, self.fluid_k)
        table = dict(zip(np.unique(t).tolist(), (s[2] for s in shapes), strict=True))
        return self.fluid_k * np.vectorize(table.__getitem__)(t)

    def __call__(self, temperature) -> np.ndarray:
        t = np.atleast_1d(np.asarray(temperature, dtype=float))
        psi = float(self.porosity)
        if self.model == "geometric":
            return np.full(t.shape, self.solid_k ** (1.0 - psi) * self.fluid_k ** psi)
        # the gas conductivity is smooth: evaluate it on a 1 K grid and interpolate,
        # so a field of a million cells costs a table of a few hundred entries
        low, high = float(np.floor(t.min())), float(np.ceil(t.max())) + 1.0
        grid = np.arange(low, high + 1.0)
        k_f = np.interp(t, grid, self.gas_k(grid))
        kappa = self.solid_k / k_f
        k_rad = (4.0 * SIGMA * t ** 3 * self.diameter
                 / ((2.0 / self.emissivity - 1.0) * k_f))
        b = self.shape * ((1.0 - psi) / psi) ** (10.0 / 9.0)
        n = 1.0 + (k_rad - b) / kappa
        k_c = (2.0 / n) * (b * (kappa + k_rad - 1.0) / (n ** 2 * kappa)
                           * np.log((kappa + k_rad) / b)
                           + (b + 1.0) / (2.0 * b) * (k_rad - b) - (b - 1.0) / n)
        root = np.sqrt(1.0 - psi)
        ratio = ((1.0 - root) * (1.0 + psi * k_rad)
                 + root * (self.flattening * kappa + (1.0 - self.flattening) * k_c))
        return ratio * k_f


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
                                      packing_fraction: float = PACKING_FRACTION_DEFAULT,
                                      temperature: float | None = None,
                                      particle_diameter: float = PARTICLE_DIAMETER_DEFAULT,
                                      model: str = "geometric") -> ThermalProperties:
        """Effective properties of a packed bed of solid particles.

        ``model="geometric"`` is the constant geometric mean (the reference the older
        callers were written against); ``"zbs"`` the Zehner-Bauer-Schlunder conductivity
        with radiation at ``temperature`` [K] (293.15 K by default), for grains of
        ``particle_diameter`` [m].
        """
        if not 0.0 < packing_fraction < 1.0:
            raise ValueError(f"packing_fraction must be in (0, 1), got {packing_fraction}")
        props = self.compute_effective_properties(solid, 1.0 - float(packing_fraction))
        if model == "geometric":
            return props
        law = self.packed_bed(solid, packing_fraction, particle_diameter, model)
        k = float(law(293.15 if temperature is None else temperature)[0])
        return replace(props, k=k, description=f"{props.description}, {model}")

    def packed_bed(self, solid: str, packing_fraction: float = PACKING_FRACTION_DEFAULT,
                   particle_diameter: float = PARTICLE_DIAMETER_DEFAULT,
                   model: str = "zbs") -> PackedBed:
        """The bed conductivity as a law of the temperature (:class:`PackedBed`)."""
        if model not in BED_MODELS:
            raise ValueError(f"unknown bed model {model!r}; expected one of {BED_MODELS}")
        if particle_diameter <= 0.0:
            raise ValueError(f"particle_diameter must be > 0, got {particle_diameter}")
        grain = self.get(solid)
        gas = self.get(self.pore_fluid)
        return PackedBed(solid_k=grain.k, porosity=1.0 - float(packing_fraction),
                         diameter=float(particle_diameter), emissivity=grain.emissivity,
                         fluid_k=gas.k, fluid=self.pore_fluid, model=model)

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
