"""The outside of the storage: film coefficients, without simulating the air.

A storage vessel sits in air that is not part of the machine: simulating it costs
cells and buys a thermal resistance that is a few percent of the envelope.  What the
model needs instead is the **film coefficient of the outer surface**, which is the sum
of two mechanisms acting on the same surface:

* **natural convection** on the vertical wall and on the top/bottom faces, from the
  Rayleigh number (Churchill-Chu correlation for a vertical plate, which is the
  standard for a cylinder of this aspect ratio);
* **forced convection from the wind**, for which the engineering practice uses the
  simple combined expression ``h = 4 + 4 v`` (v in m/s, ISO 6946 external surface) or,
  when the direction matters, the cylinder-in-crossflow correlation with a windward
  weighting.

The two add as conductances in parallel, which is what the standard combined
expressions already assume (their constant term *is* the natural-convection share).

Everything here is a *closure*, not a PDE: the error it carries is the correlation
uncertainty (tens of percent), which is why the model reports the sensitivity of the
losses to ``h_out`` instead of pretending to know it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import GRAVITY, T_AMBIENT_DEFAULT


@dataclass(frozen=True)
class AirProperties:
    """Air properties for the film correlations (film temperature)."""

    k: float = 0.0262        # [W/(m K)]
    nu: float = 1.57e-5      # [m^2/s]
    alpha: float = 2.2e-5    # [m^2/s]
    pr: float = 0.71         # [-]
    beta: float = 1.0 / 300.0   # [1/K] ideal gas at ~300 K

    @staticmethod
    def at(temperature: float) -> AirProperties:
        """Air at a temperature: power laws around the 300 K reference."""
        ratio = max(temperature, 150.0) / 300.0
        return AirProperties(k=0.0262 * ratio ** 0.8, nu=1.57e-5 * ratio ** 1.7,
                             alpha=2.2e-5 * ratio ** 1.7, pr=0.71, beta=1.0 / temperature)


def rayleigh(delta_t: float, length: float, air: AirProperties) -> float:
    """Rayleigh number of a surface of size ``length`` with a temperature jump."""
    if length <= 0 or delta_t == 0:
        return 0.0
    return float(GRAVITY * air.beta * abs(delta_t) * length ** 3 / (air.nu * air.alpha))


def h_natural_vertical(t_surface: float, t_ambient: float, height: float,
                       air: AirProperties | None = None) -> float:
    """Churchill-Chu for a vertical surface [W/(m^2 K)] (valid over all Ra)."""
    air = air or AirProperties.at(0.5 * (t_surface + t_ambient))
    ra = rayleigh(t_surface - t_ambient, height, air)
    if ra <= 0:
        return 0.0
    nu = (0.825 + 0.387 * ra ** (1.0 / 6.0)
          / (1.0 + (0.492 / air.pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    return float(nu * air.k / height)


def h_natural_horizontal(t_surface: float, t_ambient: float, width: float,
                         facing_up: bool = True,
                         air: AirProperties | None = None) -> float:
    """Natural convection on a horizontal face [W/(m^2 K)].

    Hot face up (or cold face down): ``Nu = 0.15 Ra^(1/3)``; hot face down:
    ``Nu = 0.27 Ra^(1/4)`` - the stable stratification of the latter makes it weaker.
    """
    air = air or AirProperties.at(0.5 * (t_surface + t_ambient))
    ra = rayleigh(t_surface - t_ambient, width, air)
    if ra <= 0:
        return 0.0
    hot_up = (t_surface > t_ambient) == facing_up
    nu = 0.15 * ra ** (1.0 / 3.0) if hot_up else 0.27 * ra ** (1.0 / 4.0)
    return float(max(nu, 1.0) * air.k / width)


def h_wind(wind_speed: float, simple: bool = True, diameter: float = 1.0,
           air: AirProperties | None = None) -> float:
    """Wind convection on the outer surface [W/(m^2 K)].

    ``simple``: ``h = 4 + 4 v`` (the combined external-surface expression of the
    building-physics standard, which already includes the natural share at rest).
    Otherwise the cylinder-in-crossflow correlation ``Nu = 0.26 Re^0.6 Pr^0.37`` for
    the windward side, halved to represent the average over a cylinder.
    """
    if wind_speed <= 0:
        return 0.0
    if simple:
        return float(4.0 + 4.0 * wind_speed)
    air = air or AirProperties.at(T_AMBIENT_DEFAULT)
    re = wind_speed * diameter / air.nu
    nu = 0.26 * re ** 0.6 * air.pr ** 0.37
    return float(0.5 * nu * air.k / diameter)


def h_out(t_surface: float, t_ambient: float, height: float, width: float,
          wind_speed: float = 0.0) -> dict[str, float]:
    """Total film coefficient of the envelope [W/(m^2 K)] and its two shares."""
    vertical = h_natural_vertical(t_surface, t_ambient, max(height, 1e-3))
    top = h_natural_horizontal(t_surface, t_ambient, max(width, 1e-3), facing_up=True)
    bottom = h_natural_horizontal(t_surface, t_ambient, max(width, 1e-3),
                                  facing_up=False)
    natural = float(np.mean([vertical, top, bottom]))
    forced = h_wind(wind_speed)
    return {"vertical": vertical, "top": top, "bottom": bottom,
            "natural": natural, "wind": forced, "total": natural + forced}
