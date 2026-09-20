"""Driving profiles and initial conditions for the transient solver.

All temperatures are KELVIN, all powers W, mass flows kg/s.  Every profile can
describe itself with :meth:`validate`, so callers surface a problem list instead
of silently falling back to a default behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import T_AMBIENT_DEFAULT, T_INITIAL_DEFAULT
from ..units import check_kelvin

POWER_MODES = ("off", "constant", "schedule", "csv")
EXTRACTION_MODES = ("off", "power", "flow_rate")
IC_MODES = ("uniform", "by_material", "from_file", "keep")


@dataclass
class PowerProfile:
    """Heater power as a function of time [W]."""

    mode: str = "constant"
    constant_power: float = 0.0
    times: list[float] = field(default_factory=list)     # [s], increasing
    powers: list[float] = field(default_factory=list)    # [W]
    csv_path: str = ""

    def __post_init__(self) -> None:
        self._t: np.ndarray | None = None
        self._p: np.ndarray | None = None
        if self.mode in ("schedule", "csv"):
            self._build_cache()

    def _build_cache(self) -> None:
        if self.mode == "csv":
            if not self.csv_path:
                raise ValueError("PowerProfile: csv mode requires csv_path")
            try:
                data = np.loadtxt(self.csv_path, delimiter=",", ndmin=2)
            except OSError as exc:
                raise FileNotFoundError(f"PowerProfile: cannot read {self.csv_path!r}: {exc}") from exc
            if data.shape[1] < 2:
                raise ValueError(f"PowerProfile: {self.csv_path!r} needs two columns (t, P)")
            times, powers = data[:, 0], data[:, 1]
        else:
            times = np.asarray(self.times, dtype=float)
            powers = np.asarray(self.powers, dtype=float)
        if times.size == 0:
            raise ValueError("PowerProfile: the schedule is empty - add at least one "
                             "(time, power) row")
        if times.size != powers.size:
            raise ValueError(f"PowerProfile: {times.size} times but {powers.size} powers")
        if np.any(np.diff(times) <= 0):
            raise ValueError("PowerProfile: times must be strictly increasing")
        if np.any(powers < 0):
            raise ValueError("PowerProfile: powers must be >= 0")
        self._t, self._p = times, powers

    def power_at(self, t: float) -> float:
        """Power at time ``t``; the last value is held beyond the last point."""
        if self.mode == "off":
            return 0.0
        if self.mode == "constant":
            return max(float(self.constant_power), 0.0)
        if self._t is None:
            self._build_cache()
        return float(np.interp(t, self._t, self._p))

    def validate(self) -> list[str]:
        problems = []
        if self.mode not in POWER_MODES:
            problems.append(f"PowerProfile: unknown mode {self.mode!r}")
            return problems
        if self.mode == "constant" and self.constant_power < 0:
            problems.append("PowerProfile: constant_power must be >= 0")
        if self.mode in ("schedule", "csv"):
            try:
                self._build_cache()
            except (ValueError, FileNotFoundError) as exc:
                problems.append(str(exc))
        return problems


@dataclass
class ExtractionProfile:
    """Heat extraction from the heat-transfer fluid [W] / [kg/s] / [K]."""

    mode: str = "off"
    power: float = 0.0
    mass_flow: float = 0.0
    t_inlet: float = T_AMBIENT_DEFAULT
    h_fluid: float = 300.0
    fluid_cp: float = 4182.0
    fluid_rho: float = 1000.0
    times: list[float] = field(default_factory=list)
    powers: list[float] = field(default_factory=list)

    def power_request(self, t: float = 0.0) -> float:
        """Requested power at ``t`` [W] (before the availability cap)."""
        if self.mode != "power":
            return 0.0
        if self.times:
            return max(float(np.interp(t, self.times, self.powers)), 0.0)
        return max(float(self.power), 0.0)

    def outlet_temperature(self, power: float) -> float | None:
        """Fluid outlet temperature [K] for the given extracted power."""
        if self.mode != "flow_rate" or self.mass_flow <= 0:
            return None
        return self.t_inlet + power / (self.mass_flow * self.fluid_cp)

    def validate(self) -> list[str]:
        problems = []
        if self.mode not in EXTRACTION_MODES:
            problems.append(f"ExtractionProfile: unknown mode {self.mode!r}")
            return problems
        try:
            check_kelvin(self.t_inlet, "ExtractionProfile.t_inlet")
        except ValueError as exc:
            problems.append(str(exc))
        if self.h_fluid <= 0:
            problems.append("ExtractionProfile: h_fluid must be > 0")
        if self.mode == "power":
            if self.power < 0:
                problems.append("ExtractionProfile: power must be >= 0")
            if self.times:
                if len(self.times) != len(self.powers):
                    problems.append("ExtractionProfile: times and powers length mismatch")
                elif np.any(np.diff(np.asarray(self.times, dtype=float)) <= 0):
                    problems.append("ExtractionProfile: schedule times must be increasing")
        if self.mode == "flow_rate" and self.mass_flow <= 0:
            problems.append("ExtractionProfile: flow_rate mode requires mass_flow > 0")
        return problems


@dataclass
class InitialCondition:
    """Starting field of a transient run; returns KELVIN."""

    mode: str = "uniform"
    t_uniform: float = T_INITIAL_DEFAULT
    t_by_material: dict[int, float] = field(default_factory=dict)
    file_path: str = ""

    def apply_to_mesh(self, mesh) -> np.ndarray | None:
        """Field for ``mesh`` [K], or ``None`` when the current mesh.T must be kept."""
        if self.mode == "keep":
            # the mesh already carries the field to start from (e.g. the steady
            # pre-run of "start from the steady solution")
            return None
        if self.mode == "uniform":
            check_kelvin(self.t_uniform, "InitialCondition.t_uniform")
            return np.full(mesh.T.shape, float(self.t_uniform))
        if self.mode == "by_material":
            if not self.t_by_material:
                raise ValueError("InitialCondition: by_material requires a material map")
            field = np.full(mesh.T.shape, float(T_AMBIENT_DEFAULT))
            covered = np.zeros(mesh.T.shape, dtype=bool)
            for material_id, t_value in self.t_by_material.items():
                check_kelvin(t_value, f"InitialCondition[{material_id}]")
                mask = mesh.material_id == int(material_id)
                field[mask] = float(t_value)
                covered |= mask
            if not covered.any():
                raise ValueError("InitialCondition: no cell matches the given material ids")
            return field
        if self.mode == "from_file":
            if not self.file_path:
                raise ValueError("InitialCondition: from_file requires file_path")
            try:
                field = np.load(self.file_path)
            except OSError as exc:
                raise FileNotFoundError(f"InitialCondition: cannot read {self.file_path!r}: {exc}") from exc
            if field.shape != mesh.T.shape:
                raise ValueError(f"InitialCondition: file shape {field.shape} != mesh {mesh.T.shape}")
            check_kelvin(field, "InitialCondition file")
            return np.asarray(field, dtype=float)
        raise ValueError(f"InitialCondition: unknown mode {self.mode!r}")

    def validate(self, mesh=None) -> list[str]:
        problems = []
        if self.mode not in IC_MODES:
            problems.append(f"InitialCondition: unknown mode {self.mode!r}")
            return problems
        if self.mode == "uniform":
            try:
                check_kelvin(self.t_uniform, "t_uniform")
            except ValueError as exc:
                problems.append(str(exc))
        if self.mode == "by_material" and not self.t_by_material:
            problems.append("InitialCondition: by_material needs at least one material")
        if self.mode == "from_file" and mesh is not None and self.file_path:
            try:
                field = np.load(self.file_path)
                if field.shape != mesh.T.shape:
                    problems.append("InitialCondition: file shape does not match the mesh")
            except OSError:
                problems.append(f"InitialCondition: cannot read {self.file_path!r}")
        return problems
