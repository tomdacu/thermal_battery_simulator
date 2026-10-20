"""
src.solver.results
==================
Result container of a transient run: one list per sampled series.

WHAT IT DOES
------------
Holds the time series of
:class:`~src.solver.transient.TransientSolver` - the scalar diagnostics, the
optional full fields and the wall time - and converts them to arrays, to a
summary or to a CSV file.

HOW IT WORKS
------------
-> APPEND-ONLY SERIES <-
   Every scalar is a list appended by :meth:`TransientResults.add_timestep`, so
   a sample can never fail on a missing or unexpected key: a missing value is
   recorded as zero.  ``SCALAR_FIELDS`` fixes both the append order and the
   column order of the export.

-> ARRAYS AND EXPORT <-
   :meth:`TransientResults.to_arrays` stacks the series, and
   :meth:`TransientResults.export_csv` writes them as CSV with the names of the
   series as the header.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: scalar series recorded at every step, in append and CSV-column order.
SCALAR_FIELDS = (
    "T_mean_storage", "T_max", "T_min", "T_mean_insulation", "T_mean_shell",
    "P_heaters", "P_extracted", "Q_losses_top", "Q_losses_bottom", "Q_losses_side",
    "Q_losses_total", "E_stored", "E_in_cumulative", "E_out_cumulative",
    "E_losses_cumulative", "Ex_stored", "Ex_destroyed",
)


@dataclass
class TransientResults:
    """Time series of a transient run (lists, so ``add_timestep`` never fails)."""

    times: list[float] = field(default_factory=list)
    T_mean_storage: list[float] = field(default_factory=list)
    T_max: list[float] = field(default_factory=list)
    T_min: list[float] = field(default_factory=list)
    T_mean_insulation: list[float] = field(default_factory=list)
    T_mean_shell: list[float] = field(default_factory=list)
    P_heaters: list[float] = field(default_factory=list)
    P_extracted: list[float] = field(default_factory=list)
    Q_losses_top: list[float] = field(default_factory=list)
    Q_losses_bottom: list[float] = field(default_factory=list)
    Q_losses_side: list[float] = field(default_factory=list)
    Q_losses_total: list[float] = field(default_factory=list)
    E_stored: list[float] = field(default_factory=list)
    E_in_cumulative: list[float] = field(default_factory=list)
    E_out_cumulative: list[float] = field(default_factory=list)
    E_losses_cumulative: list[float] = field(default_factory=list)
    #: exergy stored above the ambient [J].
    Ex_stored: list[float] = field(default_factory=list)
    #: exergy destroyed between the source and the stored exergy [J].
    Ex_destroyed: list[float] = field(default_factory=list)
    #: the whole temperature field of each saved sample [K], flat in mesh order.
    T_fields: list[np.ndarray] = field(default_factory=list)
    wall_time: float = 0.0

    def add_timestep(self, t: float, T_field: np.ndarray | None = None, **values) -> None:
        """Append one sample; unknown or missing keys are ignored."""
        self.times.append(float(t))
        for name in SCALAR_FIELDS:
            getattr(self, name).append(float(values.get(name, 0.0)))
        if T_field is not None:
            self.T_fields.append(np.asarray(T_field, dtype=np.float64).copy())

    def __len__(self) -> int:
        return len(self.times)

    def to_arrays(self) -> dict[str, np.ndarray]:
        out = {"times": np.asarray(self.times)}
        for name in SCALAR_FIELDS:
            out[name] = np.asarray(getattr(self, name))
        return out

    def summary(self) -> dict[str, float]:
        if not self.times:
            return {}
        return {
            "t_end": self.times[-1],
            "T_mean_storage_end": self.T_mean_storage[-1],
            "E_in_cumulative": self.E_in_cumulative[-1],
            "E_out_cumulative": self.E_out_cumulative[-1],
            "E_losses_cumulative": self.E_losses_cumulative[-1],
        }

    def export_csv(self, path: str) -> str:
        arrays = self.to_arrays()
        header = ",".join(arrays)
        data = np.column_stack([arrays[name] for name in arrays])
        np.savetxt(path, data, delimiter=",", header=header, comments="")
        return path
