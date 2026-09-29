"""Results panel: statistics, energy balance, materials, transient and log tabs."""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QFileDialog, QHBoxLayout, QTabWidget, QTextEdit, QVBoxLayout,
                             QWidget)

from src.analysis.balance import compute_balance
from src.constants import T_AMBIENT_DEFAULT
from src.core.materials import MaterialManager
from src.core.mesh import storage_mask
from src.viz.scene import MATERIAL_NAMES, grid_lines

from ..units import k_to_c
from ..widgets import button

MONO = "font-family: Consolas, monospace; font-size: 11px;"


class ResultsPanel(QWidget):
    """Read-only report tabs; every number is computed by ``src.analysis``."""

    #: the window owns the mesh, so it writes the field export
    export_requested = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        self.tabs.setUsesScrollButtons(True)
        layout.addWidget(self.tabs)
        self.stats_text = self._add_tab("Statistics")
        self.energy_text = self._add_tab("Energy balance")
        self.materials_text = self._add_tab("Materials")
        self.transient_text = self._add_tab("Transient")
        self.log_text = self._add_tab("Log", size=10)
        row = QHBoxLayout()
        row.addWidget(button("Series CSV...", self._export_csv,
                             "The time series of the last transient"))
        row.addWidget(button("Field VTK...", self.export_requested.emit,
                             "Every leaf and every field, for ParaView"))
        layout.addLayout(row)

    def _add_tab(self, title: str, size: int = 11) -> QTextEdit:
        view = QTextEdit()
        view.setReadOnly(True)
        view.setStyleSheet(f"font-family: Consolas, monospace; font-size: {size}px;")
        self.tabs.addTab(view, title)
        return view

    # ------------------------------------------------------------------ log
    def log(self, message: str) -> None:
        self.log_text.append(message)
        if len(self.log_text.toPlainText()) > 400_000:
            self.log_text.clear()
            self.log_text.append("(log truncated)")

    # ---------------------------------------------------------- statistics
    def update_statistics(self, mesh) -> None:
        storage = storage_mask(mesh.material_id)
        # the air around the vessel is excluded from the problem and held at the ambient:
        # it is not part of the battery and would only dilute the statistics
        active = ~np.asarray(mesh.excluded, dtype=bool)
        field_c = np.asarray(mesh.T)[active] - 273.15
        lines = [
            "=== Temperature (degC) ===",
            f"battery        min {field_c.min():8.2f}   max {field_c.max():8.2f}   "
            f"mean {field_c.mean():8.2f}   std {field_c.std():7.2f}",
        ]
        if storage.any():
            s = mesh.T[storage] - 273.15
            lines += [
                f"storage (sand) min {s.min():8.2f}   max {s.max():8.2f}   "
                f"mean {s.mean():8.2f}   std {s.std():7.2f}",
                f"percentiles    p10 {np.percentile(s, 10):8.2f}   "
                f"p50 {np.percentile(s, 50):8.2f}   p90 {np.percentile(s, 90):8.2f}",
            ]
        lines += [
            "",
            *grid_lines(mesh),
        ]
        self.stats_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------- energy balance
    def update_energy(self, mesh, losses=None, transient=None, ambient=None,
                      radiation: bool = False, loop=None) -> None:
        # the ambient of the *run* (the ground and the air are inputs, not defaults)
        # and the same radiation switch the solver used, or the fluxes shown would
        # not be the fluxes that were solved
        balance = compute_balance(mesh, T_AMBIENT_DEFAULT if ambient is None else ambient,
                                  radiation=radiation)
        lines = [
            "=== Power balance ===",
            f"input (gas -> bed)  {balance.p_input / 1000:9.3f} kW",
            f"extracted (bed->gas){balance.p_extracted / 1000:9.3f} kW",
            f"losses (envelope)   {balance.q_battery / 1000:9.3f} kW",
            f"    top             {balance.q_battery_top / 1000:9.3f} kW",
            f"    side            {balance.q_battery_side / 1000:9.3f} kW",
            f"    bottom          {balance.q_battery_bottom / 1000:9.3f} kW",
            f"losses (all paths)  {balance.q_domain / 1000:9.3f} kW",
            f"imbalance           {balance.imbalance / 1000:9.3f} kW",
            "",
            "=== Stored energy ===",
            f"E stored            {balance.e_stored / 3.6e6:9.3f} kWh",
            f"exergy stored       {balance.ex_stored / 3.6e6:9.3f} kWh",
            f"exergy destroyed    {balance.ex_destroyed / 3.6e6:9.3f} kWh",
            f"T storage mean      {k_to_c(balance.t_mean_storage):9.2f} degC",
            f"T max (battery)     {k_to_c(balance.t_max):9.2f} degC",
        ]
        if balance.q_battery > 0:
            hours = balance.e_stored / balance.q_battery / 3600.0
            lines.append(f"thermal autonomy    {hours:9.1f} h")
        if loop is not None:
            lines += [
                "",
                "=== Gas loop ===",
                f"gas in / out        {k_to_c(loop.t_in):9.1f} / {k_to_c(loop.t_out):.1f} degC",
                f"bed power           {loop.power / 1000:9.3f} kW",
                f"NTU (mean)          {loop.ntu:9.2f}",
                f"pressure drop       {loop.delta_p / 1000:9.3f} kPa",
                f"fan power           {loop.fan_power / 1000:9.3f} kW "
                f"({100 * loop.circulation_loss:.2f} % of the bed power)",
            ]
        if losses is not None:
            lines += [
                "",
                "=== Steady standby ===",
                f"converged           {losses.converged} ({losses.iterations} iterations)",
                f"storage held at     {k_to_c(losses.t_mean_storage):9.1f} degC",
                f"holding power       {losses.power / 1000:9.3f} kW  (= the standby losses)",
                f"power density       {losses.power_density:9.1f} W/m³",
            ]
        if transient is not None and len(transient):
            lines += [
                "",
                "=== Cumulative (transient) ===",
                f"E in                {transient.E_in_cumulative[-1] / 3.6e6:9.3f} kWh",
                f"E out               {transient.E_out_cumulative[-1] / 3.6e6:9.3f} kWh",
                f"E losses            {transient.E_losses_cumulative[-1] / 3.6e6:9.3f} kWh",
            ]
        self.energy_text.setPlainText("\n".join(lines))

    # ----------------------------------------------------------- materials
    def update_materials(self, mesh, storage_material: str, packing: float,
                         insulation_material: str) -> None:
        manager = MaterialManager()
        counts = dict(zip(*[x.tolist() for x in np.unique(mesh.material_id, return_counts=True)],
         strict=True))
        total = max(sum(counts.values()), 1)
        lines = ["=== Material map ==="]
        for material_id, count in sorted(counts.items()):
            name = MATERIAL_NAMES.get(int(material_id), f"id {material_id}")
            lines.append(f"{name:<18} {count:8,} cells  ({100 * count / total:5.1f} %)")
        storage = manager.compute_packed_bed_properties(storage_material, packing)
        insulation = manager.get(insulation_material)
        lines += [
            "",
            f"storage     {storage_material} @ phi_x={packing:.2f}",
            f"            k {storage.k:.3f} W/(m·K)  rho {storage.rho:.0f} kg/m³  "
            f"cp {storage.cp:.0f} J/(kg·K)",
            f"insulation  {insulation_material}",
            f"            k {insulation.k:.3f} W/(m·K)  rho {insulation.rho:.0f} kg/m³  "
            f"cp {insulation.cp:.0f} J/(kg·K)",
        ]
        self.materials_text.setPlainText("\n".join(lines))

    # ----------------------------------------------------------- transient
    def update_transient(self, results) -> None:
        if not results or not len(results):
            self.transient_text.setPlainText("no transient run yet")
            self._results = None
            return
        self._results = results
        times = np.asarray(results.times)
        lines = [
            f"samples          {len(times)}  (t = {times[0]:.1f} .. {times[-1]:.1f} s, "
            f"wall {results.wall_time:.1f} s)",
            "",
            "   t [s]     T_mean    T_max     T_min   P_heat   P_extract   losses",
        ]
        step = max(1, len(times) // 12)
        for i in range(0, len(times), step):
            lines.append(
                f"{times[i]:8.1f} {k_to_c(results.T_mean_storage[i]):9.2f} "
                f"{k_to_c(results.T_max[i]):9.2f} {k_to_c(results.T_min[i]):9.2f} "
                f"{results.P_heaters[i] / 1000:8.2f} {results.P_extracted[i] / 1000:10.2f} "
                f"{results.Q_losses_total[i] / 1000:8.2f}")
        self.transient_text.setPlainText("\n".join(lines))

    def _export_csv(self) -> None:
        results = getattr(self, "_results", None)
        if results is None:
            self.log("no transient results to export")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export time series", "results/results.csv",
                                              "CSV (*.csv)")
        if path:
            results.export_csv(path)
            self.log(f"time series written to {path}")
