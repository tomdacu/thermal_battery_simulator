"""Throwaway smoke run: the Pipes tab builds, paints and drives a transient."""
from __future__ import annotations

import numpy as np
from PyQt6.QtWidgets import QApplication

app = QApplication([])
from gui.main_window import ThermalBatteryGUI

win = ThermalBatteryGUI()
panel = win.geometry_panel
panel.refined.setChecked(False)
panel.auto_first.setChecked(False)
panel.spacing.setValue(0.25)
panel.tubes_active.setChecked(True)
win.build_mesh()
print("mesh", win.mesh.Nx, win.mesh.Ny, win.mesh.Nz)
print("network before:", panel.pipe_network())

panel.pipe_layout.setCurrentIndex(2)          # concentric rings
panel.pipe_insulated.setChecked(True)
panel.pipe_junction.setValue(0.05)
panel.pipe_network_requested.emit()          # what the button does
network = panel.pipe_network()
print("network:", None if network is None else network.n_risers, "risers")
print("tubes tab off:", not panel.tubes_active.isChecked())
print("info tail:", panel.pipe_info.text().splitlines()[-1])
tubes = win.mesh.material_id == 4
print("painted cells:", int(tubes.sum()), "with film:", int((win.mesh.bc_h > 0).sum()))

# the mesh band the junction asked for
spec = panel.grid_spec()
print("junction bands in the spec:", [(b.start, b.end, b.target) for b in spec.z])

config = win._run_config()
print("run config: network", config.pipe_network is not network and "set",
      "flow", config.pipe_flow)
win.analysis_panel.radio_transient.setChecked(True)
win.analysis_panel.duration.setValue(30.0)
win.analysis_panel.duration_unit.setCurrentIndex(1)      # minutes
win.analysis_panel.dt.setValue(300.0)
win.analysis_panel.power_constant.setChecked(True)
win.analysis_panel.power_value.setValue(20.0)
config = win._run_config()
config.analysis = "transient"
work = win.controller._make_work(config, win.mesh)
result = work(lambda value, message: None, lambda: False)
print("transient samples:", len(result))
print("T storage", round(result.T_mean_storage[0] - 273.15, 2), "->",
      round(result.T_mean_storage[-1] - 273.15, 2), "degC")
print("power in", round(result.E_in_cumulative[-1] / 3600.0, 2), "Wh")
print("storage spread of the run:", round(float(np.ptp(result.T_mean_storage)), 3))
win.close()
print("OK")
