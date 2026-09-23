"""Interactive 3D view: PyVista scene + clip/opacity/colormap controls."""
from __future__ import annotations


from PyQt6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import Qt

from src.viz.scene import (FIELD_ARRAYS, add_field, add_geometry_preview,
                           add_material_legend, domain_extent)

from ..widgets import button, combo

_AXES = {"x": (0, 1.0, 0.0, 0.0), "y": (1, 0.0, 1.0, 0.0), "z": (2, 0.0, 0.0, 1.0)}


class VizView(QWidget):
    """Embeds a PyVista plotter and rebuilds the scene on demand.

    When no OpenGL context is available (headless session, VM, remote desktop)
    the widget degrades to a placeholder instead of taking the application down;
    ``THERMAL_DISABLE_3D=1`` forces that mode.
    """

    def __init__(self, parent=None) -> None:
        import os

        super().__init__(parent)
        layout = QVBoxLayout(self)
        self.plotter = None
        self._mesh = None
        self._battery = None
        self._network = None
        self._mode = "field"
        self._disabled_reason = None
        if os.environ.get("THERMAL_DISABLE_3D", "") in ("1", "true", "yes"):
            self._disabled_reason = "3D view disabled by THERMAL_DISABLE_3D"
        else:
            try:
                from pyvistaqt import QtInteractor

                self.plotter = QtInteractor(self)
                self.plotter.setMinimumSize(600, 450)
                layout.addWidget(self.plotter.interactor)
                self.plotter.add_axes()
            except Exception as exc:  # pragma: no cover - display dependent
                self._disabled_reason = f"3D view unavailable ({type(exc).__name__}: {exc})"
        if self.plotter is None:
            placeholder = QLabel(self._disabled_reason or "3D view unavailable")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setMinimumSize(400, 300)
            layout.addWidget(placeholder)
        layout.addWidget(self._controls())
        self.controls.setEnabled(self.plotter is not None)

    @property
    def enabled(self) -> bool:
        return self.plotter is not None

    def _controls(self) -> QGroupBox:
        """Two compact rows with full-width sliders and explicit labels."""
        box = QGroupBox("3D view")
        outer = QVBoxLayout(box)
        outer.setContentsMargins(6, 4, 6, 4)
        outer.setSpacing(2)

        top = QHBoxLayout()
        self.field_combo = combo(tuple((f, f) for f in FIELD_ARRAYS), 0, self.render)
        self.axis_combo = combo((("x", "x"), ("y", "y"), ("z", "z")), 2, self.render)
        self.reset_btn = button("Reset camera", self._reset_camera,
                                "Restore the default point of view")
        self.field_combo.setToolTip("Quantity shown on the cut plane")
        self.axis_combo.setToolTip("Axis of the cutting plane")
        top.addWidget(QLabel("Field"))
        top.addWidget(self.field_combo, 1)
        top.addWidget(QLabel("Cut"))
        top.addWidget(self.axis_combo)
        top.addWidget(self.reset_btn)
        outer.addLayout(top)

        bottom = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(1, 99)
        self.slice_slider.setValue(50)
        self.slice_slider.setMinimumWidth(220)
        self.slice_slider.setToolTip("Move the cutting plane through the domain")
        self.slice_slider.valueChanged.connect(self.render)
        self.position_label = QLabel("--")
        self.position_label.setMinimumWidth(64)
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(5, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.setMinimumWidth(120)
        self.opacity_slider.setToolTip("Transparency of the volume")
        self.opacity_slider.valueChanged.connect(self.render)
        self.opacity_label = QLabel("80%")
        self.opacity_label.setMinimumWidth(40)
        bottom.addWidget(QLabel("Position"))
        bottom.addWidget(self.slice_slider, 3)
        bottom.addWidget(self.position_label)
        bottom.addWidget(QLabel("Opacity"))
        bottom.addWidget(self.opacity_slider, 2)
        bottom.addWidget(self.opacity_label)
        outer.addLayout(bottom)

        self.controls = box
        return box

    def _reset_camera(self) -> None:
        if self.plotter is not None:
            self.plotter.reset_camera()
            self.plotter.render()

    # ------------------------------------------------------------------ scene
    def render(self) -> None:
        mesh = self._mesh
        if self.plotter is None:
            return
        axis = self.axis_combo.currentData()
        fraction = min(max(self.slice_slider.value() / 100.0, 0.02), 0.98)
        opacity = self.opacity_slider.value() / 100.0
        self.plotter.clear()
        if self._mode == "geometry" and self._battery is not None:
            length = (domain_extent(mesh) if mesh else
                      (self._battery.cylinder.r_shell * 2 + 1,
                       self._battery.cylinder.r_shell * 2 + 1,
                       self._battery.cylinder.z_cone_apex + 0.5))
            position = fraction * length[_AXES[axis][0]]
            add_geometry_preview(self.plotter, self._battery, mesh,
                                 clip=(axis, position),
                                 opacity_scale=opacity / 0.8, network=self._network)
            self.plotter.add_text(f"geometry preview  |  cut {axis} at {position:.2f} m",
                                  position="upper_left", font_size=10)
        elif mesh is not None:
            field = self.field_combo.currentData()
            actor = None
            try:
                actor = add_field(self.plotter, mesh, field, cmap="coolwarm",
                                  opacity=opacity, axis=axis, fraction=fraction)
            except Exception as exc:                  # never take the app down
                self.plotter.add_text(f"cannot display {field}: {type(exc).__name__}: {exc}",
                                      position="upper_left", font_size=9, color="red")
            if actor is None:
                self.plotter.add_text(f"nothing to show for {field} at this cut",
                                      position="upper_left", font_size=9)
            length = domain_extent(mesh)[_AXES[axis][0]]
            self.plotter.add_text(f"{field}  |  cut {axis} at "
                                  f"{fraction * length:.2f} m",
                                  position="upper_left", font_size=10)
            add_material_legend(self.plotter)
        self.plotter.add_axes()
        self.position_label.setText(f"{fraction * self._domain_length(axis):.2f} m")
        self.opacity_label.setText(f"{self.opacity_slider.value()}%")
        self.plotter.render()

    def _domain_length(self, axis: str) -> float:
        mesh = self._mesh
        if mesh is not None:
            return domain_extent(mesh)[_AXES[axis][0]]
        if self._battery is not None:
            cyl = self._battery.cylinder
            return (cyl.r_shell * 2 + 1, cyl.r_shell * 2 + 1, cyl.z_cone_apex + 0.5
                    )[_AXES[axis][0]]
        return 1.0

    # -------------------------------------------------------------- preview
    def show_mesh(self, mesh, field: str | None = None) -> None:
        self._mesh = mesh
        self._mode = "field"
        if field:
            index = self.field_combo.findData(field)
            if index >= 0:
                self.field_combo.setCurrentIndex(index)
        self.render()

    def show_geometry(self, battery, mesh=None, network=None,
                      reset_camera: bool = True) -> None:
        """Schematic preview; the cut and opacity controls apply to it too.

        ``network`` is the pipe network the Pipes tab built and painted: the preview
        draws its circuit, which is the heat-transfer surface of the whole plant.
        Passing ``None`` on purpose clears it (a rebuilt mesh invalidates the old one);
        leaving it out keeps whatever the last preview had.
        """
        self._battery = battery
        if network is not None or mesh is not None:
            self._network = network
        self._mode = "geometry"
        if mesh is not None:
            self._mesh = mesh
        if self.plotter is not None and reset_camera:
            # an edit redraws in place: the camera stays where the user left it
            self.plotter.reset_camera()
        self.render()
