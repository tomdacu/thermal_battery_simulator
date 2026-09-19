"""Small widget factories shared by the GUI panels.

Every combo stores its machine value in ``itemData``, so no panel has to map
translated label strings back to numbers - the failure mode that silently pinned
tolerance/precision/thread selectors to their defaults.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)


def double_spin(value: float, lo: float, hi: float, step: float = 1.0, decimals: int = 2,
                suffix: str = "", tooltip: str = "", on_change: Callable = None,
                special: str = "") -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(lo, hi)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    if suffix:
        box.setSuffix(suffix)
    if special:                       # text shown at the minimum, e.g. "auto"
        box.setSpecialValueText(special)
    if tooltip:
        box.setToolTip(tooltip)
    if on_change is not None:
        box.valueChanged.connect(on_change)
    return box


def int_spin(value: int, lo: int, hi: int, step: int = 1, tooltip: str = "",
             on_change: Callable = None) -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    if tooltip:
        box.setToolTip(tooltip)
    if on_change is not None:
        box.valueChanged.connect(on_change)
    return box


def combo(items: Sequence[tuple[str, object]], index: int = 0,
          on_change: Callable = None) -> QComboBox:
    """Combo whose ``currentData()`` is the machine value of the entry."""
    box = QComboBox()
    for label, data in items:
        box.addItem(label, data)
    box.setCurrentIndex(min(max(index, 0), box.count() - 1))
    if on_change is not None:
        box.currentIndexChanged.connect(on_change)
    return box


def check(label: str, checked: bool = False, tooltip: str = "",
          on_toggle: Callable = None) -> QCheckBox:
    box = QCheckBox(label)
    box.setChecked(checked)
    if tooltip:
        box.setToolTip(tooltip)
    if on_toggle is not None:
        box.toggled.connect(on_toggle)
    return box


def button(label: str, on_click: Callable = None, tooltip: str = "") -> QPushButton:
    btn = QPushButton(label)
    if on_click is not None:
        btn.clicked.connect(on_click)
    if tooltip:
        btn.setToolTip(tooltip)
    return btn


def hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #666; font-size: 11px;")
    return label


class FormPanel(QWidget):
    """QWidget with a QFormLayout plus helpers to add rows and keep accessors."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.fields = {}

    def add(self, name: str, widget: QWidget, label: str = None) -> QWidget:
        self.form.addRow(label if label is not None else name, widget)
        self.fields[name] = widget
        return widget

    def add_row(self, widget: QWidget) -> QWidget:
        self.form.addRow(widget)
        return widget

    def add_hint(self, text: str) -> QLabel:
        return self.add_row(hint(text))

    def value(self, name: str):
        widget = self.fields[name]
        if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            data = widget.currentData()
            return widget.currentText() if data is None else data
        if isinstance(widget, QCheckBox):
            return widget.isChecked()
        raise TypeError(f"cannot read value of {type(widget).__name__}")


def set_bold(label: QLabel) -> QLabel:
    label.setStyleSheet("font-weight: 600;")
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label


def grid_of(widgets: Iterable[QWidget], columns: int = 2) -> QWidget:
    """Put widgets in a simple grid (used by compact sub-tabs)."""
    from PyQt6.QtWidgets import QGridLayout

    holder = QWidget()
    grid = QGridLayout(holder)
    for i, widget in enumerate(widgets):
        grid.addWidget(widget, i // columns, i % columns)
    return holder
