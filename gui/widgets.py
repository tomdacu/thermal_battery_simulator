"""Small widget factories shared by the GUI panels.

Every combo stores its machine value in ``itemData``, so no panel has to map
translated label strings back to numbers - the failure mode that silently pinned
tolerance/precision/thread selectors to their defaults.

Explanations never take room in a form: a control's explanation is its tooltip, and the
row's label carries an "ⓘ" when there is one; a section's explanation is one small "ⓘ"
line whose tooltip holds the text.  Tooltips are rich text, so Qt wraps them instead of
drawing one line wider than the screen.
"""
from __future__ import annotations

import html
from collections.abc import Callable, Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

#: the mark a label carries when its control has an explanation
INFO = "ⓘ"


def rich(text: str) -> str:
    """A tooltip Qt wraps: rich text, the plain text escaped."""
    if not text or text.lstrip().startswith("<"):
        return text
    return f"<p style='max-width: 420px'>{html.escape(text)}</p>"


def _tip(widget: QWidget, tooltip: str) -> None:
    if tooltip:
        widget.setToolTip(rich(tooltip))


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
    _tip(box, tooltip)
    if on_change is not None:
        box.valueChanged.connect(on_change)
    return box


def int_spin(value: int, lo: int, hi: int, step: int = 1, tooltip: str = "",
             on_change: Callable = None) -> QSpinBox:
    box = QSpinBox()
    box.setRange(lo, hi)
    box.setSingleStep(step)
    box.setValue(value)
    _tip(box, tooltip)
    if on_change is not None:
        box.valueChanged.connect(on_change)
    return box


def combo(items: Sequence[tuple[str, object]], index: int = 0,
          on_change: Callable = None, tooltip: str = "") -> QComboBox:
    """Combo whose ``currentData()`` is the machine value of the entry."""
    box = QComboBox()
    for label, data in items:
        box.addItem(label, data)
    box.setCurrentIndex(min(max(index, 0), box.count() - 1))
    _tip(box, tooltip)
    if on_change is not None:
        box.currentIndexChanged.connect(on_change)
    return box


def check(label: str, checked: bool = False, tooltip: str = "",
          on_toggle: Callable = None) -> QCheckBox:
    box = QCheckBox(label)
    box.setChecked(checked)
    _tip(box, tooltip)
    if on_toggle is not None:
        box.toggled.connect(on_toggle)
    return box


def button(label: str, on_click: Callable = None, tooltip: str = "") -> QPushButton:
    btn = QPushButton(label)
    if on_click is not None:
        btn.clicked.connect(on_click)
    _tip(btn, tooltip)
    return btn


def hint(text: str) -> QLabel:
    """A read-out: wrapped, selectable, small - it grows downwards, never sideways."""
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    label.setStyleSheet("color: #555; font-size: 11px;")
    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
    return label


def info(text: str, caption: str = "about this section") -> QLabel:
    """One small line whose tooltip holds an explanation."""
    label = QLabel(f"{INFO}  {caption}")
    label.setStyleSheet("color: #2a6fb0; font-size: 11px;")
    label.setToolTip(rich(text))
    label.setCursor(Qt.CursorShape.WhatsThisCursor)
    return label


def scrollable(widget: QWidget) -> QScrollArea:
    """A page that scrolls vertically instead of being cut by a short screen."""
    holder = QWidget()
    column = QVBoxLayout(holder)
    column.setContentsMargins(0, 0, 0, 0)
    column.addWidget(widget)
    column.addStretch(1)                  # the page packs at the top, never stretched
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(holder)
    return area


class FormPanel(QWidget):
    """QWidget with a QFormLayout plus helpers to add rows and keep accessors."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.form = QFormLayout(self)
        self.form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.fields = {}

    def add(self, name: str, widget: QWidget, label: str = None) -> QWidget:
        """A labelled row; the label shows "ⓘ" and the explanation when the control has one."""
        text = label if label is not None else name
        caption = QLabel(text)
        tooltip = widget.toolTip()
        if tooltip:
            caption.setText(f"{text} {INFO}")
            caption.setToolTip(tooltip)
        self.form.addRow(caption, widget)
        self.fields[name] = widget
        return widget

    def add_row(self, widget: QWidget) -> QWidget:
        self.form.addRow(widget)
        return widget

    def add_hint(self, text: str, caption: str = "about this section") -> QLabel:
        return self.add_row(info(text, caption))

    def section(self, title: str, explanation: str = "") -> FormPanel:
        """A titled group of rows inside this form; returns the group's own form."""
        box = QGroupBox(title)
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 4)
        inner = FormPanel()
        inner.form.setContentsMargins(0, 0, 0, 0)
        if explanation:
            inner.add_hint(explanation)
        layout.addWidget(inner)
        self.form.addRow(box)
        return inner

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
