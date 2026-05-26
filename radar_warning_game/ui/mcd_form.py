"""MCD issuance dialog with three Peak Intensity Bin pickers (plan §8).

Player workflow:
  1. Press hotkey / button → polygon editor enters draw mode on the radar panel
     (or on a zoomed CONUS subset).
  2. Click vertices, close the polygon.
  3. This dialog opens to pick duration + tornado/wind/hail PIBs.
  4. Click "Issue MCD" → caller creates the MCD via :class:`GameSession`.

Each PIB picker is a dropdown that shows ``"PIB N - <descriptor>"`` so the
forecaster can compare to SPC's published phrasing.
"""

from __future__ import annotations

from datetime import timedelta

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..verification.pibs import HAIL_PIBS, TORNADO_PIBS, WIND_PIBS, PIBSpec

DEFAULT_DURATION_MIN = 90


class MCDFormDialog(QDialog):
    """Modal dialog for issuing an MCD."""

    def __init__(self, *, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Issue MCD")
        self.setModal(True)

        # Duration
        self._duration_spin = QSpinBox(self)
        self._duration_spin.setRange(30, 240)
        self._duration_spin.setSingleStep(15)
        self._duration_spin.setSuffix(" min")
        self._duration_spin.setValue(DEFAULT_DURATION_MIN)

        # PIB pickers — one per hazard
        self._tor_combo = _pib_combo(self, TORNADO_PIBS, "Tornado")
        self._wnd_combo = _pib_combo(self, WIND_PIBS, "Wind")
        self._hal_combo = _pib_combo(self, HAIL_PIBS, "Hail")

        # Form
        form = QFormLayout()
        form.addRow("Duration:", self._duration_spin)
        form.addRow("Tornado PIB:", self._tor_combo)
        form.addRow("Wind PIB:", self._wnd_combo)
        form.addRow("Hail PIB:", self._hal_combo)

        # Hint
        hint = QLabel(
            "At least one hazard must be ≥ PIB 1 (an all-None MCD is rejected). "
            "Scoring: PIB accuracy per hazard + lead-time bonus + multi-hazard breadth.",
            self,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa; font-size: 10pt;")

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Issue MCD")
        buttons.accepted.connect(self._validate_and_accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(hint)
        self._error_label = QLabel("", self)
        self._error_label.setStyleSheet("color: #ff6b6b;")
        layout.addWidget(self._error_label)
        layout.addWidget(buttons)

    def _validate_and_accept(self) -> None:
        # Anti-spam: require at least one non-None PIB (plan §8)
        if (
            self._tor_combo.currentData() == 0
            and self._wnd_combo.currentData() == 0
            and self._hal_combo.currentData() == 0
        ):
            self._error_label.setText("Pick at least one hazard PIB ≥ 1.")
            return
        self.accept()

    def get_parameters(self) -> dict:
        """Returns kwargs ready for :meth:`GameSession.issue_mcd`."""
        return {
            "duration": timedelta(minutes=self._duration_spin.value()),
            "pib_tornado": int(self._tor_combo.currentData()),
            "pib_wind": int(self._wnd_combo.currentData()),
            "pib_hail": int(self._hal_combo.currentData()),
        }


def _pib_combo(parent: QWidget, table: tuple[PIBSpec, ...], label: str) -> QComboBox:
    combo = QComboBox(parent)
    combo.addItem("None (no expected hazard)", 0)
    for spec in table:
        combo.addItem(f"PIB {spec.pib} — {spec.descriptor}  [{spec.ibw_tag}]", spec.pib)
    return combo
