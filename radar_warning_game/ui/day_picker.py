"""Round-start day picker dialog (plan §1).

First screen shown to the host: pick a random day with thresholds, or pick a
specific date. Random-day mode keeps the date hidden from the host so they can
also play the round.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..game.round_builder import DATE_RANGE_START, ThresholdSpec


class DayPickerDialog(QDialog):
    """Modal at app start. ``exec()`` then :meth:`get_choice` to read."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Pick a day")
        self.setModal(True)

        self._random_radio = QRadioButton("Random day (hidden) with thresholds:", self)
        self._random_radio.setChecked(True)
        self._random_radio.toggled.connect(self._refresh_enabled)
        self._specific_radio = QRadioButton("Specific date (host will see it):", self)
        self._specific_radio.toggled.connect(self._refresh_enabled)
        self._live_radio = QRadioButton(
            "Live (now) — play real-time against current weather", self,
        )
        self._live_radio.toggled.connect(self._refresh_enabled)

        # Thresholds for random mode
        self._tor_spin = QSpinBox(self); self._tor_spin.setRange(0, 200); self._tor_spin.setValue(5)
        self._hail_spin = QSpinBox(self); self._hail_spin.setRange(0, 500); self._hail_spin.setValue(20)
        self._wind_spin = QSpinBox(self); self._wind_spin.setRange(0, 500); self._wind_spin.setValue(20)

        # Specific date
        self._date_edit = QDateEdit(self)
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setMinimumDate(date(DATE_RANGE_START.year, DATE_RANGE_START.month, DATE_RANGE_START.day))
        from datetime import date as _date
        self._date_edit.setMaximumDate(_date.today())
        self._date_edit.setDate(_date(2013, 5, 20))

        # Replay toggle
        self._save_replay = QCheckBox("Save replay file", self)

        # Team mode toggle
        self._team_mode = QCheckBox("Enable team mode (pre-round team lobby)", self)

        # Layout
        form = QFormLayout()
        form.addRow("Min tornado reports:", self._tor_spin)
        form.addRow("Min hail reports (≥1.0\"):", self._hail_spin)
        form.addRow("Min wind reports (≥58 mph):", self._wind_spin)
        form.addRow("Specific date:", self._date_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._random_radio)
        layout.addLayout(form)
        layout.addSpacing(8)
        layout.addWidget(self._specific_radio)
        layout.addSpacing(8)
        layout.addWidget(self._live_radio)
        layout.addSpacing(12)
        layout.addWidget(self._save_replay)
        layout.addWidget(self._team_mode)
        layout.addWidget(buttons)

        self._refresh_enabled()

    def _refresh_enabled(self) -> None:
        is_random = self._random_radio.isChecked()
        is_specific = self._specific_radio.isChecked()
        self._tor_spin.setEnabled(is_random)
        self._hail_spin.setEnabled(is_random)
        self._wind_spin.setEnabled(is_random)
        self._date_edit.setEnabled(is_specific)

    def is_live(self) -> bool:
        return self._live_radio.isChecked()

    # ---- output --------------------------------------------------------

    def is_random(self) -> bool:
        return self._random_radio.isChecked()

    def thresholds(self) -> ThresholdSpec:
        return ThresholdSpec(
            min_tornadoes=self._tor_spin.value(),
            min_hail=self._hail_spin.value(),
            min_wind=self._wind_spin.value(),
        )

    def specific_date_12z(self) -> datetime:
        d = self._date_edit.date().toPyDate()
        return datetime(d.year, d.month, d.day, 12, tzinfo=timezone.utc)

    def save_replay(self) -> bool:
        return self._save_replay.isChecked()

    def team_mode(self) -> bool:
        return self._team_mode.isChecked()
