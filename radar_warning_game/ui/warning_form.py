"""Warning issuance + revision dialog (plan §5, §7).

Player workflow:
  1. Press hotkey / button → polygon editor enters draw mode on the radar panel.
  2. Click vertices on the polygon; double-click or Enter to finish.
  3. This dialog opens to pick warning type, duration, and required magnitudes.
  4. Click "Issue" → form returns the parameters; caller creates the Warning.

For revising an existing warning, construct with ``existing=<Warning>`` to
pre-populate the fields with the warning's current revision and skip polygon
drawing — only fields change.
"""

from __future__ import annotations

from datetime import timedelta

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ..verification.reports_in_poly import Magnitudes, Warning
from ..verification.tornado_tiers import (
    PDS_TOR_MIN_EF,
    TORE_MIN_EF,
    WarningType,
)

DEFAULT_DURATION_MIN = 30
DURATION_PRESETS_MIN = (30, 45, 60)


class WarningFormDialog(QDialog):
    """Modal dialog for issuing or revising a warning.

    Use :meth:`get_parameters` after ``exec()`` returns ``Accepted``:

    .. code-block:: python

        dlg = WarningFormDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            params = dlg.get_parameters()
            session.issue_warning(player_id=me, **params)
    """

    def __init__(
        self,
        *,
        existing: Warning | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Revise Warning" if existing else "Issue Warning")
        self.setModal(True)
        self._existing = existing

        # Type picker
        self._type_combo = QComboBox(self)
        for wt in WarningType:
            self._type_combo.addItem(wt.value, wt)
        self._type_combo.currentIndexChanged.connect(self._on_type_changed)

        # Duration
        self._duration_spin = QSpinBox(self)
        self._duration_spin.setRange(5, 240)
        self._duration_spin.setSuffix(" min")
        self._duration_spin.setValue(DEFAULT_DURATION_MIN)

        # Magnitudes
        self._hail_spin = QDoubleSpinBox(self)
        self._hail_spin.setRange(0.0, 8.0)
        self._hail_spin.setSingleStep(0.25)
        self._hail_spin.setDecimals(2)
        self._hail_spin.setSuffix(" in")
        self._hail_spin.setValue(1.0)

        self._wind_spin = QSpinBox(self)
        self._wind_spin.setRange(50, 150)
        self._wind_spin.setSuffix(" mph")
        self._wind_spin.setValue(60)

        self._ef_spin = QSpinBox(self)
        self._ef_spin.setRange(-1, 5)
        self._ef_spin.setPrefix("EF ")
        self._ef_spin.setSpecialValueText("(radar-indicated)")
        self._ef_spin.setValue(-1)

        # Labels for context (hint of tier requirements)
        self._tier_hint = QLabel(self)
        self._tier_hint.setWordWrap(True)
        self._tier_hint.setStyleSheet("color: #aaa; font-size: 10pt;")

        # Layout
        form = QFormLayout()
        form.addRow("Warning type:", self._type_combo)
        form.addRow("Duration:", self._duration_spin)
        form.addRow("Expected hail (SVR family):", self._hail_spin)
        form.addRow("Expected wind gust (SVR family):", self._wind_spin)
        form.addRow("Expected EF (TOR family):", self._ef_spin)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Revise" if existing else "Issue"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._tier_hint)
        layout.addWidget(buttons)

        # Pre-fill from existing warning if present
        if existing is not None:
            cur = existing.current_revision
            idx = self._type_combo.findData(cur.warning_type)
            if idx >= 0:
                self._type_combo.setCurrentIndex(idx)
            self._duration_spin.setValue(int(cur.duration.total_seconds() // 60))
            if cur.magnitudes.hail_in is not None:
                self._hail_spin.setValue(cur.magnitudes.hail_in)
            if cur.magnitudes.wind_mph is not None:
                self._wind_spin.setValue(int(cur.magnitudes.wind_mph))
            if cur.magnitudes.ef is not None:
                self._ef_spin.setValue(int(cur.magnitudes.ef))

        self._on_type_changed()

    # ---- behavior ------------------------------------------------------

    def _on_type_changed(self) -> None:
        wt = self.selected_type()
        # Toggle field enabled states based on warning family
        is_severe = wt.is_severe_family
        is_tornado = wt.is_tornado_family
        self._hail_spin.setEnabled(is_severe)
        self._wind_spin.setEnabled(is_severe)
        self._ef_spin.setEnabled(is_tornado)
        # Tier hint
        if wt == WarningType.SVR:
            self._tier_hint.setText("SVR — hail ≥1.0\" OR wind ≥58 mph verifies.")
        elif wt == WarningType.SVRC:
            self._tier_hint.setText(
                "SVRC — bonus if peak hail ≥1.75\" or wind ≥70 mph; else scored as plain SVR."
            )
        elif wt == WarningType.SVRD:
            self._tier_hint.setText(
                "SVRD — bonus if peak hail ≥2.75\" or wind ≥80 mph; FA penalty ×1.5."
            )
        elif wt == WarningType.TOR:
            self._tier_hint.setText("TOR — any tornado report verifies. EF estimate optional.")
        elif wt == WarningType.TORR:
            self._tier_hint.setText(
                "TORR — 1.10× verified bonus. Late-warn POD allowed (≤10 min after report)."
            )
        elif wt == WarningType.PDS_TOR:
            self._tier_hint.setText(
                f"PDS TOR — 1.75× bonus only if verified by EF≥{PDS_TOR_MIN_EF} OR casualties; "
                "no bonus for weak verification. FA penalty ×1.5."
            )
        elif wt == WarningType.TORE:
            self._tier_hint.setText(
                f"TORE — 2.5× bonus if EF≥{TORE_MIN_EF}+/casualties; 0.75× if weak (over-issuance). "
                "FA penalty ×3.0 (heaviest in the game)."
            )

    # ---- output --------------------------------------------------------

    def selected_type(self) -> WarningType:
        return self._type_combo.currentData()

    def get_parameters(self) -> dict:
        """Return kwargs ready to pass to :meth:`GameSession.issue_warning`.

        Includes ``warning_type``, ``duration``, ``magnitudes``. Does NOT include
        ``polygon`` or ``player_id`` — those come from the caller's context.
        """
        wt = self.selected_type()
        hail = self._hail_spin.value() if wt.is_severe_family else None
        wind = float(self._wind_spin.value()) if wt.is_severe_family else None
        ef = (
            None if not wt.is_tornado_family
            else (None if self._ef_spin.value() < 0 else float(self._ef_spin.value()))
        )
        return {
            "warning_type": wt,
            "duration": timedelta(minutes=self._duration_spin.value()),
            "magnitudes": Magnitudes(hail_in=hail, wind_mph=wind, ef=ef),
        }
