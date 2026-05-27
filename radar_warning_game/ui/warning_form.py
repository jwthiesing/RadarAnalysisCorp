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

from datetime import datetime, timedelta

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
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
    SVRC_HAIL_THRESHOLD_IN,
    SVRC_WIND_THRESHOLD_MPH,
    SVRD_HAIL_THRESHOLD_IN,
    SVRD_WIND_THRESHOLD_MPH,
    TORE_MIN_EF,
    WarningType,
)

DEFAULT_DURATION_MIN = 30
DURATION_PRESETS_MIN = (30, 45, 60)


# Map (family, tag) → concrete WarningType. The UI shows family and tag in
# two separate dropdowns; the form recombines them at submit time. Keeps
# the underlying enum + scoring code untouched.
_SEVERE_TAGS: tuple[tuple[str, str, WarningType], ...] = (
    ("base", "Base", WarningType.SVR),
    ("considerable",
     f"Considerable — verifies bonus at hail ≥{SVRC_HAIL_THRESHOLD_IN:g} in "
     f"or wind ≥{int(SVRC_WIND_THRESHOLD_MPH)} mph",
     WarningType.SVRC),
    ("destructive",
     f"Destructive — verifies bonus at hail ≥{SVRD_HAIL_THRESHOLD_IN:g} in "
     f"or wind ≥{int(SVRD_WIND_THRESHOLD_MPH)} mph",
     WarningType.SVRD),
)

_TORNADO_TAGS: tuple[tuple[str, str, WarningType], ...] = (
    ("base", "Base", WarningType.TOR),
    ("radar_confirmed",
     "Radar-confirmed (TORR) — small bonus, late-warn POD allowed",
     WarningType.TORR),
    ("pds",
     f"PDS — bonus at EF ≥{PDS_TOR_MIN_EF} or casualties",
     WarningType.PDS_TOR),
    ("emergency",
     f"Tornado Emergency — large bonus at EF ≥{TORE_MIN_EF} + casualties",
     WarningType.TORE),
)


def _split_warning_type(wt: WarningType) -> tuple[str, str]:
    """Inverse of the (family, tag) → WarningType map. Used to pre-fill the
    dropdowns when editing an existing warning."""
    for family, tags in (("severe", _SEVERE_TAGS), ("tornado", _TORNADO_TAGS)):
        for tag, _label, mapped in tags:
            if mapped == wt:
                return family, tag
    # Fallback (shouldn't happen) — treat unknown as base severe.
    return "severe", "base"


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
        now: "datetime | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Revise Warning" if existing else "Issue Warning")
        self.setModal(True)
        self._existing = existing
        # ``now`` is the current game-clock time, used when revising so
        # the duration field can be prefilled with "minutes remaining
        # until the current expiry." Without it, the old prefill of
        # ``current_revision.duration`` had the duration meaning
        # "from-issue" while ``revise_warning`` interprets the new
        # revision's duration as "from-revision-time" — so a
        # no-change revise actually extended the warning by the
        # elapsed time since issuance. Passing ``now`` lets us
        # prefill so "no change" really is no change.
        self._now = now
        self._cancel_requested = False

        # Family picker — base warning category (Severe Thunderstorm or
        # Tornado). The IBW "tag" (Considerable, Destructive, PDS, Emergency,
        # …) is selected in a separate dropdown so the tier overlay is
        # decoupled from the underlying warning type.
        self._family_combo = QComboBox(self)
        self._family_combo.addItem("Severe Thunderstorm (SVR)", "severe")
        self._family_combo.addItem("Tornado (TOR)", "tornado")
        self._family_combo.currentIndexChanged.connect(self._on_family_changed)

        # Tag picker — populated from the active family. Each entry shows
        # the numeric thresholds the tag's bonus is tied to so the
        # forecaster picks a hazard level, not a label.
        self._tag_combo = QComboBox(self)
        self._tag_combo.currentIndexChanged.connect(self._on_type_changed)

        # Duration
        self._duration_spin = QSpinBox(self)
        self._duration_spin.setRange(5, 240)
        self._duration_spin.setSuffix(" min")
        self._duration_spin.setValue(DEFAULT_DURATION_MIN)

        # Magnitudes. Each spinbox carries a sentinel one step below its
        # real-valued range that renders as "(no hail tag)" / "(no wind
        # tag)" — picking it omits the prediction from the magnitudes
        # record and removes that hazard from magnitude scoring.
        self._hail_spin = QDoubleSpinBox(self)
        self._hail_spin.setRange(-0.25, 8.0)
        self._hail_spin.setSingleStep(0.25)
        self._hail_spin.setDecimals(2)
        self._hail_spin.setSuffix(" in")
        self._hail_spin.setSpecialValueText("(no hail tag)")
        self._hail_spin.setValue(1.0)

        self._wind_spin = QSpinBox(self)
        self._wind_spin.setRange(49, 150)
        self._wind_spin.setSuffix(" mph")
        self._wind_spin.setSpecialValueText("(no wind tag)")
        self._wind_spin.setValue(60)

        # "Tornado Possible" IBW tag — SVR-only. When the SVR catches a
        # tornado inside its footprint, the player earns a flat bonus
        # (see scoring.SVR_TORNADO_POSSIBLE_BONUS). Disabled on TOR family.
        self._tornado_possible_check = QCheckBox(
            "Tornado possible (IBW tag)", self,
        )
        self._tornado_possible_check.setToolTip(
            "NWS IBW 'Tornado Possible' tag for SVR warnings.\n"
            "Earns a flat bonus if a tornado occurs inside the polygon "
            "during the warning's valid time."
        )

        # Labels for context (hint of tier requirements)
        self._tier_hint = QLabel(self)
        self._tier_hint.setWordWrap(True)
        self._tier_hint.setStyleSheet("color: #aaa; font-size: 10pt;")

        # Layout
        form = QFormLayout()
        form.addRow("Warning type:", self._family_combo)
        form.addRow("Tag:", self._tag_combo)
        form.addRow("Duration:", self._duration_spin)
        form.addRow("Expected hail:", self._hail_spin)
        form.addRow("Expected wind gust (SVR family):", self._wind_spin)
        form.addRow("", self._tornado_possible_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Revise" if existing else "Issue"
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        # Cancel-warning button (revise mode only) — issues a
        # WarningCancel rather than a revision. Surfaced via the
        # dialog's ``cancel_requested()`` flag so the caller can
        # route to the right session method.
        self._cancel_warning_btn: "QPushButton | None" = None
        if existing is not None:
            from PyQt6.QtWidgets import QPushButton
            self._cancel_warning_btn = QPushButton("Cancel This Warning", self)
            self._cancel_warning_btn.setStyleSheet(
                "QPushButton { color: #ff6060; font-weight: bold; }"
                " QPushButton:hover { background-color: rgba(255, 60, 60, 0.10); }"
            )
            self._cancel_warning_btn.setToolTip(
                "Permanently cancel this warning at the current game time. "
                "No reports after the cancel time will verify it. Cannot be undone."
            )
            self._cancel_warning_btn.clicked.connect(self._on_cancel_warning_clicked)
            # Slot into the QDialogButtonBox so it sits inline with
            # the Revise / Cancel(close-dialog) pair.
            buttons.addButton(
                self._cancel_warning_btn,
                QDialogButtonBox.ButtonRole.DestructiveRole,
            )

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self._tier_hint)
        layout.addWidget(buttons)

        # Populate the tag dropdown from whichever family is initially active.
        self._on_family_changed()

        # Pre-fill from existing warning if present
        if existing is not None:
            cur = existing.current_revision
            family, tag = _split_warning_type(cur.warning_type)
            f_idx = self._family_combo.findData(family)
            if f_idx >= 0:
                self._family_combo.setCurrentIndex(f_idx)
                self._on_family_changed()
            t_idx = self._tag_combo.findData(tag)
            if t_idx >= 0:
                self._tag_combo.setCurrentIndex(t_idx)
            # Duration prefill: minutes from "now" (game-clock time)
            # until the warning's current expiry. ``session.revise_warning``
            # uses the new revision's duration as the time-from-revision-
            # until-expiry, and stamps the revision_time at the current
            # clock — so this prefill makes a no-change revise preserve
            # the existing expiry. Falls back to the raw current-
            # revision duration when ``now`` wasn't provided (e.g.
            # tests construct the dialog directly without the clock).
            if self._now is not None:
                remaining = (existing.end_time() - self._now).total_seconds() / 60.0
                # Clamp to the spinbox's range; a negative remaining
                # (already-expired warning) is a degenerate case but
                # we let the user pick a new positive duration to
                # effectively re-activate it.
                prefilled = max(
                    self._duration_spin.minimum(),
                    min(self._duration_spin.maximum(), int(round(remaining))),
                )
                self._duration_spin.setValue(prefilled)
            else:
                self._duration_spin.setValue(int(cur.duration.total_seconds() // 60))
            # None → land on the "(no hail tag)" / "(no wind tag)" sentinel
            # so editing an existing warning shows what it really claims.
            if cur.magnitudes.hail_in is not None:
                self._hail_spin.setValue(cur.magnitudes.hail_in)
            else:
                self._hail_spin.setValue(self._hail_spin.minimum())
            if cur.magnitudes.wind_mph is not None:
                self._wind_spin.setValue(int(cur.magnitudes.wind_mph))
            else:
                self._wind_spin.setValue(self._wind_spin.minimum())
            self._tornado_possible_check.setChecked(
                bool(cur.magnitudes.tornado_possible)
            )

        self._on_type_changed()

    # ---- behavior ------------------------------------------------------

    def _on_family_changed(self) -> None:
        """Rebuild the tag dropdown for the newly-selected warning family."""
        family = self._family_combo.currentData()
        tags = _SEVERE_TAGS if family == "severe" else _TORNADO_TAGS
        self._tag_combo.blockSignals(True)
        self._tag_combo.clear()
        for tag, label, _wt in tags:
            self._tag_combo.addItem(label, tag)
        self._tag_combo.setCurrentIndex(0)
        self._tag_combo.blockSignals(False)
        self._on_type_changed()

    def _on_type_changed(self) -> None:
        wt = self.selected_type()
        # Hail is enabled for both families — NWS tornado warnings frequently
        # carry a hail tag for the co-located hail threat. Wind gust stays as
        # SVR-only since tornado warnings rarely advertise straight-line wind
        # separately from the tornado itself. EF is never user-set (it's a
        # post-event damage-survey rating).
        self._wind_spin.setEnabled(wt.is_severe_family)
        # "Tornado Possible" tag only applies to SVR warnings.
        self._tornado_possible_check.setEnabled(wt.is_severe_family)
        if not wt.is_severe_family:
            self._tornado_possible_check.setChecked(False)
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
        """Recombine the family + tag dropdowns into the concrete WarningType."""
        family = self._family_combo.currentData()
        tag = self._tag_combo.currentData()
        tags = _SEVERE_TAGS if family == "severe" else _TORNADO_TAGS
        for entry_tag, _label, wt in tags:
            if entry_tag == tag:
                return wt
        # Fall back to the base type for the chosen family.
        return tags[0][2]

    def _on_cancel_warning_clicked(self) -> None:
        """Cancel-warning button handler — flag the request and accept
        the dialog so the caller's normal Accepted branch can route to
        ``cancel_warning(...)`` instead of ``revise_warning(...)``."""
        self._cancel_requested = True
        self.accept()

    def cancel_requested(self) -> bool:
        """``True`` iff the user clicked 'Cancel This Warning' rather
        than the Revise / Issue button. Caller branches on this after
        ``exec()`` returns Accepted."""
        return self._cancel_requested

    def get_parameters(self) -> dict:
        """Return kwargs ready to pass to :meth:`GameSession.issue_warning`.

        Includes ``warning_type``, ``duration``, ``magnitudes``. Does NOT include
        ``polygon`` or ``player_id`` — those come from the caller's context.
        """
        wt = self.selected_type()
        # Each spinbox's minimum is a "no tag" sentinel — pick it to omit
        # the prediction. The sentinel is one step below the real-valued
        # range; comparing against the spinbox's minimum() avoids
        # hard-coding the magic value here.
        hail_value = self._hail_spin.value()
        hail = None if hail_value <= self._hail_spin.minimum() else hail_value
        wind: float | None
        if wt.is_severe_family:
            wind_value = self._wind_spin.value()
            wind = None if wind_value <= self._wind_spin.minimum() else float(wind_value)
        else:
            wind = None
        # "Tornado Possible" applies only to SVR warnings — discard the
        # flag if the user picked it then switched to TOR.
        tornado_possible = (
            wt.is_severe_family and self._tornado_possible_check.isChecked()
        )
        return {
            "warning_type": wt,
            "duration": timedelta(minutes=self._duration_spin.value()),
            "magnitudes": Magnitudes(
                hail_in=hail, wind_mph=wind, ef=None,
                tornado_possible=tornado_possible,
            ),
        }
