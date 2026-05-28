"""Live in-session corner widget + end-of-round full table (plan §9).

The live widget is intentionally compact — it's docked in a corner of the main
window during gameplay so players can glance at the standings without it eating
real estate. It updates on demand (host calls :meth:`refresh` after every score-
changing event: a report's virtual time crossing now, a warning expiring).

The end-of-round table is a separate dialog showing per-team breakdowns and
clickable per-warning details.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..data.reports import Report
from ..game.session import GameSession
from ..geo.polygons import contains_with_buffer
from ..verification.reports_in_poly import (
    DEFAULT_VERIFICATION_BUFFER_KM,
    Warning,
)
from ..verification.scoring import TeamScore
from .colors import color_for_team
from .recap_map import RecapMap
from .time_format import format_player_offset, format_player_time_short


class LiveLeaderboardWidget(QFrame):
    """Compact corner widget showing ranked team scores.

    Two visual modes:
      - **compact**: name + score only (default)
      - **expanded**: name + score + POD% + FAR% + warning count

    Click the title bar to toggle between modes.
    """

    def __init__(self, local_team_id: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("LiveLeaderboardWidget { background: #111; }")
        self.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Maximum)
        self._local_team_id = local_team_id
        self._expanded = False

        self._title = QLabel("Leaderboard  (click to expand)", self)
        self._title.setStyleSheet("color: #ddd; font-weight: bold; padding: 4px;")
        self._title.mousePressEvent = self._toggle_mode  # type: ignore[assignment]

        self._list = QListWidget(self)
        self._list.setStyleSheet("""
            QListWidget { background: #111; color: #ddd; border: none; }
            QListWidget::item { padding: 2px 6px; }
        """)
        self._list.setFrameShape(QFrame.Shape.NoFrame)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        layout.addWidget(self._title)
        layout.addWidget(self._list)

    def refresh(self, scores: list[TeamScore], team_names: dict[str, str] | None = None) -> None:
        """Repopulate from a fresh score snapshot. Pass team_names for display."""
        self._list.clear()
        team_names = team_names or {}
        ranked = sorted(scores, key=lambda s: s.total, reverse=True)
        for rank, s in enumerate(ranked, start=1):
            name = team_names.get(s.team_id, s.team_id)
            color = color_for_team(s.team_id)
            you_marker = "  ←" if s.team_id == self._local_team_id else ""
            if self._expanded:
                text = (
                    f"{rank:>2}. ● {name}   {s.total:+7.1f}   "
                    f"POD {s.pod*100:>3.0f}%  FAR {s.far*100:>3.0f}%  "
                    f"#W {s.n_warnings}{you_marker}"
                )
            else:
                text = f"{rank:>2}. ● {name}   {s.total:+7.1f}{you_marker}"
            item = QListWidgetItem(text)
            item.setForeground(_qcolor(color))
            if s.team_id == self._local_team_id:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self._list.addItem(item)

    def _toggle_mode(self, _event) -> None:
        self._expanded = not self._expanded
        self._title.setText(
            "Leaderboard  (click to collapse)" if self._expanded
            else "Leaderboard  (click to expand)"
        )


_REPORT_TYPE_SYMBOL = {"tornado": "▲", "hail": "●", "wind": "■"}
_REPORT_TYPE_COLOR = {
    "tornado": "#ff4444",
    "hail":    "#44ff66",
    "wind":    "#66bbff",
}

# Display labels for the ticker's "→ TYPES" suffix. ``WarningType``'s
# enum value is the underlying string used in scoring + protocol; this
# remaps the long ones to compact NWS-style aliases that match how
# forecasters refer to them on a busy panel:
#   PDS_TOR  → TORP   (Particularly Dangerous Situation TOR)
#   TORE     → TORE   (Tornado Emergency — kept as-is)
# Anything not in this dict falls through unchanged.
_TICKER_TYPE_LABEL = {
    "PDS_TOR": "TORP",
}


def _format_report_line(
    report: Report,
    covering_warning_types: list[str],
) -> str:
    """Build the single-line HTML for one ticker entry.

    Layout: ``HH:MM Z  ▲  ST/County  EF2`` with a trailing
    ``→ TOR/SVR`` suffix (colored) listing every warning type the local
    team had over that report at its observation time. The trailing
    list is what tells the user "this report was used in scoring for
    these warnings of mine." An empty suffix means the report fell
    outside every active polygon — no scoring credit either way.
    """
    sym = _REPORT_TYPE_SYMBOL.get(report.category, "•")
    sym_color = _REPORT_TYPE_COLOR.get(report.category, "#ccc")
    time_str = format_player_time_short(report.time)
    locality = (report.county or report.state or "").strip()
    if report.category == "tornado":
        if report.magnitude < 0:
            mag_str = "EF?"
        else:
            mag_str = f"EF{int(report.magnitude)}"
    elif report.category == "hail":
        mag_str = f'{report.magnitude:.2f}"'
    elif report.category == "wind":
        mag_str = f"{int(report.magnitude)} mph"
    else:
        mag_str = ""
    head = (
        f"<span style='color:#bbb'>{time_str}</span> &nbsp; "
        f"<span style='color:{sym_color}; font-weight:bold'>{sym}</span> "
        f"<span style='color:#ddd'>{locality}</span> &nbsp; "
        f"<span style='color:#aaa'>{mag_str}</span>"
    )
    if covering_warning_types:
        joined = "/".join(
            _TICKER_TYPE_LABEL.get(t, t) for t in covering_warning_types
        )
        tail = (f" &nbsp; <span style='color:#9be38f'>→ {joined}</span>")
    else:
        tail = " &nbsp; <span style='color:#666'>→ (uncovered)</span>"
    return head + tail


def _covering_warning_types(
    report: Report,
    team_warnings: list[Warning],
    buffer_km: float = DEFAULT_VERIFICATION_BUFFER_KM,
) -> list[str]:
    """Find every warning whose polygon (at the report's time) covers
    the report. Returns the *current-revision* warning-type names in
    insertion order. Skips warnings that were canceled before the
    report and warnings issued after the report.

    This is the inverse of ``find_verifying_reports`` — instead of
    asking "which reports verify this warning?", we ask "which of my
    warnings was covering this report when it happened?". Used to
    annotate the ticker with the scoring-relevant warning types per
    report.
    """
    out: list[str] = []
    seen: set[str] = set()
    for w in team_warnings:
        if report.time < w.original_issue_time:
            continue
        if w.canceled_at is not None and report.time > w.canceled_at:
            continue
        end = w.end_time()
        if report.time > end:
            continue
        rev = w.revision_at(report.time) or w.revisions[0]
        if not contains_with_buffer(
            rev.polygon, report.lat, report.lon, buffer_km=buffer_km,
        ):
            continue
        wt = rev.warning_type.value
        if wt not in seen:
            seen.add(wt)
            out.append(wt)
    return out


class RecentReportsTicker(QFrame):
    """Scrolling list of recent reports with per-report warning coverage.

    Sits beside the live leaderboard. Each row shows
    ``time · symbol · locality · magnitude → covering warning types``.
    The trailing arrow lists every local-team warning that had the
    report inside its polygon at the report's observation time —
    i.e., the warnings whose score this report contributed to.

    Newest reports go on top. The widget is purely a view over the
    session; :meth:`refresh` rebuilds the list from the current state.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.Box)
        self.setStyleSheet("RecentReportsTicker { background: #111; }")
        title = QLabel("Recent reports", self)
        title.setStyleSheet("color: #ddd; font-weight: bold; padding: 4px;")
        self._list = QListWidget(self)
        self._list.setStyleSheet("""
            QListWidget { background: #111; color: #ddd; border: none; }
            QListWidget::item { border-bottom: 1px solid #1a1a1a; }
        """)
        self._list.setWordWrap(True)
        self._list.setFrameShape(QFrame.Shape.NoFrame)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(0)
        layout.addWidget(title)
        layout.addWidget(self._list)

    def refresh(
        self,
        reports: list[Report],
        team_warnings: list[Warning],
        now: object | None = None,
    ) -> None:
        """Repopulate the list. ``reports`` should already be filtered
        to visible-this-tick (i.e., ``report.time <= virtual_time``);
        the widget itself doesn't know the clock."""
        self._list.clear()
        # Newest first so the ticker reads top-down chronologically
        # backwards from "right now".
        for r in sorted(reports, key=lambda r: r.time, reverse=True):
            covering = _covering_warning_types(r, team_warnings)
            html = _format_report_line(r, covering)
            # QListWidgetItem doesn't render HTML directly; wrap each
            # row in a QLabel so the inline color/bold spans paint.
            label = QLabel(html, self)
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setStyleSheet("padding: 4px 8px;")
            label.setAlignment(
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
            )
            # ``QLabel.sizeHint`` undersizes inline-styled HTML on
            # Qt 6 — text gets clipped top + bottom because the row's
            # painted area is short of what the rich-text layout
            # actually needs. Force a tall minimum and a matching
            # size hint floor; vcenter the label so any leftover slop
            # is split symmetrically above and below the text.
            label.setMinimumHeight(36)
            sh = label.sizeHint()
            sh.setHeight(max(sh.height(), 36))
            item = QListWidgetItem()
            item.setSizeHint(sh)
            self._list.addItem(item)
            self._list.setItemWidget(item, label)


class LiveLeaderboardWindow(QWidget):
    """Free-standing top-level window wrapping :class:`LiveLeaderboardWidget`.

    The host central map docks the same widget in its own side panel,
    but peers and solo players don't have a host map — so without this
    they never see live scores. The window is shown for every client
    so everyone gets the running standings during play.

    Layout: leaderboard on the left, a :class:`RecentReportsTicker` on
    the right side that lists each visible report with the local
    team's warning types covering it at the report's time. Pass
    ``local_team_id`` so the "you" marker / bold row highlights the
    right team. The window's :meth:`refresh` updates both panes.
    """

    def __init__(
        self,
        local_team_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        # No parent → independent top-level window, but still owned by
        # the Qt application so it dies with the main window.
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Live leaderboard")
        self.resize(720, 360)
        self.widget = LiveLeaderboardWidget(local_team_id=local_team_id, parent=self)
        self.ticker = RecentReportsTicker(parent=self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget, stretch=2)
        layout.addWidget(self.ticker, stretch=3)

    def refresh(
        self,
        scores: list[TeamScore],
        team_names: dict[str, str] | None = None,
        *,
        visible_reports: list[Report] | None = None,
        team_warnings: list[Warning] | None = None,
    ) -> None:
        self.widget.refresh(scores, team_names)
        # Backward-compat: callers that don't pass the new args just
        # update the leaderboard, leaving the ticker at its prior
        # state. PlayView wires the new args on every tick.
        if visible_reports is not None and team_warnings is not None:
            self.ticker.refresh(visible_reports, team_warnings)


class FinalLeaderboardDialog(QDialog):
    """End-of-round full breakdown dialog (plan §9)."""

    def __init__(
        self,
        scores: list[TeamScore],
        team_names: dict[str, str],
        *,
        date_reveal: str | None = None,
        location_reveal: str | None = None,
        event_url: str | None = None,
        replay_path: str | None = None,
        session: GameSession | None = None,
        local_player_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Round complete")
        self.setModal(True)
        self.resize(1100, 700)

        # Reveal banner
        banner_parts = ["<b>Round complete</b>"]
        if date_reveal:
            banner_parts.append(f"Event date: <b>{date_reveal}</b>")
        if location_reveal:
            banner_parts.append(f"Location: {location_reveal}")
        if event_url:
            banner_parts.append(f'<a href="{event_url}">Event review</a>')
        banner = QLabel("  ·  ".join(banner_parts), self)
        banner.setOpenExternalLinks(True)
        banner.setStyleSheet("font-size: 12pt; padding: 8px;")

        # Score table
        table = QTableWidget(self)
        cols = ["Team", "Total", "Warnings", "MCDs", "POD", "FAR", "CSI",
                "Mean Lead", "Lead (P25/P75)", "#Warn", "#FA", "Verified"]
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setRowCount(len(scores))
        ranked = sorted(scores, key=lambda s: s.total, reverse=True)
        for row, s in enumerate(ranked):
            name = team_names.get(s.team_id, s.team_id)
            color = color_for_team(s.team_id)
            cells = [
                ("● " + name, color),
                (f"{s.total:+.1f}", None),
                (f"{s.warnings_total:+.1f}", None),
                (f"{s.mcd_total:+.1f}", None),
                (f"{s.pod*100:.1f}%", None),
                (f"{s.far*100:.1f}%", None),
                (f"{s.csi*100:.1f}%", None),
                (format_player_offset(s.mean_lead_time_sec), None),
                (f"{format_player_offset(s.p25_lead_time_sec)} / "
                 f"{format_player_offset(s.p75_lead_time_sec)}", None),
                (str(s.n_warnings), None),
                (str(s.n_false_alarms), None),
                (f"{s.n_verifying_reports}/{s.n_total_reports_in_game}", None),
            ]
            for col, (text, color_for_cell) in enumerate(cells):
                item = QTableWidgetItem(text)
                if color_for_cell:
                    item.setForeground(_qcolor(color_for_cell))
                table.setItem(row, col, item)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

        # Tabs: scores table + recap map. Recap is only available when
        # the caller passes us a session; offline / test invocations
        # that just want the table can omit it and we fall through to
        # a single-pane layout.
        tabs = QTabWidget(self)
        scores_tab = QWidget(self)
        scores_layout = QVBoxLayout(scores_tab)
        scores_layout.setContentsMargins(0, 0, 0, 0)
        scores_layout.addWidget(table)
        tabs.addTab(scores_tab, "Scores")
        if session is not None and local_player_id is not None:
            tabs.addTab(RecapMap(session, local_player_id, parent=self),
                         "Your warnings")

        # Close button
        close = QPushButton("Close", self)
        close.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(banner)
        layout.addWidget(tabs, stretch=1)
        if replay_path:
            replay_label = QLabel(
                f"<i>Replay saved to:</i> <a href='file://{replay_path}'>{replay_path}</a>",
                self,
            )
            replay_label.setOpenExternalLinks(True)
            replay_label.setStyleSheet("color: #888; padding: 4px;")
            layout.addWidget(replay_label)
        layout.addWidget(close, alignment=Qt.AlignmentFlag.AlignRight)


def _qcolor(hex_color: str):
    from PyQt6.QtGui import QColor
    return QColor(hex_color)
