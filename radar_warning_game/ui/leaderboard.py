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

from ..game.session import GameSession
from ..verification.scoring import TeamScore
from .colors import color_for_team
from .recap_map import RecapMap
from .time_format import format_player_offset


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


class LiveLeaderboardWindow(QWidget):
    """Free-standing top-level window wrapping :class:`LiveLeaderboardWidget`.

    The host central map docks the same widget in its own side panel,
    but peers and solo players don't have a host map — so without this
    they never see live scores. The window is shown for every client
    so everyone gets the running standings during play.

    Pass ``local_team_id`` so the "you" marker / bold row highlights
    the right team. The window's :meth:`refresh` is a thin pass-through
    to the inner widget; PlayView calls it on every tick.
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
        self.resize(360, 220)
        self.widget = LiveLeaderboardWidget(local_team_id=local_team_id, parent=self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.widget)

    def refresh(self, scores: list[TeamScore], team_names: dict[str, str] | None = None) -> None:
        self.widget.refresh(scores, team_names)


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
