"""Team lobby widget (plan §11).

Pre-round-start UI that lets players form / join / leave teams. The host has
admin powers (move others, auto-assign, rename, delete) — implemented as a
``host_mode`` flag.

Renders as a left pane of teams + a right pane of details/actions. Solo teams
(``solo:<player_id>``) are visually grouped under "Unassigned" since they
represent players who haven't joined a real team yet.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..game.session import SOLO_TEAM_PREFIX, GameSession
from .colors import color_for_team

_UNASSIGNED_GROUP_KEY = "__unassigned__"


class TeamLobbyWidget(QWidget):
    """List-based team management UI bound to a :class:`GameSession`.

    Signals
    -------
    request_create_team(str name)
    request_join_team(str team_id)
    request_leave_team()
    request_freeze_roster()
    request_move_player(str player_id, str team_id) — host only
    """

    request_create_team = pyqtSignal(str)
    request_join_team = pyqtSignal(str)
    request_leave_team = pyqtSignal()
    request_freeze_roster = pyqtSignal()
    request_move_player = pyqtSignal(str, str)

    def __init__(
        self,
        session: GameSession,
        local_player_id: str,
        *,
        host_mode: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.local_player_id = local_player_id
        self.host_mode = host_mode

        self._teams_list = QListWidget(self)
        self._teams_list.setMinimumWidth(300)
        self._teams_list.itemDoubleClicked.connect(self._on_double_click)

        # Right pane: action buttons
        right = QVBoxLayout()
        self._create_btn = QPushButton("Create team…", self)
        self._create_btn.clicked.connect(self._on_create)
        right.addWidget(self._create_btn)

        self._join_btn = QPushButton("Join selected team", self)
        self._join_btn.clicked.connect(self._on_join)
        right.addWidget(self._join_btn)

        self._leave_btn = QPushButton("Leave my team (back to solo)", self)
        self._leave_btn.clicked.connect(self.request_leave_team.emit)
        right.addWidget(self._leave_btn)

        if self.host_mode:
            right.addSpacing(20)
            host_label = QLabel("<b>Host controls</b>", self)
            right.addWidget(host_label)
            self._move_btn = QPushButton("Move selected player to…", self)
            self._move_btn.setToolTip("Select a player, then click here to move them to a different team")
            self._move_btn.clicked.connect(self._on_move_player)
            right.addWidget(self._move_btn)
            self._freeze_btn = QPushButton("Start round (freeze teams)", self)
            self._freeze_btn.clicked.connect(self.request_freeze_roster.emit)
            right.addWidget(self._freeze_btn)
        right.addStretch(1)

        layout = QHBoxLayout(self)
        layout.addWidget(self._teams_list, stretch=1)
        layout.addLayout(right, stretch=0)

        self.refresh()

    # ---- public --------------------------------------------------------

    def refresh(self) -> None:
        """Repopulate the list from the bound :class:`GameSession`."""
        self._teams_list.clear()
        # Group teams: real teams first, then unassigned (solo)
        real_teams: list[tuple[str, str, list[str]]] = []
        solo_players: list[str] = []
        for tid, members in self.session.teams.items():
            if tid.startswith(SOLO_TEAM_PREFIX):
                solo_players.extend(members)
            else:
                real_teams.append((tid, self.session.team_names.get(tid, "?"), members))

        for tid, name, members in sorted(real_teams, key=lambda t: t[1].lower()):
            color = color_for_team(tid)
            label = f"● {name}  ({len(members)} member{'s' if len(members) != 1 else ''})"
            header = QListWidgetItem(label)
            header.setData(Qt.ItemDataRole.UserRole, ("team", tid))
            header.setForeground(_qcolor(color))
            self._teams_list.addItem(header)
            for pid in members:
                player = self.session.players.get(pid)
                mark = "  ← you" if pid == self.local_player_id else ""
                item = QListWidgetItem(f"    {player.display_name if player else pid}{mark}")
                item.setData(Qt.ItemDataRole.UserRole, ("player", pid))
                self._teams_list.addItem(item)

        if solo_players:
            unassigned_header = QListWidgetItem(f"○ Unassigned  ({len(solo_players)})")
            unassigned_header.setData(Qt.ItemDataRole.UserRole, ("group", _UNASSIGNED_GROUP_KEY))
            unassigned_header.setForeground(_qcolor("#888888"))
            self._teams_list.addItem(unassigned_header)
            for pid in solo_players:
                player = self.session.players.get(pid)
                mark = "  ← you" if pid == self.local_player_id else ""
                item = QListWidgetItem(f"    {player.display_name if player else pid}{mark}")
                item.setData(Qt.ItemDataRole.UserRole, ("player", pid))
                self._teams_list.addItem(item)

    # ---- internals -----------------------------------------------------

    def _on_create(self) -> None:
        name, ok = QInputDialog.getText(self, "New team", "Team name:")
        if ok and name.strip():
            self.request_create_team.emit(name.strip())

    def _on_join(self) -> None:
        item = self._teams_list.currentItem()
        if item is None:
            return
        kind, payload = item.data(Qt.ItemDataRole.UserRole)
        if kind != "team":
            return
        self.request_join_team.emit(payload)

    def _on_double_click(self, item: QListWidgetItem) -> None:
        kind, payload = item.data(Qt.ItemDataRole.UserRole)
        if kind == "team":
            self.request_join_team.emit(payload)

    def _on_move_player(self) -> None:
        item = self._teams_list.currentItem()
        if item is None:
            return
        kind, pid = item.data(Qt.ItemDataRole.UserRole)
        if kind != "player":
            return
        # Build picker: every real team + "back to unassigned"
        team_options = [
            (self.session.team_names.get(tid, tid), tid)
            for tid in self.session.teams
            if not tid.startswith(SOLO_TEAM_PREFIX)
        ]
        team_options.append(("Unassigned (solo)", ""))
        choices = [name for name, _ in team_options]
        choice, ok = QInputDialog.getItem(self, "Move player",
                                           "Move to which team?", choices, 0, False)
        if not ok:
            return
        team_id = next(tid for name, tid in team_options if name == choice)
        self.request_move_player.emit(pid, team_id)


def _qcolor(hex_color: str):
    from PyQt6.QtGui import QColor
    return QColor(hex_color)
