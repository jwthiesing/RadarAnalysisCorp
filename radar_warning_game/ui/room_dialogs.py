"""Multiplayer connection dialogs (plan §1, §10).

  - :class:`ModeDialog` — first screen: Solo / Host / Join radio buttons.
  - :class:`JoinRoomDialog` — for non-hosts: room-code input + signaling URL.
  - :class:`HostRoomStatusDialog` — after host setup completes; shows the room
    code and a live list of connected peers. Closes when host clicks "Start
    Round" (which triggers the rest of the orchestrator).
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from ..net.peer import DEFAULT_SIGNALING_URL


class ModeDialog(QDialog):
    """First screen: choose Solo / Host a Room / Join a Room."""

    SOLO = "solo"
    HOST = "host"
    JOIN = "join"

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("RadarAnalysisCorp — Start")
        self.setModal(True)

        self._solo = QRadioButton("Solo (single-player)", self)
        self._solo.setChecked(True)
        self._host = QRadioButton("Host a multiplayer room", self)
        self._join = QRadioButton("Join an existing room", self)

        self._name_edit = QLineEdit(self)
        self._name_edit.setText("Player")
        self._name_edit.setMaximumWidth(220)

        self._sig_url_edit = QLineEdit(self)
        self._sig_url_edit.setText(DEFAULT_SIGNALING_URL)
        self._sig_url_edit.setMaximumWidth(360)

        form = QFormLayout()
        form.addRow("Display name:", self._name_edit)
        form.addRow("Signaling server URL:", self._sig_url_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Continue")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._solo)
        layout.addWidget(self._host)
        layout.addWidget(self._join)
        layout.addSpacing(8)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def mode(self) -> str:
        if self._solo.isChecked():
            return self.SOLO
        if self._host.isChecked():
            return self.HOST
        return self.JOIN

    def display_name(self) -> str:
        return self._name_edit.text().strip() or "Player"

    def signaling_url(self) -> str:
        return self._sig_url_edit.text().strip() or DEFAULT_SIGNALING_URL


class JoinRoomDialog(QDialog):
    """Prompt the joiner for the room code given by the host."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Join Room")
        self.setModal(True)

        self._code_edit = QLineEdit(self)
        self._code_edit.setPlaceholderText("e.g. STORM-FROG-72")
        self._code_edit.setMaximumWidth(280)
        self._code_edit.textChanged.connect(self._auto_upper)

        form = QFormLayout()
        form.addRow("Room code:", self._code_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Join")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _auto_upper(self, text: str) -> None:
        # Room codes are uppercase; auto-fix
        if text != text.upper():
            self._code_edit.blockSignals(True)
            cursor = self._code_edit.cursorPosition()
            self._code_edit.setText(text.upper())
            self._code_edit.setCursorPosition(cursor)
            self._code_edit.blockSignals(False)

    def room_code(self) -> str:
        return self._code_edit.text().strip()


class HostRoomStatusDialog(QDialog):
    """Shown to the host between starting the room and finishing setup.

    Displays the room code (for sharing) plus a live list of connected peers.
    The host clicks "Continue to Setup" to dismiss and move into the day picker.
    """

    def __init__(self, room_code: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Room Hosted")
        self.setModal(True)

        title = QLabel(f"<b>Room code:</b> &nbsp; <span style='font-size: 18pt; color: #ffd400'>{room_code}</span>", self)
        title.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        hint = QLabel(
            "Share this code with anyone you want to join your room. "
            "Peers will see your round setup once you click Continue.",
            self,
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #aaa;")

        self._peer_list = QListWidget(self)
        self._peer_list.addItem("(waiting for peers…)")

        continue_btn = QPushButton("Continue to Setup", self)
        continue_btn.clicked.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(QLabel("<b>Connected peers:</b>", self))
        layout.addWidget(self._peer_list)
        layout.addWidget(continue_btn)

    def add_peer(self, peer_id: str, name: str) -> None:
        # Strip the placeholder if present
        if self._peer_list.count() == 1 and self._peer_list.item(0).text().startswith("("):
            self._peer_list.clear()
        item = QListWidgetItem(f"{name}  ({peer_id})")
        self._peer_list.addItem(item)

    def remove_peer(self, peer_id: str) -> None:
        for i in range(self._peer_list.count()):
            item = self._peer_list.item(i)
            if peer_id in item.text():
                self._peer_list.takeItem(i)
                return
