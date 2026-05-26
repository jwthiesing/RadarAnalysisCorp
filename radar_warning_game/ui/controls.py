"""Clock controls widget — play/pause + bracket-key speed (plan §4).

Host-only during multiplayer (peers see ticks via the network). Provides:
  - ``Space`` toggle pause/play
  - ``[`` slower (halves speed multiplier)
  - ``]`` faster (doubles speed multiplier)
  - Visible speed indicator
  - "End round" button (host only)

The widget owns a :class:`GameClock` reference and a periodic :class:`QTimer`
that advances the clock and emits :attr:`tick` for the network layer to
broadcast. UI updates happen on every advance.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QWidget,
)

from ..game.clock import GameClock, TickState
from .time_format import format_player_time

# Host clock advance + network broadcast rate.
# 1 Hz keeps network load minimal at the 50-player room cap (1 message/sec × 50
# peers = 50 msg/sec from ticks; the host can absolutely handle that). The UI
# itself runs the QTimer at a higher rate (UI_REFRESH_HZ below) to keep on-screen
# time display smooth between actual tick broadcasts.
HOST_TICK_HZ = 1
UI_REFRESH_HZ = 4


class ClockControls(QFrame):
    """Host-side clock control bar.

    Signals
    -------
    tick(TickState)
        emitted every host tick — wire this to the network layer to broadcast
    request_end_round()
        emitted when the host clicks "End round"
    """

    tick = pyqtSignal(object)             # TickState
    request_end_round = pyqtSignal()

    def __init__(self, clock: GameClock, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.clock = clock
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._play_btn = QPushButton("▶ Play", self)
        self._play_btn.clicked.connect(self._toggle_play)

        self._slower_btn = QPushButton("[ Slower", self)
        self._slower_btn.clicked.connect(self._slower)

        self._faster_btn = QPushButton("Faster ]", self)
        self._faster_btn.clicked.connect(self._faster)

        self._time_label = QLabel("--:--:--Z", self)
        self._time_label.setStyleSheet("font-family: monospace; font-size: 14pt; padding: 0 12px;")

        self._speed_label = QLabel("1.0×", self)
        self._speed_label.setStyleSheet("font-family: monospace; font-size: 12pt; padding: 0 8px;")

        self._end_btn = QPushButton("End round", self)
        self._end_btn.setStyleSheet("color: #ff8888;")
        self._end_btn.clicked.connect(self.request_end_round.emit)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 4)
        layout.addWidget(self._play_btn)
        layout.addWidget(self._slower_btn)
        layout.addWidget(self._faster_btn)
        layout.addWidget(self._time_label)
        layout.addWidget(self._speed_label)
        layout.addStretch(1)
        layout.addWidget(self._end_btn)

        # UI timer ticks at UI_REFRESH_HZ for smooth labels; network broadcast
        # is throttled to HOST_TICK_HZ separately to keep peer bandwidth low.
        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / UI_REFRESH_HZ))
        self._timer.timeout.connect(self._on_tick)
        self._timer.start()
        self._broadcast_interval_sec = 1.0 / max(HOST_TICK_HZ, 0.001)
        self._last_broadcast: float = 0.0

        self._refresh_labels()

    # ---- public --------------------------------------------------------

    def stop(self) -> None:
        self._timer.stop()

    # ---- handlers ------------------------------------------------------

    def _toggle_play(self) -> None:
        self.clock.toggle_pause()
        self._refresh_labels()
        self.tick.emit(self.clock.snapshot())

    def _faster(self) -> None:
        self.clock.faster()
        self._refresh_labels()
        self.tick.emit(self.clock.snapshot())

    def _slower(self) -> None:
        self.clock.slower()
        self._refresh_labels()
        self.tick.emit(self.clock.snapshot())

    def _on_tick(self) -> None:
        import time as _time
        snap = self.clock.advance()
        self._refresh_labels()
        # Only emit (broadcast to peers + heavy UI refreshes) at HOST_TICK_HZ,
        # not on every UI-refresh tick.
        now = _time.monotonic()
        if now - self._last_broadcast >= self._broadcast_interval_sec:
            self._last_broadcast = now
            self.tick.emit(snap)
        if self.clock.is_over():
            self._timer.stop()
            self.request_end_round.emit()

    def _refresh_labels(self) -> None:
        self._time_label.setText(format_player_time(self.clock.virtual_time))
        self._speed_label.setText(f"{self.clock.speed:g}×")
        self._play_btn.setText("▶ Play" if self.clock.paused else "❚❚ Pause")

    # ---- keyboard ------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802
        key = event.key()
        if key == Qt.Key.Key_Space:
            self._toggle_play()
        elif key == Qt.Key.Key_BracketLeft:
            self._slower()
        elif key == Qt.Key.Key_BracketRight:
            self._faster()
        else:
            super().keyPressEvent(event)
            return
        event.accept()
