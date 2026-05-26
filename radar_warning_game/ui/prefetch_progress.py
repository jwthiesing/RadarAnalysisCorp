"""Pre-round prefetch progress UI (plan §10).

Single-player simplification of the multi-client gate from the plan: in solo
mode we just download all pre-game volumes in parallel and show per-radar
progress. When all done, we emit :attr:`ready_to_play` and the parent
transitions to the PLAYING view.

The Prefetcher itself runs background threads; we poll it on a QTimer to
update the bars.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.prefetch import Prefetcher


class PrefetchProgressWidget(QWidget):
    """Shows a progress bar per enabled radar while the pre-game window downloads."""

    ready_to_play = pyqtSignal()

    def __init__(self, prefetcher: Prefetcher, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.prefetcher = prefetcher

        self._title = QLabel("Downloading radar volumes for round start…", self)
        self._title.setStyleSheet("font-size: 13pt; padding: 8px;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._bars: dict[str, QProgressBar] = {}
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(self._title)

        for site in prefetcher.sites:
            row = QLabel(site, self)
            row.setAlignment(Qt.AlignmentFlag.AlignCenter)
            bar = QProgressBar(self)
            bar.setRange(0, 100)
            bar.setValue(0)
            layout.addWidget(row)
            layout.addWidget(bar)
            self._bars[site] = bar

        self._skip_btn = QPushButton("Start anyway", self)
        self._skip_btn.clicked.connect(self.ready_to_play.emit)
        layout.addWidget(self._skip_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def _poll(self) -> None:
        progress = self.prefetcher.pregame_progress()
        all_done = True
        any_seen = False
        for site, bar in self._bars.items():
            done, total = progress.get(site, (0, 0))
            if total > 0:
                any_seen = True
                bar.setRange(0, total)
                bar.setValue(done)
                bar.setFormat(f"{done}/{total}")
                if done < total:
                    all_done = False
            else:
                bar.setRange(0, 1)
                bar.setValue(0)
                bar.setFormat("(listing…)")
                all_done = False
        if all_done and any_seen:
            self._timer.stop()
            self.ready_to_play.emit()

    def stop(self) -> None:
        self._timer.stop()
