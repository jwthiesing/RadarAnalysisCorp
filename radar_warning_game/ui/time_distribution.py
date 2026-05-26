"""Stacked time-of-day histogram with a draggable range slider (plan §3).

After the host picks the game-area polygon, we filter the day's LSRs to those
inside it and show the temporal distribution of reports by category. The host
drags a span to pick the game's ``[start, end]`` time window.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.widgets import SpanSelector
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.reports import Report
from .time_format import format_player_time_short

# 10-minute bins are a good balance between fidelity and visual smoothness
DEFAULT_BIN_MINUTES = 10

_COLORS = {
    "tornado": "#d62728",
    "hail":    "#2ca02c",
    "wind":    "#1f77b4",
}


class TimeDistribution(QWidget):
    """Stacked bar chart with a span selector for picking the time window.

    Emits :attr:`window_changed` with ``(start_dt, end_dt)`` whenever the host
    moves or resizes the selected span.
    """

    window_changed = pyqtSignal(object, object)   # (datetime, datetime) UTC
    start_requested = pyqtSignal()                  # host clicked the Start Round button

    def __init__(
        self,
        reports: list[Report],
        day_start_12z: datetime,
        *,
        bin_minutes: int = DEFAULT_BIN_MINUTES,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.day_start = day_start_12z if day_start_12z.tzinfo else day_start_12z.replace(tzinfo=timezone.utc)
        self.day_end = self.day_start + timedelta(days=1)
        self.bin_minutes = bin_minutes

        self._figure = Figure(figsize=(8, 3.5), facecolor="#0a0a0a")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self.ax = self._figure.add_subplot(111)
        self._style_axes()
        self._draw_stack(reports)

        # Default window = full 12Z-12Z range
        self._window_start = self.day_start
        self._window_end = self.day_end
        self._span = SpanSelector(
            self.ax, self._on_span, "horizontal",
            useblit=True, props=dict(alpha=0.25, facecolor="#ffd400"),
            interactive=True, drag_from_anywhere=True,
        )
        self._span.extents = (0.0, 24.0)  # default span covers full day

        self._label = QLabel(self._format_status(), self)
        self._start_btn = QPushButton("Start round →", self)
        self._start_btn.setStyleSheet("font-weight: bold;")
        self._start_btn.clicked.connect(self.start_requested.emit)

        bottom = QHBoxLayout()
        bottom.addWidget(self._label, stretch=1)
        bottom.addWidget(self._start_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(bottom)

    # ---- public API ----------------------------------------------------

    def selected_window(self) -> tuple[datetime, datetime]:
        return self._window_start, self._window_end

    def set_window(self, start: datetime, end: datetime) -> None:
        h0 = (start - self.day_start).total_seconds() / 3600.0
        h1 = (end - self.day_start).total_seconds() / 3600.0
        self._span.extents = (h0, h1)
        self._update_window(h0, h1)

    # ---- drawing -------------------------------------------------------

    def _style_axes(self) -> None:
        self.ax.set_facecolor("#0a0a0a")
        self.ax.tick_params(colors="#bbbbbb", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color("#444")
        self.ax.set_xlim(0, 24)
        self.ax.set_xlabel("Hours after 12Z", color="#bbbbbb")
        self.ax.set_ylabel("# reports", color="#bbbbbb")

    def _draw_stack(self, reports: list[Report]) -> None:
        nbins = int(24 * 60 / self.bin_minutes)
        edges = np.linspace(0.0, 24.0, nbins + 1)
        bottom = np.zeros(nbins)
        for category, color in _COLORS.items():
            hits = [
                (r.time - self.day_start).total_seconds() / 3600.0
                for r in reports if r.category == category
            ]
            if not hits:
                continue
            counts, _ = np.histogram(hits, bins=edges)
            self.ax.bar(
                edges[:-1], counts, width=24.0 / nbins, bottom=bottom,
                color=color, align="edge", edgecolor="none", label=category,
            )
            bottom += counts
        self.ax.legend(
            loc="upper right", facecolor="#101010", edgecolor="#333",
            labelcolor="#cccccc", fontsize=8,
        )

    # ---- span selector callback ----------------------------------------

    def _on_span(self, vmin: float, vmax: float) -> None:
        self._update_window(vmin, vmax)

    def _update_window(self, h0: float, h1: float) -> None:
        h0 = max(0.0, min(24.0, float(h0)))
        h1 = max(0.0, min(24.0, float(h1)))
        if h1 < h0:
            h0, h1 = h1, h0
        self._window_start = self.day_start + timedelta(hours=h0)
        self._window_end = self.day_start + timedelta(hours=h1)
        self._label.setText(self._format_status())
        self.window_changed.emit(self._window_start, self._window_end)

    def _format_status(self) -> str:
        dur_min = int((self._window_end - self._window_start).total_seconds() / 60)
        return (
            f"Selected window: {format_player_time_short(self._window_start)} "
            f"→ {format_player_time_short(self._window_end)}    "
            f"duration: {dur_min // 60}h {dur_min % 60}m"
        )
