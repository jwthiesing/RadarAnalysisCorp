"""Stacked time-of-day histogram with a draggable range slider (plan §3).

After the host picks the game-area polygon, we filter the day's LSRs to
those inside it and show the temporal distribution of reports by category.
The host drags a span to pick the game's ``[start, end]`` time window.

Built on pyqtgraph (was matplotlib + SpanSelector). The range is a
:class:`pyqtgraph.LinearRegionItem`; bars are
:class:`pyqtgraph.BarGraphItem` stacked manually.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.reports import Report
from .time_format import format_player_time_short

DEFAULT_BIN_MINUTES = 10

_COLORS = {
    "tornado": "#d62728",
    "hail":    "#2ca02c",
    "wind":    "#1f77b4",
}


class TimeDistribution(QWidget):
    """Stacked bar chart with a span selector for picking the time window.

    Emits :attr:`window_changed` with ``(start_dt, end_dt)`` whenever the
    host moves or resizes the selected span.
    """

    window_changed = pyqtSignal(object, object)   # (datetime, datetime) UTC
    start_requested = pyqtSignal()

    def __init__(
        self,
        reports: list[Report],
        day_start_12z: datetime,
        *,
        bin_minutes: int = DEFAULT_BIN_MINUTES,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.day_start = (day_start_12z if day_start_12z.tzinfo
                          else day_start_12z.replace(tzinfo=timezone.utc))
        self.day_end = self.day_start + timedelta(days=1)
        self.bin_minutes = bin_minutes

        self._plot = pg.PlotWidget(parent=self)
        self._plot.setBackground("#0a0a0a")
        self._plot.setMenuEnabled(False)
        plot_item = self._plot.getPlotItem()
        plot_item.setLabel("bottom", "Hours after 12Z", color="#bbbbbb")
        plot_item.setLabel("left", "# reports", color="#bbbbbb")
        plot_item.showGrid(x=True, y=True, alpha=0.15)
        plot_item.setMouseEnabled(x=True, y=False)
        plot_item.setLimits(xMin=0, xMax=24, yMin=0)
        self._view: pg.ViewBox = plot_item.getViewBox()
        self._view.setRange(xRange=(0, 24), padding=0)

        self._draw_stack(reports)

        # Default window = full 12Z-12Z range.
        self._window_start = self.day_start
        self._window_end = self.day_end

        # The draggable range — colored fill + thicker boundary lines so
        # the handles are easy to grab.
        self._region = pg.LinearRegionItem(
            values=(0.0, 24.0),
            orientation="vertical",
            brush=pg.mkBrush(QColor(255, 212, 0, 60)),
            hoverBrush=pg.mkBrush(QColor(255, 212, 0, 100)),
            pen=pg.mkPen("#ffd400", width=2),
        )
        self._region.setZValue(10)
        self._plot.addItem(self._region)
        self._region.sigRegionChanged.connect(self._on_region_changed)

        self._label = QLabel(self._format_status(), self)
        self._start_btn = QPushButton("Start round →", self)
        self._start_btn.setStyleSheet("font-weight: bold;")
        self._start_btn.clicked.connect(self.start_requested.emit)

        bottom = QHBoxLayout()
        bottom.addWidget(self._label, stretch=1)
        bottom.addWidget(self._start_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._plot, stretch=1)
        layout.addLayout(bottom)

    # ---- public API --------------------------------------------------

    def selected_window(self) -> tuple[datetime, datetime]:
        return self._window_start, self._window_end

    def set_window(self, start: datetime, end: datetime) -> None:
        h0 = (start - self.day_start).total_seconds() / 3600.0
        h1 = (end - self.day_start).total_seconds() / 3600.0
        self._region.setRegion((h0, h1))
        self._update_window(h0, h1)

    # ---- drawing -----------------------------------------------------

    def _draw_stack(self, reports: list[Report]) -> None:
        nbins = int(24 * 60 / self.bin_minutes)
        edges = np.linspace(0.0, 24.0, nbins + 1)
        width = 24.0 / nbins
        bottom = np.zeros(nbins, dtype=np.float64)
        for category, color in _COLORS.items():
            hits = [
                (r.time - self.day_start).total_seconds() / 3600.0
                for r in reports if r.category == category
            ]
            if not hits:
                continue
            counts, _ = np.histogram(hits, bins=edges)
            bar = pg.BarGraphItem(
                x0=edges[:-1], height=counts, width=width,
                y0=bottom.copy(),
                brush=pg.mkBrush(color), pen=pg.mkPen(color),
            )
            bar.setZValue(2)
            self._plot.addItem(bar)
            bottom += counts
        # Legend (simple TextItems).
        legend = pg.LegendItem(offset=(-10, 10))
        legend.setParentItem(self._plot.getPlotItem().graphicsItem())
        for category, color in _COLORS.items():
            swatch = pg.PlotCurveItem(pen=pg.mkPen(color, width=4))
            legend.addItem(swatch, category)

    # ---- region callback ---------------------------------------------

    def _on_region_changed(self, _region) -> None:
        h0, h1 = self._region.getRegion()
        self._update_window(float(h0), float(h1))

    def _update_window(self, h0: float, h1: float) -> None:
        h0 = max(0.0, min(24.0, h0))
        h1 = max(0.0, min(24.0, h1))
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
