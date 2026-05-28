"""End-of-round recap map (plan §9 expansion).

A single pyqtgraph view shown alongside the final-leaderboard table that
plots every warning the local player issued during the round, color-coded
by verification outcome, against the full set of in-game storm reports.

This is the "what did I actually call?" view — the table on the other
tab tells you the score, this tells you the picture. Verified warnings
are drawn green and false alarms red; cancelled warnings get a dashed
outline so you can see which ones you pulled back. The game polygon and
state lines anchor everything geographically.

Unlike the live host map, time has stopped: reports are fully opaque
(no fade), warnings are drawn whether or not they're "active" right
now, and there is no leaderboard sidecar — every score-related summary
lives on the other dialog tab.
"""

from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..data.reports import Report
from ..data.sites import site_by_icao
from ..game.session import GameSession
from ..verification.reports_in_poly import Warning, find_verifying_reports
from ..verification.tornado_tiers import WarningType
from .overlay_loader import load_conus_lines_latlon
from .radar_panel import _concat_with_gaps
from .time_format import format_player_time_short

log = logging.getLogger(__name__)

# Outcome palette — distinguishable for the dichromacy-common red/green
# axis by leaning red toward orange-red and green toward cyan-green.
_VERIFIED_COLOR = "#33dd88"
_FALSE_ALARM_COLOR = "#ee5544"
_CANCELED_DIM = "#888888"

# Per-tier line weight — mirrors host_map._TIER_LINEWIDTH but a touch
# heavier since this view is non-interactive and benefits from contrast.
_TIER_LW = {
    WarningType.SVR: 1.6,
    WarningType.SVRC: 2.0,
    WarningType.SVRD: 2.4,
    WarningType.TOR: 1.8,
    WarningType.TORR: 2.2,
    WarningType.PDS_TOR: 2.8,
    WarningType.TORE: 3.4,
}

_REPORT_SYMBOLS = {"tornado": "t1", "hail": "o", "wind": "s"}
_REPORT_EDGES = {"tornado": "#ff3030", "hail": "#22cc55", "wind": "#3399ff"}
_REPORT_FILLS = {"tornado": "#ff4444", "hail": "#44ff66", "wind": "#66bbff"}


class RecapMap(QWidget):
    """End-of-round map of one player's warnings + the day's reports."""

    def __init__(
        self,
        session: GameSession,
        local_player_id: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.local_player_id = local_player_id

        self._plot = pg.PlotWidget(parent=self)
        self._plot.setBackground("#0a0a0a")
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.setMenuEnabled(False)
        self.view: pg.ViewBox = self._plot.getViewBox()
        self.view.setAspectLocked(True, ratio=1.0)
        self.view.setMenuEnabled(False)

        self._caption = QLabel(self)
        self._caption.setWordWrap(True)
        self._caption.setStyleSheet("color: #ccc; padding: 4px;")
        self._caption.setTextFormat(Qt.TextFormat.RichText)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._caption)
        layout.addWidget(self._plot, stretch=1)

        self._draw_basemap()
        self._draw_game_polygon()
        self._draw_radar_sites()
        n_verified, n_fa = self._draw_warnings()
        n_reports = self._draw_reports()
        self._set_caption(n_verified, n_fa, n_reports)

    # ------------------------------------------------------------------

    def _set_caption(self, verified: int, fa: int, reports: int) -> None:
        total_w = verified + fa
        if total_w == 0:
            self._caption.setText(
                "<b>Your warnings</b> &nbsp; "
                "<i>no warnings issued this round</i>"
            )
            return
        self._caption.setText(
            "<b>Your warnings</b> &nbsp; "
            f"<span style='color:{_VERIFIED_COLOR}'>● verified {verified}</span> &nbsp; "
            f"<span style='color:{_FALSE_ALARM_COLOR}'>● false alarm {fa}</span> &nbsp; "
            f"<span style='color:#aaa'>{reports} reports in game area</span>"
        )

    def _draw_basemap(self) -> None:
        try:
            data = load_conus_lines_latlon()
        except Exception as e:  # noqa: BLE001
            log.warning("recap map basemap load failed: %s", e)
            return
        if data["states"]:
            xs, ys = _concat_with_gaps(data["states"])
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#666", width=0.6),
                                     connect="finite")
            item.setZValue(1)
            self.view.addItem(item)
        if data["borders"]:
            xs, ys = _concat_with_gaps(data["borders"])
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#888", width=0.9),
                                     connect="finite")
            item.setZValue(1)
            self.view.addItem(item)

    def _draw_game_polygon(self) -> None:
        cfg = self.session.round_config
        if cfg is None:
            return
        verts = list(cfg.game_polygon.vertices) + [cfg.game_polygon.vertices[0]]
        lons = np.array([v[1] for v in verts], dtype=np.float64)
        lats = np.array([v[0] for v in verts], dtype=np.float64)
        item = pg.PlotCurveItem(lons, lats, pen=pg.mkPen("#ffcc00", width=2.0))
        item.setZValue(8)
        self.view.addItem(item)
        pad = 0.5
        self.view.setRange(
            xRange=(float(lons.min()) - pad, float(lons.max()) + pad),
            yRange=(float(lats.min()) - pad, float(lats.max()) + pad),
            padding=0,
        )

    def _draw_radar_sites(self) -> None:
        cfg = self.session.round_config
        if cfg is None:
            return
        xs, ys, names = [], [], []
        for icao in cfg.radar_sites:
            site = site_by_icao(icao)
            if site is None:
                continue
            xs.append(site.lon)
            ys.append(site.lat)
            names.append(icao)
        if not xs:
            return
        scatter = pg.ScatterPlotItem(
            x=xs, y=ys, size=10, symbol="x",
            pen=pg.mkPen("#00d4ff", width=1.8),
            brush=pg.mkBrush("#00d4ff"),
            pxMode=True,
        )
        scatter.setZValue(9)
        self.view.addItem(scatter)
        for icao, x, y in zip(names, xs, ys):
            label = pg.TextItem(icao, anchor=(0, 1), color="#00d4ff")
            label.setPos(x + 0.05, y + 0.05)
            label.setZValue(10)
            self.view.addItem(label, ignoreBounds=True)

    def _draw_warnings(self) -> tuple[int, int]:
        warnings = self.session.warnings_by_player.get(self.local_player_id, [])
        reports = self._round_reports()
        n_verified = 0
        n_fa = 0
        for w in warnings:
            verifying = find_verifying_reports(w, reports)
            is_verified = len(verifying) > 0
            if is_verified:
                n_verified += 1
            else:
                n_fa += 1
            self._draw_single_warning(w, is_verified)
        return n_verified, n_fa

    def _draw_single_warning(self, w: Warning, verified: bool) -> None:
        rev = w.current_revision
        verts = list(rev.polygon.vertices) + [rev.polygon.vertices[0]]
        lons = np.array([v[1] for v in verts], dtype=np.float64)
        lats = np.array([v[0] for v in verts], dtype=np.float64)
        lw = _TIER_LW.get(rev.warning_type, 1.8)
        color = _VERIFIED_COLOR if verified else _FALSE_ALARM_COLOR
        # Cancelled warnings get a dashed outline so you can still see
        # which ones you pulled back vs. let ride to expiration. They
        # still carry the verified/FA color since they still count for
        # any reports during their active period.
        style = (Qt.PenStyle.DashLine if w.canceled_at is not None
                 else Qt.PenStyle.SolidLine)
        pen = pg.mkPen(color=color, width=lw, style=style)
        item = pg.PlotCurveItem(lons, lats, pen=pen)
        item.setZValue(12)
        self.view.addItem(item)
        # Centroid label: warning type + issuance time so the user can
        # mentally line each polygon up against the table on the other
        # tab and the radar memory of when they drew it.
        clat, clon = rev.polygon.centroid_latlon
        label = pg.TextItem(
            f"{rev.warning_type.value} {format_player_time_short(w.original_issue_time)}",
            anchor=(0.5, 0.5),
            color=_CANCELED_DIM if w.canceled_at is not None else color,
        )
        label.setPos(clon, clat)
        label.setZValue(13)
        self.view.addItem(label, ignoreBounds=True)

    def _draw_reports(self) -> int:
        reports = self._round_reports()
        if not reports:
            return 0
        spots = []
        for r in reports:
            edge = QColor(_REPORT_EDGES.get(r.category, "#aaaaaa"))
            fill = QColor(_REPORT_FILLS.get(r.category, "#888888"))
            spots.append(dict(
                pos=(r.lon, r.lat),
                size=_report_size(r.category, r.magnitude),
                symbol=_REPORT_SYMBOLS.get(r.category, "o"),
                pen=pg.mkPen(edge, width=1.0),
                brush=pg.mkBrush(fill),
            ))
        scatter = pg.ScatterPlotItem(spots=spots, pxMode=True)
        scatter.setZValue(6)
        self.view.addItem(scatter)
        return len(reports)

    def _round_reports(self) -> list[Report]:
        """All reports that occurred during the round, regardless of
        whether they verified anything. Filtered to the game polygon's
        rough bbox so a recap centered on Oklahoma doesn't bother
        plotting a New England wind report from the same day."""
        if self.session.round_day is None or self.session.round_config is None:
            return []
        poly = self.session.round_config.game_polygon
        verts = poly.vertices
        if not verts:
            return self.session.round_day.reports
        lats = [v[0] for v in verts]
        lons = [v[1] for v in verts]
        pad = 1.5  # ~150 km in lat / less in lon — generous but not silly
        lat_lo, lat_hi = min(lats) - pad, max(lats) + pad
        lon_lo, lon_hi = min(lons) - pad, max(lons) + pad
        out: list[Report] = []
        clock = self.session.clock
        round_end = (clock.virtual_time if clock is not None else None)
        for r in self.session.round_day.reports:
            if round_end is not None and r.time > round_end:
                continue
            if not (lat_lo <= r.lat <= lat_hi and lon_lo <= r.lon <= lon_hi):
                continue
            out.append(r)
        return out


def _report_size(category: str, magnitude: float) -> float:
    if category == "tornado":
        return 14.0 + max(0.0, float(magnitude)) * 4.0
    if category == "hail":
        return 9.0 + max(0.0, float(magnitude)) * 3.0
    if category == "wind":
        return 8.0 + max(0.0, float(magnitude) - 50.0) * 0.15
    return 9.0
