"""CONUS overview map for round setup (plan §2).

Storm reports plotted with shape-by-type, size-by-magnitude, color-by-time
in the day. Radar sites are click-to-toggle (when polygon-draw mode is OFF).
A polygon editor on the same view lets the host draw the game-area boundary
once they're done toggling radars.

Built on pyqtgraph (was matplotlib + cartopy). Coordinates are plain
``(lon, lat)`` — fine at CONUS scale without a true projection. Pan, zoom,
and the polygon-edit gestures all happen in the same view.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.radar_s3 import site_has_data_on_day
from ..data.reports import Report
from ..data.sites import SITE_KIND_TDWR, load_sites
from ..geo.polygons import Polygon
from .overlay_loader import load_conus_lines_latlon
from .poly_editor import PolygonEditor
from .radar_panel import _concat_with_gaps, _report_tooltip_text

log = logging.getLogger(__name__)

_REPORT_SYMBOLS = {"tornado": "t1", "hail": "o", "wind": "s"}
_REPORT_EDGE_COLORS = {
    "tornado": "#ff3030",
    "hail":    "#22cc55",
    "wind":    "#3399ff",
}


def _marker_size(category: str, magnitude: float) -> float:
    if category == "tornado":
        return 14.0 + max(0.0, float(magnitude)) * 4.0
    if category == "hail":
        return 9.0 + max(0.0, float(magnitude)) * 3.0
    if category == "wind":
        return 8.0 + max(0.0, float(magnitude) - 50.0) * 0.15
    return 9.0


def _time_to_color(t_norm: float) -> QColor:
    """Map normalized time (0..1) to a viridis-like color. Avoids
    pulling matplotlib in just for one colormap by using a small
    interpolation over a few stops."""
    stops = [
        (0.00, (68, 1, 84)),
        (0.25, (59, 82, 139)),
        (0.50, (33, 145, 140)),
        (0.75, (94, 201, 98)),
        (1.00, (253, 231, 37)),
    ]
    t = float(np.clip(t_norm, 0.0, 1.0))
    for i in range(len(stops) - 1):
        x0, c0 = stops[i]
        x1, c1 = stops[i + 1]
        if x0 <= t <= x1:
            f = (t - x0) / max(x1 - x0, 1e-9)
            r = int(c0[0] + f * (c1[0] - c0[0]))
            g = int(c0[1] + f * (c1[1] - c0[1]))
            b = int(c0[2] + f * (c1[2] - c0[2]))
            return QColor(r, g, b)
    return QColor(*stops[-1][1])


class OverviewMap(QWidget):
    """CONUS map widget for round setup.

    Signals
    -------
    radar_sites_changed
        emitted with the new set of enabled radar ICAOs whenever the host toggles
    polygon_changed
        emitted with a :class:`Polygon` (or ``None``) as the host draws/erases
    reroll_requested
        emitted when the host clicks "Reroll" (random-day mode only)
    continue_requested
        emitted when the host clicks "Continue to time window"
    """

    radar_sites_changed = pyqtSignal(object)   # set[str]
    polygon_changed = pyqtSignal(object)        # Polygon | None
    reroll_requested = pyqtSignal()
    continue_requested = pyqtSignal()

    def __init__(
        self,
        reports: list[Report],
        *,
        is_random_day: bool,
        initial_enabled_sites: set[str] | None = None,
        day: datetime | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.reports = reports
        self.is_random_day = is_random_day
        # The convective 12Z day the round will use — needed to probe S3
        # for archive availability per site. ``None`` means "skip the
        # probe" (e.g. live mode, where every operational radar is
        # broadcasting in real time).
        self._day = day
        self._enabled_sites: set[str] = set(initial_enabled_sites or set())
        # Sites whose Unidata Level 2 archive has no objects for the
        # chosen day. Populated lazily by ``_probe_archive_availability``
        # — rendered as dim grey and refusing toggle clicks so the host
        # can't pick a radar that will yield empty panels at round
        # start. TDWR coverage on the mirror is patchy in older years,
        # which is the main motivator.
        self._unavailable_sites: set[str] = set()
        self._site_scatter: pg.ScatterPlotItem | None = None
        self._site_brush_by_icao: dict[str, str] = {}   # icao → color hex
        self._site_index: dict[int, str] = {}           # spot index → icao
        self._report_items: list[pg.ScatterPlotItem] = []
        self._report_data: list[Report] = []

        # Plot widget — flat lon/lat coords.
        self._plot = pg.PlotWidget(parent=self)
        self._plot.setBackground("#0a0a0a")
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.setMenuEnabled(False)
        self.view: pg.ViewBox = self._plot.getViewBox()
        self.view.setAspectLocked(True, ratio=1.0)
        self._home_extent = (-125.0, -66.0, 23.5, 50.0)
        self.view.setRange(
            xRange=(self._home_extent[0], self._home_extent[1]),
            yRange=(self._home_extent[2], self._home_extent[3]),
            padding=0,
        )

        # Static basemap (state lines, country borders, coastline).
        self._draw_basemap()
        self._draw_reports()
        self._draw_radar_sites()

        # Polygon editor — drawn on top, disabled by default so left-clicks
        # toggle radar sites instead of adding vertices.
        self.poly_editor = PolygonEditor(
            self.view,
            axes_to_latlon=lambda x, y: (y, x),
            color="#ffd400",
        )
        self.poly_editor.polygon_changed.connect(self.polygon_changed.emit)
        self.poly_editor.set_enabled(False)
        self._draw_mode = False

        # Hover tooltip (single TextItem).
        self._hover = pg.TextItem(
            "", anchor=(0, 1), color="#0a0a0a",
            fill=pg.mkBrush(QColor("#ffd400")),
            border=pg.mkPen("#000", width=0.6),
        )
        self._hover.setZValue(30)
        self._hover.hide()
        self.view.addItem(self._hover, ignoreBounds=True)

        # Radar-site clicks are wired via ScatterPlotItem.sigClicked (set
        # up in _draw_radar_sites) — the canonical way to detect a click
        # on a scatter symbol. Hover tooltip for reports goes through the
        # scene's mouse-moved signal.
        self._plot.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

        # Toolbar / buttons.
        toolbar = QHBoxLayout()
        self._reroll_btn = QPushButton("Reroll random day", self)
        self._reroll_btn.setEnabled(self.is_random_day)
        self._reroll_btn.clicked.connect(self.reroll_requested.emit)
        toolbar.addWidget(self._reroll_btn)
        self._draw_btn = QPushButton("Draw polygon", self)
        self._draw_btn.setCheckable(True)
        self._draw_btn.setToolTip(
            "Toggle polygon-drawing mode. When ON, left-click adds a vertex "
            "and right-click removes the nearest. When OFF, left-clicks toggle "
            "radar sites."
        )
        self._draw_btn.toggled.connect(self._set_draw_mode)
        toolbar.addWidget(self._draw_btn)
        self._clear_btn = QPushButton("Clear polygon", self)
        self._clear_btn.clicked.connect(self.poly_editor.clear)
        toolbar.addWidget(self._clear_btn)
        self._reset_view_btn = QPushButton("Reset view", self)
        self._reset_view_btn.setToolTip("Return to the full-CONUS view.")
        self._reset_view_btn.clicked.connect(self._reset_view)
        toolbar.addWidget(self._reset_view_btn)
        self._status_label = QLabel(self._status_text(), self)
        toolbar.addWidget(self._status_label)
        toolbar.addStretch(1)
        self._continue_btn = QPushButton("Continue to time window →", self)
        self._continue_btn.setStyleSheet("font-weight: bold;")
        self._continue_btn.clicked.connect(self.continue_requested.emit)
        toolbar.addWidget(self._continue_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._plot, stretch=1)
        layout.addLayout(toolbar)

    # ---- static base layer -------------------------------------------

    def _draw_basemap(self) -> None:
        try:
            data = load_conus_lines_latlon()
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load CONUS basemap: %s", e)
            return
        if data["states"]:
            xs, ys = _concat_with_gaps(data["states"])
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#888", width=0.6),
                                     connect="finite")
            item.setZValue(1)
            self.view.addItem(item)
        if data["borders"]:
            xs, ys = _concat_with_gaps(data["borders"])
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#aaa", width=0.8),
                                     connect="finite")
            item.setZValue(1)
            self.view.addItem(item)
        if data["coastlines"]:
            xs, ys = _concat_with_gaps(data["coastlines"])
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#aaa", width=0.8),
                                     connect="finite")
            item.setZValue(1)
            self.view.addItem(item)

    # ---- reports -----------------------------------------------------

    def _draw_reports(self) -> None:
        if not self.reports:
            return
        times_sec = np.array([(r.time - self.reports[0].time).total_seconds()
                              for r in self.reports], dtype=np.float64)
        if times_sec.size > 1:
            tmin, tmax = float(times_sec.min()), float(times_sec.max())
        else:
            tmin, tmax = 0.0, 86400.0
        denom = max(tmax - tmin, 1.0)
        # One ScatterPlotItem per category so symbol can differ.
        for category in _REPORT_SYMBOLS:
            spots: list[dict] = []
            cat_reports: list[Report] = []
            for r, t in zip(self.reports, times_sec, strict=False):
                if r.category != category:
                    continue
                t_norm = (t - tmin) / denom
                fill = _time_to_color(t_norm)
                spots.append(dict(
                    pos=(r.lon, r.lat),
                    size=_marker_size(r.category, r.magnitude),
                    symbol=_REPORT_SYMBOLS[category],
                    pen=pg.mkPen(_REPORT_EDGE_COLORS[category], width=1.0),
                    brush=pg.mkBrush(fill),
                ))
                cat_reports.append(r)
            if not spots:
                continue
            scatter = pg.ScatterPlotItem(spots=spots, pxMode=True)
            scatter.setZValue(5)
            self.view.addItem(scatter)
            self._report_items.append(scatter)
            self._report_data.extend(cat_reports)

    # ---- radar sites -------------------------------------------------

    def _draw_radar_sites(self) -> None:
        spots: list[dict] = []
        site_index: dict[int, str] = {}
        self._site_kind_by_icao: dict[str, str] = {}
        for site in load_sites():
            if site.state in {"AK", "HI", "GU", "PR", "KR", "JP"}:
                continue
            self._site_kind_by_icao[site.icao] = site.kind
            color = self._color_for_site(site.icao, site.kind)
            self._site_brush_by_icao[site.icao] = color
            # TDWRs get a smaller circle marker to read as visually
            # subordinate to the long-range WSR-88D X's — they're
            # terminal-airport short-range radars, not WFO survey
            # radars, and the visual hierarchy should make that clear.
            if site.kind == SITE_KIND_TDWR:
                symbol, size, pen_width = "o", 9, 1.2
            else:
                symbol, size, pen_width = "x", 14, 1.8
            spots.append(dict(
                pos=(site.lon, site.lat),
                size=size, symbol=symbol,
                pen=pg.mkPen(color, width=pen_width),
                brush=pg.mkBrush(color),
                data=site.icao,
            ))
            site_index[len(spots) - 1] = site.icao
        if not spots:
            return
        # hoverable=True enlarges the hit area; clickable signal fires on
        # release-without-drag so a left-click toggles the site reliably.
        scatter = pg.ScatterPlotItem(spots=spots, pxMode=True, hoverable=True)
        scatter.setZValue(6)
        scatter.sigClicked.connect(self._on_site_scatter_clicked)
        self.view.addItem(scatter)
        self._site_scatter = scatter
        self._site_index = site_index
        # Kick off the archive availability probe so unavailable sites
        # get dimmed once the S3 listings return.
        self._probe_archive_availability()

    def _color_for_site(self, icao: str, kind: str) -> str:
        """Resolve the marker color for a site based on enabled / kind /
        archive availability. Order matters: unavailable beats both
        enabled and kind because the dim color signals "can't pick this
        radar at all for this day."""
        if icao in self._unavailable_sites:
            # Dim grey — visibly different from a regular disabled
            # WSR-88D so the host can tell at a glance "this radar has
            # no archive data for the chosen day" vs "this radar is
            # available but I haven't enabled it."
            return "#2a2a2a"
        if icao in self._enabled_sites:
            return "#00d4ff"
        # Disabled-but-available: TDWRs a slightly different hue than
        # WSR-88Ds so the kind hierarchy reads even when nothing is
        # enabled. (Lavender for TDWR vs neutral grey for WSR-88D.)
        return "#5a4e6b" if kind == SITE_KIND_TDWR else "#404040"

    def _refresh_site_colors(self) -> None:
        if self._site_scatter is None:
            return
        # Mutate each spot's brush/pen so colors update without rebuilding
        # the entire scatter.
        for i in range(len(self._site_scatter.points())):
            pt = self._site_scatter.points()[i]
            icao = self._site_index.get(i)
            if icao is None:
                continue
            kind = self._site_kind_by_icao.get(icao, "WSR88D")
            color = self._color_for_site(icao, kind)
            pt.setBrush(pg.mkBrush(color))
            pt.setPen(pg.mkPen(color, width=1.4))
            self._site_brush_by_icao[icao] = color

    def _probe_archive_availability(self) -> None:
        """Probe Unidata S3 for archive coverage of every site on the
        host's chosen day. Runs in the background — sites get marked
        unavailable + redrawn as probes return so the UI stays
        responsive. Without a day (e.g. live mode) this is a no-op:
        live radars are assumed broadcasting and any availability
        decision is handled by the live data source instead."""
        if self._day is None or self._site_scatter is None:
            return
        from concurrent.futures import ThreadPoolExecutor

        day = self._day
        # Probe TDWRs first since they're the motivating case for this
        # check; WSR-88Ds are appended after so older dates with
        # patchy WSR coverage also get dimmed.
        icaos: list[str] = []
        for site in load_sites():
            if site.state in {"AK", "HI", "GU", "PR", "KR", "JP"}:
                continue
            icaos.append(site.icao)
        # Use the shared module-level pool pattern — capped concurrency
        # keeps us under S3's per-second request limits per origin.
        if not hasattr(self.__class__, "_AVAIL_POOL"):
            self.__class__._AVAIL_POOL = ThreadPoolExecutor(
                max_workers=8, thread_name_prefix="archive-probe",
            )
        pool = self.__class__._AVAIL_POOL

        # Each completed probe writes to ``self._unavailable_sites`` on
        # the worker thread and pings a Qt timer to refresh colors on
        # the main thread (set widget refresh is NOT thread-safe).
        self._pending_probes = len(icaos)
        self._availability_timer = QTimer(self)
        self._availability_timer.setInterval(120)
        self._availability_timer.setSingleShot(False)
        self._availability_timer.timeout.connect(self._availability_tick)
        self._availability_timer.start()

        def _probe(icao: str) -> tuple[str, bool]:
            try:
                return icao, site_has_data_on_day(icao, day)
            except Exception:  # noqa: BLE001
                return icao, True  # fall back to "available" on error

        self._availability_futures = [pool.submit(_probe, ic) for ic in icaos]

    def _availability_tick(self) -> None:
        """Main-thread tick: drain any completed availability futures
        and update the scatter colors. Stops the timer when every
        probe has reported."""
        if not getattr(self, "_availability_futures", None):
            self._availability_timer.stop()
            return
        still_pending = []
        any_dirty = False
        for fut in self._availability_futures:
            if not fut.done():
                still_pending.append(fut)
                continue
            try:
                icao, has_data = fut.result(timeout=0.001)
            except Exception:  # noqa: BLE001
                continue
            if not has_data:
                self._unavailable_sites.add(icao)
                # Don't auto-enable an unavailable site if the host had
                # previously toggled it on for a different round.
                self._enabled_sites.discard(icao)
                any_dirty = True
        self._availability_futures = still_pending
        if any_dirty:
            self._refresh_site_colors()
            self._status_label.setText(self._status_text())
        if not still_pending:
            self._availability_timer.stop()

    # ---- mouse handlers ---------------------------------------------

    # _on_scene_clicked used to live here for the old pointsAt path; it
    # was replaced by ScatterPlotItem.sigClicked (more reliable — the
    # scene-level signal can be swallowed by ViewBox's pan handler).

    def _on_site_scatter_clicked(self, scatter, points, ev=None) -> None:
        """ScatterPlotItem.sigClicked handler — fires when a left-click
        lands on one of the radar-site Xs without becoming a drag. We
        gate on draw mode here so vertices can be placed on top of a site
        without accidentally toggling it."""
        if self._draw_mode:
            return
        if len(points) == 0:
            return
        # ev may be None in older pyqtgraphs; default to left-button.
        if ev is not None and hasattr(ev, "button"):
            if ev.button() != Qt.MouseButton.LeftButton:
                return
        icao = points[0].data()
        if not icao:
            return
        if icao in self._unavailable_sites:
            # Unavailable for the chosen day — refuse the toggle and
            # surface why in the status bar so the host isn't left
            # wondering why the click did nothing.
            self._status_label.setText(
                f"{icao}: no Level 2 archive data for the chosen day"
            )
            return
        if icao in self._enabled_sites:
            self._enabled_sites.discard(icao)
        else:
            self._enabled_sites.add(icao)
        self._refresh_site_colors()
        self._status_label.setText(self._status_text())
        self.radar_sites_changed.emit(set(self._enabled_sites))

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        if not self.view.sceneBoundingRect().contains(scene_pos):
            self._hover.hide()
            return
        try:
            pt = self.view.mapSceneToView(scene_pos)
        except Exception:  # noqa: BLE001
            return
        x, y = float(pt.x()), float(pt.y())
        # Hit-test reports (across all scatter items combined).
        view_w = self.view.viewRange()[0][1] - self.view.viewRange()[0][0]
        hit_radius = view_w * 0.008
        best: Report | None = None
        best_d = float("inf")
        for r in self._report_data:
            d = np.hypot(r.lon - x, r.lat - y)
            if d < hit_radius and d < best_d:
                best = r
                best_d = d
        if best is None:
            self._hover.hide()
            return
        self._hover.setText(_report_tooltip_text(best))
        self._hover.setPos(x, y)
        self._hover.show()

    def _reset_view(self) -> None:
        self.view.setRange(
            xRange=(self._home_extent[0], self._home_extent[1]),
            yRange=(self._home_extent[2], self._home_extent[3]),
            padding=0,
        )

    def _set_draw_mode(self, enabled: bool) -> None:
        self._draw_mode = enabled
        self.poly_editor.set_enabled(enabled)
        if enabled:
            self._plot.setCursor(Qt.CursorShape.CrossCursor)
            self._draw_btn.setText("◉  Drawing — click again to finish")
            self._draw_btn.setStyleSheet(
                "QPushButton { background: #ffd400; color: #000; font-weight: bold; }"
            )
        else:
            self._plot.unsetCursor()
            self._draw_btn.setText("Draw polygon")
            self._draw_btn.setStyleSheet("")
        self._status_label.setText(self._status_text())

    def _status_text(self) -> str:
        if self._draw_mode:
            hint = "click to add vertices — Draw button again to finish"
        elif self.poly_editor.polygon() is None:
            hint = "click radar Xs to enable; Draw polygon to start the boundary"
        else:
            hint = "polygon set"
        return (
            f"{len(self.reports)} reports loaded   "
            f"{len(self._enabled_sites)} radar(s) enabled   {hint}"
        )

    # ---- public ------------------------------------------------------

    def enabled_sites(self) -> set[str]:
        return set(self._enabled_sites)

    def set_enabled_sites(self, icaos: set[str]) -> None:
        self._enabled_sites = set(icaos)
        self._refresh_site_colors()

    def mark_sites_unavailable(self, icaos: set[str]) -> None:
        """Force the given sites into the 'unavailable for this day'
        bucket — used when the host returns from the prefetch widget
        after discovering some selected radars have zero archive data.
        Removes them from ``_enabled_sites`` so the same radars can't
        be re-attempted without an explicit click, and redraws."""
        added = set(icaos) - self._unavailable_sites
        if not added:
            return
        self._unavailable_sites.update(added)
        self._enabled_sites -= added
        self._refresh_site_colors()
        self._status_label.setText(self._status_text())

    def polygon(self) -> Polygon | None:
        return self.poly_editor.polygon()

    def replace_reports(self, reports: list[Report]) -> None:
        """Called after a reroll — clear polygon + radar selections and redraw."""
        for item in self._report_items:
            try:
                self.view.removeItem(item)
            except Exception:  # noqa: BLE001
                pass
        self._report_items.clear()
        self._report_data.clear()
        if self._site_scatter is not None:
            try:
                self.view.removeItem(self._site_scatter)
            except Exception:  # noqa: BLE001
                pass
            self._site_scatter = None
            self._site_index.clear()
        self._hover.hide()
        # Reroll = fresh start: the previous polygon and radar choices are
        # tied to a different day, so wipe them.
        self.poly_editor.clear()
        self._enabled_sites.clear()
        self.reports = reports
        self._draw_reports()
        self._draw_radar_sites()
        self._status_label.setText(self._status_text())
        self.polygon_changed.emit(None)
        self.radar_sites_changed.emit(set())
