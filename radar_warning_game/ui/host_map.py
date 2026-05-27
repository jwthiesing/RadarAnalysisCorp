"""Host central map — the host's primary in-game view (plan §4c).

A single pyqtgraph view centered on the game polygon, showing:

  - The game polygon boundary (heavy outline)
  - All enabled WSR-88D site markers
  - **Every player's warning polygons** as outlines (no fill — readable when
    many overlap). Color by team, line style by warning family:
        solid  = TOR / TORR / PDS TOR / TORE
        dashed = SVR / SVRC / SVRD
        dotted = MCD
    Line weight scales with tier (PDS TOR > TORR > TOR).
  - Live storm reports per §6 fade rules (categorical symbol, magnitude size).
  - Docked :class:`LiveLeaderboardWidget` in a corner.
  - A side panel with details on a clicked polygon.
  - "Join as player" button to open the standard player gameplay window.

Coordinates are plain ``(lon, lat)``. State / country / coastline lines are
loaded from Natural Earth and drawn as flat curves — fine for the CONUS
scale this widget covers without needing a true projection.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..data.reports import Report
from ..data.sites import site_by_icao
from ..game.session import GameSession, SOLO_TEAM_PREFIX
from ..verification.reports_in_poly import MCD, Warning
from ..verification.tornado_tiers import WarningType
from .colors import color_for_team
from .leaderboard import LiveLeaderboardWidget
from .overlay_loader import load_conus_lines_latlon
from .radar_panel import _report_tooltip_text, _concat_with_gaps
from .time_format import format_player_time, format_player_time_short

log = logging.getLogger(__name__)

# Per-tier line weight (visual hierarchy)
_TIER_LINEWIDTH = {
    WarningType.SVR: 1.2,
    WarningType.SVRC: 1.6,
    WarningType.SVRD: 2.0,
    WarningType.TOR: 1.4,
    WarningType.TORR: 1.8,
    WarningType.PDS_TOR: 2.4,
    WarningType.TORE: 3.0,
}

_REPORT_SYMBOLS = {"tornado": "t1", "hail": "o", "wind": "s"}
_REPORT_EDGE_COLORS = {
    "tornado": "#ff3030",
    "hail":    "#22cc55",
    "wind":    "#3399ff",
}
_REPORT_FILL_COLORS = {"tornado": "#ff4444", "hail": "#44ff66", "wind": "#66bbff"}

REPORT_FADE_SEC = 30 * 60   # full fade after 30 game-minutes (plan §6)


class HostCentralMap(QWidget):
    """Host's in-game overview map."""

    request_join_as_player = pyqtSignal()

    def __init__(self, session: GameSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        # Item bookkeeping — wiped on every refresh().
        self._warning_items: list = []
        self._report_items: list = []
        self._report_data: list[Report] = []
        # Sidecar maps for the hover tooltip — each polygon outline's
        # (x_array, y_array, hover_text) so the mouse-move handler can
        # answer "is the cursor over a warning line, and if so what does
        # it say?" cheaply.
        self._hoverable_polygons: list[tuple[np.ndarray, np.ndarray, str]] = []

        # Plot widget — flat lon/lat coordinates.
        self._plot = pg.PlotWidget(parent=self)
        self._plot.setBackground("#0a0a0a")
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.setMenuEnabled(False)
        self.view: pg.ViewBox = self._plot.getViewBox()
        # Lock aspect so the CONUS shape isn't visibly stretched at typical
        # latitudes (note: this is still an unprojected lon/lat plot, so a
        # one-degree-lon ≠ one-degree-lat pixel ratio at CONUS lats; the
        # distortion is small enough to be tolerable for an overview map).
        self.view.setAspectLocked(True, ratio=1.0)
        self.view.setMenuEnabled(False)

        # CONUS base layer (state lines, country borders, coastlines) drawn
        # in low-saturation grey; never refreshed since it's static.
        self._draw_basemap()
        self._draw_game_polygon()
        self._draw_radar_sites()

        # Hover tooltip text item — single instance reused per mousemove.
        self._hover = pg.TextItem(
            "", anchor=(0, 1), color="#0a0a0a",
            fill=pg.mkBrush(QColor("#ffd400")),
            border=pg.mkPen("#000", width=0.6),
        )
        self._hover.setZValue(30)
        self._hover.hide()
        self.view.addItem(self._hover, ignoreBounds=True)

        # Side panel
        self._details_label = QLabel("Click a warning polygon for details", self)
        self._details_label.setWordWrap(True)
        self._details_label.setStyleSheet("color: #ccc; padding: 6px;")
        self._details_box = QTextEdit(self)
        self._details_box.setReadOnly(True)
        self._details_box.setStyleSheet(
            "background: #111; color: #eee; border: 1px solid #333;"
        )
        self._details_box.setMaximumWidth(280)
        self.leaderboard = LiveLeaderboardWidget(local_team_id=None, parent=self)
        self._join_btn = QPushButton("Join as player", self)
        self._join_btn.clicked.connect(self.request_join_as_player.emit)

        side_layout = QVBoxLayout()
        side_layout.addWidget(self._join_btn)
        side_layout.addWidget(self.leaderboard)
        side_layout.addWidget(self._details_label)
        side_layout.addWidget(self._details_box, stretch=1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._plot, stretch=1)
        layout.addLayout(side_layout, stretch=0)

        # Warning / MCD polygons are individually clickable (wired in
        # _draw_warnings / _draw_mcds via PlotCurveItem.setClickable).
        # The scene-level mouseMoved signal still drives the report hover.
        self._plot.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

        self.refresh()

    # ---- public --------------------------------------------------------

    def refresh(self) -> None:
        """Re-render warning polygons, MCDs, reports, and leaderboard.
        Called every tick AND on warning/MCD events."""
        self._clear_dynamic_items()
        self._draw_warnings()
        self._draw_mcds()
        self._draw_reports()
        scores = self.session.current_scores()
        self.leaderboard.refresh(scores, self.session.team_names)

    # ---- static layers -----------------------------------------------

    def _draw_basemap(self) -> None:
        try:
            data = load_conus_lines_latlon()
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load CONUS basemap: %s", e)
            return
        # States — solid grey.
        if data["states"]:
            xs, ys = _concat_with_gaps(data["states"])
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#888", width=0.6),
                                     connect="finite")
            item.setZValue(1)
            self.view.addItem(item)
        # Country borders — lighter grey, slightly thicker.
        if data["borders"]:
            xs, ys = _concat_with_gaps(data["borders"])
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#aaa", width=0.9),
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
        # Auto-zoom with margin.
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
            x=xs, y=ys, size=12, symbol="x",
            pen=pg.mkPen("#00d4ff", width=2.0),
            brush=pg.mkBrush("#00d4ff"),
            pxMode=True,
        )
        scatter.setZValue(9)
        self.view.addItem(scatter)
        for icao, x, y in zip(names, xs, ys):
            label = pg.TextItem(icao, anchor=(0, 1), color="#00d4ff")
            label.setPos(x + 0.1, y + 0.1)
            label.setZValue(10)
            self.view.addItem(label, ignoreBounds=True)

    def _clear_dynamic_items(self) -> None:
        for item in self._warning_items + self._report_items:
            try:
                self.view.removeItem(item)
            except Exception:  # noqa: BLE001
                pass
        self._warning_items.clear()
        self._report_items.clear()
        self._report_data.clear()
        self._hoverable_polygons.clear()
        if self._hover.isVisible():
            self._hover.hide()

    # ---- per-tick layers ---------------------------------------------

    def _draw_warnings(self) -> None:
        for issuer_id, warnings in self.session.warnings_by_player.items():
            player = self.session.players.get(issuer_id)
            team_id = player.team_id if player else f"{SOLO_TEAM_PREFIX}{issuer_id}"
            color = color_for_team(team_id or issuer_id)
            initials = _initials(player.display_name) if player else issuer_id[:2]
            for w in warnings:
                if not _warning_is_visible(w, self.session.clock):
                    continue
                rev = w.current_revision
                lw = _TIER_LINEWIDTH.get(rev.warning_type, 1.4)
                dash = (Qt.PenStyle.SolidLine if rev.warning_type.is_tornado_family
                        else Qt.PenStyle.DashLine)
                verts = list(rev.polygon.vertices) + [rev.polygon.vertices[0]]
                lons = np.array([v[1] for v in verts], dtype=np.float64)
                lats = np.array([v[0] for v in verts], dtype=np.float64)
                lookup = ("warning", issuer_id, w.warning_id)
                hover_text = _warning_hover_text(
                    w, self.session.clock,
                    name=player.display_name if player else issuer_id,
                )
                if rev.warning_type == WarningType.TORE:
                    # TORE on the host map keeps the team color as the wide
                    # outer halo (so it's still attributable to the issuing
                    # team) with a thinner black inner stroke layered on
                    # top — most-significant tier reads at a glance.
                    outer = pg.PlotCurveItem(
                        lons, lats,
                        pen=pg.mkPen(color=color, width=lw + 3.0, style=dash),
                    )
                    outer.setZValue(12)
                    outer.setClickable(True, width=8)
                    outer.sigClicked.connect(
                        lambda _curve, _ev, info=lookup: self._show_details(*info)
                    )
                    self.view.addItem(outer)
                    self._warning_items.append(outer)
                    inner = pg.PlotCurveItem(
                        lons, lats,
                        pen=pg.mkPen(color="#000000", width=lw, style=dash),
                    )
                    inner.setZValue(13)
                    inner.setClickable(True, width=8)
                    inner.sigClicked.connect(
                        lambda _curve, _ev, info=lookup: self._show_details(*info)
                    )
                    self.view.addItem(inner)
                    self._warning_items.append(inner)
                    self._hoverable_polygons.append((lons, lats, hover_text))
                else:
                    pen = pg.mkPen(color=color, width=lw, style=dash)
                    item = pg.PlotCurveItem(lons, lats, pen=pen)
                    item.setZValue(12)
                    # Make the curve respond to clicks directly — more reliable
                    # than scene-level sigMouseClicked + manual hit-testing,
                    # since the latter gets swallowed by ViewBox's pan handler.
                    item.setClickable(True, width=8)
                    item.sigClicked.connect(
                        lambda _curve, _ev, info=lookup: self._show_details(*info)
                    )
                    self.view.addItem(item)
                    self._warning_items.append(item)
                    self._hoverable_polygons.append((lons, lats, hover_text))
                # Centroid label.
                clat, clon = rev.polygon.centroid_latlon
                label = pg.TextItem(
                    f"{initials} {rev.warning_type.value}",
                    anchor=(0.5, 0.5), color=color,
                )
                label.setPos(clon, clat)
                label.setZValue(13)
                self.view.addItem(label, ignoreBounds=True)
                self._warning_items.append(label)

    def _draw_mcds(self) -> None:
        for issuer_id, mcds in self.session.mcds_by_player.items():
            player = self.session.players.get(issuer_id)
            team_id = player.team_id if player else f"{SOLO_TEAM_PREFIX}{issuer_id}"
            color = color_for_team(team_id or issuer_id)
            initials = _initials(player.display_name) if player else issuer_id[:2]
            for m in mcds:
                if not _mcd_is_visible(m, self.session.clock):
                    continue
                verts = list(m.polygon.vertices) + [m.polygon.vertices[0]]
                lons = np.array([v[1] for v in verts], dtype=np.float64)
                lats = np.array([v[0] for v in verts], dtype=np.float64)
                pen = pg.mkPen(color=color, width=1.0, style=Qt.PenStyle.DotLine)
                item = pg.PlotCurveItem(lons, lats, pen=pen)
                item.setZValue(11)
                item.setClickable(True, width=8)
                lookup = ("mcd", issuer_id, m.mcd_id)
                item.sigClicked.connect(
                    lambda _curve, _ev, info=lookup: self._show_details(*info)
                )
                self.view.addItem(item)
                self._warning_items.append(item)
                self._hoverable_polygons.append(
                    (lons, lats, _mcd_hover_text(m, self.session.clock, name=player.display_name if player else issuer_id))
                )
                clat, clon = m.polygon.centroid_latlon
                label = pg.TextItem(
                    f"{initials} MCD", anchor=(0.5, 0.5), color=color,
                )
                label.setPos(clon, clat)
                label.setZValue(12)
                self.view.addItem(label, ignoreBounds=True)
                self._warning_items.append(label)

    def _show_details(self, kind: str, owner_id: str, item_id: str) -> None:
        """Render the right-hand side panel for a clicked warning / MCD."""
        self._details_box.setHtml(_format_details(self.session, kind, owner_id, item_id))

    def _draw_reports(self) -> None:
        if self.session.clock is None or self.session.round_day is None:
            return
        now = self.session.clock.virtual_time
        spots: list[dict] = []
        kept: list[Report] = []
        for r in self.session.round_day.reports:
            if r.time > now:
                continue
            age_sec = (now - r.time).total_seconds()
            if age_sec > REPORT_FADE_SEC * 1.5:
                continue
            alpha = max(0.15, 1.0 - age_sec / REPORT_FADE_SEC)
            fill = QColor(_REPORT_FILL_COLORS[r.category])
            fill.setAlphaF(alpha)
            edge = QColor(_REPORT_EDGE_COLORS[r.category])
            edge.setAlphaF(alpha)
            spots.append(dict(
                pos=(r.lon, r.lat),
                size=_report_size(r.category, r.magnitude),
                symbol=_REPORT_SYMBOLS[r.category],
                pen=pg.mkPen(edge, width=1.0),
                brush=pg.mkBrush(fill),
            ))
            kept.append(r)
        if not spots:
            return
        scatter = pg.ScatterPlotItem(spots=spots, pxMode=True)
        scatter.setZValue(6)
        self.view.addItem(scatter)
        self._report_items.append(scatter)
        self._report_data = kept

    # ---- mouse handlers ---------------------------------------------

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        if not self.view.sceneBoundingRect().contains(scene_pos):
            self._hover.hide()
            return
        try:
            pt = self.view.mapSceneToView(scene_pos)
        except Exception:  # noqa: BLE001
            return
        x, y = float(pt.x()), float(pt.y())
        view_w = self.view.viewRange()[0][1] - self.view.viewRange()[0][0]
        # Reports first (a specific point under the cursor is the most
        # interesting target). Then warning / MCD polygon outlines.
        report_radius = view_w * 0.012
        best_report: Report | None = None
        best_d = float("inf")
        for r in self._report_data:
            d = np.hypot(r.lon - x, r.lat - y)
            if d < report_radius and d < best_d:
                best_report = r
                best_d = d
        if best_report is not None:
            self._hover.setText(_report_tooltip_text(best_report))
            self._hover.setPos(x, y)
            self._hover.show()
            return
        # Polygon outline hover — same loose-radius hit test as the click
        # path, walking each stored polyline's vertex-segment distance.
        poly_radius = view_w * 0.012
        best_text: str | None = None
        best_pd = float("inf")
        for xs, ys, text in self._hoverable_polygons:
            d = float(np.min(np.hypot(xs - x, ys - y)))
            if d < poly_radius and d < best_pd:
                best_text = text
                best_pd = d
        if best_text is None:
            self._hover.hide()
            return
        self._hover.setText(best_text)
        self._hover.setPos(x, y)
        self._hover.show()


# ---------------------------- helpers -----------------------------------------

def _format_duration(td: timedelta) -> str:
    """Compact ``HhMm`` or ``Mm`` representation of a positive timedelta."""
    total_sec = max(0, int(td.total_seconds()))
    if total_sec < 60:
        return f"{total_sec}s"
    minutes = total_sec // 60
    if minutes < 60:
        return f"{minutes}m"
    h, m = divmod(minutes, 60)
    return f"{h}h {m}m"


def _warning_hover_text(w: Warning, clock, *, name: str) -> str:
    """Hover-tooltip string for a warning polygon: type, issuer, issuance
    time, time-to-expiration, and magnitude tags."""
    rev = w.current_revision
    issued = format_player_time(w.original_issue_time)
    ends_dt = w.end_time()
    ends = format_player_time(ends_dt)
    lines = [f"{rev.warning_type.value}  —  {name}", f"Issued {issued}"]
    if clock is not None:
        now = clock.virtual_time
        if now < w.original_issue_time:
            lines.append(f"Expires {ends}")
        elif w.canceled_at is not None and now > w.canceled_at:
            lines.append(f"Canceled at {format_player_time(w.canceled_at)}")
        elif now > ends_dt:
            lines.append(f"Expired {ends}")
        else:
            remaining = ends_dt - now
            lines.append(f"Expires {ends}  ({_format_duration(remaining)} left)")
    else:
        lines.append(f"Expires {ends}")
    mags: list[str] = []
    if rev.magnitudes.hail_in is not None:
        mags.append(f"hail {rev.magnitudes.hail_in:.2f}\"")
    if rev.magnitudes.wind_mph is not None:
        mags.append(f"wind {int(rev.magnitudes.wind_mph)} mph")
    if rev.magnitudes.ef is not None:
        mags.append(f"EF{int(rev.magnitudes.ef)}")
    if getattr(rev.magnitudes, "tornado_possible", False):
        mags.append("tornado possible")
    if mags:
        lines.append(", ".join(mags))
    return "\n".join(lines)


def _mcd_hover_text(m: MCD, clock, *, name: str) -> str:
    issued = format_player_time(m.issue_time)
    ends_dt = m.end_time()
    ends = format_player_time(ends_dt)
    lines = [f"MCD  —  {name}", f"Issued {issued}"]
    if clock is not None:
        now = clock.virtual_time
        if now < m.issue_time:
            lines.append(f"Expires {ends}")
        elif m.canceled_at is not None and now > m.canceled_at:
            lines.append(f"Canceled at {format_player_time(m.canceled_at)}")
        elif now > ends_dt:
            lines.append(f"Expired {ends}")
        else:
            remaining = ends_dt - now
            lines.append(f"Expires {ends}  ({_format_duration(remaining)} left)")
    else:
        lines.append(f"Expires {ends}")
    pibs: list[str] = []
    if m.pib_tornado:
        pibs.append(f"Tor PIB {m.pib_tornado}")
    if m.pib_wind:
        pibs.append(f"Wind PIB {m.pib_wind}")
    if m.pib_hail:
        pibs.append(f"Hail PIB {m.pib_hail}")
    if pibs:
        lines.append(", ".join(pibs))
    return "\n".join(lines)


def _initials(name: str) -> str:
    parts = name.split()
    return "".join(p[0].upper() for p in parts[:2]) or name[:2].upper()


def _warning_is_visible(w: Warning, clock) -> bool:
    if clock is None:
        return True
    now = clock.virtual_time
    return w.is_active_at(now)


def _mcd_is_visible(m: MCD, clock) -> bool:
    if clock is None:
        return True
    now = clock.virtual_time
    if now < m.issue_time:
        return False
    return now <= m.end_time()


def _report_size(category: str, magnitude: float) -> float:
    """Pixel-space marker size (pxMode scatter)."""
    if category == "tornado":
        return 14.0 + max(0.0, float(magnitude)) * 4.0
    if category == "hail":
        return 9.0 + max(0.0, float(magnitude)) * 3.0
    if category == "wind":
        return 8.0 + max(0.0, float(magnitude) - 50.0) * 0.15
    return 9.0


def _format_details(session: GameSession, kind: str, owner_id: str, item_id: str) -> str:
    player = session.players.get(owner_id)
    name = player.display_name if player else owner_id
    if kind == "warning":
        for w in session.warnings_by_player.get(owner_id, []):
            if w.warning_id != item_id:
                continue
            rev = w.current_revision
            issued = format_player_time(w.original_issue_time)
            ends = format_player_time(w.end_time())
            mags = []
            if rev.magnitudes.hail_in is not None:
                mags.append(f"hail {rev.magnitudes.hail_in:.2f}\"")
            if rev.magnitudes.wind_mph is not None:
                mags.append(f"wind {int(rev.magnitudes.wind_mph)} mph")
            if rev.magnitudes.ef is not None:
                mags.append(f"EF{int(rev.magnitudes.ef)}")
            mag_str = ", ".join(mags) if mags else "(none specified)"
            return (
                f"<b>{rev.warning_type.value}</b> &nbsp; by {name}<br>"
                f"Issued: {issued}<br>"
                f"Expires: {ends}<br>"
                f"Expected: {mag_str}<br>"
                f"Revisions: {len(w.revisions)}"
                + (f"<br><span style='color:#888'>Canceled at {format_player_time(w.canceled_at)}</span>"
                   if w.canceled_at else "")
            )
    if kind == "mcd":
        for m in session.mcds_by_player.get(owner_id, []):
            if m.mcd_id != item_id:
                continue
            issued = format_player_time(m.issue_time)
            ends = format_player_time(m.end_time())
            pibs = []
            if m.pib_tornado:
                pibs.append(f"Tor PIB {m.pib_tornado}")
            if m.pib_wind:
                pibs.append(f"Wind PIB {m.pib_wind}")
            if m.pib_hail:
                pibs.append(f"Hail PIB {m.pib_hail}")
            return (
                f"<b>MCD</b> &nbsp; by {name}<br>"
                f"Issued: {issued}<br>"
                f"Expires: {ends}<br>"
                f"PIBs: {', '.join(pibs) if pibs else '(none)'}"
            )
    return "(no details)"
