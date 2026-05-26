"""Host central map — the host's primary in-game view (plan §4c).

A single Cartopy map view centered on the game polygon, showing:

  - The game polygon boundary (heavy outline)
  - All enabled WSR-88D site markers
  - **Every player's warning polygons** as outlines (no fill — readable when
    many overlap). Color by team, line style by warning family:
        solid  = TOR / TORR / PDS TOR / TORE
        dashed = SVR / SVRC / SVRD
        dotted = MCD
    Line weight scales with tier (PDS TOR > TORR > TOR).
  - Live storm reports per §6 fade rules (categorical shape, magnitude size).
  - Docked :class:`LiveLeaderboardWidget` in a corner.
  - A side panel with details on a clicked polygon.
  - "Join as player" button to open the standard player gameplay window.

This widget reads from a :class:`GameSession` and refreshes when called by the
parent (typically on every tick + on warning/MCD events).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
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

# Report visual specs (same scheme as OverviewMap, but with fade alpha)
_REPORT_MARKERS = {"tornado": "^", "hail": "o", "wind": "s"}

REPORT_FADE_SEC = 30 * 60   # full fade after 30 game-minutes (plan §6)


class HostCentralMap(QWidget):
    """Host's in-game overview map.

    Signals
    -------
    request_join_as_player
        emitted when the host clicks "Join as player"
    """

    request_join_as_player = pyqtSignal()

    def __init__(self, session: GameSession, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.session = session
        self._warning_artists: list = []
        self._report_artists: list = []
        self._site_artists: dict[str, "matplotlib.artist.Artist"] = {}
        self._polygon_lookup: dict = {}     # artist → (kind, owner_id, id) for click handling

        # Map canvas
        self._figure = Figure(figsize=(10, 7), facecolor="#0a0a0a")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self.ax = self._figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.ax.set_facecolor("#0a0a0a")
        self.ax.add_feature(cfeature.STATES.with_scale("50m"),
                            edgecolor="#888", linewidth=0.6, facecolor="none")
        self.ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                            edgecolor="#aaa", linewidth=0.8)

        # Game-polygon boundary + auto-zoom
        self._draw_game_polygon()
        self._draw_radar_sites()

        # Side panel (polygon details)
        self._details_label = QLabel("Click a warning polygon for details", self)
        self._details_label.setWordWrap(True)
        self._details_label.setStyleSheet("color: #ccc; padding: 6px;")
        self._details_box = QTextEdit(self)
        self._details_box.setReadOnly(True)
        self._details_box.setStyleSheet(
            "background: #111; color: #eee; border: 1px solid #333;"
        )
        self._details_box.setMaximumWidth(280)

        # Leaderboard
        self.leaderboard = LiveLeaderboardWidget(local_team_id=None, parent=self)

        # Toolbar
        self._join_btn = QPushButton("Join as player", self)
        self._join_btn.clicked.connect(self.request_join_as_player.emit)

        # Layout: map on left, side column on right
        side_layout = QVBoxLayout()
        side_layout.addWidget(self._join_btn)
        side_layout.addWidget(self.leaderboard)
        side_layout.addWidget(self._details_label)
        side_layout.addWidget(self._details_box, stretch=1)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(side_layout, stretch=0)

        # Pick handler
        self._canvas.mpl_connect("pick_event", self._on_pick)

        # Initial refresh
        self.refresh()

    # ---- public --------------------------------------------------------

    def refresh(self) -> None:
        """Re-render warning polygons, reports, and leaderboard.

        Called on every tick AND on warning/MCD events.
        """
        self._clear_dynamic_artists()
        self._draw_warnings()
        self._draw_mcds()
        self._draw_reports()
        scores = self.session.current_scores()
        self.leaderboard.refresh(scores, self.session.team_names)
        self._canvas.draw_idle()

    # ---- drawing -------------------------------------------------------

    def _draw_game_polygon(self) -> None:
        cfg = self.session.round_config
        if cfg is None:
            return
        verts = list(cfg.game_polygon.vertices) + [cfg.game_polygon.vertices[0]]
        lons = [v[1] for v in verts]
        lats = [v[0] for v in verts]
        self.ax.plot(lons, lats, color="#ffcc00", linewidth=2.0,
                     transform=ccrs.PlateCarree(), zorder=8)
        # Auto-zoom with margin
        pad = 0.5
        self.ax.set_extent([min(lons)-pad, max(lons)+pad, min(lats)-pad, max(lats)+pad],
                           crs=ccrs.PlateCarree())

    def _draw_radar_sites(self) -> None:
        cfg = self.session.round_config
        if cfg is None:
            return
        for icao in cfg.radar_sites:
            site = site_by_icao(icao)
            if site is None:
                continue
            artist = self.ax.scatter(
                [site.lon], [site.lat], s=80, marker="x", c="#00d4ff",
                linewidths=2.0, transform=ccrs.PlateCarree(), zorder=9,
            )
            self._site_artists[icao] = artist
            self.ax.annotate(
                icao, (site.lon, site.lat), color="#00d4ff", fontsize=9,
                xytext=(6, 4), textcoords="offset points",
                transform=ccrs.PlateCarree(), zorder=10,
            )

    def _clear_dynamic_artists(self) -> None:
        for a in self._warning_artists + self._report_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._warning_artists.clear()
        self._report_artists.clear()
        self._polygon_lookup.clear()

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
                linestyle = "-" if rev.warning_type.is_tornado_family else "--"
                verts = list(rev.polygon.vertices) + [rev.polygon.vertices[0]]
                lons = [v[1] for v in verts]
                lats = [v[0] for v in verts]
                line, = self.ax.plot(
                    lons, lats, color=color, linewidth=lw, linestyle=linestyle,
                    transform=ccrs.PlateCarree(), zorder=12,
                    picker=True, pickradius=6,
                )
                self._warning_artists.append(line)
                self._polygon_lookup[line] = ("warning", issuer_id, w.warning_id)
                # Label near centroid
                clat, clon = rev.polygon.centroid_latlon
                label = self.ax.text(
                    clon, clat, f"{initials} {rev.warning_type.value}",
                    color=color, fontsize=8, fontweight="bold",
                    ha="center", va="center",
                    transform=ccrs.PlateCarree(), zorder=13,
                )
                self._warning_artists.append(label)

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
                lons = [v[1] for v in verts]
                lats = [v[0] for v in verts]
                line, = self.ax.plot(
                    lons, lats, color=color, linewidth=1.0, linestyle=":",
                    transform=ccrs.PlateCarree(), zorder=11,
                    picker=True, pickradius=6,
                )
                self._warning_artists.append(line)
                self._polygon_lookup[line] = ("mcd", issuer_id, m.mcd_id)
                clat, clon = m.polygon.centroid_latlon
                self._warning_artists.append(self.ax.text(
                    clon, clat, f"{initials} MCD",
                    color=color, fontsize=7, ha="center", va="center",
                    transform=ccrs.PlateCarree(), zorder=12,
                ))

    def _draw_reports(self) -> None:
        if self.session.clock is None or self.session.round_day is None:
            return
        now = self.session.clock.virtual_time
        for category, marker in _REPORT_MARKERS.items():
            xs, ys, sizes, alphas = [], [], [], []
            for r in self.session.round_day.reports:
                if r.category != category or r.time > now:
                    continue
                age_sec = (now - r.time).total_seconds()
                if age_sec > REPORT_FADE_SEC * 1.5:
                    continue
                alpha = max(0.15, 1.0 - age_sec / REPORT_FADE_SEC)
                xs.append(r.lon)
                ys.append(r.lat)
                sizes.append(_report_size(category, r.magnitude))
                alphas.append(alpha)
            if not xs:
                continue
            for x, y, s, a in zip(xs, ys, sizes, alphas, strict=True):
                artist = self.ax.scatter(
                    [x], [y], s=s, c=_report_color(category), marker=marker,
                    alpha=a, edgecolors="#000", linewidths=0.4,
                    transform=ccrs.PlateCarree(), zorder=6,
                )
                self._report_artists.append(artist)

    def _on_pick(self, event) -> None:
        artist = event.artist
        info = self._polygon_lookup.get(artist)
        if info is None:
            return
        kind, owner_id, item_id = info
        self._details_box.setHtml(_format_details(self.session, kind, owner_id, item_id))


# ---------------------------- helpers -----------------------------------------

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


def _report_color(category: str) -> str:
    return {"tornado": "#ff4444", "hail": "#44ff66", "wind": "#66bbff"}[category]


def _report_size(category: str, magnitude: float) -> float:
    if category == "tornado":
        return 60.0 + max(0.0, float(magnitude)) * 30.0
    if category == "hail":
        return 30.0 + max(0.0, float(magnitude)) * 28.0
    if category == "wind":
        return 25.0 + max(0.0, float(magnitude) - 50.0) * 1.5
    return 30.0


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
