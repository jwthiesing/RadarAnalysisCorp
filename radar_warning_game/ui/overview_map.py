"""CONUS overview map for round setup (plan §2).

Host-only setup screen. Shows:

  - All LSRs for the picked day, plotted with:
      * **shape** by category (▲ tornado, ● hail, ■ wind)
      * **size**  scaled by magnitude (hail in, wind mph, tornado EF)
      * **color** by time of day (perceptual sequential colormap)
  - WSR-88D sites as toggleable dots — host clicks to enable/disable for the
    round (down-radar simulation).
  - Freehand polygon drawing for the game-area boundary, via
    :class:`PolygonEditor`.
  - A "Reroll" button (random-day mode only) that requests a new day.

Emits Qt signals for changes; the parent (room/session UI) wires those to the
session state machine.
"""

from __future__ import annotations

import logging
from datetime import datetime

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.reports import Report
from ..data.sites import Site, load_sites
from ..geo.polygons import Polygon
from .poly_editor import PolygonEditor

log = logging.getLogger(__name__)

# Report-category visual specs
_REPORT_MARKERS = {
    "tornado": "^",
    "hail": "o",
    "wind": "s",
}

# Convert raw magnitude to scatter marker area (in points²)
def _marker_size(category: str, magnitude: float) -> float:
    if category == "tornado":
        # EF rating 0-5; minimum size for unknown
        ef = max(0.0, float(magnitude))
        return 30.0 + ef * 24.0
    if category == "hail":
        # inches
        return 12.0 + max(0.0, float(magnitude)) * 16.0
    if category == "wind":
        # mph; threshold 50, max ~150
        return 8.0 + max(0.0, float(magnitude) - 50.0) * 1.2
    return 12.0


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
    """

    radar_sites_changed = pyqtSignal(object)   # set[str]
    polygon_changed = pyqtSignal(object)        # Polygon | None
    reroll_requested = pyqtSignal()
    continue_requested = pyqtSignal()           # host clicked the Continue button

    def __init__(
        self,
        reports: list[Report],
        *,
        is_random_day: bool,
        initial_enabled_sites: set[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.reports = reports
        self.is_random_day = is_random_day
        self._enabled_sites: set[str] = set(initial_enabled_sites or set())
        self._site_artists: dict[str, "matplotlib.artist.Artist"] = {}

        self._figure = Figure(figsize=(10, 7), facecolor="#0a0a0a")
        self._canvas = FigureCanvasQTAgg(self._figure)
        self.ax = self._figure.add_subplot(111, projection=ccrs.PlateCarree())
        self.ax.set_facecolor("#0a0a0a")
        self.ax.set_extent([-125, -66, 23.5, 50], crs=ccrs.PlateCarree())
        self.ax.add_feature(cfeature.STATES.with_scale("50m"),
                            edgecolor="#888", linewidth=0.6, facecolor="none")
        self.ax.add_feature(cfeature.BORDERS.with_scale("50m"),
                            edgecolor="#aaa", linewidth=0.8)
        self.ax.add_feature(cfeature.COASTLINE.with_scale("50m"),
                            edgecolor="#aaa", linewidth=0.8)

        self._draw_reports()
        self._draw_radar_sites()
        # Polygon editor in PlateCarree (x=lon, y=lat) — lat/lon swap on conversion
        self.poly_editor = PolygonEditor(
            self.ax,
            axes_to_latlon=lambda x, y: (y, x),
            color="#ffd400",
        )
        self.poly_editor.polygon_changed.connect(self.polygon_changed.emit)

        # Click-to-toggle radar sites via separate event handler that runs
        # alongside the polygon clicker (the clicker only listens to picks on
        # its own artists; site dots are separate scatters)
        self._canvas.mpl_connect("pick_event", self._on_pick)

        # Toolbar / buttons
        toolbar = QHBoxLayout()
        self._reroll_btn = QPushButton("Reroll random day", self)
        self._reroll_btn.setEnabled(self.is_random_day)
        self._reroll_btn.clicked.connect(self.reroll_requested.emit)
        toolbar.addWidget(self._reroll_btn)
        self._clear_btn = QPushButton("Clear polygon", self)
        self._clear_btn.clicked.connect(self.poly_editor.clear)
        toolbar.addWidget(self._clear_btn)
        self._status_label = QLabel(self._status_text(), self)
        toolbar.addWidget(self._status_label)
        toolbar.addStretch(1)
        self._continue_btn = QPushButton("Continue to time window →", self)
        self._continue_btn.setStyleSheet("font-weight: bold;")
        self._continue_btn.clicked.connect(self.continue_requested.emit)
        toolbar.addWidget(self._continue_btn)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self._canvas, stretch=1)
        layout.addLayout(toolbar)

    # ---- drawing -------------------------------------------------------

    def _draw_reports(self) -> None:
        if not self.reports:
            return
        # Color by time of day (seconds since min)
        times_sec = np.array([(r.time - self.reports[0].time).total_seconds()
                              for r in self.reports])
        # Normalize over the actual range so colors span the day
        if times_sec.size > 1:
            tmin, tmax = float(times_sec.min()), float(times_sec.max())
        else:
            tmin, tmax = 0.0, 86400.0
        norm = Normalize(vmin=tmin, vmax=tmax)
        cmap = "viridis"
        for category, marker in _REPORT_MARKERS.items():
            xs, ys, sizes, ts = [], [], [], []
            for r, t in zip(self.reports, times_sec, strict=False):
                if r.category != category:
                    continue
                xs.append(r.lon)
                ys.append(r.lat)
                sizes.append(_marker_size(category, r.magnitude))
                ts.append(t)
            if not xs:
                continue
            self.ax.scatter(
                xs, ys, s=sizes, c=ts, cmap=cmap, norm=norm, marker=marker,
                edgecolors="#000", linewidths=0.4, alpha=0.85,
                transform=ccrs.PlateCarree(), zorder=5,
            )

    def _draw_radar_sites(self) -> None:
        for site in load_sites():
            if site.state in {"AK", "HI", "GU", "PR", "KR", "JP"}:
                continue
            enabled = site.icao in self._enabled_sites
            color = "#00d4ff" if enabled else "#404040"
            artist = self.ax.scatter(
                [site.lon], [site.lat], s=20, c=color, marker="x",
                linewidths=1.4, transform=ccrs.PlateCarree(), zorder=6,
                picker=True, pickradius=6,
            )
            artist.set_label(site.icao)
            self._site_artists[site.icao] = artist

    def _on_pick(self, event) -> None:
        artist = event.artist
        icao = artist.get_label()
        if icao in self._enabled_sites:
            self._enabled_sites.discard(icao)
            artist.set_color("#404040")
        else:
            self._enabled_sites.add(icao)
            artist.set_color("#00d4ff")
        self._canvas.draw_idle()
        self._status_label.setText(self._status_text())
        self.radar_sites_changed.emit(set(self._enabled_sites))

    def _status_text(self) -> str:
        return (
            f"{len(self.reports)} reports loaded   "
            f"{len(self._enabled_sites)} radar(s) enabled   "
            f"{'click radars + click map to draw polygon' if not self.poly_editor.polygon() else 'polygon set'}"
        )

    # ---- public --------------------------------------------------------

    def enabled_sites(self) -> set[str]:
        return set(self._enabled_sites)

    def set_enabled_sites(self, icaos: set[str]) -> None:
        for icao, artist in self._site_artists.items():
            on = icao in icaos
            artist.set_color("#00d4ff" if on else "#404040")
        self._enabled_sites = set(icaos)
        self._canvas.draw_idle()

    def polygon(self) -> Polygon | None:
        return self.poly_editor.polygon()

    def replace_reports(self, reports: list[Report]) -> None:
        """Called after a reroll — clear polygon + radar selections and redraw."""
        # Wipe old report scatters (artists added in _draw_reports)
        for artist in list(self.ax.collections):
            artist.remove()
        self._site_artists.clear()
        # Reroll = fresh start: the previous polygon and radar choices are
        # tied to a different day, so wipe them.
        self.poly_editor.clear()
        self._enabled_sites.clear()
        self.reports = reports
        self._draw_reports()
        self._draw_radar_sites()
        self._status_label.setText(self._status_text())
        self._canvas.draw_idle()
        # Notify any wiring that depends on the cleared state
        self.polygon_changed.emit(None)
        self.radar_sites_changed.emit(set())
