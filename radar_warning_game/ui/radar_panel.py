"""Multi-panel radar display with SAILS-aware scrubbing (plan §4a).

A ``RadarPanelGrid`` widget holds 1, 2, or 4 ``RadarPanel`` instances, each
showing the same radar volume at the same elevation/time but a different product
(REF / VEL / CC / ZDR / KDP / SW). Axes are shared so pan and zoom apply to all
panels at once — matches Reference-Nowcastle's [radar_plot.py](../../Reference-Nowcastle/radar_game/radar_plot.py) idea but in PyQt6.

Keyboard, on the focused panel (or any panel — controls apply globally to the
volume time/elevation, only the *product* is per-panel):

  ``↑`` / ``↓``    next / previous elevation tilt
  ``←`` / ``→``    previous / next sweep at current elevation (SAILS-aware)
  ``Shift+←/→``    step 5 sweeps
  ``1``…``7``      change focused panel's product (REF/VEL/CC/ZDR/KDP/HCA/SW)
  ``Space``        reset view (recenters on radar)
  Mouse scroll     zoom toward cursor
  Click-drag       pan

Coordinates are km east/north from the radar site (Reference-Nowcastle pattern).
Overlays (state borders, counties, cities) are projected from lat/lon into the
same km-from-radar frame. The game-clock cap (``scan_time ≤ virtual_time``) is
enforced by the grid — peers can never scrub past current game time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pyart
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from collections import OrderedDict

from ..data.reports import Report
from ..data.sweep_index import SweepIndex, SweepRef
from ..data.sites import Site, site_by_icao
from ..geo.projection import latlon_to_xy_km
from .time_format import format_player_time

# Live storm reports drawn on the radar panel: shape/size/fade match the host
# central map (plan §6). Fade tied to display time so scrubbing back recomputes
# the report state.
_REPORT_MARKERS = {"tornado": "^", "hail": "o", "wind": "s"}
_REPORT_COLORS = {"tornado": "#ff4444", "hail": "#44ff66", "wind": "#66bbff"}
REPORT_FADE_SEC_RADAR = 30 * 60


def _report_size(category: str, magnitude: float) -> float:
    if category == "tornado":
        return 50.0 + max(0.0, float(magnitude)) * 24.0
    if category == "hail":
        return 24.0 + max(0.0, float(magnitude)) * 20.0
    if category == "wind":
        return 20.0 + max(0.0, float(magnitude) - 50.0) * 1.2
    return 24.0

log = logging.getLogger(__name__)


class DealiasMode(str, Enum):
    """How to handle velocity-aliasing (folding) on raw NEXRAD velocity data.

    REGION_BASED is the default — PyART's region-growing algorithm gives the most
    forecaster-friendly velocity images and matches what most external products
    (RadarScope, GR2Analyst) do under the hood.
    """

    NONE = "none"
    REGION_BASED = "region_based"
    PHASE_UNWRAP = "phase_unwrap"


# Product → (PyART field name, colormap, vmin/vmax) in m/s for velocity.
# Velocity field name is replaced at render time if dealiasing is active.
PRODUCTS: dict[str, tuple[str, str, float, float]] = {
    "REF": ("reflectivity",                "ChaseSpectral", -10.0, 75.0),
    "VEL": ("velocity",                    "Carbone42",     -40.0, 40.0),
    "SW":  ("spectrum_width",              "magma",           0.0, 15.0),
    "CC":  ("cross_correlation_ratio",     "NWSRef",          0.0,  1.0),
    "ZDR": ("differential_reflectivity",   "ChaseSpectral",   0.0,  7.5),
    "KDP": ("specific_differential_phase", "ChaseSpectral",   0.0,  7.5),
    "PHI": ("differential_phase",          "Wild25",          0.0, 360.0),
}

CORRECTED_VELOCITY_FIELD = "corrected_velocity"

LAYOUT_DEFAULTS = {
    1: ("REF",),
    2: ("REF", "VEL"),
    4: ("REF", "VEL", "CC", "ZDR"),
}

# Game-clock-cap default (no future peeking) — overridable per grid instance
DEFAULT_MAX_RANGE_KM = 250.0

# Radar-volume LRU cache: how many PyART Radar objects to keep in memory.
# Smooths rapid scrubbing — without this each volume re-open is ~200 ms.
RADAR_LRU_DEFAULT = 24
RADAR_LRU_MIN = 6
RADAR_LRU_MAX = 100


# ----------------------------- single panel -----------------------------------

class RadarPanel(QFrame):
    """One product axes embedded in a matplotlib canvas."""

    def __init__(self, product: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product = product
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._figure = Figure(figsize=(5, 5), facecolor="#0a0a0a")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._canvas)

        self.ax = self._figure.add_subplot(111, facecolor="#0a0a0a")
        self.ax.set_aspect("equal")
        self.ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#444")
        self._mesh = None
        self._overlay_artists: list = []
        self._report_artists: list = []
        self._title = self.ax.set_title("", color="#dddddd", fontsize=9, loc="left")
        # Pan/zoom state — set by the grid via _attach_nav so all panels stay in sync.
        self._pan_start: tuple[float, float] | None = None
        self._pan_start_xlim: tuple[float, float] | None = None
        self._pan_start_ylim: tuple[float, float] | None = None
        self._on_limits_changed = None   # (xlim, ylim) → None — broadcast to siblings
        self._cid_scroll: int | None = None
        self._cid_press: int | None = None
        self._cid_release: int | None = None
        self._cid_motion: int | None = None
        self._cid_dbl: int | None = None
        self._home_xlim: tuple[float, float] = (-250.0, 250.0)
        self._home_ylim: tuple[float, float] = (-250.0, 250.0)

    def set_product(self, product: str) -> None:
        if product not in PRODUCTS:
            raise ValueError(f"Unknown product: {product}")
        self.product = product
        self._clear_mesh()

    def render_sweep(
        self,
        radar,
        sweep_no: int,
        site: Site,
        *,
        display_time: datetime,
        overlays: "OverlayBundle | None" = None,
        velocity_field: str | None = None,
    ) -> None:
        field, cmap, vmin, vmax = PRODUCTS[self.product]
        # Velocity field name is grid-controlled (raw vs dealiased)
        if self.product == "VEL" and velocity_field is not None:
            field = velocity_field
            if field not in radar.fields and "velocity" in radar.fields:
                field = "velocity"  # graceful fallback
        elev = float(radar.fixed_angle["data"][sweep_no])
        base_title = f"{site.icao}  {self.product}  {elev:.1f}°   {format_player_time(display_time)}"
        self._clear_mesh()
        has_data = False
        if field in radar.fields:
            data = radar.fields[field]["data"]
            start = int(radar.sweep_start_ray_index["data"][sweep_no])
            end = int(radar.sweep_end_ray_index["data"][sweep_no]) + 1
            sweep_data = data[start:end]
            all_masked = hasattr(sweep_data, "mask") and sweep_data.mask.all()
            if not all_masked:
                az = radar.azimuth["data"][start:end]
                rng_m = radar.range["data"]
                az_rad = np.deg2rad(az)
                # East/north in km (matches Nowcastle: x=east, y=north)
                x = (rng_m[None, :] * np.sin(az_rad)[:, None]) / 1000.0
                y = (rng_m[None, :] * np.cos(az_rad)[:, None]) / 1000.0
                self._mesh = self.ax.pcolormesh(
                    x, y, sweep_data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
                )
                has_data = True

        if has_data:
            self._title.set_text(base_title)
        else:
            # Doppler ↔ dual-pol split is normal in NEXRAD VCPs; not a bug.
            self._title.set_text(f"{base_title}   (no {self.product} at this sweep)")
            # Force home extent so range rings have something to draw against
            self.ax.set_xlim(self._home_xlim)
            self.ax.set_ylim(self._home_ylim)
        if overlays is not None:
            self._draw_overlays(site, overlays)
        self._canvas.draw_idle()

    def _clear_mesh(self) -> None:
        if self._mesh is not None:
            try:
                self._mesh.remove()
            except (ValueError, AttributeError):
                pass
            self._mesh = None
        for a in self._overlay_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._overlay_artists.clear()
        for a in self._report_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._report_artists.clear()

    def draw_reports(self, reports: list[Report], site: Site, display_time: datetime) -> None:
        """Render storm reports onto the panel with fade tied to ``display_time``.

        Reports with ``time > display_time`` are hidden. Older reports fade
        toward fully transparent over :data:`REPORT_FADE_SEC_RADAR`. Coordinates
        are projected to km from the radar site.
        """
        for category, marker in _REPORT_MARKERS.items():
            color = _REPORT_COLORS[category]
            for r in reports:
                if r.category != category or r.time > display_time:
                    continue
                age = (display_time - r.time).total_seconds()
                if age > REPORT_FADE_SEC_RADAR * 1.5:
                    continue
                alpha = max(0.15, 1.0 - age / REPORT_FADE_SEC_RADAR)
                x_km, y_km = latlon_to_xy_km(r.lat, r.lon, site.lat, site.lon)
                a = self.ax.scatter([x_km], [y_km], s=_report_size(category, r.magnitude),
                                     c=color, marker=marker, alpha=alpha,
                                     edgecolors="#000", linewidths=0.4, zorder=7)
                self._report_artists.append(a)

    # ---- interactive navigation (scroll zoom + click-drag pan) ----------

    def attach_nav(self, on_limits_changed) -> None:
        """Wire up matplotlib event handlers; ``on_limits_changed(xlim, ylim)`` is
        called by the grid to broadcast new limits to sibling panels.
        """
        self._on_limits_changed = on_limits_changed
        self._cid_scroll = self._canvas.mpl_connect("scroll_event", self._on_scroll)
        self._cid_press = self._canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_release = self._canvas.mpl_connect("button_release_event", self._on_release)
        self._cid_motion = self._canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_dbl = self._canvas.mpl_connect("button_press_event", self._on_dblclick)

    def set_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        """External setter — used by the grid to broadcast pan/zoom across panels."""
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self._canvas.draw_idle()

    def reset_home(self) -> None:
        self.set_limits(self._home_xlim, self._home_ylim)
        if self._on_limits_changed:
            self._on_limits_changed(self._home_xlim, self._home_ylim)

    def _on_scroll(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25  # up = zoom in
        x, y = event.xdata, event.ydata
        xmin, xmax = self.ax.get_xlim()
        ymin, ymax = self.ax.get_ylim()
        new_xlim = (x + (xmin - x) * scale, x + (xmax - x) * scale)
        new_ylim = (y + (ymin - y) * scale, y + (ymax - y) * scale)
        self.set_limits(new_xlim, new_ylim)
        if self._on_limits_changed:
            self._on_limits_changed(new_xlim, new_ylim)

    def _on_press(self, event) -> None:
        if event.button != 1 or event.inaxes is not self.ax:
            return
        if event.dblclick:
            return  # handled by _on_dblclick
        self._pan_start = (event.x, event.y)
        self._pan_start_xlim = self.ax.get_xlim()
        self._pan_start_ylim = self.ax.get_ylim()

    def _on_release(self, event) -> None:
        if event.button != 1:
            return
        self._pan_start = None
        self._pan_start_xlim = None
        self._pan_start_ylim = None

    def _on_motion(self, event) -> None:
        if self._pan_start is None or self._pan_start_xlim is None or self._pan_start_ylim is None:
            return
        if event.x is None or event.y is None:
            return
        dx_px = event.x - self._pan_start[0]
        dy_px = event.y - self._pan_start[1]
        bbox = self.ax.bbox
        x0, x1 = self._pan_start_xlim
        y0, y1 = self._pan_start_ylim
        dx_data = -(dx_px / max(bbox.width, 1)) * (x1 - x0)
        dy_data = -(dy_px / max(bbox.height, 1)) * (y1 - y0)
        new_xlim = (x0 + dx_data, x1 + dx_data)
        new_ylim = (y0 + dy_data, y1 + dy_data)
        self.set_limits(new_xlim, new_ylim)
        if self._on_limits_changed:
            self._on_limits_changed(new_xlim, new_ylim)

    def _on_dblclick(self, event) -> None:
        if not event.dblclick or event.button != 1 or event.inaxes is not self.ax:
            return
        self.reset_home()

    def _draw_overlays(self, site: Site, overlays: "OverlayBundle") -> None:
        # Range rings
        for r_km in overlays.range_rings_km:
            theta = np.linspace(0, 2 * np.pi, 180)
            artist, = self.ax.plot(
                r_km * np.sin(theta), r_km * np.cos(theta),
                color="#3a3a3a", linewidth=0.6, zorder=2,
            )
            self._overlay_artists.append(artist)
        # State borders (in km from radar)
        for ring in overlays.state_borders_xy:
            artist, = self.ax.plot(ring[:, 0], ring[:, 1],
                                   color="#7a7a7a", linewidth=0.8, zorder=3)
            self._overlay_artists.append(artist)
        # County borders (if loaded; thin grey, hidden when zoomed out)
        cur_extent = self.ax.get_xlim()
        view_width_km = cur_extent[1] - cur_extent[0]
        if view_width_km < 400.0:  # show counties at moderate zoom
            for ring in overlays.county_borders_xy:
                artist, = self.ax.plot(ring[:, 0], ring[:, 1],
                                       color="#525252", linewidth=0.4, zorder=3)
                self._overlay_artists.append(artist)
        # Cities (filter by population threshold scaled to zoom)
        pop_threshold = 100_000 if view_width_km > 250.0 else 20_000
        city_dots = []
        city_texts = []
        for city in overlays.cities:
            if city.pop < pop_threshold:
                continue
            cx_km, cy_km = latlon_to_xy_km(city.lat, city.lon, site.lat, site.lon)
            dot = self.ax.scatter([cx_km], [cy_km], s=6, c="#cccccc", zorder=4)
            text = self.ax.text(cx_km + 2, cy_km + 2, city.name,
                                 color="#dddddd", fontsize=7, zorder=5)
            city_dots.append(dot)
            city_texts.append(text)
            self._overlay_artists.extend([dot, text])
        # Reduce label overlap. adjustText nudges labels apart in screen-space
        # while preferring small displacements; bounded iterations to keep
        # render time predictable.
        if city_texts:
            try:
                from adjustText import adjust_text
                adjust_text(
                    city_texts, ax=self.ax,
                    expand=(1.2, 1.4), force_text=(0.3, 0.4),
                    arrowprops=dict(arrowstyle="-", color="#666", lw=0.4),
                )
            except Exception:  # noqa: BLE001
                # adjustText is best-effort; tolerate failure (e.g., on zoom levels
                # with weird coordinate scales) without breaking the panel.
                pass


# ----------------------------- overlays ---------------------------------------

@dataclass
class CityPoint:
    name: str
    lat: float
    lon: float
    pop: int


@dataclass
class OverlayBundle:
    """Pre-projected overlay layers ready to draw on a radar-centric panel."""

    range_rings_km: list[float]
    state_borders_xy: list[np.ndarray]    # each is an Nx2 array of (x_km, y_km)
    county_borders_xy: list[np.ndarray]
    cities: list[CityPoint]

    @classmethod
    def empty(cls) -> "OverlayBundle":
        return cls(range_rings_km=[50, 100, 150, 200], state_borders_xy=[],
                   county_borders_xy=[], cities=[])


# ----------------------------- grid widget ------------------------------------

class RadarPanelGrid(QWidget):
    """1, 2, or 4 panels showing the same volume, different products."""

    sweep_changed = pyqtSignal(object)  # emits the new SweepRef (or None)

    def __init__(
        self,
        sweep_index: SweepIndex,
        site_icao: str,
        *,
        n_panels: int = 4,
        layout: tuple[str, ...] | None = None,
        max_virtual_time: datetime | None = None,
        dealias_mode: DealiasMode = DealiasMode.REGION_BASED,
        radar_lru_size: int = RADAR_LRU_DEFAULT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sweep_index = sweep_index
        site = site_by_icao(site_icao)
        if site is None:
            raise ValueError(f"Unknown radar site: {site_icao}")
        self.site = site
        self._max_virtual_time = max_virtual_time
        self._current_sweep: SweepRef | None = None
        self._current_radar = None
        self._loaded_file: Path | None = None
        # LRU cache of (file → loaded+dealiased PyART Radar) so scrubbing
        # rapidly between adjacent volumes doesn't re-parse them.
        self._radar_lru: OrderedDict[Path, object] = OrderedDict()
        self.radar_lru_size = max(RADAR_LRU_MIN, min(RADAR_LRU_MAX, int(radar_lru_size)))
        self._dealias_mode = dealias_mode
        self._dealiased_for_mode: DealiasMode | None = None
        # Live storm reports drawn on each render; tied to the panel's display
        # time so scrubbing back recomputes which reports are visible.
        self.live_reports: list[Report] = []
        # Lazy-load overlays here (cartopy/Natural Earth) so empty constructions
        # used by tests don't pay the I/O cost. Falls back to empty on failure.
        try:
            from .overlay_loader import build_overlays
            self.overlays = build_overlays(self.site)
        except Exception as e:  # noqa: BLE001
            log.warning("Overlay load failed (%s) — falling back to empty bundle", e)
            self.overlays = OverlayBundle.empty()

        self._panels: list[RadarPanel] = []
        self._build_panels(n_panels, layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_scrubber()

    def set_live_reports(self, reports: list[Report]) -> None:
        """Provide the report set whose visible subset is computed at render time."""
        self.live_reports = reports
        if self._current_sweep is not None:
            self._render_all()

    # ---- scrubber slider ------------------------------------------------

    def _build_scrubber(self) -> None:
        """Add a horizontal slider beneath the panels that scrubs through the
        full available sweep range at the current elevation.

        Slider values index the sorted-by-time list of same-elevation sweeps.
        Capped at the largest sweep ≤ max_virtual_time (game-clock cap).
        """
        self._scrubber = QSlider(Qt.Orientation.Horizontal, self)
        self._scrubber.setRange(0, 0)
        self._scrubber.setEnabled(False)
        self._scrubber.valueChanged.connect(self._on_scrubber)
        outer = self.layout()
        if outer is not None:
            outer.addWidget(self._scrubber)

    def _refresh_scrubber(self) -> None:
        """Update the slider's range + position to reflect the current elevation."""
        if not hasattr(self, "_scrubber") or self._current_sweep is None:
            return
        sweeps = sorted(self.sweep_index.at_elevation(self._current_sweep.elev_deg),
                        key=lambda s: s.start_time)
        if self._max_virtual_time:
            sweeps = [s for s in sweeps if s.start_time <= self._max_virtual_time]
        self._scrubber.blockSignals(True)
        if sweeps:
            self._scrubber.setRange(0, len(sweeps) - 1)
            try:
                idx = sweeps.index(self._current_sweep)
            except ValueError:
                idx = 0
            self._scrubber.setValue(idx)
            self._scrubber.setEnabled(True)
        else:
            self._scrubber.setRange(0, 0)
            self._scrubber.setEnabled(False)
        self._scrubber.blockSignals(False)

    def _on_scrubber(self, value: int) -> None:
        if self._current_sweep is None:
            return
        sweeps = sorted(self.sweep_index.at_elevation(self._current_sweep.elev_deg),
                        key=lambda s: s.start_time)
        if self._max_virtual_time:
            sweeps = [s for s in sweeps if s.start_time <= self._max_virtual_time]
        if 0 <= value < len(sweeps):
            self.show_sweep(sweeps[value])

    # ---- velocity dealiasing -------------------------------------------

    @property
    def dealias_mode(self) -> DealiasMode:
        return self._dealias_mode

    def set_dealias_mode(self, mode: DealiasMode) -> None:
        """Change the velocity dealiasing strategy and re-process the current volume."""
        if mode == self._dealias_mode:
            return
        self._dealias_mode = mode
        # Force re-dealiasing on the currently-loaded volume
        self._dealiased_for_mode = None
        if self._current_radar is not None:
            self._apply_dealias(self._current_radar)
            self._render_all()

    def velocity_field_name(self) -> str:
        """Field name the panels should request for the VEL product."""
        if self._dealias_mode == DealiasMode.NONE:
            return "velocity"
        return CORRECTED_VELOCITY_FIELD

    def _apply_dealias(self, radar) -> None:
        """Compute corrected_velocity on ``radar`` per current dealias mode, in-place.

        No-op if the radar already has the field populated for the current mode.
        Falls back to raw ``velocity`` (and logs a warning) on any failure so the
        UI never crashes due to dealiasing edge cases.
        """
        if self._dealias_mode == DealiasMode.NONE:
            return
        if self._dealiased_for_mode == self._dealias_mode and CORRECTED_VELOCITY_FIELD in radar.fields:
            return
        if "velocity" not in radar.fields:
            return
        try:
            if self._dealias_mode == DealiasMode.REGION_BASED:
                corrected = pyart.correct.dealias_region_based(radar)
            elif self._dealias_mode == DealiasMode.PHASE_UNWRAP:
                corrected = pyart.correct.dealias_unwrap_phase(radar)
            else:
                return
            radar.add_field(CORRECTED_VELOCITY_FIELD, corrected, replace_existing=True)
            self._dealiased_for_mode = self._dealias_mode
        except Exception as e:  # noqa: BLE001
            log.warning("Dealiasing (%s) failed: %s — falling back to raw velocity",
                        self._dealias_mode.value, e)
            # Mirror raw velocity into corrected_velocity so render still works
            raw = radar.fields["velocity"]
            radar.add_field(CORRECTED_VELOCITY_FIELD, dict(raw), replace_existing=True)
            self._dealiased_for_mode = self._dealias_mode

    # ---- layout management ----------------------------------------------

    def set_layout(self, n_panels: int, layout: tuple[str, ...] | None = None) -> None:
        # Remove existing panels from the inner grid layout (preserve outer VBox + scrubber)
        if hasattr(self, "_grid_layout"):
            while self._grid_layout.count():
                item = self._grid_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
        self._panels.clear()
        self._build_panels(n_panels, layout)
        if self._current_sweep is not None and self._current_radar is not None:
            self._render_all()

    def _build_panels(self, n_panels: int, layout: tuple[str, ...] | None) -> None:
        if n_panels not in (1, 2, 4):
            raise ValueError(f"n_panels must be 1, 2, or 4 (got {n_panels})")
        products = layout or LAYOUT_DEFAULTS[n_panels]
        if len(products) != n_panels:
            raise ValueError(f"layout has {len(products)} entries but n_panels={n_panels}")
        # Top-level VBox so the scrubber can be appended underneath the grid.
        if self.layout() is None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
        else:
            outer = self.layout()
        grid = QGridLayout()
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        for i, product in enumerate(products):
            panel = RadarPanel(product, self)
            if n_panels == 1:
                grid.addWidget(panel, 0, 0)
            elif n_panels == 2:
                grid.addWidget(panel, 0, i)
            else:  # 4
                grid.addWidget(panel, i // 2, i % 2)
            panel.attach_nav(self._broadcast_limits)
            self._panels.append(panel)
        outer.addLayout(grid, stretch=1)
        self._grid_layout = grid

    def _broadcast_limits(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        """One panel changed its view via scroll/pan → mirror to all others.

        We compare before applying to avoid an infinite signal loop (each panel
        already has the new limits when this callback fires).
        """
        for panel in self._panels:
            if panel.ax.get_xlim() != xlim or panel.ax.get_ylim() != ylim:
                panel.set_limits(xlim, ylim)

    # ---- time / elevation ----------------------------------------------

    def set_max_virtual_time(self, t: datetime | None) -> None:
        """Game-clock cap. Forward-scrubs past this are blocked."""
        self._max_virtual_time = t
        if self._current_sweep and t and self._current_sweep.start_time > t:
            self.show_latest_at(t, self._current_sweep.elev_deg)
        else:
            self._refresh_scrubber()   # cap moved forward: more sweeps reachable

    def show_sweep(self, sweep: SweepRef) -> None:
        if self._max_virtual_time and sweep.start_time > self._max_virtual_time:
            return
        if sweep.file != self._loaded_file:
            self._current_radar = self._get_radar_from_cache(sweep.file)
            self._loaded_file = sweep.file
        self._current_sweep = sweep
        self._render_all()
        self._refresh_scrubber()
        self.sweep_changed.emit(sweep)

    def _get_radar_from_cache(self, file: Path):
        """Return a loaded+dealiased PyART Radar, hitting the LRU when possible."""
        cached = self._radar_lru.get(file)
        if cached is not None:
            # Mark as most-recently used
            self._radar_lru.move_to_end(file)
            return cached
        radar = pyart.io.read_nexrad_archive(str(file))
        self._dealiased_for_mode = None
        self._apply_dealias(radar)
        self._radar_lru[file] = radar
        while len(self._radar_lru) > self.radar_lru_size:
            self._radar_lru.popitem(last=False)
        return radar

    def show_latest_at(self, display_time: datetime, elev_deg: float) -> None:
        """Pick the most recent sweep at ``elev_deg`` with start_time ≤ display_time."""
        s = self.sweep_index.latest_at_or_before(display_time, elev_deg=elev_deg)
        if s is not None:
            self.show_sweep(s)

    def step_time(self, n: int) -> None:
        """Step ``n`` sweeps forward (+) or backward (-) at the current elevation."""
        if self._current_sweep is None:
            return
        nxt = self.sweep_index.step_in_elevation(self._current_sweep, n)
        if nxt is None:
            return
        if self._max_virtual_time and nxt.start_time > self._max_virtual_time:
            return
        self.show_sweep(nxt)

    def step_elevation(self, n: int) -> None:
        """Move ``n`` tilts up (+) or down (-) at roughly the current time."""
        if self._current_sweep is None:
            return
        elevs = self.sweep_index.available_elevations(self._current_sweep.start_time)
        if not elevs:
            return
        cur_idx = min(range(len(elevs)), key=lambda i: abs(elevs[i] - self._current_sweep.elev_deg))
        new_idx = max(0, min(len(elevs) - 1, cur_idx + n))
        new_elev = elevs[new_idx]
        s = self.sweep_index.latest_at_or_before(self._current_sweep.start_time, elev_deg=new_elev)
        if s is not None:
            self.show_sweep(s)

    # ---- product control -----------------------------------------------

    def set_product(self, panel_index: int, product: str) -> None:
        if not (0 <= panel_index < len(self._panels)):
            return
        self._panels[panel_index].set_product(product)
        if self._current_sweep is not None and self._current_radar is not None:
            self._render_one(self._panels[panel_index])

    # ---- rendering -----------------------------------------------------

    def _render_all(self) -> None:
        for panel in self._panels:
            self._render_one(panel)

    def _render_one(self, panel: RadarPanel) -> None:
        if self._current_radar is None or self._current_sweep is None:
            return
        panel.render_sweep(
            self._current_radar,
            self._current_sweep.sweep_number,
            self.site,
            display_time=self._current_sweep.start_time,
            overlays=self.overlays,
            velocity_field=self.velocity_field_name(),
        )
        # Storm reports rendered with fade keyed to the sweep's display time,
        # so scrubbing back recomputes which reports are visible.
        if self.live_reports:
            panel.draw_reports(self.live_reports, self.site, self._current_sweep.start_time)

    # ---- keyboard ------------------------------------------------------

    def keyPressEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        key = event.key()
        mods = event.modifiers()
        step = 5 if mods & Qt.KeyboardModifier.ShiftModifier else 1
        if key == Qt.Key.Key_Left:
            self.step_time(-step)
        elif key == Qt.Key.Key_Right:
            self.step_time(step)
        elif key == Qt.Key.Key_Up:
            self.step_elevation(+1)
        elif key == Qt.Key.Key_Down:
            self.step_elevation(-1)
        elif Qt.Key.Key_1 <= key <= Qt.Key.Key_7:
            self._cycle_focused_product(key - Qt.Key.Key_1)
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def _cycle_focused_product(self, index: int) -> None:
        focused = self._focused_panel_index()
        product_keys = list(PRODUCTS.keys())
        if not (0 <= index < len(product_keys)):
            return
        new_product = product_keys[index]
        self.set_product(focused, new_product)

    def _focused_panel_index(self) -> int:
        for i, panel in enumerate(self._panels):
            if panel.hasFocus():
                return i
        return 0
