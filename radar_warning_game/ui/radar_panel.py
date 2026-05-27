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
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)
from collections import OrderedDict

from ..data.reports import Report
from ..data.sweep_index import SweepIndex, SweepRef
from ..data.sites import Site, site_by_icao
from ..geo.polygons import Polygon as GamePolygon
from ..geo.projection import latlon_to_xy_km
from .time_format import format_player_time

# Live storm reports drawn on the radar panel: shape/size/fade match the host
# central map (plan §6). Fade tied to display time so scrubbing back recomputes
# the report state.
_REPORT_MARKERS = {"tornado": "^", "hail": "o", "wind": "s"}
# Edge color encodes the report category (visible at any zoom even on small
# markers); fill stays bright for visibility against radar imagery.
_REPORT_EDGE_COLORS = {
    "tornado": "#ff3030",
    "hail":    "#22cc55",
    "wind":    "#3399ff",
}
_REPORT_COLORS = {"tornado": "#ff4444", "hail": "#44ff66", "wind": "#66bbff"}
REPORT_FADE_SEC_RADAR = 30 * 60


def _report_tooltip_text(r: "Report") -> str:
    """Compact label for the report-hover tooltip: time + magnitude + remarks.

    Magnitude is rendered with hazard-appropriate units (EF for tornado, in
    for hail, mph for wind). Casualty counts (when nonzero) and any free-text
    remarks are appended on extra lines so the player can see whether the
    report contains injuries/fatalities information that PIB/PDS scoring
    depends on.
    """
    if r.category == "tornado":
        mag = "EF?" if r.magnitude < 0 else f"EF{int(round(r.magnitude))}"
    elif r.category == "hail":
        mag = f"{r.magnitude:.2f}\" hail".replace("0\"", "\"")
    elif r.category == "wind":
        mag = f"{int(round(r.magnitude))} mph wind"
    else:
        mag = f"{r.magnitude:g}"
    lines = [f"{r.category.title()} — {mag}",
             format_player_time(r.time)]
    if r.injuries or r.fatalities:
        lines.append(f"{r.injuries} INJ, {r.fatalities} FAT")
    return "\n".join(lines)


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

# Display units for the data-probe / inspector readout. PHI is the raw
# differential phase, CC the cross-correlation ratio (unitless).
PRODUCT_UNITS: dict[str, str] = {
    "REF": "dBZ", "VEL": "m/s", "SW": "m/s",
    "CC":  "",    "ZDR": "dB",  "KDP": "°/km", "PHI": "°",
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

    # emitted when the user selects a product from this panel's dropdown
    product_changed = pyqtSignal(str)

    def __init__(self, product: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product = product
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        self._figure = Figure(figsize=(5, 5), facecolor="#0a0a0a")
        self._canvas = FigureCanvasQTAgg(self._figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Product picker — same options as the keybinds 1-7, labeled with the
        # keybind. Discoverable equivalent of "click panel + press digit".
        self._product_combo = QComboBox(self)
        product_keys = list(PRODUCTS.keys())
        for i, key in enumerate(product_keys):
            self._product_combo.addItem(f"{key}  ({i+1})", key)
        self._product_combo.setCurrentIndex(product_keys.index(product))
        self._product_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._product_combo.setToolTip(
            "Change this panel's product (keybind: click panel, then press 1-7)"
        )
        self._product_combo.currentIndexChanged.connect(self._on_combo_change)
        layout.addWidget(self._product_combo)
        layout.addWidget(self._canvas, stretch=1)

        self.ax = self._figure.add_subplot(111, facecolor="#0a0a0a")
        self.ax.set_aspect("equal")
        self.ax.tick_params(colors="#aaaaaa", labelsize=8)
        for spine in self.ax.spines.values():
            spine.set_color("#444")
        self._mesh = None
        self._overlay_artists: list = []
        self._report_artists: list = []
        # Sidecar map: scatter artist → underlying Report. Lets the hover
        # handler look up which report the cursor is over without rebuilding
        # picker tags. Keyed by id() since matplotlib artists are not hashable
        # in older versions.
        self._report_meta: dict[int, "Report"] = {}
        # Hover annotation — single Text artist that follows the cursor when
        # over a report and hides otherwise. Initial position is irrelevant
        # (set per hover); annotation is invisible until a report is hit.
        self._hover_annotation = self.ax.annotate(
            "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
            color="#0a0a0a", fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffd400",
                      edgecolor="#000", linewidth=0.6, alpha=0.95),
            zorder=30,
        )
        self._hover_annotation.set_visible(False)
        self._title = self.ax.set_title("", color="#dddddd", fontsize=9, loc="left")
        # Debounce redraws during continuous pan/zoom — multiple events within
        # the debounce window only repaint once. Without this, every scroll
        # notch triggers a full canvas redraw which compounds when 4 panels
        # broadcast a single zoom event to each other.
        self._redraw_timer = QTimer(self)
        self._redraw_timer.setSingleShot(True)
        self._redraw_timer.setInterval(40)   # ~25 FPS during continuous gestures
        self._redraw_timer.timeout.connect(self._canvas.draw_idle)
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
        # Data-probe / inspector state. ``_render_data`` is repopulated at
        # the end of each render_sweep so the probe can look up values
        # without re-deriving sweep geometry on every mousemove.
        self.inspector_enabled: bool = False
        self._render_data: dict | None = None

    def set_product(self, product: str) -> None:
        if product not in PRODUCTS:
            raise ValueError(f"Unknown product: {product}")
        self.product = product
        self._clear_mesh()
        # Keep the dropdown in sync when set programmatically (keybind path)
        if hasattr(self, "_product_combo"):
            idx = list(PRODUCTS.keys()).index(product)
            self._product_combo.blockSignals(True)
            self._product_combo.setCurrentIndex(idx)
            self._product_combo.blockSignals(False)

    def _on_combo_change(self, idx: int) -> None:
        product_keys = list(PRODUCTS.keys())
        if not (0 <= idx < len(product_keys)):
            return
        new = product_keys[idx]
        if new == self.product:
            return
        # Emit so the grid can re-render this panel with the right field
        self.product_changed.emit(new)

    def render_sweep(
        self,
        radar,
        sweep_no: int,
        site: Site,
        *,
        display_time: datetime,
        overlays: "OverlayBundle | None" = None,
        velocity_field: str | None = None,
        cross_volume_resolver=None,
    ) -> None:
        field, cmap, vmin, vmax = PRODUCTS[self.product]
        # Velocity field name is grid-controlled (raw vs dealiased)
        if self.product == "VEL" and velocity_field is not None:
            field = velocity_field
            if field not in radar.fields and "velocity" in radar.fields:
                field = "velocity"  # graceful fallback
        elev = float(radar.fixed_angle["data"][sweep_no])
        base_title = f"{site.icao}  {self.product}  {elev:.1f}°   {format_player_time(display_time)}"

        # Preserve the user's pan/zoom across data changes. pcolormesh
        # auto-scales the axes to fit the new data, which clobbers any zoom
        # the user has applied. Save the current limits before clearing and
        # restore them after the new mesh is in place.
        prev_xlim = self.ax.get_xlim()
        prev_ylim = self.ax.get_ylim()
        is_first_render = prev_xlim == (0.0, 1.0) and prev_ylim == (0.0, 1.0)

        self._clear_mesh()
        # Find the best sweep within this volume that actually has data for
        # this product. NEXRAD splits each low tilt into a surveillance scan
        # (dual-pol fields) and a Doppler scan (VEL/SW); when the current
        # sweep is the wrong split, fall back to a sibling at the same tilt
        # that has the requested field. The user sees the closest-available
        # CC/ZDR/etc. instead of a blank panel.
        effective_sweep = self._find_best_sweep(radar, sweep_no, field, elev)
        used_in_volume_fallback = (
            effective_sweep is not None and effective_sweep != sweep_no
        )
        # If nothing in this volume works for the requested field, fall
        # forward/backward to an adjacent volume that does. This covers the
        # case where e.g. the first sweep of a new volume is a surveillance
        # scan with no VEL: we render the previous volume's most-recent
        # Doppler sweep at the same tilt so the panel never goes blank.
        cross_volume_radar = radar
        cross_volume_time = display_time
        used_cross_volume = False
        if effective_sweep is None and cross_volume_resolver is not None:
            resolved = cross_volume_resolver(field, elev, display_time)
            if resolved is not None:
                cross_volume_radar, effective_sweep, cross_volume_time = resolved
                used_cross_volume = True

        has_data = False
        # Reset inspector state in case render fails or nothing is drawn —
        # otherwise the probe would lie about stale data.
        self._render_data = None
        if effective_sweep is not None:
            src = cross_volume_radar if used_cross_volume else radar
            # Re-resolve the field name on the cross-volume source — different
            # volumes may have raw vs dealiased velocity available.
            actual_field = field
            if used_cross_volume and field not in src.fields and "velocity" in src.fields:
                actual_field = "velocity"
            data = src.fields[actual_field]["data"]
            start = int(src.sweep_start_ray_index["data"][effective_sweep])
            end = int(src.sweep_end_ray_index["data"][effective_sweep]) + 1
            sweep_data = data[start:end]
            all_masked = hasattr(sweep_data, "mask") and sweep_data.mask.all()
            if not all_masked:
                az = src.azimuth["data"][start:end]
                rng_m = src.range["data"]
                az_rad = np.deg2rad(az)
                # East/north in km (matches Nowcastle: x=east, y=north)
                x = (rng_m[None, :] * np.sin(az_rad)[:, None]) / 1000.0
                y = (rng_m[None, :] * np.cos(az_rad)[:, None]) / 1000.0
                self._mesh = self.ax.pcolormesh(
                    x, y, sweep_data, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto",
                    rasterized=True,
                )
                has_data = True
                # Stash everything the data-probe needs to look up a value
                # at (x_km, y_km) without re-deriving sweep ranges.
                self._render_data = {
                    "azimuths_deg": np.asarray(az, dtype=float),
                    "ranges_m": np.asarray(rng_m, dtype=float),
                    "sweep_data": sweep_data,
                    "unit": PRODUCT_UNITS.get(self.product, ""),
                    "label": self.product,
                }

        if has_data and not (used_in_volume_fallback or used_cross_volume):
            self._title.set_text(base_title)
        elif has_data and used_cross_volume:
            self._title.set_text(
                f"{base_title}   (showing {self.product} from "
                f"{format_player_time(cross_volume_time)})"
            )
        elif has_data and used_in_volume_fallback:
            eff_elev = float(radar.fixed_angle["data"][effective_sweep])
            self._title.set_text(
                f"{base_title}   (showing {self.product} from "
                f"sibling sweep at {eff_elev:.1f}°)"
            )
        else:
            # Nothing — neither this volume nor any nearby one has the field.
            # Rare; typically means the radar doesn't support this product.
            self._title.set_text(f"{base_title}   (no {self.product} available)")

        # Restore the user's view (or apply the home extent on first render)
        if is_first_render:
            self.ax.set_xlim(self._home_xlim)
            self.ax.set_ylim(self._home_ylim)
        else:
            self.ax.set_xlim(prev_xlim)
            self.ax.set_ylim(prev_ylim)
        if overlays is not None:
            self._draw_overlays(site, overlays)
        self._canvas.draw_idle()

    def _find_best_sweep(
        self,
        radar,
        sweep_no: int,
        field: str,
        target_elev: float,
    ) -> int | None:
        """Pick a sweep in ``radar`` at ~``target_elev`` that has unmasked
        data for ``field``. Prefers the requested ``sweep_no`` exactly;
        otherwise returns the closest-elevation sibling within the volume.
        Returns ``None`` if no sweep in this volume has the field at all.
        """
        if field not in radar.fields:
            return None
        data = radar.fields[field]["data"]
        starts = radar.sweep_start_ray_index["data"]
        ends = radar.sweep_end_ray_index["data"]

        def _has_unmasked(sw: int) -> bool:
            s = int(starts[sw])
            e = int(ends[sw]) + 1
            sl = data[s:e]
            if hasattr(sl, "mask"):
                try:
                    return not bool(sl.mask.all())
                except (AttributeError, TypeError):
                    return True
            return sl.size > 0

        if 0 <= sweep_no < radar.nsweeps and _has_unmasked(sweep_no):
            return sweep_no

        from ..data.sweep_index import ELEV_TOLERANCE_DEG
        fixed = radar.fixed_angle["data"]
        # Closest-elevation first; on a tie prefer the nearest sweep number
        # (Doppler scans typically immediately follow their paired surveillance
        # scan in the volume, so an adjacent index is the natural pairing).
        candidates = sorted(
            (i for i in range(int(radar.nsweeps))
             if i != sweep_no
             and abs(float(fixed[i]) - target_elev) < ELEV_TOLERANCE_DEG),
            key=lambda i: (abs(float(fixed[i]) - target_elev), abs(i - sweep_no)),
        )
        for i in candidates:
            if _has_unmasked(i):
                return i
        return None

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
        self._report_meta.clear()
        if hasattr(self, "_hover_annotation"):
            self._hover_annotation.set_visible(False)

    def draw_game_polygon(self, polygon: GamePolygon, site: Site) -> None:
        """Project the game boundary into km-from-radar and draw it as an outline.

        Plan §4a: players need to see the verification boundary on every radar
        panel — warnings outside this polygon don't verify. Drawn in yellow
        and slightly transparent so it's visible against any radar product
        without obscuring storm structure.
        """
        verts = list(polygon.vertices) + [polygon.vertices[0]]
        xs, ys = [], []
        for lat, lon in verts:
            x_km, y_km = latlon_to_xy_km(lat, lon, site.lat, site.lon)
            xs.append(x_km)
            ys.append(y_km)
        line, = self.ax.plot(xs, ys, color="#ffd400", linewidth=1.8,
                              alpha=0.85, zorder=6)
        self._overlay_artists.append(line)

    def draw_player_overlays(
        self,
        warnings: list,
        mcds: list,
        site: Site,
        display_time: datetime,
        game_clock_time: datetime | None = None,
    ) -> None:
        """Draw the player's own active warnings + MCDs onto this panel.

        Visibility is gated on ``game_clock_time`` (the current game-clock
        time) rather than the panel's ``display_time``. Reason: a warning
        the player just issued may be valid right now even though the
        last available radar sweep is timestamped a minute or two before
        the issuance — the warning should still appear on that sweep.

        The polygon **shape** still respects scrubbing — we draw the
        revision active at ``display_time`` so the player can review how
        their polygon evolved over time. If ``display_time`` predates the
        warning's first revision, we fall back to the original revision so
        the polygon doesn't disappear from a slightly-stale radar frame.

        Visual encoding mirrors the host central map (plan §4c):
          - **Tornado family**: solid line, red.
          - **Severe family**: dashed line, orange.
          - **MCD**: dotted line, purple.
        Higher-tier warnings get a thicker stroke so destructive / PDS /
        emergency tags stand out without changing color.
        """
        # Color by category — matches the user's spec:
        #   SVR family    → yellow (dashed)
        #   TOR / TORR    → red    (solid)
        #   PDS / TORE    → pink   (solid)
        #   MCD           → blue   (dotted)
        SVR_COLOR = "#ffd400"
        TOR_COLOR = "#ff3030"
        PDS_COLOR = "#ff66cc"
        MCD_COLOR = "#3399ff"
        # Tier → line-weight bump. Higher tier = thicker stroke so the
        # destructive / PDS / emergency flavor reads at any zoom.
        TIER_LW = {
            "SVR": 1.5, "SVRC": 2.0, "SVRD": 2.6,
            "TOR": 1.5, "TORR": 1.8, "PDS_TOR": 2.4, "TORE": 3.0,
        }
        # "Now" for the active-warning gate. If the caller didn't pass a
        # game-clock reference, fall back to display_time (legacy behavior).
        ref_time = game_clock_time if game_clock_time is not None else display_time
        for w in warnings:
            issue_t = w.original_issue_time
            cur_rev = w.current_revision
            expiry = cur_rev.revision_time + cur_rev.duration
            if w.canceled_at is not None and ref_time > w.canceled_at:
                continue
            if ref_time < issue_t or ref_time > expiry:
                continue
            # Pick revision for the polygon shape — scrubbing back within
            # the warning's lifetime shows the historical polygon; before
            # issue, fall back to the original so the polygon stays visible
            # on radar frames that predate issuance.
            rev = w.revision_at(display_time) or w.revisions[0]
            wt = rev.warning_type
            wt_name = wt.value if hasattr(wt, "value") else str(wt)
            if wt_name in ("PDS_TOR", "TORE"):
                color, style = PDS_COLOR, "-"
            elif wt_name in ("TOR", "TORR"):
                color, style = TOR_COLOR, "-"
            else:  # SVR family
                color, style = SVR_COLOR, "--"
            lw = TIER_LW.get(wt_name, 1.6)
            verts = list(rev.polygon.vertices) + [rev.polygon.vertices[0]]
            xs, ys = [], []
            for lat, lon in verts:
                x_km, y_km = latlon_to_xy_km(lat, lon, site.lat, site.lon)
                xs.append(x_km); ys.append(y_km)
            line, = self.ax.plot(xs, ys, color=color, linewidth=lw,
                                  linestyle=style, alpha=0.95, zorder=8)
            self._overlay_artists.append(line)
        for m in mcds:
            if ref_time < m.issue_time:
                continue
            if m.canceled_at is not None and ref_time > m.canceled_at:
                continue
            if ref_time > m.end_time():
                continue
            verts = list(m.polygon.vertices) + [m.polygon.vertices[0]]
            xs, ys = [], []
            for lat, lon in verts:
                x_km, y_km = latlon_to_xy_km(lat, lon, site.lat, site.lon)
                xs.append(x_km); ys.append(y_km)
            line, = self.ax.plot(xs, ys, color=MCD_COLOR, linewidth=1.4,
                                  linestyle=":", alpha=0.9, zorder=8)
            self._overlay_artists.append(line)

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
                                     edgecolors=_REPORT_EDGE_COLORS[category],
                                     linewidths=1.0, zorder=7)
                self._report_artists.append(a)
                self._report_meta[id(a)] = r

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
        """External setter — used by the grid to broadcast pan/zoom across panels.

        Schedules a debounced redraw rather than calling draw_idle directly so
        rapid scroll/drag events coalesce into a single repaint per panel.
        """
        self.ax.set_xlim(*xlim)
        self.ax.set_ylim(*ylim)
        self._redraw_timer.start()

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
        # Report/inspector hover tooltip — runs only when the user isn't
        # panning, so we never repaint mid-drag (cheap test: pan_start is
        # None). Inspector mode triggers even when no reports are present.
        if (self._pan_start is None
                and event.inaxes is self.ax
                and (self._report_meta or self.inspector_enabled)):
            self._update_hover_tooltip(event)
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

    def _update_hover_tooltip(self, event) -> None:
        """Show the annotation when the cursor is over a report scatter
        OR when the data inspector is enabled (then it shows the
        product's value at the cursor location). Report hits win when
        both apply, since they're a sharper, more specific click target.
        """
        hit_report = None
        for artist in self._report_artists:
            try:
                contains, _info = artist.contains(event)
            except (AttributeError, ValueError):
                continue
            if contains:
                hit_report = self._report_meta.get(id(artist))
                if hit_report is not None:
                    break
        if hit_report is not None:
            self._hover_annotation.xy = (event.xdata, event.ydata)
            self._hover_annotation.set_text(_report_tooltip_text(hit_report))
            if not self._hover_annotation.get_visible():
                self._hover_annotation.set_visible(True)
            self._canvas.draw_idle()
            return
        # No report under cursor — maybe show the inspector readout.
        if self.inspector_enabled:
            probe = self._field_value_at(event.xdata, event.ydata)
            if probe is not None:
                self._hover_annotation.xy = (event.xdata, event.ydata)
                self._hover_annotation.set_text(probe)
                if not self._hover_annotation.get_visible():
                    self._hover_annotation.set_visible(True)
                self._canvas.draw_idle()
                return
        if self._hover_annotation.get_visible():
            self._hover_annotation.set_visible(False)
            self._canvas.draw_idle()

    def _field_value_at(self, x_km: float | None, y_km: float | None) -> str | None:
        """Look up the product's value at the cursor location.

        Maps ``(x_km, y_km)`` in panel coordinates → (range, azimuth) for
        the radar site → ray index + range-bin index → sweep data array.
        Returns a label like ``"REF: 47.5 dBZ\\n12 km @ 263°"`` or ``None``
        if the cursor is off-radar / outside the rendered sweep / no data.
        """
        rd = self._render_data
        if rd is None or x_km is None or y_km is None:
            return None
        rng_km = (x_km * x_km + y_km * y_km) ** 0.5
        if rng_km <= 0.0:
            return None
        # Azimuth in radar conventions: clockwise from north (=0 at +y).
        az_deg = (np.degrees(np.arctan2(x_km, y_km)) + 360.0) % 360.0
        ranges_m = rd["ranges_m"]
        if ranges_m.size == 0:
            return None
        rng_m = rng_km * 1000.0
        if rng_m > ranges_m[-1] + (ranges_m[-1] - ranges_m[-2]) / 2.0:
            return None   # past the furthest range bin
        # Closest range bin
        rb_idx = int(np.argmin(np.abs(ranges_m - rng_m)))
        # Closest ray azimuth (modular distance)
        az = rd["azimuths_deg"]
        if az.size == 0:
            return None
        diff = np.minimum(np.abs(az - az_deg), 360.0 - np.abs(az - az_deg))
        ray_idx = int(np.argmin(diff))
        sweep_data = rd["sweep_data"]
        try:
            val = sweep_data[ray_idx, rb_idx]
        except (IndexError, TypeError):
            return None
        # Mask check (numpy masked arrays carry .mask)
        if hasattr(val, "mask") and bool(val.mask):
            return None
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        unit = rd["unit"]
        unit_part = f" {unit}" if unit else ""
        return (
            f"{rd['label']}: {fval:.2f}{unit_part}\n"
            f"{rng_km:.1f} km @ {az_deg:03.0f}°"
        )

    def _on_dblclick(self, event) -> None:
        if not event.dblclick or event.button != 1 or event.inaxes is not self.ax:
            return
        self.reset_home()

    def _draw_overlays(self, site: Site, overlays: "OverlayBundle") -> None:
        # All multi-segment line overlays go through LineCollection — one
        # artist for many segments, drastically cheaper to draw than
        # hundreds of individual Line2Ds.
        from matplotlib.collections import LineCollection
        # Range rings
        if overlays.range_rings_km:
            theta = np.linspace(0, 2 * np.pi, 180)
            ring_segments = [
                np.column_stack((r * np.sin(theta), r * np.cos(theta)))
                for r in overlays.range_rings_km
            ]
            lc = LineCollection(ring_segments, colors="#3a3a3a", linewidths=0.6, zorder=2)
            self.ax.add_collection(lc)
            self._overlay_artists.append(lc)
        # State borders
        if overlays.state_borders_xy:
            lc = LineCollection(overlays.state_borders_xy,
                                colors="#7a7a7a", linewidths=0.8, zorder=3)
            self.ax.add_collection(lc)
            self._overlay_artists.append(lc)
        # County borders (only when zoomed past a threshold to avoid clutter)
        cur_extent = self.ax.get_xlim()
        view_width_km = cur_extent[1] - cur_extent[0]
        if view_width_km < 400.0 and overlays.county_borders_xy:
            lc = LineCollection(overlays.county_borders_xy,
                                colors="#525252", linewidths=0.4, zorder=3)
            self.ax.add_collection(lc)
            self._overlay_artists.append(lc)
        # Cities: greedy non-overlap placement.
        # 1. Sort by population descending so the most-important cities win
        #    when two labels would collide.
        # 2. For each candidate, transform its label position to display
        #    (pixel) coordinates, estimate the label's bbox, and skip if it
        #    overlaps any already-placed bbox.
        # 3. Filter by zoom-dependent population threshold up front.
        pop_threshold = 100_000 if view_width_km > 250.0 else 20_000
        candidates = sorted(
            (c for c in overlays.cities if c.pop >= pop_threshold),
            key=lambda c: -c.pop,
        )
        placed_bboxes: list[tuple[float, float, float, float]] = []
        city_xs: list[float] = []
        city_ys: list[float] = []
        # Cheap glyph-width approximation in pixels at fontsize=7.
        # Avoids needing to actually render text to measure it.
        CHAR_PX = 4.5
        LABEL_PX_HEIGHT = 11
        LABEL_PAD = 2
        DOT_OFFSET = 4   # pixels between dot and label start
        for city in candidates:
            cx_km, cy_km = latlon_to_xy_km(city.lat, city.lon, site.lat, site.lon)
            # Skip if the dot is outside the visible window — it would never
            # appear and we don't want to waste a bbox slot on an off-screen
            # label.
            xlim = self.ax.get_xlim(); ylim = self.ax.get_ylim()
            if not (xlim[0] <= cx_km <= xlim[1] and ylim[0] <= cy_km <= ylim[1]):
                continue
            # Compute the label's bbox in DISPLAY (pixel) coordinates.
            px, py = self.ax.transData.transform((cx_km, cy_km))
            label_w = len(city.name) * CHAR_PX
            x0 = px + DOT_OFFSET - LABEL_PAD
            x1 = x0 + label_w + 2 * LABEL_PAD
            y0 = py - LABEL_PX_HEIGHT / 2 - LABEL_PAD
            y1 = py + LABEL_PX_HEIGHT / 2 + LABEL_PAD
            # Overlap with any previously placed label?
            if any(not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1)
                   for (bx0, by0, bx1, by1) in placed_bboxes):
                continue
            placed_bboxes.append((x0, y0, x1, y1))
            city_xs.append(cx_km)
            city_ys.append(cy_km)
            text = self.ax.text(cx_km + 2, cy_km + 2, city.name,
                                 color="#dddddd", fontsize=7, zorder=5)
            self._overlay_artists.append(text)
        if city_xs:
            dots = self.ax.scatter(city_xs, city_ys, s=6, c="#cccccc", zorder=4)
            self._overlay_artists.append(dots)


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
    # Emitted after ``set_layout`` rebuilds the panel widgets. Consumers
    # that hold matplotlib artists tied to the old axes (e.g. MotionTool's
    # storm tracks) need to clear and re-bind on this signal — the old
    # canvases are deleted by the time it fires.
    panels_rebuilt = pyqtSignal()

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
        # Scrub debounce: rapid ←/→ keypresses (esp. through SAILS-active
        # low tilts with 4 sweeps/volume) used to open a new PyART volume
        # PER keystroke, freezing the UI thread and triggering the macOS
        # beachball. With this debounce only the final pending sweep is
        # rendered; `_current_sweep` is updated synchronously so step_time
        # still walks the index correctly.
        self._pending_render_sweep: SweepRef | None = None
        self._scrub_debounce_timer = QTimer(self)
        self._scrub_debounce_timer.setSingleShot(True)
        self._scrub_debounce_timer.setInterval(60)   # ~16 fps catch-up
        self._scrub_debounce_timer.timeout.connect(self._do_render_pending)
        # LRU cache of (file → loaded+dealiased PyART Radar) so scrubbing
        # rapidly between adjacent volumes doesn't re-parse them.
        self._radar_lru: OrderedDict[Path, object] = OrderedDict()
        self.radar_lru_size = max(RADAR_LRU_MIN, min(RADAR_LRU_MAX, int(radar_lru_size)))
        self._dealias_mode = dealias_mode
        self._dealiased_for_mode: DealiasMode | None = None
        # Live storm reports drawn on each render; tied to the panel's display
        # time so scrubbing back recomputes which reports are visible.
        self.live_reports: list[Report] = []
        # Game polygon (verification boundary) — drawn on every panel so the
        # player can see where their warnings can actually verify (plan §4a).
        self.game_polygon: GamePolygon | None = None
        # Player's own warnings & MCDs to overlay on each panel — drawn only
        # for the revision active at the panel's display time. Lets the
        # player see exactly which polygon was in force when scrubbing.
        self.player_warnings: list = []
        self.player_mcds: list = []
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
        self._build_toolbar()
        self._build_scrubber()

    def set_live_reports(self, reports: list[Report]) -> None:
        """Provide the report set whose visible subset is computed at render time."""
        self.live_reports = reports
        if self._current_sweep is not None:
            self._render_all()

    def set_game_polygon(self, polygon: GamePolygon | None) -> None:
        """Provide the game boundary (verification polygon). Drawn on every
        panel until cleared. ``None`` removes it."""
        self.game_polygon = polygon
        if self._current_sweep is not None:
            self._render_all()

    def set_inspector_enabled(self, enabled: bool) -> None:
        """Toggle the data-probe across every panel. When on, hovering over
        a radar pixel reveals the product's value at that point."""
        for panel in self._panels:
            panel.inspector_enabled = enabled
            # If turning off, clear any visible probe text immediately.
            if not enabled and panel._hover_annotation.get_visible():
                panel._hover_annotation.set_visible(False)
                panel._canvas.draw_idle()

    def toggle_inspector(self) -> bool:
        """Flip the inspector on/off across all panels. Returns the new state."""
        if not self._panels:
            return False
        new_state = not self._panels[0].inspector_enabled
        self.set_inspector_enabled(new_state)
        return new_state

    def set_player_warnings(self, warnings: list, mcds: list | None = None) -> None:
        """Hand the panel the player's own active warnings / MCDs so they
        render as outlines on each radar canvas. Only the revision active at
        the panel's display time is drawn — scrubbing back shows the polygon
        as it was at that moment. ``None`` is equivalent to an empty list.
        """
        self.player_warnings = list(warnings or [])
        self.player_mcds = list(mcds or [])
        if self._current_sweep is not None:
            self._render_all()

    # ---- toolbar (buttons for every keybind) ----------------------------

    def _build_toolbar(self) -> None:
        """Compact toolbar below the panels with buttons for every radar
        keybind (layout, elevation, time scrub, zoom). Each button is
        labeled with its keybind so the controls are discoverable without
        memorizing the key chart.
        """
        bar = QFrame(self)
        bar.setFrameShape(QFrame.Shape.NoFrame)
        bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        h = QHBoxLayout(bar)
        h.setContentsMargins(4, 2, 4, 2)
        h.setSpacing(4)

        def _btn(label: str, tip: str, slot) -> QToolButton:
            b = QToolButton(bar)
            b.setText(label)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.clicked.connect(slot)
            return b

        # --- panel layout ----------------------------------------------
        h.addWidget(QLabel("Panels:", bar))
        h.addWidget(_btn("1  (Alt+1)", "Show 1 panel", lambda: self.set_layout(1)))
        h.addWidget(_btn("2  (Alt+2)", "Show 2 panels", lambda: self.set_layout(2)))
        h.addWidget(_btn("4  (Alt+4)", "Show 4 panels", lambda: self.set_layout(4)))
        h.addSpacing(10)

        # --- time scrubbing --------------------------------------------
        h.addWidget(QLabel("Time:", bar))
        h.addWidget(_btn("⏮ -5  (Shift+←)", "Step 5 sweeps backward",
                         lambda: self.step_time(-5)))
        h.addWidget(_btn("◀ -1  (←)", "Step 1 sweep backward",
                         lambda: self.step_time(-1)))
        h.addWidget(_btn("+1 ▶  (→)", "Step 1 sweep forward",
                         lambda: self.step_time(+1)))
        h.addWidget(_btn("+5 ⏭  (Shift+→)", "Step 5 sweeps forward",
                         lambda: self.step_time(+5)))
        h.addSpacing(10)

        # --- elevation -------------------------------------------------
        h.addWidget(QLabel("Tilt:", bar))
        h.addWidget(_btn("↑ up  (↑)", "Move up one elevation tilt",
                         lambda: self.step_elevation(+1)))
        h.addWidget(_btn("↓ down  (↓)", "Move down one elevation tilt",
                         lambda: self.step_elevation(-1)))
        h.addSpacing(10)

        # --- zoom + view reset -----------------------------------------
        h.addWidget(QLabel("View:", bar))
        h.addWidget(_btn("+  (=)", "Zoom in", lambda: self.zoom(0.8)))
        h.addWidget(_btn("−  (-)", "Zoom out", lambda: self.zoom(1.25)))
        h.addWidget(_btn("Reset  (dbl-click)", "Reset pan/zoom to home extent",
                         lambda: (self._panels[0].reset_home() if self._panels else None)))
        h.addSpacing(10)

        # --- WASD pan --------------------------------------------------
        PAN_STEP = 0.2
        h.addWidget(QLabel("Pan:", bar))
        h.addWidget(_btn("↑ (W)", "Pan north", lambda: self.pan(0.0, +PAN_STEP)))
        h.addWidget(_btn("← (A)", "Pan west",  lambda: self.pan(-PAN_STEP, 0.0)))
        h.addWidget(_btn("↓ (S)", "Pan south", lambda: self.pan(0.0, -PAN_STEP)))
        h.addWidget(_btn("→ (D)", "Pan east",  lambda: self.pan(+PAN_STEP, 0.0)))
        h.addSpacing(10)

        # --- data inspector toggle ------------------------------------
        self._inspector_btn = QToolButton(bar)
        self._inspector_btn.setText("Inspect  (I)")
        self._inspector_btn.setCheckable(True)
        self._inspector_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._inspector_btn.setToolTip(
            "Toggle the data probe. While on, hovering over the radar shows "
            "the displayed product's value at the cursor."
        )
        self._inspector_btn.toggled.connect(self.set_inspector_enabled)
        h.addWidget(self._inspector_btn)
        h.addSpacing(10)

        # --- velocity dealiasing mode ---------------------------------
        h.addWidget(QLabel("VEL dealias:", bar))
        self._dealias_combo = QComboBox(bar)
        self._dealias_combo.addItem("Region-based", DealiasMode.REGION_BASED)
        self._dealias_combo.addItem("Phase unwrap", DealiasMode.PHASE_UNWRAP)
        self._dealias_combo.addItem("None (raw)", DealiasMode.NONE)
        # Match the constructor's initial mode so the combo isn't out of
        # sync with the actual processing pipeline.
        idx = self._dealias_combo.findData(self._dealias_mode)
        if idx >= 0:
            self._dealias_combo.setCurrentIndex(idx)
        self._dealias_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._dealias_combo.setToolTip(
            "How to undo velocity aliasing (folding) before display.\n"
            "Region-based: PyART's region-growing dealiasing (default, best for most cases).\n"
            "Phase unwrap: works better for strong shear / fast storms.\n"
            "None: shows raw Doppler velocity (will fold past ±Nyquist)."
        )
        self._dealias_combo.currentIndexChanged.connect(
            lambda _i: self.set_dealias_mode(self._dealias_combo.currentData())
        )
        h.addWidget(self._dealias_combo)
        h.addStretch(1)

        outer = self.layout()
        if outer is not None:
            outer.addWidget(bar)
        self._toolbar = bar

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
        # QSlider's default StrongFocus + native arrow-key handling would eat
        # ←/→/↑/↓ when the user clicks the scrubber, blocking the play-view
        # scrub/elevation shortcuts. Scrubber is mouse-only.
        self._scrubber.setFocusPolicy(Qt.FocusPolicy.NoFocus)
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
        # Remember the user's pan/zoom so it survives the layout swap
        keep_xlim = keep_ylim = None
        if self._panels:
            keep_xlim = self._panels[0].ax.get_xlim()
            keep_ylim = self._panels[0].ax.get_ylim()
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
        # Notify consumers (e.g. MotionTool) that their old canvas refs are
        # now dead, before any render runs on the new panels.
        self.panels_rebuilt.emit()
        if self._current_sweep is not None and self._current_radar is not None:
            self._render_all()
        # Re-apply the preserved view to every new panel
        if keep_xlim and keep_ylim and keep_xlim != (0.0, 1.0):
            for p in self._panels:
                p.set_limits(keep_xlim, keep_ylim)

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
            # Hook up the per-panel product dropdown to the grid's setter so
            # the panel re-renders on selection (mirror of the digit-keybind).
            panel.product_changed.connect(
                lambda product, idx=i: self.set_product(idx, product)
            )
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

    def zoom(self, factor: float) -> None:
        """Zoom all panels around their current view center by ``factor``.

        ``factor < 1`` zooms in (smaller view), ``factor > 1`` zooms out.
        Synced across panels via the same path as scroll-zoom.
        """
        if not self._panels:
            return
        ax = self._panels[0].ax
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        new_xlim = (cx + (xmin - cx) * factor, cx + (xmax - cx) * factor)
        new_ylim = (cy + (ymin - cy) * factor, cy + (ymax - cy) * factor)
        self._broadcast_limits(new_xlim, new_ylim)

    def pan(self, dx_frac: float, dy_frac: float) -> None:
        """Pan all panels by ``(dx_frac, dy_frac)`` of the current view size.

        Positive ``dx`` moves the view east (right), positive ``dy`` north
        (up). Synced across panels via the same path as click-drag pan.
        """
        if not self._panels:
            return
        ax = self._panels[0].ax
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        dx = (xmax - xmin) * dx_frac
        dy = (ymax - ymin) * dy_frac
        new_xlim = (xmin + dx, xmax + dx)
        new_ylim = (ymin + dy, ymax + dy)
        self._broadcast_limits(new_xlim, new_ylim)

    # ---- time / elevation ----------------------------------------------

    def set_max_virtual_time(self, t: datetime | None) -> None:
        """Game-clock cap. Forward-scrubs past this are blocked."""
        self._max_virtual_time = t
        if self._current_sweep and t and self._current_sweep.start_time > t:
            self.show_latest_at(t, self._current_sweep.elev_deg)
        else:
            self._refresh_scrubber()   # cap moved forward: more sweeps reachable

    def show_sweep(self, sweep: SweepRef) -> None:
        """Public entry point for "render this sweep". Coalesces rapid calls
        (autorepeat ←/→, fast scrubbing) via a tiny QTimer so we don't open a
        new PyART volume per keystroke.

        ``_current_sweep`` is updated **immediately** so that subsequent
        ``step_time(±1)`` calls compute the next sweep from the latest
        intended position — i.e., holding ← for 10 frames moves 10 sweeps
        back, even though only the final sweep triggers a render.
        """
        if self._max_virtual_time and sweep.start_time > self._max_virtual_time:
            return
        self._current_sweep = sweep
        self._pending_render_sweep = sweep
        # Start (or restart) the debounce timer. Rapid presses keep the
        # timer alive; only the final position's render fires.
        self._scrub_debounce_timer.start()

    def _do_render_pending(self) -> None:
        """Timer-fired worker: actually load + render the latest pending sweep."""
        sweep = self._pending_render_sweep
        self._pending_render_sweep = None
        if sweep is None:
            return
        if sweep.file != self._loaded_file:
            self._current_radar = self._get_radar_from_cache(sweep.file)
            self._loaded_file = sweep.file
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
        if s is None:
            # No sweep at this elevation at-or-before our current time —
            # happens at the start of a round when the radar's first sweep
            # is timestamped slightly after the requested time. Fall back
            # to the earliest available sweep at this elevation.
            candidates = sorted(self.sweep_index.at_elevation(new_elev),
                                key=lambda x: x.start_time)
            if not candidates:
                return
            s = candidates[0]
            if self._max_virtual_time and s.start_time > self._max_virtual_time:
                return
        self.show_sweep(s)

    # ---- product control -----------------------------------------------

    def set_product(self, panel_index: int, product: str) -> None:
        if not (0 <= panel_index < len(self._panels)):
            return
        self._panels[panel_index].set_product(product)
        if self._current_sweep is not None and self._current_radar is not None:
            self._render_one(self._panels[panel_index])

    # ---- cross-volume product fallback ---------------------------------

    def _resolve_field_across_volumes(
        self,
        field: str,
        elev_deg: float,
        near_time: datetime,
    ):
        """Find an indexed sweep at ``elev_deg`` whose source volume actually
        contains unmasked ``field`` data, preferring the volume closest in
        time to ``near_time``. Returns ``(radar, sweep_no, sweep_start_time)``
        or ``None``.

        Used by panels when the currently-loaded volume has no data for the
        requested product (e.g. scrubbing onto the first surveillance sweep
        of a new volume that lacks VEL). Bounded to a handful of nearby
        volumes so a missing-on-this-radar field doesn't trigger a scan of
        every indexed file.
        """
        from ..data.sweep_index import ELEV_TOLERANCE_DEG
        candidates = self.sweep_index.at_elevation(elev_deg, tol=ELEV_TOLERANCE_DEG)
        if self._max_virtual_time:
            candidates = [c for c in candidates if c.start_time <= self._max_virtual_time]
        if not candidates:
            return None
        # Sort by absolute distance in time from `near_time`, preferring
        # earlier-or-equal on ties (the case the user reported is "first
        # sweep of a new volume" — falling back to the previous volume
        # makes more sense than jumping forward).
        candidates.sort(key=lambda s: (
            abs((s.start_time - near_time).total_seconds()),
            0 if s.start_time <= near_time else 1,
        ))
        # Cap the search so a truly-absent field doesn't load every cached
        # volume. ~6 nearby volumes (covers ±15 min of typical VCP cadence).
        SEARCH_CAP = 6
        for ref in candidates[:SEARCH_CAP]:
            # Skip the volume we already know lacks the field for this tilt
            if ref.file == self._loaded_file:
                continue
            try:
                radar = self._get_radar_from_cache(ref.file)
            except Exception as e:  # noqa: BLE001
                log.warning("Cross-volume fallback: failed to load %s (%s)", ref.file, e)
                continue
            actual_field = field
            if actual_field not in radar.fields and "velocity" in radar.fields:
                actual_field = "velocity"
            if actual_field not in radar.fields:
                continue
            data = radar.fields[actual_field]["data"]
            start = int(radar.sweep_start_ray_index["data"][ref.sweep_number])
            end = int(radar.sweep_end_ray_index["data"][ref.sweep_number]) + 1
            sl = data[start:end]
            if hasattr(sl, "mask"):
                try:
                    if sl.mask.all():
                        continue
                except (AttributeError, TypeError):
                    pass
            elif sl.size == 0:
                continue
            return (radar, ref.sweep_number, ref.start_time)
        return None

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
            cross_volume_resolver=self._resolve_field_across_volumes,
        )
        # Game polygon (verification boundary) — drawn before reports so reports
        # appear on top when overlapping.
        if self.game_polygon is not None:
            panel.draw_game_polygon(self.game_polygon, self.site)
        # Player's own warnings / MCDs — visibility is gated by the game
        # clock (so a newly-issued warning shows even on a slightly-stale
        # radar frame), while the polygon shape uses the panel's display
        # time so scrubbing back reveals revision history.
        if self.player_warnings or self.player_mcds:
            panel.draw_player_overlays(
                self.player_warnings, self.player_mcds, self.site,
                self._current_sweep.start_time,
                game_clock_time=self._max_virtual_time,
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
