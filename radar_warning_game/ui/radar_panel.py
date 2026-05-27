"""Multi-panel radar display with SAILS-aware scrubbing (plan §4a).

A :class:`RadarPanelGrid` widget holds 1, 2, or 4 :class:`RadarPanel`
instances, each showing the same radar volume at the same elevation/time
but a different product (REF / VEL / SW / CC / ZDR / PHI). Pan and zoom
are synchronized across the panels via their pyqtgraph view ranges.
KDP isn't included — its range-derivative retrieval is too slow to
keep up with scrubbing; raw differential phase (PHI) is exposed
instead for users who want to see the underlying phase signal.

Built on **pyqtgraph**. The radar sweep is rasterized once per render
from polar (azimuth, range) → Cartesian (x_km, y_km) via numpy ufuncs
and displayed as a single :class:`pyqtgraph.ImageItem`. This *is* a
polar shader — the algorithm computes the polar coordinates of every
output pixel and samples the sweep's data array — it just runs as
vectorized numpy on the CPU instead of GLSL on a GPU. The output image
is 2048×2048, giving sub-100 m pixel resolution across the 500 km
radar window — visually indistinguishable from a true polar mesh at
typical zoom levels. Build cost is ~30 ms per sweep; pan/zoom is
free (`ImageItem` is a textured quad). Overlays (state borders,
counties, cities, warnings, reports) are pyqtgraph ``QGraphicsItem``s
on the same view so everything composes correctly. The game-clock cap
(``scan_time ≤ virtual_time``) is enforced by the grid — peers can
never scrub past current game time.

Keyboard, on the focused panel (applied globally to volume time /
elevation, only the *product* is per-panel):

  ``↑`` / ``↓``      next / previous elevation tilt
  ``←`` / ``→``      previous / next sweep at current elevation (SAILS-aware)
  ``Shift+←/→``      step 5 sweeps
  ``1``…``6``        change focused panel's product (REF/VEL/SW/CC/ZDR/PHI)
  ``W A S D``        pan view (north / west / south / east)
  ``=`` / ``-``      zoom in / out
  Mouse drag         pan
  Mouse wheel        zoom toward cursor
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pyart
import pyqtgraph as pg
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..data.reports import Report
from ..data.sweep_index import SweepIndex, SweepRef
from ..data.sites import Site, site_by_icao
from ..geo.polygons import Polygon as GamePolygon
from ..geo.projection import latlon_to_xy_km
from .time_format import format_player_time

log = logging.getLogger(__name__)

# pyqtgraph global defaults — dark theme, white foreground for axes/text.
pg.setConfigOption("background", "#0a0a0a")
pg.setConfigOption("foreground", "#dddddd")
pg.setConfigOption("antialias", True)
pg.setConfigOption("useNumba", False)
# Images use row-major; matches numpy array conventions.
pg.setConfigOption("imageAxisOrder", "row-major")


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# Report visual specs. Edge color encodes the category; fills stay bright
# so even small markers (low magnitude) are legible against radar imagery.
_REPORT_SYMBOLS = {"tornado": "t1", "hail": "o", "wind": "s"}   # pg symbols
_REPORT_EDGE_COLORS = {
    "tornado": "#ff3030",
    "hail":    "#22cc55",
    "wind":    "#3399ff",
}
_REPORT_FILL_COLORS = {"tornado": "#ff4444", "hail": "#44ff66", "wind": "#66bbff"}
REPORT_FADE_SEC_RADAR = 30 * 60


class DealiasMode(str, Enum):
    """How to handle velocity-aliasing (folding) on raw NEXRAD velocity data.

    REGION_BASED is the default — PyART's region-growing algorithm gives the
    most forecaster-friendly velocity images and matches what most external
    products (RadarScope, GR2Analyst) do under the hood.
    """

    NONE = "none"
    REGION_BASED = "region_based"
    PHASE_UNWRAP = "phase_unwrap"


# Product → (PyART field name, colormap name, vmin, vmax) in m/s for velocity.
# Velocity field name is replaced at render time if dealiasing is active.
#
# KDP (specific_differential_phase) is intentionally NOT here: PyART's
# NEXRAD reader doesn't produce the range-derivative directly, and the
# retrieval (e.g. ``pyart.retrieve.kdp_vulpiani``) takes ~4 s per
# WSR-88D volume — too slow even when run in the preload pool, since
# every prefetched volume would pay that cost. We expose the raw
# differential-phase field (PHI) so users who want phase information
# can still see it; KDP can be added later if a faster retrieval
# becomes available.
PRODUCTS: dict[str, tuple[str, str, float, float]] = {
    "REF": ("reflectivity",                "ChaseSpectral", -10.0, 75.0),
    "VEL": ("velocity",                    "Carbone42",     -40.0, 40.0),
    "SW":  ("spectrum_width",              "magma",           0.0, 15.0),
    "CC":  ("cross_correlation_ratio",     "NWSRef",          0.0,  1.0),
    "ZDR": ("differential_reflectivity",   "ChaseSpectral",   0.0,  7.5),
    "PHI": ("differential_phase",          "Wild25",          0.0, 360.0),
}

# Display units for the data-probe / inspector readout. PHI is the raw
# differential phase, CC the cross-correlation ratio (unitless).
PRODUCT_UNITS: dict[str, str] = {
    "REF": "dBZ", "VEL": "m/s", "SW": "m/s",
    "CC":  "",    "ZDR": "dB",  "PHI": "°",
}

CORRECTED_VELOCITY_FIELD = "corrected_velocity"

# Field names for which "fall back to plain 'velocity' if the named
# field is missing" is the correct behavior. Used by the in-volume
# sibling + cross-volume fallbacks when they have a candidate sweep
# but the originally-requested velocity variant isn't on it (e.g. an
# older volume that wasn't dealiased yet still has raw ``velocity``).
# **Crucially**, the fallback MUST NOT fire for non-velocity requests
# — substituting velocity for a missing ``cross_correlation_ratio`` /
# ``differential_reflectivity`` and rendering the result through the
# CC / ZDR colormap produces meaningless garbage (a uniform cyan
# blanket on CC; uniform purple on ZDR) that looks like real data.
_VELOCITY_FALLBACK_FIELDS = frozenset({"velocity", "corrected_velocity"})

LAYOUT_DEFAULTS = {
    1: ("REF",),
    2: ("REF", "VEL"),
    4: ("REF", "VEL", "CC", "ZDR"),
}

DEFAULT_MAX_RANGE_KM = 250.0

# Stylesheet shared by every toggleable tool button in the radar /
# action toolbars. The :checked state is a vivid yellow highlight so
# "this tool is currently active" is unmistakable — Qt's default
# checked rendering (especially on macOS) is a near-invisible bevel.
# The :checked:disabled selector keeps the highlight even when the
# button has been auto-disabled (Warning button while a warning
# polygon is in flight); the default :disabled greying would
# otherwise mute the highlight back to invisibility.
_TOOL_BUTTON_QSS = """
QToolButton {
    padding: 3px 8px;
    border: 1px solid #444;
    border-radius: 3px;
    background-color: #2a2a2a;
    color: #dddddd;
}
QToolButton:hover { background-color: #3a3a3a; }
QToolButton:checked {
    background-color: #ffd400;
    color: #1a1a1a;
    border: 1px solid #b89400;
    font-weight: bold;
}
QToolButton:checked:disabled {
    background-color: #ffd400;
    color: #1a1a1a;
    border: 1px solid #b89400;
    font-weight: bold;
}
QToolButton:disabled { color: #777777; }
"""

# Radar-volume LRU cache: how many PyART Radar objects to keep in memory.
# Smooths rapid scrubbing — without this each volume re-open is ~200 ms.
RADAR_LRU_DEFAULT = 24
RADAR_LRU_MIN = 6
RADAR_LRU_MAX = 100


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _report_tooltip_text(r: "Report") -> str:
    """Compact label for the report-hover tooltip: time + magnitude + remarks."""
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


def _format_remaining(td) -> str:
    total_sec = max(0, int(td.total_seconds()))
    if total_sec < 60:
        return f"{total_sec}s"
    m = total_sec // 60
    if m < 60:
        return f"{m}m"
    h, mm = divmod(m, 60)
    return f"{h}h {mm}m"


def _warning_hover_text(w, ref_time) -> str:
    """Compact tooltip for the player's own warning polygon shown on the
    radar panel: type tag, issuance/expiry times, time-to-expiration,
    and the warning's expected magnitudes."""
    rev = w.current_revision
    issued = format_player_time(w.original_issue_time)
    ends_dt = w.end_time()
    ends = format_player_time(ends_dt)
    lines = [rev.warning_type.value, f"Issued {issued}"]
    if ref_time is not None:
        if w.canceled_at is not None and ref_time > w.canceled_at:
            lines.append(f"Canceled at {format_player_time(w.canceled_at)}")
        elif ref_time > ends_dt:
            lines.append(f"Expired {ends}")
        else:
            remaining = ends_dt - ref_time
            lines.append(f"Expires {ends}  ({_format_remaining(remaining)} left)")
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
    # Hint only when the warning is still revisable — once canceled or
    # expired, clicking the polygon won't open the dialog.
    revisable = (
        ref_time is None
        or (
            (w.canceled_at is None or ref_time <= w.canceled_at)
            and ref_time <= ends_dt
        )
    )
    if revisable:
        lines.append("click to revise")
    return "\n".join(lines)


def _mcd_hover_text(m, ref_time) -> str:
    issued = format_player_time(m.issue_time)
    ends_dt = m.end_time()
    ends = format_player_time(ends_dt)
    lines = ["MCD", f"Issued {issued}"]
    if ref_time is not None:
        if m.canceled_at is not None and ref_time > m.canceled_at:
            lines.append(f"Canceled at {format_player_time(m.canceled_at)}")
        elif ref_time > ends_dt:
            lines.append(f"Expired {ends}")
        else:
            remaining = ends_dt - ref_time
            lines.append(f"Expires {ends}  ({_format_remaining(remaining)} left)")
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


def _report_size(category: str, magnitude: float) -> float:
    """Pixel diameter for a report's symbol in the radar scatter plot."""
    if category == "tornado":
        return 12.0 + max(0.0, float(magnitude)) * 4.0
    if category == "hail":
        return 9.0 + max(0.0, float(magnitude)) * 3.5
    if category == "wind":
        return 8.0 + max(0.0, float(magnitude) - 50.0) * 0.20
    return 9.0


_COLORMAP_CACHE: dict[str, pg.ColorMap] = {}

def _colormap(name: str) -> pg.ColorMap:
    """Fetch (and cache) a pyqtgraph ColorMap for the given matplotlib name.
    PyART's custom maps register with matplotlib at import time, so they
    flow through pyqtgraph's matplotlib bridge without extra work."""
    if name not in _COLORMAP_CACHE:
        _COLORMAP_CACHE[name] = pg.colormap.getFromMatplotlib(name)
    return _COLORMAP_CACHE[name]


# --------------------------------------------------------------------------
# Polar mesh renderer
# --------------------------------------------------------------------------
#
# Each (ray, bin) cell becomes a colored quadrilateral in km-east/north
# of the radar. The mesh is rendered with QPainter into a cached
# ``QPicture``; subsequent pans/zooms replay the picture for free.
# Build cost is dominated by per-color ``fillPath`` calls; we minimize
# them by quantizing values to ~192 color levels and batching all cells
# of a level into a single ``arrayToQPath(connect=int_array)`` path.


def _polar_corner_grid(
    az_deg: np.ndarray,
    rng_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(nrays+1, nbins+1)`` Cartesian corner arrays (in km) for a
    polar mesh whose cells are centered on the sweep's rays and gates.

    Azimuth corners are placed midway between adjacent rays in scan order,
    with wrap-around handling so the boundary between e.g. 359° and 0°
    rays doesn't gap or overlap. Range corners use the nominal gate
    spacing (assumed uniform — true for all NEXRAD moments)."""
    nrays = az_deg.size
    nbins = rng_m.size
    # Range corners: half-step shift on each side.
    range_res = float(rng_m[1] - rng_m[0]) if nbins > 1 else 250.0
    range_corners_m = np.empty(nbins + 1, dtype=np.float64)
    range_corners_m[:-1] = rng_m - range_res / 2.0
    range_corners_m[-1] = rng_m[-1] + range_res / 2.0
    range_corners_km = range_corners_m / 1000.0
    # Azimuth corners — average adjacent rays modulo 360. The "modular
    # midpoint" trick: shift the second value by ±360 if the wrap gap
    # would otherwise straddle the boundary, then average.
    az_corners_deg = np.empty(nrays + 1, dtype=np.float64)
    diffs = np.diff(az_deg)
    # Wrap each diff into (-180, 180] so the midpoint is the short arc
    # between rays, not the long way around — handles e.g. ray 359° →
    # ray 0° without going all the way around the disk.
    diffs_wrapped = ((diffs + 180.0) % 360.0) - 180.0
    # Inner corners: corner k+1 sits midway between rays k and k+1.
    az_corners_deg[1:-1] = az_deg[:-1] + diffs_wrapped / 2.0
    # Boundary corners: extrapolate half a ray-step out from the first /
    # last ray so the polar mesh covers the full azimuth disk.
    half_first = ((az_deg[1] - az_deg[0] + 180.0) % 360.0) - 180.0
    half_last = ((az_deg[-1] - az_deg[-2] + 180.0) % 360.0) - 180.0
    az_corners_deg[0] = az_deg[0] - half_first / 2.0
    az_corners_deg[-1] = az_deg[-1] + half_last / 2.0
    az_corners_rad = np.deg2rad(az_corners_deg)
    # 2-D corner grid (rays along axis 0, bins along axis 1). NEXRAD
    # convention: azimuth measured clockwise from north, so x = east =
    # r*sin(az), y = north = r*cos(az).
    sin_az = np.sin(az_corners_rad)
    cos_az = np.cos(az_corners_rad)
    x = range_corners_km[None, :] * sin_az[:, None]
    y = range_corners_km[None, :] * cos_az[:, None]
    return x, y


# --------------------------------------------------------------------------
# Polar shader (numpy CPU rasterizer + pg.ImageItem display)
# --------------------------------------------------------------------------
#
# For each output pixel we compute its polar coordinates relative to the
# radar, look up which (ray, range_bin) it samples, fetch that cell's
# value, and run the colormap. That is, *literally*, a polar shader —
# the lookup table from (pixel.x, pixel.y) → (ray_idx, bin_idx) is the
# fragment shader; it just happens to be implemented as vectorized numpy
# ufuncs on the CPU rather than GLSL on the GPU.
#
# The lookup table is cached per (image_size, extent, nrays, nbins, az
# signature, range signature) — at the same VCP/tilt across many sweeps
# it gets reused, so each fresh sweep only pays the gather + colormap
# (~10–30 ms). Output goes into a single :class:`pyqtgraph.ImageItem`;
# pan/zoom is essentially free (the GraphicsView just textures a quad).

# Pixel resolution of the rasterized polar→Cartesian radar image.
# Typical panel sizes are 700–1000 pixels wide, so 1024 gives roughly
# 1 image-pixel per panel-pixel (no upscaling artifacts). Bumping above
# 1024 quadruples both the rasterize cost and the per-frame drawImage
# cost without a visible quality win at normal pan/zoom.
RADAR_IMAGE_SIZE_PX = 1024


@dataclass
class _PolarLookup:
    """Cached lookup tables for projecting an (image-pixel → km) frame
    into (ray_idx, bin_idx, valid) per pixel.

    Keyed by the image pixel-resolution, the Cartesian rect the image
    covers, and the polar-grid signature so multiple zoom levels can
    coexist in the cache (radar VCP rarely changes within a session)."""

    image_size: int
    rect: tuple                # (x_min, x_max, y_min, y_max) in km
    nrays: int
    nbins: int
    az_signature: tuple
    ranges_signature: tuple
    ray_idx: np.ndarray
    bin_idx: np.ndarray
    valid: np.ndarray


_LOOKUP_CACHE: OrderedDict[tuple, _PolarLookup] = OrderedDict()
_LOOKUP_CACHE_MAX = 16


def _build_polar_lookup(
    az_deg: np.ndarray,
    rng_m: np.ndarray,
    *,
    image_size: int,
    rect: tuple,
) -> _PolarLookup:
    """Return (ray_idx, bin_idx, valid) lookup tables for every Cartesian
    pixel inside ``rect`` = ``(x_min, x_max, y_min, y_max)`` (km). Cached
    by polar-grid signature **and** rect so each zoom level reuses its
    table across multiple sweeps."""
    nrays = az_deg.size
    nbins = rng_m.size
    az_sig = (float(az_deg[0]), float(az_deg[-1]), nrays)
    rng_sig = (float(rng_m[0]), float(rng_m[-1]), nbins)
    cache_key = (image_size, tuple(rect), nrays, nbins, az_sig, rng_sig)
    cached = _LOOKUP_CACHE.get(cache_key)
    if cached is not None:
        _LOOKUP_CACHE.move_to_end(cache_key)
        return cached

    x_min, x_max, y_min, y_max = rect
    # Pixel-center coords for the requested rect.
    coords_x = np.linspace(x_min, x_max, image_size, endpoint=False) \
        + (x_max - x_min) / (2 * image_size)
    coords_y = np.linspace(y_min, y_max, image_size, endpoint=False) \
        + (y_max - y_min) / (2 * image_size)
    x_grid, y_grid = np.meshgrid(coords_x, coords_y)
    r_m = np.hypot(x_grid, y_grid) * 1000.0
    az = (np.degrees(np.arctan2(x_grid, y_grid)) + 360.0) % 360.0

    range_res = float(rng_m[1] - rng_m[0]) if nbins > 1 else 250.0
    range_start = float(rng_m[0])
    bin_idx = ((r_m - range_start) / range_res).astype(np.int32)
    np.clip(bin_idx, 0, nbins - 1, out=bin_idx)
    in_range = (r_m >= range_start - range_res / 2.0) & (
        r_m <= rng_m[-1] + range_res / 2.0
    )

    # Ray index — handle out-of-order azimuths (NEXRAD scans wrap at
    # arbitrary headings; SAILS sweeps interleave rays). Sort once,
    # searchsorted-and-pick-nearest with modular distance for the wrap.
    sorted_idx = np.argsort(az_deg)
    az_sorted = az_deg[sorted_idx]
    pos = np.searchsorted(az_sorted, az)
    left = np.where(pos == 0, nrays - 1, pos - 1)
    right = pos % nrays
    d_left = np.abs(az - az_sorted[left])
    d_left = np.minimum(d_left, 360.0 - d_left)
    d_right = np.abs(az - az_sorted[right])
    d_right = np.minimum(d_right, 360.0 - d_right)
    ray_sorted = np.where(d_left < d_right, left, right)
    ray_idx = sorted_idx[ray_sorted].astype(np.int32)

    lookup = _PolarLookup(
        image_size=image_size, rect=tuple(rect),
        nrays=nrays, nbins=nbins,
        az_signature=az_sig, ranges_signature=rng_sig,
        ray_idx=ray_idx, bin_idx=bin_idx, valid=in_range,
    )
    _LOOKUP_CACHE[cache_key] = lookup
    while len(_LOOKUP_CACHE) > _LOOKUP_CACHE_MAX:
        _LOOKUP_CACHE.popitem(last=False)
    return lookup


_LUT_CACHE: dict[tuple, np.ndarray] = {}


def _colormap_lut(colormap: pg.ColorMap, levels: int = 256) -> np.ndarray:
    """Pre-sample a colormap into a ``(levels, 4)`` uint8 RGBA LUT so the
    per-pixel colormap lookup becomes a fast ``np.take`` instead of four
    ``np.interp`` calls (~10× faster on multi-megapixel images)."""
    key = (id(colormap), levels)
    lut = _LUT_CACHE.get(key)
    if lut is None:
        positions = np.linspace(0.0, 1.0, levels, dtype=np.float32)
        lut = colormap.map(positions, mode="byte")
        if lut.shape[1] == 3:
            # ColorMap.map sometimes returns RGB; pad to RGBA.
            lut = np.concatenate(
                [lut, np.full((levels, 1), 255, dtype=np.uint8)], axis=1,
            )
        _LUT_CACHE[key] = lut
    return lut


def _rasterize_polar(
    az_deg: np.ndarray,
    rng_m: np.ndarray,
    data,
    *,
    image_size: int,
    rect: tuple,
    colormap: pg.ColorMap,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Rasterize ``data`` into an ``(image_size, image_size, 4)`` uint8
    RGBA image covering ``rect = (x_min, x_max, y_min, y_max)`` km of
    the Cartesian plane. Pixels outside the radar's range or where the
    data is masked/NaN go fully transparent."""
    lookup = _build_polar_lookup(
        az_deg, rng_m, image_size=image_size, rect=rect,
    )
    raw = np.asarray(data, dtype=np.float32)[lookup.ray_idx, lookup.bin_idx]
    valid = lookup.valid.copy()
    if hasattr(data, "mask") and data.mask is not False:
        try:
            valid &= ~np.asarray(data.mask)[lookup.ray_idx, lookup.bin_idx]
        except (IndexError, TypeError):
            pass
    valid &= np.isfinite(raw)
    # Quantize raw values to a 256-entry LUT index, then look up RGBA.
    # This replaces ~50 ms of `np.interp` on 4M pixels with ~5 ms of
    # `np.take` (and an arithmetic clip + cast).
    lut = _colormap_lut(colormap)
    levels = lut.shape[0]
    inv_span = 1.0 / max(vmax - vmin, 1e-9)
    idx = np.clip((raw - vmin) * (inv_span * (levels - 1)), 0, levels - 1) \
        .astype(np.uint8)
    rgba = lut[idx]
    rgba[~valid] = 0
    return rgba


# Discrete zoom-level snap-rects so the lookup cache doesn't thrash. The
# image always covers a square axis-aligned rect; we round the requested
# view rect *up* to the next snap level (giving 10-50% pan headroom) and
# pick the smallest level that still contains the view. With ~6 levels
# the cache survives normal interaction with room to spare.
RADAR_EXTENT_LEVELS_KM = [20, 40, 80, 120, 160, 250]


def _choose_view_rect(
    view_xrange: tuple[float, float],
    view_yrange: tuple[float, float],
) -> tuple[float, float, float, float]:
    """Pick an axis-aligned square rect that fully contains the view
    rect, snapped to one of :data:`RADAR_EXTENT_LEVELS_KM` half-widths
    and a coarse position grid.

    The rendered image is intentionally ~1.5× larger than the visible
    view in each direction. Small pans within that headroom hit the
    SAME chosen rect → no re-rasterize fires when ``set_view_range`` is
    called → pan stays smooth. Only when the user pans far enough to
    push the visible area off the rendered image, or zooms past a
    snap level, do we re-rasterize."""
    cx = 0.5 * (view_xrange[0] + view_xrange[1])
    cy = 0.5 * (view_yrange[0] + view_yrange[1])
    half_w = 0.5 * (view_xrange[1] - view_xrange[0])
    half_h = 0.5 * (view_yrange[1] - view_yrange[0])
    # 1.5x padding: image extent = visible × 1.5, so we have ~50%
    # headroom on each axis. Combined with a coarse position snap
    # (half / 2) the user can pan ~50% of the view diagonal before the
    # chosen rect shifts and a re-rasterize is needed.
    needed_half = max(half_w, half_h) * 1.5
    half = next((lvl for lvl in RADAR_EXTENT_LEVELS_KM if lvl >= needed_half),
                RADAR_EXTENT_LEVELS_KM[-1])
    snap_step = max(5.0, half / 2.0)
    cx = round(cx / snap_step) * snap_step
    cy = round(cy / snap_step) * snap_step
    return (cx - half, cx + half, cy - half, cy + half)


# Shared worker pool for parallel polar rasterization across panels.
# Created lazily so unit tests / headless harnesses that never render
# don't spin up extra threads. ``max_workers=4`` matches the panel-grid
# upper bound; numpy releases the GIL for the arithmetic / indexing /
# clip ops inside ``_rasterize_polar`` so 4 threads scale linearly on a
# typical multi-core laptop. The pool is intentionally module-level
# (process-wide) — it's safe across multiple grids because each
# rasterize call is self-contained on its input arrays.
_RASTERIZE_POOL: ThreadPoolExecutor | None = None


def _rasterize_pool() -> ThreadPoolExecutor:
    global _RASTERIZE_POOL
    if _RASTERIZE_POOL is None:
        _RASTERIZE_POOL = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="radar-raster",
        )
    return _RASTERIZE_POOL


class PolarRadarItem(pg.ImageItem):
    """Polar-shader radar display backed by :class:`pyqtgraph.ImageItem`.

    Rasterization is **zoom-aware**: when the user zooms in, the radar
    is re-rasterized to cover only the visible region (still at the
    full :data:`RADAR_IMAGE_SIZE_PX` pixel budget) so cell resolution
    stays sharp. View changes are debounced so rapid pans don't
    thrash the rasterizer.

    The rasterize step is split into ``prepare_render`` (main-thread
    bookkeeping + cache warmup) and ``commit_render`` (main-thread Qt
    update). The actual numpy work lives in :func:`_rasterize_polar`
    which is thread-safe given the prepared caches, so the grid can
    fan multiple panels' rasterizes out to the worker pool and join
    before doing the Qt-side ``setImage`` calls."""

    # Public for tests / callers that want to know the current frame.
    image_size = RADAR_IMAGE_SIZE_PX

    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
        self._set_rect((-DEFAULT_MAX_RANGE_KM, DEFAULT_MAX_RANGE_KM,
                         -DEFAULT_MAX_RANGE_KM, DEFAULT_MAX_RANGE_KM))
        # Latest sweep — kept so view changes can re-rasterize without
        # round-tripping back to the caller.
        self._az: np.ndarray | None = None
        self._rng: np.ndarray | None = None
        self._data = None
        self._colormap: pg.ColorMap | None = None
        self._vmin: float = 0.0
        self._vmax: float = 1.0
        self._current_rect: tuple = (-DEFAULT_MAX_RANGE_KM, DEFAULT_MAX_RANGE_KM,
                                      -DEFAULT_MAX_RANGE_KM, DEFAULT_MAX_RANGE_KM)
        # Debounce — view-driven re-renders coalesce so a fast pan/zoom
        # gesture only triggers one rasterize at the end.
        self._view_debounce = QTimer()
        self._view_debounce.setSingleShot(True)
        self._view_debounce.setInterval(40)
        self._view_debounce.timeout.connect(self._render_for_view)
        self._pending_view_rect: tuple | None = None

    # ---- public API -------------------------------------------------

    def set_data(
        self,
        az_deg: np.ndarray,
        rng_m: np.ndarray,
        data,
        *,
        colormap: pg.ColorMap,
        vmin: float,
        vmax: float,
    ) -> None:
        """Stash a new sweep and immediately (serially) rasterize it.

        The grid's batched per-sweep render uses :meth:`prepare_render`
        + :meth:`commit_render` instead so the rasterize step can run
        in parallel across panels. ``set_data`` stays for callers that
        only have a single radar item to update (tests, MotionTool's
        scrub overlays, etc.)."""
        nrays = az_deg.size
        nbins = rng_m.size
        if nrays < 2 or nbins < 2:
            self.clear()
            return
        rect = self.prepare_render(
            az_deg, rng_m, data,
            colormap=colormap, vmin=vmin, vmax=vmax,
        )
        rgba = _rasterize_polar(
            az_deg, rng_m, data,
            image_size=RADAR_IMAGE_SIZE_PX, rect=rect,
            colormap=colormap, vmin=vmin, vmax=vmax,
        )
        self.commit_render(rgba, rect)

    def prepare_render(
        self,
        az_deg: np.ndarray,
        rng_m: np.ndarray,
        data,
        *,
        colormap: pg.ColorMap,
        vmin: float,
        vmax: float,
    ) -> tuple:
        """Main-thread phase 1 of a batched render. Stashes the sweep
        inputs (so subsequent view-change re-rasterizes use this data),
        pre-warms the per-pixel polar lookup + colormap LUT caches (so
        the parallel workers don't race on cache writes), and returns
        the rect the worker should rasterize into."""
        self._az = az_deg
        self._rng = rng_m
        self._data = data
        self._colormap = colormap
        self._vmin = vmin
        self._vmax = vmax
        rect = self._pending_view_rect or self._current_rect
        # Pre-warm the shared lookup + colormap caches on the main
        # thread. If we left this to the workers, two panels racing
        # on the same cache key could both compute the lookup table
        # (a 4 MB int32 grid + bool mask) and the second's write
        # would discard the first. Warming here makes the worker
        # path a pure cache read.
        _build_polar_lookup(az_deg, rng_m,
                            image_size=RADAR_IMAGE_SIZE_PX, rect=rect)
        _colormap_lut(colormap)
        return rect

    def commit_render(self, rgba: np.ndarray, rect: tuple) -> None:
        """Main-thread phase 3: apply the worker-computed RGBA to the
        ImageItem. ``rgba`` is the array returned by
        :func:`_rasterize_polar` from the worker; ``rect`` is the rect
        passed back from :meth:`prepare_render` (kept paired so a stale
        view change doesn't paint the wrong rect)."""
        self.setImage(rgba, autoLevels=False)
        self._set_rect(rect)
        self._current_rect = rect

    def set_view_range(
        self,
        xrange: tuple[float, float],
        yrange: tuple[float, float],
    ) -> None:
        """Called from the host RadarPanel whenever the ViewBox range
        changes. Picks an appropriate snap-rect and (debounced)
        re-rasterizes the current sweep to that extent."""
        new_rect = _choose_view_rect(xrange, yrange)
        if new_rect == self._current_rect and self._pending_view_rect is None:
            return
        self._pending_view_rect = new_rect
        self._view_debounce.start()

    def clear(self) -> None:
        super().clear()
        self._az = self._rng = self._data = None

    # ---- internal ---------------------------------------------------

    def _render_for_view(self) -> None:
        if self._pending_view_rect is None:
            return
        rect = self._pending_view_rect
        self._pending_view_rect = None
        if self._data is None:
            # No sweep loaded yet — just remember the rect for when one arrives.
            self._current_rect = rect
            self._set_rect(rect)
            return
        # Serial re-rasterize is fine here — view-change re-renders are
        # already debounced to 40 ms tail of pan/zoom gestures.
        rgba = _rasterize_polar(
            self._az, self._rng, self._data,
            image_size=RADAR_IMAGE_SIZE_PX,
            rect=rect,
            colormap=self._colormap, vmin=self._vmin, vmax=self._vmax,
        )
        self.commit_render(rgba, rect)

    def _set_rect(self, rect: tuple) -> None:
        x_min, x_max, y_min, y_max = rect
        self.setRect(QRectF(x_min, y_min, x_max - x_min, y_max - y_min))


# --------------------------------------------------------------------------
# Overlay bundle (pre-projected geometry for the radar panel)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Single radar panel
# --------------------------------------------------------------------------

class RadarPanel(QFrame):
    """One product axes embedded in a pyqtgraph PlotWidget."""

    # Emitted when the user selects a product from this panel's dropdown.
    product_changed = pyqtSignal(str)
    # Emitted when the view range changes — used by the grid to sync siblings.
    view_range_changed = pyqtSignal(object, object)   # (xlim, ylim)
    # Emitted when the user clicks one of the local player's own warning
    # / MCD polygons rendered on the panel. PlayView listens and opens
    # the corresponding revise dialog.
    warning_clicked = pyqtSignal(str)   # warning_id
    mcd_clicked = pyqtSignal(str)       # mcd_id

    def __init__(self, product: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.product = product
        # User-tunable velocity range (m/s). The colormap is symmetric so
        # we just store the half-range — vmin = -vel_vmax, vmax =
        # +vel_vmax. Defaults to PRODUCTS["VEL"]'s 40 m/s but the grid
        # can override per-instance via :meth:`set_vel_range`. Doesn't
        # affect other products' fixed PRODUCTS entries.
        self._vel_vmax: float = float(PRODUCTS["VEL"][3])
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

        # Product picker — same options as the keybinds 1-7, labeled with
        # the keybind. Discoverable equivalent of "click panel + press digit".
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

        # Plot widget — owns the QGraphicsScene that hosts the polar
        # radar mesh + all pyqtgraph overlays (range rings, state/county
        # lines, warnings, reports, text, hover tooltip).
        self._plot = pg.PlotWidget(parent=self)
        self._plot.setBackground("#0a0a0a")
        self._plot.hideAxis("bottom")
        self._plot.hideAxis("left")
        # hideAxis() only hides; the AxisItem still listens to
        # sigXRangeChanged / sigYRangeChanged and re-runs setHtml +
        # auto-SI prefix on the (invisible) label every pan tick. Cut
        # those connections — saves ~10% of per-frame work in profile.
        plot_item = self._plot.getPlotItem()
        for axname in ("bottom", "left", "top", "right"):
            ax = plot_item.getAxis(axname)
            if ax is None:
                continue
            try:
                ax.linkedView().sigXRangeChanged.disconnect(ax.linkedViewChanged)
            except (TypeError, RuntimeError):
                pass
            try:
                ax.linkedView().sigYRangeChanged.disconnect(ax.linkedViewChanged)
            except (TypeError, RuntimeError):
                pass
        self._plot.setMouseEnabled(x=True, y=True)
        self._plot.setMenuEnabled(False)
        self.view: pg.ViewBox = self._plot.getViewBox()
        self.view.setAspectLocked(True)
        self.view.setRange(xRange=(-250, 250), yRange=(-250, 250),
                           padding=0, update=False)
        self.view.disableAutoRange()
        self.view.sigRangeChanged.connect(self._on_view_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._product_combo)
        layout.addWidget(self._plot, stretch=1)

        # ---- Items added to the view ----------------------------------
        # Polar radar mesh — GL-rendered, lives below the overlays.
        self._radar_item = PolarRadarItem()
        self._radar_item.setZValue(0)
        self.view.addItem(self._radar_item)
        # Two pools of overlay artists, split by lifetime:
        # * _static_overlay_items — range rings. Rebuilt only when the
        #   overlay bundle changes (e.g. site switch) so scrubbing
        #   through sweeps doesn't tear down a stable PlotCurveItem
        #   each tick.
        # * _overlay_items — game polygon, player warnings, MCDs.
        #   Rebuilt every render because per-sweep state (tier upgrades,
        #   cancels, etc.) needs to reflect the current display time.
        self._static_overlay_items: list = []
        self._overlay_items: list = []
        # id(overlay_bundle) the static items were built for. When the
        # next render_sweep arrives with the same bundle, _draw_overlays
        # early-returns and the cached statics are reused as-is. The
        # view-driven debouncer (_run_view_work) still re-culls the
        # border curves and re-lays out city labels when the user
        # actually pans/zooms — those passes are independent of sweep
        # scrubbing.
        self._overlays_bundle_id: int | None = None
        # Persistent city label items — re-laid out (not destroyed) on each
        # range-change so we can do non-overlap labeling without creating
        # new Qt objects each pan.
        self._city_dot_item: pg.ScatterPlotItem | None = None
        self._city_label_items: list[pg.TextItem] = []
        self._city_data: list[tuple[float, float, str, int]] = []   # (x,y,name,pop)
        # Source rings + persistent border items for the view-culling
        # path. Populated by _draw_overlays; pruned/rebuilt by
        # _refresh_culled_overlays on every view change.
        self._state_rings: list[np.ndarray] = []
        self._county_rings: list[np.ndarray] = []
        self._border_state_item: pg.PlotCurveItem | None = None
        self._border_county_item: pg.PlotCurveItem | None = None
        # Storm reports — one ScatterPlotItem holding all the visible ones.
        self._report_scatter: pg.ScatterPlotItem | None = None
        self._report_data: list[Report] = []   # parallel to scatter points
        # Title text (top-left in view coords). pg.LabelItem in a sibling
        # AxisItem would be cleaner; using TextItem inside the view keeps
        # the layout simpler.
        self._title = pg.TextItem("", anchor=(0, 0), color="#dddddd")
        self._title.setZValue(50)
        self.view.addItem(self._title, ignoreBounds=True)
        # Hover tooltip — single TextItem that floats near the cursor when
        # over a report (or in inspector mode, over any radar pixel).
        self._hover = pg.TextItem(
            "", anchor=(0, 1), color="#0a0a0a",
            fill=pg.mkBrush(QColor("#ffd400")),
            border=pg.mkPen(color="#000", width=0.6),
        )
        self._hover.setZValue(60)
        self._hover.hide()
        self.view.addItem(self._hover, ignoreBounds=True)

        # State ----------------------------------------------------------
        self._home_xlim: tuple[float, float] = (-250.0, 250.0)
        self._home_ylim: tuple[float, float] = (-250.0, 250.0)
        # Inspector / data-probe state. Re-populated at the end of each
        # render_sweep so the probe can map (x, y) → field value without
        # re-deriving the sweep geometry every mousemove.
        self.inspector_enabled: bool = False
        self._render_data: dict | None = None
        # Sidecars for warning/MCD polygon hover tooltips. Each entry is
        # ``(xs, ys, tooltip_text)`` in panel km coords — populated by
        # draw_player_overlays, cleared with the other overlays.
        self._hoverable_polygons: list[tuple[np.ndarray, np.ndarray, str]] = []

        # Connect scene mouse events for hover-tooltip + inspector. Using
        # the GraphicsScene's signal rather than per-canvas mpl_connect
        # makes the hover work over the entire view (incl. overlays).
        self._plot.scene().sigMouseMoved.connect(self._on_scene_mouse_moved)

        # Used by the grid to broadcast range changes — set by attach_nav.
        self._on_limits_changed = None
        # When the grid syncs this panel via set_limits, the resulting
        # _on_view_changed shouldn't echo back as another broadcast
        # (that's an infinite loop). The flag breaks the recursion.
        self._suppress_broadcast = False
        # Coalesce the view-change work (overlay cull + city relayout)
        # — ViewBox drag-pan fires sigRangeChanged ~60×/sec, and doing
        # the full bbox-cull of thousands of state/county rings + the
        # city-label layout per frame is what was driving the stutter.
        # Title text repositioning is cheap and stays immediate. Radar
        # rasterization has its own debounce inside PolarRadarItem.
        self._view_work_debounce = QTimer(self)
        self._view_work_debounce.setSingleShot(True)
        self._view_work_debounce.setInterval(40)
        self._view_work_debounce.timeout.connect(self._run_view_work)

        # Initial title (empty until first render).
        self._update_title_pos()

    # ---- product selection -------------------------------------------

    def set_product(self, product: str) -> None:
        if product not in PRODUCTS:
            raise ValueError(f"Unknown product: {product}")
        self.product = product
        # Keep the dropdown in sync when set programmatically (keybind path)
        idx = list(PRODUCTS.keys()).index(product)
        self._product_combo.blockSignals(True)
        self._product_combo.setCurrentIndex(idx)
        self._product_combo.blockSignals(False)
        # Wipe stale render bookkeeping so the probe doesn't misreport
        self._render_data = None

    def _on_combo_change(self, idx: int) -> None:
        product_keys = list(PRODUCTS.keys())
        if not (0 <= idx < len(product_keys)):
            return
        new = product_keys[idx]
        if new == self.product:
            return
        self.product_changed.emit(new)

    # ---- public navigation API ---------------------------------------

    def set_limits(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        """External setter — used by the grid to broadcast pan/zoom
        across panels.

        We DON'T block ``sigRangeChanged`` here, because the per-panel
        work fired by :meth:`_on_view_changed` (radar re-rasterize,
        overlay cull, city relayout) needs to run on the synced panels
        too — otherwise a zoom on one panel only re-rasterizes that one
        panel's radar and the siblings keep their stale low-res image.

        To prevent the obvious infinite loop, we set a guard flag so the
        synced panel's ``_on_view_changed`` does all its local work but
        skips re-broadcasting back to the grid."""
        self._suppress_broadcast = True
        try:
            self.view.setRange(xRange=xlim, yRange=ylim, padding=0, update=False)
        finally:
            self._suppress_broadcast = False

    def reset_home(self) -> None:
        # set_limits no longer suppresses sigRangeChanged, so the
        # _on_view_changed callback fires and broadcasts naturally.
        self.set_limits(self._home_xlim, self._home_ylim)

    def attach_nav(self, on_limits_changed) -> None:
        """Used by the grid to register a callback fired whenever this
        panel's view range changes (e.g. from user pan/zoom). The grid
        mirrors those limits to the sibling panels."""
        self._on_limits_changed = on_limits_changed

    def _on_view_changed(self, _vb, _ranges) -> None:
        xlim = tuple(self.view.viewRange()[0])
        ylim = tuple(self.view.viewRange()[1])
        # Immediate work — cheap and needs to track the view in real time
        # so the title doesn't lag visibly during a drag.
        self._update_title_pos()
        # Radar rasterization is debounced inside PolarRadarItem already;
        # this call just updates the pending rect / restarts the timer.
        self._radar_item.set_view_range(xlim, ylim)
        # Heavy view-driven work (overlay cull, city relayout) is
        # debounced so it runs ONCE at the tail of a pan gesture instead
        # of 60×/sec while the user is still dragging.
        self._view_work_debounce.start()
        # Only the originating panel broadcasts back to the grid; synced
        # panels set self._suppress_broadcast to break the recursion. The
        # callback gets a ref to *this* panel so the grid can skip
        # re-applying the same range back to us (saves a redundant
        # setRange round-trip — pyqtgraph would fire sigRangeChanged a
        # second time on us if it did).
        if not self._suppress_broadcast and self._on_limits_changed:
            self._on_limits_changed(self, xlim, ylim)
        self.view_range_changed.emit(xlim, ylim)

    def _run_view_work(self) -> None:
        """Tail of the debounced view-change work — heavy passes that we
        don't want firing every drag-pan frame."""
        self._relayout_city_labels()
        self._refresh_culled_overlays()

    # ---- rendering ---------------------------------------------------

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
        """Paint one sweep onto the panel (serial path).

        For batched per-sweep rendering the grid uses
        :meth:`prepare_sweep_render` + :meth:`commit_sweep_render` so
        the polar-shader rasterize can run in parallel across panels.
        This method exists for single-panel callers (tests, etc.) and
        wraps the same prepare/commit pair into one synchronous call."""
        prep = self.prepare_sweep_render(
            radar, sweep_no, site,
            display_time=display_time, overlays=overlays,
            velocity_field=velocity_field,
            cross_volume_resolver=cross_volume_resolver,
        )
        if prep["rasterize"] is not None:
            rgba = prep["rasterize"]()
        else:
            rgba = None
        self.commit_sweep_render(prep, rgba)

    def prepare_sweep_render(
        self,
        radar,
        sweep_no: int,
        site: Site,
        *,
        display_time: datetime,
        overlays: "OverlayBundle | None" = None,
        velocity_field: str | None = None,
        cross_volume_resolver=None,
    ) -> dict:
        """Main-thread phase 1: do all the bookkeeping (clear stale
        overlays, resolve sweep/field, slice sweep data) and return a
        dict containing a thread-safe ``rasterize()`` callable that
        produces the RGBA buffer for this sweep — or ``None`` if no
        rasterize is needed (no data / all masked). The companion
        :meth:`commit_sweep_render` applies the rasterizer output and
        finalizes the panel."""
        field, cmap_name, vmin, vmax = PRODUCTS[self.product]
        if self.product == "VEL":
            # Override the static PRODUCTS range with the panel's
            # user-tunable ±vel_vmax — the colormap is symmetric so
            # one half-range value fully describes it.
            vmin, vmax = -self._vel_vmax, self._vel_vmax
            if velocity_field is not None:
                field = velocity_field
                if field not in radar.fields and "velocity" in radar.fields:
                    field = "velocity"
        elev = float(radar.fixed_angle["data"][sweep_no])
        base_title = (f"{site.icao}  {self.product}  {elev:.1f}°   "
                      f"{format_player_time(display_time)}")

        # Wipe the previous dynamic overlays; static overlays + city
        # dots/labels are managed separately and persist across sweeps.
        self._clear_overlays_and_reports()
        self._render_data = None

        # In-volume sibling fallback (SAILS split surveillance / Doppler).
        effective_sweep = self._find_best_sweep(radar, sweep_no, field, elev)
        used_in_volume_fb = (
            effective_sweep is not None and effective_sweep != sweep_no
        )
        cross_radar = radar
        cross_time = display_time
        used_cross_volume = False
        if effective_sweep is None and cross_volume_resolver is not None:
            resolved = cross_volume_resolver(field, elev, display_time)
            if resolved is not None:
                cross_radar, effective_sweep, cross_time = resolved
                used_cross_volume = True

        rasterize_fn = None
        rasterize_rect: tuple | None = None
        has_data = False
        if effective_sweep is not None:
            src = cross_radar if used_cross_volume else radar
            actual_field = field
            # Velocity-only fallback: a sibling/cross-volume sweep that
            # lacks ``corrected_velocity`` (because it predates dealias
            # or dealias failed for that file) is acceptably substituted
            # with raw ``velocity``. Do NOT do this for non-velocity
            # requests — see ``_VELOCITY_FALLBACK_FIELDS`` for why.
            if (used_cross_volume
                    and field in _VELOCITY_FALLBACK_FIELDS
                    and field not in src.fields
                    and "velocity" in src.fields):
                actual_field = "velocity"
            if actual_field in src.fields:
                data = src.fields[actual_field]["data"]
                start = int(src.sweep_start_ray_index["data"][effective_sweep])
                end = int(src.sweep_end_ray_index["data"][effective_sweep]) + 1
                sweep_data = data[start:end]
                az = np.asarray(src.azimuth["data"][start:end], dtype=np.float64)
                rng_m = np.asarray(src.range["data"], dtype=np.float64)
                all_masked = (
                    hasattr(sweep_data, "mask") and sweep_data.mask is not False
                    and getattr(sweep_data, "mask", None) is not False
                    and bool(np.asarray(sweep_data.mask).all())
                )
                if not all_masked and az.size and rng_m.size:
                    cmap = _colormap(cmap_name)
                    # Stash inputs + warm caches; return-value is the
                    # rect the rasterize step will produce.
                    rasterize_rect = self._radar_item.prepare_render(
                        az, rng_m, sweep_data,
                        colormap=cmap, vmin=vmin, vmax=vmax,
                    )
                    # Capture all the inputs by value so the callable
                    # can run on a worker thread independent of any
                    # subsequent main-thread updates to ``_radar_item``.
                    def _do_rasterize(
                        az=az, rng_m=rng_m, sweep_data=sweep_data,
                        cmap=cmap, vmin=vmin, vmax=vmax,
                        rect=rasterize_rect,
                    ):
                        return _rasterize_polar(
                            az, rng_m, sweep_data,
                            image_size=RADAR_IMAGE_SIZE_PX,
                            rect=rect, colormap=cmap, vmin=vmin, vmax=vmax,
                        )
                    rasterize_fn = _do_rasterize
                    has_data = True
                    self._render_data = {
                        "azimuths_deg": az,
                        "ranges_m": rng_m,
                        "sweep_data": sweep_data,
                        "unit": PRODUCT_UNITS.get(self.product, ""),
                        "label": self.product,
                    }

        # Title text — annotate whichever fallback (if any) we hit.
        if has_data and not (used_in_volume_fb or used_cross_volume):
            title_text = base_title
        elif has_data and used_cross_volume:
            title_text = (f"{base_title}   (showing {self.product} from "
                          f"{format_player_time(cross_time)})")
        elif has_data and used_in_volume_fb:
            eff_elev = float(radar.fixed_angle["data"][effective_sweep])
            title_text = (f"{base_title}   (showing {self.product} from "
                          f"sibling sweep at {eff_elev:.1f}°)")
        else:
            title_text = f"{base_title}   (no {self.product} available)"

        return {
            "rasterize": rasterize_fn,
            "rect": rasterize_rect,
            "has_data": has_data,
            "title_text": title_text,
            "site": site,
            "overlays": overlays,
        }

    def commit_sweep_render(self, prep: dict, rgba: np.ndarray | None) -> None:
        """Main-thread phase 3: apply the rasterizer output and refresh
        title / static overlays. Called by the grid after the worker
        pool returns from rasterizing all panels in parallel."""
        if prep["has_data"] and rgba is not None and prep["rect"] is not None:
            self._radar_item.commit_render(rgba, prep["rect"])
        else:
            self._radar_item.clear()
        self._title.setText(prep["title_text"])
        self._update_title_pos()
        overlays = prep.get("overlays")
        if overlays is not None:
            self._draw_overlays(prep["site"], overlays)

    def _find_best_sweep(
        self,
        radar,
        sweep_no: int,
        field: str,
        target_elev: float,
    ) -> int | None:
        """Pick a sweep in ``radar`` at ~``target_elev`` that has unmasked
        data for ``field``. Prefers the requested ``sweep_no`` so the
        displayed time matches the scrub position exactly — even if a
        sibling sweep happens to carry the same field at higher
        resolution. Falls back to the closest-index sibling at the same
        elevation when ``sweep_no`` itself doesn't have the field."""
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

    # ---- overlay layers ----------------------------------------------

    def _clear_overlays_and_reports(self) -> None:
        """Remove only the *dynamic* per-render overlays: game polygon,
        player warnings, MCDs, storm reports. Static overlays (range
        rings, state/county borders, city dots/labels) persist across
        sweep changes — they're torn down only when the overlay bundle
        itself changes (see :meth:`_clear_static_overlays`) or when the
        view rect changes (debounced ``_run_view_work``).

        Skipping the static rebuild on every scrub tick saves the
        ``PlotCurveItem`` creation + Qt scene insertion for ~4 range
        rings, the state-border concat curve, the county concat curve,
        and the city scatter + label TextItems per panel. That's the
        single biggest contributor to scrub frame time after the radar
        rasterize itself."""
        scene = self.view.scene()
        for item in self._overlay_items:
            # Only remove if the item is actually still in our scene —
            # otherwise Qt logs a "scene 0x0 doesn't match" warning AND
            # the spurious removal can scramble surrounding scene state.
            if item.scene() is scene:
                self.view.removeItem(item)
        self._overlay_items.clear()
        if self._report_scatter is not None and self._report_scatter.scene() is scene:
            self.view.removeItem(self._report_scatter)
        self._report_scatter = None
        self._report_data = []
        self._hoverable_polygons.clear()
        if self._hover.isVisible():
            self._hover.hide()

    def _clear_static_overlays(self) -> None:
        """Tear down the cached static overlays. Called when the overlay
        bundle itself changes (e.g., the host switched radar sites and a
        new ``OverlayBundle`` arrived) — not on every sweep change."""
        scene = self.view.scene()
        for item in self._static_overlay_items:
            if item.scene() is scene:
                self.view.removeItem(item)
        self._static_overlay_items.clear()
        for attr in ("_border_state_item", "_border_county_item"):
            it = getattr(self, attr, None)
            if it is not None and it.scene() is scene:
                self.view.removeItem(it)
            setattr(self, attr, None)
        for lbl in self._city_label_items:
            if lbl.scene() is scene:
                self.view.removeItem(lbl)
        self._city_label_items.clear()
        if self._city_dot_item is not None and self._city_dot_item.scene() is scene:
            self.view.removeItem(self._city_dot_item)
        self._city_dot_item = None
        self._overlays_bundle_id = None

    def _draw_overlays(self, site: Site, overlays: "OverlayBundle") -> None:
        """(Re)build the per-bundle static overlays. Idempotent — if the
        same bundle was already drawn (matched by ``id()``), this is a
        no-op. Per-view work (border-ring cull, city-label relayout)
        still runs from the debounced ``_run_view_work`` path on actual
        pan/zoom."""
        if id(overlays) == self._overlays_bundle_id:
            return
        # New bundle — drop any prior statics, then rebuild.
        self._clear_static_overlays()
        self._overlays_bundle_id = id(overlays)
        # Range rings — always drawn (they're small and centered on radar).
        for r in overlays.range_rings_km:
            theta = np.linspace(0, 2 * np.pi, 180)
            xs = r * np.sin(theta)
            ys = r * np.cos(theta)
            item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#3a3a3a", width=0.6))
            item.setZValue(2)
            self.view.addItem(item)
            self._static_overlay_items.append(item)
        # State / county borders are view-culled separately so panning
        # / zooming re-prunes them. We just store the source rings here;
        # _refresh_culled_overlays builds the actual PlotCurveItem.
        self._state_rings = overlays.state_borders_xy
        self._county_rings = overlays.county_borders_xy
        self._refresh_culled_overlays()
        # Cities — stored for non-overlap label placement on every range change.
        self._city_data = []
        for c in overlays.cities:
            cx_km, cy_km = latlon_to_xy_km(c.lat, c.lon, site.lat, site.lon)
            self._city_data.append((cx_km, cy_km, c.name, c.pop))
        self._relayout_city_labels()

    def _refresh_culled_overlays(self) -> None:
        """Rebuild the state + county border ``PlotCurveItem``s using only
        the rings that intersect the current view rect. Called both when
        a new sweep is drawn and on every (debounce-pending) view change
        so the curve item only carries pixels the user can actually see.

        For CONUS-scale overlays this is the difference between drawing
        ~100k points per panel and ~2-10k — the per-panel pan cost drops
        accordingly."""
        if not getattr(self, "_state_rings", None) and \
           not getattr(self, "_county_rings", None):
            return
        scene = self.view.scene()
        xr = self.view.viewRange()[0]
        yr = self.view.viewRange()[1]
        # Tear down any prior culled border items.
        for attr in ("_border_state_item", "_border_county_item"):
            it = getattr(self, attr, None)
            if it is not None and it.scene() is scene:
                self.view.removeItem(it)
            setattr(self, attr, None)
        # State borders.
        visible_state = _rings_in_rect(self._state_rings or [], xr, yr)
        if visible_state:
            xs, ys = _concat_with_gaps(visible_state)
            item = pg.PlotCurveItem(
                xs, ys, pen=pg.mkPen("#7a7a7a", width=0.8),
                connect="finite",
            )
            item.setZValue(3)
            self.view.addItem(item)
            self._border_state_item = item
        # County borders — only when zoomed past a threshold (too noisy
        # at low zoom and the cull is no help when the whole CONUS is
        # visible).
        view_w = xr[1] - xr[0]
        if view_w < 400.0:
            visible_county = _rings_in_rect(self._county_rings or [], xr, yr)
            if visible_county:
                xs, ys = _concat_with_gaps(visible_county)
                item = pg.PlotCurveItem(
                    xs, ys, pen=pg.mkPen("#525252", width=0.4),
                    connect="finite",
                )
                item.setZValue(3)
                self.view.addItem(item)
                self._border_county_item = item

    def _relayout_city_labels(self) -> None:
        """Greedy non-overlap label placement based on the current view extent.

        Two-pass algorithm: the first pass places only "above-threshold"
        cities (the threshold scales with zoom — 100k+ when zoomed wide,
        20k+ when zoomed in) so wide views don't sprawl with hundreds
        of small towns. The second pass kicks in only when fewer than
        :data:`MIN_LABELED_CITIES` labels landed in the first pass — it
        relaxes the threshold all the way to the underlying
        Natural-Earth floor (~1k pop) so deep-zoom views over rural
        areas still show something instead of a blank panel. Population
        ordering is preserved across both passes so the biggest visible
        place always wins."""
        scene = self.view.scene()
        # Clear previous label items (city dot scatter is reused).
        for lbl in self._city_label_items:
            if lbl.scene() is scene:
                self.view.removeItem(lbl)
        self._city_label_items.clear()
        if self._city_dot_item is not None:
            if self._city_dot_item.scene() is scene:
                self.view.removeItem(self._city_dot_item)
            self._city_dot_item = None
        if not self._city_data:
            return

        xr = self.view.viewRange()[0]
        yr = self.view.viewRange()[1]
        view_w = xr[1] - xr[0]
        # Lower thresholds so even smaller towns show up — forecasters
        # often think in terms of "such-and-such 8k-pop town just got
        # hit", so we want labels for those, not just metros.
        # Wide views still cap at the larger threshold for clutter
        # control; zoomed views show everything down to small towns.
        if view_w > 400.0:
            pop_threshold = 10_000
        elif view_w > 150.0:
            pop_threshold = 5_000
        else:
            pop_threshold = 1_000
        # Cap on the upper end so the wide-view doesn't sprawl, but
        # guarantee a floor regardless of zoom so the user always has
        # geographic context to anchor what they're seeing on radar.
        MIN_LABELED_CITIES = 5
        MAX_LABELED_CITIES = 60
        # Iterate ALL in-view cities ordered by descending population
        # — the threshold gating is applied per-city below, not at
        # filter time, so the relaxation step can keep iterating the
        # same sorted list without rebuilding it.
        in_view = sorted(
            (c for c in self._city_data
             if xr[0] <= c[0] <= xr[1] and yr[0] <= c[1] <= yr[1]),
            key=lambda c: -c[3],
        )
        # Translate "label collision" detection into view coords by
        # estimating a per-character data width based on the current pixel
        # → data ratio.
        pix_per_data_x = self._plot.width() / max(view_w, 1e-9)
        char_data_w = 5.5 / max(pix_per_data_x, 1e-9)
        label_data_h = 12.0 / max(pix_per_data_x, 1e-9)
        dot_offset = 6.0 / max(pix_per_data_x, 1e-9)
        placed_boxes: list[tuple[float, float, float, float]] = []
        dot_xs: list[float] = []
        dot_ys: list[float] = []

        def _try_place(cx: float, cy: float, name: str) -> bool:
            w_data = len(name) * char_data_w
            x0 = cx + dot_offset
            x1 = x0 + w_data
            y0 = cy - label_data_h / 2.0
            y1 = cy + label_data_h / 2.0
            if any(not (x1 < bx0 or x0 > bx1 or y1 < by0 or y0 > by1)
                   for (bx0, by0, bx1, by1) in placed_boxes):
                return False
            placed_boxes.append((x0, y0, x1, y1))
            dot_xs.append(cx)
            dot_ys.append(cy)
            lbl = pg.TextItem(name, anchor=(0, 0.5), color="#bbbbbb")
            lbl.setPos(x0, cy)
            lbl.setZValue(5)
            self.view.addItem(lbl, ignoreBounds=True)
            self._city_label_items.append(lbl)
            return True

        # Pass 1 — only above-threshold cities, up to MAX. Clutter
        # control for wide views.
        for cx, cy, name, pop in in_view:
            if pop < pop_threshold:
                continue
            if len(self._city_label_items) >= MAX_LABELED_CITIES:
                break
            _try_place(cx, cy, name)

        # Pass 2 — relax threshold to backfill toward the minimum.
        # Only runs when the first pass left us short, which mostly
        # happens at deep zoom in rural areas. The hard 1,000 floor
        # keeps tiny hamlets out of the labels even when the area is
        # very rural — better to show fewer labels than to clutter the
        # panel with population-200 places that nobody recognizes.
        ABSOLUTE_POP_FLOOR = 1_000
        if len(self._city_label_items) < MIN_LABELED_CITIES:
            for cx, cy, name, pop in in_view:
                if pop >= pop_threshold:
                    continue   # already considered in pass 1
                if pop < ABSOLUTE_POP_FLOOR:
                    continue
                if len(self._city_label_items) >= MIN_LABELED_CITIES:
                    break
                _try_place(cx, cy, name)

        if dot_xs:
            self._city_dot_item = pg.ScatterPlotItem(
                x=dot_xs, y=dot_ys, size=3.5,
                pen=None, brush=pg.mkBrush("#cccccc"),
                pxMode=True,
            )
            self._city_dot_item.setZValue(5)
            self.view.addItem(self._city_dot_item)

    # ---- game / warning / report overlays ----------------------------

    def draw_game_polygon(self, polygon: GamePolygon, site: Site) -> None:
        verts = list(polygon.vertices) + [polygon.vertices[0]]
        xs = np.empty(len(verts), dtype=np.float64)
        ys = np.empty(len(verts), dtype=np.float64)
        for i, (lat, lon) in enumerate(verts):
            x_km, y_km = latlon_to_xy_km(lat, lon, site.lat, site.lon)
            xs[i] = x_km
            ys[i] = y_km
        item = pg.PlotCurveItem(xs, ys, pen=pg.mkPen("#ffd400", width=1.8))
        item.setZValue(6)
        self.view.addItem(item)
        self._overlay_items.append(item)

    def draw_player_overlays(
        self,
        warnings: list,
        mcds: list,
        site: Site,
        display_time: datetime,
        game_clock_time: datetime | None = None,
    ) -> None:
        """Draw the player's own warnings + MCDs.

        Warning/MCD state (visibility, type, magnitudes) is evaluated
        against the *game-clock* time — not the radar's scrub time —
        so a revision the player just issued immediately re-colors the
        polygon to the new tier even when the latest radar sweep is
        slightly older than the revision timestamp (typical: sweeps
        land every 30-90 s, so a revision made at 18:30:00 against a
        sweep at 18:29:50 would otherwise display the *old* tier
        until the next sweep arrived). The radar imagery underneath
        still reflects the scrub time; only the forecast overlay
        tracks the game clock."""
        SVR_COLOR = "#ffd400"
        TOR_COLOR = "#ff3030"
        PDS_COLOR = "#ff66cc"
        MCD_COLOR = "#3399ff"
        TORE_PINK = "#ff66cc"
        # Per-tier line widths (px). Doubled from the original
        # 1.5-3.0 px set so warning polygons read more clearly against
        # dense radar imagery — at typical zoom-in levels the thinner
        # lines were hard to spot under heavy reflectivity / velocity
        # gradients.
        TIER_LW = {
            "SVR": 3.0, "SVRC": 4.0, "SVRD": 5.2,
            "TOR": 3.0, "TORR": 3.6, "PDS_TOR": 4.8, "TORE": 6.0,
        }
        ref_time = game_clock_time if game_clock_time is not None else display_time
        for w in warnings:
            issue_t = w.original_issue_time
            cur_rev = w.current_revision
            expiry = cur_rev.revision_time + cur_rev.duration
            if w.canceled_at is not None and ref_time > w.canceled_at:
                continue
            if ref_time < issue_t or ref_time > expiry:
                continue
            # Use game-clock time (not the sweep's scrub time) so a
            # just-issued revision shows its new tier color even if
            # the latest cached sweep predates the revision_time.
            rev = w.revision_at(ref_time) or w.revisions[0]
            wt = rev.warning_type
            wt_name = wt.value if hasattr(wt, "value") else str(wt)
            verts = list(rev.polygon.vertices) + [rev.polygon.vertices[0]]
            xs = np.empty(len(verts), dtype=np.float64)
            ys = np.empty(len(verts), dtype=np.float64)
            for i, (lat, lon) in enumerate(verts):
                x_km, y_km = latlon_to_xy_km(lat, lon, site.lat, site.lon)
                xs[i] = x_km; ys[i] = y_km
            lw = TIER_LW.get(wt_name, 1.6)
            warning_id = getattr(w, "warning_id", None)

            def _bind_click(curve, wid=warning_id):
                """Wire ``sigClicked`` on a warning curve so the player
                can click it to revise. ``setClickable(width=10)`` widens
                the hit area beyond the visible line width so the click
                doesn't require pixel-perfect aim."""
                if wid is None:
                    return
                curve.setClickable(True, width=10)
                curve.sigClicked.connect(
                    lambda _c, _ev, w_id=wid: self.warning_clicked.emit(w_id)
                )

            if wt_name == "TORE":
                # TORE gets a double-stroked outline — wider pink halo
                # behind a narrower black core line — to read as the most
                # significant tier on the map at a glance.
                outer = pg.PlotCurveItem(
                    xs, ys, pen=pg.mkPen(color=TORE_PINK, width=lw + 3.0),
                )
                outer.setZValue(8)
                self.view.addItem(outer)
                self._overlay_items.append(outer)
                _bind_click(outer)
                inner = pg.PlotCurveItem(
                    xs, ys, pen=pg.mkPen(color="#000000", width=lw),
                )
                inner.setZValue(9)
                self.view.addItem(inner)
                self._overlay_items.append(inner)
                _bind_click(inner)
                self._hoverable_polygons.append(
                    (xs, ys, _warning_hover_text(w, ref_time))
                )
                continue
            if wt_name == "PDS_TOR":
                color, dash = PDS_COLOR, Qt.PenStyle.SolidLine
            elif wt_name in ("TOR", "TORR"):
                color, dash = TOR_COLOR, Qt.PenStyle.SolidLine
            else:
                color, dash = SVR_COLOR, Qt.PenStyle.DashLine
            pen = pg.mkPen(color=color, width=lw, style=dash)
            item = pg.PlotCurveItem(xs, ys, pen=pen)
            item.setZValue(8)
            self.view.addItem(item)
            self._overlay_items.append(item)
            _bind_click(item)
            self._hoverable_polygons.append(
                (xs, ys, _warning_hover_text(w, ref_time))
            )
        for m in mcds:
            if ref_time < m.issue_time:
                continue
            if m.canceled_at is not None and ref_time > m.canceled_at:
                continue
            if ref_time > m.end_time():
                continue
            verts = list(m.polygon.vertices) + [m.polygon.vertices[0]]
            xs = np.empty(len(verts), dtype=np.float64)
            ys = np.empty(len(verts), dtype=np.float64)
            for i, (lat, lon) in enumerate(verts):
                x_km, y_km = latlon_to_xy_km(lat, lon, site.lat, site.lon)
                xs[i] = x_km; ys[i] = y_km
            # 2× the original 1.4 px for visual parity with the
            # thicker warning lines.
            pen = pg.mkPen(color=MCD_COLOR, width=2.8, style=Qt.PenStyle.DotLine)
            item = pg.PlotCurveItem(xs, ys, pen=pen)
            item.setZValue(8)
            mcd_id = getattr(m, "mcd_id", None)
            if mcd_id is not None:
                item.setClickable(True, width=10)
                item.sigClicked.connect(
                    lambda _c, _ev, mid=mcd_id: self.mcd_clicked.emit(mid)
                )
            self.view.addItem(item)
            self._overlay_items.append(item)
            self._hoverable_polygons.append(
                (xs, ys, _mcd_hover_text(m, ref_time))
            )

    def draw_reports(
        self,
        reports: list[Report],
        site: Site,
        display_time: datetime,
    ) -> None:
        """Render storm reports onto the panel as a single scatter item.
        Fade alpha is keyed to ``display_time`` so scrubbing back in time
        recomputes which reports are visible and how solid each one is."""
        spots: list[dict] = []
        kept_reports: list[Report] = []
        for r in reports:
            if r.time > display_time:
                continue
            age = (display_time - r.time).total_seconds()
            if age > REPORT_FADE_SEC_RADAR * 1.5:
                continue
            alpha = max(0.15, 1.0 - age / REPORT_FADE_SEC_RADAR)
            x_km, y_km = latlon_to_xy_km(r.lat, r.lon, site.lat, site.lon)
            fill = QColor(_REPORT_FILL_COLORS[r.category])
            fill.setAlphaF(alpha)
            edge = QColor(_REPORT_EDGE_COLORS[r.category])
            edge.setAlphaF(alpha)
            spots.append(dict(
                pos=(x_km, y_km),
                size=_report_size(r.category, r.magnitude),
                symbol=_REPORT_SYMBOLS[r.category],
                pen=pg.mkPen(edge, width=1.0),
                brush=pg.mkBrush(fill),
            ))
            kept_reports.append(r)
        if not spots:
            self._report_data = []
            return
        scatter = pg.ScatterPlotItem(spots=spots, pxMode=True)
        scatter.setZValue(7)
        self.view.addItem(scatter)
        # Tracked via `_report_scatter` only — do NOT also append to
        # `_overlay_items`. Tracking the same item in both lists makes
        # _clear_overlays_and_reports remove it twice, which trips Qt's
        # "scene 0x0 doesn't match" warning and (worse) sometimes
        # invalidates surrounding scene state.
        self._report_scatter = scatter
        self._report_data = kept_reports

    # ---- mouse hover (reports + inspector) ---------------------------

    def _on_scene_mouse_moved(self, scene_pos) -> None:
        # Only respond when the cursor is over our view (multiple panels
        # share the same QApplication scene, so this fires for every panel).
        if not self.view.sceneBoundingRect().contains(scene_pos):
            self._hide_hover()
            return
        data_pt: QPointF = self.view.mapSceneToView(scene_pos)
        x, y = float(data_pt.x()), float(data_pt.y())
        # Hover priority: report > warning polygon > inspector probe.
        # A specific report under the cursor is the sharpest target;
        # polygons are next (they cover more area); the probe is a
        # whole-pixel fallback that only kicks in if neither matched.
        hit_report = self._hit_test_report(x, y)
        if hit_report is not None:
            self._show_hover(x, y, _report_tooltip_text(hit_report))
            return
        poly_text = self._hit_test_polygon(x, y)
        if poly_text is not None:
            self._show_hover(x, y, poly_text)
            return
        if self.inspector_enabled:
            probe = self._field_value_at(x, y)
            if probe is not None:
                self._show_hover(x, y, probe)
                return
        self._hide_hover()

    def _hit_test_polygon(self, x: float, y: float) -> str | None:
        """If the cursor is close enough to a warning / MCD polygon outline,
        return its tooltip text. Hit radius scales with the current view
        width so the test feels roughly the same at any zoom."""
        if not self._hoverable_polygons:
            return None
        view_w = self.view.viewRange()[0][1] - self.view.viewRange()[0][0]
        hit_radius = view_w * 0.012
        best_text: str | None = None
        best_d = float("inf")
        for xs, ys, text in self._hoverable_polygons:
            d = float(np.min(np.hypot(xs - x, ys - y)))
            if d < hit_radius and d < best_d:
                best_text = text
                best_d = d
        return best_text

    def _hit_test_report(self, x: float, y: float) -> Report | None:
        """Find the report whose marker the cursor is over. Uses a simple
        radius test in data coords scaled to the marker size."""
        if not self._report_data:
            return None
        # Cheap pixel→data conversion: 1 pixel = view_w / widget_w
        view_w = self.view.viewRange()[0][1] - self.view.viewRange()[0][0]
        widget_w = self._plot.width() or 1
        px_to_data = view_w / widget_w
        for r in self._report_data:
            rx_km, ry_km = latlon_to_xy_km(r.lat, r.lon,
                                            *self._cached_site_lat_lon())
            half_data = _report_size(r.category, r.magnitude) * 0.6 * px_to_data
            if abs(x - rx_km) < half_data and abs(y - ry_km) < half_data:
                return r
        return None

    def _cached_site_lat_lon(self) -> tuple[float, float]:
        """Site lat/lon used to project reports — pulled from render_data
        when possible. Fallback (0, 0) is harmless; reports won't be hit-
        tested before render_sweep has run."""
        rd = self._render_data
        if rd is None or "site_latlon" not in rd:
            # Set lazily on first hit-test against current grid site.
            return self._site_lat_lon
        return rd["site_latlon"]

    # The grid sets this immediately after construction so reports can be
    # projected without re-importing the radar each hover.
    _site_lat_lon: tuple[float, float] = (0.0, 0.0)

    def _field_value_at(self, x_km: float | None, y_km: float | None) -> str | None:
        """Look up the displayed product's value at the cursor — used by
        the inspector overlay."""
        rd = self._render_data
        if rd is None or x_km is None or y_km is None:
            return None
        rng_km = math.hypot(x_km, y_km)
        if rng_km <= 0:
            return None
        az_deg = (math.degrees(math.atan2(x_km, y_km)) + 360.0) % 360.0
        ranges_m = rd["ranges_m"]
        if ranges_m.size == 0:
            return None
        rng_m = rng_km * 1000.0
        if rng_m > ranges_m[-1] + (ranges_m[-1] - ranges_m[-2]) / 2.0:
            return None
        rb_idx = int(np.argmin(np.abs(ranges_m - rng_m)))
        az = rd["azimuths_deg"]
        diff = np.minimum(np.abs(az - az_deg), 360.0 - np.abs(az - az_deg))
        ray_idx = int(np.argmin(diff))
        try:
            val = rd["sweep_data"][ray_idx, rb_idx]
        except (IndexError, TypeError):
            return None
        if hasattr(val, "mask") and bool(val.mask):
            return None
        try:
            fval = float(val)
        except (TypeError, ValueError):
            return None
        unit = rd["unit"]
        unit_part = f" {unit}" if unit else ""
        return (f"{rd['label']}: {fval:.2f}{unit_part}\n"
                f"{rng_km:.1f} km @ {az_deg:03.0f}°")

    def _show_hover(self, x: float, y: float, text: str) -> None:
        self._hover.setText(text)
        self._hover.setPos(x, y)
        if not self._hover.isVisible():
            self._hover.show()

    def _hide_hover(self) -> None:
        if self._hover.isVisible():
            self._hover.hide()

    # ---- title placement ---------------------------------------------

    def _update_title_pos(self) -> None:
        xr = self.view.viewRange()[0]
        yr = self.view.viewRange()[1]
        # Top-left in view coords (anchor (0,0) means upper-left of text).
        self._title.setPos(xr[0], yr[1])


# --------------------------------------------------------------------------
# Helper: stitch a list of polylines into a single (x, y) trace with NaN gaps
# --------------------------------------------------------------------------

def _rings_in_rect(
    rings: list[np.ndarray],
    xrange: tuple[float, float],
    yrange: tuple[float, float],
) -> list[np.ndarray]:
    """Return only those rings whose AABB intersects the view rect.

    Cheap per-ring bounding-box test — for ~5000 county rings this is
    well under a millisecond in numpy. Massive win at high zoom where
    99% of CONUS-scale rings are off-screen and shouldn't be walked
    point-by-point by Qt's painter."""
    if not rings:
        return []
    xmin, xmax = xrange
    ymin, ymax = yrange
    out: list[np.ndarray] = []
    for ring in rings:
        rxmin = ring[:, 0].min(); rxmax = ring[:, 0].max()
        rymin = ring[:, 1].min(); rymax = ring[:, 1].max()
        if rxmax < xmin or rxmin > xmax or rymax < ymin or rymin > ymax:
            continue
        out.append(ring)
    return out


def _concat_with_gaps(rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate a list of ``(N, 2)`` arrays into 1D x and y arrays with
    NaN separators between adjacent rings. PlotCurveItem with
    ``connect='finite'`` then renders each ring without spurious bridges."""
    if not rings:
        return np.array([]), np.array([])
    chunks_x = []
    chunks_y = []
    for i, ring in enumerate(rings):
        chunks_x.append(ring[:, 0])
        chunks_y.append(ring[:, 1])
        if i != len(rings) - 1:
            chunks_x.append(np.array([np.nan]))
            chunks_y.append(np.array([np.nan]))
    return np.concatenate(chunks_x), np.concatenate(chunks_y)


# --------------------------------------------------------------------------
# Grid widget: 1, 2, or 4 panels of one site
# --------------------------------------------------------------------------

class RadarPanelGrid(QWidget):
    """1, 2, or 4 panels showing the same volume, different products."""

    sweep_changed = pyqtSignal(object)
    # Emitted after :meth:`set_layout` rebuilds the panel widgets so
    # consumers holding refs to the old panels (e.g. MotionTool) can clean
    # up and re-bind to the new ones.
    panels_rebuilt = pyqtSignal()
    # Emitted (via :meth:`_on_volume_indexed_from_prefetch`) when the
    # prefetcher has indexed a newly-downloaded volume for the active
    # site. Using a signal here lets the prefetcher's worker thread
    # marshal back to the main thread so the QSlider update happens on
    # the GUI thread (Qt widgets are not thread-safe). The slot is
    # always connected in __init__ regardless of whether a prefetcher
    # is attached, so attach can be wired up safely after construction.
    _sweep_index_extended = pyqtSignal()
    # Bubbled from any panel when the user clicks an own-player warning
    # / MCD polygon. PlayView opens the corresponding revise dialog.
    # Wired up in :meth:`_build_panels` since panels are re-created on
    # layout changes.
    warning_clicked = pyqtSignal(str)   # warning_id
    mcd_clicked = pyqtSignal(str)       # mcd_id

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
        available_sites: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sweep_index = sweep_index
        site = site_by_icao(site_icao)
        if site is None:
            raise ValueError(f"Unknown radar site: {site_icao}")
        self.site = site
        # The full list of radar ICAOs the host enabled for this round.
        # When > 1, a "Radar:" dropdown appears in the toolbar and
        # :meth:`set_active_site` swaps the grid between them on
        # selection. Defaults to just the initial site so callers that
        # only have one radar (e.g. tests, MotionTool's lookback panel)
        # don't see the dropdown.
        self.available_sites: list[str] = list(
            available_sites if available_sites else [site_icao.upper()],
        )
        self._max_virtual_time = max_virtual_time
        self._current_sweep: SweepRef | None = None
        self._current_radar = None
        self._loaded_file: Path | None = None
        self._radar_lru: OrderedDict[Path, object] = OrderedDict()
        self.radar_lru_size = max(RADAR_LRU_MIN, min(RADAR_LRU_MAX, int(radar_lru_size)))
        self._dealias_mode = dealias_mode
        self._dealiased_for_mode: DealiasMode | None = None
        # Live storm reports drawn on each render; tied to the panel's
        # display time so scrubbing back recomputes which are visible.
        self.live_reports: list[Report] = []
        # Game polygon (verification boundary) — drawn on every panel.
        self.game_polygon: GamePolygon | None = None
        # Player's own warnings & MCDs, drawn on every panel.
        self.player_warnings: list = []
        self.player_mcds: list = []
        # Tick-side render-skip caches: signatures of the warnings /
        # MCDs / reports last passed in, so a tick that hands us the
        # same data skips ``_render_all`` instead of paying the full
        # 4-panel rasterize for no visible change. See
        # ``set_player_warnings`` / ``set_live_reports``.
        self._warnings_render_key_cached: tuple | None = None
        self._reports_render_key_cached: tuple | None = None
        # Lazy-load overlays — empty bundle for tests / failure modes.
        try:
            from .overlay_loader import build_overlays
            self.overlays = build_overlays(self.site)
        except Exception as e:  # noqa: BLE001
            log.warning("Overlay load failed (%s) — falling back to empty bundle", e)
            self.overlays = OverlayBundle.empty()

        # Scrub debounce — coalesce rapid ←/→ presses into one render so
        # the main thread doesn't open a new PyART volume per keystroke.
        # 15 ms is short enough to feel instantaneous on a single
        # keypress while still folding a held-down arrow's keyboard
        # auto-repeat (typically 30-60 ms) into one render.
        self._pending_render_sweep: SweepRef | None = None
        self._scrub_debounce_timer = QTimer(self)
        self._scrub_debounce_timer.setSingleShot(True)
        self._scrub_debounce_timer.setInterval(15)
        self._scrub_debounce_timer.timeout.connect(self._do_render_pending)

        self._panels: list[RadarPanel] = []
        self._build_panels(n_panels, layout)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._build_toolbar()
        self._build_scrubber()
        # Auto-grow the time-scrubber whenever the prefetcher indexes
        # a newly-downloaded volume for the active site. The signal is
        # emitted from the prefetcher's download-pool thread (see
        # ``attach_prefetcher_preload``); the connection auto-uses
        # QueuedConnection across threads, so the slot runs on the GUI
        # thread where QSlider mutations are safe.
        self._sweep_index_extended.connect(self._refresh_scrubber)

    # ---- public setters ----------------------------------------------

    def set_live_reports(self, reports: list[Report]) -> None:
        # Skip the re-render if the visible-report set is unchanged
        # (same identities, same alpha-relevant fields). The game tick
        # hands us the same reports list every second until a new
        # report actually crosses virtual_time, so without this diff
        # we'd pay a full ``_render_all`` (~16 ms × 4 panels) every
        # tick for no visible change.
        new_key = self._reports_render_key(reports)
        if new_key == self._reports_render_key_cached:
            self.live_reports = reports
            return
        self._reports_render_key_cached = new_key
        self.live_reports = reports
        if self._current_sweep is not None:
            self._render_all()

    def set_game_polygon(self, polygon: GamePolygon | None) -> None:
        if self.game_polygon is polygon:
            return
        self.game_polygon = polygon
        if self._current_sweep is not None:
            self._render_all()

    @staticmethod
    def _reports_render_key(reports: list[Report]) -> tuple:
        # ``id()`` is fine: storm reports are immutable in-session, so
        # identity equality matches semantic equality. Avoids hashing
        # every report's fields on every tick.
        return tuple(id(r) for r in reports)

    @staticmethod
    def _warnings_render_key(items: list) -> tuple:
        # For each warning/MCD, the things that affect the rendered
        # outline are: identity, the currently-active revision number,
        # and the cancel time. Capturing those gives us a cheap signature
        # that picks up issuance, edits, and cancels without rerendering
        # on no-op ticks.
        out: list = []
        for it in items:
            rev_idx = (
                len(it.revisions) - 1
                if hasattr(it, "revisions") else 0
            )
            cancel = getattr(it, "canceled_at", None)
            out.append((id(it), rev_idx, cancel))
        return tuple(out)

    def set_inspector_enabled(self, enabled: bool) -> None:
        """Toggle the data probe across every panel."""
        for panel in self._panels:
            panel.inspector_enabled = enabled
            if not enabled:
                panel._hide_hover()

    def toggle_inspector(self) -> bool:
        if not self._panels:
            return False
        new_state = not self._panels[0].inspector_enabled
        self.set_inspector_enabled(new_state)
        return new_state

    def set_player_warnings(self, warnings: list, mcds: list | None = None) -> None:
        # Diff against the previous render-key so a tick that just hands
        # us the same warnings doesn't trigger a full ``_render_all``.
        # In solo this method fires every game-clock tick from
        # ``PlayView._on_tick`` whether the player has done anything or
        # not — without the diff that's an unconditional 4-panel
        # rasterize every second of real time.
        new_warnings = list(warnings or [])
        new_mcds = list(mcds or [])
        new_key = (
            self._warnings_render_key(new_warnings),
            self._warnings_render_key(new_mcds),
        )
        if new_key == self._warnings_render_key_cached:
            self.player_warnings = new_warnings
            self.player_mcds = new_mcds
            return
        self._warnings_render_key_cached = new_key
        self.player_warnings = new_warnings
        self.player_mcds = new_mcds
        if self._current_sweep is not None:
            self._render_all()

    # ---- toolbar / scrubber ------------------------------------------

    def _build_toolbar(self) -> None:
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

        # Radar selector — only shown when the round has > 1 enabled
        # radar. With a single radar there's nothing to switch to so
        # the dropdown would just be noise.
        if len(self.available_sites) > 1:
            h.addWidget(QLabel("Radar:", bar))
            self._site_combo = QComboBox(bar)
            for icao in self.available_sites:
                obj = site_by_icao(icao)
                if obj is None:
                    continue
                tag = "TDWR" if obj.is_tdwr else "WSR-88D"
                self._site_combo.addItem(f"{icao}  ({tag} — {obj.name})", icao)
            idx = self._site_combo.findData(self.site.icao)
            if idx >= 0:
                self._site_combo.setCurrentIndex(idx)
            self._site_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self._site_combo.setToolTip(
                "Switch the active radar. Each radar is downloaded + "
                "indexed independently — switching is instant once the "
                "new site's prefetch has completed; otherwise the "
                "panels stay blank until its first volume lands."
            )
            self._site_combo.currentIndexChanged.connect(
                lambda _i: self.set_active_site(self._site_combo.currentData())
            )
            h.addWidget(self._site_combo)
            h.addSpacing(10)
        else:
            self._site_combo = None
        h.addWidget(QLabel("Panels:", bar))
        h.addWidget(_btn("1  (Alt+1)", "Show 1 panel", lambda: self.set_layout(1)))
        h.addWidget(_btn("2  (Alt+2)", "Show 2 panels", lambda: self.set_layout(2)))
        h.addWidget(_btn("4  (Alt+4)", "Show 4 panels", lambda: self.set_layout(4)))
        h.addSpacing(10)
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
        h.addWidget(QLabel("Tilt:", bar))
        h.addWidget(_btn("↑ up  (↑)", "Move up one elevation tilt",
                         lambda: self.step_elevation(+1)))
        h.addWidget(_btn("↓ down  (↓)", "Move down one elevation tilt",
                         lambda: self.step_elevation(-1)))
        h.addSpacing(10)
        h.addWidget(QLabel("View:", bar))
        h.addWidget(_btn("+  (=)", "Zoom in", lambda: self.zoom(0.8)))
        h.addWidget(_btn("−  (-)", "Zoom out", lambda: self.zoom(1.25)))
        h.addWidget(_btn("Reset  (dbl-click)", "Reset pan/zoom to home extent",
                         lambda: (self._panels[0].reset_home() if self._panels else None)))
        h.addSpacing(10)
        PAN_STEP = 0.2
        h.addWidget(QLabel("Pan:", bar))
        h.addWidget(_btn("↑ (W)", "Pan north", lambda: self.pan(0.0, +PAN_STEP)))
        h.addWidget(_btn("← (A)", "Pan west",  lambda: self.pan(-PAN_STEP, 0.0)))
        h.addWidget(_btn("↓ (S)", "Pan south", lambda: self.pan(0.0, -PAN_STEP)))
        h.addWidget(_btn("→ (D)", "Pan east",  lambda: self.pan(+PAN_STEP, 0.0)))
        h.addSpacing(10)

        self._inspector_btn = QToolButton(bar)
        self._inspector_btn.setText("Inspect  (I)")
        self._inspector_btn.setCheckable(True)
        self._inspector_btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        # Match the action-bar tool buttons' active-state style so the
        # :checked state is a vivid yellow highlight instead of the
        # near-invisible Qt-default checked rendering on macOS.
        self._inspector_btn.setStyleSheet(_TOOL_BUTTON_QSS)
        self._inspector_btn.setToolTip(
            "Toggle the data probe. While on, hovering over the radar shows "
            "the displayed product's value at the cursor."
        )
        self._inspector_btn.toggled.connect(self.set_inspector_enabled)
        h.addWidget(self._inspector_btn)
        h.addSpacing(10)

        h.addWidget(QLabel("VEL dealias:", bar))
        self._dealias_combo = QComboBox(bar)
        self._dealias_combo.addItem("Region-based", DealiasMode.REGION_BASED)
        self._dealias_combo.addItem("Phase unwrap", DealiasMode.PHASE_UNWRAP)
        self._dealias_combo.addItem("None (raw)", DealiasMode.NONE)
        idx = self._dealias_combo.findData(self._dealias_mode)
        if idx >= 0:
            self._dealias_combo.setCurrentIndex(idx)
        self._dealias_combo.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._dealias_combo.setToolTip(
            "How to undo velocity aliasing (folding) before display.\n"
            "Region-based: PyART's region-growing dealiasing (default).\n"
            "Phase unwrap: works better for strong shear / fast storms.\n"
            "None: shows raw Doppler velocity (will fold past ±Nyquist)."
        )
        self._dealias_combo.currentIndexChanged.connect(
            lambda _i: self.set_dealias_mode(self._dealias_combo.currentData())
        )
        h.addWidget(self._dealias_combo)
        h.addSpacing(10)

        # VEL ± range — symmetric vmax for the velocity colormap.
        # Defaults to whatever PRODUCTS["VEL"] says (40 m/s) so we
        # don't break the visual baseline; host can stretch or
        # compress to match the day.
        h.addWidget(QLabel("VEL ±:", bar))
        from PyQt6.QtWidgets import QDoubleSpinBox
        self._vel_range_spin = QDoubleSpinBox(bar)
        self._vel_range_spin.setRange(5.0, 150.0)
        self._vel_range_spin.setSingleStep(5.0)
        self._vel_range_spin.setDecimals(0)
        self._vel_range_spin.setSuffix(" m/s")
        self._vel_range_spin.setValue(
            float(self._panels[0]._vel_vmax) if self._panels
            else float(PRODUCTS["VEL"][3])
        )
        self._vel_range_spin.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._vel_range_spin.setToolTip(
            "Symmetric ± range for the velocity color scale. Default "
            "±40 m/s suits most ordinary storms; raise to ±80+ when "
            "tracking violent rotation / strong shear so deep reds "
            "and greens don't saturate. Affects ALL VEL panels in "
            "the grid simultaneously."
        )
        self._vel_range_spin.valueChanged.connect(self.set_vel_range)
        h.addWidget(self._vel_range_spin)
        h.addStretch(1)

        outer = self.layout()
        if outer is None:
            outer = QVBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
        outer.addWidget(bar)
        self._toolbar = bar

    def _build_scrubber(self) -> None:
        self._scrubber = QSlider(Qt.Orientation.Horizontal, self)
        self._scrubber.setRange(0, 0)
        self._scrubber.setEnabled(False)
        # QSlider's default StrongFocus + native arrow-key handling would eat
        # ←/→/↑/↓ when the user clicks the scrubber. Scrubber is mouse-only.
        self._scrubber.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._scrubber.valueChanged.connect(self._on_scrubber)
        outer = self.layout()
        if outer is not None:
            outer.addWidget(self._scrubber)

    def _refresh_scrubber(self) -> None:
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

    # ---- velocity dealiasing -----------------------------------------

    @property
    def dealias_mode(self) -> DealiasMode:
        return self._dealias_mode

    def set_vel_range(self, vmax_ms: float) -> None:
        """Set the symmetric ±velocity colormap range (m/s) on every
        panel and trigger a re-render. Used by the toolbar's "VEL ±"
        spinbox so the host can stretch / compress the velocity color
        scale to match the day's flow — default ±40 m/s works for most
        garden-variety storms, but a derecho or strong tornado needs
        ±80+ to keep the deep reds/greens from saturating, while a
        weak-shear day looks washed out at ±40 and benefits from
        narrowing to ±20."""
        vmax_ms = float(max(1.0, min(150.0, vmax_ms)))
        for panel in self._panels:
            panel._vel_vmax = vmax_ms
        # Re-render only if a VEL panel actually exists in the current
        # layout — otherwise the new range value just sits ready for
        # the next time the user switches a panel to VEL.
        if self._current_radar is not None and any(
            p.product == "VEL" for p in self._panels
        ):
            self._render_all()

    def set_dealias_mode(self, mode: DealiasMode) -> None:
        if mode == self._dealias_mode:
            return
        self._dealias_mode = mode
        self._dealiased_for_mode = None
        # Keep the prefetcher's preload dealias in sync so future
        # background-loaded volumes use the same algorithm — otherwise
        # the grid would have to re-dealias every preloaded radar on
        # first display, defeating the preload win.
        prefetcher = getattr(self, "_prefetcher", None)
        if prefetcher is not None:
            prefetcher.set_preload_dealias_mode(
                mode.value if mode != DealiasMode.NONE else None
            )
        if self._current_radar is not None:
            self._apply_dealias(self._current_radar)
            self._render_all()

    def velocity_field_name(self) -> str:
        if self._dealias_mode == DealiasMode.NONE:
            return "velocity"
        return CORRECTED_VELOCITY_FIELD

    def _apply_dealias(self, radar) -> None:
        if self._dealias_mode == DealiasMode.NONE:
            return
        if self._dealiased_for_mode == self._dealias_mode and CORRECTED_VELOCITY_FIELD in radar.fields:
            return
        if "velocity" not in radar.fields:
            return
        # PyART's NEXRAD reader leaves nyquist_velocity all-zero for
        # TDWR + some legacy WSR-88D volumes, which crashes the dealias
        # algorithms with a divide-by-zero (cast to int via inf). Fix
        # the metadata first — same call the preloader makes — so this
        # main-thread path doesn't keep retripping the same exception.
        from ..data.radar_repair import ensure_nyquist_velocity
        ensure_nyquist_velocity(radar)
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
            raw = radar.fields["velocity"]
            radar.add_field(CORRECTED_VELOCITY_FIELD, dict(raw), replace_existing=True)
            self._dealiased_for_mode = self._dealias_mode

    # ---- layout management -------------------------------------------

    def set_layout(self, n_panels: int, layout: tuple[str, ...] | None = None) -> None:
        # Remember the user's pan/zoom so it survives the layout swap
        keep_xlim = keep_ylim = None
        if self._panels:
            keep_xlim = tuple(self._panels[0].view.viewRange()[0])
            keep_ylim = tuple(self._panels[0].view.viewRange()[1])
        if hasattr(self, "_grid_layout"):
            while self._grid_layout.count():
                item = self._grid_layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
                    w.deleteLater()
        self._panels.clear()
        self._build_panels(n_panels, layout)
        self.panels_rebuilt.emit()
        if self._current_sweep is not None and self._current_radar is not None:
            self._render_all()
        if keep_xlim and keep_ylim:
            for p in self._panels:
                p.set_limits(keep_xlim, keep_ylim)

    def _build_panels(self, n_panels: int, layout: tuple[str, ...] | None) -> None:
        if n_panels not in (1, 2, 4):
            raise ValueError(f"n_panels must be 1, 2, or 4 (got {n_panels})")
        products = layout or LAYOUT_DEFAULTS[n_panels]
        if len(products) != n_panels:
            raise ValueError(f"layout has {len(products)} entries but n_panels={n_panels}")
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
            panel._site_lat_lon = (self.site.lat, self.site.lon)
            if n_panels == 1:
                grid.addWidget(panel, 0, 0)
            elif n_panels == 2:
                grid.addWidget(panel, 0, i)
            else:
                grid.addWidget(panel, i // 2, i % 2)
            panel.attach_nav(self._broadcast_limits)
            panel.product_changed.connect(
                lambda product, idx=i: self.set_product(idx, product)
            )
            # Bubble per-panel warning/MCD click events up to the grid
            # so PlayView (one listener at the grid level) doesn't need
            # to re-wire per panel after every layout change.
            panel.warning_clicked.connect(self.warning_clicked)
            panel.mcd_clicked.connect(self.mcd_clicked)
            self._panels.append(panel)
        outer.addLayout(grid, stretch=1)
        self._grid_layout = grid

    def _broadcast_limits(
        self,
        origin: "RadarPanel | None",
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        """One panel pan/zoomed → mirror to all others. ``origin`` is the
        panel that originated the change (or ``None`` for grid-level
        synthetic events like the zoom/pan buttons) — it's skipped because
        its range is already correct and reapplying would just fire a
        redundant sigRangeChanged on it."""
        for panel in self._panels:
            if panel is origin:
                continue
            cur_x = tuple(panel.view.viewRange()[0])
            cur_y = tuple(panel.view.viewRange()[1])
            if cur_x != xlim or cur_y != ylim:
                panel.set_limits(xlim, ylim)

    def zoom(self, factor: float) -> None:
        if not self._panels:
            return
        xmin, xmax = self._panels[0].view.viewRange()[0]
        ymin, ymax = self._panels[0].view.viewRange()[1]
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        new_xlim = (cx + (xmin - cx) * factor, cx + (xmax - cx) * factor)
        new_ylim = (cy + (ymin - cy) * factor, cy + (ymax - cy) * factor)
        self._broadcast_limits(None, new_xlim, new_ylim)

    def pan(self, dx_frac: float, dy_frac: float) -> None:
        if not self._panels:
            return
        xmin, xmax = self._panels[0].view.viewRange()[0]
        ymin, ymax = self._panels[0].view.viewRange()[1]
        dx = (xmax - xmin) * dx_frac
        dy = (ymax - ymin) * dy_frac
        new_xlim = (xmin + dx, xmax + dx)
        new_ylim = (ymin + dy, ymax + dy)
        self._broadcast_limits(None, new_xlim, new_ylim)

    # ---- cross-volume product fallback -------------------------------

    def _resolve_field_across_volumes(
        self,
        field: str,
        elev_deg: float,
        near_time: datetime,
    ):
        from ..data.sweep_index import ELEV_TOLERANCE_DEG
        candidates = self.sweep_index.at_elevation(elev_deg, tol=ELEV_TOLERANCE_DEG)
        if self._max_virtual_time:
            candidates = [c for c in candidates if c.start_time <= self._max_virtual_time]
        if not candidates:
            return None
        candidates.sort(key=lambda s: (
            abs((s.start_time - near_time).total_seconds()),
            0 if s.start_time <= near_time else 1,
        ))
        SEARCH_CAP = 6
        for ref in candidates[:SEARCH_CAP]:
            if ref.file == self._loaded_file:
                continue
            try:
                radar = self._get_radar_from_cache(ref.file)
            except Exception as e:  # noqa: BLE001
                log.warning("Cross-volume fallback: failed to load %s (%s)", ref.file, e)
                continue
            actual_field = field
            # Velocity-only fallback (see _VELOCITY_FALLBACK_FIELDS):
            # an older volume that wasn't dealiased can substitute raw
            # ``velocity`` for ``corrected_velocity``. NEVER substitute
            # velocity for unrelated fields — rendering velocity values
            # through the CC / ZDR / KDP colormap produced a uniform
            # cyan/purple blanket that looked like real data.
            if (field in _VELOCITY_FALLBACK_FIELDS
                    and actual_field not in radar.fields
                    and "velocity" in radar.fields):
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

    # ---- rendering ---------------------------------------------------

    def _render_all(self) -> None:
        """Render the current sweep across every panel in parallel.

        The polar-shader rasterize is the dominant per-frame cost
        (~15 ms × N panels at warm cache). Each panel rasterizes
        independently into its own RGBA buffer, so we fan the
        ``rasterize`` callables produced by ``prepare_sweep_render``
        out to a shared :class:`ThreadPoolExecutor` and join before
        applying the results back on the main thread. With four panels
        and four cores the serial 60 ms cost collapses to ~15-20 ms."""
        if self._current_radar is None or self._current_sweep is None:
            # Show a placeholder title across every panel so a blank
            # screen is observable instead of silent. Distinguishes
            # "no sweep ever loaded" (prefetch hasn't finished, scan
            # index empty, file parse failed) from "real render but no
            # data" (which shows '(no <PRODUCT> available)' instead).
            why = []
            if self._current_radar is None:
                why.append("no radar loaded")
            if self._current_sweep is None:
                why.append("no sweep selected")
            placeholder = f"{self.site.icao}  —  {' / '.join(why)}"
            for panel in self._panels:
                panel._title.setText(placeholder)
                panel._update_title_pos()
                panel._radar_item.clear()
            log.warning(
                "_render_all skipped: %s (site=%s)", " / ".join(why), self.site.icao,
            )
            return
        # Phase 1: prepare + warm caches on main thread (each panel).
        preps: list[tuple[RadarPanel, dict]] = []
        for panel in self._panels:
            prep = panel.prepare_sweep_render(
                self._current_radar,
                self._current_sweep.sweep_number,
                self.site,
                display_time=self._current_sweep.start_time,
                overlays=self.overlays,
                velocity_field=self.velocity_field_name(),
                cross_volume_resolver=self._resolve_field_across_volumes,
            )
            preps.append((panel, prep))
        # Phase 2: rasterize across panels in parallel. numpy releases
        # the GIL for the gather + clip + take ops so threads scale.
        if len(preps) > 1 and any(p["rasterize"] is not None for _, p in preps):
            pool = _rasterize_pool()
            futures = [
                pool.submit(prep["rasterize"]) if prep["rasterize"] is not None
                else None
                for _, prep in preps
            ]
            rgbas = [f.result() if f is not None else None for f in futures]
        else:
            rgbas = [
                prep["rasterize"]() if prep["rasterize"] is not None else None
                for _, prep in preps
            ]
        # Phase 3: apply on main thread (Qt single-thread requirement).
        for (panel, prep), rgba in zip(preps, rgbas):
            panel.commit_sweep_render(prep, rgba)
            self._draw_dynamic_overlays(panel)

    def _render_one(self, panel: RadarPanel) -> None:
        """Single-panel render path — used when only one panel's state
        changed (e.g. product hotkey). Falls through to the serial
        ``render_sweep`` since parallelism doesn't buy anything for one
        panel."""
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
        self._draw_dynamic_overlays(panel)

    def _draw_dynamic_overlays(self, panel: RadarPanel) -> None:
        """Game polygon + player warnings + live reports — drawn after
        the polar shader so they layer on top. Cheap relative to
        rasterize; runs serially on each panel."""
        if self.game_polygon is not None:
            panel.draw_game_polygon(self.game_polygon, self.site)
        if self.player_warnings or self.player_mcds:
            panel.draw_player_overlays(
                self.player_warnings, self.player_mcds, self.site,
                self._current_sweep.start_time,
                game_clock_time=self._max_virtual_time,
            )
        if self.live_reports:
            panel.draw_reports(
                self.live_reports, self.site, self._current_sweep.start_time,
            )

    # ---- keyboard ----------------------------------------------------

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

    # ---- product control --------------------------------------------

    def set_product(self, panel_index: int, product: str) -> None:
        if not (0 <= panel_index < len(self._panels)):
            return
        self._panels[panel_index].set_product(product)
        if self._current_sweep is not None and self._current_radar is not None:
            self._render_one(self._panels[panel_index])

    # ---- time / elevation -------------------------------------------

    def set_max_virtual_time(self, t: datetime | None) -> None:
        self._max_virtual_time = t
        if self._current_sweep and t and self._current_sweep.start_time > t:
            self.show_latest_at(t, self._current_sweep.elev_deg)
        else:
            self._refresh_scrubber()

    def set_active_site(self, icao: str) -> None:
        """Switch the grid to display a different radar in the
        ``available_sites`` list. Repoints the sweep_index, site lat/
        lon, and overlay bundle; clears the radar LRU (different site
        = different files); and re-picks an initial sweep at the
        current display time (or as close to it as available)."""
        icao = icao.upper()
        if icao == self.site.icao:
            return
        if icao not in {s.upper() for s in self.available_sites}:
            log.warning("set_active_site(%s): not in available_sites=%s",
                        icao, self.available_sites)
            return
        new_site = site_by_icao(icao)
        if new_site is None:
            log.warning("set_active_site(%s): unknown ICAO", icao)
            return
        # Need a prefetcher to source the new site's sweep_index. In
        # tests / single-radar bare usage the grid wasn't given a
        # prefetcher; we can't switch in that case.
        prefetcher = getattr(self, "_prefetcher", None)
        if prefetcher is None:
            log.warning("set_active_site(%s): no prefetcher attached "
                        "— can't resolve sweep_index for the new site", icao)
            return
        try:
            new_index = prefetcher.sweep_index(icao)
        except KeyError:
            log.warning("set_active_site(%s): prefetcher doesn't know "
                        "this site", icao)
            return
        # Remember the time we were viewing so we can land near it.
        target_time = (
            self._current_sweep.start_time if self._current_sweep is not None
            else self._max_virtual_time
        )
        target_elev = (
            self._current_sweep.elev_deg if self._current_sweep is not None
            else 0.5
        )
        # Swap state.
        self.site = new_site
        self.sweep_index = new_index
        self._current_sweep = None
        self._current_radar = None
        self._loaded_file = None
        self._radar_lru.clear()
        # Reload overlays for the new radar's origin (state borders,
        # cities etc. are projected to the radar's lat/lon).
        try:
            from .overlay_loader import build_overlays
            self.overlays = build_overlays(new_site)
        except Exception as e:  # noqa: BLE001
            log.warning("Overlay reload failed for %s: %s — falling "
                        "back to empty bundle", icao, e)
            self.overlays = OverlayBundle.empty()
        # Per-panel state that was bound to the old site.
        for p in self._panels:
            p._site_lat_lon = (new_site.lat, new_site.lon)
            # Force a static-overlay rebuild for the new bundle —
            # the bundle-id keying in _draw_overlays would skip the
            # rebuild otherwise since the panel's cached bundle id
            # still points at the old site's bundle.
            p._clear_static_overlays()
            # Clear the dynamic overlay caches so warnings/MCDs/reports
            # re-project to the new origin on next render.
            p._warnings_render_key_cached = None
            p._reports_render_key_cached = None
        # Pick an initial sweep at the target time and elevation. If
        # the new site has no sweeps yet (its prefetch hasn't landed
        # any volumes), _render_all will show the placeholder title
        # and we'll fill in when the prefetcher fires the indexed
        # callback.
        if target_time is not None:
            initial = self.sweep_index.latest_at_or_before(
                target_time, elev_deg=target_elev,
            )
            if initial is None:
                # Fall back to the lowest-elevation sweep nearest in time.
                low = sorted(self.sweep_index.at_elevation(target_elev),
                             key=lambda s: abs((s.start_time - target_time).total_seconds()))
                initial = low[0] if low else None
            if initial is not None:
                self.show_sweep(initial)
                return
        # No matching sweep — render the placeholder so the user knows
        # the switch took effect and is just waiting on data.
        self._render_all()

    def show_sweep(self, sweep: SweepRef) -> None:
        """Public entry point — debounced. Updates ``_current_sweep``
        synchronously so step_time() walks the index from the latest
        intended position; the heavy load+render runs after a short idle."""
        if self._max_virtual_time and sweep.start_time > self._max_virtual_time:
            return
        self._current_sweep = sweep
        self._pending_render_sweep = sweep
        self._scrub_debounce_timer.start()

    def _do_render_pending(self) -> None:
        sweep = self._pending_render_sweep
        self._pending_render_sweep = None
        if sweep is None:
            return
        # Wrap the whole load+render in try/except so a single-file
        # parse failure (e.g. a corrupted Level 2 file, or a format
        # PyART doesn't recognize on older WSR-88D archives) surfaces
        # in the log instead of leaving the panels mysteriously blank.
        # The QTimer that fires this can re-fire on the next scrub
        # attempt; we don't want one bad file to wedge the grid
        # silently.
        try:
            if sweep.file != self._loaded_file:
                self._current_radar = self._get_radar_from_cache(sweep.file)
                self._loaded_file = sweep.file
            self._render_all()
        except Exception:  # noqa: BLE001
            log.exception(
                "render failed for sweep %s (file=%s) — panels will stay "
                "on last successful render until next scrub attempt",
                sweep, sweep.file,
            )
            # Reset so a re-attempt re-tries the load instead of using a
            # half-initialized cached state.
            self._loaded_file = None
            self._current_radar = None
            return
        self._refresh_scrubber()
        self.sweep_changed.emit(sweep)

    def attach_prefetcher_preload(self, prefetcher) -> None:
        """Wire the grid's radar LRU to the prefetcher's background
        preload cache. After this call, ``_get_radar_from_cache`` checks
        the prefetcher's already-parsed + already-dealiased Radar
        objects before falling back to a synchronous read on the main
        thread — eliminating the multi-hundred-ms stall that hits every
        time scrubbing crosses a volume boundary.

        Also wires the prefetcher's ``on_volume_indexed`` callback so
        the time-scrubber QSlider extends its range automatically as
        new sweeps arrive — particularly important in live mode where
        new volumes stream in every ~5 minutes, but also matters in
        historical play when the in-game lookahead buffer pulls a new
        chunk."""
        self._prefetcher = prefetcher
        # Tell the prefetcher to dealias preloaded volumes with the
        # current mode so its cache and ours stay consistent.
        mode_str = self._dealias_mode.value if self._dealias_mode \
            else None
        prefetcher.set_preload_dealias_mode(mode_str)
        # Pre-warm: any volumes the prefetcher already finished loading
        # before this hook-up get promoted into our LRU now.
        prefetcher.set_radar_preloaded_callback(self._on_radar_preloaded)
        # Scrubber auto-extension hook.
        prefetcher.set_volume_indexed_callback(self._on_volume_indexed_from_prefetch)

    def _on_volume_indexed_from_prefetch(self, site: str, file: Path) -> None:
        """Fires on the prefetcher's download-pool thread the moment a
        new volume lands in the sweep-index. Filter to our active
        site, then emit the Qt signal — connected in ``__init__`` with
        an auto-queued cross-thread connection — so the actual
        QSlider mutation runs on the GUI thread."""
        if site.upper() != self.site.icao.upper():
            return
        self._sweep_index_extended.emit()

    def _on_radar_preloaded(self, file: Path, radar) -> None:
        """Callback fired off the prefetcher's preload pool. Promote the
        loaded Radar into our LRU so the next scrub into that volume is
        instant. Safe to call from a worker thread — dict assignment is
        atomic in CPython and the LRU's bound check happens under no
        contention here (the grid only ever reads under main thread)."""
        self._radar_lru[file] = radar
        while len(self._radar_lru) > self.radar_lru_size:
            self._radar_lru.popitem(last=False)

    def _get_radar_from_cache(self, file: Path):
        cached = self._radar_lru.get(file)
        if cached is not None:
            self._radar_lru.move_to_end(file)
            return cached
        # Check the prefetcher's preload cache before falling back to a
        # blocking synchronous read on the main thread. The preloader
        # has already paid the parse + dealias cost for any volume that
        # finished downloading, so this turns the volume-crossing stall
        # from a multi-hundred-ms block into a dict lookup.
        prefetcher = getattr(self, "_prefetcher", None)
        if prefetcher is not None:
            preloaded = prefetcher.get_loaded_radar(file)
            if preloaded is None:
                # If a preload is currently in flight, give it ~250 ms
                # to land — faster than starting a duplicate parse.
                preloaded = prefetcher.wait_for_preload(file, timeout=0.25)
            if preloaded is not None:
                self._radar_lru[file] = preloaded
                while len(self._radar_lru) > self.radar_lru_size:
                    self._radar_lru.popitem(last=False)
                # If the user toggled to a different dealias mode
                # mid-round, _apply_dealias is idempotent and re-runs
                # if the cached mode doesn't match.
                self._dealiased_for_mode = None
                self._apply_dealias(preloaded)
                return preloaded
        radar = pyart.io.read_nexrad_archive(str(file))
        self._dealiased_for_mode = None
        self._apply_dealias(radar)
        self._radar_lru[file] = radar
        while len(self._radar_lru) > self.radar_lru_size:
            self._radar_lru.popitem(last=False)
        return radar

    def show_latest_at(self, display_time: datetime, elev_deg: float) -> None:
        s = self.sweep_index.latest_at_or_before(display_time, elev_deg=elev_deg)
        if s is not None:
            self.show_sweep(s)

    def step_time(self, n: int) -> None:
        if self._current_sweep is None:
            return
        nxt = self.sweep_index.step_in_elevation(self._current_sweep, n)
        if nxt is None:
            return
        if self._max_virtual_time and nxt.start_time > self._max_virtual_time:
            return
        self.show_sweep(nxt)

    def step_elevation(self, n: int) -> None:
        if self._current_sweep is None:
            return
        elevs = self.sweep_index.available_elevations(self._current_sweep.start_time)
        if not elevs:
            return
        cur_idx = min(range(len(elevs)),
                      key=lambda i: abs(elevs[i] - self._current_sweep.elev_deg))
        new_idx = max(0, min(len(elevs) - 1, cur_idx + n))
        new_elev = elevs[new_idx]
        # Pick the sweep at the new elevation whose start_time is
        # *closest* to the current sweep — this preserves the volume
        # context across elevation switches. The old behavior used
        # ``latest_at_or_before(current_time)`` which silently fell
        # back to "the first sweep at this elevation in the whole
        # indexed range" (often 20+ minutes before round start) when
        # the radar's current volume hadn't yet completed that tilt.
        # That made ↑/↓ feel like it jumped to the round's start.
        candidates = self.sweep_index.at_elevation(new_elev)
        if self._max_virtual_time:
            candidates = [
                c for c in candidates
                if c.start_time <= self._max_virtual_time
            ]
        if not candidates:
            return
        cur_time = self._current_sweep.start_time
        s = min(
            candidates,
            key=lambda c: abs((c.start_time - cur_time).total_seconds()),
        )
        self.show_sweep(s)
