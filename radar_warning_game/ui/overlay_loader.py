"""Build :class:`OverlayBundle`s from Cartopy / Natural Earth shapefiles.

Layers populated:
  - **State borders** (Cartopy admin_1 states_provinces, 10m).
  - **Cities** — union of Cartopy's Natural Earth populated_places (global
    coverage for the world bbox) and a bundled GeoNames-derived US CSV
    (``resources/cities/us_cities.csv``, ~22k entries) that fills in every
    county seat plus every place ≥500 population. Natural Earth alone is
    extremely sparse over rural CONUS — most counties had zero labeled
    towns — so for any radar inside the US the GeoNames table is what the
    user actually sees once they zoom in.
  - **US counties** (bundled TIGER ``cb_*_us_county_500k.shp``).

All geometry is projected into km east/north of the radar site for direct
drawing on the radar panel's pre-existing km-coordinate axes.
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

import cartopy.io.shapereader as shpreader
import numpy as np
from shapely.geometry import LineString

from ..data.sites import Site
from ..geo.projection import latlon_to_xy_km
from .radar_panel import CityPoint, OverlayBundle

log = logging.getLogger(__name__)

# Range-of-interest: don't bother projecting geometries that lie far outside any
# radar's coverage (saves a lot of work at CONUS-scale shapefiles).
DEFAULT_RANGE_KM = 350.0

# Douglas-Peucker simplification tolerances (km, post-projection). Picked
# so the simplification is sub-pixel at typical pan/zoom levels but cuts
# ring vertex counts to a fraction of the raw shapefile. State borders
# can lose more detail than counties because they're drawn thicker.
#
# At our deepest zoom-in (~20 km half-width on a 1024-px image →
# ~40 m/pixel), 0.2 km of simplification == 5 px of detail loss. That's
# visible only when you zoom hard *and* look right at a coast or river;
# for a synoptic scrub-through it's invisible. The win on the paint
# side is real: state-border total vertex count drops ~5-10× and county
# count drops ~3-5×, with corresponding drops in QPainter ``drawPath``
# time per frame.
_SIMPLIFY_TOL_STATE_KM = 0.2
_SIMPLIFY_TOL_COUNTY_KM = 0.1


def _simplify_ring_km(arr: np.ndarray, tolerance_km: float) -> np.ndarray:
    """Douglas-Peucker simplify a projected ``(N, 2)`` ring in km.

    Short rings (≤4 vertices) are returned unchanged — the algorithm
    can't meaningfully reduce them and the LineString round-trip would
    cost more than it saves. ``preserve_topology=False`` is fine: these
    rings are rendered as standalone lines, not as topology-bearing
    administrative areas, so non-simple intersections at the
    simplified-vertex level are visually irrelevant."""
    if len(arr) <= 4:
        return arr
    line = LineString(arr)
    simp = line.simplify(tolerance_km, preserve_topology=False)
    out = np.asarray(simp.coords, dtype=np.float64)
    # simplify() can collapse a tiny ring to <2 points; fall back.
    if out.shape[0] < 2:
        return arr
    return out


_COUNTIES_SHP = (
    Path(__file__).resolve().parent.parent.parent
    / "resources" / "counties" / "cb_2023_us_county_500k.shp"
)

_US_CITIES_CSV = (
    Path(__file__).resolve().parent.parent.parent
    / "resources" / "cities" / "us_cities.csv"
)


@lru_cache(maxsize=4)
def _load_state_geometries():
    """Load (and cache) all admin_1 polygons from Natural Earth."""
    path = shpreader.natural_earth(resolution="10m", category="cultural",
                                   name="admin_1_states_provinces_lines")
    return list(shpreader.Reader(path).geometries())


@lru_cache(maxsize=4)
def _load_populated_places():
    """Load Natural Earth populated places with population attribute."""
    path = shpreader.natural_earth(resolution="10m", category="cultural",
                                   name="populated_places")
    return list(shpreader.Reader(path).records())


@lru_cache(maxsize=1)
def _load_us_cities() -> list[tuple[str, str, float, float, int, str]]:
    """Load the bundled GeoNames-derived US cities CSV.

    Returns a list of ``(name, state, lat, lon, pop, feature_code)``
    tuples. ``feature_code`` is the GeoNames code — ``PPLA2`` flags
    county seats, ``PPLA`` flags state capitals — preserved so the
    renderer could prioritize them in the future, though right now
    every entry just lands in the same population-sorted candidate
    pool.

    Rebuild the CSV with ``scripts/build_us_cities.py`` to refresh
    against the latest GeoNames cities500 dump.
    """
    if not _US_CITIES_CSV.exists():
        log.warning(
            "US cities CSV not found at %s — run scripts/build_us_cities.py "
            "to populate; Natural Earth populated_places will be the only "
            "city source until then",
            _US_CITIES_CSV,
        )
        return []
    out: list[tuple[str, str, float, float, int, str]] = []
    with _US_CITIES_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row["lat"])
                lon = float(row["lon"])
                pop = int(row["pop"] or 0)
            except (KeyError, TypeError, ValueError):
                continue
            out.append((
                row.get("name", "?"), row.get("state", ""),
                lat, lon, pop, row.get("feature", ""),
            ))
    return out


@lru_cache(maxsize=4)
def _load_county_geometries():
    """Load (and cache) US county polygons from the bundled TIGER shapefile."""
    if not _COUNTIES_SHP.exists():
        log.warning("County shapefile not found at %s — county overlays disabled",
                    _COUNTIES_SHP)
        return []
    return list(shpreader.Reader(str(_COUNTIES_SHP)).geometries())


def _project_ring(coords, site: Site) -> np.ndarray:
    """Project a list of (lon, lat) coords to (x_km, y_km) around ``site``."""
    out = np.empty((len(coords), 2), dtype=np.float64)
    for i, (lon, lat) in enumerate(coords):
        out[i, 0], out[i, 1] = latlon_to_xy_km(lat, lon, site.lat, site.lon)
    return out


def _coords_of(geom):
    """Yield each ring's coords from a Shapely geometry (handles Multi*).

    Returns sequences of ``(lon, lat)`` tuples.
    """
    geom_type = geom.geom_type
    if geom_type == "LineString":
        yield list(geom.coords)
    elif geom_type == "MultiLineString":
        for line in geom.geoms:
            yield list(line.coords)
    elif geom_type == "Polygon":
        yield list(geom.exterior.coords)
        for hole in geom.interiors:
            yield list(hole.coords)
    elif geom_type == "MultiPolygon":
        for poly in geom.geoms:
            yield list(poly.exterior.coords)
            for hole in poly.interiors:
                yield list(hole.coords)


@lru_cache(maxsize=4)
def _load_country_border_geometries():
    """Load country admin_0 border lines from Natural Earth."""
    path = shpreader.natural_earth(resolution="50m", category="cultural",
                                   name="admin_0_boundary_lines_land")
    return list(shpreader.Reader(path).geometries())


@lru_cache(maxsize=4)
def _load_coastline_geometries():
    """Load global coastline lines from Natural Earth."""
    path = shpreader.natural_earth(resolution="50m", category="physical",
                                   name="coastline")
    return list(shpreader.Reader(path).geometries())


def _rings_in_bbox_latlon(
    geoms,
    *,
    minx: float, maxx: float, miny: float, maxy: float,
) -> list[np.ndarray]:
    """Return each line/polygon ring as an ``(N, 2)`` ``(lon, lat)`` array,
    filtered to those that touch the bbox. Used by map widgets that draw
    in plain lat/lon (no cartopy projection at draw time)."""
    out: list[np.ndarray] = []
    for geom in geoms:
        gxmin, gymin, gxmax, gymax = geom.bounds
        if gxmax < minx or gxmin > maxx or gymax < miny or gymin > maxy:
            continue
        for coords in _coords_of(geom):
            if len(coords) < 2:
                continue
            arr = np.asarray(coords, dtype=np.float64)
            out.append(arr)
    return out


@lru_cache(maxsize=2)
def load_conus_lines_latlon() -> dict:
    """Return state / country-border / coastline lines (CONUS bbox) as plain
    lon/lat numpy arrays. Used by the pyqtgraph-based maps (host map, day
    picker) — they don't need cartopy's projection machinery; lat/lon as
    flat (x, y) is good enough at CONUS scale."""
    minx, maxx, miny, maxy = -130.0, -60.0, 20.0, 55.0
    states: list[np.ndarray] = []
    try:
        states = _rings_in_bbox_latlon(
            _load_state_geometries(),
            minx=minx, maxx=maxx, miny=miny, maxy=maxy,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load state lines (lat/lon): %s", e)

    borders: list[np.ndarray] = []
    try:
        borders = _rings_in_bbox_latlon(
            _load_country_border_geometries(),
            minx=minx, maxx=maxx, miny=miny, maxy=maxy,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load borders (lat/lon): %s", e)

    coastlines: list[np.ndarray] = []
    try:
        coastlines = _rings_in_bbox_latlon(
            _load_coastline_geometries(),
            minx=minx, maxx=maxx, miny=miny, maxy=maxy,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load coastline (lat/lon): %s", e)

    return {
        "states": states,
        "borders": borders,
        "coastlines": coastlines,
    }


def build_overlays(site: Site, *, range_km: float = DEFAULT_RANGE_KM) -> OverlayBundle:
    """Build an :class:`OverlayBundle` centered on ``site``, limited to ``range_km``.

    Filters geometries by a generous bounding box around the radar so we don't
    project the entire continent. The lat/lon box is approximate; we trim more
    precisely in km after projection.
    """
    # ~1 deg lat = 111 km; 1 deg lon ≈ 111 * cos(lat) km
    dlat = range_km / 111.0
    dlon = range_km / (111.0 * max(0.1, np.cos(np.deg2rad(site.lat))))
    minx, maxx = site.lon - dlon, site.lon + dlon
    miny, maxy = site.lat - dlat, site.lat + dlat

    state_rings: list[np.ndarray] = []
    try:
        for geom in _load_state_geometries():
            gxmin, gymin, gxmax, gymax = geom.bounds
            if gxmax < minx or gxmin > maxx or gymax < miny or gymin > maxy:
                continue
            for coords in _coords_of(geom):
                ring = _project_ring(coords, site)
                # Trim out points beyond ~range_km*1.5 from radar (cleaner)
                dist = np.hypot(ring[:, 0], ring[:, 1])
                if (dist < range_km * 1.5).any():
                    state_rings.append(
                        _simplify_ring_km(ring, _SIMPLIFY_TOL_STATE_KM)
                    )
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load state borders: %s", e)

    cities: list[CityPoint] = []
    # Track (name_lower, rounded lat/lon) to suppress GeoNames/Natural-Earth
    # duplicates — both sources include big metros so without dedup a label
    # like "Dallas" would draw twice with identical text. Lat/lon are rounded
    # to ~1 km so a tiny coordinate disagreement doesn't defeat the match.
    seen: set[tuple[str, int, int]] = set()

    def _dedup_key(name: str, lat: float, lon: float) -> tuple[str, int, int]:
        return (name.casefold(), round(lat * 100), round(lon * 100))

    try:
        for record in _load_populated_places():
            attrs = record.attributes
            try:
                pop = int(attrs.get("POP_MAX") or 0)
            except (TypeError, ValueError):
                pop = 0
            if pop < 200:
                continue
            geom = record.geometry
            lon, lat = float(geom.x), float(geom.y)
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
            name = str(attrs.get("NAME") or attrs.get("NAMEASCII") or "?")
            cities.append(CityPoint(name=name, lat=lat, lon=lon, pop=pop))
            seen.add(_dedup_key(name, lat, lon))
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load populated places: %s", e)

    # Layer in the GeoNames US cities — sparse rural CONUS doesn't get
    # decent coverage from Natural Earth alone, so this fills in every
    # county seat + every place ≥500 pop. Outside the US this is a no-op
    # (the bbox filter rejects everything).
    try:
        for name, _state, lat, lon, pop, _feature in _load_us_cities():
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
            key = _dedup_key(name, lat, lon)
            if key in seen:
                continue
            seen.add(key)
            cities.append(CityPoint(name=name, lat=lat, lon=lon, pop=pop))
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load US cities CSV: %s", e)

    county_rings: list[np.ndarray] = []
    try:
        for geom in _load_county_geometries():
            gxmin, gymin, gxmax, gymax = geom.bounds
            if gxmax < minx or gxmin > maxx or gymax < miny or gymin > maxy:
                continue
            for coords in _coords_of(geom):
                ring = _project_ring(coords, site)
                dist = np.hypot(ring[:, 0], ring[:, 1])
                if (dist < range_km * 1.5).any():
                    county_rings.append(
                        _simplify_ring_km(ring, _SIMPLIFY_TOL_COUNTY_KM)
                    )
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load county borders: %s", e)

    return OverlayBundle(
        range_rings_km=[50.0, 100.0, 150.0, 200.0],
        state_borders_xy=state_rings,
        county_borders_xy=county_rings,
        cities=cities,
    )
