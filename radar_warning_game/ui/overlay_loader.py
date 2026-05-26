"""Build :class:`OverlayBundle`s from Cartopy / Natural Earth shapefiles.

Layers populated:
  - **State borders** (Cartopy admin_1 states_provinces, 10m).
  - **Cities** (Cartopy populated_places with population field, 10m).
  - **County borders** — *not yet implemented*. Natural Earth doesn't include
    US counties at usable resolution; the plan calls for the US Census TIGER
    cartographic boundary shapefile bundled with the app. Until that file
    ships in ``resources/``, county overlays remain empty.

All geometry is projected into km east/north of the radar site for direct
drawing on the radar panel's pre-existing km-coordinate axes.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import cartopy.io.shapereader as shpreader
import numpy as np

from ..data.sites import Site
from ..geo.projection import latlon_to_xy_km
from .radar_panel import CityPoint, OverlayBundle

log = logging.getLogger(__name__)

# Range-of-interest: don't bother projecting geometries that lie far outside any
# radar's coverage (saves a lot of work at CONUS-scale shapefiles).
DEFAULT_RANGE_KM = 350.0


_COUNTIES_SHP = (
    Path(__file__).resolve().parent.parent.parent
    / "resources" / "counties" / "cb_2023_us_county_500k.shp"
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
                    state_rings.append(ring)
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load state borders: %s", e)

    cities: list[CityPoint] = []
    try:
        for record in _load_populated_places():
            attrs = record.attributes
            try:
                pop = int(attrs.get("POP_MAX") or 0)
            except (TypeError, ValueError):
                pop = 0
            if pop < 1000:
                continue
            geom = record.geometry
            lon, lat = float(geom.x), float(geom.y)
            if not (minx <= lon <= maxx and miny <= lat <= maxy):
                continue
            name = str(attrs.get("NAME") or attrs.get("NAMEASCII") or "?")
            cities.append(CityPoint(name=name, lat=lat, lon=lon, pop=pop))
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load populated places: %s", e)

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
                    county_rings.append(ring)
    except Exception as e:  # noqa: BLE001
        log.warning("Failed to load county borders: %s", e)

    return OverlayBundle(
        range_rings_km=[50.0, 100.0, 150.0, 200.0],
        state_borders_xy=state_rings,
        county_borders_xy=county_rings,
        cities=cities,
    )
