"""Lightweight lat/lon ↔ local-tangent-plane projections.

For game purposes we never need true map projections — we need:

  - **Equirectangular** for plotting around a radar site (Reference-Nowcastle's
    approach; accurate to <1% at radar scales of ~250 km).
  - **Bearing + distance** in km between two (lat, lon) points (great-circle).

Cartopy handles the actual rendering projections; this module is for math.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial bearing from point 1 → point 2, in degrees clockwise from true north.

    Used by the storm-motion measurement tool (§5a of the plan): point 1 is the
    storm's earlier centroid, point 2 the later. The bearing is the storm's
    direction *of* motion (TO). FROM = ``(bearing + 180) mod 360``.
    """
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    x = math.sin(dlon) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlon)
    brg = math.degrees(math.atan2(x, y))
    return (brg + 360.0) % 360.0


def latlon_to_xy_km(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular projection around ``(lat0, lon0)``.

    Returns ``(x_east_km, y_north_km)``. Matches PyART's RadarDisplay non-map
    coordinates closely enough for plotting at radar scales (<250 km).
    """
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = EARTH_RADIUS_KM * math.cos(math.radians(lat0)) * dlon
    y = EARTH_RADIUS_KM * dlat
    return x, y


def xy_km_to_latlon(x_km: float, y_km: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Inverse of :func:`latlon_to_xy_km`."""
    lat = lat0 + math.degrees(y_km / EARTH_RADIUS_KM)
    lon = lon0 + math.degrees(x_km / (EARTH_RADIUS_KM * math.cos(math.radians(lat0))))
    return lat, lon


@dataclass(frozen=True)
class StormMotion:
    """Output of the §5a motion tool."""

    from_deg: float        # NWS convention: direction the storm is coming FROM
    to_deg: float          # direction storm is going TO (P1 → P2 bearing)
    speed_kt: float        # storm speed in knots

    def __str__(self) -> str:
        return f"from {self.from_deg:03.0f}° at {self.speed_kt:.0f} kt"


def storm_motion_from_two_points(
    lat1: float,
    lon1: float,
    t1_sec: float,
    lat2: float,
    lon2: float,
    t2_sec: float,
) -> StormMotion:
    """Compute storm motion from two timed point observations.

    ``t1_sec`` / ``t2_sec`` are seconds since any common epoch (UNIX time works).
    Raises ``ValueError`` if the two points share a timestamp.
    """
    dt = t2_sec - t1_sec
    if dt == 0:
        raise ValueError("Two observations have identical timestamps")
    distance_km = haversine_km(lat1, lon1, lat2, lon2)
    speed_kmh = (distance_km / dt) * 3600.0
    speed_kt = speed_kmh * 0.539957  # km/h → knots
    to_deg = bearing_deg(lat1, lon1, lat2, lon2)
    from_deg = (to_deg + 180.0) % 360.0
    return StormMotion(from_deg=from_deg, to_deg=to_deg, speed_kt=abs(speed_kt))
