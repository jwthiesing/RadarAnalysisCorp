"""NEXRAD WSR-88D site catalog."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_RESOURCES = Path(__file__).resolve().parent.parent.parent / "resources"
RADARS_TXT = _RESOURCES / "RADARS.txt"

CONUS_EXCLUDE_STATES = frozenset({"AK", "HI", "GU", "PR", "KR", "JP"})


@dataclass(frozen=True)
class Site:
    icao: str
    lat: float
    lon: float
    elev_m: float
    state: str
    name: str


@lru_cache(maxsize=1)
def load_sites() -> list[Site]:
    """Parse the WSR-88D (type 1) sites from RADARS.txt."""
    sites: list[Site] = []
    with open(RADARS_TXT) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue
            icao, _wfo, lat, lon, elev, kind, state, name = parts[:8]
            if kind != "1":
                continue
            sites.append(
                Site(
                    icao=icao.upper(),
                    lat=float(lat),
                    lon=float(lon),
                    elev_m=float(elev),
                    state=state,
                    name=name,
                )
            )
    return sites


def site_by_icao(icao: str) -> Site | None:
    icao = icao.upper()
    for s in load_sites():
        if s.icao == icao:
            return s
    return None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0088
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = p2 - p1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def nearest_site(lat: float, lon: float, conus_only: bool = True) -> tuple[Site, float]:
    best: tuple[Site, float] | None = None
    for s in load_sites():
        if conus_only and s.state in CONUS_EXCLUDE_STATES:
            continue
        d = haversine_km(lat, lon, s.lat, s.lon)
        if best is None or d < best[1]:
            best = (s, d)
    assert best is not None, "no sites loaded"
    return best


def sites_within_km(lat: float, lon: float, radius_km: float, conus_only: bool = True) -> list[tuple[Site, float]]:
    """All radars within radius_km of (lat, lon), sorted by distance."""
    out: list[tuple[Site, float]] = []
    for s in load_sites():
        if conus_only and s.state in CONUS_EXCLUDE_STATES:
            continue
        d = haversine_km(lat, lon, s.lat, s.lon)
        if d <= radius_km:
            out.append((s, d))
    out.sort(key=lambda x: x[1])
    return out
