"""WSR-88D + TDWR site catalog.

Both radar families are loaded from ``resources/RADARS.txt``:
  - ``kind=1`` → WSR-88D (the 160 NWS S-band sites; ICAOs all begin with K
    in CONUS, P in the Pacific, etc.)
  - ``kind=3`` → TDWR (the 45 FAA C-band terminal Doppler radars at major
    airports; ICAOs all begin with T)

TDWR Level 2 archive coverage on the Unidata S3 mirror is patchy in older
years, so a site that's in the catalog isn't automatically usable for a
given date. Day-pickers and the radar-selection map call
:func:`site_has_data_on_day` to probe S3 and grey out sites without
archive coverage on the chosen UTC day.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

log = logging.getLogger(__name__)

_RESOURCES = Path(__file__).resolve().parent.parent.parent / "resources"
RADARS_TXT = _RESOURCES / "RADARS.txt"

CONUS_EXCLUDE_STATES = frozenset({"AK", "HI", "GU", "PR", "KR", "JP"})

# RADARS.txt `kind` codes. The file uses "1" for WSR-88D and "3" for
# TDWR; other codes (research / experimental radars) are filtered out.
SITE_KIND_WSR88D = "WSR88D"
SITE_KIND_TDWR = "TDWR"
_KIND_FROM_FILE = {"1": SITE_KIND_WSR88D, "3": SITE_KIND_TDWR}


@dataclass(frozen=True)
class Site:
    icao: str
    lat: float
    lon: float
    elev_m: float
    state: str
    name: str
    kind: str = SITE_KIND_WSR88D    # "WSR88D" or "TDWR"

    @property
    def is_tdwr(self) -> bool:
        return self.kind == SITE_KIND_TDWR


@lru_cache(maxsize=1)
def load_sites() -> list[Site]:
    """Parse all supported radar sites from ``RADARS.txt`` — WSR-88D
    *and* TDWR. Other radar families (research, experimental) are
    skipped because the rest of the pipeline can't read their data."""
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
            mapped_kind = _KIND_FROM_FILE.get(kind)
            if mapped_kind is None:
                continue
            sites.append(
                Site(
                    icao=icao.upper(),
                    lat=float(lat),
                    lon=float(lon),
                    elev_m=float(elev),
                    state=state,
                    name=name,
                    kind=mapped_kind,
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


def nearest_site(
    lat: float,
    lon: float,
    conus_only: bool = True,
    kinds: frozenset[str] | None = None,
) -> tuple[Site, float]:
    """Nearest radar to ``(lat, lon)``. By default considers all radar
    families (WSR-88D *and* TDWR) — a TDWR can easily be closer than the
    WFO WSR-88D in big metros, and the rest of the pipeline can read both.
    Pass ``kinds={SITE_KIND_WSR88D}`` to restrict to long-range S-band sites."""
    best: tuple[Site, float] | None = None
    for s in load_sites():
        if conus_only and s.state in CONUS_EXCLUDE_STATES:
            continue
        if kinds is not None and s.kind not in kinds:
            continue
        d = haversine_km(lat, lon, s.lat, s.lon)
        if best is None or d < best[1]:
            best = (s, d)
    assert best is not None, "no sites loaded"
    return best


def sites_within_km(
    lat: float,
    lon: float,
    radius_km: float,
    conus_only: bool = True,
    kinds: frozenset[str] | None = None,
) -> list[tuple[Site, float]]:
    """All radars within ``radius_km`` of ``(lat, lon)``, sorted by distance.
    See :func:`nearest_site` for the ``kinds`` filter."""
    out: list[tuple[Site, float]] = []
    for s in load_sites():
        if conus_only and s.state in CONUS_EXCLUDE_STATES:
            continue
        if kinds is not None and s.kind not in kinds:
            continue
        d = haversine_km(lat, lon, s.lat, s.lon)
        if d <= radius_km:
            out.append((s, d))
    out.sort(key=lambda x: x[1])
    return out
