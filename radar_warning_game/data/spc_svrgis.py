"""SPC Severe Weather Database (SVRGIS) tornado backfill.

After every tornado, the local NWS office conducts a damage survey
that finalizes the EF rating, path length / width, and casualty
counts. Those post-survey numbers land in SPC's annual SVRGIS dataset
— a single CSV covering **1950 through the last completed publication
year** (~70 k records, ~7.5 MB total). Substantially more reliable
than the IEM LSR ``magnitude`` field, which is populated inconsistently
during real-time NWS issuance: many tornado LSRs carry no EF at all
("preliminary radar-indicated"), or a placeholder, and the casualty
counts are often parsed out of free-text remarks.

We use SVRGIS to backfill the magnitude / casualty fields on every
IEM tornado report whose start time is older than the publication-lag
window (default 180 days). For newer events, SVRGIS hasn't been
published yet — fall back to IEM + the existing SPC daily-filtered
overlay in :mod:`.reports`.

Matching is approximate: SVRGIS records ONE start point per tornado
track while IEM may emit several LSRs per track (touchdown + path
waypoints). All IEM LSRs along the path link to the same SVRGIS row.
Tolerance defaults to **50 km** of the start point and **60 min** of
the start time — wide enough to catch path-waypoint LSRs and clock-
skew on older records but tight enough to avoid mixing up clustered
events.

Schema reference (column order on the CSV's header line):
  ``om, yr, mo, dy, date, time, tz, st, stf, stn, mag, inj, fat,
  loss, closs, slat, slon, elat, elon, len, wid, ns, sn, sg,
  f1, f2, f3, f4, fc``

Key columns:
  - ``mag``: F-scale (1950-2006) or EF (2007+). ``-9`` = unrated.
  - ``inj`` / ``fat``: post-survey injury / fatality counts.
  - ``slat`` / ``slon``: tornado **start** lat/lon (decimal degrees).
  - ``tz``: timezone code. ``3`` = CST (fixed UTC-6, no DST per SPC
    convention), ``9`` = GMT/UTC. Other codes (0, 6) appear in older
    rows and we treat them as best-effort UTC; the time tolerance
    swallows the resulting clock-skew.
"""

from __future__ import annotations

import hashlib
import logging
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger(__name__)

# Defaults — both are constants so callers / tests can override.
SVRGIS_URL = "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
SVRGIS_PUBLICATION_LAG = timedelta(days=180)
"""SVRGIS publishes annually with a ~6-month damage-survey lag. Events
younger than this won't be in the file — fall back to live sources."""

# Cache location mirrors reports.py's hashed-filename pattern so the
# on-disk file doesn't leak any human-readable context.
_CACHE_ROOT = Path.home() / ".radaranalysiscorp" / "cache" / "reports"
_CACHE_KEY = "svrgis_tornadoes_1950_2023"


def _hashed_cache_path() -> Path:
    _CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha1(_CACHE_KEY.encode("utf-8")).hexdigest()
    return _CACHE_ROOT / f"{h}.csv"


_LOCK = threading.Lock()
_DF: pd.DataFrame | None = None


def _to_utc_naive(row) -> "pd.Timestamp":
    """Combine SVRGIS ``date + time + tz`` into a naive UTC pandas Timestamp.

    Returns ``pd.NaT`` if the date / time can't be parsed (very rare —
    SVRGIS is clean — but a corrupt row shouldn't poison the whole load).
    """
    try:
        dt_local = pd.Timestamp(f"{row['date']} {row['time']}")
    except (ValueError, TypeError):
        return pd.NaT
    try:
        tz = int(row["tz"]) if pd.notna(row["tz"]) else 0
    except (ValueError, TypeError):
        tz = 0
    # SPC convention: tz=3 = CST fixed UTC-6 (no DST), tz=9 = already
    # UTC. Other codes (0, 6) are inconsistent across old records;
    # treat as UTC and rely on the matcher's time tolerance to absorb
    # the skew.
    if tz == 3:
        return dt_local + pd.Timedelta(hours=6)
    return dt_local


def load_svrgis() -> pd.DataFrame:
    """Download (cached) + parse the SVRGIS tornado CSV. Idempotent;
    returns the same in-memory DataFrame across calls within a process."""
    global _DF
    with _LOCK:
        if _DF is not None:
            return _DF
        cache = _hashed_cache_path()
        if not cache.exists():
            log.info("Downloading SVRGIS tornadoes (~8 MB) from %s", SVRGIS_URL)
            r = requests.get(SVRGIS_URL, timeout=120)
            r.raise_for_status()
            cache.write_bytes(r.content)
        df = pd.read_csv(cache)
        # Pre-compute the UTC timestamp once so per-lookup matching is
        # a simple time-delta comparison.
        df["utc_dt"] = df.apply(_to_utc_naive, axis=1)
        # Drop rows we can't time-stamp (rare; usually pre-1950
        # placeholder rows aren't present, but be safe).
        df = df.dropna(subset=["utc_dt"]).reset_index(drop=True)
        log.info("SVRGIS loaded: %d tornado records from %d-%d",
                 len(df), int(df["yr"].min()), int(df["yr"].max()))
        _DF = df
        return _DF


def reset_cache() -> None:
    """Forget the in-memory DataFrame. Mostly for tests; the on-disk
    cache survives so the next call doesn't re-download."""
    global _DF
    with _LOCK:
        _DF = None


# Match tolerances. The defaults are deliberately loose because we
# want every IEM LSR along a tornado track to bind to the same
# SVRGIS row (touchdown + path-waypoint LSRs can be tens of km apart).
DEFAULT_DIST_TOL_KM = 50.0
DEFAULT_TIME_TOL_MIN = 60.0


def find_tornado_record(
    when_utc: datetime,
    lat: float,
    lon: float,
    *,
    dist_tol_km: float = DEFAULT_DIST_TOL_KM,
    time_tol_min: float = DEFAULT_TIME_TOL_MIN,
) -> "pd.Series | None":
    """Look up the SVRGIS row matching the given IEM tornado report.

    Returns ``None`` if no row is within ``dist_tol_km`` of the start
    point AND ``time_tol_min`` of the start time. If multiple rows
    qualify, the spatially-nearest wins (ties broken by smallest time
    delta — guarantees a deterministic match even for closely-clustered
    events)."""
    try:
        df = load_svrgis()
    except Exception:
        log.exception("SVRGIS load failed — returning no match")
        return None
    if df.empty:
        return None
    if when_utc.tzinfo is not None:
        when = when_utc.astimezone(timezone.utc).replace(tzinfo=None)
    else:
        when = when_utc
    when_ts = pd.Timestamp(when)
    # Narrow to a ±1 day band first (cheap filter on a sorted-ish
    # column). The time tolerance defaults to 60 min so ±1 day is
    # plenty of headroom for timezone-edge cases.
    day_window = pd.Timedelta(days=1)
    band = df[
        (df["utc_dt"] >= when_ts - day_window)
        & (df["utc_dt"] <= when_ts + day_window)
    ]
    if band.empty:
        return None
    dt_min = (band["utc_dt"] - when_ts).abs() / pd.Timedelta(minutes=1)
    band = band[dt_min <= time_tol_min]
    if band.empty:
        return None
    # Approximate km using equirectangular projection at the report's
    # latitude — good to better than 1% over the 50 km tolerance, no
    # need to bring in haversine for this many rows.
    DEG_LAT_KM = 111.0
    DEG_LON_KM = 111.0 * np.cos(np.radians(lat))
    dlat = (band["slat"] - lat) * DEG_LAT_KM
    dlon = (band["slon"] - lon) * DEG_LON_KM
    dist_km = np.sqrt(dlat * dlat + dlon * dlon)
    within = band[dist_km <= dist_tol_km]
    if within.empty:
        return None
    # Nearest in distance, then nearest in time as tiebreaker.
    candidates = within.copy()
    candidates["_dist_km"] = dist_km[within.index]
    candidates["_time_min"] = (
        candidates["utc_dt"] - when_ts
    ).abs() / pd.Timedelta(minutes=1)
    candidates = candidates.sort_values(["_dist_km", "_time_min"])
    return candidates.iloc[0]


def has_svrgis_coverage(when_utc: datetime, *, today: datetime | None = None) -> bool:
    """Whether the event is old enough that SVRGIS should have it.
    Younger events fall into the publication-lag window and we
    should not bother trying to look them up."""
    if when_utc.tzinfo is None:
        when_utc = when_utc.replace(tzinfo=timezone.utc)
    today = today or datetime.now(timezone.utc)
    return today - when_utc >= SVRGIS_PUBLICATION_LAG


def magnitude_from_svrgis(row: "pd.Series") -> float:
    """Pull the EF magnitude from a SVRGIS row, normalizing ``-9``
    (unrated) to ``-1`` so callers see the same sentinel they get
    from preliminary IEM data."""
    try:
        mag = float(row.get("mag", -1))
    except (TypeError, ValueError):
        return -1.0
    if mag < 0:    # SVRGIS uses -9; reports.py uses -1
        return -1.0
    return mag


def casualties_from_svrgis(row: "pd.Series") -> tuple[int, int]:
    """``(injuries, fatalities)`` from a SVRGIS row. Returns ``(0, 0)``
    on any parse error so callers can safely substitute these in."""
    def _int(v):
        try:
            n = int(v)
            return max(0, n)
        except (TypeError, ValueError):
            return 0
    return _int(row.get("inj", 0)), _int(row.get("fat", 0))


def _convective_day_12z(utc_dt: "pd.Timestamp") -> "pd.Timestamp":
    """Project a UTC tornado time onto the 12Z–12Z convective day it
    belongs to. A tornado at 03:14 UTC on 2013-05-21 is part of the
    2013-05-20 convective day; one at 19:56 UTC on 2013-05-20 is also
    on 2013-05-20."""
    shifted = utc_dt - pd.Timedelta(hours=12)
    return pd.Timestamp(shifted.year, shifted.month, shifted.day, 12)


def convective_days_with_min_ef(
    min_ef: float,
    *,
    range_start: datetime | None = None,
    range_end: datetime | None = None,
) -> list[datetime]:
    """Return the sorted list of UTC 12Z timestamps for every convective
    day in ``[range_start, range_end]`` that contains at least one
    SVRGIS tornado with ``mag >= min_ef``.

    This is the SVRGIS-driven equivalent of "sample random dates from
    2000-today until one has an EF4+ tornado." For ``min_ef=4`` there
    are roughly 50 such days over the full archive, so the random-
    pick becomes one O(1) selection from a 50-entry list instead of
    hundreds of HTTP round-trips against IEM hoping to land on a
    qualifying day.

    Returns an empty list if SVRGIS is unavailable (caller should
    fall back to the date-range random-sample approach)."""
    try:
        df = load_svrgis()
    except Exception:
        log.exception("SVRGIS unavailable — no candidate days returned")
        return []
    qualifying = df[df["mag"] >= min_ef]
    if qualifying.empty:
        return []
    # Project each tornado's UTC time onto its convective 12Z day.
    days = qualifying["utc_dt"].apply(_convective_day_12z)
    # Apply optional date-window filter (in NAIVE-UTC space because
    # ``utc_dt`` itself is naive — see load_svrgis).
    if range_start is not None:
        rs = range_start.replace(tzinfo=None) if range_start.tzinfo else range_start
        days = days[days >= pd.Timestamp(rs)]
    if range_end is not None:
        re_ = range_end.replace(tzinfo=None) if range_end.tzinfo else range_end
        days = days[days <= pd.Timestamp(re_)]
    if days.empty:
        return []
    # Deduplicate (one tornado per day is enough to qualify the day)
    # and convert back to aware UTC datetimes for caller ergonomics.
    unique_naive = sorted(set(days.tolist()))
    return [
        d.to_pydatetime().replace(tzinfo=timezone.utc)
        for d in unique_naive
    ]
