"""Storm-report fetching.

Primary source: IEM Local Storm Reports (LSRs) — `mesonet.agron.iastate.edu`.
For events older than ``SPC_BACKFILL_THRESHOLD_DAYS`` we additionally fetch
SPC's QC'd storm database to backfill confirmed tornado EF ratings and
injury/fatality counts that may be absent or preliminary in the LSRs.

LSR free-text remarks sometimes include casualty counts (e.g. ``"...2 INJ, 1 FAT..."``).
:func:`parse_casualties` extracts those when present; SPC backfill is preferred
when available.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from .cache import DEFAULT_CACHE_ROOT

IEM_LSR_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py"
SPC_REPORTS_URL = "https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts_filtered_torn.csv"
SPC_BACKFILL_THRESHOLD_DAYS = 30

REPORTS_CACHE = DEFAULT_CACHE_ROOT / "reports"
REPORTS_CACHE.mkdir(parents=True, exist_ok=True)


def _hashed_path(key: str, suffix: str = ".csv") -> Path:
    """Hash the cache key so the on-disk filename doesn't leak the date."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return REPORTS_CACHE / f"{h}{suffix}"

# IEM TYPECODE → category
TORNADO_CODES = frozenset({"T"})
HAIL_CODES = frozenset({"H"})
WIND_CODES = frozenset({"G", "W", "N", "D"})

_INJ_RE = re.compile(r"(\d+)\s*INJ", re.IGNORECASE)
_FAT_RE = re.compile(r"(\d+)\s*FAT", re.IGNORECASE)


@dataclass(frozen=True)
class Report:
    """A normalized storm report (regardless of IEM/SPC origin)."""

    time: datetime           # UTC
    lat: float
    lon: float
    category: str            # 'tornado' | 'hail' | 'wind'
    magnitude: float         # hail in inches; wind in mph; tornado as EF integer or -1 if unknown
    state: str
    county: str
    remark: str
    injuries: int            # 0 if unknown / not parsed
    fatalities: int          # 0 if unknown / not parsed
    source: str              # 'IEM' | 'SPC'


def _categorize(typecode: str) -> str | None:
    if typecode in TORNADO_CODES:
        return "tornado"
    if typecode in HAIL_CODES:
        return "hail"
    if typecode in WIND_CODES:
        return "wind"
    return None


def parse_casualties(remark: str) -> tuple[int, int]:
    """Extract (injuries, fatalities) from an LSR remark string. Returns (0, 0) if absent."""
    if not isinstance(remark, str):
        return 0, 0
    inj = _INJ_RE.search(remark)
    fat = _FAT_RE.search(remark)
    return (int(inj.group(1)) if inj else 0, int(fat.group(1)) if fat else 0)


def _iem_lsr_cache_path(day_utc: datetime) -> Path:
    return _hashed_path(f"iem_lsr_{day_utc:%Y%m%d}")


def fetch_iem_lsr_day(day: datetime) -> pd.DataFrame:
    """Fetch (and cache) one UTC calendar day of raw IEM LSR rows."""
    if day.tzinfo is None:
        day = day.replace(tzinfo=timezone.utc)
    day_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    cache = _iem_lsr_cache_path(day_utc)
    if not cache.exists():
        params = {
            "sts": day_utc.strftime("%Y-%m-%dT%H:%MZ"),
            "ets": (day_utc + timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ"),
            "fmt": "csv",
        }
        r = requests.get(IEM_LSR_URL, params=params, timeout=60)
        r.raise_for_status()
        cache.write_bytes(r.content)
    try:
        df = pd.read_csv(cache, dtype={"VALID": str})
    except pd.errors.ParserError:
        df = pd.read_csv(cache, dtype={"VALID": str}, engine="python", on_bad_lines="skip")
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["VALID"], format="%Y%m%d%H%M", utc=True)
    df["MAG"] = pd.to_numeric(df["MAG"], errors="coerce").fillna(0.0)
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df = df.dropna(subset=["LAT", "LON", "time"]).reset_index(drop=True)
    return df


def _iem_row_to_report(row: pd.Series) -> Report | None:
    cat = _categorize(str(row.get("TYPECODE", "")))
    if cat is None:
        return None
    remark = str(row.get("REMARK", "") or "")
    inj, fat = parse_casualties(remark)
    # Tornado magnitude in IEM LSRs uses EF rating in MAG when available; many
    # preliminary rows leave it as 0. We pass -1 to signal "unknown EF" so the
    # SPC backfill can fill it in for older events.
    mag = float(row.get("MAG", 0.0))
    if cat == "tornado" and mag == 0.0:
        mag = -1.0
    return Report(
        time=row["time"].to_pydatetime(),
        lat=float(row["LAT"]),
        lon=float(row["LON"]),
        category=cat,
        magnitude=mag,
        state=str(row.get("ST", "")),
        county=str(row.get("COUNTY", "") or ""),
        remark=remark,
        injuries=inj,
        fatalities=fat,
        source="IEM",
    )


def fetch_iem_window(start: datetime, end: datetime) -> list[Report]:
    """Fetch all IEM LSRs in ``[start, end]`` (UTC), spanning day caches."""
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    out: list[Report] = []
    d = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    while d <= end:
        try:
            df = fetch_iem_lsr_day(d)
        except Exception as e:  # noqa: BLE001
            print(f"[warn] IEM LSR fetch failed for {d:%Y-%m-%d}: {e}")
            d += timedelta(days=1)
            continue
        for _, row in df.iterrows():
            t: datetime = row["time"].to_pydatetime()
            if t < start or t > end:
                continue
            rep = _iem_row_to_report(row)
            if rep is not None:
                out.append(rep)
        d += timedelta(days=1)
    return out


# --------------------------- SPC backfill (>30 days) --------------------------

def _spc_cache_path(day_utc: datetime) -> Path:
    return _hashed_path(f"spc_torn_{day_utc:%Y%m%d}")


def fetch_spc_tornadoes_day(day: datetime) -> pd.DataFrame:
    """Fetch (and cache) one UTC day of SPC filtered tornado reports.

    SPC publishes daily CSVs with confirmed EF ratings + casualty counts as
    structured fields. Only available for past dates (typically a day or two
    after the event for filtered, longer for the final QC'd Storm Data DB).
    """
    if day.tzinfo is None:
        day = day.replace(tzinfo=timezone.utc)
    day_utc = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    cache = _spc_cache_path(day_utc)
    if not cache.exists():
        url = SPC_REPORTS_URL.format(yymmdd=day_utc.strftime("%y%m%d"))
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        cache.write_bytes(r.content)
    try:
        df = pd.read_csv(cache)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    return df


def _should_backfill_with_spc(event_day: datetime, today: datetime | None = None) -> bool:
    today = today or datetime.now(timezone.utc)
    age = today - event_day
    return age >= timedelta(days=SPC_BACKFILL_THRESHOLD_DAYS)


def fetch_reports(
    start: datetime,
    end: datetime,
    *,
    use_spc_backfill: bool = True,
    today: datetime | None = None,
) -> list[Report]:
    """Top-level entry: returns the normalized report list for ``[start, end]`` (UTC).

    When ``use_spc_backfill`` is True and the event predates today by more than
    ``SPC_BACKFILL_THRESHOLD_DAYS``, SPC tornado data overlays IEM rows where it
    has better EF / casualty info. Non-tornado reports always come from IEM.
    """
    iem = fetch_iem_window(start, end)
    if not use_spc_backfill or not _should_backfill_with_spc(start, today):
        return iem
    # Backfill tornado EF / casualties from SPC where row matches by approx (time, lat, lon).
    try:
        spc_df = fetch_spc_tornadoes_day(start)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SPC backfill fetch failed for {start:%Y-%m-%d}: {e}")
        return iem
    if spc_df.empty:
        return iem

    out = []
    for r in iem:
        if r.category != "tornado":
            out.append(r)
            continue
        match = _spc_match(r, spc_df)
        if match is None:
            out.append(r)
            continue
        out.append(_merge_spc_into_iem(r, match))
    return out


def _spc_match(report: Report, spc_df: pd.DataFrame) -> pd.Series | None:
    """Find the SPC row best matching an IEM tornado report by location.

    SPC daily CSVs are multi-section (tornado / wind / hail sections concatenated
    with repeating header rows), and column dtypes inferred by pandas can drift to
    strings. We coerce Lat/Lon to numeric, drop rows that can't be parsed, and
    silently return ``None`` rather than raising — preliminary IEM data is then used.
    """
    if "Lat" not in spc_df.columns or "Lon" not in spc_df.columns:
        return None
    try:
        df = spc_df.copy()
        df["Lat"] = pd.to_numeric(df["Lat"], errors="coerce")
        df["Lon"] = pd.to_numeric(df["Lon"], errors="coerce")
        df = df.dropna(subset=["Lat", "Lon"])
        if df.empty:
            return None
        df["_dist"] = (df["Lat"] - report.lat) ** 2 + (df["Lon"] - report.lon) ** 2
        candidate = df.sort_values("_dist").iloc[0]
        if candidate["_dist"] > 0.09:  # ~33 km — preliminary LSR coords drift this much
            return None
        return candidate
    except (KeyError, ValueError, TypeError):
        return None


def _merge_spc_into_iem(iem: Report, spc_row: pd.Series) -> Report:
    """Overlay SPC EF / casualty fields onto an IEM tornado report."""
    ef = iem.magnitude
    try:
        if "F_Scale" in spc_row.index:
            ef = float(spc_row["F_Scale"])
        elif "EF" in spc_row.index:
            ef = float(spc_row["EF"])
    except (TypeError, ValueError):
        pass
    inj = iem.injuries
    fat = iem.fatalities
    try:
        if "Injured" in spc_row.index:
            inj = max(inj, int(spc_row["Injured"]))
        if "Killed" in spc_row.index:
            fat = max(fat, int(spc_row["Killed"]))
    except (TypeError, ValueError):
        pass
    return Report(
        time=iem.time,
        lat=iem.lat,
        lon=iem.lon,
        category=iem.category,
        magnitude=ef,
        state=iem.state,
        county=iem.county,
        remark=iem.remark,
        injuries=inj,
        fatalities=fat,
        source="SPC",
    )


def filter_severe(
    reports: list[Report],
    *,
    min_hail_in: float = 1.0,
    min_wind_mph: float = 58.0,
) -> list[Report]:
    """Drop reports that don't meet severe criteria. Tornadoes always pass."""
    out: list[Report] = []
    for r in reports:
        if r.category == "tornado":
            out.append(r)
        elif r.category == "hail" and r.magnitude >= min_hail_in:
            out.append(r)
        elif r.category == "wind" and r.magnitude >= min_wind_mph:
            out.append(r)
    return out


def count_by_category(reports: list[Report]) -> dict[str, int]:
    counts = {"tornado": 0, "hail": 0, "wind": 0}
    for r in reports:
        if r.category in counts:
            counts[r.category] += 1
    return counts
