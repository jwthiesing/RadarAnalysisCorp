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

import numpy as np
import pandas as pd
import requests

from .cache import DEFAULT_CACHE_ROOT

IEM_LSR_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py"
SPC_REPORTS_URL = "https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts_filtered_torn.csv"
SPC_BACKFILL_THRESHOLD_DAYS = 30

REPORTS_CACHE = DEFAULT_CACHE_ROOT / "reports"
REPORTS_CACHE.mkdir(parents=True, exist_ok=True)

# Lightweight per-day counts index: maps "YYYY-MM-DD" → {tornado: N, hail: N, wind: N}.
# Populated lazily as days are fetched. Lets pick_random_day skip non-qualifying
# days without paying a fresh HTTP round-trip per candidate.
_DAILY_COUNTS_PATH = REPORTS_CACHE / "daily_counts.json"


def _hashed_path(key: str, suffix: str = ".csv") -> Path:
    """Hash the cache key so the on-disk filename doesn't leak the date."""
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return REPORTS_CACHE / f"{h}{suffix}"


def _load_daily_counts() -> dict[str, dict[str, int]]:
    if not _DAILY_COUNTS_PATH.exists():
        return {}
    import json
    try:
        return json.loads(_DAILY_COUNTS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_daily_counts(counts: dict[str, dict[str, int]]) -> None:
    import json
    try:
        tmp = _DAILY_COUNTS_PATH.with_suffix(".json.part")
        tmp.write_text(json.dumps(counts, separators=(",", ":")))
        tmp.rename(_DAILY_COUNTS_PATH)
    except OSError:
        pass


def get_daily_counts(day_12z: datetime) -> dict | None:
    """Return cached counts for the convective day starting at ``day_12z``, or
    ``None`` if we've never indexed it. The returned dict has at least
    ``tornado / hail / wind`` integer keys, and optionally ``peak_ef``
    (float — the day's strongest confirmed tornado EF, ``-1.0`` if no
    rated tornado, missing entirely for legacy entries that predated
    the EF-threshold feature). Safe to call without network access.
    """
    if day_12z.tzinfo is None:
        day_12z = day_12z.replace(tzinfo=timezone.utc)
    key = day_12z.strftime("%Y-%m-%d")
    return _load_daily_counts().get(key)


def _record_daily_counts(
    day_12z: datetime,
    counts: dict[str, int],
    *,
    peak_ef: float | None = None,
) -> None:
    """Persist a day's category counts (and optionally the peak
    confirmed tornado EF) into the lightweight index. Passing
    ``peak_ef=None`` leaves any existing recorded peak_ef in place —
    that way the IEM-only initial recording doesn't clobber a
    post-SVRGIS update done by the second-pass recorder."""
    if day_12z.tzinfo is None:
        day_12z = day_12z.replace(tzinfo=timezone.utc)
    key = day_12z.strftime("%Y-%m-%d")
    all_counts = _load_daily_counts()
    entry: dict = {k: int(counts.get(k, 0)) for k in ("tornado", "hail", "wind")}
    # Preserve a previously-recorded peak_ef if the current call
    # didn't supply one (e.g. fetch_iem_window can't compute the
    # post-SVRGIS peak; only fetch_reports can).
    existing = all_counts.get(key, {})
    if peak_ef is not None:
        entry["peak_ef"] = float(peak_ef)
    elif "peak_ef" in existing:
        entry["peak_ef"] = existing["peak_ef"]
    all_counts[key] = entry
    _save_daily_counts(all_counts)


def peak_tornado_ef(reports: list["Report"]) -> float:
    """Public alias of :func:`_peak_tornado_ef`. Used by callers that
    need to enrich a fresh ``count_by_category`` dict with the day's
    strongest tornado EF for the ``ThresholdSpec.is_met`` gate."""
    return _peak_tornado_ef(reports)


def _peak_tornado_ef(reports: list["Report"]) -> float:
    """Day's strongest *rated* tornado EF. Returns ``-1.0`` if no
    tornado was rated (either no tornadoes at all or only unrated /
    preliminary records). Used by the random-day picker's
    ``min_strongest_tornado_ef`` threshold."""
    peak = -1.0
    for r in reports:
        if r.category != "tornado":
            continue
        if r.magnitude > peak:
            peak = float(r.magnitude)
    return peak

# IEM TYPECODE → category
TORNADO_CODES = frozenset({"T"})
HAIL_CODES = frozenset({"H"})
WIND_CODES = frozenset({"G", "W", "N", "D"})

# Default wind-speed (mph) for an LSR that lacked a measured magnitude.
# The IEM "D" code is "thunderstorm wind damage", which is a damage-
# observed report — no anemometer reading, only that things broke.
# Such reports come in at MAG=0 (or unset, coerced to 0) which then
# fails the 58 mph severe-wind threshold, so a real damage-confirmed
# SVR with four wind reports in its valid window was getting scored
# as a FA. NWS forecasters routinely treat damage reports as ≥severe
# by definition (you don't get LSR-worthy damage from sub-severe
# wind), so we substitute a value just above the severe threshold.
UNKNOWN_WIND_DEFAULT_MPH = 60.0

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
    elif cat == "wind" and mag == 0.0:
        # Damage-only LSRs (typecode D) often arrive with MAG=0 —
        # see UNKNOWN_WIND_DEFAULT_MPH for the why.
        mag = UNKNOWN_WIND_DEFAULT_MPH
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
    """Fetch all IEM LSRs in ``[start, end]`` (UTC), spanning day caches.

    As a side-effect, records the per-12Z-day category counts into the
    lightweight daily-counts index so the random-day picker can skip
    non-qualifying days without re-fetching them.
    """
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
    # Record counts for the 12Z–12Z convective day starting at ``start`` so the
    # random-day picker can short-circuit future visits to this day.
    if start.hour == 12:
        _record_daily_counts(start, count_by_category(out))
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
    use_svrgis_backfill: bool = True,
    today: datetime | None = None,
) -> list[Report]:
    """Top-level entry: returns the normalized report list for ``[start, end]`` (UTC).

    Tornado reports get supplementary data layered in *in priority order*:

      1. **SVRGIS** (SPC Severe Weather Database) — post-survey EF +
         finalized casualty counts from NWS damage surveys. Annual
         publication with ~6 mo lag, so applied only to events older
         than ``SVRGIS_PUBLICATION_LAG`` (default 180 days). Most
         reliable source we have when the data exists.
      2. **SPC daily filtered CSV** — same-day SPC publication. Still
         preliminary but often carries EF when IEM doesn't. Applied
         to events older than ``SPC_BACKFILL_THRESHOLD_DAYS`` (30 days)
         and not already filled in by SVRGIS.
      3. **IEM LSRs** (the baseline) — real-time NWS issuance, EF
         frequently unset, casualty counts parsed from free-text
         remarks.

    Non-tornado reports always come from IEM. The merge is non-
    destructive: SVRGIS / SPC values overlay IEM where they're more
    informative, but a SVRGIS row marked unrated (``mag = -9``) leaves
    the IEM value alone."""
    iem = fetch_iem_window(start, end)

    def _record_post_backfill_peak(reports: list[Report]) -> None:
        """Re-record the daily-counts entry with the *post-backfill*
        counts AND the peak confirmed tornado EF, so the random-day
        picker's ``min_strongest_tornado_ef`` threshold can short-
        circuit future visits to this day from cache."""
        if start.hour == 12:
            _record_daily_counts(
                start, count_by_category(reports),
                peak_ef=_peak_tornado_ef(reports),
            )

    if not use_spc_backfill or not _should_backfill_with_spc(start, today):
        # No backfill is going to fire, but we can still record the
        # peak EF from the IEM-only data so the EF threshold can
        # short-circuit on this day. For events <30 days old the
        # IEM EF is preliminary / often unset, so the peak will be
        # close to -1.0 — that's fine; the random-day picker's date
        # range starts at 2000, so very-recent days are unusual.
        _record_post_backfill_peak(iem)
        return iem
    # ---- Step 1: SVRGIS backfill for tornadoes (preferred) ----
    iem = _backfill_with_svrgis(
        iem, start=start, end=end, today=today, enabled=use_svrgis_backfill,
    )
    # ---- Step 2: SPC daily-filtered CSV overlay for any remaining
    #              tornado reports that still lack a confirmed EF ----
    try:
        spc_df = fetch_spc_tornadoes_day(start)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SPC backfill fetch failed for {start:%Y-%m-%d}: {e}")
        _record_post_backfill_peak(iem)
        return iem
    if spc_df.empty:
        _record_post_backfill_peak(iem)
        return iem
    out = []
    for r in iem:
        if r.category != "tornado":
            out.append(r)
            continue
        # If SVRGIS already supplied a confirmed EF, don't overwrite
        # with the preliminary SPC daily file.
        if r.source == "SVRGIS" and r.magnitude >= 0:
            out.append(r)
            continue
        match = _spc_match(r, spc_df)
        if match is None:
            out.append(r)
            continue
        out.append(_merge_spc_into_iem(r, match))
    _record_post_backfill_peak(out)
    return out


def _backfill_with_svrgis(
    reports: list[Report],
    *,
    start: datetime,
    end: datetime,
    today: datetime | None,
    enabled: bool,
) -> list[Report]:
    """Two-pass SVRGIS integration for tornadoes:

      1. **Merge** — for every IEM tornado that has a SVRGIS match
         (by time + location tolerance), overlay the post-survey EF
         and casualty counts. Same as before.
      2. **Add SVRGIS-only** — for every SVRGIS tornado in the
         ``[start, end]`` window whose ``om`` (sequential id) wasn't
         consumed by step 1, append a new Report. This catches the
         common case where the post-survey database has tornadoes
         IEM's preliminary LSRs missed entirely — often small EF0s
         in rural areas where no spotters / damage surveys reached
         until after the LSR window closed.

    Skips entirely if disabled, if the event is too recent for SVRGIS
    coverage (publication-lag window), or if the SVRGIS load fails."""
    if not enabled:
        return reports
    from .spc_svrgis import (
        find_tornado_record, has_svrgis_coverage,
        magnitude_from_svrgis, casualties_from_svrgis,
    )
    if not has_svrgis_coverage(start, today=today):
        return reports
    try:
        from .spc_svrgis import load_svrgis
        df = load_svrgis()
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SVRGIS backfill unavailable: {e}")
        return reports

    out: list[Report] = []
    backfilled = 0
    # Track which SVRGIS rows have already been consumed by an IEM
    # match so the SVRGIS-only pass doesn't double-add them. ``om``
    # is the sequential tornado id and is unique per row.
    matched_oms: set[int] = set()

    # ---- pass 1: merge SVRGIS into matched IEM tornadoes ----
    for r in reports:
        if r.category != "tornado":
            out.append(r)
            continue
        row = find_tornado_record(r.time, r.lat, r.lon)
        if row is None:
            out.append(r)
            continue
        try:
            matched_oms.add(int(row["om"]))
        except (KeyError, TypeError, ValueError):
            pass
        mag = magnitude_from_svrgis(row)
        inj, fat = casualties_from_svrgis(row)
        # If SVRGIS has nothing better than IEM for ALL fields, keep
        # the IEM record (avoids relabeling source=SVRGIS on a row
        # where SVRGIS contributed no actual information).
        if mag < 0 and inj == 0 and fat == 0:
            out.append(r)
            continue
        out.append(Report(
            time=r.time, lat=r.lat, lon=r.lon,
            category="tornado",
            magnitude=mag if mag >= 0 else r.magnitude,
            state=r.state, county=r.county,
            remark=r.remark,
            injuries=max(inj, r.injuries),
            fatalities=max(fat, r.fatalities),
            source="SVRGIS",
        ))
        backfilled += 1

    # ---- pass 2: add SVRGIS-only tornadoes in the time window ----
    # Normalize the window to naive UTC for comparison with df['utc_dt']
    # (which is naive-UTC as built by ``spc_svrgis.load_svrgis``).
    if start.tzinfo is None:
        start_utc = start.replace(tzinfo=timezone.utc)
    else:
        start_utc = start.astimezone(timezone.utc)
    if end.tzinfo is None:
        end_utc = end.replace(tzinfo=timezone.utc)
    else:
        end_utc = end.astimezone(timezone.utc)
    start_ts = pd.Timestamp(start_utc.replace(tzinfo=None))
    end_ts = pd.Timestamp(end_utc.replace(tzinfo=None))
    in_window = df[(df["utc_dt"] >= start_ts) & (df["utc_dt"] <= end_ts)]
    added = 0
    for _, row in in_window.iterrows():
        try:
            om = int(row["om"])
        except (TypeError, ValueError):
            continue
        if om in matched_oms:
            continue
        try:
            lat = float(row["slat"])
            lon = float(row["slon"])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        # Reconstruct the UTC datetime from the precomputed naive
        # Timestamp + UTC tz tag for the Report's ``time`` field.
        utc_dt = row["utc_dt"]
        if pd.isna(utc_dt):
            continue
        report_time = utc_dt.to_pydatetime().replace(tzinfo=timezone.utc)
        mag = magnitude_from_svrgis(row)
        inj, fat = casualties_from_svrgis(row)
        state = str(row.get("st") or "").strip()
        out.append(Report(
            time=report_time,
            lat=lat, lon=lon,
            category="tornado",
            magnitude=mag,   # may be -1 (unrated) — that's still useful presence info
            state=state,
            county="",
            remark="SVRGIS-only (no IEM LSR)",
            injuries=inj,
            fatalities=fat,
            source="SVRGIS",
        ))
        added += 1

    if backfilled or added:
        msg_parts = []
        if backfilled:
            msg_parts.append(f"backfilled {backfilled} IEM tornado(es) "
                             "with post-survey EF/casualties")
        if added:
            msg_parts.append(f"added {added} SVRGIS-only tornado(es) "
                             "missing from IEM")
        print("[info] SVRGIS: " + "; ".join(msg_parts))
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
