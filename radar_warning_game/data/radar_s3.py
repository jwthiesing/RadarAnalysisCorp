"""Unsigned boto3 access to the Unidata NEXRAD Level 2 mirror on S3.

The Unidata mirror (``s3://unidata-nexrad-level2``) is fully public-read, so no
AWS credentials are needed. Files there are gzipped (``.gz`` suffix); PyART's
``read_nexrad_archive`` decompresses transparently.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Lock

import boto3
from botocore import UNSIGNED
from botocore.config import Config

from .cache import HashedCache

log = logging.getLogger(__name__)

BUCKET = "unidata-nexrad-level2"

# Level 2 filenames: KTLX20130520_204812_V06[.gz]  or  KTLX20130520_204812[.gz]
_KEY_RE = re.compile(r"^([A-Z]{4})(\d{8})_(\d{6})(?:_V\d+)?(?:\.gz)?$")

_client_lock = Lock()
_s3_client = None


def s3_client():
    """Lazy-initialized unsigned boto3 S3 client (shared across the process)."""
    global _s3_client
    with _client_lock:
        if _s3_client is None:
            _s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        return _s3_client


@dataclass(frozen=True)
class ScanRef:
    """A single volume scan listing entry.

    ``size`` is the remote file size in bytes when the listing source
    exposes one (currently only the IEM live mirror's directory
    listings). 0 means "unknown / unavailable" — historical S3 listings
    don't surface size cheaply and don't need it, since archived files
    are immutable. The live mirror's files grow as the radar appends
    new sweeps mid-volume, so we compare ``size`` against the local
    cached file's size to decide whether to re-download.
    """

    site: str          # ICAO
    time: datetime     # UTC, parsed from the filename
    key: str           # full S3 object key
    size: int = 0      # bytes (0 == unknown)


def list_volumes_for_day(site: str, day: datetime) -> list[ScanRef]:
    """List all Level 2 volume keys for ``site`` on the UTC calendar day of ``day``."""
    site = site.upper()
    if day.tzinfo is None:
        day = day.replace(tzinfo=timezone.utc)
    prefix = f"{day:%Y/%m/%d}/{site}/"
    out: list[ScanRef] = []
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            k = obj["Key"]
            if k.endswith("_MDM"):
                continue
            base = k.split("/")[-1]
            m = _KEY_RE.match(base)
            if not m:
                continue
            _, ymd, hms = m.groups()
            t = datetime.strptime(ymd + hms, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
            out.append(ScanRef(site=site, time=t, key=k))
    out.sort(key=lambda r: r.time)
    return out


# Result cache for site-day archive probes. Keyed by ``(site, YYYY-MM-DD)``
# so a probe per site per day pays one S3 round-trip; subsequent calls
# (same site/day) read from memory. The cache is small — typical setup
# probes ~200 sites for one day = ~200 entries — so we don't bother
# evicting.
_AVAILABILITY_CACHE: dict[tuple[str, str], bool] = {}


def site_has_data_on_day(site: str, day: datetime) -> bool:
    """Return ``True`` iff the Unidata mirror has at least one Level 2
    volume for ``site`` on the UTC calendar day of ``day``.

    Used by the radar-selection map to grey out sites whose archive
    coverage is missing for a chosen historical day — TDWR coverage on
    the Unidata mirror in particular is patchy in older years and we
    don't want a player picking a TDWR that will yield empty radar
    panels at round start.

    Implementation note: we use ``ListObjectsV2`` with ``MaxKeys=1`` to
    pay the smallest possible S3 round-trip — we only need to know
    whether *any* key exists under the prefix. Result is memoized per
    ``(site, day)``."""
    site = site.upper()
    if day.tzinfo is None:
        day = day.replace(tzinfo=timezone.utc)
    key = (site, day.strftime("%Y-%m-%d"))
    cached = _AVAILABILITY_CACHE.get(key)
    if cached is not None:
        return cached
    prefix = f"{day:%Y/%m/%d}/{site}/"
    try:
        resp = s3_client().list_objects_v2(
            Bucket=BUCKET, Prefix=prefix, MaxKeys=1,
        )
        has_any = (resp.get("KeyCount") or 0) > 0
    except Exception as e:  # noqa: BLE001
        # Network blip / transient — fall back to "available" so the
        # site isn't punished for a probe error. The real prefetch will
        # surface any data-missing condition at round start.
        log.warning("archive probe failed for %s on %s: %s", site, key[1], e)
        has_any = True
    _AVAILABILITY_CACHE[key] = has_any
    return has_any


def list_volumes_in_window(site: str, start: datetime, end: datetime) -> list[ScanRef]:
    """All volumes for ``site`` whose scan time falls in ``[start, end]`` (UTC).

    Spans the day boundary if needed (window may cross midnight UTC).
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    out: list[ScanRef] = []
    d = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    end_day = datetime(end.year, end.month, end.day, tzinfo=timezone.utc)
    while d <= end_day:
        out.extend(list_volumes_for_day(site, d))
        d += timedelta(days=1)
    return [r for r in out if start <= r.time <= end]


def download_volume(scan: ScanRef, cache: HashedCache) -> Path:
    """Download (with caching) a single Level 2 volume from S3. Returns local path.

    Uses an atomic ``.part`` → final rename so partial files never appear complete.
    """
    if cache.exists(scan.key):
        return cache.path(scan.key)
    tmp = cache.temp_path(scan.key)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    s3_client().download_file(BUCKET, scan.key, str(tmp))
    return cache.finalize(scan.key)
