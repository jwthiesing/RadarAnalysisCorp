"""Live Level 2 data from the IEM live mirror (plan §12).

URL pattern: ``https://mesonet-nexrad.agron.iastate.edu/level2/raw/<SITE>/``
serves an Apache directory listing. Filenames look like
``<SITE>_YYYYMMDD_HHMM`` (no extension, minute precision). Files come in at the
radar's natural cadence (~3–5 min). The listing typically covers the last
~24 hours.

This module mirrors the public surface of :mod:`radar_s3` so that the rest of
the app (sweep index, prefetcher) can talk to the live source via the same
``ScanRef`` shape.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path

import requests

from .cache import HashedCache
from .radar_s3 import ScanRef

log = logging.getLogger(__name__)

BASE_URL = "https://mesonet-nexrad.agron.iastate.edu/level2/raw"

# Live filenames: KTLX_20260525_2057  (no extension, no seconds)
_LIVE_FILE_RE = re.compile(r"^([A-Z]{4})_(\d{8})_(\d{4})$")
_LISTING_HREF_RE = re.compile(
    r'<a href="(?P<href>[A-Z]{4}_\d{8}_\d{4})">',
    re.IGNORECASE,
)


def _parse_filename(name: str) -> tuple[str, datetime] | None:
    m = _LIVE_FILE_RE.match(name)
    if not m:
        return None
    site, ymd, hm = m.groups()
    t = datetime.strptime(ymd + hm, "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return site, t


# Send a polite User-Agent — IEM's mirror logs requests by source and a
# descriptive UA helps them notice and contact us if they need to.
_HTTP_HEADERS = {"User-Agent": "RadarAnalysisCorp/0.1 (game; uses IEM live mirror)"}


def list_live_volumes(site: str, *, timeout: float = 30.0) -> list[ScanRef]:
    """Scrape the IEM live directory for ``site`` and return ``ScanRef`` list."""
    site = site.upper()
    url = f"{BASE_URL}/{site}/"
    try:
        r = requests.get(url, timeout=timeout, headers=_HTTP_HEADERS)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("IEM live listing for %s failed: %s", site, e)
        return []
    refs: list[ScanRef] = []
    for href in _LISTING_HREF_RE.findall(r.text):
        parsed = _parse_filename(href)
        if parsed is None:
            continue
        parsed_site, t = parsed
        if parsed_site != site:
            continue
        refs.append(ScanRef(site=site, time=t, key=f"{site}/{href}"))
    refs.sort(key=lambda r: r.time)
    log.info("IEM live: %s — %d volumes available", site, len(refs))
    return refs


def download_live_volume(scan: ScanRef, cache: HashedCache, *, timeout: float = 120.0) -> Path:
    """Download a live volume (caching). Stores raw NEXRAD Level 2 bytes."""
    if cache.exists(scan.key):
        return cache.path(scan.key)
    url = f"{BASE_URL}/{scan.key}"
    tmp = cache.temp_path(scan.key)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=_HTTP_HEADERS) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
    except requests.RequestException as e:
        log.warning("IEM live download failed for %s: %s", scan.key, e)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return cache.finalize(scan.key)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def recent_lsr_window_hours(hours: int = 6) -> tuple[datetime, datetime]:
    """Return a ``(start, now)`` window covering the most recent ``hours`` of LSRs."""
    end = now_utc()
    return end - _hours(hours), end


def _hours(n: int):
    from datetime import timedelta
    return timedelta(hours=n)
