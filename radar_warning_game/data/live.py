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
# Apache directory listings — IEM mirror format — render sizes in
# HUMAN-READABLE units (``mod_autoindex`` ``IndexOptions
# SuppressHTMLPreamble`` / default settings):
#
#   <td><a href="KTLX_20260527_1959">KTLX_20260527_1959</a></td>
#   <td align="right">2026-05-27 15:04  </td>
#   <td align="right">8.8M</td>
#
# So we look for a number-with-optional-K/M/G suffix in the size
# column instead of raw integer bytes. A bare integer is accepted
# too (some Apache configs emit raw bytes). The size tells the
# prefetcher when a live file has grown since we last fetched it
# (Level 2 files start small and accumulate sweeps as the radar
# finishes the volume); without it we'd cache the partial first
# version forever.
_SIZE_FIND_RE = re.compile(
    r"\b(?P<num>\d+(?:\.\d+)?)\s*(?P<unit>[KMG])?\b",
    re.IGNORECASE,
)

_UNIT_BYTES = {"": 1, "K": 1024, "M": 1024 ** 2, "G": 1024 ** 3}


def _parse_size_token(num_str: str, unit: str | None) -> int:
    """Convert a ``(num, unit)`` capture to bytes. ``unit`` is ``K``,
    ``M``, ``G``, or ``None``/empty (bytes)."""
    try:
        n = float(num_str)
    except (TypeError, ValueError):
        return 0
    mult = _UNIT_BYTES.get((unit or "").upper(), 1)
    return int(n * mult)


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
    """Scrape the IEM live directory for ``site`` and return ``ScanRef``
    entries, each carrying the remote file's byte size when the
    listing exposes one. The prefetcher uses ``size`` to detect when
    a live file has grown since it was last downloaded — Level 2
    volumes accumulate sweeps for a few minutes after the radar
    starts emitting them, and the first version we see is typically
    just one or two sweeps."""
    site = site.upper()
    url = f"{BASE_URL}/{site}/"
    try:
        r = requests.get(url, timeout=timeout, headers=_HTTP_HEADERS)
        r.raise_for_status()
    except requests.RequestException as e:
        log.warning("IEM live listing for %s failed: %s", site, e)
        return []
    text = r.text
    refs: list[ScanRef] = []
    for href_match in _LISTING_HREF_RE.finditer(text):
        href = href_match.group("href")
        parsed = _parse_filename(href)
        if parsed is None:
            continue
        parsed_site, t = parsed
        if parsed_site != site:
            continue
        # Pull the substring from this href to the next href (or end
        # of document) — the file size is the largest plausible
        # number-with-unit in that span. Apache listings have
        # ``<td>...size...</td>`` tags between hrefs, so isolating
        # per-href substring keeps us from picking up a neighbor's
        # size by accident.
        span_start = href_match.end()
        next_href = _LISTING_HREF_RE.search(text, span_start)
        span_end = next_href.start() if next_href else len(text)
        between = text[span_start:span_end]
        size = 0
        for s_match in _SIZE_FIND_RE.finditer(between):
            candidate = _parse_size_token(s_match.group("num"), s_match.group("unit"))
            # Plausible Level 2 file size: 100 KB to 500 MB. Skips
            # date fragments (4-digit years), times (HHMM), and any
            # other small numeric noise in the listing.
            if 100_000 <= candidate <= 500_000_000 and candidate > size:
                size = candidate
        refs.append(ScanRef(site=site, time=t, key=f"{site}/{href}", size=size))
    refs.sort(key=lambda r: r.time)
    log.info("IEM live: %s — %d volumes available", site, len(refs))
    return refs


def download_live_volume(scan: ScanRef, cache: HashedCache, *, timeout: float = 120.0) -> Path:
    """Download a live volume, caching to disk.

    If the volume is already cached but the listing reports a larger
    remote size (the radar appended more sweeps after our first
    download), the cached file is overwritten with the full current
    version. Without this re-download path we'd be stuck with a
    partial Level 2 file containing only the first one or two
    sweeps of a 5-minute volume.
    """
    if cache.exists(scan.key):
        local = cache.path(scan.key)
        try:
            local_size = local.stat().st_size
        except OSError:
            local_size = 0
        # Size unknown (size=0) or remote not larger → keep cached.
        # Strict ``>`` so equal sizes don't trigger pointless re-fetch.
        if scan.size <= 0 or local_size >= scan.size:
            return local
        log.debug(
            "IEM live: re-downloading %s (cached %d bytes < remote %d bytes)",
            scan.key, local_size, scan.size,
        )
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
