"""Parallel / buffered radar download worker.

Per the plan §10, the game uses a two-phase fetch strategy:

  - **Pre-game:** all clients download the first 30 min of the round in parallel
    (bounded thread pool). The host coordinates the start gate: once ≥75% of
    clients signal "ready", a 60-second countdown begins, then the round starts.
  - **In-game:** each client maintains a rolling ~20 min lookahead buffer ahead of
    the game clock per active radar. If a client falls behind, its panel shows
    "buffering…" but gameplay continues for everyone else.

This module exposes :class:`Prefetcher` which encapsulates both phases. UI / game
session code drives it.
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .cache import HashedCache
from .live import download_live_volume, list_live_volumes
from .radar_s3 import ScanRef, download_volume, list_volumes_in_window
from .sweep_index import SweepIndex

log = logging.getLogger(__name__)

PREGAME_WINDOW = timedelta(minutes=30)
INGAME_LOOKAHEAD = timedelta(minutes=20)
DEFAULT_PARALLELISM = 8


@dataclass
class RadarPrefetchState:
    site: str
    cache: HashedCache
    sweep_index: SweepIndex
    downloaded_scan_times: set[datetime] = field(default_factory=set)
    in_flight: dict[str, Future] = field(default_factory=dict)
    pregame_total: int = 0       # total scans seen by schedule_pregame (cached + new)


class Prefetcher:
    """Coordinates Level 2 downloads for one client across many radars.

    When ``live_source=True``, scan listings come from the IEM live directory
    (``mesonet-nexrad.agron.iastate.edu/level2/raw/``) instead of the S3
    Unidata mirror. Downloads are HTTP GET rather than boto3. Suitable for
    LIVE-mode rounds (plan §12) where the player is nowcasting recent weather.
    """

    def __init__(
        self,
        sites: list[str],
        cache: HashedCache,
        *,
        parallelism: int = DEFAULT_PARALLELISM,
        live_source: bool = False,
    ) -> None:
        self._states = {
            s: RadarPrefetchState(site=s.upper(), cache=cache, sweep_index=SweepIndex(s))
            for s in (site.upper() for site in sites)
        }
        self._pool = ThreadPoolExecutor(max_workers=parallelism, thread_name_prefix="radar-fetch")
        self._lock = threading.RLock()
        self._round_start: datetime | None = None
        self._round_end: datetime | None = None
        self._live_source = live_source

    @property
    def sites(self) -> list[str]:
        return list(self._states.keys())

    def sweep_index(self, site: str) -> SweepIndex:
        return self._states[site.upper()].sweep_index

    # ------------------------------ phase 1: pre-game ----------------------

    def schedule_pregame(self, start: datetime, end: datetime) -> list[Future]:
        """Schedule download of all volumes in ``[start, start + PREGAME_WINDOW]`` per site.

        Returns the list of in-flight futures so the caller can join / report progress.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        self._round_start = start
        self._round_end = end
        target_end = min(start + PREGAME_WINDOW, end)
        futures: list[Future] = []
        for site, state in self._states.items():
            scans = self._list_scans(site, start, target_end)
            state.pregame_total = len(scans)
            for scan in scans:
                fut = self._submit_if_new(state, scan)
                if fut is not None:
                    futures.append(fut)
        return futures

    def pregame_progress(self) -> dict[str, tuple[int, int]]:
        """Per-radar (completed, total) pre-game progress counts.

        ``total`` is the number of scans :meth:`schedule_pregame` enumerated for
        this site (cached and in-flight combined). ``completed`` counts files
        the prefetcher has confirmed available (cached on entry + freshly
        downloaded). A return of ``(0, 0)`` means schedule_pregame hasn't run
        yet for this site.
        """
        out: dict[str, tuple[int, int]] = {}
        for site, state in self._states.items():
            with self._lock:
                in_flight = list(state.in_flight.values())
                completed_downloads = sum(1 for f in in_flight if f.done())
                cached_at_start = len(state.downloaded_scan_times) - completed_downloads
                # Clamp negatives in the rare event a download completed and
                # was indexed before we recomputed (set membership is loose).
                cached_at_start = max(0, cached_at_start)
                done = cached_at_start + completed_downloads
                out[site] = (done, state.pregame_total)
        return out

    def pregame_done(self) -> bool:
        return all(f.done() for state in self._states.values() for f in list(state.in_flight.values()))

    # ------------------------------ phase 2: in-game -----------------------

    def advance_clock(self, virtual_time: datetime) -> list[Future]:
        """Ensure the in-game lookahead buffer covers up to ``virtual_time + INGAME_LOOKAHEAD``.

        Should be called each tick (or every few seconds) from the game-clock loop.
        """
        if self._round_end is None:
            return []
        if virtual_time.tzinfo is None:
            virtual_time = virtual_time.replace(tzinfo=timezone.utc)
        horizon = min(virtual_time + INGAME_LOOKAHEAD, self._round_end)
        futures: list[Future] = []
        for site, state in self._states.items():
            scans = self._list_scans(site, virtual_time, horizon)
            for scan in scans:
                fut = self._submit_if_new(state, scan)
                if fut is not None:
                    futures.append(fut)
        return futures

    def _list_scans(self, site: str, start: datetime, end: datetime) -> list[ScanRef]:
        if self._live_source:
            # Live source returns the directory listing; filter to the window
            return [s for s in list_live_volumes(site) if start <= s.time <= end]
        return list_volumes_in_window(site, start, end)

    def buffer_ok_at(self, virtual_time: datetime) -> dict[str, bool]:
        """Per-radar: is the latest scan at-or-before ``virtual_time`` already downloaded?"""
        out: dict[str, bool] = {}
        for site, state in self._states.items():
            scans = list_volumes_in_window(site, virtual_time - timedelta(minutes=10), virtual_time)
            latest = scans[-1] if scans else None
            out[site] = latest is None or latest.time in state.downloaded_scan_times
        return out

    # ------------------------------ shutdown -------------------------------

    def shutdown(self, *, wait: bool = False) -> None:
        self._pool.shutdown(wait=wait, cancel_futures=not wait)

    # ------------------------------ internal -------------------------------

    def _submit_if_new(self, state: RadarPrefetchState, scan: ScanRef) -> Future | None:
        with self._lock:
            if scan.key in state.in_flight or state.cache.exists(scan.key):
                if state.cache.exists(scan.key) and scan.time not in state.downloaded_scan_times:
                    # File was on disk before we started — record + index it now.
                    state.downloaded_scan_times.add(scan.time)
                    try:
                        state.sweep_index.add_file(state.cache.path(scan.key))
                    except Exception:  # noqa: BLE001
                        log.exception("Failed to index cached file for %s", scan.key)
                return None
            fut = self._pool.submit(self._download_and_index, state, scan)
            state.in_flight[scan.key] = fut
            return fut

    def _download_and_index(self, state: RadarPrefetchState, scan: ScanRef) -> Path:
        try:
            if self._live_source:
                local = download_live_volume(scan, state.cache)
            else:
                local = download_volume(scan, state.cache)
            state.sweep_index.add_file(local)
            with self._lock:
                state.downloaded_scan_times.add(scan.time)
            return local
        except Exception:
            log.exception("Failed to fetch %s", scan.key)
            raise
