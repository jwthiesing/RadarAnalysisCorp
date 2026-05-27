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
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

from .cache import HashedCache
from .live import download_live_volume, list_live_volumes
from .radar_s3 import ScanRef, download_volume, list_volumes_in_window
from .sweep_index import SweepIndex

log = logging.getLogger(__name__)

PREGAME_LOOKBACK = timedelta(minutes=20)
"""How far before the round's start time to also pull volumes for.

Gives the player visible radar history at t=0 and ensures the clock can
actually start AT the round start time (vs. snapping forward to whatever
first scan landed after it). The cost is ~3-4 extra volumes per radar at
typical NEXRAD cadence.
"""

PREGAME_WINDOW = timedelta(minutes=30)
INGAME_LOOKAHEAD = timedelta(minutes=20)
DEFAULT_PARALLELISM = 8

# How often the in-game scheduler is allowed to re-poll the source for
# new scans. The lookahead window is 20 min, so re-listing every second
# is huge overkill — once every ~15 s of game time keeps the buffer
# fresh while cutting the per-tick S3 ListObjectsV2 traffic by ~15×.
INGAME_RELIST_THROTTLE = timedelta(seconds=15)

# How many fully-processed (PyART parsed + velocity-dealiased) Radar
# objects to keep in memory across the prefetcher's preload cache. Each
# volume is ~50-150 MB live, so 24 is the same upper bound that the
# RadarPanelGrid's LRU defaults to.
PRELOAD_CACHE_MAX = 24


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
        # Tracks the last virtual_time at which ``advance_clock`` actually
        # re-listed the source for new scans. Subsequent calls within
        # :data:`INGAME_RELIST_THROTTLE` of that time short-circuit (no
        # listing, no fetches). The game ticks at 1 Hz — without this,
        # solo play paid a ~100-500 ms S3 ListObjectsV2 per radar per
        # tick, which was the dominant on-clock UI freeze.
        self._last_relist_virtual: datetime | None = None
        # advance_clock work is dispatched to this dedicated single-
        # worker pool so the main (UI) thread never blocks on S3 LIST
        # or PyART parse. The single worker is intentional: serializing
        # listings prevents bursting against S3 when many ticks fire
        # back-to-back (high game speed), and the actual download work
        # still fans out to ``self._pool``.
        self._tick_pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="radar-tick",
        )

        # ---- preload cache (PyART parse + velocity dealias) ----
        # Holds Radar objects keyed by their local file path. Populated
        # in the background by ``_preload_one`` immediately after each
        # download finishes — so by the time the user scrubs to a sweep
        # in a new volume, the parse + dealias work is already done.
        # The grid's ``_get_radar_from_cache`` checks here first;
        # without this, every volume crossing during a scrub stalls the
        # main thread for 200 ms (file parse) + 500-3000 ms (region-
        # based dealias). That stall is the single biggest source of
        # perceived "stutter" beyond the per-frame rasterize budget.
        self._loaded_radars: "OrderedDict[Path, object]" = OrderedDict()
        # In-flight preload futures, keyed by path. Lets the grid wait
        # on a partial preload instead of starting a duplicate load.
        self._preload_futures: dict[Path, Future] = {}
        # Bounded pool for the preload work. CPU-bound (dealiasing is
        # numpy + PyART C extensions that release the GIL), so 2-4
        # workers is enough; more would steal time from the rasterize
        # pool. ``_preload_pool`` is created lazily so headless tests
        # that never touch the pyart import path don't spin it up.
        self._preload_pool: ThreadPoolExecutor | None = None
        # Velocity dealias mode used by the preload step. The grid
        # later runs its own ``_apply_dealias`` if the user toggles
        # the mode mid-round — that path is idempotent and cheap on a
        # radar that's already had ``corrected_velocity`` added.
        self._preload_dealias_mode: str | None = "region_based"
        # Optional callback fired (off the prefetch pool) when a
        # preload completes, so a UI consumer can promote the loaded
        # radar into its own LRU and warm subsequent renders.
        self._on_radar_preloaded: Callable[[Path, object], None] | None = None
        # Optional callback fired (off the download pool) immediately
        # after a newly-downloaded volume has been added to its site's
        # ``SweepIndex``. Earlier than ``_on_radar_preloaded`` (which
        # waits for PyART parse + dealias), so consumers like the
        # time-scrubber UI can refresh as soon as new sweeps become
        # available for scrubbing — not 1-3 s later when the radar
        # itself is dealiased and ready to render.
        self._on_volume_indexed: Callable[[str, Path], None] | None = None

    @property
    def sites(self) -> list[str]:
        return list(self._states.keys())

    def sweep_index(self, site: str) -> SweepIndex:
        return self._states[site.upper()].sweep_index

    # ------------------------------ phase 1: pre-game ----------------------

    def schedule_pregame(self, start: datetime, end: datetime) -> list[Future]:
        """Schedule download of pre-game volumes.

        Historical rounds: ``[start - PREGAME_LOOKBACK, start + PREGAME_WINDOW]``
        — the 20 minutes leading up to ``start`` plus the first 30 minutes of
        the round itself. The lookback ensures (a) the clock can begin at the
        nominal start time with a sweep already available rather than waiting
        for whatever scan landed after it, and (b) the player sees ~20 min of
        recent radar history when the round opens.

        Live rounds: ``[now - PREGAME_WINDOW, now]`` — the most recent 30
        minutes the IEM live mirror has. Live volumes are by definition in
        the recent past; reading from ``start`` (= now) forward returns nothing.
        """
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        self._round_start = start
        self._round_end = end
        if self._live_source:
            now = datetime.now(timezone.utc)
            target_start = now - PREGAME_WINDOW
            target_end = now
        else:
            target_start = start - PREGAME_LOOKBACK
            target_end = min(start + PREGAME_WINDOW, end)
        futures: list[Future] = []
        for site, state in self._states.items():
            scans = self._list_scans(site, target_start, target_end)
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
        """Schedule the next round of in-game lookahead-buffer maintenance.

        Returns *immediately* — the S3 ListObjectsV2 calls (and any new
        downloads they trigger) run on a dedicated tick pool worker so
        the main UI thread is never blocked. The lookahead window is
        20 min, so we also throttle re-listings to one every
        :data:`INGAME_RELIST_THROTTLE` of game-time — calling this on
        every 1 Hz tick is fine, the throttle short-circuits the redundant
        calls.

        The returned list is the download futures *enqueued during the
        previous synchronous call* — kept for backward-compat with code
        that polled it. The new (async) callers can ignore it."""
        if self._round_end is None:
            return []
        if virtual_time.tzinfo is None:
            virtual_time = virtual_time.replace(tzinfo=timezone.utc)
        with self._lock:
            last = self._last_relist_virtual
            if last is not None and virtual_time - last < INGAME_RELIST_THROTTLE:
                return []
            self._last_relist_virtual = virtual_time
        # Dispatch the actual listing + scheduling to the tick pool;
        # the main thread returns immediately so the game-clock tick
        # finishes in microseconds instead of multi-hundred-ms.
        self._tick_pool.submit(self._advance_clock_blocking, virtual_time)
        return []

    def advance_clock_blocking(self, virtual_time: datetime) -> list[Future]:
        """Synchronous variant of :meth:`advance_clock` — runs the S3
        listing + download enqueue on the caller's thread. Kept for
        existing tests that assert against the returned futures and for
        the pre-game phase where blocking is desired."""
        return self._advance_clock_blocking(virtual_time)

    def _advance_clock_blocking(self, virtual_time: datetime) -> list[Future]:
        if self._live_source:
            # Live mode: poll backwards for newly-arrived volumes. The cache
            # dedupes, so re-listing is cheap (only NEW volumes
            # trigger fetches).
            now = datetime.now(timezone.utc)
            window_start = now - INGAME_LOOKAHEAD
            window_end = now
        else:
            window_start = virtual_time
            window_end = min(virtual_time + INGAME_LOOKAHEAD, self._round_end)
        futures: list[Future] = []
        for site, state in self._states.items():
            try:
                scans = self._list_scans(site, window_start, window_end)
            except Exception:  # noqa: BLE001
                log.exception("Failed to list scans for %s in [%s, %s]",
                              site, window_start, window_end)
                continue
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
        self._tick_pool.shutdown(wait=wait, cancel_futures=not wait)
        if self._preload_pool is not None:
            self._preload_pool.shutdown(wait=wait, cancel_futures=not wait)

    # ------------------------------ preload API ----------------------------

    def set_preload_dealias_mode(self, mode: str | None) -> None:
        """Tell the preloader which velocity-dealias algorithm to apply
        to freshly-loaded volumes. Pass ``None`` to skip dealias entirely
        (matches :class:`DealiasMode.NONE` on the panel grid). New
        downloads use the updated mode; already-loaded radars stay as-is,
        and the grid's own ``_apply_dealias`` will re-dealias on demand
        if the user toggles modes mid-round."""
        with self._lock:
            self._preload_dealias_mode = mode

    def set_radar_preloaded_callback(
        self, cb: Callable[[Path, object], None] | None,
    ) -> None:
        """Register (or clear) a callback fired off the preload pool
        whenever a Radar object finishes loading + dealiasing. The grid
        uses this to warm its own LRU so subsequent scrubs into that
        volume don't pay the parse + dealias stall on the main thread."""
        with self._lock:
            self._on_radar_preloaded = cb

    def set_volume_indexed_callback(
        self, cb: Callable[[str, Path], None] | None,
    ) -> None:
        """Register (or clear) a callback fired off the download pool
        as soon as a newly-downloaded volume has been added to its
        site's :class:`SweepIndex`. The callback receives
        ``(site_icao, local_file_path)``.

        Use this to refresh UI that depends on the *set of available
        sweeps* without waiting for the heavier PyART parse + dealias
        (those fire via :meth:`set_radar_preloaded_callback`). The
        time-scrubber slider is the canonical consumer — it needs to
        extend its range the moment a new sweep is available to scrub
        to, not when the radar itself is ready to render."""
        with self._lock:
            self._on_volume_indexed = cb

    def get_loaded_radar(self, file: Path) -> object | None:
        """Return the cached PyART Radar for ``file`` if preload has
        finished; ``None`` if not yet loaded. The caller (typically the
        radar grid) should treat ``None`` as "load synchronously now"
        — by-design fallback so a cold cache never breaks rendering."""
        with self._lock:
            r = self._loaded_radars.get(file)
            if r is not None:
                self._loaded_radars.move_to_end(file)
            return r

    def wait_for_preload(self, file: Path, timeout: float = 0.0) -> object | None:
        """If a preload for ``file`` is currently in flight, block up to
        ``timeout`` seconds for it to finish and return the resulting
        Radar. Useful when the user scrubs into a brand-new volume that
        the preloader is *about* to have ready — waiting 200 ms for
        the in-flight preload to finish is faster than starting a fresh
        synchronous parse on the main thread."""
        with self._lock:
            fut = self._preload_futures.get(file)
        if fut is None:
            return self.get_loaded_radar(file)
        try:
            return fut.result(timeout=timeout) if timeout > 0 else (
                fut.result() if fut.done() else None
            )
        except Exception:  # noqa: BLE001
            return None

    # ------------------------------ internal -------------------------------

    def _submit_if_new(self, state: RadarPrefetchState, scan: ScanRef) -> Future | None:
        with self._lock:
            already_cached_on_disk = state.cache.exists(scan.key)
            if scan.key in state.in_flight or already_cached_on_disk:
                if already_cached_on_disk and scan.time not in state.downloaded_scan_times:
                    # File was on disk before we started — record + index it now.
                    state.downloaded_scan_times.add(scan.time)
                    try:
                        local = state.cache.path(scan.key)
                        state.sweep_index.add_file(local)
                        # Even though the file was already on disk, the
                        # PyART parse + dealias haven't run yet — kick
                        # off the preload so the first scrub into this
                        # volume doesn't stall.
                        self._schedule_preload(local)
                        # And ping the UI so the time scrubber's range
                        # extends to cover the newly-indexed sweeps.
                        self._notify_volume_indexed(state.site, local)
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
            # Tell any UI consumer that new sweeps are scrubbable for
            # this site — fired BEFORE the preload so the scrubber UI
            # picks up the new range without waiting for parse+dealias.
            self._notify_volume_indexed(state.site, local)
            # Chain the preload: parse + dealias in the background so
            # the first render off this volume is instant.
            self._schedule_preload(local)
            return local
        except Exception:
            log.exception("Failed to fetch %s", scan.key)
            raise

    def _notify_volume_indexed(self, site: str, local: Path) -> None:
        """Best-effort fire of the ``on_volume_indexed`` callback. A
        callback error is logged but doesn't propagate — UI hooks
        shouldn't be able to corrupt the prefetcher's state."""
        with self._lock:
            cb = self._on_volume_indexed
        if cb is None:
            return
        try:
            cb(site, local)
        except Exception:  # noqa: BLE001
            log.exception("on_volume_indexed callback failed (%s)", local)

    def _schedule_preload(self, file: Path) -> None:
        """Submit a background PyART parse + velocity-dealias for
        ``file`` (no-op if already loaded or in flight)."""
        with self._lock:
            if file in self._loaded_radars or file in self._preload_futures:
                return
            if self._preload_pool is None:
                # 2 workers is plenty — dealias is the dominant cost and
                # it releases the GIL inside PyART's C extensions, so
                # more parallelism doesn't help much and would compete
                # with the rasterize pool.
                self._preload_pool = ThreadPoolExecutor(
                    max_workers=2, thread_name_prefix="radar-preload",
                )
            fut = self._preload_pool.submit(self._preload_one, file)
            self._preload_futures[file] = fut

    def _preload_one(self, file: Path) -> object | None:
        """Parse + (optionally) dealias one volume. Runs on the preload
        pool. Errors are swallowed and logged so a single bad file
        doesn't poison the preload queue."""
        try:
            import pyart   # lazy — keeps import-time cheap for tests
            radar = pyart.io.read_nexrad_archive(str(file))
        except Exception as e:  # noqa: BLE001
            log.warning("Preload failed for %s: %s", file, e)
            with self._lock:
                self._preload_futures.pop(file, None)
            return None
        # PyART's NEXRAD reader leaves Nyquist-velocity zeroed on TDWR
        # (and some legacy WSR-88D) files; the dealias algorithms then
        # divide-by-zero. Repair the metadata before handing the radar
        # to dealias — otherwise the call below throws every time and
        # we ship out a radar without a corrected_velocity field, which
        # forces the grid to fall back to raw aliased velocity on
        # display.
        from .radar_repair import ensure_nyquist_velocity
        ensure_nyquist_velocity(radar)
        with self._lock:
            mode = self._preload_dealias_mode
        if mode and "velocity" in radar.fields:
            try:
                if mode == "region_based":
                    corrected = pyart.correct.dealias_region_based(radar)
                elif mode == "phase_unwrap":
                    corrected = pyart.correct.dealias_unwrap_phase(radar)
                else:
                    corrected = None
                if corrected is not None:
                    radar.add_field(
                        "corrected_velocity", corrected, replace_existing=True,
                    )
            except Exception as e:  # noqa: BLE001
                log.warning("Preload dealias (%s) failed for %s: %s",
                            mode, file, e)
        with self._lock:
            self._loaded_radars[file] = radar
            while len(self._loaded_radars) > PRELOAD_CACHE_MAX:
                self._loaded_radars.popitem(last=False)
            self._preload_futures.pop(file, None)
            cb = self._on_radar_preloaded
        if cb is not None:
            try:
                cb(file, radar)
            except Exception:  # noqa: BLE001
                log.exception("on_radar_preloaded callback failed")
        return radar
