"""Unit tests for the per-site preload progress accessor.

The :class:`Prefetcher` reads its preload counters directly off each
:class:`RadarPrefetchState`. ``pregame_preload_progress()`` exposes them
in the same shape as ``pregame_progress()`` so the prefetch widget can
render a second per-site bar without diverging from the download bar's
layout. These tests skip the actual S3 / PyART pipeline and just poke
the counters by hand to verify the accessor's semantics.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from radar_warning_game.data.cache import HashedCache
from radar_warning_game.data.prefetch import Prefetcher, RadarPrefetchState
from radar_warning_game.data.sweep_index import SweepIndex


def _make_prefetcher(sites: list[str]) -> Prefetcher:
    with tempfile.TemporaryDirectory() as td:
        cache = HashedCache(Path(td))
        pf = Prefetcher(sites, cache)
    return pf


def test_pregame_preload_progress_initially_zero():
    pf = _make_prefetcher(["KTLX", "KOUN"])
    progress = pf.pregame_preload_progress()
    assert progress == {"KTLX": (0, 0), "KOUN": (0, 0)}


def test_pregame_preload_progress_reflects_state_counters():
    """Bumping the per-state counters changes the accessor output."""
    pf = _make_prefetcher(["KTLX", "KOUN"])
    pf._states["KTLX"].pregame_total = 10
    pf._states["KTLX"].preload_completed = 4
    pf._states["KOUN"].pregame_total = 5
    pf._states["KOUN"].preload_completed = 5
    progress = pf.pregame_preload_progress()
    assert progress == {"KTLX": (4, 10), "KOUN": (5, 5)}


def test_pregame_preload_progress_clamps_to_pregame_total():
    """In-game preloads (after the round starts) bump preload_completed
    beyond pregame_total. The pregame-phase progress display should
    cap at the pregame denominator so the bar doesn't visibly overflow
    100% during the gate wait."""
    pf = _make_prefetcher(["KTLX"])
    pf._states["KTLX"].pregame_total = 5
    pf._states["KTLX"].preload_completed = 8  # 3 in-game volumes already done
    progress = pf.pregame_preload_progress()
    assert progress == {"KTLX": (5, 5)}


def test_preload_completed_advances_even_on_failure():
    """The actual preload entry-point uses ``try / finally`` to bump
    ``preload_completed`` regardless of parse success. A corrupted file
    shouldn't prevent the readiness gate from releasing."""
    pf = _make_prefetcher(["KTLX"])
    state = pf._states["KTLX"]
    state.pregame_total = 1
    # Hand the preloader a path that doesn't exist; PyART parse will
    # fail and the function should still bump preload_completed.
    pf._preload_one(state, Path("/nonexistent/file.ar2v"))
    assert state.preload_completed == 1


def _setup_live_pf_with_cached_partial(prev_size: int):
    """Build a live-mode prefetcher with a cached partial file of
    ``prev_size`` bytes, returning ``(prefetcher, state, scan_key,
    scan_time)`` for the test to drive."""
    from datetime import datetime, timezone
    pf = _make_prefetcher(["KTLX"])
    pf._live_source = True
    state = pf._states["KTLX"]
    cache = state.cache
    scan_key = "KTLX/KTLX_20260101_2000"
    cache_path = cache.path(scan_key)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"\0" * prev_size)
    state.downloaded_sizes[scan_key] = prev_size
    scan_time = datetime(2026, 1, 1, 20, 0, tzinfo=timezone.utc)
    state.downloaded_scan_times.add(scan_time)
    return pf, state, scan_key, scan_time


def test_live_grown_file_is_re_submitted_for_download(monkeypatch):
    """Live-mode regression: when an IEM listing reports a volume's
    remote size has grown beyond what we previously downloaded, the
    prefetcher must re-queue the file. Without this the cached
    partial-volume (first one or two sweeps emitted by the radar)
    would survive forever and the user would never see the full
    volume."""
    from radar_warning_game.data.radar_s3 import ScanRef
    pf, state, scan_key, scan_time = _setup_live_pf_with_cached_partial(1_000_000)
    # Monkeypatch the actual download so the submitted future just
    # returns the cached path without hitting the network.
    import radar_warning_game.data.prefetch as prefetch_mod
    monkeypatch.setattr(
        prefetch_mod, "download_live_volume",
        lambda scan, cache: cache.path(scan.key),
    )
    scan = ScanRef(site="KTLX", time=scan_time, key=scan_key, size=4_000_000)
    fut = pf._submit_if_new(state, scan)
    assert fut is not None, "grew → re-submit"
    assert scan_key in state.in_flight


def test_live_grown_file_resubmits_after_prior_download_completed(monkeypatch):
    """Regression for a bug where ``state.in_flight`` accumulated
    finished futures and the second poll treated the same scan_key as
    "still in flight" — so growth detection never fired and the
    cached partial volume survived forever.

    Scenario: poll 1 downloads 1 MB version → future completes. Poll 2
    sees the listing reporting 4 MB. The prefetcher must look past the
    stale ``state.in_flight`` entry (the future is done) and re-submit.
    """
    from radar_warning_game.data.radar_s3 import ScanRef
    pf, state, scan_key, scan_time = _setup_live_pf_with_cached_partial(1_000_000)
    import radar_warning_game.data.prefetch as prefetch_mod
    monkeypatch.setattr(
        prefetch_mod, "download_live_volume",
        lambda scan, cache: cache.path(scan.key),
    )
    # Simulate a *completed* prior download by parking a done future in
    # state.in_flight. Without the .done() check in _submit_if_new the
    # next call would short-circuit on `in_flight=True`.
    from concurrent.futures import Future
    finished = Future()
    finished.set_result(state.cache.path(scan_key))
    state.in_flight[scan_key] = finished

    scan = ScanRef(site="KTLX", time=scan_time, key=scan_key, size=4_000_000)
    fut = pf._submit_if_new(state, scan)
    assert fut is not None, "second poll must re-submit even with a stale in_flight entry"
    assert fut is not finished, "must enqueue a fresh download, not return the old future"


def test_live_unchanged_file_is_not_re_submitted():
    """The flip-side: if the listing reports the same size we already
    have, the prefetcher must skip re-downloading."""
    from radar_warning_game.data.radar_s3 import ScanRef
    pf, state, scan_key, scan_time = _setup_live_pf_with_cached_partial(4_000_000)
    scan = ScanRef(site="KTLX", time=scan_time, key=scan_key, size=4_000_000)
    fut = pf._submit_if_new(state, scan)
    assert fut is None
    assert scan_key not in state.in_flight


def test_live_unknown_size_does_not_trigger_redownload():
    """If the listing didn't surface a size (``size=0``), we have no
    way to detect growth and must NOT speculatively re-fetch — the
    server might return 304 / large bandwidth waste."""
    from radar_warning_game.data.radar_s3 import ScanRef
    pf, state, scan_key, scan_time = _setup_live_pf_with_cached_partial(1_000_000)
    scan = ScanRef(site="KTLX", time=scan_time, key=scan_key, size=0)
    fut = pf._submit_if_new(state, scan)
    assert fut is None


def test_pregame_progress_and_preload_progress_share_keys():
    """The widget assumes both accessors use the same site keys so it
    can render aligned bars."""
    pf = _make_prefetcher(["KTLX", "KOUN", "KDDC"])
    dl = pf.pregame_progress()
    pl = pf.pregame_preload_progress()
    assert set(dl.keys()) == set(pl.keys())
