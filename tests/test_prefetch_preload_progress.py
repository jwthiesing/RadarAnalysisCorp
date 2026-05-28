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


def test_pregame_progress_and_preload_progress_share_keys():
    """The widget assumes both accessors use the same site keys so it
    can render aligned bars."""
    pf = _make_prefetcher(["KTLX", "KOUN", "KDDC"])
    dl = pf.pregame_progress()
    pl = pf.pregame_preload_progress()
    assert set(dl.keys()) == set(pl.keys())
