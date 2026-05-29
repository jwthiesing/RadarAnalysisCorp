"""Tests for :class:`PrefetchProgressWidget`'s live-mode handling.

The widget was originally written for the historical-archive path,
where a per-site pregame_total of 0 means "this day genuinely has
no archived data for this radar." Live mode borrows the same widget
but with very different semantics: pregame_total starts at 0 for
every site because scans haven't streamed in yet. These tests pin
the live-mode short-circuits that keep the historical "no archive
data" banner from misleading users in live play.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from radar_warning_game.data.cache import HashedCache
from radar_warning_game.data.prefetch import Prefetcher
from radar_warning_game.ui.prefetch_progress import PrefetchProgressWidget


@pytest.fixture(scope="module")
def qt_app():
    app = QApplication.instance() or QApplication([])
    yield app


def _prefetcher(sites: list[str], live: bool = False) -> Prefetcher:
    with tempfile.TemporaryDirectory() as td:
        cache = HashedCache(Path(td))
        pf = Prefetcher(sites, cache, live_source=live)
    return pf


def test_live_widget_does_not_show_no_archive_data_banner(qt_app):
    """Historical mode with all-zero pregame_total → "no archive data"
    banner. Live mode with the same all-zero state → no banner, just
    the normal "downloading…" title — because zero scans listed at
    construction time is the *expected* live-mode startup state, not
    a permanent condition."""
    pf = _prefetcher(["KTLX", "KOUN"], live=True)
    w = PrefetchProgressWidget(pf)
    assert w._is_live is True
    assert w._empty_sites == set()
    # The title shouldn't have flipped to the historical "no archive data"
    # error treatment — we should still see the normal progress framing.
    title_text = w._title.text()
    assert "Downloading" in title_text or "live" in title_text.lower()


def test_historical_widget_still_shows_empty_warning(qt_app):
    """Regression guard: the historical path's empty-sites diagnosis
    is intact (only the live mode short-circuits it)."""
    pf = _prefetcher(["KTLX"], live=False)
    w = PrefetchProgressWidget(pf)
    assert w._is_live is False
    # In historical mode, all-zero pregame_total is treated as "no
    # archive data for this day" — empty_sites populates accordingly.
    assert "KTLX" in w._empty_sites


def test_live_widget_advances_on_first_preloaded_scan(qt_app):
    """In live mode, the widget should emit ``local_prefetch_done``
    as soon as ANY site has at least one downloaded + preloaded
    scan — there's no "all scans done" terminal state, scans stream
    in over wall-clock time, and the round needs to start as soon
    as the first volume is renderable.
    """
    pf = _prefetcher(["KTLX", "KOUN"], live=True)
    # Simulate one site getting a scan downloaded + preloaded.
    pf._states["KTLX"].pregame_total = 1
    pf._states["KTLX"].downloaded_scan_times.add(
        __import__("datetime").datetime(2026, 1, 1)
    )
    pf._states["KTLX"].preload_completed = 1

    w = PrefetchProgressWidget(pf)
    fired = []
    w.local_prefetch_done.connect(lambda: fired.append(True))
    # Manually run one poll tick — the timer would also do this
    # automatically every 500 ms; we drive it directly to avoid
    # waiting in the test.
    w._poll()
    assert fired == [True]


def test_live_widget_waits_when_no_scans_yet(qt_app):
    """When live mode hasn't seen its first scan, the widget shouldn't
    emit ``local_prefetch_done`` — the round can't start with
    nothing to display. Bars read "waiting for live data…"."""
    pf = _prefetcher(["KTLX"], live=True)
    w = PrefetchProgressWidget(pf)
    fired = []
    w.local_prefetch_done.connect(lambda: fired.append(True))
    w._poll()
    assert fired == []
    assert "waiting" in w._bars["KTLX"].format().lower()
