"""Unit tests for the IEM live data source (offline-only parts)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from radar_warning_game.data.live import (
    BASE_URL,
    _LISTING_HREF_RE,
    _parse_filename,
    recent_lsr_window_hours,
)


def test_parse_valid_filename():
    parsed = _parse_filename("KTLX_20260525_1959")
    assert parsed is not None
    site, t = parsed
    assert site == "KTLX"
    assert t == datetime(2026, 5, 25, 19, 59, tzinfo=timezone.utc)


def test_parse_filename_rejects_metadata_files():
    """KTLX_20260525_1959_MDM.arv2 etc. should be ignored."""
    assert _parse_filename("KTLX_20260525_1959_MDM.arv2") is None
    assert _parse_filename("KTLX_20260525_1959.gz") is None
    assert _parse_filename("README.txt") is None


def test_parse_filename_rejects_garbage():
    for bad in ["", "KTLX", "KTLX_2026", "TLX_20260525_1959", "KTLX_20260525"]:
        assert _parse_filename(bad) is None


def test_listing_regex_extracts_hrefs():
    html = '''<tr><td><a href="KTLX_20260525_1959">KTLX_20260525_1959</a></td></tr>
               <tr><td><a href="KTLX_20260525_2002">KTLX_20260525_2002</a></td></tr>
               <tr><td><a href="KTLX_20260525_2002_MDM.arv2">md</a></td></tr>
               <tr><td><a href="/parent">..</a></td></tr>'''
    hrefs = _LISTING_HREF_RE.findall(html)
    assert "KTLX_20260525_1959" in hrefs
    assert "KTLX_20260525_2002" in hrefs
    # MDM-suffixed not captured (regex matches exactly the volume pattern)
    assert "KTLX_20260525_2002_MDM.arv2" not in hrefs


def test_recent_lsr_window_hours_endpoints_match_now():
    start, end = recent_lsr_window_hours(hours=6)
    # End should be very close to now
    assert (datetime.now(timezone.utc) - end).total_seconds() < 2.0
    # Window length matches request
    assert (end - start).total_seconds() == pytest.approx(6 * 3600, abs=1.0)


def test_base_url_matches_plan():
    assert BASE_URL == "https://mesonet-nexrad.agron.iastate.edu/level2/raw"


def test_list_live_volumes_parses_size_and_crosses_utc_midnight(monkeypatch):
    """Regression for two coupled bugs:

    1. A previous regex required ``</a>`` and the size to live on the
       same line with no intervening ``<`` characters. Apache listings
       break the line with ``<td>...</td>`` columns, so the regex
       matched nothing and the listing came back empty — symptom was
       "no volumes available" when there were plenty.
    2. The IEM live listing serves the radar's last ~24 hours, which
       routinely straddles UTC midnight. Filenames carry their own
       UTC date, so the parser must accept both yesterday's and
       today's volumes without any URL-side date filter (the
       endpoint takes none).
    """
    from radar_warning_game.data import live as live_mod
    html = (
        '<table>'
        '<tr><td><a href="KTLX_20251231_2330">KTLX_20251231_2330</a></td>'
        '<td>31-Dec-2025 23:30</td>'
        '<td align="right">3145728</td></tr>'
        '<tr><td><a href="KTLX_20251231_2335">KTLX_20251231_2335</a></td>'
        '<td>31-Dec-2025 23:35</td>'
        '<td align="right">4194304</td></tr>'
        '<tr><td><a href="KTLX_20260101_0000">KTLX_20260101_0000</a></td>'
        '<td>01-Jan-2026 00:00</td>'
        '<td align="right">1048576</td></tr>'
        '</table>'
    )

    class _FakeResp:
        text = html
        def raise_for_status(self): pass

    class _FakeRequests:
        @staticmethod
        def get(url, *args, **kw):
            return _FakeResp()
        class RequestException(Exception):
            pass

    monkeypatch.setattr(live_mod, "requests", _FakeRequests)
    refs = live_mod.list_live_volumes("KTLX")
    assert len(refs) == 3
    assert refs[0].time == datetime(2025, 12, 31, 23, 30, tzinfo=timezone.utc)
    assert refs[2].time == datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)
    # Sizes from the listing — extracted independently of href position.
    assert refs[0].size == 3_145_728
    assert refs[1].size == 4_194_304
    assert refs[2].size == 1_048_576
    # The 8-digit YYYYMMDD inside the filename ("20251231") must NOT
    # be misread as a file size. Word-boundary regex protects us: in
    # ``KTLX_20251231_2330`` the digits are wrapped in underscores
    # (word chars), so ``\b\d{4,}\b`` doesn't match the date.
    assert refs[0].size != 20_251_231


def test_list_live_volumes_size_defaults_to_zero_when_absent(monkeypatch):
    """If the listing doesn't surface a size (legacy index format,
    truncated response), ``size`` stays at 0 — and the prefetcher's
    growth check then treats it as 'don't speculatively re-fetch.'"""
    from radar_warning_game.data import live as live_mod
    html = (
        '<a href="KTLX_20260101_0000">KTLX_20260101_0000</a>\n'
        '<a href="KTLX_20260101_0005">KTLX_20260101_0005</a>\n'
    )

    class _FakeResp:
        text = html
        def raise_for_status(self): pass

    class _FakeRequests:
        @staticmethod
        def get(url, *args, **kw):
            return _FakeResp()
        class RequestException(Exception):
            pass

    monkeypatch.setattr(live_mod, "requests", _FakeRequests)
    refs = live_mod.list_live_volumes("KTLX")
    assert len(refs) == 2
    assert all(r.size == 0 for r in refs)
