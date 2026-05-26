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
