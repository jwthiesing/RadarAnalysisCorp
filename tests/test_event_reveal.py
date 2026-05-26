"""Unit tests for the event-reveal database (date → event metadata)."""

from __future__ import annotations

from datetime import datetime, timezone

from radar_warning_game.game.event_reveal import EventReveal, reveal_for


def test_known_event_moore_ef5():
    rev = reveal_for(datetime(2013, 5, 20, 12, tzinfo=timezone.utc))
    assert "Moore" in rev.name
    assert "OK" in rev.location
    assert "weather.gov" in rev.url
    assert rev.date_str == "2013-05-20"


def test_known_event_super_outbreak():
    rev = reveal_for(datetime(2011, 4, 27, 12, tzinfo=timezone.utc))
    assert "Super Outbreak" in rev.name


def test_unknown_event_falls_back_to_spc_url():
    rev = reveal_for(datetime(2020, 7, 4, 12, tzinfo=timezone.utc))
    assert "spc.noaa.gov" in rev.url
    assert "200704" in rev.url   # yymmdd encoding
    assert rev.date_str == "2020-07-04"


def test_reveal_handles_naive_datetime_format():
    """Function takes any datetime; date string formatted from y/m/d."""
    rev = reveal_for(datetime(2013, 5, 20, 12))
    assert rev.date_str == "2013-05-20"
