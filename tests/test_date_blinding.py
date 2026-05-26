"""Unit tests for date-blinding helpers (plan §4b lint)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from radar_warning_game.ui.time_format import (
    contains_date,
    format_dev_time,
    format_player_offset,
    format_player_time,
    format_player_time_short,
)


_T = datetime(2013, 5, 20, 20, 48, 12, tzinfo=timezone.utc)


def test_format_player_time_no_date():
    s = format_player_time(_T)
    assert s == "20:48:12Z"
    assert "2013" not in s
    assert "05" not in s.replace("48", "").replace("12", "")
    assert not contains_date(s)


def test_format_player_time_short_no_date():
    s = format_player_time_short(_T)
    assert s == "20:48Z"
    assert not contains_date(s)


def test_format_dev_time_does_include_date():
    """format_dev_time is the only allowed full-datetime formatter (logs only)."""
    s = format_dev_time(_T)
    assert "2013-05-20" in s


def test_contains_date_detects_iso():
    assert contains_date("2013-05-20") is True
    assert contains_date("started at 2013/05/20T20:48") is True
    assert contains_date("nothing here") is False
    assert contains_date("HH:MM is 20:48Z") is False


def test_format_player_time_handles_naive_datetime():
    """Naive datetimes should be treated as UTC, not error."""
    naive = datetime(2013, 5, 20, 20, 48, 12)
    s = format_player_time(naive)
    assert s == "20:48:12Z"


def test_format_player_offset_positive():
    assert format_player_offset(900) == "+15:00"
    assert format_player_offset(65) == "+01:05"


def test_format_player_offset_negative():
    assert format_player_offset(-300) == "-05:00"


def test_format_player_offset_never_leaks_date():
    for sec in [0, 60, 3600, -7200, 86400]:
        assert not contains_date(format_player_offset(sec))


# ---- "lint" sweep — every user-facing helper produces date-free output ----

@pytest.mark.parametrize("dt", [
    datetime(2000, 1, 1, 0, 0, tzinfo=timezone.utc),
    datetime(2013, 5, 20, 20, 48, 12, tzinfo=timezone.utc),
    datetime(2026, 12, 31, 23, 59, 59, tzinfo=timezone.utc),
])
def test_no_player_helper_leaks_date(dt):
    assert not contains_date(format_player_time(dt))
    assert not contains_date(format_player_time_short(dt))
