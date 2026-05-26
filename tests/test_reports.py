"""Unit tests for storm-report parsing (IEM LSR helpers)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from radar_warning_game.data.reports import (
    Report,
    count_by_category,
    filter_severe,
    parse_casualties,
)


# ---- parse_casualties -----------------------------------------------

@pytest.mark.parametrize("remark,expected", [
    ("*** 1 INJ ***", (1, 0)),
    ("*** 3 FATAL_ 8 INJ ***", (8, 3)),    # "3 FATAL" matches \d+\s*FAT
    ("2 INJ, 0 FAT REPORTED", (2, 0)),
    ("BRIEF TOUCHDOWN", (0, 0)),
    ("", (0, 0)),
    (None, (0, 0)),
    ("multiple injuries reported (no count)", (0, 0)),
    ("ESTIMATED 25 FATALITIES", (0, 25)),  # FAT inside FATALITIES still matches the count
    ("12 INJURED, 1 KILLED", (12, 0)),     # "12 INJ" matches (KILLED isn't FAT)
])
def test_parse_casualties(remark, expected):
    assert parse_casualties(remark) == expected


# ---- filter_severe --------------------------------------------------

def _r(**kwargs):
    defaults = dict(
        time=datetime(2024, 4, 1, 20, tzinfo=timezone.utc),
        lat=35.0, lon=-97.0, category="hail", magnitude=0.0,
        state="OK", county="", remark="",
        injuries=0, fatalities=0, source="IEM",
    )
    defaults.update(kwargs)
    return Report(**defaults)


def test_filter_severe_tornado_always_passes():
    rs = [_r(category="tornado", magnitude=-1.0)]  # unknown EF still passes
    assert len(filter_severe(rs)) == 1


def test_filter_severe_hail_at_threshold():
    assert len(filter_severe([_r(category="hail", magnitude=1.0)])) == 1
    assert len(filter_severe([_r(category="hail", magnitude=0.75)])) == 0


def test_filter_severe_wind_at_threshold():
    assert len(filter_severe([_r(category="wind", magnitude=58.0)])) == 1
    assert len(filter_severe([_r(category="wind", magnitude=50.0)])) == 0


def test_filter_severe_custom_thresholds():
    rs = [_r(category="hail", magnitude=1.5)]
    assert len(filter_severe(rs, min_hail_in=2.0)) == 0
    assert len(filter_severe(rs, min_hail_in=1.0)) == 1


# ---- count_by_category ----------------------------------------------

def test_count_by_category_basic():
    rs = [
        _r(category="tornado"), _r(category="tornado"),
        _r(category="hail"), _r(category="hail"), _r(category="hail"),
        _r(category="wind"),
    ]
    assert count_by_category(rs) == {"tornado": 2, "hail": 3, "wind": 1}


def test_count_by_category_empty():
    assert count_by_category([]) == {"tornado": 0, "hail": 0, "wind": 0}


def test_count_by_category_ignores_unknown():
    rs = [_r(category="snow"), _r(category="tornado")]
    counts = count_by_category(rs)
    assert counts["tornado"] == 1
    assert "snow" not in counts
