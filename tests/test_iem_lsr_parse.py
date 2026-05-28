"""Tests for IEM LSR → :class:`Report` normalization.

These tests pin down the magnitude-defaulting rules at the parser
boundary so a regression doesn't silently re-introduce the "0-mph wind
damage reports never verify an SVR" bug.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd

from radar_warning_game.data.reports import (
    UNKNOWN_WIND_DEFAULT_MPH,
    _iem_row_to_report,
)


def _row(**overrides) -> pd.Series:
    base = {
        "TYPECODE": "G",
        "MAG": 0.0,
        "LAT": 35.5,
        "LON": -97.5,
        "ST": "OK",
        "COUNTY": "Cleveland",
        "REMARK": "",
        "time": pd.Timestamp(datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)),
    }
    base.update(overrides)
    return pd.Series(base)


def test_unknown_wind_magnitude_defaults_to_60():
    """Wind LSRs with MAG=0 (damage-only reports) get defaulted to 60
    mph so they cross the 58 mph severe-wind threshold. Without this,
    an SVR with several damage reports in its valid window got marked
    as a false alarm."""
    r = _iem_row_to_report(_row(TYPECODE="D", MAG=0.0))
    assert r is not None
    assert r.category == "wind"
    assert r.magnitude == UNKNOWN_WIND_DEFAULT_MPH
    assert r.magnitude >= 58.0


def test_unknown_wind_default_applies_to_all_wind_codes():
    """The default fires for every WIND_CODES entry that lacks a
    magnitude — G/W/N/D — not just D, because the IEM CSV can come in
    with empty/zero MAG fields on any of them."""
    for code in ("G", "W", "N", "D"):
        r = _iem_row_to_report(_row(TYPECODE=code, MAG=0.0))
        assert r is not None and r.magnitude == UNKNOWN_WIND_DEFAULT_MPH


def test_measured_wind_magnitude_is_preserved():
    """A real anemometer reading must NOT be overwritten by the
    default — the parser only fills in MAG=0."""
    r = _iem_row_to_report(_row(TYPECODE="G", MAG=72.0))
    assert r is not None and r.magnitude == 72.0


def test_hail_zero_magnitude_is_not_defaulted():
    """Only wind gets the magnitude default. Hail with MAG=0 is left
    alone — a hail report claiming 0 inches is ignorable on its face,
    not a sub-severe-but-still-damage situation like wind."""
    r = _iem_row_to_report(_row(TYPECODE="H", MAG=0.0))
    assert r is not None and r.category == "hail" and r.magnitude == 0.0


def test_tornado_zero_magnitude_still_becomes_minus_one():
    """The pre-existing tornado branch still wins: unrated tornadoes
    (MAG=0) get magnitude=-1 to signal "unknown EF" for the SPC
    backfill path."""
    r = _iem_row_to_report(_row(TYPECODE="T", MAG=0.0))
    assert r is not None and r.category == "tornado" and r.magnitude == -1.0


def test_defaulted_wind_report_verifies_svr():
    """End-to-end: a damage-only wind report should now verify an
    SVR warning issued over its polygon, where previously the same
    report (MAG=0) failed the 58 mph threshold and was ignored."""
    from radar_warning_game.geo.polygons import Polygon
    from radar_warning_game.verification.reports_in_poly import (
        Magnitudes, Warning, WarningRevision, find_verifying_reports,
    )
    from radar_warning_game.verification.tornado_tiers import WarningType

    t0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
    poly = Polygon(((35.1, -97.4), (35.1, -97.1), (35.4, -97.1), (35.4, -97.4)))
    w = Warning(
        warning_id="w-wind", issuer_id="alice", team_id="alice",
        revisions=[WarningRevision(
            revision_time=t0,
            warning_type=WarningType.SVR,
            polygon=poly,
            duration=timedelta(minutes=30),
            magnitudes=Magnitudes(wind_mph=65.0),
        )],
    )
    r = _iem_row_to_report(_row(
        TYPECODE="D", MAG=0.0, LAT=35.25, LON=-97.25,
        time=pd.Timestamp(t0 + timedelta(minutes=10)),
    ))
    matches = find_verifying_reports(w, [r])
    assert len(matches) == 1
    assert matches[0].report.category == "wind"
    assert matches[0].report.magnitude == UNKNOWN_WIND_DEFAULT_MPH
