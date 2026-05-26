"""Unit tests for warning ↔ report matching (incl. late-warn TORR window)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from radar_warning_game.data.reports import Report
from radar_warning_game.geo.polygons import Polygon
from radar_warning_game.verification.reports_in_poly import (
    DEFAULT_VERIFICATION_BUFFER_KM,
    MCD,
    Magnitudes,
    Warning,
    WarningRevision,
    find_verifying_reports,
    reports_in_mcd,
    reports_in_polygon,
)
from radar_warning_game.verification.tornado_tiers import (
    TORR_LATE_WARN_WINDOW,
    WarningType,
)


_T0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
_POLY = Polygon(((35.1, -97.4), (35.1, -97.1), (35.4, -97.1), (35.4, -97.4)))


def _make_warning(
    *,
    issuer="alice",
    wt=WarningType.TOR,
    duration_min=30,
    magnitudes=None,
    issue_time=_T0,
    polygon=None,
):
    return Warning(
        warning_id=f"w-{issuer}", issuer_id=issuer, team_id=issuer,
        revisions=[WarningRevision(
            revision_time=issue_time,
            warning_type=wt,
            polygon=polygon or _POLY,
            duration=timedelta(minutes=duration_min),
            magnitudes=magnitudes or Magnitudes(),
        )],
    )


def _make_report(*, time=_T0+timedelta(minutes=10), lat=35.25, lon=-97.25,
                 category="tornado", magnitude=2.0):
    return Report(
        time=time, lat=lat, lon=lon, category=category, magnitude=magnitude,
        state="OK", county="", remark="", injuries=0, fatalities=0, source="IEM",
    )


# ---- reports_in_polygon ---------------------------------------------

def test_reports_in_polygon_filters_by_space_and_time():
    inside = _make_report(lat=35.25, lon=-97.25, time=_T0+timedelta(minutes=5))
    outside = _make_report(lat=36.0, lon=-97.25, time=_T0+timedelta(minutes=5))
    before = _make_report(time=_T0-timedelta(minutes=5))
    after = _make_report(time=_T0+timedelta(minutes=40))
    window = (_T0, _T0+timedelta(minutes=30))
    got = reports_in_polygon(_POLY, [inside, outside, before, after], time_window=window)
    assert got == [inside]


def test_reports_in_polygon_uses_buffer():
    just_outside = _make_report(lat=35.43, lon=-97.25)  # ~3km beyond box edge → within 5km buffer
    got = reports_in_polygon(_POLY, [just_outside])
    assert got == [just_outside]


# ---- find_verifying_reports: basic -----------------------------------

def test_tor_verified_by_tornado_inside():
    w = _make_warning(wt=WarningType.TOR)
    r = _make_report(category="tornado")
    matches = find_verifying_reports(w, [r])
    assert len(matches) == 1
    assert matches[0].lead_time == timedelta(minutes=10)
    assert matches[0].late_warn is False


def test_tor_not_verified_by_hail():
    w = _make_warning(wt=WarningType.TOR)
    r = _make_report(category="hail", magnitude=2.0)
    assert find_verifying_reports(w, [r]) == []


def test_svr_verified_by_hail_at_threshold():
    w = _make_warning(wt=WarningType.SVR)
    r = _make_report(category="hail", magnitude=1.0)
    assert len(find_verifying_reports(w, [r])) == 1


def test_svr_not_verified_by_subsevere_hail():
    w = _make_warning(wt=WarningType.SVR)
    r = _make_report(category="hail", magnitude=0.75)
    assert find_verifying_reports(w, [r]) == []


def test_report_past_warning_duration_not_verified():
    w = _make_warning(duration_min=15)
    r = _make_report(time=_T0+timedelta(minutes=30))
    assert find_verifying_reports(w, [r]) == []


def test_report_outside_polygon_not_verified():
    w = _make_warning()
    r = _make_report(lat=36.5, lon=-97.25)
    assert find_verifying_reports(w, [r]) == []


# ---- late-warn TORR --------------------------------------------------

def test_torr_late_warn_within_10min_window_verifies():
    """TORR issued 5 min after a tornado report still verifies (plan §6)."""
    w = _make_warning(wt=WarningType.TORR, issue_time=_T0+timedelta(minutes=5))
    r = _make_report(time=_T0)  # report 5 min BEFORE warning
    matches = find_verifying_reports(w, [r])
    assert len(matches) == 1
    assert matches[0].late_warn is True
    assert matches[0].lead_time == timedelta(minutes=-5)


def test_torr_late_warn_beyond_window_not_verified():
    """TORR issued 15 min after a tornado report (outside 10 min) doesn't verify."""
    w = _make_warning(wt=WarningType.TORR, issue_time=_T0+timedelta(minutes=15))
    r = _make_report(time=_T0)
    assert find_verifying_reports(w, [r]) == []


def test_late_warn_rejected_for_plain_tor():
    """Only TORR allows late-warn — TOR issued after the report doesn't verify."""
    w = _make_warning(wt=WarningType.TOR, issue_time=_T0+timedelta(minutes=5))
    r = _make_report(time=_T0)
    assert find_verifying_reports(w, [r]) == []


def test_late_warn_rejected_for_svr():
    w = _make_warning(wt=WarningType.SVR, issue_time=_T0+timedelta(minutes=5))
    r = _make_report(time=_T0, category="hail", magnitude=2.0)
    assert find_verifying_reports(w, [r]) == []


def test_late_warn_exactly_at_window_edge():
    """Boundary: warning issued exactly at TORR_LATE_WARN_WINDOW seconds after report."""
    w = _make_warning(wt=WarningType.TORR, issue_time=_T0+TORR_LATE_WARN_WINDOW)
    r = _make_report(time=_T0)
    matches = find_verifying_reports(w, [r])
    assert len(matches) == 1   # ≤ window → verifies


# ---- revisions: active-revision lookup ------------------------------

def test_revision_at_returns_correct_revision():
    w = _make_warning()
    w.revisions.append(WarningRevision(
        revision_time=_T0+timedelta(minutes=10),
        warning_type=WarningType.TORR,
        polygon=_POLY, duration=timedelta(minutes=30),
        magnitudes=Magnitudes(),
    ))
    # At T+5 → original revision (TOR)
    assert w.revision_at(_T0+timedelta(minutes=5)).warning_type == WarningType.TOR
    # At T+15 → new revision (TORR)
    assert w.revision_at(_T0+timedelta(minutes=15)).warning_type == WarningType.TORR
    # Before issue → None
    assert w.revision_at(_T0-timedelta(minutes=1)) is None


def test_verification_uses_active_revision():
    """Type at report time determines whether it verifies (upgrade SVR→TOR)."""
    # Issue SVR at T0, upgrade to TOR at T+10
    w = _make_warning(wt=WarningType.SVR)
    w.revisions.append(WarningRevision(
        revision_time=_T0+timedelta(minutes=10),
        warning_type=WarningType.TOR,
        polygon=_POLY, duration=timedelta(minutes=30),
        magnitudes=Magnitudes(),
    ))
    # Tornado report at T+5 → SVR active → tornado doesn't verify
    early_torn = _make_report(time=_T0+timedelta(minutes=5))
    assert find_verifying_reports(w, [early_torn]) == []
    # Tornado report at T+15 → TOR active → verifies
    late_torn = _make_report(time=_T0+timedelta(minutes=15))
    matches = find_verifying_reports(w, [late_torn])
    assert len(matches) == 1
    assert matches[0].revision.warning_type == WarningType.TOR


# ---- cancellation ----------------------------------------------------

def test_canceled_warning_stops_at_cancel_time():
    w = _make_warning(duration_min=60)
    w.canceled_at = _T0+timedelta(minutes=20)
    # Report at T+30 should NOT verify (canceled at T+20)
    r = _make_report(time=_T0+timedelta(minutes=30))
    assert find_verifying_reports(w, [r]) == []
    # Report at T+10 (before cancel) should verify
    early = _make_report(time=_T0+timedelta(minutes=10))
    assert len(find_verifying_reports(w, [early])) == 1


# ---- MCD matching ---------------------------------------------------

def test_mcd_returns_reports_in_window_regardless_of_category():
    m = MCD(mcd_id="m1", issuer_id="alice", team_id="alice",
            polygon=_POLY, issue_time=_T0, duration=timedelta(hours=2),
            pib_tornado=3, pib_wind=2, pib_hail=4)
    reports = [
        _make_report(time=_T0+timedelta(minutes=10), category="tornado"),
        _make_report(time=_T0+timedelta(minutes=20), category="hail", magnitude=2.0),
        _make_report(time=_T0-timedelta(minutes=5)),    # before issue
        _make_report(time=_T0+timedelta(hours=3)),       # after end
    ]
    got = reports_in_mcd(m, reports)
    assert len(got) == 2  # the tornado and hail inside the window
