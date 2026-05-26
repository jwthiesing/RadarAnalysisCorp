"""Unit tests for end-of-round scoring + team aggregation + magnitude revisions."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from radar_warning_game.data.reports import Report
from radar_warning_game.geo.polygons import Polygon
from radar_warning_game.verification.reports_in_poly import (
    MCD,
    Magnitudes,
    Warning,
    WarningRevision,
)
from radar_warning_game.verification.scoring import (
    BASE_SVR_FA_PENALTY,
    BASE_SVR_POINTS,
    BASE_TOR_FA_PENALTY,
    BASE_TOR_POINTS,
    MAG_ACCURACY_WEIGHT,
    score_round,
    score_single_mcd,
    score_single_warning,
    score_team,
)
from radar_warning_game.verification.tornado_tiers import WarningType


_T0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
_GAME_POLY = Polygon(((34.5, -98.0), (34.5, -96.5), (36.0, -96.5), (36.0, -98.0)))
_WARN_POLY = Polygon(((35.1, -97.4), (35.1, -97.1), (35.4, -97.1), (35.4, -97.4)))


def _w(*, wid="w1", issuer="alice", wt=WarningType.TOR, issue=_T0,
       duration_min=30, mag=None, revisions_extra=None):
    revs = [WarningRevision(
        revision_time=issue, warning_type=wt, polygon=_WARN_POLY,
        duration=timedelta(minutes=duration_min),
        magnitudes=mag or Magnitudes(),
    )]
    if revisions_extra:
        revs.extend(revisions_extra)
    return Warning(warning_id=wid, issuer_id=issuer, team_id=issuer, revisions=revs)


def _r(*, time=_T0+timedelta(minutes=10), lat=35.25, lon=-97.25,
       category="tornado", magnitude=2.0, inj=0, fat=0):
    return Report(
        time=time, lat=lat, lon=lon, category=category, magnitude=magnitude,
        state="OK", county="", remark="", injuries=inj, fatalities=fat, source="IEM",
    )


# ---- single-warning baseline ----------------------------------------

def test_tor_verified_perfect_ef_match():
    w = _w(wt=WarningType.TOR, mag=Magnitudes(ef=2.0))
    r = _r(magnitude=2.0)
    ws = score_single_warning(w, [r])
    # base TOR × 1.0 tier × (1 + 0.5 mag bonus) = 200 × 1 × 1.5 = 300
    assert ws.points == pytest.approx(BASE_TOR_POINTS * 1.5)
    assert ws.is_false_alarm is False


def test_tor_false_alarm_negative_score():
    w = _w(wt=WarningType.TOR)
    ws = score_single_warning(w, [])
    assert ws.points == pytest.approx(-BASE_TOR_FA_PENALTY)
    assert ws.is_false_alarm is True


def test_svr_verified_perfect_hail_match_penalized_unverified_wind():
    """SVR with predicted hail+wind, only hail verifies → wind contributes 0
    to the magnitude mean (component-wise penalty for unverified prediction).
    """
    w = _w(wt=WarningType.SVR, mag=Magnitudes(hail_in=2.0, wind_mph=70))
    r = _r(category="hail", magnitude=2.0)
    ws = score_single_warning(w, [r])
    # hail accuracy = 1.0, wind accuracy = 0 (predicted but unverified)
    # mean = 0.5 → bonus = 0.5 × 0.5 = 0.25 → 100 × 1 × 1.25 = 125
    assert ws.magnitude_bonus == pytest.approx(0.25)
    assert ws.points == pytest.approx(BASE_SVR_POINTS * 1.25)


def test_svr_verified_both_hazards():
    """If both hazards verify and predictions match, full bonus."""
    w = _w(wt=WarningType.SVR, mag=Magnitudes(hail_in=2.0, wind_mph=70))
    rs = [
        _r(category="hail", magnitude=2.0),
        _r(category="wind", magnitude=70.0, time=_T0+timedelta(minutes=15)),
    ]
    ws = score_single_warning(w, rs)
    # Both perfect → mean=1.0 → bonus=0.5 → 100 × 1 × 1.5 = 150
    assert ws.magnitude_bonus == pytest.approx(0.5)


def test_svr_with_only_one_predicted_component():
    """Predicting only hail (wind=None) and verifying hail → full bonus."""
    w = _w(wt=WarningType.SVR, mag=Magnitudes(hail_in=2.0, wind_mph=None))
    r = _r(category="hail", magnitude=2.0)
    ws = score_single_warning(w, [r])
    # Only hail component contributes → mean=1.0 → bonus=0.5
    assert ws.magnitude_bonus == pytest.approx(0.5)


def test_pds_tor_with_ef3_gets_175x():
    w = _w(wt=WarningType.PDS_TOR, mag=Magnitudes(ef=3.0))
    r = _r(magnitude=3.0, inj=3, fat=1)
    ws = score_single_warning(w, [r])
    assert ws.tier_mult == pytest.approx(1.75)
    # 200 × 1.75 × 1.5 = 525
    assert ws.points == pytest.approx(525.0)


def test_tore_fa_is_heaviest_penalty():
    w = _w(wt=WarningType.TORE, mag=Magnitudes(ef=3.0))
    ws = score_single_warning(w, [])
    assert ws.fa_penalty_mult == pytest.approx(3.0)
    assert ws.points == pytest.approx(-BASE_TOR_FA_PENALTY * 3.0)


# ---- magnitude-revision semantics (plan §5) -------------------------

def test_magnitude_revision_picks_active_at_peak():
    """Hail estimate revision-aware. With both hail+wind predicted but only
    hail verifying, the mean of components is (hail_accuracy + 0) / 2."""
    revs_extra = [WarningRevision(
        revision_time=_T0+timedelta(minutes=15),
        warning_type=WarningType.SVR, polygon=_WARN_POLY,
        duration=timedelta(minutes=30),
        magnitudes=Magnitudes(hail_in=1.0, wind_mph=60),
    )]
    w = _w(wt=WarningType.SVR, mag=Magnitudes(hail_in=2.0, wind_mph=60),
           revisions_extra=revs_extra)
    r_early = _r(time=_T0+timedelta(minutes=10), category="hail", magnitude=2.5)
    ws = score_single_warning(w, [r_early])
    # Hail: |2.0 - 2.5|/2.5 = 0.2 → accuracy 0.8
    # Wind: predicted but unverified → 0
    # Mean = 0.4, bonus = 0.4 × 0.5 = 0.2
    assert ws.magnitude_bonus == pytest.approx(0.2, abs=0.01)


def test_magnitude_revision_uses_new_revision_for_later_peak():
    revs_extra = [WarningRevision(
        revision_time=_T0+timedelta(minutes=15),
        warning_type=WarningType.SVR, polygon=_WARN_POLY,
        duration=timedelta(minutes=30),
        magnitudes=Magnitudes(hail_in=1.0, wind_mph=60),
    )]
    w = _w(wt=WarningType.SVR, mag=Magnitudes(hail_in=2.0, wind_mph=60),
           revisions_extra=revs_extra)
    r_late = _r(time=_T0+timedelta(minutes=20), category="hail", magnitude=1.5)
    ws = score_single_warning(w, [r_late])
    # Hail: |1.0 - 1.5|/1.5 = 0.333 → accuracy 0.667
    # Wind: predicted but unverified → 0
    # Mean = 0.333, bonus = 0.333 × 0.5 ≈ 0.167
    assert ws.magnitude_bonus == pytest.approx(0.167, abs=0.01)


# ---- team aggregation -----------------------------------------------

def test_team_score_aggregates_member_warnings():
    """Two teammates' warnings sum into one team score."""
    w_alice = _w(wid="w1", issuer="alice", wt=WarningType.TOR, mag=Magnitudes(ef=2.0))
    w_bob = _w(wid="w2", issuer="bob", wt=WarningType.SVR,
                mag=Magnitudes(hail_in=1.5, wind_mph=70))
    # Bob's warning covers a different area; needs its own report
    w_bob.revisions[0] = WarningRevision(
        revision_time=_T0, warning_type=WarningType.SVR,
        polygon=Polygon(((35.0, -97.6), (35.0, -97.5), (35.1, -97.5), (35.1, -97.6))),
        duration=timedelta(minutes=30),
        magnitudes=Magnitudes(hail_in=1.5, wind_mph=70),
    )
    reports = [
        _r(category="tornado", magnitude=2.0),
        _r(category="hail", magnitude=1.5, lat=35.05, lon=-97.55),
    ]
    score = score_team(
        team_id="storm_chasers", member_ids=["alice", "bob"],
        warnings=[w_alice, w_bob], mcds=[],
        reports_in_game=reports, game_polygon=_GAME_POLY,
    )
    assert score.n_warnings == 2
    assert score.n_false_alarms == 0
    assert score.n_verifying_reports == 2
    assert score.pod == pytest.approx(1.0)
    assert score.far == pytest.approx(0.0)
    assert score.warnings_total > 0


def test_team_pod_with_partial_verification():
    """Team warns one of two reports → POD 0.5."""
    w = _w(wt=WarningType.TOR, mag=Magnitudes())
    verified = _r(category="tornado")
    missed = _r(category="tornado", time=_T0+timedelta(minutes=20),
                lat=35.7, lon=-97.0)   # outside warning polygon
    score = score_team(
        team_id="t1", member_ids=["alice"],
        warnings=[w], mcds=[],
        reports_in_game=[verified, missed], game_polygon=_GAME_POLY,
    )
    assert score.pod == pytest.approx(0.5)


def test_score_round_sorts_high_to_low():
    """score_round returns teams sorted by total descending."""
    w_alice = _w(wid="w-a", issuer="alice", wt=WarningType.TOR, mag=Magnitudes(ef=2.0))
    w_bob = _w(wid="w-b", issuer="bob", wt=WarningType.TORE, mag=Magnitudes(ef=2.0))
    # Bob's polygon outside the report → FA → heavy penalty
    w_bob.revisions[0] = WarningRevision(
        revision_time=_T0, warning_type=WarningType.TORE,
        polygon=Polygon(((35.0, -96.8), (35.0, -96.5), (35.1, -96.5), (35.1, -96.8))),
        duration=timedelta(minutes=30),
        magnitudes=Magnitudes(ef=2.0),
    )
    r = _r(category="tornado", magnitude=2.0)
    teams = {"alice_team": ["alice"], "bob_team": ["bob"]}
    warnings_by_player = {"alice": [w_alice], "bob": [w_bob]}
    results = score_round(teams, warnings_by_player, {}, [r], _GAME_POLY)
    assert [s.team_id for s in results] == ["alice_team", "bob_team"]


# ---- MCD scoring ----------------------------------------------------

def test_mcd_perfect_pib_match_per_hazard():
    m = MCD(mcd_id="m1", issuer_id="alice", team_id="alice",
            polygon=_WARN_POLY, issue_time=_T0,
            duration=timedelta(hours=2),
            pib_tornado=4, pib_wind=4, pib_hail=4)
    reports = [
        _r(category="tornado", magnitude=2.0, time=_T0+timedelta(minutes=10)),
        _r(category="wind", magnitude=78.0, time=_T0+timedelta(minutes=20)),
        _r(category="hail", magnitude=2.0, time=_T0+timedelta(minutes=30)),
    ]
    ms = score_single_mcd(m, reports)
    # All three hazards predicted PIB 4 and observed PIB 4 → full hazard score each
    for cat in ("tornado", "wind", "hail"):
        assert ms.hazard_scores[cat] > 0


def test_mcd_predicted_but_unobserved_penalized():
    m = MCD(mcd_id="m1", issuer_id="alice", team_id="alice",
            polygon=_WARN_POLY, issue_time=_T0,
            duration=timedelta(hours=2),
            pib_tornado=5, pib_wind=0, pib_hail=0)
    # No reports at all
    ms = score_single_mcd(m, [])
    assert ms.hazard_scores["tornado"] < 0


def test_mcd_lead_time_bonus():
    """MCD issued >60 min before first verifying report gets full lead bonus."""
    m = MCD(mcd_id="m1", issuer_id="alice", team_id="alice",
            polygon=_WARN_POLY, issue_time=_T0,
            duration=timedelta(hours=3),
            pib_tornado=3, pib_wind=0, pib_hail=0)
    r = _r(time=_T0+timedelta(minutes=90), category="tornado", magnitude=1.0)
    ms = score_single_mcd(m, [r])
    assert ms.lead_bonus > 0


# ---- per-revision tier scoring (plan §5) ---------------------------

def test_tier_upgrade_only_credits_post_upgrade_report():
    """Issue TOR, upgrade to TORR mid-warning; only post-upgrade reports earn TORR bonus."""
    revs_extra = [WarningRevision(
        revision_time=_T0+timedelta(minutes=10),
        warning_type=WarningType.TORR, polygon=_WARN_POLY,
        duration=timedelta(minutes=30), magnitudes=Magnitudes(ef=2.0),
    )]
    w = _w(wt=WarningType.TOR, mag=Magnitudes(ef=2.0), revisions_extra=revs_extra)
    # Pre-upgrade report (TOR active): no TORR bonus
    r_pre = _r(time=_T0+timedelta(minutes=5), magnitude=2.0)
    # Post-upgrade report (TORR active): 1.10x bonus
    r_post = _r(time=_T0+timedelta(minutes=15), lat=35.27, lon=-97.20, magnitude=2.0)
    ws = score_single_warning(w, [r_pre, r_post])
    # Per-match mults: [1.0 (TOR), 1.10 (TORR)] → mean = 1.05
    assert ws.tier_mult == pytest.approx(1.05)


def test_tier_pds_tor_per_match_with_significant_match():
    """PDS TOR earns full 1.75x for ANY match that triggers the significance check."""
    w = _w(wt=WarningType.PDS_TOR, mag=Magnitudes(ef=3.0))
    # Two matches: one weak (no bonus), one EF3+casualties (full bonus)
    weak = _r(time=_T0+timedelta(minutes=5), magnitude=1.0)
    strong = _r(time=_T0+timedelta(minutes=15), lat=35.27, lon=-97.20,
                 magnitude=3.0, inj=3, fat=1)
    ws = score_single_warning(w, [weak, strong])
    # Per-match: weak=1.0 (no sig), strong=1.75 → mean = 1.375
    assert ws.tier_mult == pytest.approx(1.375)


def test_per_hazard_mcd_lead_time():
    """MCD predicting tornado + hail: lead bonus is average across hazards."""
    m = MCD(mcd_id="m1", issuer_id="alice", team_id="alice",
            polygon=_WARN_POLY, issue_time=_T0,
            duration=timedelta(hours=3),
            pib_tornado=3, pib_wind=0, pib_hail=4)
    # Tornado at +30min, hail at +60min
    reports = [
        _r(time=_T0+timedelta(minutes=30), category="tornado", magnitude=2.0),
        _r(time=_T0+timedelta(minutes=60), category="hail", magnitude=2.0),
    ]
    ms = score_single_mcd(m, reports)
    # Tornado lead = 30/60 = 0.5 → 30 pts; hail lead = 60/60 = 1.0 → 60 pts
    # Mean of [30, 60] = 45
    assert ms.lead_bonus == pytest.approx(45.0, abs=1.0)


def test_per_hazard_mcd_lead_zero_for_unverified_predicted_hazard():
    """Predicted-but-unverified hazards drag the lead bonus down (contribute 0)."""
    m = MCD(mcd_id="m1", issuer_id="alice", team_id="alice",
            polygon=_WARN_POLY, issue_time=_T0,
            duration=timedelta(hours=3),
            pib_tornado=3, pib_wind=0, pib_hail=4)
    # Only tornado verifies; hail predicted but no hail report
    reports = [_r(time=_T0+timedelta(minutes=60), category="tornado", magnitude=2.0)]
    ms = score_single_mcd(m, reports)
    # Tornado lead = 60 pts; hail = 0 (unverified) → mean = 30
    assert ms.lead_bonus == pytest.approx(30.0, abs=1.0)
