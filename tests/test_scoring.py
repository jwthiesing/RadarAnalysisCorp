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


def test_tor_hail_tag_perfect_match_earns_bonus():
    """Tornado warning with an IBW hail tag scores the hail prediction
    against actual hail inside the polygon — even though hail doesn't
    'verify' a tornado warning. Verified by a tornado report; perfect
    hail match → full magnitude bonus."""
    w = _w(wt=WarningType.TOR, mag=Magnitudes(hail_in=2.0))
    rs = [
        _r(category="tornado", magnitude=1.0),
        _r(category="hail", magnitude=2.0, time=_T0+timedelta(minutes=15)),
    ]
    ws = score_single_warning(w, rs)
    assert ws.magnitude_bonus == pytest.approx(0.5)


def test_tor_hail_tag_no_hail_observed_zero_bonus():
    """Tornado warning with hail tag but no hail in the polygon → 0 bonus
    (penalizes the unrealized hail prediction)."""
    w = _w(wt=WarningType.TOR, mag=Magnitudes(hail_in=2.0))
    r = _r(category="tornado", magnitude=1.0)
    ws = score_single_warning(w, [r])
    assert ws.magnitude_bonus == pytest.approx(0.0)


def test_svr_tornado_possible_adds_bonus_when_tornado_in_polygon():
    """SVR with 'Tornado Possible' tag + tornado inside the polygon during
    the warning's valid time earns a flat bonus on top of any hail/wind
    score. The tornado does NOT verify the SVR (POD-wise), but it
    contributes to points."""
    from radar_warning_game.verification.scoring import SVR_TORNADO_POSSIBLE_BONUS
    w = _w(wt=WarningType.SVR, mag=Magnitudes(
        hail_in=2.0, wind_mph=70, tornado_possible=True,
    ))
    rs = [
        _r(category="hail", magnitude=2.0),
        _r(category="tornado", magnitude=1.0, time=_T0+timedelta(minutes=15)),
    ]
    ws = score_single_warning(w, rs)
    # Base SVR with perfect hail (wind unverified) = mag_bonus 0.25 → 125
    # Plus SVR_TORNADO_POSSIBLE_BONUS for the tornado in the polygon.
    assert ws.points == pytest.approx(BASE_SVR_POINTS * 1.25 + SVR_TORNADO_POSSIBLE_BONUS)
    assert ws.is_false_alarm is False


def test_svr_tornado_possible_no_tornado_means_no_bonus():
    """If the SVR's tornado-possible tag is set but no tornado actually
    occurs, the bonus is NOT awarded — the SVR scores like a normal SVR."""
    w = _w(wt=WarningType.SVR, mag=Magnitudes(
        hail_in=2.0, wind_mph=70, tornado_possible=True,
    ))
    rs = [_r(category="hail", magnitude=2.0)]
    ws = score_single_warning(w, rs)
    # Same as the no-TP case: base 100 × 1 × (1 + 0.25 mag) = 125
    assert ws.points == pytest.approx(BASE_SVR_POINTS * 1.25)


def test_svr_tornado_possible_rescues_from_fa_with_only_tornado():
    """An SVR-TP that catches ONLY a tornado (no hail/wind verifies) is
    not a false alarm — the bonus is the entire score."""
    from radar_warning_game.verification.scoring import SVR_TORNADO_POSSIBLE_BONUS
    w = _w(wt=WarningType.SVR, mag=Magnitudes(
        hail_in=None, wind_mph=None, tornado_possible=True,
    ))
    rs = [_r(category="tornado", magnitude=1.0)]
    ws = score_single_warning(w, rs)
    assert ws.is_false_alarm is False
    assert ws.points == pytest.approx(SVR_TORNADO_POSSIBLE_BONUS)


def test_svr_tornado_possible_off_tornado_doesnt_help():
    """An SVR without the tornado-possible tag does NOT get a bonus from
    a tornado in its polygon — the tag is opt-in."""
    w = _w(wt=WarningType.SVR, mag=Magnitudes(
        hail_in=2.0, wind_mph=70, tornado_possible=False,
    ))
    rs = [
        _r(category="hail", magnitude=2.0),
        _r(category="tornado", magnitude=1.0, time=_T0+timedelta(minutes=15)),
    ]
    ws = score_single_warning(w, rs)
    assert ws.points == pytest.approx(BASE_SVR_POINTS * 1.25)   # no TP bonus


def test_hail_predicted_zero_no_hail_observed_gets_full_credit():
    """An SVR that correctly predicts 'no hail' (hail_in=0) and verifies
    via wind only should earn full hail-component credit, not zero."""
    w = _w(wt=WarningType.SVR, mag=Magnitudes(hail_in=0.0, wind_mph=70))
    r = _r(category="wind", magnitude=70.0)
    ws = score_single_warning(w, [r])
    # Hail component = 1.0 (correct "no hail" call); wind = 1.0 → mean 1.0 → bonus 0.5
    assert ws.magnitude_bonus == pytest.approx(0.5)


def test_svr_only_predicted_hazard_counts_for_fa():
    """An SVR predicting only hail should NOT be rescued from FA by a
    severe wind report — the player staked a claim on hail, not wind.
    Conversely, the same SVR getting only sub-SVRD-tier severe hail
    should be partial verification (positive score), not FA — severe
    hail is still hail, just below the player's bonus aspiration.
    """
    # Predicted hail only; wind verifies SVR-family but isn't a claimed hazard.
    w_hail = _w(wt=WarningType.SVR, mag=Magnitudes(hail_in=1.5))
    wind_only_report = _r(category="wind", magnitude=70)
    ws = score_single_warning(w_hail, [wind_only_report])
    assert ws.is_false_alarm is True, "unpredicted wind shouldn't save SVR(hail) from FA"
    # Predicted hail+wind, only severe hail materializes → partial verification.
    w_both = _w(wt=WarningType.SVRD, mag=Magnitudes(hail_in=3.0, wind_mph=85))
    severe_but_sub_svrd_hail = _r(category="hail", magnitude=1.5)
    ws = score_single_warning(w_both, [severe_but_sub_svrd_hail])
    assert ws.is_false_alarm is False, "1.5\" hail must at least partially verify SVRD"
    assert ws.points > 0, "partial verification must score positively"


def test_canceled_warning_still_scores_during_active_period():
    """A canceled warning isn't excluded from scoring — reports that
    occurred between issue and cancel still verify it, earning normal
    credit. Cancel just truncates the valid window; it doesn't void
    the warning's scoring history."""
    w = _w(wt=WarningType.TOR, duration_min=30)
    w.canceled_at = _T0 + timedelta(minutes=15)
    r_during = _r(time=_T0+timedelta(minutes=5), magnitude=2.0)
    ws = score_single_warning(w, [r_during])
    assert ws.is_false_alarm is False
    assert ws.points == pytest.approx(BASE_TOR_POINTS)


def test_canceled_warning_ignores_reports_after_cancel():
    """Reports that occur after the cancel time don't verify a canceled
    warning — by canceling the player effectively retracted the
    forecast for the remaining window."""
    w = _w(wt=WarningType.TOR, duration_min=30)
    w.canceled_at = _T0 + timedelta(minutes=15)
    r_after = _r(time=_T0+timedelta(minutes=20), magnitude=2.0)
    ws = score_single_warning(w, [r_after])
    assert ws.is_false_alarm is True
    assert ws.points == pytest.approx(-BASE_TOR_FA_PENALTY)


def test_tor_with_no_magnitudes_earns_no_magnitude_bonus():
    """A bare tornado warning (no hail, no EF) has nothing to score
    against, so magnitude_bonus is 0 — the player only earns base TOR
    points × tier multiplier."""
    w = _w(wt=WarningType.TOR, mag=Magnitudes())
    r = _r(category="tornado", magnitude=2.0)
    ws = score_single_warning(w, [r])
    assert ws.magnitude_bonus == pytest.approx(0.0)
    assert ws.points == pytest.approx(BASE_TOR_POINTS)


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


# ---- SVRGIS-sourced reports (post-survey EF + casualties) ------------
# Verifies that scoring is source-agnostic: a SVRGIS report verifies
# warnings, contributes to peak-EF + casualty sums, and triggers tier
# bonuses identically to an IEM report. The data-source labeling
# differs (``source="SVRGIS"`` vs ``"IEM"``) but the scoring contract
# is value-based, not source-based.


def test_svrgis_report_verifies_tor():
    """A SVRGIS-sourced tornado report (rated or unrated) verifies a
    TOR warning the same way an IEM report does."""
    w = _w(wt=WarningType.TOR)
    r = Report(
        time=_T0 + timedelta(minutes=10),
        lat=35.25, lon=-97.25, category="tornado",
        magnitude=2.0, state="OK", county="",
        remark="SVRGIS-only (no IEM LSR)",
        injuries=0, fatalities=0, source="SVRGIS",
    )
    ws = score_single_warning(w, [r])
    # Verified TOR base (no magnitudes predicted → no bonus).
    assert ws.is_false_alarm is False
    assert ws.points == pytest.approx(BASE_TOR_POINTS)


def test_svrgis_post_survey_ef_triggers_pds_tor_bonus():
    """The PDS-TOR significance gate fires when SVRGIS reports a
    confirmed EF ≥ 2 — even if IEM's preliminary record had EF=-1.
    This is exactly the scenario the SVRGIS backfill was added for."""
    w = _w(wt=WarningType.PDS_TOR, mag=Magnitudes(ef=3.0))
    # Imagine IEM had this as EF=-1 ("radar-indicated"); SVRGIS
    # backfilled to EF3 post-survey. Scoring sees the SVRGIS value.
    r = Report(
        time=_T0 + timedelta(minutes=10),
        lat=35.25, lon=-97.25, category="tornado",
        magnitude=3.0, state="OK", county="", remark="",
        injuries=0, fatalities=0, source="SVRGIS",
    )
    ws = score_single_warning(w, [r])
    # peak_ef = 3 → significant → 1.75× tier multiplier.
    assert ws.tier_mult == pytest.approx(1.75)


def test_svrgis_casualties_trigger_significance_on_weak_tornado():
    """The 'significance' branch is ``EF≥2 OR casualties > 0``. SVRGIS
    authoritative casualty counts let a confirmed-weak (EF1) tornado
    with deaths still trigger the PDS bonus — important because IEM's
    free-text casualty parsing is unreliable."""
    w = _w(wt=WarningType.PDS_TOR, mag=Magnitudes(ef=1.0))
    r = Report(
        time=_T0 + timedelta(minutes=10),
        lat=35.25, lon=-97.25, category="tornado",
        magnitude=1.0,                  # weak, post-survey EF1
        state="OK", county="", remark="",
        injuries=5, fatalities=2,       # post-survey confirmed deaths
        source="SVRGIS",
    )
    ws = score_single_warning(w, [r])
    # casualties > 0 → significant → 1.75× even though EF1 < 2.
    assert ws.tier_mult == pytest.approx(1.75)


def test_svrgis_only_tornado_counts_for_pod():
    """SVRGIS-only tornadoes (no matching IEM LSR) should expand the
    POD denominator just like any other tornado in the game polygon."""
    from radar_warning_game.verification.reports_in_poly import (
        reports_in_polygon,
    )
    iem_tornado = _r(magnitude=2.0)   # in-poly IEM tornado
    svrgis_only = Report(
        time=_T0 + timedelta(minutes=20),
        lat=35.30, lon=-97.20, category="tornado",
        magnitude=1.0, state="OK", county="",
        remark="SVRGIS-only (no IEM LSR)",
        injuries=0, fatalities=0, source="SVRGIS",
    )
    in_game = reports_in_polygon(_GAME_POLY, [iem_tornado, svrgis_only])
    # Both report sources land in the denominator, regardless of source.
    sources = sorted(r.source for r in in_game)
    assert sources == ["IEM", "SVRGIS"]
    assert len(in_game) == 2


def test_unrated_svrgis_tornado_doesnt_corrupt_peak_ef():
    """A SVRGIS row with no confirmed EF (``magnitude = -1``, e.g. a
    historical record where the survey couldn't pin a rating) is
    correctly skipped by the peak-EF computation when a rated report
    exists. (Note: the per-warning ``tier_mult`` field is a *mean of
    per-match* multipliers, not the peak — so adding an unrated
    tornado does dilute that mean toward 1.0×. That's the existing
    scoring contract, independent of SVRGIS.)"""
    from radar_warning_game.verification.scoring import _peak_observed_ef
    from radar_warning_game.verification.reports_in_poly import (
        VerifyingMatch,
    )
    w = _w(wt=WarningType.PDS_TOR, mag=Magnitudes(ef=3.0))
    rated = Report(
        time=_T0 + timedelta(minutes=10),
        lat=35.25, lon=-97.25, category="tornado",
        magnitude=3.0, state="OK", county="", remark="",
        injuries=0, fatalities=0, source="SVRGIS",
    )
    unrated = Report(
        time=_T0 + timedelta(minutes=15),
        lat=35.30, lon=-97.20, category="tornado",
        magnitude=-1.0, state="OK", county="",
        remark="SVRGIS-only (no IEM LSR)",
        injuries=0, fatalities=0, source="SVRGIS",
    )
    # The peak-EF helper takes verifying matches; build minimal ones
    # bound to the warning's current revision.
    rev = w.current_revision
    matches = [
        VerifyingMatch(warning=w, revision=rev, report=rated,
                       lead_time=timedelta(minutes=10), late_warn=False),
        VerifyingMatch(warning=w, revision=rev, report=unrated,
                       lead_time=timedelta(minutes=15), late_warn=False),
    ]
    # Peak from the helper should pick the rated 3.0, not the -1.
    assert _peak_observed_ef(matches) == pytest.approx(3.0)


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
