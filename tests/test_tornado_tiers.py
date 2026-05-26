"""Unit tests for tornado/severe tier multipliers."""

from __future__ import annotations

import pytest

from radar_warning_game.verification.tornado_tiers import (
    PDS_TOR_MIN_EF,
    TORE_MIN_EF,
    SVRC_HAIL_THRESHOLD_IN,
    SVRC_WIND_THRESHOLD_MPH,
    SVRD_HAIL_THRESHOLD_IN,
    SVRD_WIND_THRESHOLD_MPH,
    WarningType,
    allows_late_warn,
    severe_tier_multiplier,
    tornado_tier_multiplier,
    verifies_warning_type,
)


def test_warning_type_families():
    assert WarningType.SVR.is_severe_family
    assert WarningType.SVRD.is_severe_family
    assert not WarningType.SVR.is_tornado_family
    assert WarningType.TOR.is_tornado_family
    assert WarningType.TORE.is_tornado_family
    assert not WarningType.TOR.is_severe_family


# ---- tornado tiers ---------------------------------------------------

def test_tor_baseline():
    t = tornado_tier_multiplier(WarningType.TOR, peak_observed_ef=0, casualties=0)
    assert t.score_multiplier == 1.0
    assert t.fa_penalty_multiplier == 1.0


def test_torr_modest_bonus():
    t = tornado_tier_multiplier(WarningType.TORR, peak_observed_ef=1, casualties=0)
    assert t.score_multiplier == pytest.approx(1.10)
    assert t.fa_penalty_multiplier == 1.0


def test_pds_tor_with_ef2_gets_full_bonus():
    t = tornado_tier_multiplier(WarningType.PDS_TOR, peak_observed_ef=2, casualties=0)
    assert t.score_multiplier == pytest.approx(1.75)
    assert t.fa_penalty_multiplier == 1.5


def test_pds_tor_with_ef1_only_no_bonus():
    t = tornado_tier_multiplier(WarningType.PDS_TOR, peak_observed_ef=1, casualties=0)
    assert t.score_multiplier == 1.0


def test_pds_tor_with_casualties_only_gets_bonus():
    t = tornado_tier_multiplier(WarningType.PDS_TOR, peak_observed_ef=-1, casualties=3)
    assert t.score_multiplier == pytest.approx(1.75)


def test_tore_with_ef4_gets_full_bonus():
    t = tornado_tier_multiplier(WarningType.TORE, peak_observed_ef=4, casualties=0)
    assert t.score_multiplier == pytest.approx(2.5)
    assert t.fa_penalty_multiplier == 3.0


def test_tore_with_ef0_only_gets_reduced_score():
    """TORE over-issuance penalty: 0.75× when only a weak tornado verifies."""
    t = tornado_tier_multiplier(WarningType.TORE, peak_observed_ef=0, casualties=0)
    assert t.score_multiplier == pytest.approx(0.75)


def test_tornado_tier_rejects_non_tornado_type():
    with pytest.raises(ValueError):
        tornado_tier_multiplier(WarningType.SVR, peak_observed_ef=0, casualties=0)


# ---- severe tiers ---------------------------------------------------

def test_svr_baseline():
    t = severe_tier_multiplier(WarningType.SVR, peak_hail_in=1.0, peak_wind_mph=60.0)
    assert t.score_multiplier == 1.0


def test_svrc_hail_threshold_met():
    t = severe_tier_multiplier(
        WarningType.SVRC, peak_hail_in=SVRC_HAIL_THRESHOLD_IN, peak_wind_mph=0,
    )
    assert t.score_multiplier == pytest.approx(1.10)


def test_svrc_wind_threshold_met():
    t = severe_tier_multiplier(
        WarningType.SVRC, peak_hail_in=0, peak_wind_mph=SVRC_WIND_THRESHOLD_MPH,
    )
    assert t.score_multiplier == pytest.approx(1.10)


def test_svrc_neither_threshold_no_bonus():
    t = severe_tier_multiplier(WarningType.SVRC, peak_hail_in=1.0, peak_wind_mph=60.0)
    assert t.score_multiplier == 1.0


def test_svrd_threshold_met():
    t = severe_tier_multiplier(
        WarningType.SVRD, peak_hail_in=SVRD_HAIL_THRESHOLD_IN, peak_wind_mph=0,
    )
    assert t.score_multiplier == pytest.approx(1.25)
    assert t.fa_penalty_multiplier == 1.5


def test_svrd_neither_threshold_no_bonus_but_heavy_fa():
    t = severe_tier_multiplier(WarningType.SVRD, peak_hail_in=1.0, peak_wind_mph=60.0)
    assert t.score_multiplier == 1.0
    assert t.fa_penalty_multiplier == 1.5


def test_severe_tier_rejects_tornado_type():
    with pytest.raises(ValueError):
        severe_tier_multiplier(WarningType.TOR, peak_hail_in=0, peak_wind_mph=0)


# ---- late-warn + report verification --------------------------------

def test_only_torr_allows_late_warn():
    assert allows_late_warn(WarningType.TORR) is True
    for wt in (WarningType.TOR, WarningType.PDS_TOR, WarningType.TORE,
               WarningType.SVR, WarningType.SVRC, WarningType.SVRD):
        assert allows_late_warn(wt) is False


def test_tornado_warning_verified_only_by_tornado_report():
    assert verifies_warning_type(WarningType.TOR, "tornado", 0.0) is True
    assert verifies_warning_type(WarningType.TOR, "hail", 2.0) is False
    assert verifies_warning_type(WarningType.TOR, "wind", 80.0) is False


def test_severe_warning_verified_by_hail_or_wind_at_threshold():
    assert verifies_warning_type(WarningType.SVR, "hail", 1.0) is True
    assert verifies_warning_type(WarningType.SVR, "hail", 0.99) is False
    assert verifies_warning_type(WarningType.SVR, "wind", 58.0) is True
    assert verifies_warning_type(WarningType.SVR, "wind", 57.0) is False
    assert verifies_warning_type(WarningType.SVR, "tornado", 0.0) is False
