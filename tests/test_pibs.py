"""Unit tests for SPC Peak Intensity Bin scoring helpers."""

from __future__ import annotations

import pytest

from radar_warning_game.verification.pibs import (
    HAIL_PIBS,
    TORNADO_PIBS,
    WIND_PIBS,
    ef_to_mph,
    max_pib_for_category,
    observed_to_pib,
    pib_table,
)


# ---- table consistency ----------------------------------------------

def test_tables_have_expected_lengths():
    assert len(TORNADO_PIBS) == 7
    assert len(WIND_PIBS) == 7
    assert len(HAIL_PIBS) == 6


def test_max_pib_per_category():
    assert max_pib_for_category("tornado") == 7
    assert max_pib_for_category("wind") == 7
    assert max_pib_for_category("hail") == 6
    assert max_pib_for_category("nonsense") == 0


def test_pib_table_lookup_or_raises():
    assert pib_table("tornado") is TORNADO_PIBS
    with pytest.raises(ValueError):
        pib_table("foo")


# ---- ef_to_mph ------------------------------------------------------

@pytest.mark.parametrize("ef,mph", [(0, 75), (1, 95), (2, 120), (3, 152), (4, 184), (5, 210)])
def test_ef_to_mph_midpoints(ef, mph):
    assert ef_to_mph(ef) == pytest.approx(mph)


def test_ef_to_mph_negative_returns_zero():
    assert ef_to_mph(-1) == 0.0


def test_ef_to_mph_clamps_above_5():
    assert ef_to_mph(7) == ef_to_mph(5)


# ---- observed_to_pib: tornado ---------------------------------------

def test_tornado_ef0_to_pib1():
    # EF0 → 75 mph → PIB 1 (lower bound 65)
    assert observed_to_pib("tornado", 0) == 1


def test_tornado_ef2_to_pib4():
    # EF2 → 120 mph → highest PIB whose lower bound ≤ 120: PIB 4 (120)
    assert observed_to_pib("tornado", 2) == 4


def test_tornado_ef3_to_pib5():
    # EF3 → 152 mph → PIB 5 (140) is highest with lower bound ≤ 152
    assert observed_to_pib("tornado", 3) == 5


def test_tornado_ef5_to_pib7():
    # EF5 → 210 mph → PIB 7 (>175)
    assert observed_to_pib("tornado", 5) == 7


# ---- observed_to_pib: wind ------------------------------------------

@pytest.mark.parametrize("mph,expected_pib", [
    (50, 1),    # < 60 → PIB 1
    (62, 2),    # 55-70 → PIB 2 (lower 55)
    (78, 4),    # 75-90 → PIB 4 (lower 75)
    (95, 6),    # 95-115 → PIB 6 (lower 95)
    (120, 7),   # > 115 → PIB 7
])
def test_wind_observed_to_pib(mph, expected_pib):
    assert observed_to_pib("wind", mph) == expected_pib


# ---- observed_to_pib: hail ------------------------------------------

@pytest.mark.parametrize("inches,expected_pib", [
    (0.5, 1),   # ≤ 1.25 → PIB 1
    (1.5, 3),   # PIB 3 (lower 1.5)
    (2.0, 4),   # PIB 4 (lower 2.0)
    (3.0, 5),   # PIB 5 (lower 2.75)
    (4.5, 6),   # ≥ 4.0 → PIB 6
])
def test_hail_observed_to_pib(inches, expected_pib):
    assert observed_to_pib("hail", inches) == expected_pib


def test_observed_negative_tornado_is_unknown_no_observation():
    # EF=-1 sentinel means "no tornado observation" → PIB 0
    assert observed_to_pib("tornado", -1.0) == 0


def test_observed_ef0_is_real_tornado_pib1():
    # EF=0 is a real (weak) tornado — NOT a sentinel — and maps to PIB 1
    assert observed_to_pib("tornado", 0.0) == 1


def test_observed_zero_wind_or_hail_returns_zero():
    # Wind 0 mph or hail 0" = "no observation"
    assert observed_to_pib("wind", 0.0) == 0
    assert observed_to_pib("wind", -5.0) == 0
    assert observed_to_pib("hail", 0.0) == 0


def test_observed_unknown_category_returns_zero():
    assert observed_to_pib("snow", 1.0) == 0
