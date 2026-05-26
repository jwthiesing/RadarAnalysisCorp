"""Unit tests for ThresholdSpec; offline-only (network-dependent paths skipped)."""

from __future__ import annotations

from radar_warning_game.game.round_builder import ThresholdSpec


def test_threshold_default_passes_any_day():
    spec = ThresholdSpec()
    assert spec.is_met({"tornado": 0, "hail": 0, "wind": 0})
    assert spec.is_met({"tornado": 10, "hail": 50, "wind": 100})


def test_threshold_all_categories_must_be_met():
    spec = ThresholdSpec(min_tornadoes=5, min_hail=20, min_wind=20)
    assert not spec.is_met({"tornado": 4, "hail": 50, "wind": 50})
    assert not spec.is_met({"tornado": 5, "hail": 19, "wind": 50})
    assert not spec.is_met({"tornado": 5, "hail": 20, "wind": 19})
    assert spec.is_met({"tornado": 5, "hail": 20, "wind": 20})


def test_threshold_handles_missing_categories():
    spec = ThresholdSpec(min_tornadoes=1)
    assert not spec.is_met({})    # no tornado key
    assert spec.is_met({"tornado": 1})
