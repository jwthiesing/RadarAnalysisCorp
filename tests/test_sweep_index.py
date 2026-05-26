"""Unit tests for the SAILS-aware sweep index.

We test the index logic by populating it with synthetic SweepRefs directly —
no real PyART file is needed, since the SAILS handling is just data structure
manipulation (the PyART parsing is exercised in the data-layer smoke test).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radar_warning_game.data.sweep_index import (
    ELEV_TOLERANCE_DEG,
    SweepIndex,
    SweepRef,
    _parse_units_epoch,
)


_T0 = datetime(2013, 5, 20, 20, 0, 0, tzinfo=timezone.utc)


def _synthetic_volume(file_path: Path, volume_start: datetime, sails_count: int = 2):
    """Build SweepRefs that mimic a real VCP 12-with-SAILS volume.

    sweep_no 0: 0.5° at volume_start (surveillance)
    sweep_no 1: 0.9° at +1s
    sweep_no 2: 0.5° at +17s (SAILS supplemental)
    sweep_no 3: 1.3° at +32s
    sweep_no 4: 1.8° at +47s
    ...
    """
    refs = []
    elevations = [0.5, 0.9] + ([0.5] * (sails_count - 1)) + [1.3, 1.8, 2.4, 3.1, 4.0]
    for i, elev in enumerate(elevations):
        refs.append(SweepRef(
            site="KTLX", start_time=volume_start + timedelta(seconds=i * 16),
            elev_deg=elev, file=file_path, sweep_number=i,
        ))
    return refs


def test_add_file_idempotent(tmp_path, monkeypatch):
    si = SweepIndex("KTLX")
    fake = tmp_path / "vol1.ar2v"
    # Patch index_volume_file to return synthetic refs
    refs = _synthetic_volume(fake, _T0)
    monkeypatch.setattr(
        "radar_warning_game.data.sweep_index.index_volume_file",
        lambda f: tuple(refs),
    )
    n1 = si.add_file(fake)
    assert n1 == len(refs)
    n2 = si.add_file(fake)
    assert n2 == 0   # second call ignored


def test_at_elevation_returns_sails_count(tmp_path, monkeypatch):
    si = SweepIndex("KTLX")
    fake = tmp_path / "vol1.ar2v"
    refs = _synthetic_volume(fake, _T0, sails_count=2)
    monkeypatch.setattr(
        "radar_warning_game.data.sweep_index.index_volume_file",
        lambda f: tuple(refs),
    )
    si.add_file(fake)
    low = si.at_elevation(0.5)
    assert len(low) == 2  # SAILS gives 2 sweeps at 0.5° per volume


def test_at_elevation_tolerance():
    """Sweeps reported as 0.48° should match at_elevation(0.5)."""
    si = SweepIndex("KTLX")
    si._sweeps = [
        SweepRef("KTLX", _T0, 0.48, Path("x"), 0),
        SweepRef("KTLX", _T0 + timedelta(seconds=1), 0.90, Path("x"), 1),
    ]
    si._times = [_T0, _T0 + timedelta(seconds=1)]
    matches = si.at_elevation(0.5, tol=ELEV_TOLERANCE_DEG)
    assert len(matches) == 1
    assert matches[0].elev_deg == pytest.approx(0.48)


def test_latest_at_or_before(tmp_path, monkeypatch):
    si = SweepIndex("KTLX")
    refs = []
    for vol in range(3):
        vol_start = _T0 + timedelta(minutes=vol * 5)
        refs.extend(_synthetic_volume(tmp_path / f"v{vol}.ar2v", vol_start))
    monkeypatch.setattr(
        "radar_warning_game.data.sweep_index.index_volume_file",
        lambda f, _refs=refs: tuple(r for r in _refs if r.file == f),
    )
    for vol in range(3):
        si.add_file(tmp_path / f"v{vol}.ar2v")

    # Query exactly at the second volume start time → should return that sweep
    target = _T0 + timedelta(minutes=5)
    s = si.latest_at_or_before(target, elev_deg=0.5)
    assert s is not None
    assert s.start_time <= target


def test_latest_at_or_before_returns_none_when_no_match():
    si = SweepIndex("KTLX")
    # No sweeps loaded
    assert si.latest_at_or_before(_T0, elev_deg=0.5) is None


def test_step_in_elevation_forward_backward(tmp_path, monkeypatch):
    si = SweepIndex("KTLX")
    refs = []
    for vol in range(3):
        vol_start = _T0 + timedelta(minutes=vol * 5)
        refs.extend(_synthetic_volume(tmp_path / f"v{vol}.ar2v", vol_start))
    monkeypatch.setattr(
        "radar_warning_game.data.sweep_index.index_volume_file",
        lambda f, _refs=refs: tuple(r for r in _refs if r.file == f),
    )
    for vol in range(3):
        si.add_file(tmp_path / f"v{vol}.ar2v")

    low_sorted = sorted(si.at_elevation(0.5), key=lambda s: s.start_time)
    assert len(low_sorted) == 6   # 2 per vol × 3 vols

    # Pick the middle one; step +1 and -1 should match the adjacent
    middle = low_sorted[2]
    assert si.step_in_elevation(middle, +1) == low_sorted[3]
    assert si.step_in_elevation(middle, -1) == low_sorted[1]
    # Step past the end returns None
    assert si.step_in_elevation(low_sorted[-1], +1) is None
    assert si.step_in_elevation(low_sorted[0], -1) is None


def test_available_elevations_filters_to_window(tmp_path, monkeypatch):
    si = SweepIndex("KTLX")
    refs = _synthetic_volume(tmp_path / "v.ar2v", _T0)
    monkeypatch.setattr(
        "radar_warning_game.data.sweep_index.index_volume_file",
        lambda f: tuple(refs),
    )
    si.add_file(tmp_path / "v.ar2v")
    elevs = si.available_elevations(_T0)
    # Should include all the unique tilts (rounded to 2dp)
    assert 0.5 in elevs
    assert 0.9 in elevs
    assert 1.3 in elevs


def test_nearest_elevation_picks_closest(tmp_path, monkeypatch):
    si = SweepIndex("KTLX")
    refs = _synthetic_volume(tmp_path / "v.ar2v", _T0)
    monkeypatch.setattr(
        "radar_warning_game.data.sweep_index.index_volume_file",
        lambda f: tuple(refs),
    )
    si.add_file(tmp_path / "v.ar2v")
    # Request 1.5° → nearest is 1.3° (closer than 1.8°)
    s = si.nearest_elevation(_T0, target_elev=1.5)
    assert s.elev_deg == pytest.approx(1.3)


# ---- units parser ---------------------------------------------------

def test_parse_units_epoch_iso():
    epoch = _parse_units_epoch("seconds since 2013-05-20T20:03:56Z")
    assert epoch == datetime(2013, 5, 20, 20, 3, 56, tzinfo=timezone.utc)


def test_parse_units_epoch_space_format():
    epoch = _parse_units_epoch("seconds since 2013-05-20 20:03:56")
    assert epoch == datetime(2013, 5, 20, 20, 3, 56, tzinfo=timezone.utc)


def test_parse_units_epoch_garbage_raises():
    with pytest.raises(ValueError):
        _parse_units_epoch("days since something")
