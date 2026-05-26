"""Unit tests for radar_warning_game.geo.projection."""

from __future__ import annotations

import math

import pytest

from radar_warning_game.geo.projection import (
    EARTH_RADIUS_KM,
    bearing_deg,
    haversine_km,
    latlon_to_xy_km,
    storm_motion_from_two_points,
    xy_km_to_latlon,
)


def test_haversine_identity():
    assert haversine_km(35.0, -97.0, 35.0, -97.0) == pytest.approx(0.0, abs=1e-9)


def test_haversine_one_degree_lat_is_about_111_km():
    d = haversine_km(35.0, -97.0, 36.0, -97.0)
    assert d == pytest.approx(111.19, rel=1e-3)


def test_haversine_one_degree_lon_at_35_lat():
    # cos(35°) ≈ 0.819 → 1 deg lon ≈ 91.1 km at 35°N
    d = haversine_km(35.0, -97.0, 35.0, -96.0)
    assert d == pytest.approx(91.1, abs=0.5)


def test_haversine_known_pair():
    # OKC (35.47, -97.52) to Tulsa (36.15, -95.99): ~157 km great-circle
    d = haversine_km(35.47, -97.52, 36.15, -95.99)
    assert 150 < d < 165


def test_bearing_due_east():
    assert bearing_deg(35.0, -97.0, 35.0, -96.0) == pytest.approx(90.0, abs=0.5)


def test_bearing_due_north():
    assert bearing_deg(35.0, -97.0, 36.0, -97.0) == pytest.approx(0.0, abs=0.5)


def test_bearing_due_south():
    assert bearing_deg(35.0, -97.0, 34.0, -97.0) == pytest.approx(180.0, abs=0.5)


def test_bearing_due_west():
    assert bearing_deg(35.0, -97.0, 35.0, -98.0) == pytest.approx(270.0, abs=0.5)


def test_latlon_xy_roundtrip():
    lat0, lon0 = 35.0, -97.0
    for dlat, dlon in [(0.5, 0.5), (-0.3, 0.8), (1.0, -1.0)]:
        x, y = latlon_to_xy_km(lat0 + dlat, lon0 + dlon, lat0, lon0)
        lat, lon = xy_km_to_latlon(x, y, lat0, lon0)
        assert lat == pytest.approx(lat0 + dlat, abs=1e-6)
        assert lon == pytest.approx(lon0 + dlon, abs=1e-6)


def test_storm_motion_eastbound():
    """50 km due east in 1 hour ≈ 27 kt at 90° (TO) / 270° (FROM)."""
    m = storm_motion_from_two_points(35.0, -97.5, 0.0, 35.0, -96.95, 3600.0)
    assert m.to_deg == pytest.approx(90.0, abs=1.5)
    assert m.from_deg == pytest.approx(270.0, abs=1.5)
    assert m.speed_kt == pytest.approx(27.0, abs=2.0)


def test_storm_motion_zero_time_raises():
    with pytest.raises(ValueError):
        storm_motion_from_two_points(35.0, -97.0, 100.0, 35.0, -96.5, 100.0)


def test_storm_motion_str_format():
    m = storm_motion_from_two_points(35.0, -97.0, 0.0, 35.5, -97.0, 1800.0)
    s = str(m)
    assert "from " in s and "at" in s and "kt" in s
