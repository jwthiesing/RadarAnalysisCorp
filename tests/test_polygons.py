"""Unit tests for radar_warning_game.geo.polygons (incl. 5km buffer)."""

from __future__ import annotations

import pytest

from radar_warning_game.geo.polygons import (
    Polygon,
    buffered_union_contains,
    contains_with_buffer,
    polygon_area_km2,
    polygon_fraction_of,
    polygon_union,
)


# Square polygon ~ 55 km × 55 km centered on (35.25, -97.25)
_BOX = Polygon(((35.0, -97.5), (35.0, -97.0), (35.5, -97.0), (35.5, -97.5)))


def test_polygon_requires_three_vertices():
    with pytest.raises(ValueError):
        Polygon(vertices=((35.0, -97.0), (35.0, -97.5)))


def test_centroid_is_average():
    lat, lon = _BOX.centroid_latlon
    assert lat == pytest.approx(35.25)
    assert lon == pytest.approx(-97.25)


def test_contains_inside():
    assert contains_with_buffer(_BOX, 35.25, -97.25) is True


def test_contains_just_outside_within_buffer():
    # ~3 km north of the box edge (well within 5 km buffer)
    assert contains_with_buffer(_BOX, 35.527, -97.25) is True


def test_contains_just_outside_beyond_buffer():
    # ~10 km north of the box edge (outside 5 km buffer)
    assert contains_with_buffer(_BOX, 35.6, -97.25) is False


def test_contains_far_outside():
    assert contains_with_buffer(_BOX, 36.0, -97.25) is False


def test_polygon_area_reasonable():
    area = polygon_area_km2(_BOX)
    # 55 km × ~46 km (cos(35°) shrink) ≈ 2500 km² — should be in the right ballpark
    assert 2000 < area < 3500


def test_fraction_of_self_is_one():
    assert polygon_fraction_of(_BOX, _BOX) == pytest.approx(1.0)


def test_fraction_of_smaller_polygon():
    inner = Polygon(((35.1, -97.4), (35.1, -97.1), (35.4, -97.1), (35.4, -97.4)))
    f = polygon_fraction_of(inner, _BOX)
    # Inner is roughly 60% × 60% = 36% of outer
    assert 0.2 < f < 0.5


def test_polygon_union_overlapping():
    a = Polygon(((35.0, -97.5), (35.0, -97.0), (35.3, -97.0), (35.3, -97.5)))
    b = Polygon(((35.2, -97.3), (35.2, -96.8), (35.5, -96.8), (35.5, -97.3)))
    merged = polygon_union([a, b])
    # Overlapping → single component
    assert len(merged) == 1
    # Union area > each individual
    assert polygon_area_km2(merged[0]) > polygon_area_km2(a)
    assert polygon_area_km2(merged[0]) > polygon_area_km2(b)


def test_polygon_union_disjoint():
    a = Polygon(((35.0, -97.5), (35.0, -97.0), (35.3, -97.0), (35.3, -97.5)))
    b = Polygon(((40.0, -90.0), (40.0, -89.5), (40.3, -89.5), (40.3, -90.0)))
    merged = polygon_union([a, b])
    # Disjoint → two components
    assert len(merged) == 2


def test_buffered_union_contains():
    a = Polygon(((35.0, -97.5), (35.0, -97.0), (35.3, -97.0), (35.3, -97.5)))
    b = Polygon(((35.4, -97.5), (35.4, -97.0), (35.7, -97.0), (35.7, -97.5)))
    # Point inside b
    assert buffered_union_contains([a, b], 35.55, -97.25) is True
    # Point in the gap between a and b (~5.55 km from each, just outside 5km buffer)
    assert buffered_union_contains([a, b], 35.35, -97.25) is False
    # Empty list returns False
    assert buffered_union_contains([], 35.0, -97.0) is False
