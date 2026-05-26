"""Unit tests for the WSR-88D site catalog."""

from __future__ import annotations

import pytest

from radar_warning_game.data.sites import (
    haversine_km,
    load_sites,
    nearest_site,
    site_by_icao,
    sites_within_km,
)


def test_load_sites_returns_many_conus_radars():
    sites = load_sites()
    assert len(sites) > 100
    # All ICAOs are 4-char uppercase
    for s in sites:
        assert len(s.icao) == 4
        assert s.icao == s.icao.upper()


def test_site_by_icao_lookup():
    s = site_by_icao("KTLX")
    assert s is not None
    assert s.state == "OK"
    assert -98 < s.lon < -97
    assert 35 < s.lat < 36


def test_site_by_icao_case_insensitive():
    assert site_by_icao("ktlx") == site_by_icao("KTLX")


def test_site_by_icao_missing_returns_none():
    assert site_by_icao("ZZZZ") is None


def test_nearest_site_to_okc_is_ktlx():
    site, dist = nearest_site(35.47, -97.52)
    assert site.icao == "KTLX"
    assert dist < 50


def test_nearest_site_conus_only_excludes_alaska():
    # Anchorage area — without conus_only would return PAEC/PAHG; with conus_only
    # returns the nearest CONUS radar (still hundreds of km away)
    site, dist = nearest_site(61.2, -149.9, conus_only=True)
    assert site.state not in {"AK", "HI", "PR"}
    assert dist > 1000   # far from Alaska


def test_sites_within_km_returns_sorted_list():
    near = sites_within_km(35.5, -97.5, radius_km=200)
    assert len(near) >= 1
    distances = [d for _, d in near]
    assert distances == sorted(distances)


def test_haversine_via_sites_module():
    # Independent test of the module's haversine helper (same as projection's)
    d = haversine_km(35.0, -97.0, 36.0, -97.0)
    assert 110 < d < 113
