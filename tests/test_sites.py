"""Unit tests for the WSR-88D + TDWR site catalog."""

from __future__ import annotations

import pytest

from radar_warning_game.data.sites import (
    SITE_KIND_TDWR,
    SITE_KIND_WSR88D,
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


def test_load_sites_includes_both_wsr88d_and_tdwr():
    sites = load_sites()
    wsr = [s for s in sites if s.kind == SITE_KIND_WSR88D]
    tdwr = [s for s in sites if s.kind == SITE_KIND_TDWR]
    # ~160 WSR-88Ds + ~45 TDWRs in the catalog. Exact counts aren't
    # important here, but if either drops below the minimums something
    # parsed wrong in RADARS.txt.
    assert len(wsr) >= 150
    assert len(tdwr) >= 40
    # TDWRs all start with T per the FAA naming convention; WSR-88D
    # CONUS sites start with K.
    assert all(s.icao.startswith("T") for s in tdwr)


def test_tdwr_is_tdwr_flag():
    tokc = site_by_icao("TOKC")
    assert tokc is not None
    assert tokc.is_tdwr is True
    assert tokc.kind == SITE_KIND_TDWR
    ktlx = site_by_icao("KTLX")
    assert ktlx is not None
    assert ktlx.is_tdwr is False
    assert ktlx.kind == SITE_KIND_WSR88D


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


def test_nearest_site_to_okc_picks_tdwr_by_default():
    # TOKC (the TDWR at Will Rogers airport) is closer to downtown OKC
    # than KTLX (the WSR-88D in Norman). Default ``nearest_site`` is
    # kind-agnostic so it picks the genuinely-nearest radar.
    site, dist = nearest_site(35.47, -97.52)
    assert site.icao == "TOKC"
    assert dist < 30


def test_nearest_wsr88d_to_okc_is_ktlx():
    # Restricting to WSR-88D filters out TDWRs and falls back to the
    # WFO long-range S-band radar.
    site, dist = nearest_site(35.47, -97.52, kinds=frozenset({SITE_KIND_WSR88D}))
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
