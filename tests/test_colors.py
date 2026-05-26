"""Unit tests for the per-player/team color palette."""

from __future__ import annotations

from radar_warning_game.ui.colors import (
    EXTENDED_PALETTE,
    FULL_PALETTE,
    WONG_PALETTE,
    color_for_player,
    color_for_team,
    lighten,
)


def test_palette_sizes_match_plan():
    assert len(WONG_PALETTE) == 8
    assert len(FULL_PALETTE) == len(WONG_PALETTE) + len(EXTENDED_PALETTE)
    assert len(FULL_PALETTE) >= 50    # plan §4c calls for ~50


def test_all_palette_colors_are_hex_format():
    for c in FULL_PALETTE:
        assert c.startswith("#")
        assert len(c) == 7
        int(c[1:], 16)   # parseable


def test_color_assignment_is_deterministic():
    a1 = color_for_player("alice")
    a2 = color_for_player("alice")
    assert a1 == a2


def test_color_assignment_differs_across_typical_ids():
    """50 distinct IDs should land on >25 distinct colors (statistical sanity)."""
    seen = {color_for_player(f"player_{i}") for i in range(50)}
    assert len(seen) > 25


def test_color_for_team_is_deterministic():
    assert color_for_team("team:abc") == color_for_team("team:abc")


def test_lighten_moves_toward_white():
    light = lighten("#000000", 0.5)
    # Half-lightened black → #808080 region
    r, g, b = int(light[1:3], 16), int(light[3:5], 16), int(light[5:7], 16)
    assert 120 < r < 135 and 120 < g < 135 and 120 < b < 135


def test_lighten_zero_is_identity():
    assert lighten("#123456", 0.0) == "#123456"


def test_lighten_one_is_white():
    assert lighten("#123456", 1.0) == "#FFFFFF"
