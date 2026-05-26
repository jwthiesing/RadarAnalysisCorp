"""Per-player / per-team color palette (plan §4c).

We need ~50 perceptually distinct colors (one room can hold up to 50 players).
The first 8 are from the Wong colorblind-safe palette so small groups always
get accessible colors. Beyond 8 we cycle through a hand-curated set of
saturated hues that remain distinguishable from each other and from the Wong 8.

Assignment is stable for a given (room, player_id) pair: we hash the player_id
into the palette so two players with the same ID always get the same color —
useful for replay re-rendering.
"""

from __future__ import annotations

import hashlib

# Wong 2011 colorblind-safe (8 colors)
WONG_PALETTE: tuple[str, ...] = (
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
)

# Extended palette — saturated and distinct from Wong; cycle when room > 8 players
EXTENDED_PALETTE: tuple[str, ...] = (
    "#A6324E", "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
    "#FFBB78", "#98DF8A", "#FF9896", "#C5B0D5", "#C49C94", "#F7B6D2",
    "#C7C7C7", "#DBDB8D", "#9EDAE5", "#393B79", "#637939", "#8C6D31",
    "#843C39", "#7B4173", "#5254A3", "#8CA252", "#BD9E39", "#AD494A",
    "#A55194", "#6B6ECF", "#B5CF6B", "#E7BA52", "#D6616B", "#CE6DBD",
    "#9C9EDE", "#CEDB9C", "#E7CB94", "#E7969C", "#DE9ED6", "#3182BD",
)

FULL_PALETTE: tuple[str, ...] = WONG_PALETTE + EXTENDED_PALETTE


def color_for_player(player_id: str) -> str:
    """Deterministically pick a palette color for ``player_id``.

    Same ID → same color across runs and across clients (so replay rendering
    matches the original game).
    """
    h = int(hashlib.sha1(player_id.encode("utf-8")).hexdigest()[:8], 16)
    return FULL_PALETTE[h % len(FULL_PALETTE)]


def color_for_team(team_id: str) -> str:
    """Color for a team id. Solo teams reuse the player's color."""
    return color_for_player(team_id)


def lighten(hex_color: str, fraction: float = 0.5) -> str:
    """Lighten a #RRGGBB color toward white by ``fraction`` (0..1)."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = int(r + (255 - r) * fraction)
    g = int(g + (255 - g) * fraction)
    b = int(b + (255 - b) * fraction)
    return f"#{r:02X}{g:02X}{b:02X}"
