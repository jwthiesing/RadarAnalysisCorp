"""Regression tests for warning visibility across teams (plan §11).

A player should see their OWN warnings plus their teammates' on their
radar panel — never opposing teams'. The host central map (covered
elsewhere) is the only view that sees everything.

We exercise :meth:`PlayView._teammate_ids_including_self` directly
because that's the function whose output gets unioned into the
overlay-push call; testing it isolates the visibility rule from the
heavier Qt/pyqtgraph rendering plumbing.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from radar_warning_game.game.session import GameSession, Player
from radar_warning_game.ui import play_view


def _stub_play_view(session: GameSession, local_id: str):
    """Build a minimal stand-in that mimics the bits of PlayView the
    helper reaches into. Skipping ``__init__`` avoids the full
    Qt-widget construction the real PlayView does."""
    pv = play_view.PlayView.__new__(play_view.PlayView)
    pv.session = session
    pv.local_player_id = local_id
    return pv


def test_solo_player_sees_only_themselves():
    """No team_id assigned (or the synthetic solo-team-of-one) →
    teammate list is just the local player."""
    sess = GameSession()
    sess.add_player(Player(player_id="alice", display_name="Alice"))
    pv = _stub_play_view(sess, "alice")
    assert pv._teammate_ids_including_self() == ["alice"]


def test_teammates_visible_within_team():
    """All players sharing a team_id show up — including the local
    player, even if they aren't yet listed in the team's members
    array (defensive against transient mid-join state)."""
    sess = GameSession()
    sess.add_player(Player(player_id="alice", display_name="Alice"))
    sess.add_player(Player(player_id="bob",   display_name="Bob"))
    sess.add_player(Player(player_id="carol", display_name="Carol"))
    sess.teams["team:storm"] = ["alice", "bob"]
    sess.team_names["team:storm"] = "Storm Chasers"
    sess.players["alice"].team_id = "team:storm"
    sess.players["bob"].team_id = "team:storm"

    pv = _stub_play_view(sess, "alice")
    visible = set(pv._teammate_ids_including_self())
    assert visible == {"alice", "bob"}
    # Carol is an opponent — she must NOT be in the visibility set.
    assert "carol" not in visible


def test_opposing_team_warnings_hidden_from_push():
    """End-to-end behavior: ``_push_player_overlays`` should only hand
    teammate warnings (incl. local) to the radar grid; opposing teams'
    warnings are filtered out before they ever reach the renderer."""
    from radar_warning_game.geo.polygons import Polygon
    from radar_warning_game.verification.reports_in_poly import (
        Magnitudes, Warning, WarningRevision,
    )
    from radar_warning_game.verification.tornado_tiers import WarningType

    sess = GameSession()
    sess.add_player(Player(player_id="alice", display_name="Alice"))
    sess.add_player(Player(player_id="bob",   display_name="Bob"))
    sess.add_player(Player(player_id="carol", display_name="Carol"))
    sess.teams["team:storm"] = ["alice", "bob"]
    sess.teams["team:hail"] = ["carol"]
    sess.players["alice"].team_id = "team:storm"
    sess.players["bob"].team_id = "team:storm"
    sess.players["carol"].team_id = "team:hail"

    t0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
    poly = Polygon(((35.1, -97.4), (35.1, -97.1), (35.4, -97.1), (35.4, -97.4)))
    def mkw(wid: str, issuer: str) -> Warning:
        return Warning(
            warning_id=wid, issuer_id=issuer, team_id=sess.players[issuer].team_id,
            revisions=[WarningRevision(
                revision_time=t0, warning_type=WarningType.TOR, polygon=poly,
                duration=timedelta(minutes=30), magnitudes=Magnitudes(),
            )],
        )
    sess.warnings_by_player["alice"] = [mkw("alice-1", "alice")]
    sess.warnings_by_player["bob"]   = [mkw("bob-1",   "bob")]
    sess.warnings_by_player["carol"] = [mkw("carol-1", "carol")]

    pv = _stub_play_view(sess, "alice")
    # Capture what gets handed to the radar grid.
    pushed: dict = {}
    pv.radar_grid = SimpleNamespace(
        set_player_warnings=lambda warns, mcds: pushed.update(w=warns, m=mcds),
    )
    pv._push_player_overlays()
    visible_ids = {w.warning_id for w in pushed["w"]}
    assert visible_ids == {"alice-1", "bob-1"}
    assert "carol-1" not in visible_ids


def test_missing_player_falls_back_to_self_only():
    """If the local player isn't yet in session.players (mid-join
    handshake), the helper degrades gracefully to ``[self]`` rather
    than raising — otherwise a transient race would crash the
    overlay push."""
    sess = GameSession()
    pv = _stub_play_view(sess, "ghost")
    assert pv._teammate_ids_including_self() == ["ghost"]
