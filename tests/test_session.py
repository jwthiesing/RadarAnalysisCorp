"""Unit tests for GameSession state machine + team management."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from radar_warning_game.data.reports import Report
from radar_warning_game.geo.polygons import Polygon
from radar_warning_game.game.clock import GameClock
from radar_warning_game.game.round_builder import RoundDay
from radar_warning_game.game.session import (
    SOLO_TEAM_PREFIX,
    GameSession,
    Player,
    RoundConfig,
    RoundMode,
    SessionState,
)
from radar_warning_game.verification.reports_in_poly import Magnitudes
from radar_warning_game.verification.tornado_tiers import WarningType


_T0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
_POLY = Polygon(((34.5, -98.0), (34.5, -96.5), (36.0, -96.5), (36.0, -98.0)))


def _new_session_with(*player_names):
    s = GameSession()
    for i, n in enumerate(player_names):
        s.add_player(Player(player_id=n.lower(), display_name=n, is_host=(i == 0)))
    return s


# ---- lobby + solo teams -------------------------------------------

def test_add_player_creates_solo_team():
    s = _new_session_with("Alice")
    assert "alice" in s.players
    assert any(t.startswith(SOLO_TEAM_PREFIX) for t in s.teams)


def test_remove_player_cleans_team():
    s = _new_session_with("Alice")
    s.remove_player("alice")
    assert "alice" not in s.players
    # No teams referencing alice should remain
    assert all("alice" not in members for members in s.teams.values())


# ---- team operations ----------------------------------------------

def test_create_team_moves_creator_out_of_solo():
    s = _new_session_with("Alice", "Bob", "Carol")
    s.enter_team_lobby()
    tid = s.create_team("Storm Chasers", creator_id="alice")
    assert tid in s.teams
    # Alice should be in exactly one team (the new one), not also in solo:alice
    alice_teams = [t for t, m in s.teams.items() if "alice" in m]
    assert alice_teams == [tid]


def test_join_team_moves_player_no_ghost_solo():
    s = _new_session_with("Alice", "Bob")
    s.enter_team_lobby()
    tid = s.create_team("Storm Chasers", creator_id="alice")
    s.join_team("bob", tid)
    assert s.teams[tid] == ["alice", "bob"]
    bob_teams = [t for t, m in s.teams.items() if "bob" in m]
    assert bob_teams == [tid]


def test_leave_team_returns_to_solo():
    s = _new_session_with("Alice", "Bob")
    s.enter_team_lobby()
    tid = s.create_team("Storm Chasers", creator_id="alice")
    s.join_team("bob", tid)
    s.leave_team("alice")
    # Alice back in a solo team
    alice_teams = [t for t, m in s.teams.items() if "alice" in m]
    assert len(alice_teams) == 1
    assert alice_teams[0].startswith(SOLO_TEAM_PREFIX)
    # Storm Chasers still has Bob
    assert s.teams[tid] == ["bob"]


def test_empty_team_auto_deleted():
    s = _new_session_with("Alice")
    s.enter_team_lobby()
    tid = s.create_team("Solo Squad", creator_id="alice")
    s.leave_team("alice")
    assert tid not in s.teams


# ---- round setup + state transitions ------------------------------

def _make_config(*, mode=RoundMode.HISTORICAL):
    return RoundConfig(
        convective_day_12z=_T0.replace(hour=12),
        game_polygon=_POLY,
        radar_sites=["KTLX"], time_start=_T0, time_end=_T0+timedelta(hours=2),
        save_replay=False, team_mode=False, mode=mode,
    )


def test_full_state_machine_walks_through():
    s = _new_session_with("Alice")
    s.freeze_roster()
    assert s.state == SessionState.SETUP
    s.set_round(RoundDay(_T0, [], {}, False), _make_config())
    s.begin_prefetch()
    assert s.state == SessionState.PREFETCH
    s.begin_play()
    assert s.state == SessionState.PLAYING
    assert s.clock is not None


def test_round_mode_recorded():
    s = _new_session_with("Alice")
    s.freeze_roster()
    s.set_round(RoundDay(_T0, [], {}, False), _make_config(mode=RoundMode.LIVE))
    assert s.round_config.mode == RoundMode.LIVE


def test_begin_play_filters_reports_to_polygon():
    s = _new_session_with("Alice")
    s.freeze_roster()
    inside = Report(time=_T0, lat=35.0, lon=-97.5, category="hail", magnitude=1.5,
                    state="OK", county="", remark="", injuries=0, fatalities=0, source="IEM")
    outside = Report(time=_T0, lat=40.0, lon=-90.0, category="hail", magnitude=1.5,
                     state="MO", county="", remark="", injuries=0, fatalities=0, source="IEM")
    s.set_round(RoundDay(_T0, [inside, outside], {}, False), _make_config())
    s.begin_prefetch(); s.begin_play()
    # cached_reports filtered to those inside the game polygon
    assert inside in s.cached_reports
    assert outside not in s.cached_reports


# ---- warning issuance & revisions ----------------------------------

def test_issue_warning_attaches_team_id():
    s = _new_session_with("Alice")
    s.freeze_roster()
    s.set_round(RoundDay(_T0, [], {}, False), _make_config())
    s.begin_prefetch(); s.begin_play()
    w = s.issue_warning(
        player_id="alice", warning_type=WarningType.TOR,
        polygon=_POLY, duration=timedelta(minutes=30), magnitudes=Magnitudes(),
    )
    assert w.team_id == s.players["alice"].team_id


def test_issue_warning_with_explicit_id_used_for_replication():
    """When peers reconstruct a warning from the wire, the ID must be preserved."""
    s = _new_session_with("Alice")
    s.freeze_roster()
    s.set_round(RoundDay(_T0, [], {}, False), _make_config())
    s.begin_prefetch(); s.begin_play()
    w = s.issue_warning(
        player_id="alice", warning_type=WarningType.TOR,
        polygon=_POLY, duration=timedelta(minutes=30), magnitudes=Magnitudes(),
        warning_id="custom-id-123",
    )
    assert w.warning_id == "custom-id-123"


def test_revise_warning_appends_revision():
    s = _new_session_with("Alice")
    s.freeze_roster()
    s.set_round(RoundDay(_T0, [], {}, False), _make_config())
    s.begin_prefetch(); s.begin_play()
    w = s.issue_warning(
        player_id="alice", warning_type=WarningType.SVR,
        polygon=_POLY, duration=timedelta(minutes=30),
        magnitudes=Magnitudes(hail_in=1.0, wind_mph=60),
    )
    s.revise_warning(
        warning_id=w.warning_id, player_id="alice",
        warning_type=WarningType.SVRC,
        magnitudes=Magnitudes(hail_in=2.0, wind_mph=80),
    )
    assert len(w.revisions) == 2
    assert w.revisions[-1].warning_type == WarningType.SVRC


def test_cancel_warning_sets_canceled_at():
    s = _new_session_with("Alice")
    s.freeze_roster()
    s.set_round(RoundDay(_T0, [], {}, False), _make_config())
    s.begin_prefetch(); s.begin_play()
    w = s.issue_warning(
        player_id="alice", warning_type=WarningType.TOR,
        polygon=_POLY, duration=timedelta(minutes=30), magnitudes=Magnitudes(),
    )
    s.cancel_warning(warning_id=w.warning_id, player_id="alice")
    assert w.canceled_at is not None


# ---- reroll guard --------------------------------------------------

def test_reroll_refuses_non_random_day():
    s = _new_session_with("Alice")
    s.round_day = RoundDay(_T0, [], {}, is_random=False)
    s.state = SessionState.SETUP
    with pytest.raises(RuntimeError, match="not.*reroll"):
        s.reroll_random_day(spec=None)


def test_reroll_refuses_post_setup():
    s = _new_session_with("Alice")
    s.round_day = RoundDay(_T0, [], {}, is_random=True)
    s.state = SessionState.PREFETCH
    with pytest.raises(RuntimeError):
        s.reroll_random_day(spec=None)
