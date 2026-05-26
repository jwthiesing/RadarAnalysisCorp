"""Unit tests for the _SessionApplier mixin in multiplayer.py.

These tests verify the **state-reconstruction** logic: given a wire message,
apply it to a fresh GameSession and check the result. No networking required.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from radar_warning_game.game.clock import GameClock
from radar_warning_game.game.session import GameSession, Player
from radar_warning_game.net import protocol as proto
from radar_warning_game.net.multiplayer import _SessionApplier, _payload_poly


_T0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
# Warning/MCD polygon (~890 km²) — well above the 200 km² MCD minimum.
_POLY = [[35.1, -97.4], [35.1, -97.1], [35.4, -97.1], [35.4, -97.4]]
# Game polygon — significantly larger so an MCD covering _POLY is <50% of game area.
_GAME_POLY = [[34.5, -98.0], [34.5, -96.5], [36.0, -96.5], [36.0, -98.0]]


class _Harness(_SessionApplier):
    """Bare-bones applier for testing in isolation."""

    def __init__(self) -> None:
        self.session = GameSession()
        self.round_epoch: datetime | None = None


def _harness_with_setup():
    h = _Harness()
    h.session.add_player(Player(player_id="alice", display_name="Alice", is_host=True))
    msg = proto.RoundSetup(
        convective_day_12z_iso=_T0.replace(hour=12).isoformat(),
        time_start_iso=_T0.isoformat(),
        time_end_iso=(_T0+timedelta(hours=2)).isoformat(),
        game_polygon_latlon=_GAME_POLY, radar_sites=["KTLX"],
        team_mode=False, save_replay=False,
    )
    h.apply_wire_message(msg)
    h.session.clock = GameClock(_T0, _T0+timedelta(hours=2))
    return h


# ---- camelCase → snake_case dispatch -------------------------------

def test_dispatch_converts_camelcase_to_snake():
    """RoundSetup must dispatch to _apply_round_setup, not _apply_roundsetup."""
    h = _harness_with_setup()
    assert h.session.round_config is not None
    assert h.session.round_config.radar_sites == ["KTLX"]


# ---- RoundSetup ----------------------------------------------------

def test_round_setup_reconstructs_polygon_and_sites():
    h = _harness_with_setup()
    cfg = h.session.round_config
    assert len(cfg.game_polygon.vertices) == 4
    assert cfg.radar_sites == ["KTLX"]
    assert cfg.time_start == _T0
    assert cfg.time_end == _T0 + timedelta(hours=2)


def test_round_setup_sets_round_epoch():
    h = _harness_with_setup()
    assert h.round_epoch == _T0


# ---- Tick ----------------------------------------------------------

def test_tick_updates_clock_via_offset():
    h = _harness_with_setup()
    h.apply_wire_message(proto.Tick(virtual_time_offset_sec=900.0, speed=1.0, paused=False))
    assert h.session.clock.virtual_time == _T0 + timedelta(minutes=15)


# ---- PlayerJoin / Leave --------------------------------------------

def test_player_join_adds_player():
    h = _harness_with_setup()
    h.apply_wire_message(proto.PlayerJoin("bob", "Bob"))
    assert "bob" in h.session.players


def test_player_join_idempotent():
    h = _harness_with_setup()
    h.apply_wire_message(proto.PlayerJoin("bob", "Bob"))
    h.apply_wire_message(proto.PlayerJoin("bob", "Bob"))   # no-op
    assert "bob" in h.session.players


def test_player_leave_removes():
    h = _harness_with_setup()
    h.apply_wire_message(proto.PlayerJoin("bob", "Bob"))
    h.apply_wire_message(proto.PlayerLeave("bob"))
    assert "bob" not in h.session.players


# ---- Team ops ------------------------------------------------------

def test_team_create_join_leave_roundtrip():
    h = _harness_with_setup()
    h.apply_wire_message(proto.PlayerJoin("bob", "Bob"))
    h.apply_wire_message(proto.TeamCreate("team:abc", "Storm Chasers", "alice"))
    h.apply_wire_message(proto.TeamJoin("team:abc", "bob"))
    assert h.session.teams["team:abc"] == ["alice", "bob"]
    h.apply_wire_message(proto.TeamLeave("alice"))
    assert h.session.teams["team:abc"] == ["bob"]


# ---- WarningIssue / Revise / Cancel --------------------------------

def test_warning_issue_apply():
    h = _harness_with_setup()
    msg = proto.WarningIssue(
        warning_id="w1", issuer_id="alice", warning_type="PDS_TOR",
        polygon_latlon=_POLY, duration_sec=1800, issue_offset_sec=300.0,
        ef=3.0,
    )
    h.apply_wire_message(msg)
    warnings = h.session.warnings_by_player.get("alice", [])
    assert len(warnings) == 1
    assert warnings[0].warning_id == "w1"
    # Issue time = round_epoch + 300s = 20:05
    assert warnings[0].original_issue_time == _T0 + timedelta(minutes=5)


def test_warning_issue_idempotent_by_id():
    """Re-applying the same WarningIssue (echo back) must not duplicate."""
    h = _harness_with_setup()
    msg = proto.WarningIssue(
        warning_id="w1", issuer_id="alice", warning_type="TOR",
        polygon_latlon=_POLY, duration_sec=1800, issue_offset_sec=0.0,
    )
    h.apply_wire_message(msg)
    h.apply_wire_message(msg)   # echo
    assert len(h.session.warnings_by_player["alice"]) == 1


def test_warning_revise_appends():
    h = _harness_with_setup()
    h.apply_wire_message(proto.WarningIssue(
        "w1", "alice", "TOR", _POLY, 1800, 0.0,
    ))
    h.apply_wire_message(proto.WarningRevise(
        "w1", "alice", "TORE", _POLY, 1800, 300.0, ef=3.0,
    ))
    w = h.session.warnings_by_player["alice"][0]
    assert len(w.revisions) == 2
    assert w.revisions[-1].warning_type.value == "TORE"


def test_warning_cancel_sets_canceled_at():
    h = _harness_with_setup()
    h.apply_wire_message(proto.WarningIssue(
        "w1", "alice", "TOR", _POLY, 1800, 0.0,
    ))
    h.apply_wire_message(proto.WarningCancel("w1", "alice", 600.0))
    w = h.session.warnings_by_player["alice"][0]
    assert w.canceled_at == _T0 + timedelta(minutes=10)


# ---- MCD -----------------------------------------------------------

def test_mcd_issue_apply():
    h = _harness_with_setup()
    h.apply_wire_message(proto.MCDIssue(
        mcd_id="m1", issuer_id="alice", polygon_latlon=_POLY,
        duration_sec=5400, issue_offset_sec=0.0,
        pib_tornado=4, pib_wind=3, pib_hail=3,
    ))
    mcds = h.session.mcds_by_player["alice"]
    assert len(mcds) == 1
    assert mcds[0].pib_tornado == 4


def test_mcd_issue_idempotent_by_id():
    h = _harness_with_setup()
    msg = proto.MCDIssue("m1", "alice", _POLY, 5400, 0.0, 3, 0, 0)
    h.apply_wire_message(msg)
    h.apply_wire_message(msg)   # echo
    assert len(h.session.mcds_by_player["alice"]) == 1


# ---- offsets require RoundSetup ------------------------------------

def test_apply_without_round_setup_raises_for_offset_msgs():
    h = _Harness()
    h.session.add_player(Player(player_id="alice", display_name="Alice"))
    # No RoundSetup yet → no round_epoch → WarningIssue can't compute time
    msg = proto.WarningIssue("w1", "alice", "TOR", _POLY, 1800, 0.0)
    with pytest.raises(RuntimeError):
        h.apply_wire_message(msg)
