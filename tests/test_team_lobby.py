"""End-to-end tests for the team-lobby feature (plan §11).

Covers both the wire-level appliers (TeamLobbyOpen → state transition,
TeamCreate/Join/Leave → roster mutations, TeamRosterFreeze → lock and
exit) and the outgoing methods on :class:`MultiplayerHost` and
:class:`MultiplayerPeer` that the UI calls.
"""

from __future__ import annotations

import asyncio

import pytest

from radar_warning_game.game.session import (
    GameSession,
    Player,
    SessionState,
)
from radar_warning_game.net import protocol as proto
from radar_warning_game.net.multiplayer import MultiplayerHost, MultiplayerPeer


# ---------------------------- fake transports ----------------------------

class _FakeHostTransport:
    def __init__(self, peer_ids: list[str]) -> None:
        self._peer_ids = list(peer_ids)
        self.broadcasts: list[object] = []
        self.on_message = None
        self.on_peer_joined = None
        self.on_peer_left = None

    @property
    def peer_ids(self) -> list[str]:
        return list(self._peer_ids)

    async def broadcast(self, raw: str) -> None:
        self.broadcasts.append(proto.decode(raw))

    async def send_to(self, peer_id: str, raw: str) -> None:
        self.broadcasts.append(("to", peer_id, proto.decode(raw)))


class _FakeClientTransport:
    def __init__(self, peer_id: str = "peer-1") -> None:
        self._peer_id = peer_id
        self.sent: list[object] = []
        self.on_message = None

    @property
    def peer_id(self) -> str:
        return self._peer_id

    async def send(self, raw: str) -> None:
        self.sent.append(proto.decode(raw))


def _seeded_host_session() -> GameSession:
    sess = GameSession()
    sess.add_player(Player(player_id="host", display_name="Host", is_host=True))
    return sess


# ---------------------------- applier-level ----------------------------

def test_team_lobby_open_transitions_state():
    """Receiving TeamLobbyOpen on a fresh peer session moves LOBBY →
    TEAM_LOBBY and fires the ``on_team_lobby_open`` callback."""
    sess = GameSession()
    sess.add_player(Player(player_id="me", display_name="Me"))
    transport = _FakeClientTransport(peer_id="me")
    mp = MultiplayerPeer(sess, transport)
    fired = []
    mp.on_team_lobby_open = lambda: fired.append(True)
    mp.apply_wire_message(proto.TeamLobbyOpen())
    assert sess.state == SessionState.TEAM_LOBBY
    assert fired == [True]


def test_team_lobby_open_idempotent():
    """A duplicate TeamLobbyOpen shouldn't crash the state machine."""
    sess = GameSession()
    sess.add_player(Player(player_id="me", display_name="Me"))
    transport = _FakeClientTransport(peer_id="me")
    mp = MultiplayerPeer(sess, transport)
    mp.apply_wire_message(proto.TeamLobbyOpen())
    mp.apply_wire_message(proto.TeamLobbyOpen())   # no-op
    assert sess.state == SessionState.TEAM_LOBBY


def test_team_roster_freeze_locks_and_exits_lobby():
    """TeamRosterFreeze writes the snapshot AND transitions to SETUP."""
    sess = GameSession()
    sess.add_player(Player(player_id="me", display_name="Me"))
    sess.add_player(Player(player_id="them", display_name="Them"))
    transport = _FakeClientTransport(peer_id="me")
    mp = MultiplayerPeer(sess, transport)
    mp.apply_wire_message(proto.TeamLobbyOpen())
    fired = []
    mp.on_team_roster_freeze = lambda: fired.append(True)
    mp.apply_wire_message(proto.TeamRosterFreeze(
        roster={"team:abc": ["me", "them"]},
        team_names={"team:abc": "Storm Chasers"},
    ))
    assert sess.state == SessionState.SETUP
    assert sess.teams == {"team:abc": ["me", "them"]}
    assert sess.team_names == {"team:abc": "Storm Chasers"}
    assert sess.players["me"].team_id == "team:abc"
    assert sess.players["them"].team_id == "team:abc"
    assert fired == [True]


def test_team_state_changed_fires_on_join_and_leave():
    sess = GameSession()
    sess.add_player(Player(player_id="me", display_name="Me"))
    transport = _FakeClientTransport(peer_id="me")
    mp = MultiplayerPeer(sess, transport)
    mp.apply_wire_message(proto.TeamLobbyOpen())
    bumps = []
    mp.on_team_state_changed = lambda: bumps.append(1)
    mp.apply_wire_message(proto.TeamCreate(
        team_id="team:1", name="Alpha", creator_id="me",
    ))
    mp.apply_wire_message(proto.TeamLeave(player_id="me"))
    assert bumps == [1, 1]


# ---------------------------- host outgoing ----------------------------

def test_host_announce_team_lobby_transitions_and_broadcasts(event_loop):
    sess = _seeded_host_session()
    transport = _FakeHostTransport(["peer-1"])
    mp = MultiplayerHost(sess, transport)
    event_loop.run_until_complete(mp.announce_team_lobby())
    assert sess.state == SessionState.TEAM_LOBBY
    assert any(isinstance(m, proto.TeamLobbyOpen) for m in transport.broadcasts)


def test_host_create_team_mutates_and_broadcasts(event_loop):
    sess = _seeded_host_session()
    transport = _FakeHostTransport(["peer-1"])
    mp = MultiplayerHost(sess, transport)
    event_loop.run_until_complete(mp.announce_team_lobby())
    tid = event_loop.run_until_complete(mp.create_team("Alpha"))
    # Local mutation
    assert tid in sess.teams
    assert sess.team_names[tid] == "Alpha"
    assert sess.players["host"].team_id == tid
    # Broadcast
    creates = [m for m in transport.broadcasts if isinstance(m, proto.TeamCreate)]
    assert len(creates) == 1
    assert creates[0].team_id == tid
    assert creates[0].creator_id == "host"


def test_host_move_player_to_team(event_loop):
    sess = _seeded_host_session()
    sess.add_player(Player(player_id="bob", display_name="Bob"))
    transport = _FakeHostTransport(["bob"])
    mp = MultiplayerHost(sess, transport)
    event_loop.run_until_complete(mp.announce_team_lobby())
    tid = event_loop.run_until_complete(mp.create_team("Alpha"))
    event_loop.run_until_complete(mp.move_player("bob", tid))
    assert "bob" in sess.teams[tid]
    # Move to unassigned
    event_loop.run_until_complete(mp.move_player("bob", ""))
    assert "bob" not in sess.teams.get(tid, [])


def test_host_freeze_roster_broadcasts_snapshot(event_loop):
    sess = _seeded_host_session()
    sess.add_player(Player(player_id="bob", display_name="Bob"))
    transport = _FakeHostTransport(["bob"])
    mp = MultiplayerHost(sess, transport)
    event_loop.run_until_complete(mp.announce_team_lobby())
    tid = event_loop.run_until_complete(mp.create_team("Alpha"))
    event_loop.run_until_complete(mp.move_player("bob", tid))
    event_loop.run_until_complete(mp.broadcast_team_roster_freeze())
    assert sess.state == SessionState.SETUP
    frozen = [m for m in transport.broadcasts if isinstance(m, proto.TeamRosterFreeze)]
    assert len(frozen) == 1
    # The frozen snapshot should preserve the team we just built.
    assert frozen[0].roster.get(tid) == ["host", "bob"]
    assert frozen[0].team_names.get(tid) == "Alpha"


# ---------------------------- peer outgoing ----------------------------

def test_peer_create_team_sends_to_host(event_loop):
    sess = GameSession()
    sess.add_player(Player(player_id="peer-1", display_name="P1"))
    transport = _FakeClientTransport(peer_id="peer-1")
    mp = MultiplayerPeer(sess, transport)
    # The host would have announced the lobby; simulate that on our side.
    mp.apply_wire_message(proto.TeamLobbyOpen())
    tid = event_loop.run_until_complete(mp.create_team("Bravo"))
    # Local effect
    assert tid in sess.teams
    assert sess.players["peer-1"].team_id == tid
    # Wire effect
    sent_creates = [m for m in transport.sent if isinstance(m, proto.TeamCreate)]
    assert len(sent_creates) == 1
    assert sent_creates[0].creator_id == "peer-1"


def test_peer_join_then_leave(event_loop):
    sess = GameSession()
    sess.add_player(Player(player_id="peer-1", display_name="P1"))
    transport = _FakeClientTransport(peer_id="peer-1")
    mp = MultiplayerPeer(sess, transport)
    mp.apply_wire_message(proto.TeamLobbyOpen())
    mp.apply_wire_message(proto.TeamCreate(
        team_id="team:host", name="HostTeam", creator_id="host",
    ))
    # Even though the host isn't a registered player on our session, the
    # team itself exists; we should be able to join it.
    event_loop.run_until_complete(mp.join_team("team:host"))
    assert "peer-1" in sess.teams["team:host"]
    event_loop.run_until_complete(mp.leave_team())
    assert "peer-1" not in sess.teams.get("team:host", [])
    join_msgs = [m for m in transport.sent if isinstance(m, proto.TeamJoin)]
    leave_msgs = [m for m in transport.sent if isinstance(m, proto.TeamLeave)]
    assert len(join_msgs) == 1
    assert len(leave_msgs) == 1


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
