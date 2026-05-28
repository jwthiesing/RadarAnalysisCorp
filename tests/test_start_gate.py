"""Unit tests for the pre-game start-gate (plan §10).

The gate logic lives in :class:`MultiplayerHost` (host side, decides
when ≥75% of clients are ready and runs the countdown) and
:class:`MultiplayerPeer` (peer side, signals ready and reacts to the
countdown). These tests use a fake transport that captures broadcast
messages without actually opening WebRTC connections.
"""

from __future__ import annotations

import asyncio

import pytest

from radar_warning_game.game.session import GameSession
from radar_warning_game.net import protocol as proto
from radar_warning_game.net.multiplayer import MultiplayerHost, MultiplayerPeer


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


def _make_host(peer_ids: list[str]) -> MultiplayerHost:
    return MultiplayerHost(GameSession(), _FakeHostTransport(peer_ids))


def test_no_peers_starts_immediately(event_loop):
    """Host with zero peers: mark_host_ready fires on_round_start
    synchronously — no countdown, since there's nothing to wait on."""
    h = _make_host([])
    fired = []
    h.on_round_start = lambda: fired.append(True)
    h.mark_host_ready()
    assert fired == [True]
    # No broadcasts — the gate path took the shortcut.
    assert h.transport.broadcasts == []


def test_threshold_for_two_clients_requires_both(event_loop):
    """1 host + 1 peer = 2 clients; ceil(0.75 * 2) = 2. The host being
    ready alone shouldn't kick off the countdown."""
    h = _make_host(["peer-1"])
    started = []
    h.on_round_start = lambda: started.append(True)
    h.mark_host_ready()
    # Threshold not met → no countdown task, no broadcasts, no fire.
    assert h._countdown_task is None
    assert started == []
    # Peer reports ready → threshold met.
    h._apply_peer_ready(proto.PeerReady(player_id="peer-1"))
    assert h._countdown_task is not None


def test_threshold_for_four_clients_at_three_ready(event_loop):
    """1 host + 3 peers = 4 clients; ceil(0.75 * 4) = 3. Two peers
    ready (plus host) = 3 → countdown kicks off even with one slow
    client outstanding."""
    h = _make_host(["a", "b", "c"])
    h.mark_host_ready()
    h._apply_peer_ready(proto.PeerReady(player_id="a"))
    assert h._countdown_task is None       # only 2 of 4 ready (host + a)
    h._apply_peer_ready(proto.PeerReady(player_id="b"))
    assert h._countdown_task is not None    # 3 of 4 ready → fire


def test_peer_leaves_can_unblock_threshold(event_loop):
    """Two peers connected, one is ready, host is ready. A third peer
    that never reported ready disconnects → 1+1=2 ready of 2 remaining
    clients → threshold (ceil(0.75*2)=2) is met. The departing peer
    pushes the gate over the line."""
    h = _make_host(["ready", "deadbeat"])
    h.mark_host_ready()
    h._apply_peer_ready(proto.PeerReady(player_id="ready"))
    assert h._countdown_task is None       # 2/3 ready, need 3 (ceil 2.25)
    # Deadbeat disconnects.
    h.transport._peer_ids.remove("deadbeat")
    h._on_peer_left("deadbeat")
    assert h._countdown_task is not None


def test_double_peer_ready_is_idempotent(event_loop):
    """Two PeerReady from the same peer shouldn't double-count toward
    the threshold."""
    h = _make_host(["a", "b", "c"])
    h.mark_host_ready()
    h._apply_peer_ready(proto.PeerReady(player_id="a"))
    h._apply_peer_ready(proto.PeerReady(player_id="a"))   # duplicate
    assert h._countdown_task is None        # still 2 of 4 ready


def test_countdown_broadcasts_and_fires_on_zero(event_loop):
    """With stragglers — threshold reached at <100% ready — the host
    runs the full countdown and fires on_round_start at zero."""
    h = _make_host(["a", "b", "c"])
    started = []
    h.on_round_start = lambda: started.append(True)
    h.COUNTDOWN_SECONDS = 1
    h.mark_host_ready()
    # 3 of 4 ready (host + a + b) → 75% threshold, "c" is the straggler.
    h._apply_peer_ready(proto.PeerReady(player_id="a"))
    h._apply_peer_ready(proto.PeerReady(player_id="b"))
    event_loop.run_until_complete(h._countdown_task)
    seen_seconds = [
        m.seconds_remaining for m in h.transport.broadcasts
        if isinstance(m, proto.RoundCountdown)
    ]
    # Full countdown should include both the 1 and the trailing 0.
    assert seen_seconds == [1, 0]
    assert started == [True]


def test_full_readiness_skips_countdown(event_loop):
    """100%-ready (no stragglers) → no point waiting 60s. Host should
    broadcast a single RoundCountdown(0) start signal and fire
    on_round_start immediately."""
    h = _make_host(["peer-1"])
    started = []
    h.on_round_start = lambda: started.append(True)
    h.mark_host_ready()
    h._apply_peer_ready(proto.PeerReady(player_id="peer-1"))
    event_loop.run_until_complete(h._countdown_task)
    seen_seconds = [
        m.seconds_remaining for m in h.transport.broadcasts
        if isinstance(m, proto.RoundCountdown)
    ]
    assert seen_seconds == [0]
    assert started == [True]


def test_force_start_round_broadcasts_zero_and_fires(event_loop):
    """``force_start_round`` (host's 'Start anyway' button) cancels
    whatever the gate was doing and sends RoundCountdown(0) so peers
    enter play in lockstep with the host."""
    h = _make_host(["slow-peer"])
    started = []
    h.on_round_start = lambda: started.append(True)
    # Host is ready but the peer never reported — countdown not started.
    h.mark_host_ready()
    assert h._countdown_task is None
    h.force_start_round()
    event_loop.run_until_complete(h._countdown_task)
    seen_seconds = [
        m.seconds_remaining for m in h.transport.broadcasts
        if isinstance(m, proto.RoundCountdown)
    ]
    assert seen_seconds == [0]
    assert started == [True]


def test_force_start_round_cancels_in_progress_countdown(event_loop):
    """If a slow countdown is already running when the host hits
    'Start anyway', the in-flight countdown is cancelled and replaced
    by the fast 0-tick path."""
    h = _make_host(["a", "b", "c"])
    started = []
    h.on_round_start = lambda: started.append(True)
    h.COUNTDOWN_SECONDS = 5
    h.mark_host_ready()
    h._apply_peer_ready(proto.PeerReady(player_id="a"))
    h._apply_peer_ready(proto.PeerReady(player_id="b"))
    # Slow countdown is now running.
    slow = h._countdown_task
    h.force_start_round()
    event_loop.run_until_complete(h._countdown_task)
    assert slow.cancelled() or slow.done()
    assert started == [True]


# ---- peer side ----------------------------------------------------------

def test_peer_sends_ready_only_once(event_loop):
    sess = GameSession()
    transport = _FakeClientTransport(peer_id="peer-1")
    mp = MultiplayerPeer(sess, transport)
    mp.mark_peer_ready()
    mp.mark_peer_ready()
    event_loop.run_until_complete(asyncio.sleep(0))
    ready_msgs = [m for m in transport.sent if isinstance(m, proto.PeerReady)]
    assert len(ready_msgs) == 1
    assert ready_msgs[0].player_id == "peer-1"


def test_peer_countdown_callback_and_start(event_loop):
    sess = GameSession()
    transport = _FakeClientTransport(peer_id="peer-1")
    mp = MultiplayerPeer(sess, transport)
    counts: list[int] = []
    started = []
    mp.on_countdown = counts.append
    mp.on_round_start = lambda: started.append(True)
    mp._apply_round_countdown(proto.RoundCountdown(seconds_remaining=3))
    mp._apply_round_countdown(proto.RoundCountdown(seconds_remaining=2))
    mp._apply_round_countdown(proto.RoundCountdown(seconds_remaining=0))
    assert counts == [3, 2, 0]
    assert started == [True]
    # A late duplicate 0 must not re-fire on_round_start.
    mp._apply_round_countdown(proto.RoundCountdown(seconds_remaining=0))
    assert started == [True]


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()
