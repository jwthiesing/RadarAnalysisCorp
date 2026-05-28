"""Replicated-session multiplayer orchestrator (plan §10).

Each client (host + every peer) maintains its **own** :class:`GameSession`.
Local mutations are applied locally first then broadcast over WebRTC; incoming
messages from peers are applied to the local session so all clients converge
on the same state. The host is the single source of truth for clock ticks and
the round setup; peer mutations (warning/MCD issues) flow through the host as
a relay (and the host applies them too, so its session matches).

Two facades over the same idea:
  - :class:`MultiplayerHost` — wraps :class:`HostTransport`. Re-broadcasts peer
    mutations to other peers so everyone sees them.
  - :class:`MultiplayerPeer` — wraps :class:`ClientTransport`. Sends own
    mutations to the host; applies received mutations (including its own
    echoed back via the host) to its local session, deduplicating by ID.

Both expose ``issue_warning`` / ``revise_warning`` / ``cancel_warning`` /
``issue_mcd`` methods that take the same kwargs as :class:`GameSession`. The
caller is the UI layer (PlayView) — it calls these instead of the session
directly when multiplayer is on.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from ..game.clock import TickState
from ..game.session import GameSession, Player, RoundConfig, SOLO_TEAM_PREFIX
from ..geo.polygons import Polygon
from ..verification.reports_in_poly import MCD, Magnitudes, Warning, WarningRevision
from ..verification.tornado_tiers import WarningType
from . import protocol as proto
from .peer import ClientTransport, HostTransport

log = logging.getLogger(__name__)


def _payload_poly(payload: list[list[float]]) -> Polygon:
    return Polygon(vertices=tuple((float(p[0]), float(p[1])) for p in payload))


# ----------------------------- shared mixin ------------------------------

class _SessionApplier:
    """Mixin: apply wire messages to a local GameSession."""

    session: GameSession
    round_epoch: datetime | None

    def _offset_to_dt(self, offset: float) -> datetime:
        if self.round_epoch is None:
            raise RuntimeError("round_epoch not set — RoundSetup not received yet")
        return self.round_epoch + timedelta(seconds=float(offset))

    def _apply_round_setup(self, msg: proto.RoundSetup) -> None:
        from datetime import datetime as _dt
        cday = _dt.fromisoformat(msg.convective_day_12z_iso)
        if cday.tzinfo is None:
            cday = cday.replace(tzinfo=timezone.utc)
        tstart = _dt.fromisoformat(msg.time_start_iso)
        if tstart.tzinfo is None:
            tstart = tstart.replace(tzinfo=timezone.utc)
        tend = _dt.fromisoformat(msg.time_end_iso)
        if tend.tzinfo is None:
            tend = tend.replace(tzinfo=timezone.utc)
        self.round_epoch = tstart
        from ..game.round_builder import RoundDay
        self.session.round_day = RoundDay(
            convective_day_12z=cday, reports=[], counts={}, is_random=False,
        )
        self.session.round_config = RoundConfig(
            convective_day_12z=cday,
            game_polygon=_payload_poly(msg.game_polygon_latlon),
            radar_sites=list(msg.radar_sites),
            time_start=tstart, time_end=tend,
            save_replay=bool(msg.save_replay),
            team_mode=bool(msg.team_mode),
        )

    def _apply_tick(self, msg: proto.Tick) -> None:
        if self.session.clock is None or self.round_epoch is None:
            return
        vt = self._offset_to_dt(msg.virtual_time_offset_sec)
        self.session.clock.apply_tick(
            TickState(virtual_time=vt, speed=msg.speed, paused=msg.paused)
        )

    def _apply_player_join(self, msg: proto.PlayerJoin) -> None:
        if msg.player_id not in self.session.players:
            self.session.add_player(
                Player(player_id=msg.player_id, display_name=msg.display_name)
            )

    def _apply_player_leave(self, msg: proto.PlayerLeave) -> None:
        self.session.remove_player(msg.player_id)

    def _fire_team_changed(self) -> None:
        """Notify the UI that the team roster changed. Subclasses set
        ``on_team_state_changed`` to a no-arg callable; the base mixin
        leaves it as None so unit tests of the applier work without
        wiring the hook."""
        cb = getattr(self, "on_team_state_changed", None)
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.exception("on_team_state_changed callback raised")

    def _apply_team_lobby_open(self, msg: proto.TeamLobbyOpen) -> None:
        # Idempotent — if a duplicate or out-of-order TeamLobbyOpen
        # arrives we just stay in TEAM_LOBBY without re-transitioning.
        from ..game.session import SessionState
        if self.session.state == SessionState.LOBBY:
            self.session.enter_team_lobby()
        cb = getattr(self, "on_team_lobby_open", None)
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.exception("on_team_lobby_open callback raised")

    def _apply_team_create(self, msg: proto.TeamCreate) -> None:
        self.session.teams.setdefault(msg.team_id, [])
        self.session.team_names[msg.team_id] = msg.name
        if msg.creator_id in self.session.players:
            self.session.join_team(msg.creator_id, msg.team_id)
        self._fire_team_changed()

    def _apply_team_join(self, msg: proto.TeamJoin) -> None:
        if msg.player_id in self.session.players and msg.team_id in self.session.teams:
            self.session.join_team(msg.player_id, msg.team_id)
        self._fire_team_changed()

    def _apply_team_leave(self, msg: proto.TeamLeave) -> None:
        if msg.player_id in self.session.players:
            self.session.leave_team(msg.player_id)
        self._fire_team_changed()

    def _apply_team_roster_freeze(self, msg: proto.TeamRosterFreeze) -> None:
        self.session.teams.clear()
        self.session.team_names.clear()
        for tid, members in msg.roster.items():
            self.session.teams[tid] = list(members)
            self.session.team_names[tid] = msg.team_names.get(tid, tid)
            for pid in members:
                if pid in self.session.players:
                    self.session.players[pid].team_id = tid
        self.session.freeze_roster()
        self._fire_team_changed()
        cb = getattr(self, "on_team_roster_freeze", None)
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.exception("on_team_roster_freeze callback raised")

    def _apply_warning_issue(self, msg: proto.WarningIssue) -> None:
        # Skip if we already have this warning (echo of our own send)
        for w in self.session.warnings_by_player.get(msg.issuer_id, []):
            if w.warning_id == msg.warning_id:
                return
        self.session.issue_warning(
            player_id=msg.issuer_id,
            warning_type=WarningType(msg.warning_type),
            polygon=_payload_poly(msg.polygon_latlon),
            duration=timedelta(seconds=msg.duration_sec),
            magnitudes=Magnitudes(
                hail_in=msg.hail_in, wind_mph=msg.wind_mph, ef=msg.ef,
                tornado_possible=getattr(msg, "tornado_possible", False),
            ),
            warning_id=msg.warning_id,
            issue_time=self._offset_to_dt(msg.issue_offset_sec),
        )

    def _apply_warning_revise(self, msg: proto.WarningRevise) -> None:
        for w in self.session.warnings_by_player.get(msg.issuer_id, []):
            if w.warning_id != msg.warning_id:
                continue
            rev_time = self._offset_to_dt(msg.revision_offset_sec)
            # Avoid duplicating an echoed revision we already have
            if any(abs((r.revision_time - rev_time).total_seconds()) < 1e-3 for r in w.revisions):
                return
            w.revisions.append(WarningRevision(
                revision_time=rev_time,
                warning_type=WarningType(msg.warning_type),
                polygon=_payload_poly(msg.polygon_latlon),
                duration=timedelta(seconds=msg.duration_sec),
                magnitudes=Magnitudes(
                    hail_in=msg.hail_in, wind_mph=msg.wind_mph, ef=msg.ef,
                    tornado_possible=getattr(msg, "tornado_possible", False),
                ),
            ))
            return

    def _apply_warning_cancel(self, msg: proto.WarningCancel) -> None:
        for w in self.session.warnings_by_player.get(msg.issuer_id, []):
            if w.warning_id == msg.warning_id and w.canceled_at is None:
                w.canceled_at = self._offset_to_dt(msg.cancel_offset_sec)
                return

    def _apply_mcd_issue(self, msg: proto.MCDIssue) -> None:
        for m in self.session.mcds_by_player.get(msg.issuer_id, []):
            if m.mcd_id == msg.mcd_id:
                return
        self.session.issue_mcd(
            player_id=msg.issuer_id,
            polygon=_payload_poly(msg.polygon_latlon),
            duration=timedelta(seconds=msg.duration_sec),
            pib_tornado=msg.pib_tornado, pib_wind=msg.pib_wind, pib_hail=msg.pib_hail,
            mcd_id=msg.mcd_id,
            issue_time=self._offset_to_dt(msg.issue_offset_sec),
        )

    def _apply_mcd_cancel(self, msg: proto.MCDCancel) -> None:
        for m in self.session.mcds_by_player.get(msg.issuer_id, []):
            if m.mcd_id == msg.mcd_id and m.canceled_at is None:
                m.canceled_at = self._offset_to_dt(msg.cancel_offset_sec)
                return

    def apply_wire_message(self, msg) -> None:
        """Single dispatch entry — routes any protocol message to its applier.

        Method names use snake_case (``_apply_round_setup``). Handles
        initialisms correctly: ``MCDIssue`` → ``_apply_mcd_issue`` (not
        ``_apply_m_c_d_issue``).
        """
        import re
        n = type(msg).__name__
        # Two-pass: insert _ between an acronym and a Camel word, then between
        # a lowercase/digit and an uppercase.
        n = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", n)
        n = re.sub(r"([a-z\d])([A-Z])", r"\1_\2", n)
        name = n.lower()
        method = getattr(self, f"_apply_{name}", None)
        if method:
            method(msg)


# ----------------------------- MultiplayerHost ---------------------------

class MultiplayerHost(_SessionApplier):
    """Host-side wrapper: drives the authoritative session + broadcasts.

    When a peer joins mid-round, the host re-sends RoundSetup + a full snapshot
    of active warnings/MCDs/players so the late-joiner converges to the same
    state as everyone else.
    """

    def __init__(self, session: GameSession, transport: HostTransport,
                 *, host_player_id: str = "host") -> None:
        self.session = session
        self.transport = transport
        self.host_player_id = host_player_id
        self.round_epoch: datetime | None = None
        # Outstanding relay tasks; keep refs so the garbage collector doesn't
        # cancel them mid-flight (Python 3.13 warns and may drop messages).
        self._relay_tasks: set[asyncio.Task] = set()
        # Pre-game start gate (plan §10): collect PeerReady votes from
        # connected peers; once threshold reached AND host is ready,
        # broadcast a 60-second RoundCountdown then call
        # ``on_round_start``. Solo / no-peer setups skip the gate and
        # start instantly.
        self._ready_peer_ids: set[str] = set()
        self._host_ready: bool = False
        self._countdown_task: asyncio.Task | None = None
        self._round_started: bool = False
        self.on_round_start: Any = None       # () -> None — set by the UI
        # Team-lobby (plan §11) UI hooks: fired after a wire team op is
        # applied or a local team op is performed. The UI sets these to
        # refresh the team-lobby widget. The host's own actions also
        # call ``_fire_team_changed`` for symmetry so the widget can
        # use a single refresh entry point.
        self.on_team_state_changed: Any = None    # () -> None
        self.on_team_roster_freeze: Any = None    # () -> None
        transport.on_message = self._on_peer_message
        transport.on_peer_joined = self._on_peer_joined
        transport.on_peer_left = self._on_peer_left

    # ---- peer lifecycle ------------------------------------------------

    def _on_peer_joined(self, peer_id: str) -> None:
        """Add the peer to our session, broadcast a PlayerJoin, then catch the
        new peer up with RoundSetup + active state snapshot."""
        if peer_id not in self.session.players:
            self.session.add_player(Player(player_id=peer_id, display_name=peer_id))
        # Tell everyone (including the new peer) that this player joined
        asyncio.ensure_future(self._broadcast(proto.PlayerJoin(
            player_id=peer_id, display_name=peer_id,
        )))
        # Send the new peer a snapshot of everything they missed
        asyncio.ensure_future(self._send_snapshot_to(peer_id))

    def _on_peer_left(self, peer_id: str) -> None:
        if peer_id in self.session.players:
            self.session.remove_player(peer_id)
        # A departing peer can never become "ready" — drop them from the
        # gate's vote set so the remaining clients' ratio isn't dragged
        # down by ghosts. If their absence now pushes us over the
        # threshold, fire the gate check.
        self._ready_peer_ids.discard(peer_id)
        self._maybe_start_countdown()
        asyncio.ensure_future(self._broadcast(proto.PlayerLeave(player_id=peer_id)))

    async def _send_snapshot_to(self, peer_id: str) -> None:
        """Send a fresh RoundSetup + warning/MCD/player snapshot to one peer."""
        if self.session.round_config is None:
            return
        cfg = self.session.round_config
        await self.transport.send_to(peer_id, proto.encode(proto.RoundSetup(
            convective_day_12z_iso=cfg.convective_day_12z.isoformat(),
            time_start_iso=cfg.time_start.isoformat(),
            time_end_iso=cfg.time_end.isoformat(),
            game_polygon_latlon=[[lat, lon] for lat, lon in cfg.game_polygon.vertices],
            radar_sites=list(cfg.radar_sites),
            team_mode=cfg.team_mode, save_replay=cfg.save_replay,
        )))
        # Existing players
        for pid, p in self.session.players.items():
            if pid == peer_id:
                continue
            await self.transport.send_to(peer_id, proto.encode(
                proto.PlayerJoin(player_id=pid, display_name=p.display_name)
            ))
        # Active warnings + MCDs (cancellations are part of the warning record)
        if self.round_epoch:
            for pwarns in self.session.warnings_by_player.values():
                for w in pwarns:
                    await self.transport.send_to(peer_id, proto.encode(
                        proto.warning_issue_from(w, self.round_epoch)))
                    for rev in w.revisions[1:]:
                        # Replay each revision so the joiner has full history
                        await self.transport.send_to(peer_id, proto.encode(
                            proto.WarningRevise(
                                warning_id=w.warning_id, issuer_id=w.issuer_id,
                                warning_type=rev.warning_type.value,
                                polygon_latlon=[[lat, lon] for lat, lon in rev.polygon.vertices],
                                duration_sec=int(rev.duration.total_seconds()),
                                revision_offset_sec=(rev.revision_time - self.round_epoch).total_seconds(),
                                hail_in=rev.magnitudes.hail_in,
                                wind_mph=rev.magnitudes.wind_mph,
                                ef=rev.magnitudes.ef,
                                tornado_possible=rev.magnitudes.tornado_possible,
                            )))
                    if w.canceled_at:
                        await self.transport.send_to(peer_id, proto.encode(
                            proto.WarningCancel(
                                warning_id=w.warning_id, issuer_id=w.issuer_id,
                                cancel_offset_sec=(w.canceled_at - self.round_epoch).total_seconds(),
                            )))
            for pmcds in self.session.mcds_by_player.values():
                for m in pmcds:
                    await self.transport.send_to(peer_id, proto.encode(
                        proto.mcd_issue_from(m, self.round_epoch)))
        # Tick last so the late joiner gets the current clock state
        if self.session.clock is not None and self.round_epoch:
            await self.transport.send_to(peer_id, proto.encode(
                proto.tick_from(self.session.clock.snapshot(), self.round_epoch)))

    async def _broadcast(self, msg) -> None:
        await self.transport.broadcast(proto.encode(msg))

    # ---- host-initiated broadcasts -------------------------------------

    async def announce_round_setup(self) -> None:
        cfg = self.session.round_config
        if cfg is None:
            raise RuntimeError("No round config yet")
        self.round_epoch = cfg.time_start
        msg = proto.RoundSetup(
            convective_day_12z_iso=cfg.convective_day_12z.isoformat(),
            time_start_iso=cfg.time_start.isoformat(),
            time_end_iso=cfg.time_end.isoformat(),
            game_polygon_latlon=[[lat, lon] for lat, lon in cfg.game_polygon.vertices],
            radar_sites=list(cfg.radar_sites),
            team_mode=cfg.team_mode,
            save_replay=cfg.save_replay,
        )
        await self.transport.broadcast(proto.encode(msg))

    # ---- team lobby (plan §11) -----------------------------------------

    async def announce_team_lobby(self) -> None:
        """Tell peers the host has entered the pre-round team lobby.

        Idempotent on receivers — applying TeamLobbyOpen twice is a no-op.
        The local session also transitions if it hasn't already.
        """
        from ..game.session import SessionState
        if self.session.state == SessionState.LOBBY:
            self.session.enter_team_lobby()
        await self.transport.broadcast(proto.encode(proto.TeamLobbyOpen()))

    async def create_team(self, name: str, creator_id: str | None = None) -> str:
        """Create a team locally and broadcast TeamCreate.

        ``creator_id`` defaults to the host's own player id, so a host
        clicking "Create team…" winds up in their own new team.
        """
        creator = creator_id or self.host_player_id
        tid = self.session.create_team(name, creator)
        self._fire_team_changed()
        await self.transport.broadcast(proto.encode(proto.TeamCreate(
            team_id=tid, name=name, creator_id=creator,
        )))
        return tid

    async def join_team(self, team_id: str, player_id: str | None = None) -> None:
        pid = player_id or self.host_player_id
        self.session.join_team(pid, team_id)
        self._fire_team_changed()
        await self.transport.broadcast(proto.encode(proto.TeamJoin(
            team_id=team_id, player_id=pid,
        )))

    async def leave_team(self, player_id: str | None = None) -> None:
        pid = player_id or self.host_player_id
        self.session.leave_team(pid)
        self._fire_team_changed()
        await self.transport.broadcast(proto.encode(proto.TeamLeave(player_id=pid)))

    async def move_player(self, player_id: str, target_team_id: str) -> None:
        """Host admin: move ``player_id`` into ``target_team_id``. An empty
        string for ``target_team_id`` returns the player to unassigned
        (solo)."""
        if not target_team_id:
            await self.leave_team(player_id=player_id)
        else:
            await self.join_team(team_id=target_team_id, player_id=player_id)

    async def broadcast_team_roster_freeze(self) -> None:
        """Snapshot the current roster, lock it, and broadcast to peers.

        Called when the host clicks "Start round (freeze teams)" — pushes
        the session out of TEAM_LOBBY into SETUP and stops further team
        edits.
        """
        roster = {tid: list(members) for tid, members in self.session.teams.items()}
        names = dict(self.session.team_names)
        self.session.freeze_roster()
        await self.transport.broadcast(proto.encode(proto.TeamRosterFreeze(
            roster=roster, team_names=names,
        )))
        self._fire_team_changed()

    # ---- pre-game start gate (plan §10) -------------------------------

    READY_THRESHOLD = 0.75          # fraction of clients needed before countdown
    COUNTDOWN_SECONDS = 60          # length of countdown once threshold hit

    def mark_host_ready(self) -> None:
        """Called by the host UI when its own pre-game prefetch finishes.

        If the threshold is already met (or no peers are connected — solo
        play / nobody-joined-yet host), this also kicks off the countdown
        (or, for the no-peers case, fires ``on_round_start`` immediately).
        """
        if self._host_ready:
            return
        self._host_ready = True
        self._maybe_start_countdown()

    def _apply_peer_ready(self, msg: proto.PeerReady) -> None:
        if msg.player_id in self._ready_peer_ids:
            return
        self._ready_peer_ids.add(msg.player_id)
        self._maybe_start_countdown()

    def _maybe_start_countdown(self) -> None:
        if self._countdown_task is not None or self._round_started:
            return
        if not self._host_ready:
            return
        peer_ids = set(self.transport.peer_ids)
        total_clients = 1 + len(peer_ids)
        # If no peers are connected, skip the gate — there's nothing to
        # coordinate. Host plays solo (or simply starts ahead of any
        # late joiner, which is fine: late joiners get mid-round-join
        # treatment per plan §4).
        if not peer_ids:
            self._round_started = True
            self._fire_round_start()
            return
        ready_peers = self._ready_peer_ids & peer_ids
        ready_count = 1 + len(ready_peers)
        # ``≥75%`` (plan §10). For 2 clients (host+1) this is effectively
        # "both ready" (ceil(1.5) = 2); for 4 it's 3 of 4.
        import math
        need = math.ceil(self.READY_THRESHOLD * total_clients)
        if ready_count >= need:
            self._countdown_task = asyncio.ensure_future(self._run_countdown())

    async def _run_countdown(self) -> None:
        """Broadcast RoundCountdown(60), (59), ..., (0), then call
        on_round_start. Always sends the trailing 0 so peers have an
        unambiguous start signal."""
        try:
            for n in range(self.COUNTDOWN_SECONDS, -1, -1):
                await self.transport.broadcast(
                    proto.encode(proto.RoundCountdown(seconds_remaining=n))
                )
                if n == 0:
                    break
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            return
        self._round_started = True
        self._fire_round_start()

    def _fire_round_start(self) -> None:
        cb = self.on_round_start
        if cb is not None:
            try:
                cb()
            except Exception:  # noqa: BLE001
                log.exception("on_round_start callback raised")

    async def broadcast_tick(self, tick: TickState) -> None:
        if self.round_epoch is None:
            return
        await self.transport.broadcast(
            proto.encode(proto.tick_from(tick, self.round_epoch))
        )

    async def issue_warning(self, **kwargs) -> Warning:
        w = self.session.issue_warning(**kwargs)
        if self.round_epoch:
            await self.transport.broadcast(
                proto.encode(proto.warning_issue_from(w, self.round_epoch))
            )
        return w

    async def revise_warning(self, **kwargs) -> Warning:
        w = self.session.revise_warning(**kwargs)
        if self.round_epoch:
            await self.transport.broadcast(
                proto.encode(proto.warning_revise_from(w, self.round_epoch))
            )
        return w

    async def cancel_warning(self, *, warning_id: str, player_id: str) -> None:
        self.session.cancel_warning(warning_id=warning_id, player_id=player_id)
        if self.round_epoch and self.session.clock:
            from .protocol import WarningCancel
            offset = (self.session.clock.virtual_time - self.round_epoch).total_seconds()
            await self.transport.broadcast(proto.encode(
                WarningCancel(warning_id=warning_id, issuer_id=player_id, cancel_offset_sec=offset)
            ))

    async def issue_mcd(self, **kwargs) -> MCD:
        m = self.session.issue_mcd(**kwargs)
        if self.round_epoch:
            await self.transport.broadcast(
                proto.encode(proto.mcd_issue_from(m, self.round_epoch))
            )
        return m

    # ---- peer-message handler ------------------------------------------

    def _on_peer_message(self, peer_id: str, raw: str) -> None:
        """Called by HostTransport when a peer sends us anything. Apply locally
        AND re-broadcast to other peers so everyone converges.
        """
        try:
            msg = proto.decode(raw)
        except (ValueError, KeyError) as e:
            log.warning("Bad peer message from %s: %s", peer_id, e)
            return
        # Apply to host's local session, catching validation errors
        try:
            self.apply_wire_message(msg)
        except Exception as e:  # noqa: BLE001
            log.warning("Peer %s sent invalid %s: %s",
                        peer_id, type(msg).__name__, e)
            return
        # Relay to every other peer (so they see this peer's action). Keep a
        # ref to the task so it doesn't get garbage-collected mid-flight.
        task = asyncio.create_task(self._relay_to_others(peer_id, raw))
        self._relay_tasks.add(task)
        task.add_done_callback(self._relay_tasks.discard)

    async def _relay_to_others(self, exclude_peer: str, raw: str) -> None:
        for pid in self.transport.peer_ids:
            if pid != exclude_peer:
                await self.transport.send_to(pid, raw)


# ----------------------------- MultiplayerPeer ---------------------------

class MultiplayerPeer(_SessionApplier):
    """Non-host wrapper: applies host's authoritative session state, sends own mutations."""

    def __init__(self, session: GameSession, transport: ClientTransport) -> None:
        self.session = session
        self.transport = transport
        self.round_epoch: datetime | None = None
        # Pre-game start-gate hooks for the peer side. The peer UI calls
        # ``mark_peer_ready`` once its own prefetch finishes; later the
        # host broadcasts RoundCountdown messages, which we surface via
        # ``on_countdown`` (per-tick) and ``on_round_start`` (when the
        # 0-tick arrives).
        self.on_countdown: Any = None         # (seconds_remaining: int) -> None
        self.on_round_start: Any = None       # () -> None
        # Team-lobby hooks (plan §11). The UI uses these to swap to /
        # from the TeamLobbyWidget and refresh its roster.
        self.on_team_lobby_open: Any = None       # () -> None
        self.on_team_roster_freeze: Any = None    # () -> None
        self.on_team_state_changed: Any = None    # () -> None
        self._sent_ready: bool = False
        self._round_started: bool = False
        transport.on_message = self._on_host_message

    # ---- team lobby (plan §11) -----------------------------------------

    async def create_team(self, name: str) -> str:
        """Locally create the team and send TeamCreate to the host.

        The host re-broadcasts to other peers and applies to its own
        session, which echoes back to us — but our applier is keyed on
        ``team_id`` and is idempotent, so the echo is harmless.
        """
        if self.transport.peer_id is None:
            raise RuntimeError("Peer not yet assigned an id (joined?)")
        tid = self.session.create_team(name, self.transport.peer_id)
        self._fire_team_changed()
        await self.transport.send(proto.encode(proto.TeamCreate(
            team_id=tid, name=name, creator_id=self.transport.peer_id,
        )))
        return tid

    async def join_team(self, team_id: str) -> None:
        if self.transport.peer_id is None:
            raise RuntimeError("Peer not yet assigned an id")
        self.session.join_team(self.transport.peer_id, team_id)
        self._fire_team_changed()
        await self.transport.send(proto.encode(proto.TeamJoin(
            team_id=team_id, player_id=self.transport.peer_id,
        )))

    async def leave_team(self) -> None:
        if self.transport.peer_id is None:
            raise RuntimeError("Peer not yet assigned an id")
        self.session.leave_team(self.transport.peer_id)
        self._fire_team_changed()
        await self.transport.send(proto.encode(proto.TeamLeave(
            player_id=self.transport.peer_id,
        )))

    def mark_peer_ready(self) -> None:
        """Tell the host our local prefetch is done. Idempotent — calling
        twice does nothing. The peer keeps showing its prefetch / waiting
        screen until the host's RoundCountdown reaches 0.
        """
        if self._sent_ready:
            return
        self._sent_ready = True
        asyncio.ensure_future(self.transport.send(
            proto.encode(proto.PeerReady(player_id=self.transport.peer_id))
        ))

    def _apply_round_countdown(self, msg: proto.RoundCountdown) -> None:
        if self._round_started:
            return
        n = int(msg.seconds_remaining)
        cb = self.on_countdown
        if cb is not None:
            try:
                cb(n)
            except Exception:  # noqa: BLE001
                log.exception("on_countdown callback raised")
        if n <= 0:
            self._round_started = True
            cb2 = self.on_round_start
            if cb2 is not None:
                try:
                    cb2()
                except Exception:  # noqa: BLE001
                    log.exception("on_round_start callback raised")

    async def issue_warning(self, **kwargs) -> Warning:
        w = self.session.issue_warning(**kwargs)
        if self.round_epoch:
            await self.transport.send(
                proto.encode(proto.warning_issue_from(w, self.round_epoch))
            )
        return w

    async def revise_warning(self, **kwargs) -> Warning:
        w = self.session.revise_warning(**kwargs)
        if self.round_epoch:
            await self.transport.send(
                proto.encode(proto.warning_revise_from(w, self.round_epoch))
            )
        return w

    async def cancel_warning(self, *, warning_id: str, player_id: str) -> None:
        self.session.cancel_warning(warning_id=warning_id, player_id=player_id)
        if self.round_epoch and self.session.clock:
            from .protocol import WarningCancel
            offset = (self.session.clock.virtual_time - self.round_epoch).total_seconds()
            await self.transport.send(proto.encode(
                WarningCancel(warning_id=warning_id, issuer_id=player_id, cancel_offset_sec=offset)
            ))

    async def issue_mcd(self, **kwargs) -> MCD:
        m = self.session.issue_mcd(**kwargs)
        if self.round_epoch:
            await self.transport.send(
                proto.encode(proto.mcd_issue_from(m, self.round_epoch))
            )
        return m

    def _on_host_message(self, peer_id: str, raw: str) -> None:
        try:
            msg = proto.decode(raw)
        except (ValueError, KeyError) as e:
            log.warning("Bad host message: %s", e)
            return
        self.apply_wire_message(msg)
