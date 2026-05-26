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

    def _apply_team_create(self, msg: proto.TeamCreate) -> None:
        self.session.teams.setdefault(msg.team_id, [])
        self.session.team_names[msg.team_id] = msg.name
        if msg.creator_id in self.session.players:
            self.session.join_team(msg.creator_id, msg.team_id)

    def _apply_team_join(self, msg: proto.TeamJoin) -> None:
        if msg.player_id in self.session.players and msg.team_id in self.session.teams:
            self.session.join_team(msg.player_id, msg.team_id)

    def _apply_team_leave(self, msg: proto.TeamLeave) -> None:
        if msg.player_id in self.session.players:
            self.session.leave_team(msg.player_id)

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
            magnitudes=Magnitudes(hail_in=msg.hail_in, wind_mph=msg.wind_mph, ef=msg.ef),
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
                magnitudes=Magnitudes(hail_in=msg.hail_in, wind_mph=msg.wind_mph, ef=msg.ef),
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
        transport.on_message = self._on_host_message

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
