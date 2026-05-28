"""WebRTC DataChannel message schema (plan §10).

All messages are JSON dicts with a top-level ``"type"`` field discriminating the
variant. Each variant maps to a dataclass below; :func:`encode` / :func:`decode`
convert between dict ↔ dataclass.

Date-blinding in the protocol: tick + warning + MCD timestamps are sent as
seconds-since-``round_epoch`` offsets, where ``round_epoch`` is the round's
``time_start`` (sent in :class:`RoundSetup`). The convective day IS sent in
RoundSetup so peers can construct S3 keys; the UI never displays it.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from ..verification.tornado_tiers import WarningType


# ---------------------------- helpers -----------------------------------------

def _dt_to_offset(dt: datetime, epoch: datetime) -> float:
    return (dt - epoch).total_seconds()


def _offset_to_dt(offset: float, epoch: datetime) -> datetime:
    return epoch + timedelta(seconds=offset)


def _poly_payload(vertices) -> list[list[float]]:
    return [[lat, lon] for lat, lon in vertices]


def _payload_poly(payload: list[list[float]]):
    from ..geo.polygons import Polygon
    return Polygon(vertices=tuple((float(p[0]), float(p[1])) for p in payload))


# ---------------------------- message dataclasses -----------------------------

@dataclass(frozen=True)
class RoundSetup:
    convective_day_12z_iso: str          # full ISO date — clients need for S3
    time_start_iso: str                  # start of round in UTC (epoch for offsets)
    time_end_iso: str                    # end of round (UTC)
    game_polygon_latlon: list[list[float]]
    radar_sites: list[str]
    team_mode: bool
    save_replay: bool
    type: str = "RoundSetup"


@dataclass(frozen=True)
class Tick:
    virtual_time_offset_sec: float       # seconds since round epoch (time_start)
    speed: float
    paused: bool
    type: str = "Tick"


@dataclass(frozen=True)
class PlayerJoin:
    player_id: str
    display_name: str
    type: str = "PlayerJoin"


@dataclass(frozen=True)
class PlayerLeave:
    player_id: str
    type: str = "PlayerLeave"


@dataclass(frozen=True)
class TeamCreate:
    team_id: str
    name: str
    creator_id: str
    type: str = "TeamCreate"


@dataclass(frozen=True)
class TeamJoin:
    team_id: str
    player_id: str
    type: str = "TeamJoin"


@dataclass(frozen=True)
class TeamLeave:
    player_id: str
    type: str = "TeamLeave"


@dataclass(frozen=True)
class TeamRosterFreeze:
    roster: dict[str, list[str]]         # {team_id: [player_id, ...]}
    team_names: dict[str, str]
    type: str = "TeamRosterFreeze"


@dataclass(frozen=True)
class TeamLobbyOpen:
    """Host → peers: enter the pre-round team lobby.

    Sent when the host has opted into team mode (plan §11) and reached the
    lobby phase. Peers transition LOBBY → TEAM_LOBBY locally so their
    waiting-room screen can swap to the :class:`TeamLobbyWidget`. The
    lobby closes when a subsequent :class:`TeamRosterFreeze` arrives.
    """

    type: str = "TeamLobbyOpen"


@dataclass(frozen=True)
class WarningIssue:
    warning_id: str
    issuer_id: str
    warning_type: str                    # WarningType.value
    polygon_latlon: list[list[float]]
    duration_sec: int
    issue_offset_sec: float              # since round epoch
    hail_in: float | None = None
    wind_mph: float | None = None
    ef: float | None = None
    tornado_possible: bool = False
    type: str = "WarningIssue"


@dataclass(frozen=True)
class WarningRevise:
    """Same fields as WarningIssue plus the revision_offset (when the change happened)."""
    warning_id: str
    issuer_id: str
    warning_type: str
    polygon_latlon: list[list[float]]
    duration_sec: int
    revision_offset_sec: float
    hail_in: float | None = None
    wind_mph: float | None = None
    ef: float | None = None
    tornado_possible: bool = False
    type: str = "WarningRevise"


@dataclass(frozen=True)
class WarningCancel:
    warning_id: str
    issuer_id: str
    cancel_offset_sec: float
    type: str = "WarningCancel"


@dataclass(frozen=True)
class MCDIssue:
    mcd_id: str
    issuer_id: str
    polygon_latlon: list[list[float]]
    duration_sec: int
    issue_offset_sec: float
    pib_tornado: int = 0
    pib_wind: int = 0
    pib_hail: int = 0
    type: str = "MCDIssue"


@dataclass(frozen=True)
class MCDCancel:
    mcd_id: str
    issuer_id: str
    cancel_offset_sec: float
    type: str = "MCDCancel"


@dataclass(frozen=True)
class Chat:
    sender_id: str
    text: str
    type: str = "Chat"


# Pre-game start-gate (plan §10 "Prefetch stall"). Peers signal readiness;
# the host coordinates a countdown so the round starts in lockstep instead
# of whenever each client's local prefetch happens to finish.
@dataclass(frozen=True)
class PeerReady:
    """Peer → host: local pre-game prefetch is done; we're ready to play.

    The host tallies these against its known peer set and, once enough
    clients (75% threshold including the host) have reported in, starts
    a fixed 60-second countdown broadcast to everyone via
    :class:`RoundCountdown`.
    """

    player_id: str
    type: str = "PeerReady"


@dataclass(frozen=True)
class RoundCountdown:
    """Host → all peers: time remaining until the round begins.

    Broadcast once per second. ``seconds_remaining == 0`` is the start
    signal — when peers see it they enter play immediately. Re-broadcasts
    are idempotent on the receive side: peers just overwrite their UI
    countdown each tick.
    """

    seconds_remaining: int
    type: str = "RoundCountdown"


# ---------------------------- (de)serialization -------------------------------

_TYPE_REGISTRY: dict[str, type] = {
    cls.__name__: cls for cls in (
        RoundSetup, Tick,
        PlayerJoin, PlayerLeave,
        TeamCreate, TeamJoin, TeamLeave, TeamRosterFreeze, TeamLobbyOpen,
        WarningIssue, WarningRevise, WarningCancel,
        MCDIssue, MCDCancel,
        Chat,
        PeerReady, RoundCountdown,
    )
}


def encode(msg) -> str:
    if not is_dataclass(msg):
        raise TypeError(f"Not a protocol dataclass: {type(msg).__name__}")
    return json.dumps(asdict(msg), separators=(",", ":"))


def decode(raw: str | bytes):
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    obj = json.loads(raw)
    kind = obj.pop("type", None)
    if kind not in _TYPE_REGISTRY:
        raise ValueError(f"Unknown message type: {kind!r}")
    cls = _TYPE_REGISTRY[kind]
    return cls(**obj)


# ---------------------------- conversion helpers ------------------------------

def warning_issue_from(warning, round_epoch: datetime) -> WarningIssue:
    """Build a WarningIssue wire message from an in-process :class:`Warning`."""
    cur = warning.current_revision
    return WarningIssue(
        warning_id=warning.warning_id,
        issuer_id=warning.issuer_id,
        warning_type=cur.warning_type.value,
        polygon_latlon=_poly_payload(cur.polygon.vertices),
        duration_sec=int(cur.duration.total_seconds()),
        issue_offset_sec=_dt_to_offset(warning.original_issue_time, round_epoch),
        hail_in=cur.magnitudes.hail_in,
        wind_mph=cur.magnitudes.wind_mph,
        ef=cur.magnitudes.ef,
        tornado_possible=cur.magnitudes.tornado_possible,
    )


def warning_revise_from(warning, round_epoch: datetime) -> WarningRevise:
    cur = warning.current_revision
    return WarningRevise(
        warning_id=warning.warning_id,
        issuer_id=warning.issuer_id,
        warning_type=cur.warning_type.value,
        polygon_latlon=_poly_payload(cur.polygon.vertices),
        duration_sec=int(cur.duration.total_seconds()),
        revision_offset_sec=_dt_to_offset(cur.revision_time, round_epoch),
        hail_in=cur.magnitudes.hail_in,
        wind_mph=cur.magnitudes.wind_mph,
        ef=cur.magnitudes.ef,
        tornado_possible=cur.magnitudes.tornado_possible,
    )


def mcd_issue_from(mcd, round_epoch: datetime) -> MCDIssue:
    return MCDIssue(
        mcd_id=mcd.mcd_id,
        issuer_id=mcd.issuer_id,
        polygon_latlon=_poly_payload(mcd.polygon.vertices),
        duration_sec=int(mcd.duration.total_seconds()),
        issue_offset_sec=_dt_to_offset(mcd.issue_time, round_epoch),
        pib_tornado=mcd.pib_tornado,
        pib_wind=mcd.pib_wind,
        pib_hail=mcd.pib_hail,
    )


def tick_from(tick_state, round_epoch: datetime) -> Tick:
    return Tick(
        virtual_time_offset_sec=_dt_to_offset(tick_state.virtual_time, round_epoch),
        speed=tick_state.speed,
        paused=tick_state.paused,
    )
