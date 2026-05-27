"""Game session state machine — the central hub.

A ``GameSession`` owns the full state of one room from creation through end-of-round:

  - players / teams (lobby + roster freeze, §11)
  - round configuration (day, polygon, radars, time window, §1-§3)
  - prefetch state (§10 pre-game gate)
  - clock (§4 host control)
  - active warnings + MCDs by player (§5, §8)
  - live + final scoring (§6, §7, §9)

State transitions:
    LOBBY → TEAM_LOBBY (if team mode) → SETUP → PREFETCH → PLAYING → ENDED

This module is networking-agnostic — the host's network layer drives state
transitions by calling methods here; the local UI reads state and renders.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

from ..data.reports import Report
from ..geo.polygons import Polygon
from ..verification.reports_in_poly import MCD, Magnitudes, Warning, WarningRevision
from ..verification.scoring import TeamScore, score_round
from ..verification.tornado_tiers import WarningType
from .clock import GameClock
from .round_builder import RoundDay


class SessionState(str, Enum):
    LOBBY = "LOBBY"
    TEAM_LOBBY = "TEAM_LOBBY"
    SETUP = "SETUP"            # host picking polygon, sites, time window
    PREFETCH = "PREFETCH"      # clients downloading first chunk of radar data
    PLAYING = "PLAYING"
    ENDED = "ENDED"


class RoundMode(str, Enum):
    """Distinguishes a historical (replay) round from a live (real-time) one.

    Live mode (plan §12) disables date-blinding (today is today, no spoiler),
    locks the clock to wall time, and polls IEM live data for radar + reports.
    """

    HISTORICAL = "HISTORICAL"
    LIVE = "LIVE"


SOLO_TEAM_PREFIX = "solo:"

# MCD anti-spam (plan §8). Enforced at session.issue_mcd so wire MCDs from
# peers can't bypass the host UI's validation.
#
# The upper cap is an absolute area rather than a fraction of the game
# polygon. The fraction-based rule was a poor proxy: small training
# rounds (e.g. a county-scale game area) would force MCDs to be tiny
# and stop them from representing realistic SPC-scale mesoscale
# discussions; large continental rounds would allow MCDs that aren't
# meaningfully bounded. 250,000 km² is roughly the area of a large
# real-world MCD ("4-5 states worth"); generous enough for any
# realistic forecast scenario but small enough to prevent blanket-
# the-game-area abuse.
MCD_MIN_AREA_KM2 = 200.0
MCD_MAX_AREA_KM2 = 250_000.0
MCD_MIN_VERTICES = 4
MCD_RATE_LIMIT_PER_TEAM = timedelta(minutes=30)


class MCDValidationError(ValueError):
    """Raised when an MCD violates the §8 anti-spam rules."""


# Allowed-transition table for the session state machine. Caller-friendly:
# raising on illegal transitions catches orchestration bugs early.
_ALLOWED_TRANSITIONS = {
    SessionState.LOBBY: {SessionState.TEAM_LOBBY, SessionState.SETUP, SessionState.ENDED},
    SessionState.TEAM_LOBBY: {SessionState.SETUP, SessionState.ENDED},
    SessionState.SETUP: {SessionState.PREFETCH, SessionState.ENDED},
    # PREFETCH can fall back to SETUP if the host cancels the prefetch
    # to revise their radar selection (e.g. they enabled a site that
    # has no archive data for the chosen day and want to deselect it
    # before retrying).
    SessionState.PREFETCH: {SessionState.PLAYING, SessionState.SETUP, SessionState.ENDED},
    SessionState.PLAYING: {SessionState.ENDED},
    SessionState.ENDED: set(),
}


class IllegalStateTransition(RuntimeError):
    """Raised when a session is asked to transition to an unreachable state."""


@dataclass
class Player:
    player_id: str
    display_name: str
    is_host: bool = False
    team_id: str | None = None
    joined_at: datetime | None = None


@dataclass
class RoundConfig:
    """Frozen configuration sent to peers at PREFETCH start."""

    convective_day_12z: datetime
    game_polygon: Polygon
    radar_sites: list[str]
    time_start: datetime
    time_end: datetime
    save_replay: bool = False
    team_mode: bool = False
    mode: RoundMode = RoundMode.HISTORICAL


@dataclass
class GameSession:
    """In-memory state of one game session. The host's instance is authoritative."""

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: SessionState = SessionState.LOBBY
    players: dict[str, Player] = field(default_factory=dict)
    teams: dict[str, list[str]] = field(default_factory=dict)         # team_id → [player_id]
    team_names: dict[str, str] = field(default_factory=dict)           # team_id → display name
    round_day: RoundDay | None = None
    round_config: RoundConfig | None = None
    clock: GameClock | None = None
    warnings_by_player: dict[str, list[Warning]] = field(default_factory=dict)
    mcds_by_player: dict[str, list[MCD]] = field(default_factory=dict)
    cached_reports: list[Report] = field(default_factory=list)         # filtered to game polygon + time window
    final_scores: list[TeamScore] | None = None

    # ---- lobby ----------------------------------------------------------

    def add_player(self, player: Player) -> None:
        self.players[player.player_id] = player
        # Until team lobby resolves, treat new player as solo
        self._ensure_solo_team(player.player_id)

    def remove_player(self, player_id: str) -> None:
        self.players.pop(player_id, None)
        self.warnings_by_player.pop(player_id, None)
        self.mcds_by_player.pop(player_id, None)
        # Detach from any team
        for tid, members in list(self.teams.items()):
            if player_id in members:
                members.remove(player_id)
                if not members:
                    del self.teams[tid]
                    self.team_names.pop(tid, None)

    def _ensure_solo_team(self, player_id: str) -> None:
        tid = f"{SOLO_TEAM_PREFIX}{player_id}"
        self.teams.setdefault(tid, [player_id])
        self.team_names.setdefault(tid, self.players[player_id].display_name)
        self.players[player_id].team_id = tid

    # ---- team lobby (§11) ----------------------------------------------

    def _transition(self, new_state: SessionState) -> None:
        if new_state not in _ALLOWED_TRANSITIONS[self.state]:
            raise IllegalStateTransition(
                f"Cannot transition from {self.state.value} to {new_state.value}"
            )
        self.state = new_state

    def enter_team_lobby(self) -> None:
        self._transition(SessionState.TEAM_LOBBY)

    def create_team(self, name: str, creator_id: str) -> str:
        """Create a new team and move ``creator_id`` into it. Returns team_id."""
        tid = f"team:{uuid.uuid4().hex[:8]}"
        self.teams[tid] = []
        self.team_names[tid] = name
        self.join_team(creator_id, tid)
        return tid

    def join_team(self, player_id: str, team_id: str) -> None:
        if team_id not in self.teams:
            raise ValueError(f"Unknown team: {team_id}")
        self._remove_from_all_teams(player_id)
        self.teams[team_id].append(player_id)
        self.players[player_id].team_id = team_id

    def leave_team(self, player_id: str) -> None:
        """Remove from current team and put back into a fresh solo placeholder."""
        self._remove_from_all_teams(player_id)
        self._ensure_solo_team(player_id)

    def _remove_from_all_teams(self, player_id: str) -> None:
        """Internal: drop ``player_id`` from every team and clean up empties.

        Does NOT auto-recreate a solo team — that's the caller's responsibility,
        so ``join_team`` can move a player without leaving a ghost solo behind.
        """
        for tid, members in list(self.teams.items()):
            if player_id in members:
                members.remove(player_id)
                if not members:
                    del self.teams[tid]
                    self.team_names.pop(tid, None)
        if player_id in self.players:
            self.players[player_id].team_id = None

    def freeze_roster(self) -> None:
        """At round start, lock teams. Unassigned players (still in solo teams) stay solo."""
        # Already represented; nothing to do beyond stopping further team changes.
        self._transition(SessionState.SETUP)

    # ---- setup → playing ----------------------------------------------

    def set_round(self, round_day: RoundDay, config: RoundConfig) -> None:
        self.round_day = round_day
        self.round_config = config

    def reroll_random_day(self, spec, **picker_kwargs) -> "RoundDay":
        """Resample a new random day with the same thresholds (host UI option).

        Only valid pre-confirmation (state ∈ {SETUP, LOBBY, TEAM_LOBBY}) and only
        if the current ``round_day`` was randomly picked. The new day overwrites
        ``round_day``; the host then sees a fresh CONUS report map and can either
        reroll again or proceed to polygon drawing.
        """
        from .round_builder import pick_random_day
        if self.state not in (SessionState.LOBBY, SessionState.TEAM_LOBBY, SessionState.SETUP):
            raise RuntimeError(f"Cannot reroll once in {self.state.value}")
        if self.round_day is not None and not self.round_day.is_random:
            raise RuntimeError("Cannot reroll a specifically-chosen day; pick a new one explicitly")
        new_day = pick_random_day(spec, **picker_kwargs)
        self.round_day = new_day
        # If a polygon/time/sites were already chosen they're now stale — clear config.
        self.round_config = None
        return new_day

    def begin_prefetch(self) -> None:
        if self.round_config is None:
            raise RuntimeError("Cannot enter PREFETCH without a round config")
        self._transition(SessionState.PREFETCH)

    def cancel_prefetch(self) -> None:
        """Abandon a started-but-not-yet-playing round. Returns the session
        to ``SETUP`` so the host can revise their radar selection / day
        and retry from the overview map. Idempotent — no-op if not in
        ``PREFETCH``."""
        if self.state != SessionState.PREFETCH:
            return
        # Clear the half-built round config so the next start path
        # rebuilds it from scratch with the host's revised selections.
        self.round_config = None
        self._transition(SessionState.SETUP)

    def begin_play(self) -> None:
        if self.round_config is None:
            raise RuntimeError("Cannot begin play without a round config")
        # PLAYING is reachable only from PREFETCH per the transition table
        self.clock = GameClock(self.round_config.time_start, self.round_config.time_end)
        self.clock.play()
        # Pre-filter reports to the game polygon (verification denominator)
        from ..verification.reports_in_poly import reports_in_polygon
        if self.round_day is not None:
            self.cached_reports = reports_in_polygon(
                self.round_config.game_polygon,
                self.round_day.reports,
                time_window=(self.round_config.time_start, self.round_config.time_end),
            )
        self._transition(SessionState.PLAYING)

    def end_round(self) -> list[TeamScore]:
        """Compute final scores, transition to ENDED, return team scores high→low."""
        if self.round_config is None or self.round_day is None:
            raise RuntimeError("No round to end")
        self.final_scores = score_round(
            teams=self.teams,
            warnings_by_player=self.warnings_by_player,
            mcds_by_player=self.mcds_by_player,
            all_reports=self.round_day.reports,
            game_polygon=self.round_config.game_polygon,
        )
        self._transition(SessionState.ENDED)
        return self.final_scores

    # ---- live actions ---------------------------------------------------

    def issue_warning(
        self,
        *,
        player_id: str,
        warning_type: WarningType,
        polygon: Polygon,
        duration: timedelta,
        magnitudes: Magnitudes,
        warning_id: str | None = None,
        issue_time: datetime | None = None,
    ) -> Warning:
        """Issue a warning. Pass ``warning_id``/``issue_time`` only when reconstructing
        a warning received from the network (so all peers share the same IDs).
        """
        if self.clock is None and issue_time is None:
            raise RuntimeError("No clock — call begin_play() first")
        warning = Warning(
            warning_id=warning_id or uuid.uuid4().hex[:10],
            issuer_id=player_id,
            team_id=(self.players.get(player_id).team_id if player_id in self.players
                     else f"{SOLO_TEAM_PREFIX}{player_id}")
                    or f"{SOLO_TEAM_PREFIX}{player_id}",
            revisions=[WarningRevision(
                revision_time=issue_time or self.clock.virtual_time,
                warning_type=warning_type,
                polygon=polygon,
                duration=duration,
                magnitudes=magnitudes,
            )],
        )
        self.warnings_by_player.setdefault(player_id, []).append(warning)
        return warning

    def revise_warning(
        self,
        *,
        warning_id: str,
        player_id: str,
        warning_type: WarningType | None = None,
        polygon: Polygon | None = None,
        duration: timedelta | None = None,
        magnitudes: Magnitudes | None = None,
    ) -> Warning:
        """Append a new revision to an active warning (§5 in-place amend semantics)."""
        if self.clock is None:
            raise RuntimeError("No clock — call begin_play() first")
        existing = self._find_warning(warning_id, player_id)
        cur = existing.current_revision
        new_rev = WarningRevision(
            revision_time=self.clock.virtual_time,
            warning_type=warning_type or cur.warning_type,
            polygon=polygon or cur.polygon,
            duration=duration if duration is not None else cur.duration,
            magnitudes=magnitudes or cur.magnitudes,
        )
        existing.revisions.append(new_rev)
        return existing

    def cancel_warning(self, *, warning_id: str, player_id: str) -> None:
        if self.clock is None:
            raise RuntimeError("No clock — call begin_play() first")
        existing = self._find_warning(warning_id, player_id)
        existing.canceled_at = self.clock.virtual_time

    def issue_mcd(
        self,
        *,
        player_id: str,
        polygon: Polygon,
        duration: timedelta,
        pib_tornado: int = 0,
        pib_wind: int = 0,
        pib_hail: int = 0,
        mcd_id: str | None = None,
        issue_time: datetime | None = None,
    ) -> MCD:
        if self.clock is None and issue_time is None:
            raise RuntimeError("No clock — call begin_play() first")
        effective_issue = issue_time or self.clock.virtual_time
        team_id = (
            (self.players.get(player_id).team_id if player_id in self.players
             else f"{SOLO_TEAM_PREFIX}{player_id}")
            or f"{SOLO_TEAM_PREFIX}{player_id}"
        )
        self._validate_mcd_or_raise(
            team_id=team_id, polygon=polygon, issue_time=effective_issue,
            pib_tornado=pib_tornado, pib_wind=pib_wind, pib_hail=pib_hail,
        )
        mcd = MCD(
            mcd_id=mcd_id or uuid.uuid4().hex[:10],
            issuer_id=player_id,
            team_id=team_id,
            polygon=polygon,
            issue_time=effective_issue,
            duration=duration,
            pib_tornado=pib_tornado,
            pib_wind=pib_wind,
            pib_hail=pib_hail,
        )
        self.mcds_by_player.setdefault(player_id, []).append(mcd)
        return mcd

    def _validate_mcd_or_raise(
        self, *,
        team_id: str, polygon: Polygon, issue_time: datetime,
        pib_tornado: int, pib_wind: int, pib_hail: int,
    ) -> None:
        """Enforce the plan §8 anti-spam rules at session level (server-side).

        Wire MCDs from peers are validated here, not just in the dialog UI.
        """
        from ..geo.polygons import polygon_area_km2
        # At least one hazard PIB ≥ 1
        if pib_tornado == 0 and pib_wind == 0 and pib_hail == 0:
            raise MCDValidationError("MCD must have ≥1 hazard PIB > 0")
        # Min vertices
        if len(polygon.vertices) < MCD_MIN_VERTICES:
            raise MCDValidationError(
                f"MCD polygon must have ≥{MCD_MIN_VERTICES} vertices"
            )
        # Min area
        area = polygon_area_km2(polygon)
        if area < MCD_MIN_AREA_KM2:
            raise MCDValidationError(
                f"MCD polygon area {area:.0f} km² < min {MCD_MIN_AREA_KM2:.0f} km²"
            )
        # Max area — absolute km² cap, independent of game-polygon size.
        if area > MCD_MAX_AREA_KM2:
            raise MCDValidationError(
                f"MCD polygon area {area:.0f} km² exceeds max "
                f"{MCD_MAX_AREA_KM2:.0f} km²"
            )
        # Per-team rate limit
        team_mcds = [
            m for player_id, mcds in self.mcds_by_player.items()
            for m in mcds
            if m.team_id == team_id
        ]
        if team_mcds:
            most_recent = max(team_mcds, key=lambda m: m.issue_time)
            since = issue_time - most_recent.issue_time
            if since < MCD_RATE_LIMIT_PER_TEAM:
                remaining = MCD_RATE_LIMIT_PER_TEAM - since
                raise MCDValidationError(
                    f"Team rate-limited — next MCD allowed in "
                    f"{int(remaining.total_seconds() / 60)}m"
                )

    def _find_warning(self, warning_id: str, player_id: str) -> Warning:
        for w in self.warnings_by_player.get(player_id, []):
            if w.warning_id == warning_id:
                return w
        raise KeyError(f"Warning {warning_id} not found for player {player_id}")

    # ---- live scoring (corner-widget feed) -----------------------------

    def current_scores(self) -> list[TeamScore]:
        """Provisional scores at the current virtual time (live leaderboard).

        Only reports whose ``time ≤ clock.virtual_time`` participate, matching
        the live-feedback rule from plan §9 ("as reports come in").
        """
        if self.clock is None or self.round_config is None or self.round_day is None:
            return []
        now = self.clock.virtual_time
        visible_reports = [r for r in self.round_day.reports if r.time <= now]
        return score_round(
            teams=self.teams,
            warnings_by_player=self.warnings_by_player,
            mcds_by_player=self.mcds_by_player,
            all_reports=visible_reports,
            game_polygon=self.round_config.game_polygon,
        )
