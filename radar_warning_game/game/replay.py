"""Replay log writer + reader (plan §9).

When the host toggles "Save replay" at round setup, a JSON file is written to
``~/.radaranalysiscorp/replays/<timestamp>.json`` containing every action that
shaped the round: setup config, player joins/leaves, team changes, warning
issues / revisions / cancels, MCD issues / cancels, final scores.

Radar data is NOT embedded — replay playback re-fetches from S3 using the same
machinery as live play (per the locked-in S3-per-client design).

The file is append-only-via-events: each entry is a tagged dict, written in
chronological order. The "events" array can be replayed deterministically to
reconstruct the session, modulo S3 fetch differences (transient list ordering).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..geo.polygons import Polygon
from ..verification.reports_in_poly import MCD, Warning

DEFAULT_REPLAY_ROOT = Path(os.path.expanduser("~/.radaranalysiscorp/replays"))


@dataclass
class ReplayEvent:
    """One timestamped event in a replay log."""

    real_time: str          # ISO-8601 wall-clock when event was recorded
    virtual_time: str | None  # ISO-8601 game-virtual time (None for pre-play events)
    kind: str               # e.g. "setup", "warning_issue", "mcd_issue", ...
    payload: dict[str, Any]


class ReplayWriter:
    """Append-only JSON log for one session."""

    def __init__(self, path: Path | None = None) -> None:
        DEFAULT_REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
        if path is None:
            path = DEFAULT_REPLAY_ROOT / f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}.json"
        self.path = Path(path)
        self._events: list[ReplayEvent] = []
        self._open = True

    def log(self, kind: str, payload: dict[str, Any], *, virtual_time: datetime | None = None) -> None:
        if not self._open:
            raise RuntimeError("Replay writer closed")
        evt = ReplayEvent(
            real_time=datetime.now(timezone.utc).isoformat(),
            virtual_time=virtual_time.isoformat() if virtual_time else None,
            kind=kind,
            payload=payload,
        )
        self._events.append(evt)

    def log_warning_issue(self, warning: Warning, *, virtual_time: datetime) -> None:
        self.log("warning_issue", _warning_payload(warning), virtual_time=virtual_time)

    def log_warning_revise(self, warning: Warning, *, virtual_time: datetime) -> None:
        self.log("warning_revise", _warning_payload(warning), virtual_time=virtual_time)

    def log_warning_cancel(self, warning_id: str, player_id: str, *, virtual_time: datetime) -> None:
        self.log(
            "warning_cancel",
            {"warning_id": warning_id, "player_id": player_id},
            virtual_time=virtual_time,
        )

    def log_mcd_issue(self, mcd: MCD, *, virtual_time: datetime) -> None:
        self.log("mcd_issue", _mcd_payload(mcd), virtual_time=virtual_time)

    def log_final_scores(self, scores: list) -> None:
        self.log("final_scores", {"scores": [_score_payload(s) for s in scores]})

    def close(self) -> None:
        if not self._open:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "written_at": datetime.now(timezone.utc).isoformat(),
            "events": [asdict(e) for e in self._events],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".part")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.rename(self.path)
        self._open = False


def load_replay(path: Path) -> list[ReplayEvent]:
    raw = json.loads(Path(path).read_text())
    return [ReplayEvent(**e) for e in raw.get("events", [])]


# ---------------------------- payload encoders --------------------------------

def _polygon_payload(p: Polygon) -> list[list[float]]:
    return [[lat, lon] for lat, lon in p.vertices]


def _warning_payload(w: Warning) -> dict[str, Any]:
    cur = w.current_revision
    return {
        "warning_id": w.warning_id,
        "issuer_id": w.issuer_id,
        "team_id": w.team_id,
        "type": cur.warning_type.value,
        "polygon": _polygon_payload(cur.polygon),
        "duration_sec": int(cur.duration.total_seconds()),
        "magnitudes": {
            "hail_in": cur.magnitudes.hail_in,
            "wind_mph": cur.magnitudes.wind_mph,
            "ef": cur.magnitudes.ef,
        },
        "revision_count": len(w.revisions),
    }


def _mcd_payload(m: MCD) -> dict[str, Any]:
    return {
        "mcd_id": m.mcd_id,
        "issuer_id": m.issuer_id,
        "team_id": m.team_id,
        "polygon": _polygon_payload(m.polygon),
        "duration_sec": int(m.duration.total_seconds()),
        "pib_tornado": m.pib_tornado,
        "pib_wind": m.pib_wind,
        "pib_hail": m.pib_hail,
    }


def _score_payload(s) -> dict[str, Any]:
    return {
        "team_id": s.team_id,
        "members": s.member_ids,
        "total": s.total,
        "warnings_total": s.warnings_total,
        "mcd_total": s.mcd_total,
        "pod": s.pod,
        "far": s.far,
        "csi": s.csi,
        "mean_lead_sec": s.mean_lead_time_sec,
        "n_warnings": s.n_warnings,
        "n_false_alarms": s.n_false_alarms,
        "n_verifying_reports": s.n_verifying_reports,
        "n_total_reports_in_game": s.n_total_reports_in_game,
    }
