"""Unit tests for the wire-protocol schema (round-trip + conversion helpers)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from radar_warning_game.geo.polygons import Polygon
from radar_warning_game.game.clock import TickState
from radar_warning_game.net import protocol as proto
from radar_warning_game.verification.reports_in_poly import (
    MCD,
    Magnitudes,
    Warning,
    WarningRevision,
)
from radar_warning_game.verification.tornado_tiers import WarningType


_POLY_PAYLOAD = [[35.1, -97.4], [35.1, -97.1], [35.4, -97.1], [35.4, -97.4]]
_POLY_OBJ = Polygon(((35.1, -97.4), (35.1, -97.1), (35.4, -97.1), (35.4, -97.4)))
_T0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize("msg", [
    proto.RoundSetup(
        convective_day_12z_iso=_T0.replace(hour=12).isoformat(),
        time_start_iso=_T0.isoformat(),
        time_end_iso=(_T0 + timedelta(hours=2)).isoformat(),
        game_polygon_latlon=_POLY_PAYLOAD, radar_sites=["KTLX", "KVNX"],
        team_mode=True, save_replay=False,
    ),
    proto.Tick(virtual_time_offset_sec=300.0, speed=2.0, paused=False),
    proto.PlayerJoin("alice", "Alice"),
    proto.PlayerLeave("alice"),
    proto.TeamCreate("team:abc", "Storm Chasers", "alice"),
    proto.TeamJoin("team:abc", "bob"),
    proto.TeamLeave("alice"),
    proto.TeamRosterFreeze(
        roster={"team:abc": ["alice", "bob"]},
        team_names={"team:abc": "Storm Chasers"},
    ),
    proto.WarningIssue("w1", "alice", "PDS_TOR", _POLY_PAYLOAD, 1800, 600.0, ef=3.0),
    proto.WarningRevise("w1", "alice", "TORE", _POLY_PAYLOAD, 1800, 900.0, ef=3.0),
    proto.WarningCancel("w1", "alice", 1200.0),
    proto.MCDIssue("m1", "alice", _POLY_PAYLOAD, 5400, 0.0,
                   pib_tornado=4, pib_wind=3, pib_hail=3),
    proto.MCDCancel("m1", "alice", 600.0),
    proto.Chat("alice", "wedge approaching norman"),
])
def test_round_trip(msg):
    raw = proto.encode(msg)
    decoded = proto.decode(raw)
    assert decoded == msg


def test_encode_rejects_non_dataclass():
    with pytest.raises(TypeError):
        proto.encode({"type": "Tick"})


def test_decode_rejects_unknown_type():
    with pytest.raises(ValueError):
        proto.decode('{"type":"BogusMessage"}')


def test_decode_handles_bytes_input():
    msg = proto.Chat("alice", "hi")
    raw = proto.encode(msg).encode("utf-8")
    decoded = proto.decode(raw)
    assert decoded == msg


# ---- conversion helpers -------------------------------------------

def test_warning_issue_from_with_epoch():
    w = Warning(
        warning_id="w1", issuer_id="alice", team_id="alice",
        revisions=[WarningRevision(
            revision_time=_T0, warning_type=WarningType.PDS_TOR,
            polygon=_POLY_OBJ, duration=timedelta(minutes=30),
            magnitudes=Magnitudes(ef=3.0),
        )],
    )
    msg = proto.warning_issue_from(w, round_epoch=_T0)
    assert msg.issue_offset_sec == 0.0
    assert msg.warning_type == "PDS_TOR"
    assert msg.ef == 3.0
    assert msg.duration_sec == 1800


def test_warning_revise_from_uses_current_revision():
    revs = [
        WarningRevision(revision_time=_T0, warning_type=WarningType.TOR,
                        polygon=_POLY_OBJ, duration=timedelta(minutes=30),
                        magnitudes=Magnitudes()),
        WarningRevision(revision_time=_T0 + timedelta(minutes=5),
                        warning_type=WarningType.TORE, polygon=_POLY_OBJ,
                        duration=timedelta(minutes=30),
                        magnitudes=Magnitudes(ef=3.0)),
    ]
    w = Warning(warning_id="w1", issuer_id="alice", team_id="alice", revisions=revs)
    msg = proto.warning_revise_from(w, round_epoch=_T0)
    assert msg.warning_type == "TORE"
    assert msg.revision_offset_sec == 300.0


def test_mcd_issue_from():
    m = MCD(mcd_id="m1", issuer_id="alice", team_id="alice",
            polygon=_POLY_OBJ, issue_time=_T0 + timedelta(minutes=10),
            duration=timedelta(minutes=90),
            pib_tornado=4, pib_wind=3, pib_hail=3)
    msg = proto.mcd_issue_from(m, round_epoch=_T0)
    assert msg.issue_offset_sec == 600.0
    assert msg.pib_tornado == 4
    assert msg.duration_sec == 5400


def test_tick_from():
    ts = TickState(virtual_time=_T0 + timedelta(minutes=15), speed=4.0, paused=False)
    msg = proto.tick_from(ts, round_epoch=_T0)
    assert msg.virtual_time_offset_sec == 900.0
    assert msg.speed == 4.0
