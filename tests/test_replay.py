"""Unit tests for the replay log writer."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from radar_warning_game.game.replay import ReplayWriter, load_replay
from radar_warning_game.geo.polygons import Polygon
from radar_warning_game.verification.reports_in_poly import (
    MCD,
    Magnitudes,
    Warning,
    WarningRevision,
)
from radar_warning_game.verification.tornado_tiers import WarningType


_T0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
_POLY = Polygon(((35.1, -97.4), (35.1, -97.1), (35.4, -97.1), (35.4, -97.4)))


def _warning():
    return Warning(
        warning_id="w1", issuer_id="alice", team_id="alice",
        revisions=[WarningRevision(
            revision_time=_T0, warning_type=WarningType.PDS_TOR, polygon=_POLY,
            duration=timedelta(minutes=30), magnitudes=Magnitudes(ef=3.0),
        )],
    )


def _mcd():
    return MCD(mcd_id="m1", issuer_id="alice", team_id="alice", polygon=_POLY,
                issue_time=_T0, duration=timedelta(minutes=90),
                pib_tornado=3, pib_wind=2, pib_hail=4)


def test_writer_logs_warning_issue(tmp_path):
    rp = ReplayWriter(tmp_path / "r.json")
    rp.log_warning_issue(_warning(), virtual_time=_T0)
    rp.close()
    events = load_replay(rp.path)
    assert events[0].kind == "warning_issue"
    assert events[0].payload["warning_id"] == "w1"
    assert events[0].payload["type"] == "PDS_TOR"
    assert events[0].payload["magnitudes"]["ef"] == 3.0


def test_writer_logs_revise_cancel(tmp_path):
    w = _warning()
    rp = ReplayWriter(tmp_path / "r.json")
    rp.log_warning_revise(w, virtual_time=_T0+timedelta(minutes=5))
    rp.log_warning_cancel(w.warning_id, w.issuer_id, virtual_time=_T0+timedelta(minutes=10))
    rp.close()
    events = load_replay(rp.path)
    kinds = [e.kind for e in events]
    assert kinds == ["warning_revise", "warning_cancel"]
    assert events[1].payload["warning_id"] == "w1"


def test_writer_logs_mcd(tmp_path):
    rp = ReplayWriter(tmp_path / "r.json")
    rp.log_mcd_issue(_mcd(), virtual_time=_T0)
    rp.close()
    events = load_replay(rp.path)
    assert events[0].kind == "mcd_issue"
    p = events[0].payload
    assert p["pib_tornado"] == 3 and p["pib_wind"] == 2 and p["pib_hail"] == 4


def test_writer_atomic_rename(tmp_path):
    """After close(), the .part file should be gone and the final exists."""
    rp = ReplayWriter(tmp_path / "r.json")
    rp.log("synthetic", {"k": 1})
    rp.close()
    assert (tmp_path / "r.json").exists()
    assert not (tmp_path / "r.json.part").exists()


def test_writer_refuses_after_close(tmp_path):
    rp = ReplayWriter(tmp_path / "r.json")
    rp.close()
    with pytest.raises(RuntimeError):
        rp.log("synthetic", {})


def test_format_version_in_file(tmp_path):
    rp = ReplayWriter(tmp_path / "r.json")
    rp.log("x", {})
    rp.close()
    raw = json.loads((tmp_path / "r.json").read_text())
    assert raw["format_version"] == 1
    assert "written_at" in raw
