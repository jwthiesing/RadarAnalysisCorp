"""Unit tests for GameClock and LiveClock."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from radar_warning_game.game.clock import (
    DEFAULT_SPEED,
    MAX_SPEED,
    MIN_SPEED,
    GameClock,
    LiveClock,
    TickState,
)


_T0 = datetime(2024, 4, 1, 20, 0, tzinfo=timezone.utc)
_T1 = _T0 + timedelta(hours=2)


def test_constructor_rejects_inverted_window():
    with pytest.raises(ValueError):
        GameClock(_T1, _T0)


def test_initial_state_paused_at_start():
    c = GameClock(_T0, _T1)
    assert c.virtual_time == _T0
    assert c.speed == DEFAULT_SPEED
    assert c.paused is True


def test_play_toggles_paused():
    c = GameClock(_T0, _T1)
    c.play()
    assert c.paused is False
    c.play()  # already playing → no-op
    assert c.paused is False
    c.pause()
    assert c.paused is True


def test_faster_doubles_speed_bounded():
    c = GameClock(_T0, _T1)
    c.set_speed(8.0)
    c.faster()
    assert c.speed == pytest.approx(16.0)
    c.faster()  # already at max
    assert c.speed == pytest.approx(MAX_SPEED)


def test_slower_halves_speed_bounded():
    c = GameClock(_T0, _T1)
    c.slower()
    c.slower()
    c.slower()
    c.slower()
    c.slower()    # past min
    assert c.speed == pytest.approx(MIN_SPEED)


def test_set_speed_clamps_to_range():
    c = GameClock(_T0, _T1)
    c.set_speed(100.0)
    assert c.speed == MAX_SPEED
    c.set_speed(0.01)
    assert c.speed == MIN_SPEED


def test_advance_progresses_virtual_time_when_playing():
    c = GameClock(_T0, _T1)
    c.play()
    c.advance(now=100.0)   # initialize
    c.advance(now=130.0)   # 30 real-s @ 1x → +30 vt
    assert c.virtual_time == _T0 + timedelta(seconds=30)


def test_advance_paused_does_not_progress():
    c = GameClock(_T0, _T1)
    c.advance(now=100.0)
    c.advance(now=200.0)
    assert c.virtual_time == _T0


def test_advance_clamps_to_end():
    c = GameClock(_T0, _T0 + timedelta(seconds=10))
    c.play()
    c.set_speed(16.0)
    c.advance(now=0.0)
    c.advance(now=1.0)   # 16 vt sec → would be at +16s, clamps to end
    assert c.virtual_time == _T0 + timedelta(seconds=10)
    assert c.paused is True   # auto-paused at end


def test_apply_tick_clamps_out_of_range():
    c = GameClock(_T0, _T1)
    c.apply_tick(TickState(virtual_time=_T0 - timedelta(hours=10),
                            speed=1.0, paused=False))
    assert c.virtual_time == _T0
    c.apply_tick(TickState(virtual_time=_T1 + timedelta(hours=10),
                            speed=1.0, paused=False))
    assert c.virtual_time == _T1


# ---- LiveClock ------------------------------------------------------

def test_live_clock_speed_is_always_one():
    c = LiveClock(_T0, _T1)
    assert c.speed == 1.0
    c.faster()
    c.set_speed(16.0)
    assert c.speed == 1.0


def test_live_clock_pause_is_noop():
    c = LiveClock(_T0, _T1)
    assert c.paused is False
    c.toggle_pause()
    assert c.paused is False
    c.pause()
    assert c.paused is False


def test_live_clock_advance_reads_wall_time():
    """LiveClock should snap virtual_time to ``now()``."""
    # Make a clock whose window includes "now"
    now = datetime.now(timezone.utc)
    c = LiveClock(now - timedelta(hours=1), now + timedelta(hours=1))
    c.advance()
    # virtual_time should now be ≈ now, within a couple seconds
    assert abs((c.virtual_time - datetime.now(timezone.utc)).total_seconds()) < 2.0


def test_live_clock_apply_tick_ignored():
    """Live mode: each client reads wall clock locally — ticks must NOT alter it."""
    c = LiveClock(_T0, _T1)
    c.apply_tick(TickState(virtual_time=_T0 + timedelta(minutes=30),
                            speed=4.0, paused=True))
    # speed/paused should still be the live defaults
    assert c.speed == 1.0
    assert c.paused is False
