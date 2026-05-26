"""Virtual game clock with host-controlled speed (plan §4 / §10).

The host owns one ``GameClock`` instance and advances it as real time passes,
scaled by the current speed multiplier. The host periodically broadcasts the
clock state (``virtual_time``, ``speed``, ``paused``) over the network; peers
hold a passive ``GameClock`` and apply received ticks directly.

Speed control on the host:
  - ``[`` (slower)  → halves the multiplier (1.0 → 0.5 → 0.25)
  - ``]`` (faster) → doubles the multiplier (1.0 → 2.0 → 4.0 → ...)
  - ``Space`` → toggle pause

This module is GUI-agnostic — Qt/Tk just calls :meth:`GameClock.advance` from a
periodic timer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta

MIN_SPEED = 0.125
MAX_SPEED = 16.0
DEFAULT_SPEED = 1.0


@dataclass
class TickState:
    """Snapshot suitable for broadcasting over the wire."""

    virtual_time: datetime
    speed: float
    paused: bool


class GameClock:
    """Wall-clock-driven virtual clock bounded by ``[start, end]``."""

    def __init__(self, start: datetime, end: datetime) -> None:
        if end <= start:
            raise ValueError("end must be after start")
        self.start = start
        self.end = end
        self._virtual_time = start
        self._speed = DEFAULT_SPEED
        self._paused = True
        self._last_real_time: float | None = None

    # ---- state -----------------------------------------------------------

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def paused(self) -> bool:
        return self._paused

    @property
    def virtual_time(self) -> datetime:
        return self._virtual_time

    def is_over(self) -> bool:
        return self._virtual_time >= self.end

    def snapshot(self) -> TickState:
        return TickState(virtual_time=self._virtual_time, speed=self._speed, paused=self._paused)

    # ---- host control ----------------------------------------------------

    def faster(self) -> None:
        self._speed = min(MAX_SPEED, self._speed * 2.0)

    def slower(self) -> None:
        self._speed = max(MIN_SPEED, self._speed / 2.0)

    def set_speed(self, speed: float) -> None:
        self._speed = max(MIN_SPEED, min(MAX_SPEED, float(speed)))

    def toggle_pause(self) -> None:
        self._paused = not self._paused
        self._last_real_time = None  # reset so next advance() doesn't see stale elapsed

    def play(self) -> None:
        if self._paused:
            self.toggle_pause()

    def pause(self) -> None:
        if not self._paused:
            self.toggle_pause()

    def advance(self, *, now: float | None = None) -> TickState:
        """Host: tick the clock forward by ``elapsed_real * speed``.

        Call this from a periodic timer (e.g. 10–30 Hz). Returns the snapshot
        the host should broadcast.
        """
        now = now if now is not None else time.monotonic()
        if self._last_real_time is None:
            self._last_real_time = now
            return self.snapshot()
        dt_real = max(0.0, now - self._last_real_time)
        self._last_real_time = now
        if self._paused:
            return self.snapshot()
        dt_virtual = dt_real * self._speed
        self._virtual_time = min(self.end, self._virtual_time + timedelta(seconds=dt_virtual))
        if self._virtual_time >= self.end:
            self._paused = True
        return self.snapshot()

    # ---- peer side -------------------------------------------------------

    def apply_tick(self, tick: TickState) -> None:
        """Peer: overwrite local state from a host broadcast.

        Clamps to ``[start, end]`` to defend against malformed network input.
        """
        vt = tick.virtual_time
        if vt < self.start:
            vt = self.start
        elif vt > self.end:
            vt = self.end
        self._virtual_time = vt
        self._speed = max(MIN_SPEED, min(MAX_SPEED, float(tick.speed)))
        self._paused = bool(tick.paused)


class LiveClock(GameClock):
    """Wall-clock clock used in LIVE mode (plan §12).

    Locks ``virtual_time = now()``. Speed adjustments are no-ops (always 1×),
    pause is no-op for multiplayer fairness. Useful when nowcasting current
    weather: time only moves when real time moves.
    """

    def __init__(self, start, end) -> None:
        super().__init__(start, end)
        self._paused = False

    @property
    def speed(self) -> float:
        return 1.0

    @property
    def paused(self) -> bool:
        return False

    def faster(self) -> None:
        return

    def slower(self) -> None:
        return

    def set_speed(self, speed: float) -> None:
        return

    def toggle_pause(self) -> None:
        return

    def play(self) -> None:
        return

    def pause(self) -> None:
        return

    def advance(self, *, now: float | None = None) -> TickState:
        from datetime import datetime as _dt, timezone as _tz
        real = _dt.now(_tz.utc)
        if real < self.start:
            self._virtual_time = self.start
        elif real > self.end:
            self._virtual_time = self.end
        else:
            self._virtual_time = real
        return self.snapshot()

    def apply_tick(self, tick: TickState) -> None:
        # Live mode is peer-equal: everyone reads wall clock locally. Ignore.
        return
