"""Random-day / specific-day round picker for the host.

Plan §1 lets the host choose either:

  - a **random day** in [2000-01-01, today-2] that meets minimum thresholds for
    hail / wind / tornado reports (each threshold independent), or
  - a **specific day** by date.

For random-day mode we sample candidate days uniformly and reject ones that don't
meet the thresholds. Stops after :data:`MAX_RANDOM_TRIES` to avoid infinite loops
when thresholds are unreachable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from ..data.reports import (
    Report,
    count_by_category,
    fetch_iem_window,
    get_daily_counts,
)

# Date range bounds (plan locked decision)
DATE_RANGE_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
DEFAULT_TODAY_LAG_DAYS = 2

MAX_RANDOM_TRIES = 200


@dataclass(frozen=True)
class ThresholdSpec:
    """Minimum report counts the picked day must meet."""

    min_tornadoes: int = 0
    min_hail: int = 0
    min_wind: int = 0

    def is_met(self, counts: dict[str, int]) -> bool:
        return (
            counts.get("tornado", 0) >= self.min_tornadoes
            and counts.get("hail", 0) >= self.min_hail
            and counts.get("wind", 0) >= self.min_wind
        )


@dataclass
class RoundDay:
    """Result of picking a day for a round."""

    convective_day_12z: datetime           # 12Z start of the day's 24h window
    reports: list[Report]                  # all IEM reports in [12Z, 12Z+24h]
    counts: dict[str, int]
    is_random: bool                        # True if picked at random, False if explicit


def _convective_day_bounds(day_12z: datetime) -> tuple[datetime, datetime]:
    """Return (start_12z, end_12z+24h) UTC."""
    if day_12z.tzinfo is None:
        day_12z = day_12z.replace(tzinfo=timezone.utc)
    return day_12z, day_12z + timedelta(days=1)


def pick_specific_day(day_12z_date: datetime) -> RoundDay:
    """Pick a specific 12Z convective day. ``day_12z_date`` should be the 12Z timestamp."""
    if day_12z_date.tzinfo is None:
        day_12z_date = day_12z_date.replace(tzinfo=timezone.utc)
    start, end = _convective_day_bounds(day_12z_date)
    reports = fetch_iem_window(start, end)
    return RoundDay(
        convective_day_12z=start,
        reports=reports,
        counts=count_by_category(reports),
        is_random=False,
    )


def pick_random_day(
    spec: ThresholdSpec,
    *,
    range_start: datetime = DATE_RANGE_START,
    today: datetime | None = None,
    today_lag_days: int = DEFAULT_TODAY_LAG_DAYS,
    rng: random.Random | None = None,
    max_tries: int = MAX_RANDOM_TRIES,
) -> RoundDay:
    """Sample 12Z convective days until one meets ``spec``.

    Raises ``RuntimeError`` after ``max_tries`` if no qualifying day was found —
    typically means the thresholds are unrealistically high.
    """
    rng = rng or random.Random()
    today = today or datetime.now(timezone.utc)
    range_end = today.replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=today_lag_days)
    if range_end <= range_start:
        raise ValueError("Effective date range is empty (range_start ≥ range_end)")
    total_days = (range_end - range_start).days
    # The daily-counts index lets us skip non-qualifying days without a fresh
    # network round-trip per attempt. For days we've never indexed we still
    # fetch (which is what populates the index).
    seen_keys: set[str] = set()
    for _ in range(max_tries):
        offset = rng.randint(0, total_days)
        candidate = range_start + timedelta(days=offset)
        candidate = candidate.replace(hour=12, minute=0, second=0, microsecond=0)
        key = candidate.strftime("%Y-%m-%d")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cached_counts = get_daily_counts(candidate)
        if cached_counts is not None:
            if not spec.is_met(cached_counts):
                continue   # known to not qualify — skip without a fetch
            # Counts qualify; still need to fetch the full reports list
        try:
            day = pick_specific_day(candidate)
        except Exception as e:  # noqa: BLE001
            print(f"[round_builder] fetch failed for {candidate:%Y-%m-%d}: {e}")
            continue
        if spec.is_met(day.counts):
            return RoundDay(
                convective_day_12z=day.convective_day_12z,
                reports=day.reports,
                counts=day.counts,
                is_random=True,
            )
    raise RuntimeError(
        f"No day meeting thresholds {spec} found in {max_tries} tries — "
        "consider lowering the thresholds."
    )
