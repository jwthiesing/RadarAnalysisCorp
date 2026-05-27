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
    fetch_reports,
    get_daily_counts,
    peak_tornado_ef,
)

# Date range bounds (plan locked decision)
DATE_RANGE_START = datetime(2000, 1, 1, tzinfo=timezone.utc)
DEFAULT_TODAY_LAG_DAYS = 2

MAX_RANDOM_TRIES = 200


@dataclass(frozen=True)
class ThresholdSpec:
    """Minimum thresholds the picked day must meet.

    ``min_tornadoes`` / ``min_hail`` / ``min_wind`` are simple count
    floors. ``min_strongest_tornado_ef`` filters by the day's *peak*
    confirmed tornado EF — useful for "give me a day with at least one
    EF3+" picks. The peak EF is sourced post-SVRGIS-backfill, so
    older events (>180 days) use the NWS-survey-confirmed rating
    rather than the often-unset IEM preliminary value. A value of
    ``-1.0`` (default) is "no EF constraint" — any tornado activity
    that satisfies the count floor passes.
    """

    min_tornadoes: int = 0
    min_hail: int = 0
    min_wind: int = 0
    min_strongest_tornado_ef: float = -1.0

    def is_met(self, counts: dict) -> bool:
        if counts.get("tornado", 0) < self.min_tornadoes:
            return False
        if counts.get("hail", 0) < self.min_hail:
            return False
        if counts.get("wind", 0) < self.min_wind:
            return False
        if self.min_strongest_tornado_ef >= 0.0:
            # ``peak_ef`` is the day's strongest confirmed tornado EF
            # (or -1.0 if no rated tornado occurred). Days for which
            # we haven't yet recorded a post-SVRGIS peak EF (legacy
            # cache entries, very-recent days) report ``None``; we
            # fail the gate in that case so the picker falls through
            # to a fresh fetch + re-record. After that one round-trip
            # the cache is current and subsequent picks short-circuit.
            peak = counts.get("peak_ef")
            if peak is None or float(peak) < self.min_strongest_tornado_ef:
                return False
        return True


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
    """Pick a specific 12Z convective day. ``day_12z_date`` should be the 12Z timestamp.

    Goes through :func:`fetch_reports` (not raw ``fetch_iem_window``) so
    tornado magnitudes / casualty counts get backfilled from SPC SVRGIS
    + SPC daily-filtered overlays before the day's reports propagate
    into the CONUS overview map, the time-distribution histogram, and
    scoring. For events older than ~6 months that means actual
    post-survey EF ratings; for events 30 days-6 months old that means
    the SPC daily-filtered overlay; for very recent events it's a
    no-op fallthrough to the raw IEM data."""
    if day_12z_date.tzinfo is None:
        day_12z_date = day_12z_date.replace(tzinfo=timezone.utc)
    start, end = _convective_day_bounds(day_12z_date)
    reports = fetch_reports(start, end)
    counts: dict = dict(count_by_category(reports))
    # Pack the day's strongest confirmed tornado EF into the counts
    # dict so ``ThresholdSpec.is_met`` can apply its EF floor on a
    # freshly-fetched day (same as the cached-counts short-circuit).
    counts["peak_ef"] = peak_tornado_ef(reports)
    return RoundDay(
        convective_day_12z=start,
        reports=reports,
        counts=counts,
        is_random=False,
    )


def _candidate_days(
    spec: ThresholdSpec,
    range_start: datetime,
    range_end: datetime,
) -> list[datetime] | None:
    """Build the list of dates to sample from.

    When ``spec`` carries a tornado-EF floor, we restrict the random
    pool to SVRGIS days that actually contain a tornado at that EF —
    a 70 k-row CSV filter is instant and avoids the wasteful
    "sample uniformly + reject" pattern. For EF4+ that's ~50 days
    across the full archive; an unguided uniform-over-9000-days
    sampler would burn dozens of IEM HTTP round-trips to land on
    even one.

    Returns ``None`` when the EF floor is unset (caller falls back to
    the uniform sampler) or when SVRGIS load fails (also uniform-
    sampler fallback)."""
    if spec.min_strongest_tornado_ef < 0:
        return None
    try:
        from ..data.spc_svrgis import convective_days_with_min_ef
        days = convective_days_with_min_ef(
            spec.min_strongest_tornado_ef,
            range_start=range_start,
            range_end=range_end,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[round_builder] SVRGIS candidate-day lookup failed: {e}")
        return None
    if not days:
        # Either no days match (very high threshold + small archive)
        # or SVRGIS just returned empty. Either way returning None
        # falls back to the date-range sampler, which will explore
        # the recent (post-publication-lag) days that SVRGIS hasn't
        # ingested yet.
        return None
    return days


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

    Sampling strategy depends on whether a tornado-EF floor is set:

      - **EF floor unset** (default): uniform random sample over
        ``[range_start, range_end]``. The daily-counts cache lets us
        short-circuit known-non-qualifying days, but unfamiliar days
        still pay a network fetch.
      - **EF floor set** (``min_strongest_tornado_ef >= 0``): sample
        from the SVRGIS-derived list of convective days that contain
        at least one tornado at the requested EF. Filtering 70 k
        SVRGIS rows is instant; this collapses "explore 9 000 days
        looking for ~50 EF4+ days" into one O(1) pick from a 50-entry
        list, no wasted HTTP round-trips.

    Raises ``RuntimeError`` after ``max_tries`` if no qualifying day was found —
    typically means the thresholds are unrealistically high.
    """
    rng = rng or random.Random()
    today = today or datetime.now(timezone.utc)
    range_end = today.replace(hour=12, minute=0, second=0, microsecond=0) - timedelta(days=today_lag_days)
    if range_end <= range_start:
        raise ValueError("Effective date range is empty (range_start ≥ range_end)")
    total_days = (range_end - range_start).days

    # If an EF floor is set, build a pre-filtered candidate-day list
    # from SVRGIS. Falls back to uniform sampling when SVRGIS is
    # unavailable or the floor wasn't requested.
    candidate_days = _candidate_days(spec, range_start, range_end)
    if candidate_days is not None:
        print(f"[round_builder] EF floor "
              f"{spec.min_strongest_tornado_ef:.0f}+ → "
              f"{len(candidate_days)} SVRGIS-qualified candidate days")

    # The daily-counts index lets us skip non-qualifying days without a fresh
    # network round-trip per attempt. For days we've never indexed we still
    # fetch (which is what populates the index).
    seen_keys: set[str] = set()
    for _ in range(max_tries):
        if candidate_days is not None:
            # SVRGIS-driven path: pick from the pre-filtered list. Once
            # every day's been tried, stop — there's nothing more to
            # explore.
            unseen = [d for d in candidate_days
                      if d.strftime("%Y-%m-%d") not in seen_keys]
            if not unseen:
                break
            candidate = rng.choice(unseen)
        else:
            # Uniform-sample path (no EF floor).
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
