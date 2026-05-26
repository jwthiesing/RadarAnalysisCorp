"""Centralized timestamp formatters for date-blinding (plan §4b).

Every UI-facing timestamp goes through one of these helpers. The date is
deliberately *omitted* — players see ``HH:MM:SS Z`` and never the calendar date
of the event they're playing. The helpers also guard against accidentally
formatting naive datetimes (we always render as UTC).

There is a separate :func:`format_dev_time` for log files / debug output where
date visibility is OK and useful.

A lint check (or unit test) should scan rendered UI strings for any
``YYYY-MM-DD`` pattern and fail if one slips through.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

DATE_LIKE_RE = re.compile(r"(?<!\d)\d{4}[-/]\d{2}[-/]\d{2}(?!\d)")


def _as_utc(dt: datetime) -> datetime:
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def format_player_time(dt: datetime) -> str:
    """Full-precision time-of-day in UTC: ``"20:48:12Z"``."""
    return f"{_as_utc(dt):%H:%M:%S}Z"


def format_player_time_short(dt: datetime) -> str:
    """Minute-precision: ``"20:48Z"``."""
    return f"{_as_utc(dt):%H:%M}Z"


def format_player_offset(seconds: float) -> str:
    """Signed offset shown as ``"+05:23"`` or ``"-00:42"``. For lead-time displays."""
    sign = "+" if seconds >= 0 else "-"
    s = int(abs(seconds))
    return f"{sign}{s // 60:02d}:{s % 60:02d}"


def format_dev_time(dt: datetime) -> str:
    """Full ISO datetime — for logs ONLY, never UI."""
    return _as_utc(dt).isoformat()


def contains_date(text: str) -> bool:
    """True if ``text`` contains anything that looks like a calendar date.

    Used by the date-blinding lint test to catch UI strings that accidentally
    include a date.
    """
    return DATE_LIKE_RE.search(text) is not None
