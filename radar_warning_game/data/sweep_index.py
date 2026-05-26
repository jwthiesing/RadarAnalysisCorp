"""SAILS-aware per-sweep index across loaded NEXRAD Level 2 volumes.

NEXRAD's SAILS / MESO-SAILS scan strategies insert additional 0.5-degree sweeps
mid-volume — a single Level 2 file can hold 1, 2, 3, or 4 sweeps at the lowest
elevation, while higher tilts have exactly one sweep per volume. Indexing only by
volume start time produces a jumpy ~5-min cadence at low elevation; indexing by
*sweep* start time produces the smooth ~60-90 second cadence forecasters expect.

This module builds a global per-radar index of ``SweepRef(start_time, elev_deg,
file, sweep_number)`` and answers two questions:

  - "scrub left/right at elevation E from time T" → next/prev sweep with similar elev
  - "change elevation to E at fixed time T" → closest tilt available in the nearest volume

No special-casing per VCP. The SAILS configuration is detected empirically by
counting per-elevation sweeps in each volume.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock

import numpy as np
import pyart

ELEV_TOLERANCE_DEG = 0.15


@dataclass(frozen=True)
class SweepRef:
    site: str
    start_time: datetime
    elev_deg: float
    file: Path
    sweep_number: int

    def __repr__(self) -> str:
        return (
            f"SweepRef({self.site} t={self.start_time:%H:%M:%S} "
            f"el={self.elev_deg:.2f}° sw={self.sweep_number})"
        )


@lru_cache(maxsize=64)
def index_volume_file(file: Path) -> tuple[SweepRef, ...]:
    """Build a sweep-level index of a single Level 2 file.

    Reads only the metadata (delay_field_loading=True) — the actual ray data is
    lazy-loaded when a sweep is rendered.
    """
    radar = pyart.io.read_nexrad_archive(str(file), delay_field_loading=True)
    site = _site_from_radar(radar, fallback_filename=file.name)
    base_time = _parse_units_epoch(radar.time["units"])

    # PyART exposes per-ray seconds-since-volume-start in radar.time['data'];
    # we use the first ray of each sweep as its start time.
    sweep_starts = radar.sweep_start_ray_index["data"]
    time_offsets = radar.time["data"]
    fixed_angles = radar.fixed_angle["data"]
    out: list[SweepRef] = []
    for sweep_no, start_ray in enumerate(sweep_starts):
        t_offset = float(time_offsets[start_ray])
        sweep_time = base_time + _seconds_to_timedelta(t_offset)
        elev = float(fixed_angles[sweep_no])
        out.append(
            SweepRef(
                site=site,
                start_time=sweep_time,
                elev_deg=elev,
                file=file,
                sweep_number=int(sweep_no),
            )
        )
    return tuple(out)


class SweepIndex:
    """Mutable per-radar index of all loaded sweeps, sorted by start time.

    One instance per site (typically). Thread-safe.
    """

    def __init__(self, site: str) -> None:
        self.site = site.upper()
        self._lock = RLock()
        self._sweeps: list[SweepRef] = []          # sorted by start_time
        self._times: list[datetime] = []           # parallel to _sweeps; lets us bisect
        self._seen_files: set[Path] = set()

    def add_file(self, file: Path) -> int:
        """Index a Level 2 file's sweeps. Returns number of new sweeps added."""
        if file in self._seen_files:
            return 0
        new_refs = [r for r in index_volume_file(file) if r.site == self.site]
        with self._lock:
            self._seen_files.add(file)
            for r in new_refs:
                pos = bisect.bisect_left(self._times, r.start_time)
                self._sweeps.insert(pos, r)
                self._times.insert(pos, r.start_time)
        return len(new_refs)

    def all_sweeps(self) -> list[SweepRef]:
        with self._lock:
            return list(self._sweeps)

    def at_elevation(self, elev_deg: float, *, tol: float = ELEV_TOLERANCE_DEG) -> list[SweepRef]:
        """All indexed sweeps whose elevation is within ``tol`` of ``elev_deg``."""
        with self._lock:
            return [s for s in self._sweeps if abs(s.elev_deg - elev_deg) < tol]

    def latest_at_or_before(
        self,
        time: datetime,
        elev_deg: float,
        *,
        tol: float = ELEV_TOLERANCE_DEG,
    ) -> SweepRef | None:
        """Most recent sweep at ``elev_deg`` whose start_time ≤ ``time``."""
        candidates = self.at_elevation(elev_deg, tol=tol)
        best: SweepRef | None = None
        for s in candidates:
            if s.start_time <= time and (best is None or s.start_time > best.start_time):
                best = s
        return best

    def step_in_elevation(
        self,
        current: SweepRef,
        step: int,
        *,
        tol: float = ELEV_TOLERANCE_DEG,
    ) -> SweepRef | None:
        """Walk ``step`` sweeps forward (positive) or backward (negative) at the same elevation."""
        same_el = self.at_elevation(current.elev_deg, tol=tol)
        same_el.sort(key=lambda s: s.start_time)
        try:
            idx = same_el.index(current)
        except ValueError:
            return None
        new_idx = idx + step
        if 0 <= new_idx < len(same_el):
            return same_el[new_idx]
        return None

    def nearest_elevation(self, time: datetime, target_elev: float) -> SweepRef | None:
        """At a fixed ``time``, find the sweep with elevation closest to ``target_elev``.

        Looks at sweeps within roughly one volume cadence (±5 min) of ``time``.
        """
        with self._lock:
            window_start = bisect.bisect_left(self._times, _add_seconds(time, -300))
            window_end = bisect.bisect_right(self._times, _add_seconds(time, 300))
            window = self._sweeps[window_start:window_end]
        if not window:
            return None
        return min(window, key=lambda s: abs(s.elev_deg - target_elev))

    def available_elevations(self, time: datetime, *, window_sec: int = 300) -> list[float]:
        """Sorted unique elevations available in volumes near ``time`` (one-volume window)."""
        with self._lock:
            ws = bisect.bisect_left(self._times, _add_seconds(time, -window_sec))
            we = bisect.bisect_right(self._times, _add_seconds(time, window_sec))
            elevs = sorted({round(s.elev_deg, 2) for s in self._sweeps[ws:we]})
        return elevs


# --------------------------- helpers ------------------------------------------

def _parse_units_epoch(units: str) -> datetime:
    """Parse a CF-style ``'seconds since YYYY-MM-DDTHH:MM:SSZ'`` string into a UTC datetime.

    PyART's ``radar.time['units']`` reliably has this format for NEXRAD volumes.
    """
    prefix = "seconds since "
    s = units.strip()
    if s.startswith(prefix):
        s = s[len(prefix):]
    s = s.rstrip("Z").rstrip()
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"Unrecognized time units string: {units!r}")


def _seconds_to_timedelta(seconds: float):
    from datetime import timedelta as _td
    return _td(seconds=seconds)


def _add_seconds(t: datetime, seconds: int) -> datetime:
    from datetime import timedelta as _td
    return t + _td(seconds=seconds)


def _site_from_radar(radar, *, fallback_filename: str) -> str:
    """Extract the ICAO from a PyART radar object, fallback to filename."""
    try:
        meta = radar.metadata or {}
        instr = meta.get("instrument_name") or meta.get("instrument") or ""
        instr = str(instr).strip().upper()
        if len(instr) == 4:
            return instr
    except AttributeError:
        pass
    # Fallback: first 4 chars of e.g. "KTLX20130520_204812_V06"
    return fallback_filename[:4].upper()
