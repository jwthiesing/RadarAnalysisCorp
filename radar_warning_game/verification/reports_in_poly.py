"""Match storm reports to active warnings (5 km buffered geometry, time windows).

This module defines the warning / MCD data model used by the scoring engine
(:mod:`radar_warning_game.verification.scoring`) and answers the question:

  "Given a list of warnings and a list of reports, which reports verify which
  warnings?"

Each warning is a sequence of timestamped revisions (plan §5). At any moment its
*effective* type / polygon / magnitudes are those of the most-recent revision.

For verification:
  - The polygon test uses the revision active at the report's time.
  - The lead-time test uses the warning's **original** issue time (not the
    revision time) so a tier upgrade doesn't penalize already-earned lead.
  - Late-warn POD credit (warning issued *after* the report) is only allowed
    for TORR, within ``TORR_LATE_WARN_WINDOW`` minutes (plan §6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..data.reports import Report
from ..geo.polygons import Polygon, contains_with_buffer
from .tornado_tiers import (
    TORR_LATE_WARN_WINDOW,
    WarningType,
    allows_late_warn,
    verifies_warning_type,
)

DEFAULT_VERIFICATION_BUFFER_KM = 5.0


@dataclass(frozen=True)
class Magnitudes:
    """Per-warning expected magnitudes (whichever apply to the type)."""

    hail_in: float | None = None
    wind_mph: float | None = None
    ef: float | None = None


@dataclass(frozen=True)
class WarningRevision:
    """One revision in a warning's history (issuance counts as revision 0)."""

    revision_time: datetime
    warning_type: WarningType
    polygon: Polygon
    duration: timedelta
    magnitudes: Magnitudes


@dataclass
class Warning:
    """A player-issued warning with full revision history."""

    warning_id: str
    issuer_id: str
    team_id: str                       # equals issuer_id in solo mode
    revisions: list[WarningRevision]   # sorted by revision_time, first is original issue
    canceled_at: datetime | None = None

    @property
    def original_issue_time(self) -> datetime:
        return self.revisions[0].revision_time

    @property
    def current_revision(self) -> WarningRevision:
        return self.revisions[-1]

    def revision_at(self, t: datetime) -> WarningRevision | None:
        """Revision active at time ``t``. Returns ``None`` if t is before original issue."""
        active = None
        for rev in self.revisions:
            if rev.revision_time <= t:
                active = rev
            else:
                break
        return active

    def end_time(self) -> datetime:
        """When the warning expires (current revision's issue_time + duration), or cancel."""
        rev = self.current_revision
        natural = rev.revision_time + rev.duration
        if self.canceled_at is not None:
            return min(natural, self.canceled_at)
        return natural

    def is_active_at(self, t: datetime) -> bool:
        if t < self.original_issue_time:
            return False
        if self.canceled_at is not None and t > self.canceled_at:
            return False
        # The current revision's duration applies; revising mid-warning extends.
        return t <= self.current_revision.revision_time + self.current_revision.duration


@dataclass
class MCD:
    """Mesoscale Convective Discussion with PIB picks (plan §8)."""

    mcd_id: str
    issuer_id: str
    team_id: str
    polygon: Polygon
    issue_time: datetime
    duration: timedelta
    pib_tornado: int                   # 0 = None, 1-7
    pib_wind: int                      # 0 = None, 1-7
    pib_hail: int                      # 0 = None, 1-6
    canceled_at: datetime | None = None

    def end_time(self) -> datetime:
        natural = self.issue_time + self.duration
        return min(natural, self.canceled_at) if self.canceled_at else natural


@dataclass(frozen=True)
class VerifyingMatch:
    """A single (warning, report) verification result."""

    warning: Warning
    revision: WarningRevision          # the revision active at report time
    report: Report
    lead_time: timedelta               # report.time - warning.original_issue_time
    late_warn: bool                    # True iff issued after report (TORR-only)


def reports_in_polygon(
    polygon: Polygon,
    reports: list[Report],
    *,
    buffer_km: float = DEFAULT_VERIFICATION_BUFFER_KM,
    time_window: tuple[datetime, datetime] | None = None,
) -> list[Report]:
    """All reports inside ``polygon`` (with 5 km buffer) within ``time_window``."""
    out: list[Report] = []
    t0, t1 = time_window if time_window else (None, None)
    for r in reports:
        if t0 is not None and r.time < t0:
            continue
        if t1 is not None and r.time > t1:
            continue
        if contains_with_buffer(polygon, r.lat, r.lon, buffer_km=buffer_km):
            out.append(r)
    return out


def find_verifying_reports(
    warning: Warning,
    reports: list[Report],
    *,
    buffer_km: float = DEFAULT_VERIFICATION_BUFFER_KM,
) -> list[VerifyingMatch]:
    """Find reports that verify ``warning`` per plan §6 rules.

    A report verifies if:
      1. Its category meets the active revision's warning type (e.g. tornado for TOR-family).
      2. It lies inside the active revision's polygon ⊕ ``buffer_km``.
      3. Its time falls in the warning's valid window — with the late-warn allowance
         for TORR (warning issued ≤10 min after the report still counts).
    """
    out: list[VerifyingMatch] = []
    issue_time = warning.original_issue_time
    end_time = warning.end_time()
    for report in reports:
        # Time window
        if report.time > end_time:
            continue
        if report.time < issue_time:
            # only TORR can claim a report that predates the warning, within window
            if not allows_late_warn(warning.current_revision.warning_type):
                continue
            if issue_time - report.time > TORR_LATE_WARN_WINDOW:
                continue
        # Which revision was active at the report's time?
        rev = warning.revision_at(report.time)
        if rev is None:
            # Late-warn case: report predates issue_time. Use the original revision.
            rev = warning.revisions[0]
        # Type-of-report check
        if not verifies_warning_type(rev.warning_type, report.category, report.magnitude):
            continue
        # Polygon test against the active revision's polygon
        if not contains_with_buffer(rev.polygon, report.lat, report.lon, buffer_km=buffer_km):
            continue
        lead = report.time - issue_time
        out.append(
            VerifyingMatch(
                warning=warning,
                revision=rev,
                report=report,
                lead_time=lead,
                late_warn=report.time < issue_time,
            )
        )
    return out


def reports_in_mcd(
    mcd: MCD,
    reports: list[Report],
    *,
    buffer_km: float = DEFAULT_VERIFICATION_BUFFER_KM,
) -> list[Report]:
    """All reports that fall inside the MCD's polygon during its valid time.

    MCDs don't restrict by hazard type at the match step — PIB scoring (§8)
    separates the verifying reports by category itself.
    """
    return reports_in_polygon(
        mcd.polygon,
        reports,
        buffer_km=buffer_km,
        time_window=(mcd.issue_time, mcd.end_time()),
    )
