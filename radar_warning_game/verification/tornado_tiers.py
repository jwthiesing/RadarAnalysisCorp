"""Tornado warning tier thresholds + per-tier scoring multipliers (plan §7).

NWS Impact-Based Warning (IBW) ladder for tornado warnings:

  - **TOR**:     base tornado warning. Any tornado verifies.
  - **TORR**:    "Considerable" tag — typically radar-indicated strong rotation.
                 Earns a small bonus over plain TOR. **Allowed late-warn POD**
                 (warning issued within 10 min after the report still verifies),
                 because real TORRs are often issued post-spotter-confirmation.
  - **PDS TOR**: "Particularly Dangerous Situation" — large bonus when verified by
                 a significant tornado (EF2+) OR injuries/fatalities; no bonus for
                 weak verification; heavier false-alarm penalty than plain TOR.
  - **TORE**:    "Tornado Emergency" — biggest bonus for significant verification;
                 reduced score for weak verification (penalizes over-issuance);
                 heaviest false-alarm penalty.

Severe Thunderstorm warning ladder is analogous:

  - **SVR**: base. Hail ≥1.0" OR Wind ≥58 mph verifies.
  - **SVRC**: "Considerable" tag — bonus if peak hail ≥1.75" OR wind ≥70 mph.
  - **SVRD**: "Destructive" tag — bigger bonus if peak hail ≥2.75" OR wind ≥80 mph.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from enum import Enum

TORR_LATE_WARN_WINDOW = timedelta(minutes=10)
"""How long after a report a TORR may still claim POD credit. Plan §6."""

PDS_TOR_MIN_EF = 2
"""Minimum EF rating for a PDS TOR to earn the upgraded multiplier."""

TORE_MIN_EF = 2
"""Minimum EF rating for a TORE to earn the upgraded multiplier."""

SVRC_HAIL_THRESHOLD_IN = 1.75
SVRC_WIND_THRESHOLD_MPH = 70.0
SVRD_HAIL_THRESHOLD_IN = 2.75
SVRD_WIND_THRESHOLD_MPH = 80.0


class WarningType(str, Enum):
    SVR = "SVR"
    SVRC = "SVRC"
    SVRD = "SVRD"
    TOR = "TOR"
    TORR = "TORR"
    PDS_TOR = "PDS_TOR"
    TORE = "TORE"

    @property
    def is_tornado_family(self) -> bool:
        return self in (WarningType.TOR, WarningType.TORR, WarningType.PDS_TOR, WarningType.TORE)

    @property
    def is_severe_family(self) -> bool:
        return self in (WarningType.SVR, WarningType.SVRC, WarningType.SVRD)


@dataclass(frozen=True)
class TierResult:
    """The multiplier and FA-penalty multiplier earned by a warning at verification time."""

    score_multiplier: float       # multiplies base type points when verified
    fa_penalty_multiplier: float  # multiplies base FA penalty when unverified


def tornado_tier_multiplier(
    warning_type: WarningType,
    *,
    peak_observed_ef: float,
    casualties: int,
) -> TierResult:
    """Compute the tier-aware multiplier for a tornado-family warning.

    ``peak_observed_ef`` should be the strongest verifying tornado EF inside the
    warning polygon, or -1 if unknown (preliminary IEM data often has unknown EF).
    ``casualties`` = injuries + fatalities (sum) summed across verifying reports.

    Cases:
      - TOR:     always 1.0× / 1.0×
      - TORR:    1.10× verified / 1.0× FA  (slight encouragement; standard FA)
      - PDS TOR: 1.75× if EF≥2 or casualties>0; else 1.0×; FA penalty 1.5×
      - TORE:    2.5×  if EF≥2 or casualties>0; else 0.75× (over-issuance); FA 3.0×
    """
    significant = (peak_observed_ef >= PDS_TOR_MIN_EF) or (casualties > 0)
    if warning_type == WarningType.TOR:
        return TierResult(1.0, 1.0)
    if warning_type == WarningType.TORR:
        return TierResult(1.10, 1.0)
    if warning_type == WarningType.PDS_TOR:
        return TierResult(1.75 if significant else 1.0, 1.5)
    if warning_type == WarningType.TORE:
        return TierResult(2.5 if significant else 0.75, 3.0)
    raise ValueError(f"Not a tornado-family warning type: {warning_type}")


def severe_tier_multiplier(
    warning_type: WarningType,
    *,
    peak_hail_in: float,
    peak_wind_mph: float,
) -> TierResult:
    """Compute the tier-aware multiplier for a severe-thunderstorm-family warning.

    ``peak_hail_in`` / ``peak_wind_mph`` are the strongest verifying observations
    inside the polygon, or 0 if no observation of that hazard.

    Cases:
      - SVR:  always 1.0× / 1.0×
      - SVRC: 1.10× if hail ≥1.75" or wind ≥70 mph; else 1.0×; FA 1.0×
      - SVRD: 1.25× if hail ≥2.75" or wind ≥80 mph; else 1.0×; FA 1.5×
    """
    if warning_type == WarningType.SVR:
        return TierResult(1.0, 1.0)
    if warning_type == WarningType.SVRC:
        meets = peak_hail_in >= SVRC_HAIL_THRESHOLD_IN or peak_wind_mph >= SVRC_WIND_THRESHOLD_MPH
        return TierResult(1.10 if meets else 1.0, 1.0)
    if warning_type == WarningType.SVRD:
        meets = peak_hail_in >= SVRD_HAIL_THRESHOLD_IN or peak_wind_mph >= SVRD_WIND_THRESHOLD_MPH
        return TierResult(1.25 if meets else 1.0, 1.5)
    raise ValueError(f"Not a severe-family warning type: {warning_type}")


def allows_late_warn(warning_type: WarningType) -> bool:
    """Only TORR is allowed to claim POD credit when issued after a verifying report."""
    return warning_type == WarningType.TORR


def verifies_warning_type(warning_type: WarningType, report_category: str, magnitude: float) -> bool:
    """Does a single report meet the verifying criteria for ``warning_type``?

    Note this only checks **kind** of report — the time + polygon test is in
    :mod:`radar_warning_game.verification.reports_in_poly`.
    """
    if warning_type.is_tornado_family:
        return report_category == "tornado"
    if warning_type.is_severe_family:
        if report_category == "hail":
            return magnitude >= 1.0
        if report_category == "wind":
            return magnitude >= 58.0
        return False
    return False
