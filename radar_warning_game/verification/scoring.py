"""End-of-round (and live) scoring for warnings and MCDs.

Implements the rules from plan §6 (verification), §7 (tier multipliers) and §8
(MCD PIB scoring). Works for solo or team play uniformly — a solo player is
modeled as a team of one.

A team's score is the sum of:

  - **Warning score**: per-warning ``tier_mult × (1 + magnitude_accuracy_bonus) × base_pts``
    summed over verified warnings, minus ``fa_penalty_mult × base_fa_pts`` for false alarms.
  - **MCD score**: per-MCD ``PIB_accuracy_score + lead_bonus + breadth_bonus``,
    minus penalties for false-alarm hazards or unpredicted significant hazards.

Reports inside the *game polygon* (not just any warning) form the denominator for
team POD. A team's verifying warning set = union of teammates' warnings; if any
teammate verified a report, the team gets credit once (no double-counting).

This module owns the *numeric* policy. All thresholds / multipliers live in
:mod:`tornado_tiers` and :mod:`pibs`; this module just composes them.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from ..data.reports import Report
from ..geo.polygons import Polygon, contains_with_buffer
from .pibs import max_pib_for_category, observed_to_pib
from .reports_in_poly import (
    DEFAULT_VERIFICATION_BUFFER_KM,
    MCD,
    VerifyingMatch,
    Warning,
    find_verifying_reports,
    reports_in_mcd,
    reports_in_polygon,
)
from .tornado_tiers import (
    PDS_TOR_MIN_EF,
    WarningType,
    severe_tier_multiplier,
    tornado_tier_multiplier,
)

# ---------------------------- base point values ------------------------------
# Tunable knobs; collected here for easy adjustment.

BASE_SVR_POINTS = 100.0
BASE_TOR_POINTS = 200.0
BASE_SVR_FA_PENALTY = 60.0
BASE_TOR_FA_PENALTY = 120.0

MAG_ACCURACY_WEIGHT = 0.5   # max boost from perfect magnitude prediction (0 → 1.0×, 1 → 1.5×)

# MCD scoring
MCD_PIB_PERFECT_HAZARD_POINTS = 80.0  # per hazard for delta=0 (max accuracy)
MCD_PIB_FA_PENALTY_PER_PIB = 20.0    # for predicting PIB N when observed=0; scales with N
MCD_PIB_MISS_PENALTY_PER_PIB = 25.0  # for predicting None when PIB observed; scales with N
MCD_LEAD_MAX_POINTS = 60.0           # max for ≥60 min lead on first verifying report
MCD_BREADTH_TWO_HAZARDS = 40.0
MCD_BREADTH_ALL_THREE = 100.0


# ---------------------------- result types ------------------------------------

@dataclass
class WarningScore:
    """Detailed scoring for a single warning."""

    warning: Warning
    verifying: list[VerifyingMatch]
    tier_mult: float
    fa_penalty_mult: float
    magnitude_bonus: float          # 0..MAG_ACCURACY_WEIGHT
    points: float                   # signed: positive if verified, negative if FA
    is_false_alarm: bool


@dataclass
class MCDScore:
    """Detailed scoring for a single MCD."""

    mcd: MCD
    verifying_reports: list[Report]
    hazard_scores: dict[str, float] = field(default_factory=dict)   # {'tornado': pts, ...}
    lead_bonus: float = 0.0
    breadth_bonus: float = 0.0
    points: float = 0.0


@dataclass
class TeamScore:
    """Aggregated scoring for one team (solo player = team of 1)."""

    team_id: str
    member_ids: list[str]
    warnings_total: float = 0.0
    mcd_total: float = 0.0
    pod: float = 0.0
    far: float = 0.0
    csi: float = 0.0
    mean_lead_time_sec: float = 0.0
    p25_lead_time_sec: float = 0.0
    p75_lead_time_sec: float = 0.0
    median_lead_time_sec: float = 0.0
    n_warnings: int = 0
    n_false_alarms: int = 0
    n_verifying_reports: int = 0
    n_total_reports_in_game: int = 0
    per_warning: list[WarningScore] = field(default_factory=list)
    per_mcd: list[MCDScore] = field(default_factory=list)

    @property
    def total(self) -> float:
        return self.warnings_total + self.mcd_total


# ---------------------------- warning scoring ---------------------------------

def _peak_observed_ef(matches: list[VerifyingMatch]) -> float:
    """Strongest verifying tornado EF; -1 if no tornado verified or all unknown EF."""
    peak = -1.0
    for m in matches:
        if m.report.category == "tornado" and m.report.magnitude > peak:
            peak = m.report.magnitude
    return peak


def _peak_observed_hail(matches: list[VerifyingMatch]) -> float:
    return max((m.report.magnitude for m in matches if m.report.category == "hail"), default=0.0)


def _peak_observed_wind(matches: list[VerifyingMatch]) -> float:
    return max((m.report.magnitude for m in matches if m.report.category == "wind"), default=0.0)


def _sum_casualties(matches: list[VerifyingMatch]) -> int:
    return sum(m.report.injuries + m.report.fatalities for m in matches)


def _magnitude_accuracy(warning, matches: list[VerifyingMatch]) -> float:
    """Return a 0..1 accuracy fraction for the warning's magnitude estimates.

    Per plan §5: peak observed is computed over the full warning lifetime, and
    magnitude accuracy uses the **revision active when that peak was observed**
    — so a player who initially nailed an estimate and later revised it isn't
    penalized for the later revision.

    SVR is scored component-wise (hail + wind separately):
      - If the player predicted a value AND a report of that type verified:
        standard accuracy = 1 − |predicted − peak observed| / peak observed.
      - If the player predicted a value but NO report of that type verified
        within the warning: contribute 0 (penalizes unrealized over-predictions).
      - If the player predicted no value for that component (None): component
        skipped entirely.
      - Final magnitude = mean of the contributing components.
    """
    warning_type = warning.current_revision.warning_type
    if warning_type.is_severe_family:
        components: list[float] = []
        cur_mag = warning.current_revision.magnitudes
        # Hail component
        if cur_mag.hail_in is not None:
            hail_matches = [m for m in matches if m.report.category == "hail"]
            if hail_matches:
                peak = max(hail_matches, key=lambda m: m.report.magnitude)
                if peak.report.magnitude > 0:
                    rev = warning.revision_at(peak.report.time) or warning.revisions[0]
                    if rev.magnitudes.hail_in is not None:
                        err = abs(rev.magnitudes.hail_in - peak.report.magnitude) / max(peak.report.magnitude, 0.5)
                        components.append(max(0.0, 1.0 - err))
                    else:
                        components.append(0.0)
                else:
                    components.append(0.0)
            else:
                # Predicted but no hail report verified — count as zero accuracy
                components.append(0.0)
        # Wind component
        if cur_mag.wind_mph is not None:
            wind_matches = [m for m in matches if m.report.category == "wind"]
            if wind_matches:
                peak = max(wind_matches, key=lambda m: m.report.magnitude)
                if peak.report.magnitude > 0:
                    rev = warning.revision_at(peak.report.time) or warning.revisions[0]
                    if rev.magnitudes.wind_mph is not None:
                        err = abs(rev.magnitudes.wind_mph - peak.report.magnitude) / max(peak.report.magnitude, 30.0)
                        components.append(max(0.0, 1.0 - err))
                    else:
                        components.append(0.0)
                else:
                    components.append(0.0)
            else:
                components.append(0.0)
        return sum(components) / len(components) if components else 0.0
    if warning_type.is_tornado_family:
        # Peak EF: scored against revision active at the peak EF's report time
        tor_matches = [m for m in matches if m.report.category == "tornado" and m.report.magnitude >= 0]
        if tor_matches:
            peak = max(tor_matches, key=lambda m: m.report.magnitude)
            rev = warning.revision_at(peak.report.time) or warning.revisions[0]
            if rev.magnitudes.ef is not None:
                err = abs(rev.magnitudes.ef - peak.report.magnitude) / max(peak.report.magnitude, 1.0)
                return max(0.0, 1.0 - err)
        return 0.0
    return 0.0


def _tier_multiplier_for_match(warning_type: WarningType, match: VerifyingMatch) -> float:
    """Per-match tier multiplier — uses ``warning_type`` (the revision active at
    report time) plus the report's own severity to decide if PDS/TORE bonuses fire.
    """
    r = match.report
    if warning_type.is_tornado_family:
        ef = r.magnitude if r.category == "tornado" else -1.0
        casualties = r.injuries + r.fatalities
        return tornado_tier_multiplier(warning_type,
                                        peak_observed_ef=ef,
                                        casualties=casualties).score_multiplier
    if warning_type.is_severe_family:
        hail = r.magnitude if r.category == "hail" else 0.0
        wind = r.magnitude if r.category == "wind" else 0.0
        return severe_tier_multiplier(warning_type,
                                       peak_hail_in=hail,
                                       peak_wind_mph=wind).score_multiplier
    return 1.0


def score_single_warning(warning: Warning, reports: list[Report]) -> WarningScore:
    """Compute the score earned (or lost) by one warning.

    Verified warnings use **per-match revision-aware tier multipliers** (plan §5):
    each verifying report scores against the revision active at its time. The
    warning's total score is the mean of per-match contributions × base points.
    A warning that was upgraded SVR→TOR mid-event earns SVR tier for hail/wind
    reports during the SVR window and TOR tier for tornado reports after the
    upgrade.

    FA penalties use the current revision's tier (it's the player's final claim).
    """
    matches = find_verifying_reports(warning, reports)
    is_fa = len(matches) == 0
    current_type = warning.current_revision.warning_type

    mag_acc = _magnitude_accuracy(warning, matches)
    mag_bonus = mag_acc * MAG_ACCURACY_WEIGHT

    if is_fa:
        # FA: tier from the most-recent revision
        if current_type.is_tornado_family:
            tier = tornado_tier_multiplier(current_type, peak_observed_ef=-1, casualties=0)
            base_fa = BASE_TOR_FA_PENALTY
        else:
            tier = severe_tier_multiplier(current_type, peak_hail_in=0.0, peak_wind_mph=0.0)
            base_fa = BASE_SVR_FA_PENALTY
        return WarningScore(
            warning=warning, verifying=matches,
            tier_mult=tier.score_multiplier, fa_penalty_mult=tier.fa_penalty_multiplier,
            magnitude_bonus=mag_bonus,
            points=-tier.fa_penalty_multiplier * base_fa,
            is_false_alarm=True,
        )

    # Verified: per-match per-revision tier. Base points depend on the warning's
    # current family (which dictates BASE_TOR_POINTS vs BASE_SVR_POINTS).
    base_pts = BASE_TOR_POINTS if current_type.is_tornado_family else BASE_SVR_POINTS
    per_match_mults = [
        _tier_multiplier_for_match(m.revision.warning_type, m) for m in matches
    ]
    mean_tier = sum(per_match_mults) / len(per_match_mults)
    points = mean_tier * (1.0 + mag_bonus) * base_pts
    return WarningScore(
        warning=warning, verifying=matches,
        tier_mult=mean_tier,
        fa_penalty_mult=1.0,    # not used when verified
        magnitude_bonus=mag_bonus,
        points=points,
        is_false_alarm=False,
    )


# ---------------------------- MCD scoring -------------------------------------

def _observed_pib_for_category(reports_in: list[Report], category: str) -> int:
    """Highest observed PIB inside the MCD polygon during its valid time."""
    relevant = [r for r in reports_in if r.category == category]
    if not relevant:
        return 0
    peak = max(r.magnitude for r in relevant)
    return observed_to_pib(category, peak)


def _score_pib_pair(predicted: int, observed: int, category: str) -> float:
    """Score for a single hazard prediction."""
    max_p = max_pib_for_category(category)
    if predicted == 0 and observed == 0:
        # Correct "None" — small credit, no big bonus
        return MCD_PIB_PERFECT_HAZARD_POINTS * 0.25
    if predicted == 0 and observed > 0:
        # Miss — penalty scaled with observed magnitude
        return -MCD_PIB_MISS_PENALTY_PER_PIB * observed
    if predicted > 0 and observed == 0:
        # False alarm — penalty scaled with predicted magnitude
        return -MCD_PIB_FA_PENALTY_PER_PIB * predicted
    # Both nonzero — accuracy score
    delta = abs(predicted - observed)
    accuracy = max(0.0, 1.0 - delta / max_p)
    return MCD_PIB_PERFECT_HAZARD_POINTS * accuracy


def _lead_bonus_for_mcd(mcd: MCD, in_poly: list[Report]) -> float:
    """Per-hazard lead-time bonus.

    For each hazard the MCD predicted (non-None PIB), compute lead time vs the
    first verifying report of that hazard. Average across predicted hazards.
    A predicted hazard with no verifying report contributes 0 lead.
    """
    predicted_hazards = [
        (cat, pib)
        for cat, pib in (("tornado", mcd.pib_tornado),
                         ("wind", mcd.pib_wind),
                         ("hail", mcd.pib_hail))
        if pib > 0
    ]
    if not predicted_hazards or not in_poly:
        return 0.0
    per_hazard_bonuses: list[float] = []
    for cat, _ in predicted_hazards:
        cat_reports = [r for r in in_poly if r.category == cat]
        if not cat_reports:
            per_hazard_bonuses.append(0.0)
            continue
        first = min(cat_reports, key=lambda r: r.time)
        lead_sec = (first.time - mcd.issue_time).total_seconds()
        if lead_sec <= 0:
            per_hazard_bonuses.append(0.0)
        else:
            per_hazard_bonuses.append(
                min(MCD_LEAD_MAX_POINTS, MCD_LEAD_MAX_POINTS * (lead_sec / 3600.0))
            )
    return sum(per_hazard_bonuses) / len(per_hazard_bonuses)


def score_single_mcd(mcd: MCD, reports: list[Report]) -> MCDScore:
    in_poly = reports_in_mcd(mcd, reports)
    hazard_scores: dict[str, float] = {}
    for cat, predicted in (
        ("tornado", mcd.pib_tornado),
        ("wind", mcd.pib_wind),
        ("hail", mcd.pib_hail),
    ):
        observed = _observed_pib_for_category(in_poly, cat)
        hazard_scores[cat] = _score_pib_pair(predicted, observed, cat)

    # Breadth bonus: how many hazards were predicted AND verified
    predicted_nonzero = sum(1 for p in (mcd.pib_tornado, mcd.pib_wind, mcd.pib_hail) if p > 0)
    verified_nonzero = sum(
        1 for cat in ("tornado", "wind", "hail")
        if _observed_pib_for_category(in_poly, cat) > 0
    )
    breadth_bonus = 0.0
    if predicted_nonzero >= 2 and verified_nonzero >= 2:
        breadth_bonus = MCD_BREADTH_TWO_HAZARDS
    if predicted_nonzero == 3 and verified_nonzero == 3:
        breadth_bonus = MCD_BREADTH_ALL_THREE

    lead_bonus = _lead_bonus_for_mcd(mcd, in_poly)
    total = sum(hazard_scores.values()) + lead_bonus + breadth_bonus

    return MCDScore(
        mcd=mcd,
        verifying_reports=in_poly,
        hazard_scores=hazard_scores,
        lead_bonus=lead_bonus,
        breadth_bonus=breadth_bonus,
        points=total,
    )


# ---------------------------- team aggregation --------------------------------

def score_team(
    team_id: str,
    member_ids: list[str],
    warnings: list[Warning],
    mcds: list[MCD],
    reports_in_game: list[Report],
    game_polygon: Polygon,
    *,
    buffer_km: float = DEFAULT_VERIFICATION_BUFFER_KM,
) -> TeamScore:
    """Compute one team's full score.

    ``reports_in_game`` should be all reports inside the game polygon during the
    session (the denominator for POD calculations). ``warnings`` and ``mcds``
    must be filtered to those issued by this team.
    """
    per_warning = [score_single_warning(w, reports_in_game) for w in warnings]
    per_mcd = [score_single_mcd(m, reports_in_game) for m in mcds]

    # POD: count of game-polygon reports that any team warning verified.
    verified_report_ids: set[int] = set()
    for ws in per_warning:
        for m in ws.verifying:
            verified_report_ids.add(id(m.report))

    # FAR
    n_fa = sum(1 for ws in per_warning if ws.is_false_alarm)
    n_total = len(per_warning)
    n_verified_reports = len(verified_report_ids)
    n_reports_in_game = len(reports_in_game)

    pod = n_verified_reports / n_reports_in_game if n_reports_in_game else 0.0
    far = n_fa / n_total if n_total else 0.0
    # Critical Success Index: hits / (hits + misses + false alarms)
    misses = max(0, n_reports_in_game - n_verified_reports)
    denom = n_verified_reports + misses + n_fa
    csi = n_verified_reports / denom if denom else 0.0

    # Lead-time distribution across verifying matches (positive = lead,
    # negative = TORR late-warn). We report mean, median, and quartiles so a
    # few late-warn TORRs don't pull the average misleadingly negative.
    leads: list[float] = []
    for ws in per_warning:
        for m in ws.verifying:
            leads.append(m.lead_time.total_seconds())
    if leads:
        import statistics
        mean_lead = sum(leads) / len(leads)
        median_lead = statistics.median(leads)
        # quantiles needs n>=2 for n=4 quartiles
        if len(leads) >= 2:
            q = statistics.quantiles(leads, n=4)
            p25, p75 = float(q[0]), float(q[2])
        else:
            p25 = p75 = median_lead
    else:
        mean_lead = median_lead = p25 = p75 = 0.0

    warnings_total = sum(ws.points for ws in per_warning)
    mcd_total = sum(ms.points for ms in per_mcd)

    return TeamScore(
        team_id=team_id,
        member_ids=list(member_ids),
        warnings_total=warnings_total,
        mcd_total=mcd_total,
        pod=pod,
        far=far,
        csi=csi,
        mean_lead_time_sec=mean_lead,
        median_lead_time_sec=median_lead,
        p25_lead_time_sec=p25,
        p75_lead_time_sec=p75,
        n_warnings=n_total,
        n_false_alarms=n_fa,
        n_verifying_reports=n_verified_reports,
        n_total_reports_in_game=n_reports_in_game,
        per_warning=per_warning,
        per_mcd=per_mcd,
    )


def score_round(
    teams: dict[str, list[str]],          # {team_id: [player_id, ...]}
    warnings_by_player: dict[str, list[Warning]],
    mcds_by_player: dict[str, list[MCD]],
    all_reports: list[Report],
    game_polygon: Polygon,
    *,
    buffer_km: float = DEFAULT_VERIFICATION_BUFFER_KM,
) -> list[TeamScore]:
    """Score every team in a round. Returns a list sorted high → low by total.

    Note: the POD denominator uses the game polygon with the same 5 km buffer
    that warning verification uses, so a report 3 km outside the strict
    polygon edge that verifies a warning is also counted in the denominator
    (symmetrical accounting).
    """
    reports_in_game = reports_in_polygon(game_polygon, all_reports, buffer_km=buffer_km)
    results: list[TeamScore] = []
    for team_id, members in teams.items():
        team_warnings = [w for m in members for w in warnings_by_player.get(m, [])]
        team_mcds = [d for m in members for d in mcds_by_player.get(m, [])]
        results.append(
            score_team(
                team_id, members, team_warnings, team_mcds,
                reports_in_game, game_polygon, buffer_km=buffer_km,
            )
        )
    results.sort(key=lambda t: t.total, reverse=True)
    return results
