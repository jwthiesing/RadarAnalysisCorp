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

# Flat bonus added to an SVR-family warning that carries the "Tornado
# Possible" IBW tag when a tornado actually occurs inside the polygon
# during its valid time. Tuned to roughly half of BASE_TOR_POINTS — the
# player flagged the possibility but didn't commit to a TOR, so partial
# credit is appropriate. Also rescues the warning from FA classification
# if neither hail nor wind verified.
SVR_TORNADO_POSSIBLE_BONUS = 80.0

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


def claimed_hazards(warning: Warning) -> set[str]:
    """Hazard categories the warning's current revision actually
    *claims* — i.e. those the issuer explicitly tagged.

    For an SVR-family warning, that's any of ``{hail, wind, tornado}``
    based on which magnitude fields are set (hail_in > 0, wind_mph > 0,
    or the tornado_possible flag). For a TOR-family warning it's just
    ``{tornado}``. A bare warning with no tags returns the empty set,
    which the FA logic treats as "any verifying report counts" — same
    as before the partial-verification refactor.

    Used by FA / partial / verified classification (in both
    ``score_single_warning`` and the recap map) to ensure a wind-only
    report against a hail-only-predicted SVR doesn't accidentally
    rescue the warning from FA — the user predicted hail, so hail is
    what must materialize.
    """
    rev = warning.current_revision
    if rev.warning_type.is_tornado_family:
        return {"tornado"}
    mag = rev.magnitudes
    out: set[str] = set()
    if mag.hail_in is not None and mag.hail_in > 0:
        out.add("hail")
    if mag.wind_mph is not None and mag.wind_mph > 0:
        out.add("wind")
    if getattr(mag, "tornado_possible", False):
        out.add("tornado")
    return out


def _reports_in_warning(
    warning: Warning,
    reports: list[Report],
    category: str,
    *,
    buffer_km: float = DEFAULT_VERIFICATION_BUFFER_KM,
) -> list[Report]:
    """All reports of ``category`` inside the warning polygon during its valid
    window — irrespective of whether the report's category verifies the
    warning type. Used to score cross-hazard magnitude predictions (e.g. the
    hail tag on a tornado warning).
    """
    out: list[Report] = []
    issue = warning.original_issue_time
    end = warning.end_time()
    for r in reports:
        if r.category != category:
            continue
        if not (issue <= r.time <= end):
            continue
        rev = warning.revision_at(r.time) or warning.revisions[0]
        if contains_with_buffer(rev.polygon, r.lat, r.lon, buffer_km=buffer_km):
            out.append(r)
    return out


def _hail_component(
    warning: Warning,
    hail_reports: list[Report],
) -> float | None:
    """0..1 accuracy for the hail prediction.

    Returns:
      - ``None`` when the warning carries no hail prediction (``hail_in =
        None``, i.e. the player picked "(no hail tag)") — component is
        skipped entirely and contributes nothing to the magnitude mean.
      - A numeric score otherwise. A predicted value of ``0`` means "no
        hail expected" — full credit if none materializes, miss if hail
        occurs. Positive predictions are scored by standard accuracy
        (``1 − |predicted − peak| / peak``), with the revision active at
        the peak hail's report time.
    """
    cur_mag = warning.current_revision.magnitudes
    if cur_mag.hail_in is None:
        return None
    predicted = float(cur_mag.hail_in)
    if not hail_reports:
        # No hail observed inside the warning footprint.
        # Predicted no hail → correct call (full credit).
        # Predicted hail → over-prediction (zero accuracy).
        return 1.0 if predicted <= 0.0 else 0.0
    peak = max(hail_reports, key=lambda r: r.magnitude)
    if peak.magnitude <= 0:
        return 1.0 if predicted <= 0.0 else 0.0
    rev = warning.revision_at(peak.time) or warning.revisions[0]
    if rev.magnitudes.hail_in is None:
        return 0.0
    pred_at_peak = float(rev.magnitudes.hail_in)
    if pred_at_peak <= 0.0:
        return 0.0   # claimed no hail but hail occurred → miss
    err = abs(pred_at_peak - peak.magnitude) / max(peak.magnitude, 0.5)
    return max(0.0, 1.0 - err)


def _wind_component(
    warning: Warning,
    wind_reports: list[Report],
) -> float | None:
    """0..1 accuracy for the wind-gust prediction. ``None`` when the
    warning carries no wind tag; a predicted ``0`` means "no wind threat
    expected" and is scored symmetrically to the hail component."""
    cur_mag = warning.current_revision.magnitudes
    if cur_mag.wind_mph is None:
        return None
    predicted = float(cur_mag.wind_mph)
    if not wind_reports:
        return 1.0 if predicted <= 0.0 else 0.0
    peak = max(wind_reports, key=lambda r: r.magnitude)
    if peak.magnitude <= 0:
        return 1.0 if predicted <= 0.0 else 0.0
    rev = warning.revision_at(peak.time) or warning.revisions[0]
    if rev.magnitudes.wind_mph is None:
        return 0.0
    pred_at_peak = float(rev.magnitudes.wind_mph)
    if pred_at_peak <= 0.0:
        return 0.0
    err = abs(pred_at_peak - peak.magnitude) / max(peak.magnitude, 30.0)
    return max(0.0, 1.0 - err)


def _magnitude_accuracy(
    warning,
    matches: list[VerifyingMatch],
    reports: list[Report],
) -> float:
    """Return a 0..1 accuracy fraction for the warning's magnitude estimates.

    Per plan §5: peak observed is computed over the full warning lifetime, and
    magnitude accuracy uses the **revision active when that peak was observed**
    — so a player who initially nailed an estimate and later revised it isn't
    penalized for the later revision.

    Component-wise scoring:
      - SVR family: hail (if predicted) + wind (if predicted). Hail/wind
        observations come from the verifying matches — i.e. hail ≥1.0" or
        wind ≥58 mph that actually verified the SVR.
      - TOR family: hail (if predicted; this is the NWS IBW hail tag on a
        tornado warning) + EF (legacy/programmatic-only — the UI no longer
        exposes it, but the data model keeps it). Hail reports come from the
        full ``reports`` list filtered to the warning polygon and window,
        since they are not "verifying" reports for a tornado warning but
        the player did stake a magnitude claim that should be scored.
      - Either family: components that the player did NOT predict are
        skipped (no contribution).
      - Final magnitude = mean of contributing components.
    """
    warning_type = warning.current_revision.warning_type
    components: list[float] = []
    if warning_type.is_severe_family:
        # SVR hail / wind components — sourced from verifying matches so
        # sub-severe hail (<1") doesn't count toward an SVR's magnitude
        # bonus (it never verified the SVR in the first place).
        hail_reports = [m.report for m in matches if m.report.category == "hail"]
        wind_reports = [m.report for m in matches if m.report.category == "wind"]
        for comp in (_hail_component(warning, hail_reports),
                     _wind_component(warning, wind_reports)):
            if comp is not None:
                components.append(comp)
    elif warning_type.is_tornado_family:
        # Tornado-warning hail tag (NWS IBW practice) — uses any hail report
        # in the polygon during the warning, not just severe-threshold hail.
        cur_mag = warning.current_revision.magnitudes
        if cur_mag.hail_in is not None:
            hail_reports = _reports_in_warning(warning, reports, "hail")
            comp = _hail_component(warning, hail_reports)
            if comp is not None:
                components.append(comp)
        # Legacy EF magnitude scoring — kept so programmatic/test code that
        # sets `Magnitudes.ef` still earns a bonus, even though the UI no
        # longer exposes the field.
        if cur_mag.ef is not None:
            tor_matches = [m for m in matches
                           if m.report.category == "tornado" and m.report.magnitude >= 0]
            if tor_matches:
                peak = max(tor_matches, key=lambda m: m.report.magnitude)
                rev = warning.revision_at(peak.report.time) or warning.revisions[0]
                if rev.magnitudes.ef is not None:
                    err = abs(rev.magnitudes.ef - peak.report.magnitude) / max(peak.report.magnitude, 1.0)
                    components.append(max(0.0, 1.0 - err))
                else:
                    components.append(0.0)
            else:
                components.append(0.0)
    return sum(components) / len(components) if components else 0.0


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
    current_type = warning.current_revision.warning_type
    cur_mag = warning.current_revision.magnitudes

    # FA / partial / verified decision is based on **covered claimed
    # hazards**: only reports of categories the warning explicitly
    # tagged count for verification. A wind report against a
    # hail-only-predicted SVR doesn't rescue the warning from FA;
    # conversely, a severe hail report on a hail+wind SVRD verifies
    # the hail leg even if no severe wind materializes (the wind
    # gets a 0 magnitude-accuracy hit but the warning is NOT a FA).
    # Tier / magnitude-accuracy bonuses are separate components so
    # an SVRD with severe-but-sub-SVRD-tier hail (e.g. 1.5") still
    # scores positively — just without the SVRD tier multiplier.
    claimed = claimed_hazards(warning)
    if claimed:
        relevant_matches = [m for m in matches if m.report.category in claimed]
        is_fa = len(relevant_matches) == 0
    else:
        # Bare warning with no tags — preserve historical behavior
        # (any verifying report saves it from FA).
        is_fa = len(matches) == 0

    mag_acc = _magnitude_accuracy(warning, matches, reports)
    mag_bonus = mag_acc * MAG_ACCURACY_WEIGHT

    # "Tornado Possible" IBW tag on an SVR: flat bonus when an actual
    # tornado fell inside the polygon during the warning's valid time.
    # Tornadoes aren't part of `find_verifying_reports` for the SVR family
    # (verifies_warning_type filters to hail/wind), so we look them up
    # separately and only add the bonus once per warning.
    tp_bonus = 0.0
    if current_type.is_severe_family and cur_mag.tornado_possible:
        tor_in_poly = _reports_in_warning(warning, reports, "tornado")
        if tor_in_poly:
            tp_bonus = SVR_TORNADO_POSSIBLE_BONUS
            # Catching a tornado the player flagged saves the warning from
            # FA classification even if no hail/wind verified.
            is_fa = False

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
    # When claimed hazards are set, only count claimed-category matches
    # toward the tier mean — otherwise an unpredicted hazard could
    # bend the multiplier away from what the player actually
    # forecast.
    scoring_matches = (
        [m for m in matches if m.report.category in claimed]
        if claimed else matches
    )
    base_pts = BASE_TOR_POINTS if current_type.is_tornado_family else BASE_SVR_POINTS
    per_match_mults = [
        _tier_multiplier_for_match(m.revision.warning_type, m) for m in scoring_matches
    ]
    if per_match_mults:
        mean_tier = sum(per_match_mults) / len(per_match_mults)
        base_score = mean_tier * (1.0 + mag_bonus) * base_pts
    else:
        # No hail/wind verifying matches — the warning was rescued from FA
        # by the tornado-possible tag alone. No hail/wind base score; only
        # the tp_bonus contributes.
        mean_tier = 1.0
        base_score = 0.0
    points = base_score + tp_bonus
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
