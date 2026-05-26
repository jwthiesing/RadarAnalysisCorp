"""SPC Peak Intensity Bin (PIB) tables for MCD scoring (plan §8).

Each hazard (tornado / wind / hail) has a numbered ladder of bins, each with a
lower magnitude threshold. PIB ranges *overlap* in the published SPC tables (a
PIB represents a probabilistic forecast with characteristic intensity centered in
the range), but for **scoring** we map an observation to the highest PIB whose
lower bound ≤ the observed magnitude. PIB 0 is "None" (no expected hazard).

Units:
  - Tornado magnitude in EF rating (0-5). Internally we map EF → wind mph
    using NWS midpoint speeds and then look up the wind-mph lower bounds.
  - Wind magnitude in mph (gust).
  - Hail magnitude in inches (diameter).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PIBSpec:
    """A single Peak Intensity Bin entry."""

    pib: int            # 0 means "None" (not used in tables, used in scoring)
    lower_mph: float    # lower bound, in mph (for tornado/wind) or inches (for hail)
    upper_mph: float    # upper bound (overlapping with next; informational)
    descriptor: str
    ibw_tag: str


# Tornado PIB table — wind speeds in mph (NWS damage-survey upper-bound estimates)
TORNADO_PIBS: tuple[PIBSpec, ...] = (
    PIBSpec(1, 65.0, 95.0,  "Weak",                          "Base"),
    PIBSpec(2, 85.0, 115.0, "Weak / Strong",                 "Base"),
    PIBSpec(3, 100.0, 130.0, "Weak / Strong",                "Base / Considerable"),
    PIBSpec(4, 120.0, 150.0, "Strong",                       "Considerable"),
    PIBSpec(5, 140.0, 170.0, "Intense",                      "Considerable / Catastrophic"),
    PIBSpec(6, 155.0, 190.0, "Intense / Violent",            "Catastrophic"),
    PIBSpec(7, 175.0, float("inf"), "Violent / Exceptionally Rare", "Catastrophic"),
)

# Wind PIB table — wind gust in mph
WIND_PIBS: tuple[PIBSpec, ...] = (
    PIBSpec(1, 0.0, 60.0,    "Locally Damaging",                        "Base"),
    PIBSpec(2, 55.0, 70.0,   "Severe",                                  "Base"),
    PIBSpec(3, 65.0, 80.0,   "Severe / Some Significant",               "Base / Considerable"),
    PIBSpec(4, 75.0, 90.0,   "Significant",                             "Considerable / Destructive"),
    PIBSpec(5, 85.0, 100.0,  "Significant / Some Intense",              "Destructive"),
    PIBSpec(6, 95.0, 115.0,  "Intense",                                 "Destructive"),
    PIBSpec(7, 115.0, float("inf"), "Intense to Extreme",               "Destructive"),
)

# Hail PIB table — diameter in inches
HAIL_PIBS: tuple[PIBSpec, ...] = (
    PIBSpec(1, 0.0,  1.25, "Locally Large",        "Base"),
    PIBSpec(2, 1.0,  1.75, "Large",                "Base"),
    PIBSpec(3, 1.5,  2.5,  "Large to Very Large",  "Base / Considerable"),
    PIBSpec(4, 2.0,  3.5,  "Very Large to Giant",  "Considerable / Destructive"),
    PIBSpec(5, 2.75, 4.25, "Very Large to Giant",  "Destructive"),
    PIBSpec(6, 4.0,  float("inf"), "Giant",        "Destructive"),
)

# EF rating → midpoint wind mph (for mapping tornado reports to tornado PIBs).
# Uses the NWS EF scale midpoint (or upper bound for EF5+).
_EF_TO_MPH = {
    0: 75.0,
    1: 95.0,
    2: 120.0,
    3: 152.0,
    4: 184.0,
    5: 210.0,
}


def ef_to_mph(ef: int | float) -> float:
    """Map an EF rating to a representative wind speed in mph.

    Used to look up the tornado PIB corresponding to an observed tornado report.
    """
    if ef < 0:
        return 0.0
    ef_int = int(round(float(ef)))
    return _EF_TO_MPH.get(min(ef_int, 5), 75.0)


def observed_to_pib(category: str, magnitude: float) -> int:
    """Map an observed magnitude to the highest PIB whose lower bound ≤ magnitude.

    For tornadoes: ``magnitude`` is the EF rating (0-5); EF=-1 (or any negative)
    means "unknown / no report" and returns 0. EF=0 is a real weak tornado and
    returns PIB 1 (its corresponding wind range starts at 65 mph).

    For wind: ``magnitude`` is gust mph; ``<=0`` returns 0 ("no observation").
    For hail: ``magnitude`` is diameter inches; ``<=0`` returns 0.
    """
    if category == "tornado":
        if magnitude < 0:
            return 0
        mag_mph = ef_to_mph(magnitude)
        table = TORNADO_PIBS
    elif category == "wind":
        if magnitude <= 0:
            return 0
        mag_mph = magnitude
        table = WIND_PIBS
    elif category == "hail":
        if magnitude <= 0:
            return 0
        mag_mph = magnitude
        table = HAIL_PIBS
    else:
        return 0
    best = 0
    for spec in table:
        if mag_mph >= spec.lower_mph:
            best = spec.pib
    return best


def max_pib_for_category(category: str) -> int:
    """Highest valid PIB number for a category (7 for tornado/wind, 6 for hail)."""
    if category == "tornado":
        return TORNADO_PIBS[-1].pib
    if category == "wind":
        return WIND_PIBS[-1].pib
    if category == "hail":
        return HAIL_PIBS[-1].pib
    return 0


def pib_table(category: str) -> tuple[PIBSpec, ...]:
    if category == "tornado":
        return TORNADO_PIBS
    if category == "wind":
        return WIND_PIBS
    if category == "hail":
        return HAIL_PIBS
    raise ValueError(f"Unknown PIB category: {category!r}")
