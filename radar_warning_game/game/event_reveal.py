"""Post-round event reveal data (plan §9).

When the round ends, the leaderboard screen reveals the real date and a brief
location summary, plus a link to an NWS event review when available.

For days NOT in the curated dict we fall back to SPC's daily storm reports
page (which is auto-generated for every date and always exists).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class EventReveal:
    date_str: str           # "2013-05-20" (UTC)
    name: str               # "Moore EF5 Tornado"
    location: str           # "Newcastle / Moore / South OKC, OK"
    url: str                # NWS WFO event review or SPC daily reports

# Hand-curated for the most-significant US severe-weather events.
# Convective day = 12Z–12Z UTC; the date here is the UTC date the day began.
_FAMOUS_EVENTS: dict[str, EventReveal] = {
    "2011-04-27": EventReveal(
        "2011-04-27", "Super Outbreak",
        "Alabama / Mississippi / Tennessee / Georgia",
        "https://www.weather.gov/bmx/event_04272011",
    ),
    "2011-05-22": EventReveal(
        "2011-05-22", "Joplin EF5 Tornado",
        "Joplin, MO",
        "https://www.weather.gov/sgf/news_events_2011may22",
    ),
    "2013-05-19": EventReveal(
        "2013-05-19", "Edmond / Shawnee / Carney Tornado Outbreak",
        "Central Oklahoma",
        "https://www.weather.gov/oun/events-20130519",
    ),
    "2013-05-20": EventReveal(
        "2013-05-20", "Moore EF5 Tornado",
        "Newcastle / Moore / South OKC, OK",
        "https://www.weather.gov/oun/events-20130520",
    ),
    "2013-05-31": EventReveal(
        "2013-05-31", "El Reno EF3 Tornado",
        "El Reno, OK",
        "https://www.weather.gov/oun/events-20130531",
    ),
    "2013-11-17": EventReveal(
        "2013-11-17", "November 2013 Tornado Outbreak",
        "Illinois / Indiana / Michigan",
        "https://www.weather.gov/ilx/nov172013",
    ),
    "2019-05-27": EventReveal(
        "2019-05-27", "Dayton EF3 Tornado Outbreak",
        "Western Ohio",
        "https://www.weather.gov/iln/20190527",
    ),
    "2020-04-12": EventReveal(
        "2020-04-12", "Easter Sunday Outbreak",
        "Mississippi / Alabama / Georgia / Tennessee",
        "https://www.weather.gov/jan/april122020",
    ),
    "2021-12-10": EventReveal(
        "2021-12-10", "Mayfield / Western Kentucky Tornadoes",
        "Western Kentucky / Tennessee",
        "https://www.weather.gov/pah/December-10th-11th-2021-Tornado",
    ),
    "2023-12-09": EventReveal(
        "2023-12-09", "Clarksville TN Tornadoes",
        "Middle Tennessee",
        "https://www.weather.gov/ohx/20231209",
    ),
}


def reveal_for(convective_day_12z: datetime) -> EventReveal:
    """Return a reveal for the day. Falls back to SPC's daily reports URL."""
    date_str = convective_day_12z.strftime("%Y-%m-%d")
    if date_str in _FAMOUS_EVENTS:
        return _FAMOUS_EVENTS[date_str]
    yymmdd = convective_day_12z.strftime("%y%m%d")
    return EventReveal(
        date_str=date_str,
        name=f"Severe Weather Event of {date_str}",
        location="(see SPC reports)",
        url=f"https://www.spc.noaa.gov/climo/reports/{yymmdd}_rpts.html",
    )
