"""Build resources/cities/us_cities.csv from the GeoNames cities500 dump.

GeoNames is CC-BY 4.0; cities500 includes every populated place with
population ≥ 500 globally. We filter to the US subset and slim the
columns to what the overlay loader needs.

Feature codes we keep (per https://www.geonames.org/export/codes.html):
  - PPL   populated place
  - PPLA  seat of a first-order admin division (state capital)
  - PPLA2 seat of a second-order admin division (county seat)
  - PPLA3 seat of a third-order admin division
  - PPLA4 seat of a fourth-order admin division
  - PPLC  capital of a political entity (Washington DC)
  - PPLS  populated places (generic)

Run once whenever the GeoNames dump should be refreshed:

    python scripts/build_us_cities.py

Output schema (CSV with header): name,state,lat,lon,pop,feature
"""

from __future__ import annotations

import csv
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

GEONAMES_URL = "https://download.geonames.org/export/dump/cities500.zip"
OUT_PATH = Path(__file__).resolve().parent.parent / "resources" / "cities" / "us_cities.csv"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {GEONAMES_URL} ...", file=sys.stderr)
    with urllib.request.urlopen(GEONAMES_URL) as resp:
        zipped = resp.read()
    print(f"Got {len(zipped):,} bytes", file=sys.stderr)
    with zipfile.ZipFile(io.BytesIO(zipped)) as zf:
        with zf.open("cities500.txt") as f:
            raw = f.read().decode("utf-8")
    n_in = n_out = 0
    with OUT_PATH.open("w", encoding="utf-8", newline="") as out:
        w = csv.writer(out)
        w.writerow(["name", "state", "lat", "lon", "pop", "feature"])
        for line in raw.splitlines():
            n_in += 1
            cols = line.split("\t")
            if len(cols) < 18:
                continue
            country = cols[8]
            if country != "US":
                continue
            asciiname = cols[2]
            lat = cols[4]
            lon = cols[5]
            feature = cols[7]
            state = cols[10]
            try:
                pop = int(cols[14] or 0)
            except ValueError:
                pop = 0
            w.writerow([asciiname, state, lat, lon, pop, feature])
            n_out += 1
    print(f"Read {n_in:,} GeoNames rows, wrote {n_out:,} US rows to {OUT_PATH}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
