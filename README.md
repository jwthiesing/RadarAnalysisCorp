# RadarAnalysisCorp

A Python radar-nowcasting + warning-issuance game inspired by [Reference-Nowcastle/](Reference-Nowcastle/). Single-player or **online P2P multiplayer** (up to 50 players, star topology over WebRTC DataChannels). Play replays of historical severe-weather events, or **LIVE mode** against current weather streaming from IEM.

The player sits at a forecaster's desk: live-feeling radar data streams in from selected WSR-88D sites, storm reports appear as they happen and fade with time, and the player issues **warnings** (SVR / SVRC / SVRD / TOR / TORR / PDS TOR / TORE) and **MCDs** (with SPC-style Peak Intensity Bins) inside a host-defined game polygon. Forecasts are verified against actual storm reports for **POD / FAR / lead-time / magnitude-accuracy** scoring with IBW tier-aware multipliers.

---

## Quick start

```bash
# Clone, install, run
git clone <this-repo>
cd RadarAnalysisCorp
pip install -e .

# Launch the game
python -m radar_warning_game.play
```

Dependencies installed automatically:
- **PyQt6** + **qasync** — GUI + Qt/asyncio bridge
- **PyART (`arm_pyart`)** — NEXRAD Level 2 parsing + velocity dealiasing
- **matplotlib + cartopy** — basemaps, radar rendering, county overlays
- **boto3** — radar S3 access (unsigned; no AWS account needed)
- **requests** — IEM LSR fetch + live-mirror radar download
- **aiortc + aiohttp** — WebRTC DataChannels + signaling
- **shapely** — polygon math (5 km buffer, point-in-polygon, union)
- **mpl_point_clicker** — interactive polygon vertex placement

---

## Game flow (single player)

```
ModeDialog → DayPicker → CONUSOverviewMap → TimeDistribution → PrefetchProgress → PlayView → FinalLeaderboard
                            (polygon +        (pick game           (download
                             radar sites)      window)              volumes)
```

In the **Day Picker** you choose one of three modes:

- **Random day** with thresholds (e.g. ≥5 tornadoes AND ≥20 hail AND ≥20 wind reports). Date hidden from everyone for fair play.
- **Specific date.** Host sees the date (effectively a quizmaster); peers don't.
- **Live (now).** Real-time wall-clock play against today's weather, streaming from the IEM live mirror.

On the **CONUS overview** screen, click **`Draw polygon`** to enter polygon-drawing mode (cursor changes to a crosshair and the button turns bright yellow). Outside draw mode the map behaves like Google Maps — see [Map controls](#map-controls) below.

The host clicks **WSR-88D X markers** to enable (cyan) or disable (grey) each radar for the round. Inactive radars are unavailable to all players, simulating downed radars.

After the polygon is drawn, the host's status bar shows `polygon set` and the **`Continue to time window →`** button advances to the time histogram. In historical mode the host drags a span over the day's storm-report distribution to pick the game window. **`Start round →`** kicks off radar prefetch. Once the radar data lands, the **PlayView** opens.

Live mode skips the time histogram — the window is automatically set to `now → now + 2h`.

---

## In-game keyboard shortcuts

Every shortcut is a `QShortcut` at the `PlayView` level, so they fire regardless of which child widget currently has focus.

| Key | Action |
|---|---|
| **Movement / view** | |
| `←` / `→` | Scrub time backward / forward by one sweep |
| `Shift+←` / `Shift+→` | Scrub by 5 sweeps |
| `↑` / `↓` | Change radar elevation up / down |
| `=` (or `+`) | Zoom in (center of view, all panels synced) |
| `-` | Zoom out |
| **Clock** | |
| `Space` | Pause / play |
| `[` / `]` | Slower / faster game clock |
| **Layout** | |
| `Alt+1` | One radar panel (REF) |
| `Alt+2` | Two panels (REF / VEL) |
| `Alt+4` | Four panels (REF / VEL / CC / ZDR) |
| `1` – `7` | Cycle the **focused** panel's product (click a panel to focus it) |
| **Issuance** | |
| `N` | New warning — opens polygon-draw mode on the focused radar panel |
| `C` | New MCD — opens polygon-draw mode |
| `M` | Toggle storm-motion measurement tool (2 clicks, on 2 different scans) |
| `Esc` | Cancel an in-flight polygon draw |
| `Enter` | Finish the current polygon and open the form dialog |

Product key map: **1**=REF • **2**=VEL • **3**=SW • **4**=CC • **5**=ZDR • **6**=KDP • **7**=PHI

---

## Map controls

### CONUS day-selection map

Polygon drawing is **opt-in** via a toolbar button so it doesn't conflict with radar-site toggling and panning.

| Gesture | Draw mode OFF (default) | Draw mode ON |
|---|---|---|
| **Left-click** on a radar X marker | Toggle that radar on/off | Toggle radar AND add vertex |
| **Left-click on empty map** | Nothing | Add a polygon vertex |
| **Left-click and drag** | Pan the map | (use Shift+left to pan) |
| **Right-click** | Pan (drag) | Remove the nearest polygon vertex |
| **Middle-click drag** | Pan | Pan |
| **Shift + left-drag** | Pan | Pan (escape hatch — doesn't add vertices) |
| **Scroll wheel** | Zoom toward cursor | Zoom toward cursor |
| **`Draw polygon` button** | Enter draw mode | Exit draw mode (preserves vertices) |
| **`Clear polygon` button** | — | Reset to no vertices |
| **`Reset view` button** | Snap back to full CONUS | Snap back to full CONUS |
| **`Reroll random day` button** | Sample a new random day (clears polygon + radar selections) | — |

### Radar panels (in-game)

| Gesture | Action |
|---|---|
| **Scroll wheel** | Zoom around cursor (debounced ~40 ms) |
| **Left-click drag** | Pan |
| **Double-click** | Reset to home view (±250 km centered on radar) |
| **Click a panel** | Focus it (border highlights; subsequent product hotkeys apply here) |
| **Scrubber slider** (below grid) | Drag to jump to any time within the current elevation's sweep range |
| **Keyboard zoom (`=`/`-`)** | Center-based zoom — cursor-position-independent |

Pan, zoom, scrub, and elevation change all **preserve the current view** across re-renders. The first render uses a default 500 km extent; after that, your pan/zoom sticks even when scrubbing through new sweeps.

---

## Map overlays (radar panels)

Drawn on every panel:

- **Game polygon** — yellow outline of the verification boundary (where your warnings can verify). Plan §4a.
- **State borders** — Cartopy Natural Earth admin_1 lines, projected to the radar's km frame.
- **US county borders** — bundled Census TIGER 2023 shapefile, drawn as thin grey lines when zoomed past ~400 km width.
- **Cities** — population-tiered (>100 k always shown; >20 k when zoomed in). **Greedy non-overlap labeling**: drawn in descending-population order, smaller cities are skipped if their label would collide with an already-placed one.
- **Range rings** at 50 / 100 / 150 / 200 km.
- **Live storm reports** — fade in over the last ~30 min of display time, scrubbing back recomputes which reports are visible. Markers:
  - ▲ **red-edged triangle** = tornado
  - ● **green-edged circle** = hail
  - ■ **blue-edged square** = wind
  - Fill color: time-of-day colormap; size: magnitude.

The host-only **Central Map** (multiplayer) additionally shows every player's warning polygons colored by team. **In single-player the central map is hidden** since you only have your own warnings — the radar panels themselves show everything you need.

---

## Scoring

For each verified warning:
```
points = mean(per_match_tier_mult) × (1 + magnitude_bonus) × base_pts
```

Each verifying report contributes a tier multiplier based on the **revision active at the report's time**. Upgrading a warning SVR → TOR mid-event earns the upgraded multiplier only for reports that arrive after the upgrade.

| Type | Verified bonus | FA penalty multiplier |
|---|---|---|
| **SVR** | 1.0× | 1.0× |
| **SVRC** | 1.10× (if hail ≥ 1.75" or wind ≥ 70 mph) | 1.0× |
| **SVRD** | 1.25× (if hail ≥ 2.75" or wind ≥ 80 mph) | 1.5× |
| **TOR** | 1.0× | 1.0× |
| **TORR** | 1.10× (+10-min late-warn POD credit allowed) | 1.0× |
| **PDS TOR** | 1.75× (if EF ≥ 2 OR injuries/fatalities) | 1.5× |
| **TORE** | 2.5× (if EF ≥ 2 OR casualties; 0.75× if only weak tornado) | **3.0×** (heaviest in game) |

**Magnitude bonus** is the mean of hail/wind/EF accuracy components. **Predicted-but-unverified hazards contribute 0** to that mean — predict only what you expect. (So an SVR with predicted hail=2.0" + wind=100 mph that verifies only via hail scores half the mag bonus of one with verified hail + matching wind prediction.)

POD denominator uses the game polygon with the same 5 km verification buffer, so reports just outside the strict polygon edge that verify a warning also count in the denominator (symmetric accounting).

**MCDs** (Mesoscale Convective Discussions) use SPC's Peak Intensity Bin system. Players pick PIB 1–7 (tornado/wind) or 1–6 (hail), or "None" per hazard. Scoring components:
- **Per-hazard PIB accuracy** (closer to observed = better; observed PIB 0 + predicted None = small credit)
- **Per-hazard lead-time bonus** — separate lead for tornado / wind / hail, averaged. A predicted-but-unverified hazard contributes 0 to the average.
- **Multi-hazard breadth bonus** when 2+ predicted hazards verify
- **Anti-spam (server-side enforced):** min 200 km² area, min 4 vertices, max 50 % game-polygon coverage, max 1 active MCD per team per 30 game-minutes. Wire MCDs from peers are validated too, not just dialog-side.

**Team mode** (host-optional) lets players form teams in a pre-round lobby. Each team is scored as one entity — the union of teammates' warnings.

**End-of-round leaderboard** shows: total / warnings / MCDs / POD / FAR / CSI / **mean lead + P25/P75 quartiles** / counts. Below the table is a link to the saved replay file (if enabled) and an NWS event review URL for famous outbreaks (Super Outbreak, Joplin, Moore, El Reno, etc.).

---

## Multiplayer setup

Multiplayer needs a small **signaling server** running somewhere reachable by all players (used only for the WebRTC handshake; gameplay traffic is peer-to-peer once connected).

```bash
# On any reachable machine (your laptop on LAN, a $5 VPS, etc.):
python -m signaling_server.server --host 0.0.0.0 --port 8765
```

In the game's **Mode** dialog, set `ws://<that-host>:8765/ws` as the signaling URL. Default is `ws://localhost:8765/ws` for LAN play.

- **Hosting:** pick "Host a multiplayer room" — you get a memorable room code like `STORM-FROG-72` to share. Walk through the same day-picker / polygon / time-window setup as solo. When you click "Start round," peers receive the full setup over the wire and begin downloading radar data.
- **Joining:** pick "Join an existing room", enter the code, wait for the host to finish setup. Peers send their warning/MCD actions to the host, which re-broadcasts to all others so every client converges on identical session state.
- **Mid-round join** is supported: the host re-sends RoundSetup + active warnings/MCDs/players + a clock snapshot when a late peer connects. They'll catch up immediately.

**NAT traversal:** uses Google + Twilio public STUN servers. Works on cone NATs (most home routers). **Symmetric NATs require a TURN server**, which we don't ship — affected players will see the WebRTC handshake fail. For private LAN play, no STUN is needed.

**Date blinding (honor-system):**
- All player-facing timestamps go through `ui/time_format.py` and render as `HH:MM:SSZ` only
- Radar cache files use SHA-1 filenames instead of the original `YYYY/MM/DD/...` keys
- Protocol Tick messages send `(virtual_time_offset_sec, speed, paused)`, never absolute dates
- A `pytest` sweep (`test_date_blinding.py`) lints UI strings for any `YYYY-MM-DD` patterns

This blocks casual snooping but a determined player can still find the date in process memory (S3 keys must be constructed locally). Document for your group.

---

## Live mode

Live mode plays out in **wall-clock real time** against IEM's live radar mirror and recent LSRs. Distinguishing characteristics:

- **`LiveClock`** locks `virtual_time = now()`. The `[`/`]` speed-control and pause keys are no-ops in live mode (each client reads its own wall clock locally so multiplayer stays synchronized without any host clock broadcasting).
- **Radar source** switches from S3 (`unidata-nexrad-level2`) to IEM's live HTTP mirror (`mesonet-nexrad.agron.iastate.edu/level2/raw/<SITE>/`).
- **Prefetch window** is reversed: we fetch the most recent 30 min of available radar at round start, and poll backwards every tick for newly-arrived volumes (the live mirror has past volumes, not future ones).
- **Date blinding is off** — you know it's today.
- **Storm reports** stream in with realistic IEM delay (minutes to hours after the event). You forecast under realistic information uncertainty.

Peer clients in live mode run a local 1 Hz timer that drives the in-game tick — necessary because the host's network ticks are no-ops in live mode.

---

## Performance

The radar grid is optimized for smooth scrubbing through hundreds of NEXRAD volumes:

- **Configurable radar LRU cache** keeps recent PyART `Radar` objects in memory (default 24, min 6, max 100 via the `radar_lru_size` constructor kwarg). Scrubbing back to a recently-viewed volume is ~12× faster than re-parsing the Level 2 file.
- **`LineCollection`** for state / county / range-ring borders instead of N separate `Line2D` artists — much cheaper matplotlib draw path.
- **Batched scatter** for city dots (single call vs N scatter calls).
- **Greedy non-overlap city labeling** — sort by population descending, skip overlapping labels via fast pixel-bbox comparison. Replaces the previous adjustText-based approach which iterated per render and was the main scrub bottleneck.
- **Rasterized `pcolormesh`** — the heavy 1.3M-cell radar mesh is treated as a raster image during draw rather than a vector path.
- **Debounced repaints** during continuous pan/zoom — a 40 ms QTimer coalesces rapid scroll/drag events into one repaint per panel.
- **View preservation across data changes** — `set_xlim`/`set_ylim` are saved and restored across `pcolormesh` calls so scrubbing time doesn't blow away your zoom.

Measured: **305 ms → 124 ms per scrub** with 4 panels rendering simultaneously, after greedy non-overlap + off-screen culling.

---

## Caching

Per-day IEM CSVs are cached at `~/.radaranalysiscorp/cache/reports/<sha1>.csv` (hashed filenames so the date isn't visible on disk). On top of that:

- **Daily counts index** at `~/.radaranalysiscorp/cache/reports/daily_counts.json` maps `YYYY-MM-DD → {tornado: N, hail: N, wind: N}`. After a day is fetched once, the random-day picker can skip non-qualifying days **without a network round-trip** — a 50-attempt random pick warms in ~0.7 s once the index is populated (down from 15 s+ cold).
- **Radar volume cache** at `~/.radaranalysiscorp/cache/radar/<sha1>.ar2v`. Auto-cleaned on startup: files older than 30 days are purged.

---

## Architecture

```
radar_warning_game/
├── data/
│   ├── sites.py            # WSR-88D catalog from resources/RADARS.txt
│   ├── radar_s3.py         # Unsigned boto3 client for unidata-nexrad-level2
│   ├── live.py             # IEM live mirror scraper for LIVE-mode rounds
│   ├── sweep_index.py      # SAILS-aware per-sweep index across loaded volumes
│   ├── reports.py          # IEM LSR + SPC backfill + daily-counts index
│   ├── prefetch.py         # Background-thread parallel downloads, live + S3 modes
│   └── cache.py            # HashedCache + 30-day startup cleanup
├── geo/
│   ├── projection.py       # haversine, bearing, lat/lon ↔ km, storm motion
│   └── polygons.py         # Shapely-backed Polygon, 5 km buffer, union, fraction
├── verification/
│   ├── pibs.py             # SPC Peak Intensity Bin tables
│   ├── tornado_tiers.py    # IBW tier multipliers, TORR late-warn window
│   ├── reports_in_poly.py  # Warning + MCD data models + report matching
│   └── scoring.py          # Per-revision tier scoring, team aggregation, MCD scoring
├── game/
│   ├── clock.py            # GameClock + LiveClock (wall-time)
│   ├── round_builder.py    # ThresholdSpec, random/specific day picker
│   ├── session.py          # State machine + team mgmt + MCD anti-spam enforcement
│   ├── replay.py           # JSON event-log writer
│   └── event_reveal.py     # Famous-event names + NWS review URLs
├── ui/
│   ├── play.py             # (defined in radar_warning_game.play)
│   ├── app.py              # MainWindow state machine
│   ├── play_view.py        # In-game composite widget + all gameplay shortcuts
│   ├── radar_panel.py      # 1/2/4-panel grid with SAILS scrub + LRU + overlays
│   ├── overview_map.py     # CONUS map for setup (polygon + radar picker)
│   ├── time_distribution.py # Stacked time-of-day histogram with span selector
│   ├── poly_editor.py      # mpl_point_clicker wrapper with set_enabled()
│   ├── warning_form.py     # Warning issue/revision dialog
│   ├── mcd_form.py         # MCD dialog with 3 PIB pickers
│   ├── motion_tool.py      # 2-point storm motion measurement
│   ├── host_map.py         # Host's central in-game overview (multiplayer-only)
│   ├── leaderboard.py      # Live corner widget + final dialog with quartiles
│   ├── controls.py         # Play/pause/speed + 1 Hz network throttle
│   ├── team_lobby.py       # Pre-round team lobby
│   ├── room_dialogs.py     # Mode/Join/Host status dialogs
│   ├── day_picker.py       # Random / specific / live mode picker
│   ├── prefetch_progress.py
│   ├── overlay_loader.py   # Cartopy + TIGER county shapefile projection
│   ├── colors.py           # 50-color palette (Wong + extended)
│   └── time_format.py      # Date-blinding timestamp formatters
├── net/
│   ├── protocol.py         # 14 JSON message types
│   ├── peer.py             # HostTransport + ClientTransport (aiortc + STUN)
│   ├── lobby.py            # (reserved for future lobby browser)
│   └── multiplayer.py      # MultiplayerHost / MultiplayerPeer orchestrators
├── play.py                 # Top-level entry point (qasync loop)
└── resources/
    ├── RADARS.txt
    └── counties/           # TIGER 2023 cb_2023_us_county_500k shapefile (~17 MB)

signaling_server/
└── server.py               # aiohttp WebSocket server for room codes + SDP exchange

tests/                      # 253 pytest tests across 20 files
```

---

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

253 tests covering: SAILS sweep index, 5-km buffered point-in-polygon, scoring metrics (incl. team aggregation, magnitude revisions, per-revision tier), tier multipliers, MCD PIB scoring and anti-spam, protocol round-trip, multiplayer state appliers, date-blinding scrape, casualty regex, cache, clock, session state machine, replay, event reveal, colors, sites, live source, round builder. Runs in ~2 seconds.

GitHub Actions CI at [`.github/workflows/test.yml`](.github/workflows/test.yml) runs the suite on Python 3.11 and 3.12 with Qt offscreen + cartopy system deps.

---

## Known limitations & future work

- **TURN server** not bundled; symmetric-NAT users can't connect in multiplayer
- **Host disconnect** ends the round in v1 (no host migration)
- **Late-warn POD credit** is allowed only for TORR (10-minute window)
- **Replay playback** UI not implemented; replay JSON files are written but only readable by tooling
- **SPC final-EF backfill** is best-effort; SPC's daily filtered CSV has a messy multi-section format that defeats naive `pd.read_csv` and we fall back to IEM preliminary data when matching fails
- **Single-radar focus per panel** during play; site switching requires re-running the setup
- **Town labels** beyond major cities not wired (counties drawn but unlabeled)
- **`adjustText`** is no longer installed/used; some city labels may overlap at certain zoom levels (the greedy non-overlap algorithm hides smaller cities to compensate)

---

## Credits

- NEXRAD Level 2 data: NOAA via the [Unidata IDD](https://www.unidata.ucar.edu/data/) S3 mirror (historical) and [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) live HTTP mirror
- LSRs: Iowa Environmental Mesonet
- County polygons: U.S. Census Bureau TIGER 2023 cartographic boundaries
- PIB tables: Storm Prediction Center mesoscale discussion conventions
- Reference game: [Reference-Nowcastle/](Reference-Nowcastle/) (gitignored; not part of this repo)
