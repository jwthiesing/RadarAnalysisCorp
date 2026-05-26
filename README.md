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

# Launch the game (single-player or multiplayer host/join)
python -m radar_warning_game.play
```

Dependencies installed automatically:
- **PyQt6** — GUI
- **PyART (`arm_pyart`)** — NEXRAD Level 2 parsing + dealiasing
- **matplotlib + cartopy** — basemaps, radar rendering, polygon drawing
- **boto3** — radar S3 access (unsigned; no AWS account needed)
- **aiortc + aiohttp + qasync** — WebRTC DataChannels + signaling
- **shapely** — polygon math + 5 km buffer
- **adjustText** — city label collision avoidance

---

## Multiplayer setup

Multiplayer needs a small **signaling server** running somewhere reachable by all players (used only for the WebRTC handshake; gameplay traffic is peer-to-peer once connected).

```bash
# On any reachable machine (your laptop on LAN, a $5 VPS, etc.):
python -m signaling_server.server --host 0.0.0.0 --port 8765
```

In the game's **Mode** dialog, fill in `ws://<that-host>:8765/ws` as the signaling URL. The default is `ws://localhost:8765/ws` for LAN play.

**Hosting a room:** picks "Host a multiplayer room", connects to the signaling server, gets a memorable room code like `STORM-FROG-72` to share with peers, walks through the same day-picker / polygon / time-window setup as solo play. When the host clicks "Start round," peers receive the full round setup over the wire and begin downloading radar data.

**Joining a room:** picks "Join an existing room", enters the room code, waits for the host to finish setup, then plays as a peer. Peers send their warning/MCD actions to the host, which re-broadcasts them so every client converges on identical session state.

**NAT traversal:** uses public STUN servers (Google + Twilio). Works on cone NATs (most home routers). **Symmetric NATs require a TURN server, which we don't ship** — affected players will see the handshake fail. Document this for your group.

---

## Game flow

```
ModeDialog → DayPicker → CONUSOverviewMap → TimeDistribution → PrefetchProgress → PlayView → FinalLeaderboard
                            (host only)         (host only)
                            ↓                   ↓
                         polygon +           drag a span to
                         active radars       pick the game window
```

### Day selection

Three options:
- **Random day** with thresholds (e.g. ≥5 tornadoes AND ≥20 hail AND ≥20 wind reports). Date stays hidden from everyone including the host — fair multiplayer.
- **Specific date.** Host sees the date (effectively a quizmaster); peers don't.
- **Live (now).** Real-time wall-clock play against today's weather, IEM live radar mirror, recent LSRs.

### Polygon + radar-site setup

The host clicks-toggles WSR-88D sites on the CONUS map (cyan = enabled, grey = off) and freehand-draws a game polygon defining the verification boundary. Reports are plotted with shape-by-type, size-by-magnitude, color-by-time-of-day so the host can see at a glance where the day's activity was.

### Time window

For historical rounds, a stacked histogram of report counts by 10-minute bin lets the host drag a span to pick `[start, end]`. Live rounds skip this step (window is `now → now + 2h`).

### Prefetch

All clients independently download radar volumes from S3 (unsigned, `unidata-nexrad-level2`) or the IEM live mirror. Once 75 % of clients are ready, a 60-second countdown starts and play begins.

### Gameplay

Each player has a **radar panel grid** (1, 2, or 4 panels — defaults REF / REF+VEL / REF+VEL+CC+ZDR) showing one active radar at a time with SAILS-aware time scrubbing (low-tilt updates at ~60–90 s cadence; higher tilts at volume cadence). Pan with click-drag, zoom with scroll, scrub with `←/→`, change tilt with `↑/↓`, switch product with `1`–`7`. Velocity is dealiased by default (region-based; switchable to phase-unwrap or none).

Overlays include state borders, **US counties** (when zoomed in past ~400 km width; from bundled TIGER 2023 shapefile), city labels with collision avoidance, range rings, the game polygon, the player's own warnings, faint peer warnings, and **live storm reports** that fade in/out tied to the panel's display time.

The **host** additionally has a central map view showing every player's warning polygons (color-by-team, line-style-by-family: solid TOR-family / dashed SVR-family / dotted MCD; line-weight scales with tier) and a live leaderboard widget. The host can play in a separate window by clicking "Join as player".

**Keyboard shortcuts during play:**
- `N` — new warning (polygon draw → form)
- `C` — new MCD (polygon draw → PIB form)
- `M` — toggle storm-motion measurement tool (two clicks across two scans gives `from XXX° at YY kt`)
- `[` / `]` — slower / faster game speed (host only)
- `Space` — pause/play (host only)

---

## Scoring

For each verified warning:
```
points = mean_per_match_tier_mult × (1 + magnitude_bonus) × base_pts
```

where each verifying report contributes a tier multiplier based on **the revision active at the report's time** (so upgrades SVR→TOR earn the upgraded multiplier only for reports occurring after the upgrade). Tier multipliers:

| Type | Verified bonus | FA penalty multiplier |
|---|---|---|
| **SVR** | 1.0× | 1.0× |
| **SVRC** | 1.10× (if hail≥1.75" or wind≥70 mph) | 1.0× |
| **SVRD** | 1.25× (if hail≥2.75" or wind≥80 mph) | 1.5× |
| **TOR** | 1.0× | 1.0× |
| **TORR** | 1.10× (+10-min late-warn POD allowed) | 1.0× |
| **PDS TOR** | 1.75× (if EF≥2 OR casualties) | 1.5× |
| **TORE** | 2.5× (if EF≥2 OR casualties; 0.75× for weak verifications) | **3.0×** (heaviest in game) |

**Magnitude bonus** is mean of hail/wind/EF accuracy components. Predicted-but-unverified hazards contribute 0 — predict only what you expect.

**MCDs** (Mesoscale Convective Discussions) use SPC's Peak Intensity Bin system. Players pick PIB 1–7 (tornado/wind) or 1–6 (hail), or "None". Scoring:
- **Per-hazard PIB accuracy** (closer = better)
- **Per-hazard lead-time bonus** (max 60 pts at 60-min lead)
- **Multi-hazard breadth bonus** (correct on 2+ hazards)
- **Anti-spam**: min 200 km² area, min 4 vertices, max 50 % coverage of game polygon, max 1 active MCD per team per 30 game-minutes

**Team mode (host-optional)** lets players form teams in a pre-round lobby. Each team is scored as a single entity (union of teammates' warnings).

**End-of-round** leaderboard shows total / warnings / MCDs / POD / FAR / CSI / mean lead / lead quartiles (P25/P75) / counts, plus the real date and a link to NWS event reviews for famous events (Super Outbreak, Joplin, Moore, etc.).

---

## Date blinding

The plan calls for hiding the calendar date from players in random-day mode so they can't cheat by looking up the event. Implementation:
- All player-facing timestamps go through [`ui/time_format.py`](radar_warning_game/ui/time_format.py): `format_player_time(dt) → "HH:MM:SSZ"` strips the date
- Radar cache files use SHA-1 filenames instead of the original `YYYY/MM/DD/...` keys
- Protocol ticks send `(virtual_time_offset_sec, speed, paused)` not absolute dates
- A pytest sweep (`test_date_blinding.py`) lints UI strings for `YYYY-MM-DD` patterns

This is **honor-system** blinding: each client must know the date in memory to construct S3 keys, so a determined player can read it from their own process. Document this for your group.

---

## Architecture

```
radar_warning_game/
├── data/
│   ├── sites.py            # WSR-88D catalog from resources/RADARS.txt
│   ├── radar_s3.py         # Unsigned boto3 client for unidata-nexrad-level2
│   ├── live.py             # IEM live mirror scraper for LIVE-mode rounds
│   ├── sweep_index.py      # SAILS-aware per-sweep index across loaded volumes
│   ├── reports.py          # IEM LSR fetching + SPC backfill for old events
│   ├── prefetch.py         # Background-thread parallel downloads with lookahead
│   └── cache.py            # HashedCache (sha1 names) + 30-day startup cleanup
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
│   ├── replay.py           # JSON event log writer
│   └── event_reveal.py     # Famous-event names + NWS review URLs
├── ui/
│   ├── app.py              # MainWindow state machine (DAY_PICKER → ... → END)
│   ├── play.py             # qasync entry point
│   ├── radar_panel.py      # 1/2/4-panel grid with SAILS scrub + dealias + overlays
│   ├── overview_map.py     # CONUS map for setup phase (polygon + radar picker)
│   ├── time_distribution.py # Stacked time-of-day histogram with span selector
│   ├── poly_editor.py      # mpl_point_clicker wrapper for polygon drawing
│   ├── warning_form.py     # Warning issue/revision dialog
│   ├── mcd_form.py         # MCD dialog with 3 PIB pickers
│   ├── motion_tool.py      # 2-point storm motion measurement
│   ├── host_map.py         # Host's central in-game overview
│   ├── leaderboard.py      # Live corner widget + final dialog
│   ├── controls.py         # Play/pause/speed + 1 Hz network throttle
│   ├── team_lobby.py       # Pre-round team lobby
│   ├── room_dialogs.py     # Mode/Join/Host status dialogs
│   ├── day_picker.py       # Random / specific / live mode picker
│   ├── prefetch_progress.py
│   ├── overlay_loader.py   # Cartopy + TIGER county shapefile projection
│   ├── colors.py           # 50-color palette (Wong + extended)
│   └── time_format.py      # Date-blinding timestamp formatters
├── net/
│   ├── protocol.py         # 14 JSON message types (RoundSetup, Tick, ...)
│   ├── peer.py             # HostTransport + ClientTransport (aiortc DataChannels)
│   ├── signaling_client.py # (consolidated into peer.py)
│   ├── lobby.py            # (reserved for future lobby browser)
│   └── multiplayer.py      # MultiplayerHost / MultiplayerPeer session orchestrators
├── play.py                 # Top-level entry point
└── resources/
    ├── RADARS.txt          # NEXRAD WSR-88D catalog
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

---

## Known limitations & future work

- **TURN server** not bundled; symmetric-NAT users can't connect in multiplayer
- **Host disconnect** ends the round in v1 (no host migration)
- **Late-warn POD credit** is allowed only for TORR (10-minute window). Live mode loosens this rationale but the rule isn't relaxed yet — TBD
- **Replay playback** is not yet a UI feature; replay files are written but only readable by tooling
- **SPC final-EF backfill** is best-effort; SPC's daily filtered CSV has a messy multi-section format that defeats naive `pd.read_csv` and we fall back to IEM preliminary data when matching fails
- **Single-radar focus** during play; no multi-radar mosaic. Tab-style cycling between sites planned but currently you switch by re-running the setup
- **County labels and town names** not yet wired (counties are drawn as outlines only)

---

## Credits

- NEXRAD Level 2 data: NOAA via the [Unidata IDD](https://www.unidata.ucar.edu/data/) S3 mirror
- LSRs and recent radar listings: [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/)
- County polygons: U.S. Census Bureau TIGER 2023 cartographic boundaries
- PIB tables: Storm Prediction Center mesoscale discussion conventions
- Reference game: [Reference-Nowcastle/](Reference-Nowcastle/) (gitignored; not part of this repo)
