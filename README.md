# RadarAnalysisCorp

A Python radar-nowcasting + warning-issuance game inspired by [Reference-Nowcastle/](Reference-Nowcastle/). Single-player or **online P2P multiplayer** (up to 50 players, star topology over WebRTC DataChannels). Play replays of historical severe-weather events, or **LIVE mode** against current weather streaming from IEM.

The player sits at a forecaster's desk: live-feeling radar data streams in from selected radar sites (long-range **WSR-88D** + short-range terminal **TDWR**), storm reports appear as they happen and fade with time, and the player issues **warnings** (SVR / SVRC / SVRD / TOR / TORR / PDS TOR / TORE) and **MCDs** (with SPC-style Peak Intensity Bins) inside a host-defined game polygon. Forecasts are verified against actual storm reports for **POD / FAR / lead-time / magnitude-accuracy** scoring with IBW tier-aware multipliers.

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
- **PyQt6** + **qasync** — GUI + Qt/asyncio bridge (one event loop drives both)
- **pyqtgraph** — high-performance 2D plotting (`ImageItem` for the radar polar shader, `PlotCurveItem` for vector overlays, `ScatterPlotItem` for reports, `TextItem` for labels)
- **PyART (`arm_pyart`)** — NEXRAD Level 2 parsing + velocity dealiasing
- **matplotlib** — only used for its colormap registry (pyart's `ChaseSpectral`, `Carbone42`, …) which pyqtgraph reads through its matplotlib bridge
- **cartopy** — Natural Earth shapefile loader (states, country borders, coastlines, populated places). No cartopy projection or matplotlib axes are used at render time; we project to lon/lat (maps) or km-from-radar (panels) up front
- **boto3** — radar S3 access (unsigned; no AWS account needed)
- **requests** — IEM LSR fetch + live-mirror radar download
- **aiortc + aiohttp** — WebRTC DataChannels + signaling
- **numpy** + **pandas** + **shapely** — array math (rasterizer + lookup tables), LSR CSV parsing, polygon math (5 km buffer, point-in-polygon, union)

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

The host clicks **radar-site markers** to enable (cyan) or disable each radar for the round. **WSR-88Ds** are drawn as X's, **TDWRs** as smaller circles in lavender — the same toggle gesture applies to both. Inactive radars are unavailable to all players, simulating downed radars. A background thread probes the Unidata Level 2 archive for each site on the chosen day; sites with no coverage for that date are dimmed and refuse selection. TDWR archive coverage on the Unidata mirror is patchy in older years, so for pre-2010ish events the TDWR fleet will mostly grey out — but for modern events you can grab a TDWR for inner-city tornado scenarios where its sub-1°-elevation low scans see things the WSR-88D's beam overshoots.

When the initial radar of a round is a TDWR, the play view opens in a 2-panel **REF / VEL** layout by default (Alt+1 / Alt+2 / Alt+4 still toggle). TDWR Level 2 from Unidata only carries REF, VEL, and SW — there is no CC / ZDR / KDP / PHI, so the default 4-panel REF/VEL/CC/ZDR layout would leave the dual-pol panels permanently blank.

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
- **Your own warning polygons** — colored by tier-family and weighted by tier:
  - **Yellow dashed** = SVR / SVRC / SVRD (weight grows with tier)
  - **Red solid** = TOR / TORR
  - **Pink solid** = PDS TOR
  - **TORE** is drawn as a *double-stroked* polygon — a wide pink halo with a thinner black core line on top — so the most-significant tier reads at a glance against any radar background.

The host-only **Central Map** (multiplayer) additionally shows every player's warning polygons colored by team. On that map TORE keeps the team color as its outer halo with a black inner stroke. **In single-player the central map is hidden** since you only have your own warnings — the radar panels themselves show everything you need.

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
- **Anti-spam (server-side enforced):** min 200 km² area, max 250 000 km² area (absolute cap — roughly a large real-world SPC MCD covering 4-5 states), min 4 vertices, max 1 active MCD per team per 30 game-minutes. Wire MCDs from peers are validated too, not just dialog-side.

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

### Hosting the signaling server from your laptop

If you don't have a VPS, the signaling server runs fine on your laptop — you only need somewhere `ws://...` accessible to every player. The signaling traffic itself is tiny (SDP-offer / SDP-answer / a few KB per join), so a residential connection handles it without issue. Once peers' WebRTC DataChannels are up, **game-state traffic flows directly through the host's data channel**, not through your laptop's signaling server.

Pick the path that matches who needs to reach you:

**Same Wi-Fi only (no port forwarding needed):**
```bash
python -m signaling_server.server --host 0.0.0.0 --port 8765
```
Find your laptop's LAN IP (`ipconfig getifaddr en0` on macOS Wi-Fi; `ip route get 1` then look for `src` on Linux; `ipconfig` then "IPv4 Address" on Windows) and share `ws://<lan-ip>:8765/ws` with peers on the same network.

**Friends on the open Internet (router port-forward):**
1. Run the server with `--host 0.0.0.0 --port 8765` (the default).
2. In your router admin page (usually `http://192.168.1.1` or `http://192.168.0.1`), find **Port Forwarding** — sometimes labeled NAT / Virtual Servers / Applications. Add a TCP rule:

   | Field | Value |
   |---|---|
   | External port | 8765 |
   | Internal port | 8765 |
   | Internal IP | your laptop's LAN IP from above |
   | Protocol | TCP |

3. Find your public IP: `curl -s ifconfig.me`. Share `ws://<public-ip>:8765/ws` with peers.
4. Sanity-check from outside your LAN (a phone on cellular works) before relying on it.

Real-world gotchas with home port-forwarding:
- **Dynamic public IPs**: most residential ISPs rotate yours every few days. Either re-share when it changes, or sign up for a free dynamic-DNS hostname (DuckDNS, No-IP) and hand peers `ws://yourname.duckdns.org:8765/ws`.
- **CGNAT**: some ISPs (especially mobile / rural / cellular fallback) put you behind their own NAT and the public IP from `ifconfig.me` isn't directly reachable. If port-forwarding "looks right" but outside connections time out, this is the usual culprit. Use the tunnel option below.
- **macOS sleep kills connections**: existing peers' WebRTC channels drop the moment your laptop sleeps. `caffeinate -i` for the duration of your session prevents that.
- **Plain WebSocket, no TLS** — fine for friends-only ad-hoc sessions; do not expose this publicly. Signaling messages aren't encrypted on the wire and the protocol has no auth beyond "do you have the room code." For an Internet-facing deployment, terminate TLS at a reverse proxy (nginx, Caddy) or use a tunnel (next).

**No router access / behind CGNAT (tunnel):**
Outbound-only tunnels skip port forwarding entirely.

```bash
# ngrok (free tier has time-limited sessions)
brew install ngrok            # or download from ngrok.com
ngrok http 8765

# Cloudflare Tunnel (free, no session time limit)
brew install cloudflared
cloudflared tunnel --url http://localhost:8765
```

Both spit out an `https://<random>.ngrok-free.app` / `https://<random>.trycloudflare.com` URL. WebSocket upgrades work on the http side, so tell peers to use `ws://<random>.ngrok-free.app/ws` — the tunnel routes it through to your local 8765 transparently. Cloudflare's free tunnels are the lower-friction option for play sessions longer than ngrok's free-tier idle limit.

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

The radar grid is optimized for smooth scrubbing through hundreds of NEXRAD volumes and smooth pan/zoom across the synced panels:

- **Polar shader (CPU).** The sweep is rasterized from polar (azimuth, range) → Cartesian (km east/north) via vectorized numpy ufuncs and displayed as a single `pg.ImageItem`. For each output pixel we look up the `(ray, range_bin)` it samples, fetch the cell value, and run the colormap — that's the polar shader; it just lives on the CPU as numpy gather + LUT, not as GLSL on a GPU.
- **Parallel rasterize across panels.** Each panel's polar shader runs on its own worker thread from a shared `ThreadPoolExecutor(max_workers=4)`. numpy releases the GIL for the gather + clip + take ops inside `_rasterize_polar`, so a 4-panel grid's 4 × 15 ms serial cost collapses to ~15-20 ms wall-clock. The grid does a three-phase render — main-thread prepare (warm caches, slice sweep data), worker-thread rasterize (parallel), main-thread commit (`setImage` + overlays).
- **1024² image at 1.5× padding.** The rasterizer always renders at `RADAR_IMAGE_SIZE_PX = 1024` (chosen to match the per-panel pixel count on typical displays without leaving Qt's raster engine in slow paths) into an image rect that is **1.5× the visible view in each direction**. Small pans within that headroom hit the same chosen rect → no re-rasterize fires → pan is essentially a scene-transform update.
- **Snap-to-discrete-extent zoom.** The view rect is rounded *up* to one of six discrete extent levels (20 / 40 / 80 / 120 / 160 / 250 km half-width) and its center snapped to a coarse grid (`half / 2`). The cache reuses lookup tables across many sweeps at the same zoom level.
- **Cached per-pixel polar lookup.** The `(ray_idx, bin_idx, valid)` table is built once per `(extent_rect, polar_signature)` and cached as an `OrderedDict` (max 16 entries). A fresh sweep at the same VCP/tilt and zoom level skips the trigonometry entirely; only the colormap + image upload runs. The lookup + LUT caches are pre-warmed on the main thread before the worker fan-out so concurrent panels never race on cache writes.
- **256-entry colormap LUT** (one `np.take` instead of four `np.interp` calls) — ~10× faster per render than going through pyqtgraph's `ColorMap.map` directly.
- **Cached static overlays across sweep changes.** Range rings, state borders, county borders, and city dots/labels persist across time-scrubs — `_draw_overlays` is keyed on the overlay bundle's `id()` and no-ops when the bundle is unchanged. Only the *dynamic* overlay layer (game polygon, warnings, MCDs, live reports) gets torn down and rebuilt per render. Saves ~10–20 ms per scrub tick before the rasterize even runs.
- **Geometry simplification at load time.** State and county rings get Douglas-Peucker-simplified (`shapely.simplify`, tolerance 0.2 km for states / 0.1 km for counties) when the overlay bundle is built. State borders collapse from ~10k total vertices to ~500, counties from ~80k to ~18k — drops `QPainter.drawPath` time per paint and shrinks the per-frame concat buffer used by the bbox cull.
- **View-culled state / county borders.** Each border ring is bbox-tested against the current view rect; only rings that actually intersect the visible area are concatenated into the `PlotCurveItem` data. Drops per-panel pan cost further at high zoom.
- **Greedy non-overlap city labels** with off-screen culling.
- **Configurable radar LRU cache** keeps recent PyART `Radar` objects in memory (default 24, min 6, max 100 via the `radar_lru_size` constructor kwarg).
- **Background pre-load + pre-dealias.** As soon as the prefetcher finishes downloading a Level 2 volume, a 2-worker `ThreadPoolExecutor` runs `pyart.io.read_nexrad_archive` and `dealias_region_based` on it and stashes the result in a shared preload cache. The grid checks that cache before falling back to a synchronous parse — turning the per-volume-crossing main-thread stall from **~1.9 seconds** (270 ms read + 1600 ms region-based dealias on a busy WSR-88D volume) into a **~1 μs** dict lookup. This is the single biggest perceived-latency win for time-bar scrubbing through multi-volume rounds.
- **Async + throttled lookahead refill.** `prefetcher.advance_clock` is called once per game-clock tick (1 Hz) but the actual S3 ListObjectsV2 + download enqueue runs on a dedicated single-worker tick pool — the main thread returns immediately. A 15-second game-time throttle further short-circuits redundant relists (the lookahead window is 20 min, so polling every second was overkill). Drops the per-tick main-thread cost from ~100-500 ms (synchronous S3 LIST per enabled radar) to ~10 μs.
- **Render-key diffs in tick setters.** `set_player_warnings`, `set_live_reports`, and `set_game_polygon` short-circuit when the input set's identity-and-revision signature matches the last render. The game tick fires every second whether the player has issued a warning or not — without the diff the radar grid was re-rasterizing all four panels on every tick (~16 ms × 4 = 60 ms wasted per idle tick). Now only ticks that actually change visible state pay the rerender.
- **Scrub debouncing.** Rapid ←/→/↑/↓ keypresses (and slider drags) update the logical position synchronously but coalesce into a single render via a 15 ms QTimer — short enough to feel instantaneous on a single keystroke while still folding keyboard auto-repeat (~30-60 ms) into one render.
- **View-change debouncing — two-layer.** Pan/zoom drags update the radar's pending view rect but coalesce into one re-rasterize after 40 ms of idle. A separate 40 ms timer batches the *overlay* work (border-ring bbox cull, city-label relayout) so the dozens-per-second `sigRangeChanged` ticks during a drag don't re-walk thousands of ring AABBs each frame.
- **Origin-skip broadcast.** When one panel pans/zooms, the grid mirrors the limits to all *other* panels but skips the originator (whose view is already correct) — avoids a redundant `setRange` round-trip and the second `sigRangeChanged` it would fire on the source panel.
- **Hidden axes fully unlinked.** `hideAxis()` alone leaves the AxisItem listening to `sigXRangeChanged` and recomputing its (invisible) HTML label every tick. We disconnect the link in `RadarPanel.__init__` so hidden axes are truly inert during pan.

Measured (M-class laptop, super-res NEXRAD 720 × 1832 = 1.3 M cells, Qt offscreen):

| Operation | Latency |
|---|---|
| Warm-cache rasterize, 1 panel | **~15 ms** (1024², lookup-table hit) |
| Cold-cache rasterize, 1 panel | ~50–70 ms (lookup-table build) |
| **Time-bar scrub, 4-panel grid (parallel)** | **~16 ms / frame** (~60 fps, 100% of frames <33 ms) |
| Volume-crossing scrub stall (WSR-88D, region-based dealias) | **~0 ms** with preload (vs ~1900 ms cold-cache) |
| Game tick on main thread, solo (idle, 4-panel grid) | **~0.9 ms / tick** (vs ~100-500 ms before — S3 LIST + always-rerender) |
| Pan / zoom frame, 4-panel grid (synced) | **~2 ms / frame** (>400 fps headroom, 100% of frames <16 ms) |
| Mouse-drag pan (120 consecutive frames) | **~2 ms / frame** (max 2.5 ms, 0 frames >16 ms) |
| Cold-cache zoom-level snap change | one-time ~50–70 ms per new snap rect |

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

tests/                      # 261 pytest tests across 20 files
```

---

## Testing

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

261 tests covering: SAILS sweep index, 5-km buffered point-in-polygon, scoring metrics (incl. team aggregation, magnitude revisions, per-revision tier), tier multipliers, MCD PIB scoring and anti-spam, protocol round-trip, multiplayer state appliers, date-blinding scrape, casualty regex, cache, clock, session state machine, replay, event reveal, colors, sites, live source, round builder. Runs in ~2 seconds.

GitHub Actions CI at [`.github/workflows/test.yml`](.github/workflows/test.yml) runs the suite on Python 3.11 and 3.12 with Qt offscreen + cartopy system deps.

---

## Known limitations & future work

- **TURN server** not bundled; symmetric-NAT users can't connect in multiplayer
- **Host disconnect** ends the round in v1 (no host migration)
- **Late-warn POD credit** is allowed only for TORR (10-minute window)
- **Replay playback** UI not implemented; replay JSON files are written but only readable by tooling
- **SPC final-EF backfill** is best-effort; SPC's daily filtered CSV has a messy multi-section format that defeats naive `pd.read_csv` and we fall back to IEM preliminary data when matching fails
- **All four panels show the same radar at a time.** Switching between enabled radars uses the **Radar:** dropdown in the toolbar (appears when the host enabled more than one site). Each site's volumes are downloaded + indexed independently in the background, so switching is instant once the new site has at least one volume cached; otherwise the panels show a "no sweep selected" placeholder until the new site's first volume lands.
- **Town labels** beyond major cities not wired (counties drawn but unlabeled)
- **`adjustText`** is no longer installed/used; some city labels may overlap at certain zoom levels (the greedy non-overlap algorithm hides smaller cities to compensate)

---

## Credits

- NEXRAD Level 2 data: NOAA via the [Unidata IDD](https://www.unidata.ucar.edu/data/) S3 mirror (historical) and [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/) live HTTP mirror
- LSRs: Iowa Environmental Mesonet
- County polygons: U.S. Census Bureau TIGER 2023 cartographic boundaries
- PIB tables: Storm Prediction Center mesoscale discussion conventions
- Reference game: [Reference-Nowcastle/](Reference-Nowcastle/) (gitignored; not part of this repo)
