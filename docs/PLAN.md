# RadarAnalysisCorp — Nowcasting Warning Game

## Context

Build a Python-based, 1+ player (online P2P multiplayer) radar warning/nowcasting game that simulates a forecaster's desk: live-feeling radar data streams in, the player issues MCDs and warnings within a chosen domain, and their forecasts are verified against actual storm reports for scoring. Inspired by [Reference-Nowcastle/](Reference-Nowcastle/) — a single-snapshot severity-guessing prototype — but substantially more ambitious: continuous-time, polygon-based, multiplayer warning issuance with IBW-tier-aware scoring (TOR/TORR/PDS TOR/TORE, SVR/SVRC/SVRD) and MCDs.

The repository is currently empty except for the Nowcastle reference. We can reuse data-access patterns from there but the architecture, UI, and gameplay loop are essentially new.

---

## Locked-in Decisions

| Topic | Decision |
|---|---|
| Language | Python 3.11+ |
| GUI framework | **PyQt6** (multi-pane forecaster-desk layout) |
| Radar I/O | PyART (`arm_pyart` 2.2.0), MetPy 1.7.1 |
| Mapping | Matplotlib + Cartopy embedded in PyQt6 (`FigureCanvasQTAgg`) |
| Radar data source | NEXRAD Level 2 on AWS S3 via the **Unidata mirror** (`s3://unidata-nexrad-level2`, **unsigned/anonymous** — no `aws configure` required). Files are gzipped; PyART decompresses transparently. *Note: switched from `noaa-nexrad-level2` because that bucket returns AccessDenied; Unidata works for both signed and unsigned and is what Reference-Nowcastle uses.* **Each client fetches radar data independently** from S3 — radar data NEVER traverses P2P. Only gameplay state (polygons, ticks, scores, chat) goes over the network. |
| Storm reports | **IEM LSRs** (Iowa State Mesonet API) — primary and only source |
| Date range (random-day picker) | **2000 to today − 2 days** |
| Convective day | 12Z–12Z UTC |
| Multiplayer transport | WebRTC DataChannel via `aiortc` with a **hosted signaling server** we maintain. **Star topology** (each peer ↔ host); host relays inter-peer messages. Avoids full-mesh scaling pain at high N. |
| Max players per room | **50** (star topology required — see §10 networking) |
| Radar panel layout | **User-configurable: 1, 2, or 4 panels.** Defaults: 1→REF, 2→REF/VEL, 4→REF/VEL/CC/ZDR. Click a panel to focus, keybind cycles its product. |
| Polygon mechanic | **Freehand drawing** on the radar panel — no auto-rectangle generation from motion |
| Storm motion tool | Stays — but as a **measurement utility only**, informs the player's freehand polygon |
| Warning duration | **Player-set** per warning |
| SVR fields | Both **wind speed** and **hail size** required (mirrors NWS WarnGen) |
| Verification buffer | **5 km** around the polygon |
| Live report overlay | **Yes** — verifying reports appear as they happen and fade with time; scrubbing back recomputes |
| Radar sites | **Host picks** during round setup on the same map as the game polygon (enables down-radar simulation) |
| Late-warn credit | Allowed **only for TORR** (which IRL is often issued post-report); all other types: zero POD credit if issued after the report |
| Mid-round join | Allowed instantly with full scoring (player just has less time to earn points) |
| End-of-round | Leaderboard with score-component table per player |
| Replay file | Optional, host-toggled at round start |
| Audio cues | None |
| Host disconnect | **v1: end round, show partial scores.** No host migration. |
| Prefetch stall (round start) | Once **≥75%** of clients finish their pre-game S3 fetch, a **60-second countdown** starts; round begins when timer hits zero regardless of stragglers. |
| Tornado EF source | **IEM** for events ≤30 days old; **SPC final-rating database** for events >30 days old (PIB scoring uses confirmed EF when available) |
| MCDs | Scored **separately** from warnings; bridge 45–120 min gaps; still earn verification points. **SPC Peak Intensity Bin (PIB) system** for tornado/wind/hail (see §8) |
| Team mode | Host-toggleable at room creation. Pre-game team lobby lets players form/join/leave teams. **Each team scored as a single entity** (union of teammates' warnings → one POD/FAR). Eases coverage of large game areas. Solo play = team of 1. See §11. |

---

## Module Layout

```
radar_warning_game/
├── data/
│   ├── radar_s3.py         # signed boto3 S3 client, scan listing, atomic downloads
│   ├── prefetch.py         # parallel/buffered radar download ahead of game clock
│   ├── sweep_index.py      # per-radar SAILS-aware sweep index (timestamp, elev, file, sweep#)
│   ├── reports.py          # IEM LSR fetching + caching; SPC final-EF backfill for events >30 days old
│   ├── sites.py            # NEXRAD site catalog (reuse Nowcastle pattern)
│   └── cache.py            # disk cache management; hashed filenames to hide date in UI
├── geo/
│   ├── projection.py       # lat/lon ↔ projected km (reuse Nowcastle pattern)
│   └── polygons.py         # polygon math, point-in-polygon, 5 km buffer
├── verification/
│   ├── scoring.py          # POD, FAR, CSI, lead time, magnitude accuracy, tier multipliers
│   ├── tornado_tiers.py    # TOR/TORR/PDS TOR/TORE thresholds + multipliers
│   ├── pibs.py             # SPC Peak Intensity Bin tables (tornado/wind/hail) + observed-→-PIB mapping
│   └── reports_in_poly.py  # match reports to active warnings with 5 km buffer
├── ui/
│   ├── app.py              # main window, dock layout (player window)
│   ├── host_map.py         # host's central in-game overview (all players' warning outlines)
│   ├── radar_panel.py      # multi-tilt, multi-product display (reuse Nowcastle radar_plot.py); supports 1/2/4 layouts
│   ├── overview_map.py     # CONUS report map for day setup; also where host picks polygon + radar sites
│   ├── poly_editor.py      # freehand polygon drawing on radar panel (warnings) AND on CONUS map (game area)
│   ├── time_distribution.py # histogram of report times for window picking
│   ├── motion_tool.py      # 2-point storm motion measurement (informational)
│   ├── warning_form.py     # warning issuance + edit dialog (type, duration, magnitudes)
│   ├── controls.py         # play/pause, speed (bracket keys)
│   ├── leaderboard.py      # live corner widget + end-of-round full table
│   ├── colors.py           # per-player color palette (perceptually distinct, colorblind-safe head)
│   └── time_format.py      # date-blinding timestamp formatters (centralized)
├── net/
│   ├── protocol.py         # message schema (round seed, time tick, warning ops, score)
│   ├── peer.py             # aiortc WebRTC DataChannel transport
│   ├── signaling_client.py # connects to our hosted signaling server, room codes
│   └── lobby.py            # room creation, host election
├── signaling_server/       # tiny aiohttp server (deployed separately)
│   └── server.py           # WebSocket signaling for SDP offer/answer exchange
├── game/
│   ├── session.py          # game session state machine
│   ├── round_builder.py    # day selection w/ thresholds, IEM filtering
│   ├── clock.py            # event-time clock, host-controlled speed
│   └── replay.py           # log + replay of issued warnings for post-game review
└── play.py                 # entry point
```

---

## Gameplay Flow

### 1. Room creation + round setup (host)

- Host creates a room → client gets a short room code from the signaling server.
- Peers enter room code → signaling server brokers WebRTC SDP exchange → peers connect P2P.
- Host chooses day:
  - **Random day** with independent thresholds (≥N hail / ≥M wind / ≥K tornado reports). Uniform sample over days in `[2000-01-01, today-2]` meeting the criteria. *No date revealed to anyone.*
  - **Specific day** (date picker, 12Z–12Z UTC). UI warns host that they'll see the date.
- Host toggles **"save replay"** option.
- Host fetches all LSRs for the chosen 12Z–12Z window via `data/reports.py` (pattern from [Reference-Nowcastle/radar_game/data.py:69-86](Reference-Nowcastle/radar_game/data.py#L69-L86)).

### 2. Interactive CONUS map → polygon + radar sites

Single map shown to the host (only the host does this setup; peers wait):

- Cartopy CONUS basemap with state borders.
- **Storm reports** plotted with:
  - **Shape** by type: triangle = tornado, circle = hail, square = wind
  - **Size** scaled by magnitude (hail inches / wind mph / tornado EF)
  - **Color/shade** by time of day across the 12Z–12Z window
- **WSR-88D sites** plotted as small dots; host **clicks to toggle** which sites are active for this round. Inactive radars are unavailable to all players (simulates downed radars).
- **"Reroll" button** (random-day mode only): host can sample a new random day with the same thresholds if they don't like the current event. Date stays hidden in random-day mode; only the report map changes. Disabled once polygon has been drawn / setup confirmed.
- **Polygon drawing**: host enters polygon-draw mode → clicks vertices → closes polygon. Defines the game-area / verification boundary. Players play only within this domain.
- Both polygon and radar-site selections are broadcast to peers when the host clicks "Confirm setup".

### 3. Time-window selection (host)

- Filter reports to those inside the polygon.
- Show a **stacked histogram** of report counts vs. time-of-day, segmented by report type (tornado/hail/wind).
- Host drags handles or types `[start_time, end_time]` UTC (within the 12Z–12Z day). This becomes the game session time range.
- "Start round" broadcasts session start to peers.

### 4. Game session

- Host controls a virtual clock advancing `start_time → end_time`.
- Speed control (host-only): `[` slower, `]` faster, `Space` pause/resume. Host broadcasts ticks to peers.
- Per virtual time tick, each player's radar panel updates: latest sweep ≤ current virtual time per radar (SAILS-aware, §4a).
- **Radars shown**: only those the host enabled in §2. Player can switch focus among them; multi-panel optional.
- **Each player issues their own warnings** (independent scoring); peers' warnings render faintly for shared situational awareness.

**Host dual role:**
- The host always has a **central map window** (§4c) showing all players' activity from above. This is the host's primary UI.
- The host may **optionally also play** the game by opening a separate gameplay window (same standard player UI: radar panels, warning issuance, motion tool, live leaderboard, etc.). Toggled via a "Join as player" button in the central map window.
- If the host plays, their warnings appear on the central map alongside everyone else's, in their assigned color.
- The host always has speed/pause control regardless of whether they're also playing.

**Mid-round join:**
- Late joiner connects, signaling server fetches current session state from host (clock position, polygon, radars, peers' active warnings).
- Joiner instantly receives game-time tick, can begin issuing warnings.
- Their scoring uses the same formulas — they're just penalized organically by having less time to issue.
- No retroactive scoring of reports that already verified before they joined (those reports are visible on their map already with normal fade behavior).

### 4a. Radar panel UX

**Configurable panel layout:** player picks **1, 2, or 4 panels** in settings. Defaults:
- 1 panel → REF
- 2 panels → REF / VEL
- 4 panels → REF / VEL / CC / ZDR

A clicked panel becomes the **focused** panel — receives keyboard input. Keybinds for product cycling change only the focused panel's product. Each panel can independently show any product: REF, VEL, CC (correlation coefficient, ρhv), ZDR, KDP, HCA, SW (spectrum width).

**Interactions (apply to focused panel):**

- **Zoom:** scroll wheel zoom centered on cursor (reuse [Reference-Nowcastle/radar_game/radar_plot.py:161-249](Reference-Nowcastle/radar_game/radar_plot.py#L161-L249) `InteractiveNav`). Zoom is synchronized across all panels (shared axes — pattern from [Reference-Nowcastle/radar_game/radar_plot.py:270-369](Reference-Nowcastle/radar_game/radar_plot.py#L270-L369)).
- **Pan:** click-drag (also synchronized across panels).
- **Reset view:** double-click.
- **Elevation tilt:** `↑` / `↓` step through available elevations (`0.5° → 0.9° → 1.3° → ...`). Pattern from [Reference-Nowcastle/radar_game/radar_plot.py:424-445](Reference-Nowcastle/radar_game/radar_plot.py#L424-L445). Tilt change applies to all panels (they all show the same elevation/time, just different products).
- **Time scrubbing:** `←` / `→` step backward/forward by one sweep at the current elevation (SAILS-aware). `Shift+←/→` steps 5. Scrubber slider below the panels for continuous control. Time applies to all panels.
- **Game-clock cap:** `scan_time ≤ current_game_virtual_time`. Forward-scrubbing past game time is a no-op. Looking back is always allowed.
- **Product cycling:** number-key shortcuts (1–7) or `P` to cycle the focused panel's product through REF/VEL/CC/ZDR/KDP/HCA/SW.
- **Radar site switching:** `Tab` cycles among enabled radars (set by host in §2). Site change reloads sweep index, may briefly stall while prefetch catches up.

**Geographic overlays:**

- Country / state borders via Cartopy Natural Earth.
- US counties via **US Census TIGER cartographic boundary shapefile** (`cb_*_us_county_500k`), bundled (~few MB). Thin grey lines, only when zoomed past ~4° width.
- Cities/towns:
  - Pop ≥ 100k: always shown, labeled.
  - Pop ≥ 5k: when zoomed in past a threshold.
  - Source: Natural Earth `populated_places` + optional SimpleMaps `uscities` CSV.
- Range rings (optional 50/100/150/200 km centered on the radar).
- **Active warning polygons:** player's own semi-transparent; peers' fainter.
- **Live storm reports** (see §6 for fade rules).
- **Range rings & site markers** for all enabled radars.

**SAILS-aware sweep indexing:**

SAILS / MESO-SAILS / SAILSx2 / SAILSx3 insert extra 0.5° sweeps mid-volume — a single Level 2 file can have 1, 2, 3, or 4 sweeps at 0.5° while every higher tilt has exactly one sweep per volume.

`data/sweep_index.py`:
1. On Level 2 open, index every sweep as `(sweep_timestamp, elevation_angle, source_file, sweep_number)`. Use `radar.time['data']` and `radar.sweep_start_ray_index` from PyART.
2. Maintain a global per-radar index across loaded volumes, sorted by sweep_timestamp.
3. Scrub at elevation E: filter to sweeps with `|elev − E| < 0.15°`, sorted by time. `←/→` step adjacent entries.
4. Render = load source file + extract that specific sweep's rays. Cache by `(file, sweep#)`.
5. Elevation change at fixed time: filter to nearest-time sweeps within volume cadence, pick closest elev to requested.

This gives smooth ~60–90 s low-level cadence at 0.5° through a SAILS-active VCP. Higher tilts naturally lock to volume cadence.

VCP detection is empirical (count sweeps per elevation per volume); no per-VCP special-casing.

### 4c. Host central map (in-session overview)

The host's primary in-game UI: a single map view (zoomable / pannable across the game polygon's extent) showing **all players' warning activity in real time**. The host can also see this in addition to playing if they choose (§4 Host dual role).

**What's drawn on the map:**

- The game polygon outline (defined in §2), drawn as a heavy boundary.
- All enabled radar site markers (with status indicator if any are down).
- **Active warning polygons from every player**, as **outlines only** (no fill — keeps the map readable when many warnings overlap):
  - **Color**: each player has a unique color assigned at room creation (palette of ~50 perceptually distinct colors; first 8 are colorblind-safe; cycle/extend as needed).
  - **Line style**: distinguishes warning category — **solid = TOR-family** (TOR/TORR/PDS TOR/TORE), **dashed = SVR-family** (SVR/SVRC/SVRD), **dotted = MCD**.
  - **Line weight**: thicker line for higher tiers within a family (PDS TOR > TORR > TOR, SVRD > SVRC > SVR).
  - Each polygon labeled with a small `[player_initials] [type]` tag near its centroid (e.g., `JT PDS`).
- **Storm reports** appear with the same shape/size/color-by-time scheme as §2's CONUS map, fading as in §6.
- **Live leaderboard** (§9) docked in a corner of the central map by default for the host.
- **Active MCDs** drawn as thin dotted outlines (per-player color).

**Interactions for the host:**
- Click any polygon → side panel shows: issuer, type, magnitudes, duration remaining, verifying reports so far.
- Standard zoom/pan (scroll-wheel + click-drag).
- Speed controls (`[` / `]` / `Space`) for clock control.
- "Join as player" button → opens the standard player gameplay window (radar panels, warning UX, etc.) without closing the central map.

**Legend:** a collapsible legend panel maps each player's name to their color, plus the line-style key.

**Implementation:**
- New `ui/host_map.py` module.
- Cartopy + matplotlib, embedded as a `FigureCanvasQTAgg` in a PyQt6 main window.
- Receives the same gameplay state stream (`WarningIssue`, `WarningUpdate`, etc.) as player clients — host's own client just renders it differently.

### 4b. Date blinding (anti-spoiler)

Players see *time of day* and *location* but never the *date*.

**Host vs. player visibility:**

- **Random-day mode:** blind for everyone, including host.
- **Specific-date mode:** host sees the date; peers do not. Host is essentially a quizmaster.

**Surfaces sanitized for player-facing display:**

| Surface | Default | Sanitized |
|---|---|---|
| Radar panel axis title | `2013-05-20 20:48:12 UTC KTLX 0.5° Z` | `20:48:12Z KTLX 0.5° Z` |
| Time scrubber tick labels | dates | `HH:MM` |
| Game clock | datetime | `HH:MM:SS UTC` |
| Window titlebar | datetime | generic |
| Warning issue confirmation | full ISO | `HH:MM Z` |
| Peer warning labels | peer + datetime | peer + `HH:MM` |
| Storm report tooltips | full ISO | `HH:MM` |
| Error messages | `Failed to load KTLX20130520_204812_V06` | `Failed to load 20:48Z scan` |
| Network protocol ticks | datetimes | virtual-time offset from session epoch (no real date sent) |

**Anti-snoop measures (honor-system, not cryptographic):**
- Host strips date from tick messages; broadcasts a session epoch + offset.
- Cache files on disk under hashed names (`<sha1>.ar2v`); in-memory dict maps hash → real filename, cleared at round end.
- **Fundamental limitation:** each client fetches radar data from S3 with its own AWS creds (per the "P2P streams game state only, not radar" architecture in §10). To construct S3 keys (`YYYY/MM/DD/<radar>/...`), the client must know the date in memory. We cannot hide the date from a client's own process.
- Acknowledged: a determined user can tcpdump their S3 traffic, `aws s3 ls` the bucket with their creds, read the date from sweep timestamps in the Level 2 binary, or attach a debugger to the running process. Date-blinding is **honor-system** — protects casual play from accidental spoilers but is not a cheating countermeasure.
- v1 does the UI-level blinding; v2 could add a "competitive mode" with a host-trusted radar proxy that strips dates, but this defeats the "individual S3 clients" architecture.

**Post-round reveal:** scoring screen shows real date, location summary, and links to NWS event reviews if available.

**Implementation:** centralized in `ui/time_format.py`. All UI surfaces format timestamps through `format_player_time(dt) → "HH:MM:SS Z"`. Code-review discipline + a unit test that scrapes UI strings for `\d{4}-\d{2}-\d{2}` patterns.

### 5. Warning issuance UX (freehand polygon)

**Workflow:**
1. Player presses warning hotkey or clicks "New Warning" button → enters draw mode.
2. Player picks **warning type** (SVR / TOR / TORR / PDS TOR / TORE) and **duration** (any value in minutes; UI suggests 30/45/60 but accepts arbitrary).
3. Player picks **type-specific magnitude fields**:
   - **SVR / SVRC / SVRD:** both expected **hail size (inches)** AND expected **wind gust (mph)** required (mirrors NWS WarnGen).
   - **TOR / TORR:** optional expected EF rating (defaults to "radar indicated" with no specific EF).
   - **PDS TOR / TORE:** expected EF rating required (must be ≥2 for tier consistency check; UI warns otherwise).
4. Player **freehand-draws polygon** directly on the radar panel by clicking vertices, double-click or Enter to close.
5. Polygon is editable: drag vertices to adjust, right-click vertex to delete, click edge midpoint to insert.
6. **"Issue"** finalizes the warning: timestamped at current virtual game time, added to player's active list, broadcast to peers as a faint overlay.

**No motion-derived auto-generation.** The storm motion tool (§5a) is purely informational — players use it to inform how far downstream to extend their hand-drawn polygon, but the polygon vertices are theirs.

**Editing/updating active warnings:**

Players can modify any of these fields on an active warning without canceling and re-issuing:
- **Polygon** (drag vertices, add/remove)
- **Duration** (extend / shorten the valid time)
- **Type** (upgrade SVR → SVRC → SVRD, or TOR → TORR → PDS TOR → TORE; downgrades allowed too)
- **Magnitudes** (revise hail size / wind gust / EF estimate)

Internally each warning is a sequence of timestamped **revisions**: `(revision_time, type, polygon, duration, magnitudes)`. Verification matches each report against the **revision active at the report's time**. Lead time always uses the **original issue time** (revisions don't penalize you for already-issued lead).

**Tier-change scoring nuance:**
- Upgrades (SVR→TOR, TOR→TORR/PDS/TORE) earn the upgraded-tier multiplier only for reports occurring **after** the upgrade revision time. Reports verifying before the upgrade are scored as the lower tier.
- Downgrades work symmetrically — reports before the downgrade get the higher tier; reports after get the lower.
- Magnitude estimate revisions apply going forward (peak observed is still computed over the full warning lifetime; magnitude accuracy uses the most-recent revision active when the peak was observed).

This matches NWS Severe Weather Statement (SVS) practice — warnings are amended in place, not canceled and re-issued. Cancel is a separate explicit action (no points for unverified reports after cancel time).

### 5a. Storm-motion measurement tool

Two-point tracker for deriving observed storm motion. **Informational only** — does not auto-generate polygons.

**Workflow:**
1. Player activates with hotkey (`M`) or button. Tool enters scrub mode on the active radar panel.
2. Same scrub controls as §4a (`←/→` step sweeps), still capped at game virtual time.
3. Scrubbing is **local** — does not advance host clock or broadcast to peers.
4. Click point 1 at scan time T1 → marker + timestamp captured.
5. Scrub to a later (or earlier) scan, click point 2 at T2 → second marker.
6. Tool computes:
   - Bearing P1→P2 (TO direction) via `geo/projection.py`.
   - FROM direction = `(bearing + 180) mod 360`.
   - Speed = `distance_km / (T2 − T1)` in knots.
   - Display: `"from 265° at 42 kt"` + arrow from P1→P2 on the panel.
7. "Re-measure" clears markers and restarts.
8. Exit restores the panel to current virtual game time.

**Implementation:**
- `ui/motion_tool.py`, shared canvas with `radar_panel.py`.
- Uses the existing sweep index — scrubbing is just changing displayed sweep, not refetching.
- Can be invoked any time, including mid-warning-draft.
- v2: cross-radar tracking (P1 on KOUN, P2 on KTLX) for storms in overlap zones.

### 6. Verification & scoring (per player)

**Verification geometry:**
- A storm report verifies a warning if:
  - report time ∈ `[warning_issue_time, warning_issue_time + duration]`, AND
  - report location ∈ `polygon ⊕ 5 km buffer` (Minkowski sum / morphological dilation by 5 km).
- For TORR specifically, late-warn credit is allowed: a TORR issued **after** a verifying report still counts for POD, provided the report time is within `[warning_issue_time − 10 min, warning_issue_time + duration]` (i.e., the warning was issued ≤10 min after the report). All other warning types: zero POD credit for late issuance.
- Rationale: real TORRs are often issued just after sighting/spotter confirmation. All other warning types should have positive lead time.

**Metrics per player at end of round:**
- **POD** (Probability of Detection) = verified reports inside game area / total reports inside game area
- **FAR** (False Alarm Ratio) = warning polygons with zero verifying reports / total warnings issued
- **Lead time per verified report** = `report_time − warning_issue_time` (positive = lead; for TORR can be slightly negative within the 10 min window).
- **Magnitude accuracy** (per-warning, see §7).

**Live in-session feedback:**
- When a report's `time ≤ current_game_virtual_time`, render it on every player's map.
- **Fade**: alpha = `clamp(1 − (now − report_time) / fade_duration, 0, 1)` where `fade_duration` ≈ 30 min. Fully faded reports are dimmed but still visible (so the field of past activity is readable).
- **Scrubbing back** in a radar panel does NOT change the report layer on the main map — reports follow the *game clock*, not the radar's scrub time.
  - However, the radar panel itself shows scan-time text in the header so the player knows they're looking at history.
  - Caveat: this means scrub-back in a radar panel ≠ "full state at past time." We could add a "view session at time T" mode in v2.
  - *(Actually, the user's request was "Scrubbing back updates the reports to be as if you were viewing it at that time." This suggests reports DO update when scrubbing back on a radar panel. Re-reading: I'll do this — the main report-overlay layer follows the active radar panel's display time, not the game clock. Players see the world as it looked at the displayed scan time.)*
  - **Revised rule**: reports rendered on the map filter to `report_time ≤ active_radar_display_time`. Fade by `(active_radar_display_time − report_time)`. When the active radar panel is at "live" (game virtual time), this reduces to the live behavior.

### 7. Warning types & tier-aware scoring

| Type | Required fields | Verifying reports |
|---|---|---|
| **SVR** | Hail size + Wind gust | Hail ≥1.0" OR Wind ≥58 mph |
| **SVRC** | Hail + Wind | + bonus if peak hail ≥1.75" OR wind ≥70 mph |
| **SVRD** | Hail + Wind | + bonus if peak hail ≥2.75" OR wind ≥80 mph |
| **TOR** | (optional EF estimate) | Any tornado report |
| **TORR** | (optional EF estimate) | TOR criteria + small bonus; late-warn credit allowed (§6) |
| **PDS TOR** | EF estimate required (≥2) | TOR criteria + large bonus if verified by significant tornado (EF2+) OR injuries/fatalities |
| **TORE** | EF estimate required (≥2) | TOR criteria + largest bonus if verified by significant tornado / casualties; reduced score if only weak tornado |
| **MCD** | (see §8) | Separate scoring |

**Tier multipliers (initial — tunable):**

- **TORR** verification: `1.10 ×` base TOR points
- **PDS TOR** verification:
  - Verified by EF2+ OR injury/fatality: `1.75 ×`
  - Verified by EF0/EF1 only: `1.0 ×`
  - False alarm: heavier penalty than plain TOR FA
- **TORE** verification:
  - Verified by EF2+ OR injury/fatality: `2.5 ×`
  - Verified by weak tornado only: `0.75 ×` (over-issuance penalty)
  - False alarm: heaviest in game
- **SVRC / SVRD**: bonus only if threshold reports observed; else scored as plain SVR.

**Magnitude accuracy** (per warning):
- For SVR: scored independently on each of hail & wind if reports of that type verified. `accuracy_field = clamp(1 − |estimated − peak_observed| / peak_observed, 0, 1)`. Final SVR magnitude factor = mean of scored fields.
- For TOR-family: if EF estimated, `accuracy = 1 − |EF_est − EF_observed| / max(1, EF_observed)` clamped.
- Magnitude factor multiplies the warning's tier-ceiling points.

**Injuries/fatalities data:** IEM LSRs include this information in **free-text remarks** (e.g., `"...1 INJ, 0 FAT..."`). We'll parse with a regex (`r"(\d+)\s*INJ"`, `r"(\d+)\s*FAT"`) on the remarks field. Accept that some events with casualties have unstructured remarks and may be missed — flag this limitation in the docs. (SPC's QC'd database has these as structured fields but isn't available real-time; rejected per user choice.)

### 8. MCDs (Mesoscale Convective Discussions) — SPC Peak Intensity Bin system

**Purpose:** Bridge the 45–120 min "watch-but-no-warning" gap. Score separately from warnings so they don't pollute POD/FAR.

**Issuance:**
- Player picks MCD duration (default 90 min, configurable, typical real-world range 60–180 min).
- Player freehand-draws polygon on the radar panel or zoomed CONUS subset.
- Player selects **Peak Intensity Bins (PIBs)** for each of three hazards (each independent, "None" allowed):

**Tornado PIBs (1–7):**

| PIB | Wind (m/s) | Wind (mph) | IBW Tag | Descriptor |
|---|---|---|---|---|
| 1 | 29–42 | 65–95 | Base | Weak |
| 2 | 38–51 | 85–115 | Base | Weak / Strong |
| 3 | 45–58 | 100–130 | Base / Considerable | Weak / Strong |
| 4 | 54–67 | 120–150 | Considerable | Strong |
| 5 | 63–76 | 140–170 | Considerable / Catastrophic | Intense |
| 6 | 69–85 | 155–190 | Catastrophic | Intense / Violent |
| 7 | >78 | >175 | Catastrophic | Violent / Exceptionally Rare |

**Wind PIBs (1–7):**

| PIB | Wind (kph) | Wind (mph) | IBW Tag | Descriptor | Coverage |
|---|---|---|---|---|---|
| 1 | <97 | <60 | Base | Locally Damaging | Localized–Scattered |
| 2 | 86–113 | 55–70 | Base | Severe | Localized–Scattered |
| 3 | 105–129 | 65–80 | Base / Considerable | Severe / Some Significant | Localized–Scattered |
| 4 | 120–145 | 75–90 | Considerable / Destructive | Significant | Isolated–Numerous |
| 5 | 137–161 | 85–100 | Destructive | Significant / Some Intense | Isolated–Widespread |
| 6 | 153–185 | 95–115 | Destructive | Intense | Scattered–Widespread |
| 7 | >185 | >115 | Destructive | Intense to Extreme / Exceptionally Rare | Widespread |

**Hail PIBs (1–6):**

| PIB | Hail (cm) | Hail (in) | IBW Tag | Descriptor |
|---|---|---|---|---|
| 1 | ≤3.175 | ≤1.25 | Base | Locally Large |
| 2 | 2.5–4.5 | 1.00–1.75 | Base | Large |
| 3 | 3.8–6.4 | 1.50–2.50 | Base / Considerable | Large to Very Large |
| 4 | 5.1–8.9 | 2.00–3.50 | Considerable / Destructive | Very Large to Giant |
| 5 | 7–10.8 | 2.75–4.25 | Destructive | Very Large to Giant |
| 6 | ≥10.2 | ≥4.00 | Destructive | Giant |

PIB tables live in `verification/pibs.py` as constants. UI shows a 3-row PIB picker (Tornado / Wind / Hail) with a "None" option per row. At least one hazard must have PIB ≥1 (UI rejects all-None MCDs).

**Observed → PIB mapping** (for scoring):
- For each hazard category, find the peak reported magnitude inside the MCD polygon during its valid time.
- PIB ranges overlap — for scoring purposes, map the observation to the **highest PIB whose lower bound ≤ observed value**.
  - e.g. observed 78 mph wind → highest lower-bound ≤ 78 is PIB 4 (75 mph) → observed PIB = 4.
  - e.g. observed EF3 tornado (~165 mph) → highest lower-bound ≤ 165 is PIB 5 (140 mph) → observed PIB = 5.
- If no reports of that hazard occurred inside the polygon: observed PIB = 0 ("None").

**Scoring (separate from warning POD/FAR):**
- **Per-hazard PIB accuracy** (independent for tornado / wind / hail):
  - `delta = |player_PIB − observed_PIB|`
  - `score_hazard = max(0, 1 − delta / max_PIB_for_category)` (max 7 for tornado/wind, 6 for hail)
  - Correct prediction of "None" (player picks None AND observed = 0) scores full credit, but no big bonus (it's a small claim).
  - Predicting a PIB when observed = 0 = false alarm (heavier penalty than under-predicting).
  - Predicting None when observed ≥1 = miss (large penalty if observed PIB high).
- **Lead-time component:** for the first verifying report of any hazard, credit = `clamp((report_time − MCD_issue_time) / 60 min, 0, 1)`. Caps at full credit for 60+ min lead. Encourages early MCDs.
- **Coverage breadth bonus:** if 2+ hazards were predicted (non-None) and 2+ verified, small bonus. If all 3 predicted and verified, larger bonus.
- **Anti-spam:** UI rejects MCDs covering >50% of the game polygon. Penalty for trivial small MCDs (<200 km² or fewer than 4 vertices). Per-player rate-limit: max 1 active MCD per 30 minutes of game time.

**MCDs and warnings interact:**
- A warning issued *inside* an active MCD by the same player gets a small "consistency bonus" if both verify.
- MCDs do NOT count against warning FAR.

**Note on IEM data limits:** PIB scoring needs peak wind speed (mph) and peak hail size (inches) per report. IEM LSRs include `magnitude` and `magtype` (G for gust, M for measured, etc.). Tornado EF rating is in the LSR (preliminary, may be revised by SPC later). We use IEM's reported EF and map to PIB via the wind range. Preliminary nature is a known limitation — flagged in docs.

### 9. Leaderboard (live + end-of-round), replay, reveal

**Live in-session leaderboard** (corner widget, always visible during gameplay):
- Compact widget docked in a corner of the main window (player-configurable corner; default top-right).
- Shows each connected player's name + current cumulative score.
- Sorted by score, highest at top. Local player highlighted.
- **Provisional scoring:** updated live as reports come in and verify against active warnings. Tier multipliers (TORR/PDS TOR/TORE) are applied provisionally based on observed reports so far; magnitude accuracy uses the running peak.
- Compact mode (just name + score) and expanded mode (name + score + POD% + FAR% + # warnings) — toggle by click.
- **Update trigger:** the leaderboard refreshes **as storm reports come in** (i.e. when game virtual time crosses a report's timestamp) and when warnings expire — those are the only moments score can change. Score deltas are computed by each client locally from the shared game state, so no extra protocol load.

**End-of-round leaderboard screen** (`ui/leaderboard.py`):
- Sorted by total score.
- Table columns per player:
  - Total score
  - POD, FAR, CSI
  - Mean lead time (positive better)
  - # warnings of each type
  - Magnitude accuracy mean (separate for hail/wind/EF)
  - MCD score component
  - Tier multiplier bonuses applied
- Click a player to expand: per-warning breakdown (which reports verified each, lead time, magnitude error).

**Reveal:**
- Real date, primary location (e.g., "Northern AL, 2011-04-27 Super Outbreak").
- Link to NWS event review if available (hand-maintained dict mapping `date → URL` for famous events; otherwise just SPC's daily storm reports URL).

**Replay file** (if host enabled at round start):
- JSON file written to `~/.radaranalysiscorp/replays/<timestamp>.json`.
- Contains: chosen day, polygon, radar sites, time window, per-player action log (warning issuances, edits, cancels, MCDs), final scores.
- No radar data embedded — replay needs S3 access (same `aws configure`) to refetch volumes.
- Replay playback: single-player mode, no scoring (or "ghost mode" where the user can compare their re-play warnings against the original).

### 12. Live mode (real-time, "now-casting" against current weather)

A separate game mode that plays out in **wall-clock real time** against
live-streaming radar data and storm reports. Instead of replaying a historical
event, the player nowcasts whatever's happening on radar at this very moment.

**Distinguishing characteristics:**

- **Clock is real time.** No host speed control; no `[` / `]` keys. The "Pause"
  button is functionally a "step away" — the game keeps going. (Optionally, allow
  pause for solo play; disable for multiplayer.)
- **No date blinding.** Players know it's today; that's the whole point.
- **Reports stream in with realistic delay.** IEM LSRs typically appear minutes
  to hours after the event, mirroring real forecasting. Players judge with
  incomplete information just like a real shift forecaster.
- **Round duration is host-set** (e.g., "play for 2 hours from now"), but the
  game clock can't be sped past wall-clock time.
- **Polygon + radar-site setup** happens the same as historical mode, but
  reports shown for selection are the last 24 hours of LSRs (so the host has
  some context for picking an area).

**New data sources:**

- **Live Level 2 radar:** IEM mirror at
  `http://mesonet-nexrad.agron.iastate.edu/level2/raw/<RADAR>/` — exposes the
  most recent volumes per radar (typically last 1–2 hours), unsigned HTTP, no
  S3 needed. Poll every ~30 s for new files.
- **Live LSRs:** the same IEM LSR endpoint we already use
  (`mesonet.agron.iastate.edu/cgi-bin/request/gis/lsr.py`) supports a
  ``sts``/``ets`` window that ends at "now"; poll every ~60 s for new reports.
- (Optional v2: NWS warnings feed for ground-truth comparison —
  `api.weather.gov/alerts/active/zone/{zone_id}` so post-round we can show
  "here's what NWS actually issued.")

**New module:** `data/live.py`

```
class LiveDataSource:
    def __init__(self, sites: list[str], polygon: Polygon, poll_sec=30): ...
    def start(self) -> None: ...                    # begins background polling
    def stop(self) -> None: ...
    def latest_volumes(self) -> dict[str, list[ScanRef]]: ...
    def new_reports_since(self, t: datetime) -> list[Report]: ...
    def on_new_volume(callback) -> None:
    def on_new_reports(callback) -> None:
```

Internally: a background thread that scrapes the IEM live directory listing
per radar, downloads new volumes into the same `HashedCache`, and re-fetches
recent LSRs. Emits Qt signals for the UI to consume.

**Clock changes:** `game/clock.py` gets a `LiveClock` subclass that locks
``virtual_time = now()``, ignores speed adjustments, ignores pause for
multiplayer.

**Session changes:** add a `RoundMode` enum (`HISTORICAL` / `LIVE`) on
`RoundConfig`. Live mode skips the prefetch phase entirely — it just starts
listening and the gameplay UI populates as volumes arrive.

**UI changes:**

- Day picker grows a third radio button: "Live (now)" — disables threshold and
  date inputs.
- `OverviewMap` shows the **last 24 h of LSRs** by default in live mode, with
  a "show recent" filter slider (1h / 6h / 24h).
- A "LIVE" badge in the corner indicates real-time play.
- Date-blinding is disabled in live mode (display full datetimes since players
  know it's today).
- New "Recent reports" panel that adds incoming LSRs to a scrolling feed
  during gameplay (since they may not be inside the player's warning polygon).

**Scoring caveats:**

- Lead time is computed normally, but accounting for reporting delay: a
  warning issued at 19:00 verifying a report that *occurred at 19:05* but was
  *reported at 19:32* still earns 5-minute lead time. Use the report's `time`
  field (the event time IEM provides), not when it arrived at our client.
- Round can be ended early by host (no fixed end time; play "until you stop").
- Optional: scale POD/FAR by how mature the report set is at the round-end
  cutoff (more reports may trickle in later, so live scores are provisional).

**Risks / open questions (live mode):**

- IEM live link may go down — need graceful degradation.
- Race condition: a player warns, then a report appears that occurred BEFORE
  the warning. Plan §6 currently disallows late-warn for non-TORR types. In
  live mode the delay is the rule, not the exception, so maybe loosen this to
  allow late-by-report-arrival but not late-by-event-time for all types.
- Storms may stop happening — live mode can be very quiet. Host should be
  warned ("forecast quietness expected" via SPC outlook integration?). v2.

**Purpose:** ease coverage of large game-area polygons or long time windows by letting players coordinate as a single scoring unit. Useful for classroom / training scenarios where a group works the event together, and for casual co-op play.

**Enabling:** host toggles "Team mode" at room creation. Default = OFF (free-for-all).

**Pre-game team lobby (only when team mode = ON):**
- After all players have joined the room but before the host clicks "Start round," a **team lobby** screen appears.
- Players can:
  - **Create a new team** with a chosen name → join it as the first member.
  - **Join an existing team** from a list of teams in the room.
  - **Leave their current team** (returns to "Unassigned" pool).
- An "Unassigned" pool catches players who haven't picked a team.
- Host sees the same screen plus an admin panel — can:
  - Move any player between teams (drag-and-drop, or right-click → "Move to…").
  - Auto-assign unassigned players (round-robin across existing teams, or random).
  - Rename teams, delete empty teams.
- Host clicks "Start round" — at that moment, the team roster freezes:
  - Unassigned players become solo teams of 1 (auto-named with their handle).
  - No team changes during gameplay (avoids mid-round scoring chaos).
- Mid-round joiners (§4) join as a solo team of 1.

**Scoring in team mode:**
- Each team is a single scoring entity. Its warning set = union of all teammates' warnings.
- POD/FAR computed over the team's warnings against all reports inside the game polygon.
- Magnitude accuracy / tier bonuses computed per warning as before, attributed to the team.
- Lead time per verified report uses the earliest teammate warning whose polygon contained the report at the report's time.
- **Conflict / overlap rule:** if two teammates issue warnings over the same area, they don't double-count. The team gets credit once per report. Use the union of polygons for verification.
- **Conflict / contradiction:** if two teammates issue warnings of conflicting type (one SVR, one TOR) covering the same report, the team gets credit for the *higher* tier that verifies (TOR wins over SVR if a tornado occurs; SVR wins if only hail/wind). This rewards bold calls but doesn't penalize the cautious teammate.
- MCDs from any teammate are credited to the team. Multiple MCDs per team are allowed (no per-player rate limit, but the per-30-min limit still applies *per team* to prevent spam).

**UI changes in team mode:**
- **Per-player color scheme is replaced by per-team color scheme** on the host central map (§4c) and in the live leaderboard (§9). Within a team, individual contributors are distinguished by line *style* variations or small initials labels.
- Teammates' warnings appear **fully visible** (same opacity as your own) on every teammate's gameplay window — they're "yours" too.
- Other teams' warnings appear faintly (current behavior for other players).
- Live leaderboard widget shows per-team scores, with an expand-on-click view showing per-teammate contributions.
- End-of-round leaderboard shows team totals; click a team to see per-member breakdown.

**Network protocol additions** (`net/protocol.py`):
- `TeamCreate(team_id, name)` — peer → host → all.
- `TeamJoin(team_id, player_id)` — peer → host → all.
- `TeamLeave(player_id)` — peer → host → all.
- `TeamRosterFreeze(roster)` — host → all, broadcast at start of round.
- Existing `WarningIssue / WarningUpdate / WarningCancel` already carry the issuing player_id; the team aggregation happens server-side in scoring, not in the wire format.

**Solo play:** team mode OFF reduces to current behavior. Team mode ON with one player per team also reduces to current behavior. Both paths exercise the same scoring code with team_size=1.

### 10. Networking architecture

**Strict separation of two data planes:**

1. **Radar data plane** (per-client, direct from AWS S3):
   - Each client uses its own boto3 signed client against `noaa-nexrad-level2`.
   - On round setup, every client receives `{day, polygon, radar_sites, time_window}` from the host. Using these, it independently lists S3 keys and downloads the needed Level 2 files.
   - Radar data **never** traverses P2P. The host has no special role in fetching/serving radar.
   - `data/prefetch.py` runs a background worker:
     - **Pre-game**: downloads all scans for the first 30 minutes of the time window in parallel (bounded thread pool, ~8 concurrent fetches) before the game clock starts. UI shows per-radar progress bars and an overall room-wide "X / N players ready" status.
     - **Pre-game start gate**: host monitors download completion via heartbeat messages from each peer. When **≥75% of connected clients** signal "ready," host starts a **60-second countdown** broadcast to all peers. When timer hits zero, round starts regardless of who hasn't finished. Late-finishers behave like mid-round joiners (§4 mid-round join) — they begin playing as soon as their first scan loads.
     - **In-game**: maintains a rolling ~20-minute lookahead buffer ahead of the game clock per active radar. New downloads stream in as the clock advances. If a client's buffer empties (S3 stall mid-game), that client's radar panel stays on the latest cached scan with a "buffering — waiting for radar data" overlay; gameplay continues for everyone else. The slow client cannot scrub past their buffer's edge. No room-wide pause.
   - Cache: `~/.radaranalysiscorp/cache/<sha1>.ar2v` with hash-of-key filenames (UI date-blinding; not a security measure — see §4b).
   - On round end, cache is **not** purged (next round may reuse if same day).

2. **Gameplay state plane** (P2P over WebRTC DataChannel, star topology):
   - **Star topology**, not full mesh. Each peer maintains one WebRTC connection to the host. Host relays all inter-peer messages.
   - Rationale: full mesh at 50 players = 1,225 pairwise connections, with NAT-traversal failures multiplying. Star = 49 connections, much more tractable. Game messages are tiny (kB/s, not MB/s), so host bandwidth cost is acceptable.
   - **Signaling**: small `aiohttp` WebSocket server we host. Workflow:
     1. Host creates room → server issues short code (e.g., `STORM-FROG-72`).
     2. Peers enter code → server forwards their SDP offer to the host.
     3. Host returns SDP answer via server → peers connect P2P directly to host.
     4. Server's role ends once the DataChannel is up (it only re-engages for new joiners / reconnects).
   - **Message types** (defined in `net/protocol.py`):
     - `RoundSetup` (day, polygon, radar_sites, time_window, replay_enabled) — sent once at setup, plus to late joiners.
     - `Tick` (virtual_time_offset, speed_multiplier) — host → all, ~1/s during play.
     - `PlayerState` (join, leave, ready) — bidirectional via host.
     - `WarningIssue / WarningUpdate / WarningCancel` (polygon, type, duration, magnitudes) — peer → host → all.
     - `MCDIssue / MCDCancel` (polygon, duration, PIBs) — peer → host → all.
     - `Chat` (text) — peer → host → all.
     - `ScoreSnapshot` — clients compute scores locally; host periodically sends a snapshot only for disagreement detection (rare).
   - **Date-blinding in protocol:** all `Tick` and timestamp fields are sent as offsets from a session epoch, not absolute dates. The day itself IS sent in `RoundSetup` (clients need it for S3) but never displayed in UI — see §4b honor-system caveat.
   - **Host disconnect (v1):** if the host's connection drops, the round **ends immediately** and the leaderboard shows partial scores. No host migration in v1 — this keeps the protocol simple. v2 could add migration if needed.
   - **Late joiner sync:** signaling server hands joiner the current `RoundSetup` + active warnings/MCDs + leaderboard snapshot from the host. Joiner starts S3 prefetch immediately and begins ticking once first scan loads.

**Bandwidth estimate** (50-player room, host upload):
- Tick: ~50 B × 50 peers × 1 Hz = 2.5 kB/s
- Warning issuance: ~500 B (polygon vertices + metadata) × 50 peers per issue, ~1 every few minutes = negligible avg
- Chat: variable but tiny
- Total host upload: well under 50 kB/s at maximum activity. Acceptable on any home broadband.

---

## Files to Reuse from Nowcastle

- [Reference-Nowcastle/radar_game/data.py](Reference-Nowcastle/radar_game/data.py) — LSR fetching from IEM; S3 download with atomic rename; scan-time lookup. **Adapt** S3 client to signed boto3 against `noaa-nexrad-level2` (Nowcastle uses unsigned Unidata mirror).
- [Reference-Nowcastle/radar_game/radar_plot.py](Reference-Nowcastle/radar_game/radar_plot.py) — 6-panel REF/VEL/RHO/ZDR/KDP/HCA display, `InteractiveNav` scroll/pan/zoom, tilt stepping, KDP/HCA derivation.
- [Reference-Nowcastle/radar_game/sites.py](Reference-Nowcastle/radar_game/sites.py) — `Site` dataclass, `latlon_to_xy_km()`, nearest-site lookup. Largely reusable.
- [Reference-Nowcastle/RADARS.txt](Reference-Nowcastle/RADARS.txt) — WSR-88D catalog.

**Not applicable from Nowcastle:**
- `game.py` round logic (single-circle severity guess) — fundamentally different mechanic.
- `stats.py` (MAE/bias on guesses) — replaced by POD/FAR/lead-time scoring.

---

## Remaining Open Questions (minor / tunable parameters only)

1. **Replay ghost mode:** v1 just plays back what happened, or v1.x allows the user to re-issue warnings during replay and compare against original?
2. **Reveal database:** hand-curated `date → event_name + URL` for famous outbreaks (Super Outbreak, El Reno, etc.)? Or just always link to SPC daily reports?
3. **MCD constraints:** max 50% of game polygon, min 200 km² + 4 vertices, max 1 active per 30 game-minutes — proposed defaults; confirm or adjust.
4. **Hosted signaling server:** where to deploy (small VPS, Fly.io, Cloudflare Workers)? Doesn't affect game code but needs to exist before multiplayer works.

---

## Verification (how to test the finished game end-to-end)

- **Manual single-player** on known events (e.g., 2013-05-20 Moore, 2011-04-27 Super Outbreak, 2013-05-31 El Reno).
- **Local LAN multiplayer**: two processes on one machine, then two machines on LAN, then two over the internet via the signaling server.
- **Unit tests** for:
  - SAILS-aware sweep index (synthetic Level 2 with mixed VCP, verify 0.5° step order)
  - 5 km buffered point-in-polygon (synthetic polygon + edge-case points)
  - POD/FAR/lead-time computation against synthetic reports & warnings
  - Tier multiplier logic (TORR/PDS TOR/TORE × verified-by-EF2 / weak-tornado / FA)
  - MCD scoring (lead-coverage, multi-hazard bonus, anti-spam cap)
  - LSR remarks injury/fatality regex parsing on real LSR samples
  - Date-blinding sweep: scrape rendered UI strings, fail on any `YYYY-MM-DD`
- **Replay round-trip**: record a session, replay it, confirm deterministic scoring (modulo S3 fetch).
- **Network**: signaling server load test (room create / join / SDP exchange), DataChannel reconnect after brief disconnect, mid-round join state sync.
