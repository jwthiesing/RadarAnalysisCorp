"""In-game composite widget: radar panels + host central map + clock + leaderboard.

The player's main view during a round. Composes:

  - :class:`ClockControls` (top, host only — peers see read-only time)
  - :class:`RadarPanelGrid` (center-left) — main forecasting display
  - :class:`HostCentralMap` (right) — host overview, also has Join-as-player button
  - :class:`LiveLeaderboardWidget` (corner of host_map)

Wires keyboard shortcuts to the actions:
  - ``N`` → New warning (polygon draw mode on radar panel → form)
  - ``M`` → Activate motion tool
  - ``C`` → New MCD (polygon draw + PIB form)

For solo play, the host map and the radar panel both render the same session.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import timedelta

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..game.session import RoundMode

from ..data.prefetch import Prefetcher
from ..game.replay import ReplayWriter
from ..game.session import GameSession
from ..geo.polygons import Polygon
from ..net.multiplayer import MultiplayerHost, MultiplayerPeer
from .controls import ClockControls
from .host_map import HostCentralMap
from .mcd_form import MCDFormDialog
from .motion_tool import MotionTool
from .poly_editor import PolygonEditor
from .radar_panel import RadarPanelGrid, _TOOL_BUTTON_QSS
from .warning_form import WarningFormDialog

log = logging.getLogger(__name__)


class PlayView(QWidget):
    """In-game composite widget."""

    round_ended = pyqtSignal()
    # Optional path of the saved replay file; emitted on round end after the
    # leaderboard signal.
    replay_saved = pyqtSignal(object)   # str | None

    def __init__(
        self,
        session: GameSession,
        prefetcher: Prefetcher,
        local_player_id: str,
        *,
        multiplayer: MultiplayerHost | MultiplayerPeer | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.prefetcher = prefetcher
        self.local_player_id = local_player_id
        self.multiplayer = multiplayer
        self._is_host = isinstance(multiplayer, MultiplayerHost) or multiplayer is None
        # Replay logging: host- and solo-only. Peers in a multiplayer round
        # would otherwise each write a duplicate file. The host owns the
        # canonical record of the round.
        self._replay: ReplayWriter | None = None
        if (
            session.round_config is not None
            and session.round_config.save_replay
            and not isinstance(multiplayer, MultiplayerPeer)
        ):
            try:
                self._replay = ReplayWriter()
            except Exception as e:  # noqa: BLE001
                log.warning("Could not open replay writer: %s", e)

        # Default to first radar site
        sites = session.round_config.radar_sites if session.round_config else []
        if not sites:
            raise RuntimeError("PlayView needs at least one radar site")
        initial_site = sites[0]
        # TDWRs are single-pol C-band — Unidata's Level 2 files only carry
        # REF / VEL / SW (no CC, ZDR, PHI). Default-laying-out four
        # panels with two of them dual-pol would leave the host staring
        # at two permanently-blank panels every render, which reads as
        # "the panels aren't rendering" even though REF/VEL are fine.
        # Switch the initial layout to a 2-panel REF/VEL grid when the
        # initial radar is a TDWR; users can still bump back up to 4
        # via Alt+4 (panels just won't have dual-pol data to show).
        from ..data.sites import site_by_icao
        initial_site_obj = site_by_icao(initial_site)
        if initial_site_obj is not None and initial_site_obj.is_tdwr:
            initial_n_panels = 2
            initial_layout = ("REF", "VEL")
        else:
            initial_n_panels = 4
            initial_layout = None  # default REF/VEL/CC/ZDR
        # Construct without the game-clock cap so the initial display can show
        # the first available sweep even if it's timestamped slightly after the
        # round's nominal start time (NEXRAD scans align to their own schedule,
        # not the round window). The cap is enforced from the first tick onward
        # via _on_tick → set_max_virtual_time.
        self.radar_grid = RadarPanelGrid(
            sweep_index=prefetcher.sweep_index(initial_site),
            site_icao=initial_site,
            n_panels=initial_n_panels,
            layout=initial_layout,
            max_virtual_time=None,
            # Surface the full list of radars the host enabled so the
            # grid's toolbar can show a switcher dropdown. Without
            # this, the grid is permanently locked to ``initial_site``
            # — even though the prefetcher has been faithfully
            # downloading every site's volumes in the background.
            available_sites=list(sites),
        )
        # Hand the prefetcher to the grid so its volume-load LRU stays
        # warm — PyART parse + region-based velocity dealias runs on
        # the prefetcher's worker pool the moment a download finishes,
        # which eliminates the 200-3000 ms stall that otherwise hits the
        # main thread the first time a scrub steps into a new volume.
        self.radar_grid.attach_prefetcher_preload(prefetcher)
        # Click an own warning/MCD polygon on any radar panel → opens
        # the revise dialog. Wired here (not in the grid) so the
        # multiplayer-or-solo routing of the resulting revision stays
        # localized to PlayView.
        self.radar_grid.warning_clicked.connect(self._on_warning_clicked)
        self.radar_grid.mcd_clicked.connect(self._on_mcd_clicked)
        # Push the game verification polygon onto the radar grid so it's drawn
        # on every panel — players need to see the boundary their warnings
        # must fall within (plan §4a).
        if session.round_config is not None:
            self.radar_grid.set_game_polygon(session.round_config.game_polygon)
        if session.clock:
            si = self.radar_grid.sweep_index
            initial = si.latest_at_or_before(session.clock.virtual_time, elev_deg=0.5)
            if initial is None:
                low = sorted(si.at_elevation(0.5), key=lambda s: s.start_time)
                initial = low[0] if low else None
            if initial is not None:
                self.radar_grid.show_sweep(initial)

        # Host central map — only created when the local user is the host
        # of a multiplayer round. It opens as a **separate top-level
        # window** so the host's play window stays focused on their own
        # gameplay (radar + their own warnings) instead of seeing every
        # other player's polygons in real time. Solo and peer clients
        # never see a host map.
        self._is_solo = multiplayer is None
        self._is_host_player = isinstance(multiplayer, MultiplayerHost)
        self.host_map: HostCentralMap | None = None
        if self._is_host_player:
            self.host_map = HostCentralMap(session)
            self.host_map.setWindowTitle("Host overview — all players")
            self.host_map.resize(900, 700)
            self.host_map.show()

        # Clock controls (host only — single-player IS the host).
        # On peer clients we still need _on_tick to fire so the map/leaderboard
        # update; we use a local timer (1 Hz in live mode where wall-clock
        # drives everything locally; in historical mode the network tick from
        # the host arrives anyway and triggers our handler indirectly via
        # MultiplayerPeer.apply_tick).
        self.clock_controls = ClockControls(session.clock)
        self.clock_controls.tick.connect(self._on_tick)
        self.clock_controls.request_end_round.connect(self._on_end_round)
        # Peer-side local timer: drives _on_tick locally so the map refreshes
        # even when the network tick is a no-op (live mode) or arrives at
        # 1 Hz cadence (historical mode — we tick UI faster than network).
        self._peer_timer: QTimer | None = None
        if isinstance(multiplayer, MultiplayerPeer):
            self._peer_timer = QTimer(self)
            self._peer_timer.setInterval(1000)   # 1 Hz peer-local tick
            self._peer_timer.timeout.connect(self._peer_local_tick)
            self._peer_timer.start()

        # Motion tool (lazy-attached). Tracks persist across activate/
        # deactivate cycles; right-click removes a single track at any time.
        self.motion_tool = MotionTool(self.radar_grid)
        # When the panel layout changes (Alt+1/2/4), the old canvases die
        # and any storm-track artists go with them. Tell the tool to drop
        # stale state and re-attach to the new canvases.
        self.radar_grid.panels_rebuilt.connect(
            self.motion_tool.reinstall_handlers_for_new_panels
        )

        # Layout: clock at top; action bar below it; radar grid fills the
        # rest. The action bar gives every keybind a discoverable button
        # equivalent — players don't have to memorize the key chart.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.addWidget(self.clock_controls)
        self._action_bar = self._build_action_bar()
        layout.addWidget(self._action_bar)
        # The play window only shows the player's own radar — the host's
        # all-players overview lives in its own top-level window
        # (created above when applicable).
        layout.addWidget(self.radar_grid, stretch=1)

        # Keyboard shortcuts — registered at the PlayView level so they fire
        # regardless of which child widget (radar grid, clock bar, host map)
        # currently holds Qt focus. Per-widget keyPressEvent handlers only
        # fire when that widget owns focus, which is unreliable in practice.
        # Keep refs on ``self`` so Python's GC can't collect the
        # QShortcut objects while their parent QWidget is alive. PyQt6
        # used to keep the connection alive via the parent argument
        # alone, but observed cases of shortcuts going dead led us to
        # keep explicit references.
        self._shortcuts: list[QShortcut] = []
        def _sc(seq, slot):
            s = QShortcut(QKeySequence(seq), self, activated=slot)
            self._shortcuts.append(s)
            return s
        _sc("N", self._begin_warning_polygon)
        _sc("C", self._begin_mcd_polygon)
        _sc("M", self._toggle_motion_tool)
        # Polygon-draw finalize/cancel — registered ONCE here at
        # PlayView level, not per-draw. (The old code re-created the
        # Return shortcut on every ``_start_polygon_draw`` call which
        # piled up dangling QShortcut objects and had no Escape
        # equivalent at all.) The handlers no-op when no polygon is
        # being drawn, so they're safe to fire from any state.
        _sc(Qt.Key.Key_Return, self._finish_polygon)
        _sc(Qt.Key.Key_Enter, self._finish_polygon)    # numeric keypad
        _sc(Qt.Key.Key_Escape, self._cancel_polygon_draw)
        # Clock controls
        QShortcut(QKeySequence(Qt.Key.Key_Space), self,
                   activated=self.clock_controls._toggle_play)
        QShortcut(QKeySequence(Qt.Key.Key_BracketLeft), self,
                   activated=self.clock_controls._slower)
        QShortcut(QKeySequence(Qt.Key.Key_BracketRight), self,
                   activated=self.clock_controls._faster)
        # Radar scrub: arrow keys + Shift modifiers
        QShortcut(QKeySequence(Qt.Key.Key_Left), self,
                   activated=lambda: self.radar_grid.step_time(-1))
        QShortcut(QKeySequence(Qt.Key.Key_Right), self,
                   activated=lambda: self.radar_grid.step_time(+1))
        QShortcut(QKeySequence("Shift+Left"), self,
                   activated=lambda: self.radar_grid.step_time(-5))
        QShortcut(QKeySequence("Shift+Right"), self,
                   activated=lambda: self.radar_grid.step_time(+5))
        QShortcut(QKeySequence(Qt.Key.Key_Up), self,
                   activated=lambda: self.radar_grid.step_elevation(+1))
        QShortcut(QKeySequence(Qt.Key.Key_Down), self,
                   activated=lambda: self.radar_grid.step_elevation(-1))
        # Panel-count selectors. Alt+digit because plain digits cycle products
        # on the focused panel.
        QShortcut(QKeySequence("Alt+1"), self,
                   activated=lambda: self.radar_grid.set_layout(1))
        QShortcut(QKeySequence("Alt+2"), self,
                   activated=lambda: self.radar_grid.set_layout(2))
        QShortcut(QKeySequence("Alt+4"), self,
                   activated=lambda: self.radar_grid.set_layout(4))
        # Product cycling on the focused panel — same key-to-product map as
        # PRODUCTS dict ordering: 1=REF, 2=VEL, 3=SW, 4=CC, 5=ZDR, 6=PHI.
        # Click a panel to focus it before pressing the digit.
        # (KDP used to be 6 but was dropped — its retrieval is too slow.)
        for i, key in enumerate(("1", "2", "3", "4", "5", "6")):
            QShortcut(QKeySequence(key), self,
                       activated=lambda idx=i: self.radar_grid._cycle_focused_product(idx))
        # Keyboard zoom (center-of-view). Scroll-wheel zoom still works but
        # these give precise stepwise zoom that doesn't depend on cursor position.
        QShortcut(QKeySequence("="), self,
                   activated=lambda: self.radar_grid.zoom(0.8))  # zoom in
        QShortcut(QKeySequence("-"), self,
                   activated=lambda: self.radar_grid.zoom(1.25))  # zoom out
        # Plus key without shift on some layouts; treat it the same as =
        QShortcut(QKeySequence("+"), self,
                   activated=lambda: self.radar_grid.zoom(0.8))
        # Data inspector toggle — mousing over the radar reveals the
        # displayed product's value at the cursor location while on.
        QShortcut(QKeySequence("I"), self,
                   activated=lambda: self._toggle_inspector())
        # WASD pan — each step shifts the view by 20% of its current size.
        # Mirrors the mouse-drag pan but is keyboard-only so the player can
        # nudge the view without leaving the focused panel.
        PAN_STEP = 0.2
        QShortcut(QKeySequence("W"), self,
                   activated=lambda: self.radar_grid.pan(0.0, +PAN_STEP))
        QShortcut(QKeySequence("S"), self,
                   activated=lambda: self.radar_grid.pan(0.0, -PAN_STEP))
        QShortcut(QKeySequence("A"), self,
                   activated=lambda: self.radar_grid.pan(-PAN_STEP, 0.0))
        QShortcut(QKeySequence("D"), self,
                   activated=lambda: self.radar_grid.pan(+PAN_STEP, 0.0))

        # In-flight polygon editor (set during a draw)
        self._active_poly_editor: PolygonEditor | None = None
        self._pending_action: str | None = None     # 'warning' or 'mcd'
        self._original_cursor = None
        # Action bar starts in idle mode (Finish/Cancel hidden until draw starts)
        self._update_action_bar_mode(drawing=False)

    # ---- action bar ----------------------------------------------------

    def _build_action_bar(self) -> QFrame:
        """Buttons for every PlayView-level keybind: N/C/M/Esc/Enter.

        Each button's label includes its keybind. Buttons are focus-skipping
        (NoFocus) so clicking them never traps keyboard input — the global
        QShortcuts continue to work afterward.

        The tool buttons (New Warning / New MCD / Motion) are checkable
        so the :checked stylesheet kicks in while the corresponding tool
        is active — a vivid yellow highlight that survives the disabled
        state (Qt's default :disabled would otherwise mute it back to
        invisibility).
        """
        bar = QFrame(self)
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        h = QHBoxLayout(bar)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(6)

        def _btn(label, tip, slot, *, danger=False, checkable=False):
            b = QToolButton(bar)
            b.setText(label)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            b.setCheckable(checkable)
            if danger:
                # Danger buttons (Cancel Draw) override the base style
                # with a red tint instead of the toolbar look.
                b.setStyleSheet("color: #ff8888;")
            else:
                b.setStyleSheet(_TOOL_BUTTON_QSS)
            # autoExclusive=False so the three tool buttons can't
            # silently uncheck each other if Qt's radio-group default
            # ever kicked in.
            b.setAutoExclusive(False)
            b.clicked.connect(slot)
            return b

        self._btn_new_warning = _btn(
            "▲ New Warning  (N)", "Begin a freehand warning polygon",
            self._begin_warning_polygon, checkable=True,
        )
        self._btn_new_mcd = _btn(
            "◇ New MCD  (C)", "Begin a freehand Mesoscale Convective Discussion",
            self._begin_mcd_polygon, checkable=True,
        )
        self._btn_motion = _btn(
            "↗ Motion Tool  (M)", "Two-click storm-motion measurement",
            self._toggle_motion_tool, checkable=True,
        )
        # Dynamic draw-mode buttons — hidden when no draw is in flight.
        self._btn_finish = _btn(
            "✓ Finish Polygon  (Enter)",
            "Close and submit the in-flight polygon",
            self._finish_polygon,
        )
        self._btn_cancel = _btn(
            "✗ Cancel Draw  (Esc)",
            "Discard the in-flight polygon",
            self._cancel_polygon_draw,
            danger=True,
        )

        for b in (self._btn_new_warning, self._btn_new_mcd, self._btn_motion,
                  self._btn_finish, self._btn_cancel):
            h.addWidget(b)
        h.addStretch(1)
        self._draw_hint = QLabel("", bar)
        self._draw_hint.setStyleSheet("color: #ffd400; font-weight: bold;")
        h.addWidget(self._draw_hint)
        return bar

    def _update_action_bar_mode(self, *, drawing: bool) -> None:
        """Toggle visibility + checked-state of the action-bar buttons.

        While a draw is in flight, the corresponding tool button (warning
        or MCD) stays *checked* (yellow highlight) AND disabled — so the
        player can see at a glance which tool produced the in-flight
        polygon, but can't accidentally start a second one. The other
        tool buttons go to unchecked + disabled. The Finish/Cancel
        buttons appear only while drawing."""
        if not hasattr(self, "_btn_finish"):
            return
        self._btn_finish.setVisible(drawing)
        self._btn_cancel.setVisible(drawing)
        self._btn_new_warning.setEnabled(not drawing)
        self._btn_new_mcd.setEnabled(not drawing)
        self._btn_motion.setEnabled(not drawing)
        # Reflect the active tool with the :checked highlight. block-
        # Signals because we set state programmatically and don't want
        # the click handler to fire.
        for b, active in (
            (self._btn_new_warning, drawing and self._pending_action == "warning"),
            (self._btn_new_mcd,     drawing and self._pending_action == "mcd"),
        ):
            b.blockSignals(True)
            b.setChecked(active)
            b.blockSignals(False)
        if drawing and self._pending_action:
            kind = "warning" if self._pending_action == "warning" else "MCD"
            self._draw_hint.setText(
                f"Drawing {kind} polygon — click to add vertices"
            )
        else:
            self._draw_hint.setText("")

    # ---- tick handling -------------------------------------------------

    def _on_tick(self, tick) -> None:
        # Update game-clock cap on the radar panel so scrubbing is bounded.
        self.radar_grid.set_max_virtual_time(tick.virtual_time)
        # Tell prefetcher to advance its lookahead buffer. This now
        # returns immediately — the actual S3 LIST + dispatch happens
        # on the prefetcher's dedicated tick worker, so the main
        # thread is never blocked on network.
        self.prefetcher.advance_clock(tick.virtual_time)
        # Push reports up to virtual_time onto the radar panel for
        # live overlay. Route through the setter so its render-key
        # diff can short-circuit the re-render when the visible
        # report set is unchanged (the common case — most ticks
        # don't cross a new report's timestamp).
        if self.session.round_day is not None:
            visible = [
                r for r in self.session.round_day.reports
                if r.time <= tick.virtual_time
            ]
            self.radar_grid.set_live_reports(visible)
        # Player's own warnings/MCDs overlaid on each radar panel — drawn for
        # whichever revision is active at the panel's display time, so
        # scrubbing back shows the polygon as it was then. The setter
        # diffs against the previously-rendered set, so a tick with no
        # warning changes (the common case) doesn't trigger a re-render.
        self._push_player_overlays()
        # Refresh the host map (so reports fade + leaderboard updates).
        # In solo there's no host map; the radar panel itself shows reports
        # via the live-reports overlay we just updated.
        if self.host_map is not None:
            self.host_map.refresh()
        # Broadcast the tick to peers (host only)
        if isinstance(self.multiplayer, MultiplayerHost):
            asyncio.ensure_future(self.multiplayer.broadcast_tick(tick))

    def _peer_local_tick(self) -> None:
        """Peer-side local 1 Hz tick: drives _on_tick so map/leaderboard
        refresh even when the network's tick handler is a no-op (live mode)."""
        if self.session.clock is None:
            return
        # In live mode, call advance() so wall-clock virtual_time updates;
        # in historical mode the network tick from the host is authoritative
        # (apply_tick already wrote it) so we just need to fire _on_tick to
        # rerender. In both cases passing the current snapshot is correct.
        if (self.session.round_config is not None
                and self.session.round_config.mode == RoundMode.LIVE):
            self.session.clock.advance()
        self._on_tick(self.session.clock.snapshot())

    def _on_end_round(self) -> None:
        scores = self.session.end_round()
        replay_path: str | None = None
        if self._replay is not None:
            try:
                self._replay.log_final_scores(scores)
                self._replay.close()
                replay_path = str(self._replay.path)
                log.info("Replay file saved: %s", replay_path)
            except Exception as e:  # noqa: BLE001
                log.warning("Failed to finalize replay file: %s", e)
        self.replay_saved.emit(replay_path)
        self.round_ended.emit()

    # ---- warning / mcd issuance ----------------------------------------

    def _begin_warning_polygon(self) -> None:
        if self._active_poly_editor is not None:
            return
        self._start_polygon_draw(action="warning", color="#ffd400")

    def _begin_mcd_polygon(self) -> None:
        if self._active_poly_editor is not None:
            return
        self._start_polygon_draw(action="mcd", color="#cc88ff")

    def _start_polygon_draw(self, *, action: str, color: str) -> None:
        """Shared entry point for warning / MCD polygon draw modes.

        Adds a visible affordance: crosshair cursor + status-bar message.
        Esc / right-click cancels; double-click or Enter finishes.

        Hands ALL panels' ViewBoxes to the editor so a vertex click on
        any panel registers and the in-flight polygon mirrors across
        every panel. Without this, only the focused panel responded
        and the other three sat ignoring clicks — confusing because
        they all show the same view.
        """
        if not self.radar_grid._panels:
            return
        # If the storm-motion tool is currently listening, deactivate
        # it before we start drawing. Both tools subscribe to the
        # panel scenes' ``sigMouseClicked``; if motion-tool stays
        # active, every polygon-vertex click ALSO drops a motion-tool
        # P2 (or sets a stray P1 if it had none), which manifests as
        # "my warning placement is putting a point at where my last
        # motion-tool click was." Cleaner UX: starting a polygon
        # cancels any in-flight motion measurement.
        if self.motion_tool.is_active:
            self._toggle_motion_tool()
        site = self.radar_grid.site
        from ..geo.projection import xy_km_to_latlon
        views = [p.view for p in self.radar_grid._panels]
        self._active_poly_editor = PolygonEditor(
            views,
            axes_to_latlon=lambda x, y: xy_km_to_latlon(x, y, site.lat, site.lon),
            color=color,
        )
        self._pending_action = action
        # Cursor change — local to the radar panels' canvases
        self._original_cursor = self.radar_grid.cursor()
        self.radar_grid.setCursor(Qt.CursorShape.CrossCursor)
        # Status bar message via the parent window
        parent = self.window()
        if hasattr(parent, "statusBar"):
            kind = "warning" if action == "warning" else "MCD"
            parent.statusBar().showMessage(
                f"Drawing {kind} polygon — click to add vertices, "
                f"Enter to finish, Esc to cancel"
            )
        # Return / Enter / Escape are registered once at PlayView
        # construction (see ``__init__``); their handlers gate on
        # ``_active_poly_editor`` so they only act while drawing.
        self._update_action_bar_mode(drawing=True)

    def _cancel_polygon_draw(self) -> None:
        """Esc handler — clear in-flight polygon and reset the cursor / status."""
        if self._active_poly_editor is None:
            return
        # ``dispose`` removes the outline + marker artists from every
        # panel's scene (vs ``clear`` which only empties their data).
        # Necessary so a re-render of the panel can't paint stale
        # vertices back in from a leftover Qt-internal cache.
        self._active_poly_editor.dispose()
        self._active_poly_editor = None
        self._pending_action = None
        self._restore_default_cursor()
        self._update_action_bar_mode(drawing=False)
        parent = self.window()
        if hasattr(parent, "statusBar"):
            parent.statusBar().showMessage("Draw canceled", 2000)

    def _restore_default_cursor(self) -> None:
        if self._original_cursor is not None:
            self.radar_grid.setCursor(self._original_cursor)
            self._original_cursor = None
        else:
            self.radar_grid.unsetCursor()

    def _finish_polygon(self) -> None:
        if self._active_poly_editor is None:
            return
        polygon = self._active_poly_editor.polygon()
        action = self._pending_action
        editor = self._active_poly_editor
        self._active_poly_editor = None
        self._pending_action = None
        self._restore_default_cursor()
        self._update_action_bar_mode(drawing=False)
        if polygon is None:
            log.info("Polygon has < 3 vertices; canceled")
            editor.clear()
            return
        if action == "warning":
            dlg = WarningFormDialog(parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                params = dlg.get_parameters()
                if self.multiplayer is not None:
                    asyncio.ensure_future(self.multiplayer.issue_warning(
                        player_id=self.local_player_id, polygon=polygon, **params,
                    ))
                    # Best-effort local replay log: synthesize a placeholder
                    # warning so the writer has something to log. The
                    # session-side warning will be created when the broadcast
                    # echoes back from the host.
                else:
                    w = self.session.issue_warning(
                        player_id=self.local_player_id, polygon=polygon, **params,
                    )
                    if self._replay is not None and self.session.clock:
                        self._replay.log_warning_issue(w, virtual_time=self.session.clock.virtual_time)
                self._push_player_overlays()
                if self.host_map is not None:
                    self.host_map.refresh()
        elif action == "mcd":
            dlg = MCDFormDialog(parent=self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                params = dlg.get_parameters()
                if self.multiplayer is not None:
                    asyncio.ensure_future(self.multiplayer.issue_mcd(
                        player_id=self.local_player_id, polygon=polygon, **params,
                    ))
                else:
                    m = self.session.issue_mcd(
                        player_id=self.local_player_id, polygon=polygon, **params,
                    )
                    if self._replay is not None and self.session.clock:
                        self._replay.log_mcd_issue(m, virtual_time=self.session.clock.virtual_time)
                self._push_player_overlays()
                if self.host_map is not None:
                    self.host_map.refresh()
        # Permanently remove the in-progress editor's vertex markers +
        # outline from every panel's scene. ``clear()`` only empties
        # their data; ``dispose()`` detaches them outright so leftover
        # Qt items can't flash content back in next time the user
        # starts another draw.
        editor.dispose()

    def _on_warning_clicked(self, warning_id: str) -> None:
        """Open the revise dialog for an own warning the user clicked.

        The dialog opens prefilled from the warning's current revision
        (type, duration, magnitudes — see ``WarningFormDialog.__init__``'s
        ``existing=`` branch). On Accept we call ``revise_warning(...)``
        via the multiplayer wrapper if we're online, or the session
        directly if we're solo — both append a new revision at the
        current clock time, broadcast (in MP), and re-render via
        ``_push_player_overlays``."""
        # Find the warning in our own bucket. Ignore clicks on other
        # players' warnings (which currently aren't even drawn on the
        # radar panel but defensive coding for future relax of that).
        my_warnings = self.session.warnings_by_player.get(
            self.local_player_id, [],
        )
        target = next(
            (w for w in my_warnings if w.warning_id == warning_id), None,
        )
        if target is None:
            log.debug("warning click on %s — not in our bucket; ignoring",
                      warning_id)
            return
        # Don't bother opening the dialog for warnings that have already
        # expired or been canceled — they're frozen and revising them
        # would just confuse the scoring.
        ref = (self.session.clock.virtual_time
               if self.session.clock is not None
               else target.original_issue_time)
        if target.canceled_at is not None and ref > target.canceled_at:
            log.info("warning %s canceled — not opening revise dialog", warning_id)
            return
        if ref > target.end_time():
            log.info("warning %s expired — not opening revise dialog", warning_id)
            return
        # Pass the current game-clock time so the dialog can prefill
        # the duration field with "minutes remaining until current
        # expiry" — a no-change revise will then preserve the
        # original expiry instead of inadvertently extending it by
        # the time elapsed since issuance.
        now = self.session.clock.virtual_time if self.session.clock else None
        dlg = WarningFormDialog(existing=target, now=now, parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        # The user can pick between two destructive options from the
        # dialog: Cancel-this-warning (which permanently terminates
        # the warning at the current clock time) or Revise (which
        # appends a new revision). The dialog flags the former via
        # ``cancel_requested()``.
        if dlg.cancel_requested():
            if self.multiplayer is not None:
                asyncio.ensure_future(self.multiplayer.cancel_warning(
                    warning_id=warning_id, player_id=self.local_player_id,
                ))
            else:
                self.session.cancel_warning(
                    warning_id=warning_id, player_id=self.local_player_id,
                )
                if self._replay is not None and self.session.clock:
                    self._replay.log_warning_cancel(
                        warning_id, self.local_player_id,
                        virtual_time=self.session.clock.virtual_time,
                    )
            self._push_player_overlays()
            if self.host_map is not None:
                self.host_map.refresh()
            return

        params = dlg.get_parameters()
        # ``params`` carries warning_type, duration, magnitudes — exactly
        # the kwargs ``revise_warning`` overlays onto the prior revision
        # (polygon is preserved by passing None). The reused MP/solo
        # split mirrors the issue flow in _finish_polygon so the same
        # routing applies whether we're online or in solo play.
        if self.multiplayer is not None:
            asyncio.ensure_future(self.multiplayer.revise_warning(
                warning_id=warning_id, player_id=self.local_player_id,
                **params,
            ))
        else:
            w = self.session.revise_warning(
                warning_id=warning_id, player_id=self.local_player_id,
                **params,
            )
            if self._replay is not None and self.session.clock:
                self._replay.log_warning_revise(
                    w, virtual_time=self.session.clock.virtual_time,
                )
        self._push_player_overlays()
        if self.host_map is not None:
            self.host_map.refresh()

    def _on_mcd_clicked(self, mcd_id: str) -> None:
        """Stub — MCDs aren't revisable in the current dialog, but we
        still log clicks so a future ``MCDFormDialog(existing=...)``
        path can drop in without rewiring the signal."""
        log.debug("MCD %s clicked — MCD revision not yet supported in UI",
                  mcd_id)

    def _push_player_overlays(self) -> None:
        """Hand the local player's + their teammates' warnings/MCDs to
        the radar grid so they appear immediately after issuance /
        revision / cancel without waiting for the next tick.

        Visibility rule (plan §11): a player sees their own warnings
        plus those of teammates only. Opposing teams' warnings are
        hidden on the per-player radar panel — they're visible on the
        host central map (which is the host's view of the whole room).
        Solo play degenerates correctly because the local player's
        "team" is the synthetic solo-team-of-one and the union below
        reduces to just their own warnings.

        Filtering to revisions-active-at-display-time happens inside
        the grid's draw routine.
        """
        team_ids = self._teammate_ids_including_self()
        warnings: list = []
        mcds: list = []
        for pid in team_ids:
            warnings.extend(self.session.warnings_by_player.get(pid, []))
            mcds.extend(self.session.mcds_by_player.get(pid, []))
        self.radar_grid.set_player_warnings(warnings, mcds)

    def _teammate_ids_including_self(self) -> list[str]:
        """All player ids on the same team as the local player, including
        the local player themselves. Falls back to just the local id
        when the player isn't yet bound to a team — happens briefly
        between joining the room and the host's TeamRosterFreeze."""
        me = self.session.players.get(self.local_player_id)
        if me is None or me.team_id is None:
            return [self.local_player_id]
        members = self.session.teams.get(me.team_id, [])
        if self.local_player_id not in members:
            members = list(members) + [self.local_player_id]
        return members

    def _toggle_inspector(self) -> None:
        """Keybind-driven inspector toggle. Reflects the new state back on
        the toolbar's checkable button so the UI stays in sync."""
        new_state = self.radar_grid.toggle_inspector()
        btn = getattr(self.radar_grid, "_inspector_btn", None)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(new_state)
            btn.blockSignals(False)

    def _toggle_motion_tool(self) -> None:
        # Motion-tool and polygon-draw both consume scene-level clicks.
        # If a polygon draw is in flight when the user toggles motion
        # on, cancel the draw first — otherwise every motion-tool
        # click would ALSO add a polygon vertex.
        if (not self.motion_tool.is_active
                and self._active_poly_editor is not None):
            self._cancel_polygon_draw()
        if self.motion_tool.is_active:
            # Stop listening for new clicks; existing tracks remain drawn
            # until the player right-clicks them.
            self.motion_tool.deactivate()
        else:
            self.motion_tool.activate()
        # Mirror the new state on the action-bar button so it lights up
        # while the tool is listening. blockSignals so the click handler
        # doesn't re-fire from our programmatic setChecked.
        btn = getattr(self, "_btn_motion", None)
        if btn is not None:
            btn.blockSignals(True)
            btn.setChecked(self.motion_tool.is_active)
            btn.blockSignals(False)

    # ---- shutdown ------------------------------------------------------

    def shutdown(self) -> None:
        self.clock_controls.stop()
        if self._peer_timer is not None:
            self._peer_timer.stop()
        # Close the separate host-overview window if we opened one.
        if self.host_map is not None:
            self.host_map.close()
            self.host_map = None
        self.prefetcher.shutdown(wait=False)
