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
    QSplitter,
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
from .radar_panel import RadarPanelGrid
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
        # Construct without the game-clock cap so the initial display can show
        # the first available sweep even if it's timestamped slightly after the
        # round's nominal start time (NEXRAD scans align to their own schedule,
        # not the round window). The cap is enforced from the first tick onward
        # via _on_tick → set_max_virtual_time.
        self.radar_grid = RadarPanelGrid(
            sweep_index=prefetcher.sweep_index(initial_site),
            site_icao=initial_site,
            n_panels=4,
            max_virtual_time=None,
        )
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

        # Host central map — only meaningful in multiplayer where it shows
        # all players' warnings. In solo the player only has their own
        # warnings (which they can see directly on the radar panels and the
        # game-polygon overlay), so the central map is redundant; skip it
        # and give the radar grid the full width.
        self._is_solo = multiplayer is None
        self.host_map: HostCentralMap | None = (
            None if self._is_solo else HostCentralMap(session)
        )

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
        if self.host_map is not None:
            splitter = QSplitter(Qt.Orientation.Horizontal, self)
            splitter.addWidget(self.radar_grid)
            splitter.addWidget(self.host_map)
            splitter.setSizes([900, 700])
            layout.addWidget(splitter, stretch=1)
        else:
            layout.addWidget(self.radar_grid, stretch=1)

        # Keyboard shortcuts — registered at the PlayView level so they fire
        # regardless of which child widget (radar grid, clock bar, host map)
        # currently holds Qt focus. Per-widget keyPressEvent handlers only
        # fire when that widget owns focus, which is unreliable in practice.
        QShortcut(QKeySequence("N"), self, activated=self._begin_warning_polygon)
        QShortcut(QKeySequence("C"), self, activated=self._begin_mcd_polygon)
        QShortcut(QKeySequence("M"), self, activated=self._toggle_motion_tool)
        QShortcut(QKeySequence("Esc"), self, activated=self._cancel_polygon_draw)
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
        # PRODUCTS dict ordering: 1=REF, 2=VEL, 3=SW, 4=CC, 5=ZDR, 6=KDP, 7=PHI.
        # Click a panel to focus it before pressing the digit.
        for i, key in enumerate(("1", "2", "3", "4", "5", "6", "7")):
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
        """
        bar = QFrame(self)
        bar.setFrameShape(QFrame.Shape.StyledPanel)
        h = QHBoxLayout(bar)
        h.setContentsMargins(6, 4, 6, 4)
        h.setSpacing(6)

        def _btn(label: str, tip: str, slot, *, danger: bool = False) -> QToolButton:
            b = QToolButton(bar)
            b.setText(label)
            b.setToolTip(tip)
            b.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            if danger:
                b.setStyleSheet("color: #ff8888;")
            b.clicked.connect(slot)
            return b

        self._btn_new_warning = _btn(
            "▲ New Warning  (N)", "Begin a freehand warning polygon",
            self._begin_warning_polygon,
        )
        self._btn_new_mcd = _btn(
            "◇ New MCD  (C)", "Begin a freehand Mesoscale Convective Discussion",
            self._begin_mcd_polygon,
        )
        self._btn_motion = _btn(
            "↗ Motion Tool  (M)", "Two-click storm-motion measurement",
            self._toggle_motion_tool,
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
        """Toggle visibility of the Finish/Cancel buttons + hint label."""
        if not hasattr(self, "_btn_finish"):
            return
        self._btn_finish.setVisible(drawing)
        self._btn_cancel.setVisible(drawing)
        self._btn_new_warning.setEnabled(not drawing)
        self._btn_new_mcd.setEnabled(not drawing)
        self._btn_motion.setEnabled(not drawing)
        if drawing and self._pending_action:
            kind = "warning" if self._pending_action == "warning" else "MCD"
            self._draw_hint.setText(
                f"Drawing {kind} polygon — click to add vertices"
            )
        else:
            self._draw_hint.setText("")

    # ---- tick handling -------------------------------------------------

    def _on_tick(self, tick) -> None:
        # Update game-clock cap on the radar panel so scrubbing is bounded
        self.radar_grid.set_max_virtual_time(tick.virtual_time)
        # Tell prefetcher to advance its lookahead buffer
        self.prefetcher.advance_clock(tick.virtual_time)
        # Push reports up to virtual_time onto the radar panel for live overlay
        if self.session.round_day is not None:
            visible = [r for r in self.session.round_day.reports if r.time <= tick.virtual_time]
            self.radar_grid.live_reports = visible
        # Player's own warnings/MCDs overlaid on each radar panel — drawn for
        # whichever revision is active at the panel's display time, so
        # scrubbing back shows the polygon as it was then.
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
        """
        if not self.radar_grid._panels:
            return
        panel = self.radar_grid._panels[self.radar_grid._focused_panel_index()]
        site = self.radar_grid.site
        from ..geo.projection import xy_km_to_latlon
        self._active_poly_editor = PolygonEditor(
            panel.ax,
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
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, activated=self._finish_polygon)
        self._update_action_bar_mode(drawing=True)

    def _cancel_polygon_draw(self) -> None:
        """Esc handler — clear in-flight polygon and reset the cursor / status."""
        if self._active_poly_editor is None:
            return
        self._active_poly_editor.clear()
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
        editor.clear()

    def _push_player_overlays(self) -> None:
        """Hand the local player's warnings/MCDs to the radar grid so they
        appear immediately after issuance/revision/cancel without waiting
        for the next tick. Filtering to revisions-active-at-display-time
        happens inside the grid's draw routine.
        """
        self.radar_grid.set_player_warnings(
            self.session.warnings_by_player.get(self.local_player_id, []),
            self.session.mcds_by_player.get(self.local_player_id, []),
        )

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
        if self.motion_tool.is_active:
            # Stop listening for new clicks; existing tracks remain drawn
            # until the player right-clicks them.
            self.motion_tool.deactivate()
        else:
            self.motion_tool.activate()

    # ---- shutdown ------------------------------------------------------

    def shutdown(self) -> None:
        self.clock_controls.stop()
        if self._peer_timer is not None:
            self._peer_timer.stop()
        self.prefetcher.shutdown(wait=False)
