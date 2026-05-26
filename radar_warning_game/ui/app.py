"""Top-level MainWindow that drives the single-player E2E flow.

State machine (matches GameSession.state with extra UI-only stages):

    DAY_PICKER → fetching reports → SETUP_OVERVIEW → SETUP_TIME → PREFETCH → PLAY → END

The window swaps its central widget at each transition. All cross-widget
plumbing happens here so the individual UI widgets stay self-contained.

Networking is NOT wired in yet — this is the solo-player driver. The same view
hierarchy will be reused for multiplayer; the host will additionally broadcast
each session mutation over the WebRTC DataChannel.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from ..data.cache import DEFAULT_CACHE_ROOT, HashedCache
from ..data.prefetch import Prefetcher
from ..data.reports import Report, fetch_iem_window
from ..game.clock import GameClock, LiveClock
from ..game.event_reveal import reveal_for
from ..game.round_builder import (
    RoundDay,
    ThresholdSpec,
    pick_random_day,
    pick_specific_day,
)
from ..game.session import GameSession, Player, RoundConfig, RoundMode
from ..net.multiplayer import MultiplayerHost, MultiplayerPeer
from ..net.peer import ClientTransport, HostTransport
from .day_picker import DayPickerDialog
from .leaderboard import FinalLeaderboardDialog
from .overview_map import OverviewMap
from .play_view import PlayView
from .prefetch_progress import PrefetchProgressWidget
from .room_dialogs import HostRoomStatusDialog, JoinRoomDialog, ModeDialog
from .time_distribution import TimeDistribution

log = logging.getLogger(__name__)


class _DayFetchWorker(QThread):
    """Background fetch of reports for the chosen day so the UI doesn't freeze."""

    done = pyqtSignal(object)         # RoundDay
    failed = pyqtSignal(str)

    def __init__(self, *, is_random: bool = False, spec: ThresholdSpec | None = None,
                 specific_date: datetime | None = None, is_live: bool = False,
                 live_lookback_hours: int = 6) -> None:
        super().__init__()
        self.is_random = is_random
        self.spec = spec
        self.specific_date = specific_date
        self.is_live = is_live
        self.live_lookback_hours = live_lookback_hours

    def run(self) -> None:
        try:
            if self.is_live:
                # For LIVE: fetch the last N hours of LSRs so the host has
                # spatial context when picking the polygon. Convective_day is
                # set to today 12Z (just for bookkeeping).
                from datetime import timedelta as _td, timezone as _tz
                now = datetime.now(_tz.utc)
                start = now - _td(hours=self.live_lookback_hours)
                reports = fetch_iem_window(start, now)
                today_12z = now.replace(hour=12, minute=0, second=0, microsecond=0)
                day = RoundDay(
                    convective_day_12z=today_12z, reports=reports,
                    counts={c: sum(1 for r in reports if r.category == c)
                            for c in ("tornado", "hail", "wind")},
                    is_random=False,
                )
            elif self.is_random:
                day = pick_random_day(self.spec)
            else:
                day = pick_specific_day(self.specific_date)
            self.done.emit(day)
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    """Top-level orchestrator window."""

    def __init__(self, *, local_player_name: str = "You") -> None:
        super().__init__()
        self.setWindowTitle("RadarAnalysisCorp")
        self.resize(1500, 950)
        self.local_player_id = "local"
        self.local_player_name = local_player_name

        self._stack = QStackedWidget(self)
        self.setCentralWidget(self._stack)

        bar = QStatusBar(self)
        self.setStatusBar(bar)

        self.session = GameSession()
        self.session.add_player(
            Player(player_id=self.local_player_id, display_name=local_player_name, is_host=True)
        )
        self._cache = HashedCache(DEFAULT_CACHE_ROOT / "radar", suffix=".ar2v")

        # Holders set as we progress
        self._round_day: RoundDay | None = None
        self._overview_map: OverviewMap | None = None
        self._time_dist: TimeDistribution | None = None
        self._prefetcher: Prefetcher | None = None
        self._prefetch_progress: PrefetchProgressWidget | None = None
        self._play_view: PlayView | None = None
        self._day_picker_is_random = True

        # Multiplayer state
        self._mode = ModeDialog.SOLO              # set in _show_mode_dialog
        self._host_transport: HostTransport | None = None
        self._peer_transport: ClientTransport | None = None
        self._multiplayer: MultiplayerHost | MultiplayerPeer | None = None
        self._signaling_url: str | None = None

        self._show_mode_dialog()

    # ----------------------------------------------------------------------
    # mode selection
    # ----------------------------------------------------------------------

    def _show_mode_dialog(self) -> None:
        dlg = ModeDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.close()
            return
        self._mode = dlg.mode()
        self.local_player_name = dlg.display_name()
        self.session.players[self.local_player_id].display_name = self.local_player_name
        self._signaling_url = dlg.signaling_url()
        if self._mode == ModeDialog.SOLO:
            self._show_day_picker()
        elif self._mode == ModeDialog.HOST:
            asyncio.ensure_future(self._begin_host_mode())
        else:
            asyncio.ensure_future(self._begin_join_mode())

    async def _begin_host_mode(self) -> None:
        """Start a HostTransport, show the room-status dialog while accepting peers."""
        self._host_transport = HostTransport(
            name=self.local_player_name, signaling_url=self._signaling_url,
        )
        try:
            room_code = await self._host_transport.start()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Host failed", f"Could not host room: {e}")
            self.close()
            return
        # Build the multiplayer wrapper now (so it can register peer callbacks)
        self._multiplayer = MultiplayerHost(self.session, self._host_transport)
        # Show room-status dialog; host clicks "Continue to Setup" when ready
        room_dlg = HostRoomStatusDialog(room_code, self)
        # Hook the peer-joined callback to update the dialog list
        self._host_transport._on_peer_joined = lambda pid: room_dlg.add_peer(pid, pid)
        self._host_transport._on_peer_left = lambda pid: room_dlg.remove_peer(pid)
        if room_dlg.exec() == QDialog.DialogCode.Accepted:
            self._show_day_picker()

    async def _begin_join_mode(self) -> None:
        """Prompt for a room code, connect, wait for RoundSetup from host, then enter prefetch."""
        join_dlg = JoinRoomDialog(self)
        if join_dlg.exec() != QDialog.DialogCode.Accepted:
            self.close()
            return
        room_code = join_dlg.room_code()
        self._peer_transport = ClientTransport(
            name=self.local_player_name, signaling_url=self._signaling_url,
        )
        try:
            await self._peer_transport.join(room_code)
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Join failed", f"Could not join room {room_code}: {e}")
            self.close()
            return
        # Build the peer wrapper (registers the message handler)
        self._multiplayer = MultiplayerPeer(self.session, self._peer_transport)
        # Show a "waiting for host setup..." placeholder while we wait
        placeholder = QWidget(self)
        pl = QVBoxLayout(placeholder)
        pl.addStretch(1)
        msg = QLabel(
            f"Joined room <b>{room_code}</b>. Waiting for host to start the round…",
            placeholder,
        )
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(msg)
        pl.addStretch(1)
        self._stack.addWidget(placeholder)
        self._stack.setCurrentWidget(placeholder)
        # Poll for the RoundSetup message to arrive (it sets round_config on the session)
        for _ in range(600):  # up to 5 minutes
            await asyncio.sleep(0.5)
            if self.session.round_config is not None:
                break
        if self.session.round_config is None:
            QMessageBox.warning(self, "No round started",
                                "Host didn't start a round in time. Returning to mode selection.")
            self.close()
            return
        # Peer skips polygon/time setup — config came from the wire. Go straight to prefetch.
        cfg = self.session.round_config
        self.session.clock = GameClock(cfg.time_start, cfg.time_end)
        self.session.begin_play()           # peer's session is now PLAYING
        self._prefetcher = Prefetcher(list(cfg.radar_sites), self._cache)
        self._prefetcher.schedule_pregame(cfg.time_start, cfg.time_end)
        self._prefetch_progress = PrefetchProgressWidget(self._prefetcher)
        self._prefetch_progress.ready_to_play.connect(self._enter_play)
        self._stack.addWidget(self._prefetch_progress)
        self._stack.setCurrentWidget(self._prefetch_progress)

    # ----------------------------------------------------------------------
    # state transitions
    # ----------------------------------------------------------------------

    def _show_day_picker(self) -> None:
        dlg = DayPickerDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            self.close()
            return
        self._day_picker_is_random = dlg.is_random()
        self._save_replay = dlg.save_replay()
        self._team_mode = dlg.team_mode()
        self._is_live = dlg.is_live()
        self.statusBar().showMessage("Fetching reports…")

        if dlg.is_live():
            worker = _DayFetchWorker(is_live=True, live_lookback_hours=6)
        elif dlg.is_random():
            worker = _DayFetchWorker(is_random=True, spec=dlg.thresholds())
        else:
            worker = _DayFetchWorker(is_random=False, specific_date=dlg.specific_date_12z())
        worker.done.connect(self._on_day_fetched)
        worker.failed.connect(self._on_day_failed)
        self._day_worker = worker
        worker.start()
        # Show a placeholder while fetching
        placeholder = QWidget(self)
        pl = QVBoxLayout(placeholder)
        pl.addStretch(1)
        msg = QLabel("Fetching reports…", placeholder)
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pl.addWidget(msg)
        pl.addStretch(1)
        self._stack.addWidget(placeholder)
        self._stack.setCurrentWidget(placeholder)

    def _on_day_failed(self, message: str) -> None:
        QMessageBox.critical(self, "Day fetch failed", message)
        self._show_day_picker()

    def _on_day_fetched(self, day: RoundDay) -> None:
        self._round_day = day
        log.info("Day fetched: %d reports (random=%s)", len(day.reports), day.is_random)
        self.statusBar().showMessage(
            f"{len(day.reports)} reports loaded — pick game polygon + active radars"
        )
        self._overview_map = OverviewMap(
            reports=day.reports,
            is_random_day=day.is_random,
        )
        self._overview_map.reroll_requested.connect(self._on_reroll)
        self._overview_map.polygon_changed.connect(self._on_polygon_changed)
        self._overview_map.continue_requested.connect(self._continue_to_time)
        self._stack.addWidget(self._overview_map)
        self._stack.setCurrentWidget(self._overview_map)

    def _on_polygon_changed(self, polygon) -> None:
        if polygon is not None:
            self.statusBar().showMessage(
                f"Polygon set ({len(polygon.vertices)} vertices), "
                f"{len(self._overview_map.enabled_sites())} radar(s) — Ctrl+Enter to continue"
            )

    def _on_reroll(self) -> None:
        if self._round_day is None or not self._round_day.is_random:
            return
        # Fire a fresh fetch with the same thresholds (we kept day_picker_is_random True)
        self.statusBar().showMessage("Rerolling…")
        # In v1 we don't preserve the threshold spec across reroll — assume a re-pick
        # with the same defaults. A future improvement is to stash the spec.
        worker = _DayFetchWorker(is_random=True, spec=ThresholdSpec(5, 20, 20))
        worker.done.connect(self._on_reroll_fetched)
        worker.failed.connect(self._on_day_failed)
        self._day_worker = worker
        worker.start()

    def _on_reroll_fetched(self, day: RoundDay) -> None:
        self._round_day = day
        if self._overview_map is not None:
            self._overview_map.replace_reports(day.reports)
            self.statusBar().showMessage(f"Rerolled — {len(day.reports)} reports")

    def _continue_to_time(self) -> None:
        if self._overview_map is None or self._round_day is None:
            return
        polygon = self._overview_map.polygon()
        enabled_sites = self._overview_map.enabled_sites()
        if polygon is None:
            QMessageBox.warning(self, "Missing polygon",
                                "Click 3+ points on the map to define the game polygon.")
            return
        if not enabled_sites:
            QMessageBox.warning(self, "No radars",
                                "Click ≥1 radar site (X markers) to enable it for the round.")
            return
        # Filter reports to those inside the polygon for the histogram
        from ..verification.reports_in_poly import reports_in_polygon
        inside = reports_in_polygon(polygon, self._round_day.reports)
        log.info("Polygon contains %d reports of %d", len(inside), len(self._round_day.reports))
        # Cache picks for prefetch transition
        self._chosen_polygon = polygon
        self._chosen_sites = enabled_sites

        if getattr(self, "_is_live", False):
            # Live mode: window is "now → now + 2h" by default (host can end early)
            from datetime import timedelta as _td, timezone as _tz
            now = datetime.now(_tz.utc)
            self._live_time_start = now
            self._live_time_end = now + _td(hours=2)
            self._continue_to_prefetch()
            return

        self._time_dist = TimeDistribution(
            inside, day_start_12z=self._round_day.convective_day_12z,
        )
        self._time_dist.start_requested.connect(self._continue_to_prefetch)
        self._stack.addWidget(self._time_dist)
        self._stack.setCurrentWidget(self._time_dist)
        self.statusBar().showMessage(
            "Drag the yellow span to pick the game window, then click Start round"
        )

    def _continue_to_prefetch(self) -> None:
        if self._round_day is None:
            return
        if getattr(self, "_is_live", False):
            time_start = self._live_time_start
            time_end = self._live_time_end
        else:
            if self._time_dist is None:
                return
            time_start, time_end = self._time_dist.selected_window()
            if time_end <= time_start:
                QMessageBox.warning(self, "Bad time window", "End time must be after start.")
                return
        # Build round config + session state
        config = RoundConfig(
            convective_day_12z=self._round_day.convective_day_12z,
            game_polygon=self._chosen_polygon,
            radar_sites=sorted(self._chosen_sites),
            time_start=time_start,
            time_end=time_end,
            save_replay=self._save_replay,
            team_mode=self._team_mode,
            mode=RoundMode.LIVE if getattr(self, "_is_live", False) else RoundMode.HISTORICAL,
        )
        self.session.set_round(self._round_day, config)
        self.session.freeze_roster()
        self.session.begin_prefetch()

        # Kick off the prefetcher (live mode uses the IEM live source, not S3)
        is_live = config.mode == RoundMode.LIVE
        self._prefetcher = Prefetcher(sorted(self._chosen_sites), self._cache,
                                       live_source=is_live)
        self._prefetcher.schedule_pregame(time_start, time_end)
        self._prefetch_progress = PrefetchProgressWidget(self._prefetcher)
        self._prefetch_progress.ready_to_play.connect(self._enter_play)
        self._stack.addWidget(self._prefetch_progress)
        self._stack.setCurrentWidget(self._prefetch_progress)
        self.statusBar().showMessage("Downloading radar volumes…")

    def _enter_play(self) -> None:
        if self._prefetcher is None or self.session.round_config is None:
            return
        # Live mode: replace the bare GameClock with a wall-clock LiveClock
        if self.session.round_config.mode == RoundMode.LIVE:
            self.session.clock = LiveClock(
                self.session.round_config.time_start,
                self.session.round_config.time_end,
            )
        if self.session.state.value != "PLAYING":
            self.session.begin_play()
        # If we're hosting, announce the round to peers now (before play view exists).
        if isinstance(self._multiplayer, MultiplayerHost):
            asyncio.ensure_future(self._multiplayer.announce_round_setup())
        self._play_view = PlayView(
            session=self.session,
            prefetcher=self._prefetcher,
            local_player_id=self.local_player_id,
            multiplayer=self._multiplayer,
        )
        self._replay_path: str | None = None
        self._play_view.replay_saved.connect(lambda p: setattr(self, "_replay_path", p))
        self._play_view.round_ended.connect(self._show_final_screen)
        self._stack.addWidget(self._play_view)
        self._stack.setCurrentWidget(self._play_view)
        mode_label = {"solo": "Solo", "host": "Hosting", "join": "Peer"}.get(self._mode, "")
        self.statusBar().showMessage(
            f"Playing ({mode_label}) — N=new warning · C=new MCD · M=motion tool · [ slower · ] faster · Space pause"
        )

    def _show_final_screen(self) -> None:
        if self.session.final_scores is None:
            return
        cfg = self.session.round_config
        reveal = reveal_for(cfg.convective_day_12z) if cfg else None
        dlg = FinalLeaderboardDialog(
            self.session.final_scores,
            self.session.team_names,
            date_reveal=reveal.date_str if reveal else None,
            location_reveal=f"{reveal.name} — {reveal.location}" if reveal else None,
            event_url=reveal.url if reveal else None,
            replay_path=getattr(self, "_replay_path", None),
            parent=self,
        )
        dlg.exec()
        if self._play_view is not None:
            self._play_view.shutdown()
        self.close()

    # ----------------------------------------------------------------------
    # cleanup
    # ----------------------------------------------------------------------

    def closeEvent(self, event) -> None:  # noqa: N802
        if self._play_view is not None:
            self._play_view.shutdown()
        if self._prefetcher is not None:
            self._prefetcher.shutdown(wait=False)
        # Best-effort: schedule transport shutdown but don't block the close
        if self._host_transport is not None:
            asyncio.ensure_future(self._host_transport.stop())
        if self._peer_transport is not None:
            asyncio.ensure_future(self._peer_transport.stop())
        super().closeEvent(event)
