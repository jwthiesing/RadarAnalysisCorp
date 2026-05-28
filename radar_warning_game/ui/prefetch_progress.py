"""Pre-round prefetch progress UI (plan §10).

Single-player simplification of the multi-client gate from the plan: in solo
mode we just download all pre-game volumes in parallel and show per-radar
progress. When all done, we emit :attr:`ready_to_play` and the parent
transitions to the PLAYING view.

The Prefetcher itself runs background threads; we poll it on a QTimer to
update the bars.

If a radar's listing returns 0 scans (the day/site has no archive data
on the Unidata mirror) the widget surfaces that loudly instead of
sitting at "(listing…)" forever — and offers a back-to-radar-selection
escape hatch so the host can deselect dead sites or pick a different
day instead of staring at blank panels with no way out.
"""

from __future__ import annotations

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..data.prefetch import Prefetcher


class PrefetchProgressWidget(QWidget):
    """Shows a progress bar per enabled radar while the pre-game window downloads.

    Signals
    -------
    local_prefetch_done
        emitted once every *populated* site this client downloads has
        finished. Distinct from :attr:`ready_to_play` so the multiplayer
        start-gate (plan §10) can intercept it — peers send a PeerReady
        to the host and wait for the countdown to hit zero before
        actually entering play. In solo / "Start anyway" paths the app
        layer wires this straight through to ``ready_to_play``.
    ready_to_play
        emitted when the round should actually begin — i.e. the gate has
        released. Always wired to ``_enter_play`` by app.py. The widget
        itself only emits this from the host's "Start anyway" button;
        all other paths go through ``local_prefetch_done`` first.
    back_requested
        emitted when the user clicks the back/leave button. The
        appropriate parent behavior depends on ``is_peer``:
          - host: discard the prefetcher and return to the radar-selection
            map so the host can revise radar/day choices.
          - peer: disconnect from the room and return to the mode dialog
            (peers can't pick radars — the only escape is to leave).
        Always available so neither role is stuck on a misconfigured round.
    """

    local_prefetch_done = pyqtSignal()
    ready_to_play = pyqtSignal()
    # Emitted when the host clicks "Start anyway" — the multiplayer
    # path needs to broadcast a start signal to peers in addition to
    # entering play locally, so the connection target differs by mode.
    # In solo mode the app layer wires this straight to ``ready_to_play``.
    force_start_requested = pyqtSignal()
    back_requested = pyqtSignal()

    def __init__(
        self,
        prefetcher: Prefetcher,
        parent: QWidget | None = None,
        *,
        is_peer: bool = False,
    ) -> None:
        super().__init__(parent)
        self.prefetcher = prefetcher
        self._is_peer = is_peer

        self._title = QLabel("Downloading radar volumes for round start…", self)
        self._title.setStyleSheet("font-size: 13pt; padding: 8px;")
        self._title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Snapshot the per-site availability at construction time.
        # ``schedule_pregame`` is synchronous, so by the time this widget
        # is built every site's ``pregame_total`` is final — sites that
        # report 0 here have no archive data for the chosen day, which
        # is a permanent condition (not a "still listing" race).
        initial = self.prefetcher.pregame_progress()
        self._empty_sites: set[str] = {
            s for s, (_done, total) in initial.items() if total == 0
        }
        all_empty = (
            bool(self.prefetcher.sites)
            and len(self._empty_sites) == len(self.prefetcher.sites)
        )

        self._bars: dict[str, QProgressBar] = {}
        self._site_status_labels: dict[str, QLabel] = {}
        layout = QVBoxLayout(self)
        layout.addStretch(1)
        layout.addWidget(self._title)

        if all_empty:
            if is_peer:
                explanation = (
                    "<b>The host picked radars with no archive data for this day.</b><br><br>"
                    "Wait for the host to revise the round, or click<br>"
                    "<b>Leave room</b> to disconnect and return to mode select."
                )
            else:
                explanation = (
                    "<b>No archive data found for any selected radar on this day.</b><br><br>"
                    "The Unidata Level 2 mirror has no objects under any of the<br>"
                    "per-site prefixes for this UTC day. This usually means:<br>"
                    "&nbsp;&nbsp;• the chosen day is before the radar fleet's<br>"
                    "&nbsp;&nbsp;&nbsp;&nbsp;archive coverage, or<br>"
                    "&nbsp;&nbsp;• every selected radar was offline / not yet<br>"
                    "&nbsp;&nbsp;&nbsp;&nbsp;installed on that day.<br><br>"
                    "Click <b>Back to radar selection</b> to pick a different<br>"
                    "day or a different set of radars."
                )
            warn = QLabel(explanation)
            warn.setStyleSheet(
                "color: #ff6060; padding: 12px; font-size: 11pt; "
                "background-color: rgba(255, 60, 60, 0.08);"
            )
            warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warn.setTextFormat(Qt.TextFormat.RichText)
            layout.addWidget(warn)
        elif self._empty_sites:
            n_bad = len(self._empty_sites)
            n_ok = len(self.prefetcher.sites) - n_bad
            bad_csv = ', '.join(sorted(self._empty_sites))
            if is_peer:
                partial_text = (
                    f"<b>{n_bad} of {len(self.prefetcher.sites)} radars the "
                    f"host picked have no archive data for this day:</b> "
                    f"{bad_csv}.<br>The round will play with the "
                    f"{n_ok} working radar(s) — those dead radars' panels "
                    f"will stay blank."
                )
            else:
                partial_text = (
                    f"<b>{n_bad} of {len(self.prefetcher.sites)} selected "
                    f"radars have no archive data for this day:</b> "
                    f"{bad_csv}.<br>"
                    f"You can <b>Start anyway</b> with the {n_ok} working "
                    f"radar(s), or <b>Back to radar selection</b> to revise."
                )
            warn = QLabel(partial_text)
            warn.setStyleSheet(
                "color: #ffb060; padding: 12px; font-size: 10pt; "
                "background-color: rgba(255, 160, 0, 0.08);"
            )
            warn.setAlignment(Qt.AlignmentFlag.AlignCenter)
            warn.setTextFormat(Qt.TextFormat.RichText)
            warn.setWordWrap(True)
            layout.addWidget(warn)

        # Per-site preprocessing (PyART parse + velocity dealias) bars.
        # Same gating logic as the download bars: empty sites get the
        # muted-red "no data" treatment and don't gate the round.
        self._preload_bars: dict[str, QProgressBar] = {}
        for site in prefetcher.sites:
            row = QLabel(site, self)
            row.setAlignment(Qt.AlignmentFlag.AlignCenter)
            bar = QProgressBar(self)
            bar.setRange(0, 100)
            bar.setValue(0)
            preload_bar = QProgressBar(self)
            preload_bar.setRange(0, 100)
            preload_bar.setValue(0)
            preload_bar.setFormat("preprocess: 0/0")
            if site in self._empty_sites:
                # Permanently empty — render both bars in muted red and
                # set their text to the diagnosis so it's not confused
                # with the "(listing…)" startup state.
                for b in (bar, preload_bar):
                    b.setRange(0, 1)
                    b.setValue(0)
                    b.setStyleSheet(
                        "QProgressBar { background-color: #3a1a1a; color: #ff8a8a; "
                        "border: 1px solid #5a2a2a; text-align: center; }"
                    )
                bar.setFormat(f"{site}: no archive data for this day")
                preload_bar.setFormat("preprocess: —")
            layout.addWidget(row)
            layout.addWidget(bar)
            layout.addWidget(preload_bar)
            self._bars[site] = bar
            self._preload_bars[site] = preload_bar

        # Buttons row — always show "back to radar selection" so there's
        # an escape even when the prefetcher is happily downloading
        # (sometimes the host realizes they picked the wrong site).
        # The forward-button visibility tracks the availability state:
        # all-empty → no "Start anyway" (it would just open blank panels).
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        # Back/leave button — wording depends on role since peers can't
        # change the host's radar selection; the only escape is to
        # disconnect from the room.
        back_label = "← Leave room" if is_peer else "← Back to radar selection"
        self._back_btn = QPushButton(back_label, self)
        self._back_btn.clicked.connect(self.back_requested.emit)
        btn_row.addWidget(self._back_btn)
        # Peers never get a "Start anyway" — they auto-advance when the
        # host's downloads are done (or via ready_to_play). The button
        # would imply manual control they don't have.
        if not all_empty and not is_peer:
            # "Start anyway" deliberately *bypasses* the multiplayer
            # readiness gate — host explicitly chose to begin
            # regardless of peer readiness. ``force_start_requested``
            # routes through the MultiplayerHost so peers get a
            # RoundCountdown(0) start signal in addition to the local
            # ``ready_to_play``; solo mode wires force_start straight
            # to ready_to_play.
            self._skip_btn = QPushButton("Start anyway", self)
            self._skip_btn.clicked.connect(self.force_start_requested.emit)
            btn_row.addWidget(self._skip_btn)
        else:
            self._skip_btn = None
        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addStretch(1)

        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._poll)
        # Don't bother polling if there's literally nothing to download.
        if not all_empty:
            self._timer.start()

    def _poll(self) -> None:
        progress = self.prefetcher.pregame_progress()
        preload_progress = self.prefetcher.pregame_preload_progress()
        # Gate auto-advance on whether *non-empty* sites have finished
        # BOTH the download AND the PyART parse + velocity dealias.
        # Empty sites stay at 0/0 forever and would otherwise block
        # ``local_prefetch_done`` from ever firing. The user-visible
        # "ready" state needs both phases done so the round doesn't
        # start with the host idle and peers still chewing through
        # dealias on a slow CPU.
        all_done = True
        any_seen = False
        for site, bar in self._bars.items():
            if site in self._empty_sites:
                continue
            done, total = progress.get(site, (0, 0))
            pl_done, pl_total = preload_progress.get(site, (0, 0))
            if total > 0:
                any_seen = True
                bar.setRange(0, total)
                bar.setValue(done)
                bar.setFormat(f"download: {done}/{total}")
                if done < total:
                    all_done = False
                pl_bar = self._preload_bars[site]
                # preload_total tracks pregame_total once schedule_pregame
                # finishes listing; until the first download completes,
                # it's 0. Show as the same denominator as downloads so
                # the user sees a clear 0/N progression rather than
                # 0/0 → 1/1 → 2/2 jitter.
                pl_denom = total
                pl_bar.setRange(0, max(pl_denom, 1))
                pl_bar.setValue(pl_done)
                pl_bar.setFormat(f"preprocess: {pl_done}/{pl_denom}")
                if pl_done < pl_denom:
                    all_done = False
            else:
                bar.setRange(0, 1)
                bar.setValue(0)
                bar.setFormat("(listing…)")
                all_done = False
        if all_done and any_seen:
            self._timer.stop()
            self.local_prefetch_done.emit()

    def set_countdown(self, seconds_remaining: int) -> None:
        """Update the title to display the start-gate countdown.

        Called from the multiplayer path when the host broadcasts each
        ``RoundCountdown`` tick. Idempotent — set to the same value
        twice just rerenders the same string. Counts down once, then
        the parent calls ``ready_to_play.emit()`` separately to actually
        enter play.
        """
        if seconds_remaining <= 0:
            self._title.setText("Starting round…")
        else:
            self._title.setText(
                f"Starting round in {seconds_remaining}s — radar prefetch continues"
            )

    def set_waiting_for_peers(self) -> None:
        """Title-only update shown after local prefetch finishes but
        before the gate threshold is met — i.e. we're waiting on other
        clients to catch up. Replaced by ``set_countdown`` once the
        host kicks off the timer."""
        self._title.setText("Local prefetch complete — waiting for other clients…")

    def stop(self) -> None:
        self._timer.stop()
