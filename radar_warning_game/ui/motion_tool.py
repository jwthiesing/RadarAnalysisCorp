"""Two-point storm-motion measurement tool (plan §5a).

Attaches to a :class:`RadarPanelGrid`. While active, the next two clicks on the
focused radar panel drop markers at scan-time-stamped points; the tool then
computes the storm's bearing (TO + FROM), speed in kt, and displays the
NWS-style annotation plus an arrow from P1 → P2.

Each completed track is **persistent** — it stays drawn until the player
right-clicks on any of its artists to delete it. The player can keep
left-clicking to drop additional tracks alongside existing ones. This is
**purely informational** (plan §5a) — tracks do NOT auto-generate warning
polygons.

Scrubbing while the tool is active uses the same controls as normal (``←/→``,
``Shift+←/→``, ``↑/↓``), and the per-player game-clock cap still applies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from PyQt6.QtCore import QObject, pyqtSignal

from ..data.sites import Site
from ..geo.projection import StormMotion, storm_motion_from_two_points

log = logging.getLogger(__name__)


@dataclass
class _StormTrack:
    """One completed two-point measurement and all its matplotlib artists.

    Held in :attr:`MotionTool._tracks` so the player can keep multiple
    measurements on screen at once and right-click any of them to remove
    just that track.
    """

    p1: tuple[float, float, float]    # (x_km, y_km, t_sec)
    p2: tuple[float, float, float]
    motion: StormMotion | None
    artists: list = field(default_factory=list)


class MotionTool(QObject):
    """Mouse-click motion measurement controller.

    Lifecycle::

        tool = MotionTool(grid)
        tool.activate()        # enable left-click point-drops
        # ... player clicks two points for each track ...
        tool.deactivate()      # stop listening; tracks remain visible

    Right-clicking on any drawn track (while the tool is active) removes
    just that track.

    Signals
    -------
    motion_measured(StormMotion)
        emitted when the second click of a track lands
    point_added(int)
        emitted with 1 after first click of a track, 2 after second
    cleared()
        emitted when every track is wiped (e.g. layout change)
    """

    motion_measured = pyqtSignal(object)   # StormMotion
    point_added = pyqtSignal(int)
    cleared = pyqtSignal()
    error = pyqtSignal(str)                # user-visible error (e.g. same-sweep clicks)

    def __init__(self, grid) -> None:
        super().__init__()
        self._grid = grid
        # ``_active`` only gates *new-point* left-clicks. Right-click track
        # deletion stays armed even when the tool is "off" so the player can
        # tidy up old measurements without re-arming the tool.
        self._active = False
        # In-progress measurement state (cleared after second click commits a track).
        self._p1: tuple[float, float, float] | None = None
        self._inprogress_artists: list = []
        # All completed tracks. Each owns its own matplotlib artist list.
        self._tracks: list[_StormTrack] = []
        # mpl callback ids — one per panel canvas. Installed lazily on
        # first activate and kept attached for the panel's lifetime so
        # right-click removal works even when the tool is inactive.
        self._cids: list[tuple[object, int]] = []
        self._handlers_installed = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Begin accepting left-clicks for new measurements. Right-click
        track removal is always armed (after first activation) regardless
        of this flag."""
        self._active = True
        self._ensure_handlers_installed()

    def deactivate(self) -> None:
        """Stop accepting new left-clicks. Existing tracks remain drawn and
        can still be removed via right-click."""
        if not self._active:
            return
        self._active = False
        # Drop any half-finished measurement (single un-paired P1 marker)
        # so the next activation starts cleanly. Committed tracks are kept.
        if self._p1 is not None:
            self._discard_inprogress()

    def _ensure_handlers_installed(self) -> None:
        """Attach the button_press_event handler to every panel canvas
        exactly once. Idempotent — safe to call repeatedly."""
        if self._handlers_installed:
            return
        for panel in self._grid._panels:
            canvas = panel._canvas
            cid = canvas.mpl_connect("button_press_event", self._on_click)
            self._cids.append((canvas, cid))
        self._handlers_installed = True

    def reinstall_handlers_for_new_panels(self) -> None:
        """Called after :class:`RadarPanelGrid` rebuilds its panels (e.g. on
        layout change) — old canvas cids are stale, and any track artists
        whose axes were destroyed are unrecoverable. Wipes stale tracks and
        re-attaches handlers on the fresh canvases."""
        # The old cids point to canvases that have been deleted; just drop
        # them rather than trying to disconnect (would raise or be a no-op).
        self._cids.clear()
        self._handlers_installed = False
        # Stale track artists belong to dead axes. Forget them entirely.
        self._tracks.clear()
        self._p1 = None
        self._inprogress_artists.clear()
        # If the tool was active when the layout changed, keep it active —
        # just refresh the handler attachments on the new canvases.
        if self._active:
            self._ensure_handlers_installed()
        else:
            # Inactive: still install handlers so right-click removal works
            # if the user later draws new tracks and wants to remove them.
            # No-op until they activate the tool, but having handlers
            # attached doesn't hurt.
            self._ensure_handlers_installed()

    def reset(self) -> None:
        """Cancel an in-progress measurement without disturbing completed
        tracks. Used by the host's "exit motion tool" path."""
        if self._p1 is not None:
            self._discard_inprogress()
            self.cleared.emit()

    def clear_all(self) -> None:
        """Remove every track (and any in-progress P1). Used on layout
        changes that invalidate the underlying axes/artists."""
        for track in self._tracks:
            for a in track.artists:
                try:
                    a.remove()
                except (ValueError, AttributeError):
                    pass
        self._tracks.clear()
        if self._p1 is not None:
            self._discard_inprogress()
        for panel in self._grid._panels:
            panel._canvas.draw_idle()
        self.cleared.emit()

    def motion(self) -> StormMotion | None:
        """Motion of the most-recently completed track (or ``None``)."""
        if not self._tracks:
            return None
        return self._tracks[-1].motion

    # ---- event handler --------------------------------------------------

    def _on_click(self, event) -> None:
        # Right-click → try to delete a track whose artist we hit. Works
        # independent of ``_active`` — the player can remove old tracks at
        # any time, even with the motion tool "off".
        if event.button == 3:
            self._handle_right_click(event)
            return
        if event.button != 1:
            return
        # New-point left-clicks only fire when the tool is active.
        if not self._active:
            return
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        if self._grid._current_sweep is None:
            return
        t_sec = self._grid._current_sweep.start_time.timestamp()
        point = (float(event.xdata), float(event.ydata), t_sec)
        if self._p1 is None:
            self._p1 = point
            marker = self._draw_marker(ax, point[0], point[1])
            self._inprogress_artists.append(marker)
            self.point_added.emit(1)
            return
        # Second click → commit a track.
        if point[2] == self._p1[2]:
            # Same sweep → Δt = 0, can't compute motion. Ask user to scrub.
            self.error.emit("Scrub time forward (← →) before clicking point 2.")
            return
        p2 = point
        # Draw P2 marker, arrow, forecast track and accumulate into a new
        # track group. The in-progress P1 marker is folded into the track
        # so the right-click hit-test can remove the whole group.
        artists = list(self._inprogress_artists)
        artists.append(self._draw_marker(ax, p2[0], p2[1]))
        motion = self._compute_motion(self._p1, p2)
        artists.extend(self._draw_arrow_and_label(ax, self._p1, p2, motion))
        artists.extend(self._draw_forecast_track(ax, self._p1, p2))
        track = _StormTrack(p1=self._p1, p2=p2, motion=motion, artists=artists)
        self._tracks.append(track)
        ax.figure.canvas.draw_idle()
        # Reset in-progress state — next left-click starts a NEW track,
        # alongside the one we just committed.
        self._p1 = None
        self._inprogress_artists = []
        self.point_added.emit(2)
        if motion is not None:
            self.motion_measured.emit(motion)

    def _handle_right_click(self, event) -> None:
        """Delete the track whose artist was right-clicked. Hit-tests every
        artist of every track; first hit wins."""
        if not self._tracks:
            return
        for i in range(len(self._tracks) - 1, -1, -1):   # newest first
            track = self._tracks[i]
            for artist in track.artists:
                try:
                    contains, _info = artist.contains(event)
                except (AttributeError, ValueError):
                    continue
                if contains:
                    for a in track.artists:
                        try:
                            a.remove()
                        except (ValueError, AttributeError):
                            pass
                    del self._tracks[i]
                    event.canvas.draw_idle()
                    return

    # ---- drawing --------------------------------------------------------

    def _draw_marker(self, ax, x: float, y: float):
        m, = ax.plot([x], [y], marker="x", markersize=10, color="#ffff00",
                     markeredgewidth=2, zorder=20)
        ax.figure.canvas.draw_idle()
        return m

    def _compute_motion(self, p1, p2) -> StormMotion | None:
        x1, y1, t1 = p1
        x2, y2, t2 = p2
        site = self._grid.site
        from ..geo.projection import xy_km_to_latlon
        lat1, lon1 = xy_km_to_latlon(x1, y1, site.lat, site.lon)
        lat2, lon2 = xy_km_to_latlon(x2, y2, site.lat, site.lon)
        return storm_motion_from_two_points(lat1, lon1, t1, lat2, lon2, t2)

    def _draw_arrow_and_label(self, ax, p1, p2, motion: StormMotion | None) -> list:
        x1, y1, _t1 = p1
        x2, y2, _t2 = p2
        out: list = []
        arrow = ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#ffff00", lw=2),
            zorder=20,
        )
        out.append(arrow)
        if motion is not None:
            label = ax.text(
                (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                f"  {motion}",
                color="#ffff00", fontsize=10, fontweight="bold",
                zorder=21,
            )
            out.append(label)
        return out

    def _draw_forecast_track(self, ax, p1, p2,
                              *, minutes: tuple[int, ...] = (15, 30, 45, 60),
                              tick_len_km: float = 8.0) -> list:
        """Extrapolate the storm's motion forward from P2 and draw a dashed
        forecast track with perpendicular tick marks every ``minutes``
        interval. Provides a visual ruler for sizing warning polygons.
        Returns the list of artists drawn so they can be tracked with the
        rest of the track group.
        """
        x1, y1, t1 = p1
        x2, y2, t2 = p2
        out: list = []
        dt = t2 - t1
        if dt <= 0:
            return out
        vx = (x2 - x1) / dt
        vy = (y2 - y1) / dt
        speed_km_s = (vx * vx + vy * vy) ** 0.5
        if speed_km_s == 0:
            return out
        perp_x = -vy / speed_km_s
        perp_y = vx / speed_km_s

        # Forecast track line from P2 to the furthest tick.
        max_sec = max(minutes) * 60
        track_x = (x2, x2 + vx * max_sec)
        track_y = (y2, y2 + vy * max_sec)
        track_line, = ax.plot(track_x, track_y, color="#ffff00",
                               linewidth=1.4, linestyle="--", alpha=0.7,
                               zorder=19)
        out.append(track_line)

        # Perpendicular ticks at each interval + minute label.
        for m in minutes:
            sec = m * 60
            tx = x2 + vx * sec
            ty = y2 + vy * sec
            tick_xs = (tx - tick_len_km * perp_x, tx + tick_len_km * perp_x)
            tick_ys = (ty - tick_len_km * perp_y, ty + tick_len_km * perp_y)
            tick, = ax.plot(tick_xs, tick_ys, color="#ffff00",
                            linewidth=1.6, zorder=20)
            out.append(tick)
            label = ax.text(
                tx + tick_len_km * 1.4 * perp_x,
                ty + tick_len_km * 1.4 * perp_y,
                f"+{m}m",
                color="#ffff00", fontsize=8, fontweight="bold",
                ha="center", va="center", zorder=21,
            )
            out.append(label)
        return out

    def _discard_inprogress(self) -> None:
        """Remove any in-progress P1 marker (called when the user
        deactivates the tool without finishing a measurement)."""
        for a in self._inprogress_artists:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._inprogress_artists.clear()
        self._p1 = None
        for panel in self._grid._panels:
            panel._canvas.draw_idle()

    def _focused_panel(self):
        return self._grid._panels[self._grid._focused_panel_index()]
