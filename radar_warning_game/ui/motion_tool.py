"""Two-point storm-motion measurement tool (plan §5a).

Attaches to a :class:`RadarPanelGrid`. While active, the next two clicks on the
focused radar panel drop markers at scan-time-stamped points; the tool then
computes the storm's bearing (TO + FROM), speed in kt, and displays the
NWS-style annotation plus an arrow from P1 → P2.

This is **purely informational** (plan §5a) — it does NOT auto-generate warning
polygons. The player uses the measured motion to inform how far downstream to
draw their hand-drawn warning polygon.

Scrubbing while the tool is active uses the same controls as normal (``←/→``,
``Shift+←/→``, ``↑/↓``), and the per-player game-clock cap still applies.
"""

from __future__ import annotations

import logging

from PyQt6.QtCore import QObject, pyqtSignal

from ..data.sites import Site
from ..geo.projection import StormMotion, storm_motion_from_two_points

log = logging.getLogger(__name__)


class MotionTool(QObject):
    """Mouse-click motion measurement controller.

    Lifecycle::

        tool = MotionTool(grid)
        tool.activate()        # next two clicks register points
        # ... player clicks twice ...
        tool.motion()          # → StormMotion or None
        tool.deactivate()      # stop listening, clear markers

    Signals
    -------
    motion_measured(StormMotion)
        emitted when the second click lands
    point_added(int)
        emitted with 1 after first click, 2 after second
    cleared()
        emitted when markers are cleared
    """

    motion_measured = pyqtSignal(object)   # StormMotion
    point_added = pyqtSignal(int)
    cleared = pyqtSignal()
    error = pyqtSignal(str)                # user-visible error (e.g. same-sweep clicks)

    def __init__(self, grid) -> None:
        super().__init__()
        self._grid = grid
        self._active = False
        self._p1: tuple[float, float, float] | None = None  # (x_km, y_km, t_sec)
        self._p2: tuple[float, float, float] | None = None
        self._markers: list = []
        self._arrow = None
        self._cid_press: int | None = None

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        """Begin listening for clicks on the focused panel."""
        if self._active:
            return
        self._active = True
        # Attach a click handler to every panel — focus determines which gets clicks
        panel = self._focused_panel()
        canvas = panel._canvas
        self._cid_press = canvas.mpl_connect("button_press_event", self._on_click)

    def deactivate(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._cid_press is not None:
            self._focused_panel()._canvas.mpl_disconnect(self._cid_press)
            self._cid_press = None

    def reset(self) -> None:
        """Clear both points; ready to measure again."""
        self._p1 = self._p2 = None
        self._clear_artists()
        self.cleared.emit()

    def motion(self) -> StormMotion | None:
        if self._p1 is None or self._p2 is None:
            return None
        x1, y1, t1 = self._p1
        x2, y2, t2 = self._p2
        # Convert km-from-radar to lat/lon for projection module
        site = self._grid.site
        from ..geo.projection import xy_km_to_latlon
        lat1, lon1 = xy_km_to_latlon(x1, y1, site.lat, site.lon)
        lat2, lon2 = xy_km_to_latlon(x2, y2, site.lat, site.lon)
        return storm_motion_from_two_points(lat1, lon1, t1, lat2, lon2, t2)

    # ---- event handler --------------------------------------------------

    def _on_click(self, event) -> None:
        if not self._active or event.button != 1:
            return
        # The clicker on overlapping axes may also fire — let it for non-active mode.
        # Use the panel whose axes received the click.
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        if self._grid._current_sweep is None:
            return
        t_sec = self._grid._current_sweep.start_time.timestamp()
        point = (float(event.xdata), float(event.ydata), t_sec)
        if self._p1 is None:
            self._p1 = point
            self._draw_marker(ax, point[0], point[1])
            self.point_added.emit(1)
        elif self._p2 is None:
            # Both clicks on the same sweep → can't compute motion (Δt = 0).
            # Tell the user to scrub forward before clicking again.
            if point[2] == self._p1[2]:
                self.error.emit("Scrub time forward (← →) before clicking point 2.")
                return
            self._p2 = point
            self._draw_marker(ax, point[0], point[1])
            self._draw_arrow(ax)
            self.point_added.emit(2)
            motion = self.motion()
            if motion is not None:
                self.motion_measured.emit(motion)
        else:
            # Third click → start over with the new point as P1
            self.reset()
            self._p1 = point
            self._draw_marker(ax, point[0], point[1])
            self.point_added.emit(1)

    # ---- drawing --------------------------------------------------------

    def _draw_marker(self, ax, x: float, y: float) -> None:
        m, = ax.plot([x], [y], marker="x", markersize=10, color="#ffff00",
                     markeredgewidth=2, zorder=20)
        self._markers.append(m)
        ax.figure.canvas.draw_idle()

    def _draw_arrow(self, ax) -> None:
        if self._p1 is None or self._p2 is None:
            return
        x1, y1, _ = self._p1
        x2, y2, _ = self._p2
        self._arrow = ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#ffff00", lw=2),
            zorder=20,
        )
        motion = self.motion()
        if motion is not None:
            label = ax.text(
                (x1 + x2) / 2.0, (y1 + y2) / 2.0,
                f"  {motion}",
                color="#ffff00", fontsize=10, fontweight="bold",
                zorder=21,
            )
            self._markers.append(label)
        ax.figure.canvas.draw_idle()

    def _clear_artists(self) -> None:
        for a in self._markers:
            try:
                a.remove()
            except (ValueError, AttributeError):
                pass
        self._markers.clear()
        if self._arrow is not None:
            try:
                self._arrow.remove()
            except (ValueError, AttributeError):
                pass
            self._arrow = None
        if self._grid._panels:
            self._grid._panels[0]._canvas.draw_idle()

    def _focused_panel(self):
        return self._grid._panels[self._grid._focused_panel_index()]
