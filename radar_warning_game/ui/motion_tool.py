"""Two-point storm-motion measurement tool (plan §5a).

Attaches to a :class:`RadarPanelGrid`. While active, the next two left-clicks
on any radar panel drop markers at scan-time-stamped points; the tool
computes the storm's bearing (TO + FROM), speed in kt, and displays the
NWS-style annotation plus an arrow from P1 → P2 and a dashed forecast track
with 15-min tick marks.

Each completed track is **persistent** — it stays drawn until the player
right-clicks on any of its artists to delete it. The player can keep
left-clicking to drop additional tracks alongside existing ones. This is
**purely informational** (plan §5a) — tracks do NOT auto-generate warning
polygons.

Scrubbing while the tool is active uses the same controls as normal
(``←/→``, ``Shift+←/→``, ``↑/↓``), and the per-player game-clock cap still
applies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, pyqtSignal

from ..geo.projection import StormMotion, storm_motion_from_two_points

log = logging.getLogger(__name__)


@dataclass
class _StormTrack:
    """One completed two-point measurement and all its pyqtgraph items.

    Held in :attr:`MotionTool._tracks` so the player can keep multiple
    measurements on screen at once and right-click any of them to remove
    just that track.
    """

    p1: tuple[float, float, float]    # (x_km, y_km, t_sec)
    p2: tuple[float, float, float]
    motion: StormMotion | None
    view: pg.ViewBox                  # the view they were drawn on
    items: list = field(default_factory=list)


class MotionTool(QObject):
    """Mouse-click motion measurement controller.

    Lifecycle::

        tool = MotionTool(grid)
        tool.activate()        # enable left-click point-drops
        # ... player clicks two points for each track ...
        tool.deactivate()      # stop listening; tracks remain visible

    Right-clicking on any drawn track (in any state) removes just that track.

    Signals
    -------
    motion_measured(StormMotion)
        emitted when the second click of a track lands
    point_added(int)
        emitted with 1 after first click of a track, 2 after second
    cleared()
        emitted when all tracks are wiped (e.g. layout change)
    """

    motion_measured = pyqtSignal(object)   # StormMotion
    point_added = pyqtSignal(int)
    cleared = pyqtSignal()
    error = pyqtSignal(str)                # user-visible error (e.g. same-sweep clicks)

    def __init__(self, grid) -> None:
        super().__init__()
        self._grid = grid
        # ``_active`` only gates *new-point* left-clicks. Right-click track
        # deletion stays armed whenever the tool's scene-click handler is
        # installed (i.e. after the first activate).
        self._active = False
        self._p1: tuple[float, float, float] | None = None
        self._p1_view: pg.ViewBox | None = None
        self._inprogress_items: list = []
        self._tracks: list[_StormTrack] = []
        # Connections to the per-panel scene().sigMouseClicked signal.
        # Stored as (view, connection) pairs so we can disconnect cleanly.
        self._connections: list[tuple[pg.ViewBox, object]] = []
        self._handlers_installed = False

    @property
    def is_active(self) -> bool:
        return self._active

    def activate(self) -> None:
        self._active = True
        self._ensure_handlers_installed()

    def deactivate(self) -> None:
        if not self._active:
            return
        self._active = False
        if self._p1 is not None:
            self._discard_inprogress()

    def _ensure_handlers_installed(self) -> None:
        if self._handlers_installed:
            return
        for panel in self._grid._panels:
            view = panel.view
            scene = view.scene()
            scene.sigMouseClicked.connect(self._on_scene_clicked)
            self._connections.append((view, scene))
        self._handlers_installed = True

    def reinstall_handlers_for_new_panels(self) -> None:
        """Called after the grid rebuilds its panels. Old views are dead;
        clear stale tracks and rewire to the new scenes."""
        # Stale track items belong to deleted views — drop them.
        self._tracks.clear()
        self._p1 = None
        self._inprogress_items.clear()
        # Old connections will be garbage-collected with their scenes; no
        # explicit disconnect needed.
        self._connections.clear()
        self._handlers_installed = False
        self._ensure_handlers_installed()

    def reset(self) -> None:
        """Cancel an in-progress measurement without disturbing committed
        tracks."""
        if self._p1 is not None:
            self._discard_inprogress()
            self.cleared.emit()

    def clear_all(self) -> None:
        """Remove every track (and any in-progress P1)."""
        for track in self._tracks:
            for item in track.items:
                try:
                    track.view.removeItem(item)
                except Exception:  # noqa: BLE001
                    pass
        self._tracks.clear()
        if self._p1 is not None:
            self._discard_inprogress()
        self.cleared.emit()

    def motion(self) -> StormMotion | None:
        if not self._tracks:
            return None
        return self._tracks[-1].motion

    # ---- event handler ----------------------------------------------

    def _on_scene_clicked(self, ev) -> None:
        # Figure out which panel's view received this click, since the
        # scene fires for the whole graphics scene the panel is in. Only
        # one panel's view contains the click location.
        view = self._view_under(ev.scenePos())
        if view is None:
            return
        # Right-click: delete a track hit-test in this view.
        if ev.button() == Qt.MouseButton.RightButton:
            self._handle_right_click(view, ev)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        if not self._active:
            return
        try:
            pt = view.mapSceneToView(ev.scenePos())
        except Exception:  # noqa: BLE001
            return
        x = float(pt.x())
        y = float(pt.y())
        if self._grid._current_sweep is None:
            return
        t_sec = self._grid._current_sweep.start_time.timestamp()
        point = (x, y, t_sec)
        if self._p1 is None:
            self._p1 = point
            self._p1_view = view
            marker = self._draw_marker(view, x, y)
            self._inprogress_items.append(marker)
            self.point_added.emit(1)
            ev.accept()
            return
        # Second click → commit a track on the same view as P1.
        if view is not self._p1_view:
            # Switching panels mid-track is awkward; ignore.
            return
        if point[2] == self._p1[2]:
            self.error.emit("Scrub time forward (← →) before clicking point 2.")
            return
        p2 = point
        items = list(self._inprogress_items)
        items.append(self._draw_marker(view, p2[0], p2[1]))
        motion_obj = self._compute_motion(self._p1, p2)
        items.extend(self._draw_arrow_and_label(view, self._p1, p2, motion_obj))
        items.extend(self._draw_forecast_track(view, self._p1, p2))
        track = _StormTrack(p1=self._p1, p2=p2, motion=motion_obj,
                            view=view, items=items)
        self._tracks.append(track)
        self._p1 = None
        self._p1_view = None
        self._inprogress_items = []
        self.point_added.emit(2)
        if motion_obj is not None:
            self.motion_measured.emit(motion_obj)
        ev.accept()

    def _view_under(self, scene_pos) -> pg.ViewBox | None:
        for panel in self._grid._panels:
            if panel.view.sceneBoundingRect().contains(scene_pos):
                return panel.view
        return None

    def _handle_right_click(self, view: pg.ViewBox, ev) -> None:
        if not self._tracks:
            return
        try:
            pt = view.mapSceneToView(ev.scenePos())
        except Exception:  # noqa: BLE001
            return
        x, y = float(pt.x()), float(pt.y())
        # Hit-test in this view, newest tracks first.
        # Loose radius scaled to view size.
        view_w = view.viewRange()[0][1] - view.viewRange()[0][0]
        hit_radius = view_w * 0.05
        for i in range(len(self._tracks) - 1, -1, -1):
            track = self._tracks[i]
            if track.view is not view:
                continue
            # Check distance to each leg of the arrow (P1, P2, midpoint).
            p1x, p1y, _ = track.p1
            p2x, p2y, _ = track.p2
            midx = (p1x + p2x) / 2.0
            midy = (p1y + p2y) / 2.0
            d = min(
                np.hypot(x - p1x, y - p1y),
                np.hypot(x - p2x, y - p2y),
                np.hypot(x - midx, y - midy),
            )
            if d < hit_radius:
                for item in track.items:
                    try:
                        track.view.removeItem(item)
                    except Exception:  # noqa: BLE001
                        pass
                del self._tracks[i]
                ev.accept()
                return

    # ---- drawing ----------------------------------------------------

    def _draw_marker(self, view: pg.ViewBox, x: float, y: float):
        scatter = pg.ScatterPlotItem(
            x=[x], y=[y], size=12, symbol="x",
            pen=pg.mkPen("#ffff00", width=2.0),
            brush=pg.mkBrush("#ffff00"),
            pxMode=True,
        )
        scatter.setZValue(20)
        view.addItem(scatter)
        return scatter

    def _compute_motion(self, p1, p2) -> StormMotion | None:
        x1, y1, t1 = p1
        x2, y2, t2 = p2
        site = self._grid.site
        from ..geo.projection import xy_km_to_latlon
        lat1, lon1 = xy_km_to_latlon(x1, y1, site.lat, site.lon)
        lat2, lon2 = xy_km_to_latlon(x2, y2, site.lat, site.lon)
        return storm_motion_from_two_points(lat1, lon1, t1, lat2, lon2, t2)

    def _draw_arrow_and_label(
        self, view: pg.ViewBox, p1, p2, motion_obj: StormMotion | None,
    ) -> list:
        x1, y1, _t1 = p1
        x2, y2, _t2 = p2
        out: list = []
        # Arrow as a line + arrowhead (pg.ArrowItem at the tip).
        line = pg.PlotCurveItem(
            x=[x1, x2], y=[y1, y2],
            pen=pg.mkPen("#ffff00", width=2),
        )
        line.setZValue(20)
        view.addItem(line)
        out.append(line)
        # Arrowhead at P2 pointing along the segment.
        ang_rad = np.arctan2(y2 - y1, x2 - x1)
        # ArrowItem expects degrees with 0=right, going CCW.
        ang_deg = np.degrees(ang_rad) + 180.0   # arrow tail points back along segment
        arrow = pg.ArrowItem(
            angle=ang_deg, tipAngle=30, baseAngle=20, headLen=14,
            tailLen=None, brush=pg.mkBrush("#ffff00"),
            pen=pg.mkPen("#ffff00"),
        )
        arrow.setPos(x2, y2)
        arrow.setZValue(20)
        view.addItem(arrow)
        out.append(arrow)
        if motion_obj is not None:
            label = pg.TextItem(
                f" {motion_obj}", anchor=(0, 0.5), color="#ffff00",
            )
            label.setPos((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            label.setZValue(21)
            view.addItem(label, ignoreBounds=True)
            out.append(label)
        return out

    def _draw_forecast_track(
        self, view: pg.ViewBox, p1, p2,
        *, minutes: tuple[int, ...] = (15, 30, 45, 60),
        tick_len_km: float = 8.0,
    ) -> list:
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

        max_sec = max(minutes) * 60
        track_xs = [x2, x2 + vx * max_sec]
        track_ys = [y2, y2 + vy * max_sec]
        track_line = pg.PlotCurveItem(
            x=track_xs, y=track_ys,
            pen=pg.mkPen("#ffff00", width=1.4, style=Qt.PenStyle.DashLine),
        )
        track_line.setZValue(19)
        view.addItem(track_line)
        out.append(track_line)

        for m in minutes:
            sec = m * 60
            tx = x2 + vx * sec
            ty = y2 + vy * sec
            tick_xs = [tx - tick_len_km * perp_x, tx + tick_len_km * perp_x]
            tick_ys = [ty - tick_len_km * perp_y, ty + tick_len_km * perp_y]
            tick = pg.PlotCurveItem(
                x=tick_xs, y=tick_ys,
                pen=pg.mkPen("#ffff00", width=1.6),
            )
            tick.setZValue(20)
            view.addItem(tick)
            out.append(tick)
            label = pg.TextItem(f"+{m}m", anchor=(0.5, 0.5), color="#ffff00")
            label.setPos(tx + tick_len_km * 1.4 * perp_x,
                         ty + tick_len_km * 1.4 * perp_y)
            label.setZValue(21)
            view.addItem(label, ignoreBounds=True)
            out.append(label)
        return out

    def _discard_inprogress(self) -> None:
        if self._p1_view is not None:
            for item in self._inprogress_items:
                try:
                    self._p1_view.removeItem(item)
                except Exception:  # noqa: BLE001
                    pass
        self._inprogress_items.clear()
        self._p1 = None
        self._p1_view = None
