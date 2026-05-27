"""Freehand polygon drawing on a pyqtgraph ViewBox (plan §2, §5).

Click-to-add-vertex, right-click-to-remove-nearest, drag-to-move-vertex.
Used by:
  - the CONUS overview map for the game-area polygon (§2)
  - the radar panel for warning polygons (§5)

The editor operates in the view's native coordinate system. For the radar
panel that's km from the radar; for the CONUS map it's plain ``(lon, lat)``.
The returned :class:`Polygon` is always in ``(lat, lon)`` — callers pass an
``axes_to_latlon`` function that maps from view coords back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import QObject, Qt, pyqtSignal

from ..geo.polygons import Polygon

log = logging.getLogger(__name__)


class PolygonEditor(QObject):
    """Click-to-draw polygon editor wired into one *or more* pyqtgraph
    ViewBoxes.

    Emits :attr:`polygon_changed` with the updated :class:`Polygon` (or
    ``None`` if fewer than 3 vertices) every time a vertex is added or
    removed. Caller decides when to "commit" — for the CONUS game-area
    picker that's a "Confirm" button, for a warning issuance that's the
    "Issue" button.

    Multi-view support is what makes the radar grid usable: when the
    host opens a 4-panel layout, the editor should accept clicks on
    **any** panel and mirror the in-flight polygon outline + vertex
    markers across all four. Pass either a single :class:`pyqtgraph.ViewBox`
    (for the CONUS overview map, which is single-view) or a list of
    them (for the radar grid).

    Parameters
    ----------
    view : a :class:`pyqtgraph.ViewBox` or list of them. All views are
        assumed to share the same data coordinate system — the editor's
        vertex list is in the *shared* coord space and is rendered
        identically on each.
    axes_to_latlon : function that converts ``(x, y)`` view coordinates to
        ``(lat, lon)``. For a radar-centric km axes this projects through
        the radar site; for a plain lat/lon map it's ``lambda x, y: (y, x)``.
    color : outline / marker color.
    """

    polygon_changed = pyqtSignal(object)   # emits Polygon or None

    def __init__(
        self,
        view,   # pg.ViewBox | list[pg.ViewBox]
        axes_to_latlon: Callable[[float, float], tuple[float, float]],
        *,
        color: str = "#ffd400",
        linewidth: float = 1.6,
        marker_size: int = 10,
    ) -> None:
        super().__init__()
        # Normalize to a list of views — single-view callers (the
        # CONUS map) still work via this list path. ``self.view``
        # stays as the first/primary view for back-compat with any
        # external code that reaches for it.
        if isinstance(view, list):
            self.views: list[pg.ViewBox] = list(view)
        else:
            self.views = [view]
        self.view = self.views[0]
        self._axes_to_latlon = axes_to_latlon
        self._color = color
        self._linewidth = linewidth
        self._marker_size = marker_size
        self._enabled = True
        self._vertices: list[tuple[float, float]] = []   # in view coords (x, y)

        # One pair of persistent artists per view so the in-flight
        # outline + vertex markers appear on every panel as the user
        # builds the polygon. ``_refresh_artists`` updates them in
        # lockstep so the panels stay visually identical.
        self._outlines: list[pg.PlotCurveItem] = []
        self._markers_items: list[pg.ScatterPlotItem] = []
        for v in self.views:
            outline = pg.PlotCurveItem(
                x=[], y=[], pen=pg.mkPen(color=color, width=linewidth),
            )
            outline.setZValue(15)
            v.addItem(outline)
            self._outlines.append(outline)
            markers = pg.ScatterPlotItem(
                x=[], y=[], size=marker_size,
                pen=pg.mkPen("white", width=1.0),
                brush=pg.mkBrush(color),
                pxMode=True,
            )
            markers.setZValue(16)
            v.addItem(markers)
            self._markers_items.append(markers)
        # Back-compat aliases used by some tests / external callers
        # that look for the singular ``_outline`` / ``_markers``.
        self._outline = self._outlines[0]
        self._markers = self._markers_items[0]

        # Subscribe to EVERY view's scene so a click on any panel adds
        # a vertex. The lambda captures the originating view so we can
        # map the scene-space click back to data coordinates via that
        # view's projection — synced views share the same data range
        # so the resulting (x, y) is identical regardless of which
        # panel the user clicked.
        for v in self.views:
            v.scene().sigMouseClicked.connect(
                lambda ev, src=v: self._on_scene_clicked(ev, src)
            )

    # ---- public API --------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, enabled: bool) -> None:
        """When disabled, left-clicks no longer add vertices. The existing
        polygon stays visible. Right-click vertex deletion is suspended too."""
        self._enabled = enabled

    def vertices_axes(self) -> np.ndarray:
        if not self._vertices:
            return np.empty((0, 2))
        return np.asarray(self._vertices, dtype=np.float64)

    def vertices_latlon(self) -> list[tuple[float, float]]:
        return [self._axes_to_latlon(float(x), float(y)) for x, y in self._vertices]

    def polygon(self) -> Polygon | None:
        verts = self.vertices_latlon()
        if len(verts) < 3:
            return None
        return Polygon(vertices=tuple(verts))

    def clear(self) -> None:
        self._vertices.clear()
        self._refresh_artists()
        self.polygon_changed.emit(None)

    def dispose(self) -> None:
        """Remove every outline + marker artist from every view's scene
        and forget them. Use this when the editor is permanently done
        (e.g. user finished or canceled a draw) — ``clear()`` only
        empties the data on the existing artists, which leaves them
        attached to the scenes. If a subsequent action then re-uses
        the same artist instances (or a child item type that pyqtgraph
        re-paints), the cleared-but-still-attached artists can flicker
        previous content back into view. ``dispose()`` makes the
        cleanup hermetic — after this call the editor is dead, do not
        use it further."""
        # Empty data first (defensive against any held reference).
        self._vertices.clear()
        for v, outline in zip(self.views, self._outlines):
            try:
                if outline.scene() is v.scene():
                    v.removeItem(outline)
            except (RuntimeError, AttributeError):
                pass
        for v, markers in zip(self.views, self._markers_items):
            try:
                if markers.scene() is v.scene():
                    v.removeItem(markers)
            except (RuntimeError, AttributeError):
                pass
        self._outlines.clear()
        self._markers_items.clear()

    def set_polygon(
        self,
        polygon: Polygon,
        latlon_to_axes: Callable[[float, float], tuple[float, float]],
    ) -> None:
        self._vertices = [latlon_to_axes(lat, lon) for lat, lon in polygon.vertices]
        self._refresh_artists()
        self.polygon_changed.emit(self.polygon())

    # ---- mouse handling ---------------------------------------------

    def _on_scene_clicked(self, ev, src_view: "pg.ViewBox | None" = None) -> None:
        if not self._enabled:
            return
        # The originating view is passed in by the lambda we registered
        # per-view in __init__; fall back to the primary if a caller
        # invokes this directly (tests / introspection).
        view = src_view if src_view is not None else self.view
        if not view.sceneBoundingRect().contains(ev.scenePos()):
            return
        try:
            data_pt = view.mapSceneToView(ev.scenePos())
        except Exception:  # noqa: BLE001
            return
        x, y = float(data_pt.x()), float(data_pt.y())
        if ev.button() == Qt.MouseButton.LeftButton:
            # Add a vertex at the click position.
            self._vertices.append((x, y))
            self._refresh_artists()
            self.polygon_changed.emit(self.polygon())
            ev.accept()
        elif ev.button() == Qt.MouseButton.RightButton:
            # Remove the nearest vertex (if any within a generous radius).
            if not self._vertices:
                return
            idx = self._nearest_vertex(x, y)
            if idx is None:
                return
            del self._vertices[idx]
            self._refresh_artists()
            self.polygon_changed.emit(self.polygon())
            ev.accept()

    def _nearest_vertex(self, x: float, y: float) -> int | None:
        """Index of the vertex closest to ``(x, y)`` in view coords, or None."""
        if not self._vertices:
            return None
        arr = np.asarray(self._vertices)
        d = np.hypot(arr[:, 0] - x, arr[:, 1] - y)
        idx = int(np.argmin(d))
        # Loose hit radius proportional to view width — generous so the
        # user doesn't have to click on the dot exactly.
        view_w = self.view.viewRange()[0][1] - self.view.viewRange()[0][0]
        if d[idx] > view_w * 0.04:
            return None
        return idx

    # ---- rendering ---------------------------------------------------

    def _refresh_artists(self) -> None:
        """Push the current vertex list to every view's outline +
        markers so the in-flight polygon mirrors across all panels."""
        verts = self.vertices_axes()
        if len(verts) >= 1:
            marker_x, marker_y = list(verts[:, 0]), list(verts[:, 1])
        else:
            marker_x, marker_y = [], []
        if len(verts) >= 2:
            xs = list(verts[:, 0])
            ys = list(verts[:, 1])
            if len(verts) >= 3:
                xs.append(xs[0])
                ys.append(ys[0])
            outline_x, outline_y = xs, ys
        else:
            outline_x, outline_y = [], []
        for markers in self._markers_items:
            markers.setData(x=marker_x, y=marker_y)
        for outline in self._outlines:
            outline.setData(x=outline_x, y=outline_y)
