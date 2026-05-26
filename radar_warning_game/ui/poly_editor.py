"""Freehand polygon drawing on a matplotlib axes (plan §2, §5).

Built on top of ``mpl_point_clicker`` — that library provides the click handlers
for adding/removing vertices; we layer a closed-polygon outline + signal hooks
on top so the rest of the app sees a clean ``PolygonEditor`` API.

Used by:
  - the CONUS overview map for the game-area polygon (§2)
  - the radar panel for warning polygons (§5)

The editor operates in the axes' native coordinate system. For the radar panel
that's km from the radar; for the CONUS map it's PlateCarree (lon, lat). The
returned :class:`Polygon` is in (lat, lon) regardless — callers pass an
``axes_to_latlon`` function that maps from axis coords back.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
from mpl_point_clicker import clicker
from PyQt6.QtCore import QObject, pyqtSignal

from ..geo.polygons import Polygon

log = logging.getLogger(__name__)


class PolygonEditor(QObject):
    """Attach click-to-draw polygon editing to any matplotlib axes.

    Emits :attr:`polygon_changed` with the updated :class:`Polygon` (or ``None``
    if fewer than 3 vertices) every time a vertex is added or removed. Caller
    decides when to "commit" — for the CONUS game-area picker that's a "Confirm"
    button, for a warning issuance that's the "Issue" button.

    Parameters
    ----------
    ax : matplotlib axes the polygon is drawn on.
    axes_to_latlon : function that converts an ``(x, y)`` axes coordinate to a
        ``(lat, lon)`` pair. For a radar-centric km axes this projects through
        the radar site; for a PlateCarree map it's ``lambda x, y: (y, x)``.
    color : outline color (per-player or neutral white).
    """

    polygon_changed = pyqtSignal(object)   # emits Polygon or None

    def __init__(
        self,
        ax,
        axes_to_latlon: Callable[[float, float], tuple[float, float]],
        *,
        color: str = "#ffd400",
        linewidth: float = 1.6,
        marker: str = "o",
        marker_size: int = 60,
    ) -> None:
        super().__init__()
        self.ax = ax
        self._axes_to_latlon = axes_to_latlon
        self._color = color
        self._linewidth = linewidth
        self._clicker = clicker(
            ax,
            classes=["poly"],
            markers=[marker],
            colors=[color],
            disable_legend=True,
        )
        self._outline_line = None
        self._clicker.on_point_added(self._on_changed)
        self._clicker.on_point_removed(self._on_changed)
        self._clicker.on_positions_set(self._on_changed)

    # ---- public API ------------------------------------------------------

    def vertices_axes(self) -> np.ndarray:
        """Vertex array in axes coords, shape ``(N, 2)``."""
        positions = self._clicker.get_positions()
        return positions.get("poly", np.empty((0, 2)))

    def vertices_latlon(self) -> list[tuple[float, float]]:
        """Vertex list in ``(lat, lon)`` order, suitable for :class:`Polygon`."""
        return [self._axes_to_latlon(float(x), float(y)) for x, y in self.vertices_axes()]

    def polygon(self) -> Polygon | None:
        """Return a closed :class:`Polygon`, or ``None`` if <3 vertices yet."""
        verts = self.vertices_latlon()
        if len(verts) < 3:
            return None
        return Polygon(vertices=tuple(verts))

    def clear(self) -> None:
        self._clicker.clear_positions()
        self._redraw_outline()

    def set_polygon(self, polygon: Polygon, latlon_to_axes: Callable[[float, float], tuple[float, float]]) -> None:
        """Load an existing polygon (for editing an active warning, etc.)."""
        pts = np.array([latlon_to_axes(lat, lon) for lat, lon in polygon.vertices])
        self._clicker.set_positions({"poly": pts})
        self._redraw_outline()

    # ---- internal -------------------------------------------------------

    def _on_changed(self, *_args: Any) -> None:
        self._redraw_outline()
        self.polygon_changed.emit(self.polygon())

    def _redraw_outline(self) -> None:
        if self._outline_line is not None:
            try:
                self._outline_line.remove()
            except (ValueError, AttributeError):
                pass
            self._outline_line = None
        verts = self.vertices_axes()
        if len(verts) >= 2:
            # Close back to first vertex if 3+ points
            xs = list(verts[:, 0])
            ys = list(verts[:, 1])
            if len(verts) >= 3:
                xs.append(xs[0])
                ys.append(ys[0])
            self._outline_line, = self.ax.plot(
                xs, ys, color=self._color, linewidth=self._linewidth, zorder=10,
            )
        self.ax.figure.canvas.draw_idle()
