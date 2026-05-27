"""Polygon math for warning verification.

A "polygon" in this codebase is a list of ``(lat, lon)`` vertices defining a
simple, non-self-intersecting closed ring. For verification we need:

  - Point-in-polygon test with a **5 km buffer** (Minkowski dilation), per plan §6.
  - Area in km² (for the MCD anti-spam cap).
  - Union of polygons (team-mode aggregation, per plan §11).

We use Shapely for geometry. Since we operate at the scale of one game polygon
(~few hundred km), we project to a local equirectangular tangent plane centered
on the polygon's centroid for buffer/area operations — much cheaper than a
real geodesic library and accurate to <1 % at this scale.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from shapely.geometry import MultiPolygon, Point, Polygon as ShapelyPolygon
from shapely.ops import unary_union

from .projection import latlon_to_xy_km, xy_km_to_latlon

LatLon = tuple[float, float]
Ring = Sequence[LatLon]


@dataclass(frozen=True)
class Polygon:
    """A geographic polygon defined as a list of ``(lat, lon)`` vertices.

    Closure is implicit: the last vertex need not equal the first. Vertices
    should be in either order (CW or CCW); we normalize internally.
    """

    vertices: tuple[LatLon, ...]

    def __post_init__(self) -> None:
        if len(self.vertices) < 3:
            raise ValueError("Polygon needs at least 3 vertices")

    @property
    def centroid_latlon(self) -> LatLon:
        lat = sum(v[0] for v in self.vertices) / len(self.vertices)
        lon = sum(v[1] for v in self.vertices) / len(self.vertices)
        return (lat, lon)

    def shapely_local(self, lat0: float | None = None, lon0: float | None = None) -> ShapelyPolygon:
        """Return a Shapely polygon in km from ``(lat0, lon0)`` (default: own centroid)."""
        if lat0 is None or lon0 is None:
            lat0, lon0 = self.centroid_latlon
        xy = [latlon_to_xy_km(lat, lon, lat0, lon0) for lat, lon in self.vertices]
        return ShapelyPolygon(xy)


def contains_with_buffer(
    polygon: Polygon,
    lat: float,
    lon: float,
    buffer_km: float = 5.0,
) -> bool:
    """True if ``(lat, lon)`` lies inside ``polygon`` dilated by ``buffer_km`` (Minkowski sum).

    Buffer is applied in the local equirectangular plane centered on the polygon's
    centroid, which is accurate to <1 % at typical warning-polygon scales.
    """
    lat0, lon0 = polygon.centroid_latlon
    poly_xy = polygon.shapely_local(lat0, lon0).buffer(buffer_km)
    px, py = latlon_to_xy_km(lat, lon, lat0, lon0)
    return poly_xy.contains(Point(px, py)) or poly_xy.touches(Point(px, py))


def polygon_area_km2(polygon: Polygon) -> float:
    """Polygon area in km² (using local equirectangular projection)."""
    return polygon.shapely_local().area


def polygon_fraction_of(inner: Polygon, outer: Polygon) -> float:
    """Fraction of ``outer``'s area covered by ``inner ∩ outer``.

    General utility — useful for "how much of the game polygon does this
    warning cover?" coverage metrics in scoring breakdowns. (The MCD
    anti-spam cap used to live here but was switched to an absolute
    km² maximum in :data:`session.MCD_MAX_AREA_KM2`.)
    Both polygons are projected to ``outer``'s centroid for consistency.
    """
    lat0, lon0 = outer.centroid_latlon
    o = outer.shapely_local(lat0, lon0)
    i = inner.shapely_local(lat0, lon0)
    if o.area == 0:
        return 0.0
    inter = o.intersection(i)
    return inter.area / o.area


def polygon_union(polygons: Iterable[Polygon]) -> list[Polygon]:
    """Geometric union of multiple polygons.

    Returns a list because the union may be disjoint (one Polygon per component).
    All inputs and outputs share a single reference centroid for projection.
    Used by team-mode scoring: a team's coverage = union of all teammates' polygons.
    """
    polygons = list(polygons)
    if not polygons:
        return []
    # Use the first polygon's centroid as the common reference for all projections.
    lat0, lon0 = polygons[0].centroid_latlon
    locals_ = [p.shapely_local(lat0, lon0) for p in polygons]
    merged = unary_union(locals_)
    return _shapely_to_polygons(merged, lat0, lon0)


def buffered_union_contains(
    polygons: Iterable[Polygon],
    lat: float,
    lon: float,
    buffer_km: float = 5.0,
) -> bool:
    """True if ``(lat, lon)`` lies inside the buffered union of ``polygons``.

    Faster than building the union as Polygons and re-buffering: we do the
    buffer once on the merged Shapely geometry.
    """
    polygons = list(polygons)
    if not polygons:
        return False
    lat0, lon0 = polygons[0].centroid_latlon
    locals_ = [p.shapely_local(lat0, lon0) for p in polygons]
    merged = unary_union(locals_).buffer(buffer_km)
    px, py = latlon_to_xy_km(lat, lon, lat0, lon0)
    return merged.contains(Point(px, py)) or merged.touches(Point(px, py))


def _shapely_to_polygons(
    geom,
    lat0: float,
    lon0: float,
) -> list[Polygon]:
    """Convert a Shapely (Multi)Polygon in local km back to our Polygon list."""
    if geom.is_empty:
        return []
    if isinstance(geom, ShapelyPolygon):
        return [_shapely_ring_to_polygon(geom, lat0, lon0)]
    if isinstance(geom, MultiPolygon):
        return [_shapely_ring_to_polygon(g, lat0, lon0) for g in geom.geoms]
    # GeometryCollection or other — pull out any polygonal components.
    return [
        _shapely_ring_to_polygon(g, lat0, lon0)
        for g in getattr(geom, "geoms", [])
        if isinstance(g, ShapelyPolygon)
    ]


def _shapely_ring_to_polygon(poly: ShapelyPolygon, lat0: float, lon0: float) -> Polygon:
    coords = list(poly.exterior.coords)
    if coords and coords[0] == coords[-1]:
        coords = coords[:-1]
    vertices = tuple(xy_km_to_latlon(x, y, lat0, lon0) for x, y in coords)
    return Polygon(vertices=vertices)
