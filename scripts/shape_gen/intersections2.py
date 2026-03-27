from __future__ import annotations

from typing import Tuple, List
import numpy as np


def _dedupe_points_xy(xy: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Merge points closer than eps. Returns unique points in original order (approx).
    """
    if xy.size == 0:
        return xy.reshape(0, 2)

    kept: List[np.ndarray] = []
    for p in xy:
        if not kept:
            kept.append(p)
            continue
        d2 = np.sum((np.vstack(kept) - p) ** 2, axis=1)
        if np.min(d2) > eps**2:
            kept.append(p)
    return np.vstack(kept)


def find_intersection_points_multiple(
    silhouette_xy: np.ndarray,
    occluder_xy: np.ndarray,
    *,
    eps_merge: float = 1e-6,
) -> np.ndarray:
    """
    Python version of your MATLAB polyxpoly-based function.

    Uses:
      - silhouette polyline (closed for safety)
      - occluder treated as a POLYGON, and intersects against its BOUNDARY (closed ring)

    Returns:
      - (0,2) if <2 intersections
      - (2,2) if exactly 2, or if >2 selects the two most separated points
    """
    try:
        from shapely.geometry import (
            LineString,
            Point,
            MultiPoint,
            GeometryCollection,
            Polygon,
            LineString as ShpLineString,
        )
        from shapely.ops import unary_union
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: shapely. Install it to compute intersections."
        ) from e

    sil = np.asarray(silhouette_xy, dtype=np.float64)
    occ = np.asarray(occluder_xy, dtype=np.float64)

    if sil.ndim != 2 or sil.shape[1] != 2:
        raise ValueError(f"silhouette_xy must be Nx2. Got {sil.shape}")
    if occ.ndim != 2 or occ.shape[1] != 2:
        raise ValueError(f"occluder_xy must be Mx2. Got {occ.shape}")
    if sil.shape[0] < 2 or occ.shape[0] < 3:
        return np.zeros((0, 2), dtype=np.float64)

    # Close silhouette for robust wraparound intersection
    if not np.allclose(sil[0], sil[-1]):
        sil = np.vstack([sil, sil[0]])

    sil_ls = LineString(sil.tolist())

    # Treat occluder as polygon and intersect with its boundary (always closed)
    occ_poly = Polygon(occ.tolist())
    if not occ_poly.is_valid:
        # Fix common issues (e.g., minor self-crossing, wrong winding)
        occ_poly = occ_poly.buffer(0)
        if occ_poly.is_empty or (not occ_poly.is_valid):
            return np.zeros((0, 2), dtype=np.float64)

    occ_boundary = occ_poly.boundary
    inter = sil_ls.intersection(occ_boundary)

    pts: List[Tuple[float, float]] = []

    def add_point(p: Point):
        pts.append((float(p.x), float(p.y)))

    def handle_geom(g):
        # Points
        if isinstance(g, Point):
            add_point(g)
            return

        # MultiPoint
        if isinstance(g, MultiPoint):
            for p in g.geoms:
                add_point(p)
            return

        # Line overlap. polyxpoly can return overlaps as multiple points.
        if isinstance(g, ShpLineString):
            coords = list(g.coords)
            if len(coords) >= 2:
                pts.append((float(coords[0][0]), float(coords[0][1])))
                pts.append((float(coords[-1][0]), float(coords[-1][1])))
            return

        # GeometryCollection or MultiLineString-like
        if isinstance(g, GeometryCollection) or hasattr(g, "geoms"):
            for gg in g.geoms:
                handle_geom(gg)
            return

    handle_geom(inter)

    if not pts:
        return np.zeros((0, 2), dtype=np.float64)

    xy = np.array(pts, dtype=np.float64)
    xy = _dedupe_points_xy(xy, eps=eps_merge)

    if xy.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)

    if xy.shape[0] == 2:
        return xy

    # More than 2. Pick two most separated.
    diffs = xy[:, None, :] - xy[None, :, :]
    d2 = np.sum(diffs**2, axis=2)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return xy[[i, j], :]
