# File: shape_gen/generate2.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math
import numpy as np

from shapely.geometry import Point, Polygon, LineString, MultiPoint, GeometryCollection
from shapely.prepared import prep
from shapely.ops import unary_union, nearest_points

from shape_gen.segments import extract_random_segment
from shape_gen.render import draw_and_save
from shape_gen.io_mat import unit_to_pixel
from shape_gen.warp_inside import warp_segment_inside


# Donor-based fitting (natural shapes).
from shape_gen.donor_fit import DonorFitParams, donor_occluder_fit


# ============================================================
# Metadata
# ============================================================

@dataclass
class CompletionMeta:
    silhouette_index: int
    completion_index: int
    sil_class: str
    donor_class: str
    donor_record_index: int
    donor_img_id: Optional[int]
    fraction: float

    # donor-fit params (logged)
    shrink_gamma: float
    max_shrink_iters: int
    smooth_win: int
    try_mirror: bool
    n_samples_mode: str

    # spline refit params (logged)
    refit_enabled: bool
    refit_n_ctrl: int
    refit_subdiv: int
    refit_jitter_sigma: float
    refit_max_attempts: int

    out_file: str
    attempts: int
    valid: bool



def save_metadata_jsonl(metas: List[CompletionMeta], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m)) + "\n")


# ============================================================
# Core helpers
# ============================================================

def _choose_donor_index(
    rng: np.random.Generator,
    records,
    classes: List[str],
    byClass: Dict[str, np.ndarray],
    sil_class: str,
) -> Tuple[int, Optional[str]]:
    if classes:
        if sil_class and (sil_class in classes) and any(c != sil_class for c in classes):
            other = [c for c in classes if c != sil_class]
        else:
            other = classes

        if other:
            donor_class = other[int(rng.integers(0, len(other)))]
            pool = byClass[donor_class]
            idx = int(pool[int(rng.integers(0, len(pool)))])
            return idx, donor_class

    return int(rng.integers(0, len(records))), None


def _validate_inside(curve_xy: np.ndarray, prepared_occ) -> bool:
    pts = np.asarray(curve_xy, dtype=np.float64)
    for p in pts:
        P = Point(float(p[0]), float(p[1]))
        if not (prepared_occ.contains(P) or prepared_occ.touches(P)):
            return False
    return True


def _is_simple_polyline(curve_xy: np.ndarray) -> bool:
    curve = np.asarray(curve_xy, dtype=np.float64)
    if curve.shape[0] < 4:
        return True
    ls = LineString(curve.tolist())
    return bool(ls.is_simple)


# ============================================================
# Intersection extraction and visible arc
# ============================================================

def _pick_two_farthest_points_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xy.shape[0] < 2:
        raise RuntimeError("Need at least 2 intersection points.")
    if xy.shape[0] == 2:
        return xy[0], xy[1]
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return xy[i], xy[j]


def _nearest_index(curve_xy: np.ndarray, p: np.ndarray) -> int:
    d2 = np.sum((curve_xy - p[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def _indices_between(a: int, b: int, n: int) -> np.ndarray:
    if a <= b:
        return np.arange(a, b + 1)
    return np.concatenate([np.arange(a, n), np.arange(0, b + 1)])


def _arc_points(curve_xy: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    return curve_xy[idxs, :]

def _unique_points_eps(P: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if P.size == 0:
        return P.reshape(0, 2)
    out = []
    for p in P:
        if not out:
            out.append(p)
            continue
        d2 = np.sum((np.asarray(out) - p[None, :]) ** 2, axis=1)
        if float(np.min(d2)) > (eps * eps):
            out.append(p)
    return np.asarray(out, dtype=np.float64)


def _edge_boundary_crossings(sil_xy: np.ndarray, occ_poly: Polygon) -> np.ndarray:
    """
    Robust fallback. Finds crossings of the occluder boundary by scanning
    silhouette segments for inside/outside changes, then interpolating.
    Returns an array of candidate intersection points.
    """
    sil = np.asarray(sil_xy, dtype=np.float64)
    if sil.shape[0] < 2:
        return np.zeros((0, 2), dtype=np.float64)

    # ensure closed for edge scan
    if not np.allclose(sil[0], sil[-1]):
        sil = np.vstack([sil, sil[0]])

    # inside test at vertices
    inside = np.array([occ_poly.contains(Point(float(x), float(y))) or occ_poly.touches(Point(float(x), float(y)))
                       for x, y in sil], dtype=bool)

    pts = []
    boundary = occ_poly.boundary

    for i in range(len(sil) - 1):
        a = sil[i]
        b = sil[i + 1]
        ia = inside[i]
        ib = inside[i + 1]

        if ia == ib:
            continue

        # segment crosses boundary somewhere. Use shapely intersection on this small segment.
        seg = LineString([a.tolist(), b.tolist()])
        inter = seg.intersection(boundary)

        if inter.is_empty:
            continue

        if inter.geom_type == "Point":
            pts.append([inter.x, inter.y])
        elif inter.geom_type == "MultiPoint":
            for g in inter.geoms:
                pts.append([g.x, g.y])
        elif inter.geom_type in ("LineString", "MultiLineString"):
            u = unary_union(inter)
            if u.geom_type == "LineString":
                coords = np.asarray(u.coords, dtype=np.float64)
                pts.append(coords[0].tolist())
                pts.append(coords[-1].tolist())
            elif u.geom_type == "MultiLineString":
                for ls in u.geoms:
                    coords = np.asarray(ls.coords, dtype=np.float64)
                    pts.append(coords[0].tolist())
                    pts.append(coords[-1].tolist())

    P = np.asarray(pts, dtype=np.float64)
    P = _unique_points_eps(P, eps=1e-6)
    return P


def _extract_intersections_AB_and_visible_arc_robust(
    sil_xy: np.ndarray,
    occ_xy: np.ndarray,
    *,
    snap_to_silhouette_vertices: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    """
    Drop-in replacement for _extract_intersections_AB_and_visible_arc.
    Uses your original logic. Adds fallback paths when Shapely yields <2 points.
    """
    sil = np.asarray(sil_xy, dtype=np.float64)
    occ = np.asarray(occ_xy, dtype=np.float64)

    # clean polygons, avoids some degeneracies
    occ_poly = Polygon(occ.tolist())
    if not occ_poly.is_valid:
        occ_poly = occ_poly.buffer(0)

    # Ensure silhouette closed for intersection
    sil_closed = sil
    if not np.allclose(sil_closed[0], sil_closed[-1]):
        sil_closed = np.vstack([sil_closed, sil_closed[0]])

    occ_boundary = occ_poly.boundary
    sil_line = LineString(sil_closed.tolist())

    def _collect_pts_from_inter(inter) -> np.ndarray:
        pts = []
        if inter.is_empty:
            return np.zeros((0, 2), dtype=np.float64)
        if isinstance(inter, MultiPoint):
            for g in inter.geoms:
                pts.append([g.x, g.y])
        elif isinstance(inter, GeometryCollection):
            for g in inter.geoms:
                if g.geom_type == "Point":
                    pts.append([g.x, g.y])
                elif g.geom_type == "MultiPoint":
                    for gg in g.geoms:
                        pts.append([gg.x, gg.y])
                elif g.geom_type in ("LineString", "MultiLineString"):
                    u = unary_union(g)
                    if u.geom_type == "LineString":
                        coords = np.asarray(u.coords, dtype=np.float64)
                        pts.append(coords[0].tolist())
                        pts.append(coords[-1].tolist())
                    elif u.geom_type == "MultiLineString":
                        for ls in u.geoms:
                            coords = np.asarray(ls.coords, dtype=np.float64)
                            pts.append(coords[0].tolist())
                            pts.append(coords[-1].tolist())
        elif inter.geom_type == "Point":
            pts.append([inter.x, inter.y])
        elif inter.geom_type in ("LineString", "MultiLineString"):
            u = unary_union(inter)
            if u.geom_type == "LineString":
                coords = np.asarray(u.coords, dtype=np.float64)
                pts.append(coords[0].tolist())
                pts.append(coords[-1].tolist())
            elif u.geom_type == "MultiLineString":
                for ls in u.geoms:
                    coords = np.asarray(ls.coords, dtype=np.float64)
                    pts.append(coords[0].tolist())
                    pts.append(coords[-1].tolist())

        P = np.asarray(pts, dtype=np.float64)
        return _unique_points_eps(P, eps=1e-6)

    # 1) original shapely intersection
    inter = sil_line.intersection(occ_boundary)
    P = _collect_pts_from_inter(inter)

    # 2) fallback: scan edges for crossings
    if P.shape[0] < 2:
        P2 = _edge_boundary_crossings(sil, occ_poly)
        if P2.shape[0] >= 2:
            P = P2

    # 3) tangency breaker: tiny buffer around boundary
    if P.shape[0] < 2:
        eps = 1e-3
        boundary2 = occ_poly.buffer(eps).boundary
        inter2 = sil_line.intersection(boundary2)
        P3 = _collect_pts_from_inter(inter2)
        if P3.shape[0] >= 2:
            P = P3

    if P.shape[0] < 2:
        raise RuntimeError(f"Expected >=2 intersection points, got {P.shape[0]}.")

    # Now reuse the remainder of your original logic
    pA, pB = _pick_two_farthest_points_xy(P)

    if snap_to_silhouette_vertices:
        n = sil.shape[0]
        idxA = _nearest_index(sil, pA)
        idxB = _nearest_index(sil, pB)
        pA = sil[idxA].copy()
        pB = sil[idxB].copy()
    else:
        idxA = _nearest_index(sil, pA)
        idxB = _nearest_index(sil, pB)

    n = sil.shape[0]
    arc_AtoB = _indices_between(idxA, idxB, n)
    arc_BtoA = _indices_between(idxB, idxA, n)

    arc1 = _arc_points(sil, arc_AtoB)
    arc2 = _arc_points(sil, arc_BtoA)

    prepared = prep(occ_poly)
    inside1 = np.array([prepared.contains(Point(float(x), float(y))) for x, y in arc1], dtype=np.float64).mean()
    inside2 = np.array([prepared.contains(Point(float(x), float(y))) for x, y in arc2], dtype=np.float64).mean()

    if inside1 >= inside2:
        visible_arc = arc2
    else:
        visible_arc = arc1[::-1].copy()

    visible_arc = np.asarray(visible_arc, dtype=np.float64)
    visible_arc[0] = pB
    visible_arc[-1] = pA

    return pA, pB, visible_arc, prepared

def _extract_intersections_AB_and_visible_arc(
    sil_xy: np.ndarray,
    occ_xy: np.ndarray,
    *,
    snap_to_silhouette_vertices: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object]:
    sil = np.asarray(sil_xy, dtype=np.float64)
    occ = np.asarray(occ_xy, dtype=np.float64)

    sil_closed = sil
    if not np.allclose(sil_closed[0], sil_closed[-1]):
        sil_closed = np.vstack([sil_closed, sil_closed[0]])

    occ_poly = Polygon(occ.tolist())
    occ_boundary = occ_poly.boundary

    sil_line = LineString(sil_closed.tolist())
    inter = sil_line.intersection(occ_boundary)

    pts: List[List[float]] = []
    if inter.is_empty:
        raise RuntimeError("Silhouette does not intersect occluder boundary.")

    if isinstance(inter, MultiPoint):
        for g in inter.geoms:
            pts.append([g.x, g.y])
    elif isinstance(inter, GeometryCollection):
        for g in inter.geoms:
            if g.geom_type == "Point":
                pts.append([g.x, g.y])
            elif g.geom_type == "MultiPoint":
                for gg in g.geoms:
                    pts.append([gg.x, gg.y])
    elif inter.geom_type == "Point":
        pts.append([inter.x, inter.y])
    else:
        if inter.geom_type in ("LineString", "MultiLineString"):
            u = unary_union(inter)
            if u.geom_type == "LineString":
                coords = np.asarray(u.coords, dtype=np.float64)
                pts.append(coords[0].tolist())
                pts.append(coords[-1].tolist())
            elif u.geom_type == "MultiLineString":
                for ls in u.geoms:
                    coords = np.asarray(ls.coords, dtype=np.float64)
                    pts.append(coords[0].tolist())
                    pts.append(coords[-1].tolist())

    P = np.asarray(pts, dtype=np.float64)
    if P.shape[0] < 2:
        raise RuntimeError(f"Expected >=2 intersection points, got {P.shape[0]}.")

    pA, pB = _pick_two_farthest_points_xy(P)

    # Optionally snap A/B to nearest sampled silhouette vertex for stable indexing.
    # If you want more variability and avoid vertex locking, set this False.
    if snap_to_silhouette_vertices:
        n = sil.shape[0]
        idxA = _nearest_index(sil, pA)
        idxB = _nearest_index(sil, pB)
        pA = sil[idxA].copy()
        pB = sil[idxB].copy()
    else:
        # Keep true intersection points.
        idxA = _nearest_index(sil, pA)
        idxB = _nearest_index(sil, pB)

    n = sil.shape[0]
    arc_AtoB = _indices_between(idxA, idxB, n)
    arc_BtoA = _indices_between(idxB, idxA, n)

    arc1 = _arc_points(sil, arc_AtoB)
    arc2 = _arc_points(sil, arc_BtoA)

    prepared = prep(occ_poly)
    inside1 = np.array([prepared.contains(Point(float(x), float(y))) for x, y in arc1], dtype=np.float64).mean()
    inside2 = np.array([prepared.contains(Point(float(x), float(y))) for x, y in arc2], dtype=np.float64).mean()

    if inside1 >= inside2:
        visible_arc = arc2
    else:
        visible_arc = arc1[::-1].copy()

    visible_arc = np.asarray(visible_arc, dtype=np.float64)
    visible_arc[0] = pB
    visible_arc[-1] = pA

    return pA, pB, visible_arc, prepared


# ============================================================
# Flexible spline refit, smooth and non self-intersecting
# ============================================================

def _polyline_arclength(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def _resample_by_arclength(xy: np.ndarray, n: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] == 0:
        return xy.reshape(0, 2)
    if xy.shape[0] == 1:
        return np.repeat(xy, n, axis=0)
    s = _polyline_arclength(xy)
    total = float(s[-1])
    if total <= 1e-12:
        return np.repeat(xy[:1], n, axis=0)
    t = np.linspace(0.0, total, n)
    out = np.zeros((n, 2), dtype=np.float64)
    j = 0
    for i, ti in enumerate(t):
        while j < len(s) - 2 and s[j + 1] < ti:
            j += 1
        s0, s1 = float(s[j]), float(s[j + 1])
        p0, p1 = xy[j], xy[j + 1]
        if s1 - s0 <= 1e-12:
            out[i] = p0
        else:
            a = (ti - s0) / (s1 - s0)
            out[i] = (1.0 - a) * p0 + a * p1
    return out


def _project_points_inside(points_xy: np.ndarray, occ_poly: Polygon) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).copy()
    for i in range(pts.shape[0]):
        P = Point(float(pts[i, 0]), float(pts[i, 1]))
        if occ_poly.contains(P) or occ_poly.touches(P):
            continue
        # nearest point on polygon boundary, then pull slightly inward by blending.
        nearest_on_poly = nearest_points(P, occ_poly)[1]
        q = np.array([nearest_on_poly.x, nearest_on_poly.y], dtype=np.float64)
        pts[i] = q
    return pts


def _catmull_rom_centripetal(P: np.ndarray, n_per_seg: int = 25, alpha: float = 0.5) -> np.ndarray:
    """
    Centripedal Catmull-Rom spline through control points.
    This yields smooth, natural curvature without overshoot as often as uniform parameterization.
    """
    P = np.asarray(P, dtype=np.float64)
    m = P.shape[0]
    if m < 2:
        return P.copy()
    if m == 2:
        return _resample_by_arclength(P, max(2, n_per_seg + 1))

    def tj(ti: float, pi: np.ndarray, pj: np.ndarray) -> float:
        return ti + (np.linalg.norm(pj - pi) ** alpha)

    # Extend endpoints by duplication for boundary conditions.
    Pext = np.vstack([P[0], P, P[-1]])
    out: List[np.ndarray] = []

    for i in range(1, m):
        p0, p1, p2, p3 = Pext[i - 1], Pext[i], Pext[i + 1], Pext[i + 2]

        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)

        # Guard degenerate cases.
        if (t2 - t1) <= 1e-12:
            seg = np.vstack([p1, p2])
            if i == 1:
                out.append(seg[0:1])
            out.append(seg[1:2])
            continue

        ts = np.linspace(t1, t2, n_per_seg, endpoint=(i == m - 1))

        for t in ts:
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1 if (t1 - t0) > 1e-12 else p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3 if (t3 - t2) > 1e-12 else p2

            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2 if (t2 - t0) > 1e-12 else A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3 if (t3 - t1) > 1e-12 else A3

            C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
            out.append(C[None, :])

    return np.vstack(out).astype(np.float64, copy=False)


def _flexible_refit_spline_inside(
    curve_xy: np.ndarray,
    *,
    pA: np.ndarray,
    pB: np.ndarray,
    occ_poly: Polygon,
    prepared_occ,
    rng: np.random.Generator,
    n_ctrl: int = 14,
    subdiv: int = 25,
    jitter_sigma: float = 0.0,
    max_attempts: int = 10,
) -> np.ndarray:
    """
    Produces a smoother, more flexible curve inside the occluder.
    Keeps endpoints fixed to pA and pB.
    Ensures curve is inside occluder and is_simple (no self intersections).
    Uses centripetal Catmull-Rom through a reduced set of control points, with optional jitter.
    """

    base = np.asarray(curve_xy, dtype=np.float64)
    if base.shape[0] < 4:
        out = base.copy()
        out[0] = pA
        out[-1] = pB
        return out

    # Build control points by arclength downsampling.
    ctrl0 = _resample_by_arclength(base, int(max(4, n_ctrl)))
    ctrl0[0] = pA
    ctrl0[-1] = pB

    # Scale jitter to occluder size.
    minx, miny, maxx, maxy = occ_poly.bounds
    scale = math.hypot(maxx - minx, maxy - miny)
    sigma_px = float(jitter_sigma) * float(scale)

    best: Optional[np.ndarray] = None

    for _ in range(int(max_attempts)):
        ctrl = ctrl0.copy()

        # Optional jitter of interior control points only.
        if sigma_px > 0.0 and ctrl.shape[0] > 2:
            noise = rng.normal(0.0, sigma_px, size=(ctrl.shape[0] - 2, 2))
            ctrl[1:-1] += noise

        # Project control points inside occluder.
        ctrl = _project_points_inside(ctrl, occ_poly)
        ctrl[0] = pA
        ctrl[-1] = pB

        dense = _catmull_rom_centripetal(ctrl, n_per_seg=int(max(5, subdiv)), alpha=0.5)

        # Enforce exact endpoints.
        dense[0] = pA
        dense[-1] = pB

        # Project dense points inside occluder.
        dense = _project_points_inside(dense, occ_poly)
        dense[0] = pA
        dense[-1] = pB

        if not _validate_inside(dense, prepared_occ):
            sigma_px *= 0.65
            continue

        if not _is_simple_polyline(dense):
            sigma_px *= 0.65
            continue

        best = dense
        break

    if best is None:
        # Fall back to original curve if refit fails.
        out = base.copy()
        out[0] = pA
        out[-1] = pB
        return out

    return best.astype(np.float64, copy=False)


# ============================================================
# Generator
# ============================================================

def generate_completions(
    *,
    silhouette: np.ndarray,
    occluder: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    minX: int,
    minY: int,
    wBB: int,
    hBB: int,
    out_w: int,
    out_h: int,
    out_dir: str | Path,
    silhouette_index: int,
    sil_class: str,
    base_grid: int,
    records,
    classes: List[str],
    byClass: Dict[str, np.ndarray],
    n_images: int,
    rng: np.random.Generator,
        # corridor warping behavior (NEW)
    warp_enabled: bool = True,
    warp_k_modes: int = 6,
    warp_amp_rel: float = 0.06,
    warp_n_steps: int = 2,
    warp_max_attempts: int = 10,
    warp_decay: float = 0.65,


    # donor segment sampling
    fraction: float = 0.40,

    # donor-fit behavior (preserve donor curvature; shrink only if needed)
    shrink_gamma: float = 0.88,
    max_shrink_iters: int = 30,
    smooth_win: int = 5,
    try_mirror: bool = True,

    # behavior
    final_n_samples_mode: str = "match_segment",  # or "matlab_100"
    supersample: int = 4,
    flush_every: int = 500,
    max_attempts_per_image: int = 50,
    require_valid: bool = True,
    save_invalid: bool = False,
    invalid_subdir: str = "_invalid",

    # intersection behavior
    snap_intersections_to_vertices: bool = True,

    # flexible refit spline behavior
    refit_enabled: bool = True,
    refit_n_ctrl: int = 14,
    refit_subdiv: int = 25,
    refit_jitter_sigma: float = 0.015,
    refit_max_attempts: int = 10,
) -> tuple[List[CompletionMeta], List[str], List[np.ndarray]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_invalid:
        (out_dir / invalid_subdir).mkdir(parents=True, exist_ok=True)

    if n_images <= 0:
        return [], [], []

    sil = np.asarray(silhouette, dtype=np.float64)
    occ_xy = np.asarray(occluder, dtype=np.float64)

    occ_poly = Polygon(occ_xy.tolist())
    prepared_occ = prep(occ_poly)

    # Use caller-provided endpoints. This avoids intersection degeneracies in small ovals.
    pA_true = np.asarray(start_pt, dtype=np.float64).copy()
    pB_true = np.asarray(end_pt, dtype=np.float64).copy()

    # Visible arc should still be computed, but we can compute it WITHOUT relying on intersection geometry.
    # We’ll derive indices by nearest silhouette vertices to pA_true/pB_true.
    sil0 = sil
    n = sil0.shape[0]
    idxA = _nearest_index(sil0, pA_true)
    idxB = _nearest_index(sil0, pB_true)

    arc_AtoB = _indices_between(idxA, idxB, n)
    arc_BtoA = _indices_between(idxB, idxA, n)

    arc1 = _arc_points(sil0, arc_AtoB)
    arc2 = _arc_points(sil0, arc_BtoA)

    inside1 = np.array([prepared_occ.contains(Point(float(x), float(y))) for x, y in arc1], dtype=np.float64).mean()
    inside2 = np.array([prepared_occ.contains(Point(float(x), float(y))) for x, y in arc2], dtype=np.float64).mean()

    # pick the arc that is mostly OUTSIDE the occluder as "visible"
    if inside1 >= inside2:
        visible_arc = arc2
    else:
        visible_arc = arc1[::-1].copy()

    visible_arc = np.asarray(visible_arc, dtype=np.float64)
    visible_arc[0] = pB_true
    visible_arc[-1] = pA_true


    metas: List[CompletionMeta] = []
    out_files_xy: List[str] = []
    polygons_xy: List[np.ndarray] = []

    produced = 0
    target = int(n_images)
    i = 1

    while produced < target:
        attempts = 0
        valid = False
        out_file: Optional[Path] = None
        new_shape_valid: Optional[np.ndarray] = None

        donor_idx: Optional[int] = None
        donor_class: Optional[str] = None
        donor_img_id: Optional[int] = None

        while attempts < max_attempts_per_image and (not valid):
            attempts += 1

            donor_idx, donor_class = _choose_donor_index(rng, records, classes, byClass, sil_class)
            donor = records[donor_idx]
            donor_img_id = getattr(donor, "img_id", None)

            alt_shape = unit_to_pixel(donor.contour_u, base_grid)
            random_segment, _ = extract_random_segment(alt_shape, fraction=fraction, rng=rng)

            n_samples = 100 if final_n_samples_mode == "matlab_100" else int(len(random_segment))
            if n_samples < 20:
                n_samples = 20

            params = DonorFitParams(
                n_samples=int(n_samples),
                enforce_simple=True,
                max_shrink_iters=int(max_shrink_iters),
                shrink_gamma=float(shrink_gamma),
                smooth_win=int(smooth_win),
                try_mirror=bool(try_mirror),
            )

            aligned_segment = donor_occluder_fit(
                donor_segment=random_segment,
                start_pt=start_pt,
                end_pt=end_pt,
                occluder=occ_xy,
                rng=rng,
                params=params,
            )

            if aligned_segment.size == 0:
                continue

            aligned_segment = np.asarray(aligned_segment, dtype=np.float64)

            # Validate inside. Also ensure not self-intersecting.
            if not _validate_inside(aligned_segment, prepared_occ):
                continue
            if not _is_simple_polyline(aligned_segment):
                continue

            # Orient donor segment so it runs pA -> pB.
            dA0 = np.sum((aligned_segment[0] - pA_true) ** 2)
            dA1 = np.sum((aligned_segment[-1] - pA_true) ** 2)
            if dA1 < dA0:
                aligned_segment = np.flipud(aligned_segment)

            # Force exact endpoints.
            aligned_segment = aligned_segment.copy()
            aligned_segment[0] = pA_true
            aligned_segment[-1] = pB_true

            # Flexible spline refit for natural curvature and more flexibility.
            if refit_enabled:
                refined = _flexible_refit_spline_inside(
                    aligned_segment,
                    pA=pA_true,
                    pB=pB_true,
                    occ_poly=occ_poly,
                    prepared_occ=prepared_occ,
                    rng=rng,
                    n_ctrl=int(refit_n_ctrl),
                    subdiv=int(refit_subdiv),
                    jitter_sigma=float(refit_jitter_sigma),
                    max_attempts=int(refit_max_attempts),
                )
                if refined.size == 0:
                    continue
                if not _validate_inside(refined, prepared_occ):
                    continue
                if not _is_simple_polyline(refined):
                    continue
                aligned_segment = refined
                        # Corridor warping. Donor is the prior, warp explores feasible diversity.
            if warp_enabled:
                warped = warp_segment_inside(
                    aligned_segment,
                    pA=pA_true,
                    pB=pB_true,
                    occ_poly=occ_poly,
                    prepared_occ=prepared_occ,
                    rng=rng,
                    validate_inside_fn=_validate_inside,
                    is_simple_fn=_is_simple_polyline,
                    k_modes=int(warp_k_modes),
                    amp_rel=float(warp_amp_rel),
                    n_steps=int(warp_n_steps),
                    max_attempts=int(warp_max_attempts),
                    decay=float(warp_decay),
                )
                if warped.size == 0:
                    continue
                aligned_segment = warped

            # Stitch polygon as: (inside completion) pA->pB, then (visible arc) pB->pA.
            new_shape = np.vstack([aligned_segment, visible_arc[1:]])

            # Explicitly close polygon.
            if not np.allclose(new_shape[0], new_shape[-1]):
                new_shape = np.vstack([new_shape, new_shape[0]])

            # Final simple check for polygon validity and obvious self-crossings.
            poly = Polygon(new_shape.tolist())
            if (not poly.is_valid) or (poly.area <= 1e-6):
                continue

            out_file = out_dir / f"completion_{silhouette_index:04d}_{i:05d}.png"
            draw_and_save(
                polygons=[new_shape],
                colors=[[0, 0, 0]],
                minX=minX,
                minY=minY,
                wBB=wBB,
                hBB=hBB,
                out_w=out_w,
                out_h=out_h,
                out_file=out_file,
                supersample=supersample,
            )

            valid = True
            new_shape_valid = new_shape.astype(np.float32, copy=False)

        if require_valid and (not valid):
            metas.append(
                CompletionMeta(
                    silhouette_index=int(silhouette_index),
                    completion_index=int(i),
                    sil_class=str(sil_class or ""),
                    donor_class=str(donor_class or (records[donor_idx].category if donor_idx is not None else "") or ""),
                    donor_record_index=int(donor_idx) if donor_idx is not None else -1,
                    donor_img_id=donor_img_id,
                    fraction=float(fraction),
                    shrink_gamma=float(shrink_gamma),
                    max_shrink_iters=int(max_shrink_iters),
                    smooth_win=int(smooth_win),
                    try_mirror=bool(try_mirror),
                    n_samples_mode=str(final_n_samples_mode),
                    refit_enabled=bool(refit_enabled),
                    refit_n_ctrl=int(refit_n_ctrl),
                    refit_subdiv=int(refit_subdiv),
                    refit_jitter_sigma=float(refit_jitter_sigma),
                    refit_max_attempts=int(refit_max_attempts),
                    out_file=str(out_file) if out_file is not None else "",
                    attempts=int(attempts),
                    valid=False,
                )
            )
            if (len(metas) % flush_every) == 0:
                print(f"Attempts logged {len(metas)}. Valid produced {produced}/{target}...")
            i += 1
            continue

        if out_file is not None and new_shape_valid is not None:
            produced += 1
            out_files_xy.append(str(out_file))
            polygons_xy.append(new_shape_valid)

        metas.append(
            CompletionMeta(
                silhouette_index=int(silhouette_index),
                completion_index=int(i),
                sil_class=str(sil_class or ""),
                donor_class=str(donor_class or (records[donor_idx].category if donor_idx is not None else "") or ""),
                donor_record_index=int(donor_idx) if donor_idx is not None else -1,
                donor_img_id=donor_img_id,
                fraction=float(fraction),
                shrink_gamma=float(shrink_gamma),
                max_shrink_iters=int(max_shrink_iters),
                smooth_win=int(smooth_win),
                try_mirror=bool(try_mirror),
                n_samples_mode=str(final_n_samples_mode),
                refit_enabled=bool(refit_enabled),
                refit_n_ctrl=int(refit_n_ctrl),
                refit_subdiv=int(refit_subdiv),
                refit_jitter_sigma=float(refit_jitter_sigma),
                refit_max_attempts=int(refit_max_attempts),
                out_file=str(out_file) if out_file is not None else "",
                attempts=int(attempts),
                valid=bool(valid),
            )
        )

        if (produced % flush_every) == 0:
            print(f"Generated {produced}/{target} completions... (last attempts={attempts})")

        i += 1

    return metas, out_files_xy, polygons_xy
