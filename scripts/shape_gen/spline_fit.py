from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep


@dataclass
class SplineFitParams:
    num_ctrl_points: int = 16
    alpha_shape: float = 1.5
    beta_curvature: float = 0.1
    max_iters: int = 200
    penalty_factor: float = 1e8


def simple_procrustes_reference(segment_xy: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """
    Procrustes-like transform: rotate + scale 'segment_xy' so first point maps to p1
    and last point maps to p2.
    """
    seg = np.asarray(segment_xy, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64).reshape(2,)
    p2 = np.asarray(p2, dtype=np.float64).reshape(2,)

    seg_shifted = seg - seg[0, :]
    seg_vec = seg_shifted[-1, :] - seg_shifted[0, :]
    tgt_vec = p2 - p1

    scale_factor = np.linalg.norm(tgt_vec) / (np.linalg.norm(seg_vec) + np.finfo(float).eps)
    angle = np.arctan2(tgt_vec[1], tgt_vec[0]) - np.arctan2(seg_vec[1], seg_vec[0])

    R = np.array(
        [[np.cos(angle), -np.sin(angle)],
         [np.sin(angle),  np.cos(angle)]],
        dtype=np.float64,
    )

    X0 = (seg_shifted @ R.T) * scale_factor + p1
    X0[0, :] = p1
    X0[-1, :] = p2
    return X0


def _chord_length_param(points: np.ndarray) -> np.ndarray:
    """Cumulative chord-length parameterization."""
    p = np.asarray(points, dtype=np.float64)
    if p.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)

    d = np.diff(p, axis=0)
    seglen = np.sqrt(np.sum(d * d, axis=1))
    t = np.concatenate([[0.0], np.cumsum(seglen)])

    # ensure strictly increasing
    eps = np.finfo(float).eps
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + eps
    return t


def sample_parametric_cubic(ctrl_pts: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Replacement for MATLAB's cscvn(...) + fnval(...).

    Interpolates a parametric cubic curve through ctrl_pts using chord-length t.
    """
    try:
        from scipy.interpolate import CubicSpline
    except Exception as e:
        raise RuntimeError("Missing dependency: scipy (needed for cubic spline sampling).") from e

    P = np.asarray(ctrl_pts, dtype=np.float64)
    if P.ndim != 2 or P.shape[1] != 2:
        raise ValueError(f"ctrl_pts must be Kx2. Got {P.shape}")
    if n_samples < 2:
        raise ValueError("n_samples must be >= 2")

    t = _chord_length_param(P)

    csx = CubicSpline(t, P[:, 0], bc_type="not-a-knot")
    csy = CubicSpline(t, P[:, 1], bc_type="not-a-knot")

    tt = np.linspace(t[0], t[-1], n_samples)
    return np.column_stack([csx(tt), csy(tt)])


def _points_in_polygon_shapely(points_xy: np.ndarray, poly: Polygon, prepared=None) -> np.ndarray:
    """Vectorized-ish inside check (includes boundary)."""
    pts = np.asarray(points_xy, dtype=np.float64)
    if prepared is None:
        prepared = prep(poly)
    return np.array(
        [prepared.contains(Point(float(p[0]), float(p[1]))) or prepared.touches(Point(float(p[0]), float(p[1])))
         for p in pts],
        dtype=bool,
    )


def _outside_distance(points_xy: np.ndarray, poly: Polygon, prepared=None) -> np.ndarray:
    """
    Returns 0 for inside (or on boundary), else distance to polygon boundary.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if prepared is None:
        prepared = prep(poly)

    out = np.zeros((pts.shape[0],), dtype=np.float64)
    for i, p in enumerate(pts):
        P = Point(float(p[0]), float(p[1]))
        if prepared.contains(P) or prepared.touches(P):
            out[i] = 0.0
        else:
            out[i] = poly.exterior.distance(P)
    return out


def bspline_obj_fun(
    x_flat: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    ref_pts: np.ndarray,
    occluder: np.ndarray,
    alpha: float,
    beta: float,
    num_ctrl_points: int,
    penalty_factor: float = 1e8,
    eval_samples: int = 300,
    self_intersect_penalty: float = 1e10,
) -> float:
    """
    Objective:
      - match Procrustes reference (alpha)
      - smooth curvature (beta)
      - stay inside occluder (distance-based penalty)
      - avoid self-intersection (hard penalty)
    """
    start_pt = np.asarray(start_pt, dtype=np.float64).reshape(2,)
    end_pt = np.asarray(end_pt, dtype=np.float64).reshape(2,)
    ref_pts = np.asarray(ref_pts, dtype=np.float64)
    occ = np.asarray(occluder, dtype=np.float64)

    interior_count = num_ctrl_points - 2
    ctrl = np.zeros((num_ctrl_points, 2), dtype=np.float64)
    ctrl[0, :] = start_pt
    ctrl[-1, :] = end_pt
    if interior_count > 0:
        ctrl[1:-1, :] = np.asarray(x_flat, dtype=np.float64).reshape(interior_count, 2)

    # Dense curve for constraints (catch overshoot)
    dense_n = int(max(eval_samples, ref_pts.shape[0]))
    curve_dense = sample_parametric_cubic(ctrl, dense_n)

    # Ref-resolution curve for shape/curvature costs
    curve_ref = sample_parametric_cubic(ctrl, int(ref_pts.shape[0]))

    # 1) shape closeness
    diff = curve_ref - ref_pts
    shape_cost = float(np.sum(np.sum(diff * diff, axis=1)))

    # 2) curvature penalty (second differences)
    d1 = np.diff(curve_ref, axis=0)
    d2 = np.diff(d1, axis=0)
    curvature_cost = float(np.sum(np.sum(d2 * d2, axis=1)))

    # 3) out-of-bounds (distance-based)
    poly = Polygon(occ.tolist())
    prepared = prep(poly)

    dist_out_curve = _outside_distance(curve_dense, poly, prepared=prepared)
    dist_out_ctrl = _outside_distance(ctrl, poly, prepared=prepared)

    outside_cost_curve = float(np.sum(dist_out_curve ** 2))
    outside_cost_ctrl = float(np.sum(dist_out_ctrl ** 2))
    out_of_bounds_penalty = penalty_factor * (outside_cost_curve + 3.0 * outside_cost_ctrl)

    # 4) self-intersection
    ls = LineString(curve_dense.tolist())
    si_pen = float(self_intersect_penalty) if (not ls.is_simple) else 0.0

    return alpha * shape_cost + beta * curvature_cost + out_of_bounds_penalty + si_pen


def bspline_occluder_fit(
    original_segment: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    occluder: np.ndarray,
    *,
    num_ctrl_points: int = 16,
    alpha_shape: float = 1.5,
    beta_curvature: float = 0.1,
    max_iters: int = 200,
    final_n_samples: Optional[int] = None,
    penalty_factor: float = 1e8,
) -> np.ndarray:
    """
    Optimization-based curve fit inside a polygonal occluder.

    Uses SLSQP with:
      - warm-start control points from Procrustes reference
      - retries with stronger penalties until curve is inside + simple
    """
    try:
        from scipy.optimize import minimize
    except Exception as e:
        raise RuntimeError("Missing dependency: scipy (needed for optimization).") from e

    seg = np.asarray(original_segment, dtype=np.float64)
    start_pt = np.asarray(start_pt, dtype=np.float64).reshape(2,)
    end_pt = np.asarray(end_pt, dtype=np.float64).reshape(2,)
    occ = np.asarray(occluder, dtype=np.float64)

    if final_n_samples is None:
        final_n_samples = int(seg.shape[0])
    final_n_samples = int(final_n_samples)

    # 1) Procrustes reference (same length as original segment)
    ref = simple_procrustes_reference(seg, start_pt, end_pt)

    # 2) Warm-start control points by sampling along ref
    k = int(num_ctrl_points)
    if k < 2:
        raise ValueError("num_ctrl_points must be >= 2")

    idx = np.linspace(0, ref.shape[0] - 1, k).round().astype(int)
    ctrl0 = ref[idx, :].copy()
    ctrl0[0, :] = start_pt
    ctrl0[-1, :] = end_pt

    interior_count = k - 2
    x0 = ctrl0[1:-1, :].reshape(-1) if interior_count > 0 else np.array([], dtype=np.float64)

    best_x = None
    best_f = np.inf

    penalties = [
        (penalty_factor, 1e10),
        (penalty_factor * 10.0, 1e11),
        (penalty_factor * 50.0, 5e11),
    ]

    poly = Polygon(occ.tolist())
    prepared = prep(poly)

    for pf, sip in penalties:
        def obj(x):
            return bspline_obj_fun(
                x,
                start_pt=start_pt,
                end_pt=end_pt,
                ref_pts=ref,
                occluder=occ,
                alpha=alpha_shape,
                beta=beta_curvature,
                num_ctrl_points=k,
                penalty_factor=pf,
                eval_samples=max(300, ref.shape[0] * 2),
                self_intersect_penalty=sip,
            )

        res = minimize(
            obj,
            x0,
            method="SLSQP",
            options={"maxiter": int(max_iters), "disp": False},
        )

        if float(res.fun) < best_f:
            best_f = float(res.fun)
            best_x = np.array(res.x, copy=True)

        # Validate best so far
        x_try = best_x if best_x is not None else x0

        ctrl = np.zeros((k, 2), dtype=np.float64)
        ctrl[0, :] = start_pt
        ctrl[-1, :] = end_pt
        if interior_count > 0:
            ctrl[1:-1, :] = x_try.reshape(interior_count, 2)

        curve_dense = sample_parametric_cubic(ctrl, max(400, final_n_samples * 3))
        inside_all = bool(np.all(_points_in_polygon_shapely(curve_dense, poly, prepared=prepared)))
        simple_ok = bool(LineString(curve_dense.tolist()).is_simple)

        if inside_all and simple_ok:
            best_x = x_try
            break

    x_opt = best_x if best_x is not None else x0

    # Build final curve
    ctrl = np.zeros((k, 2), dtype=np.float64)
    ctrl[0, :] = start_pt
    ctrl[-1, :] = end_pt
    if interior_count > 0:
        ctrl[1:-1, :] = x_opt.reshape(interior_count, 2)

    aligned = sample_parametric_cubic(ctrl, final_n_samples)
    return aligned
