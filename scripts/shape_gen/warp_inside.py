# File: shape_gen/warp_inside.py
from __future__ import annotations

import math
from typing import Optional, Tuple, List

import numpy as np
from shapely.geometry import Point, Polygon


def _polyline_arclength(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    s = np.concatenate([[0.0], np.cumsum(d)])
    return s


def _unit_normals(xy: np.ndarray) -> np.ndarray:
    """
    Approximate unit normals along a polyline using forward/backward differences.
    """
    xy = np.asarray(xy, dtype=np.float64)
    n = xy.shape[0]
    if n < 2:
        return np.zeros((n, 2), dtype=np.float64)

    tang = np.zeros((n, 2), dtype=np.float64)
    tang[1:-1] = xy[2:] - xy[:-2]
    tang[0] = xy[1] - xy[0]
    tang[-1] = xy[-1] - xy[-2]

    tnorm = np.linalg.norm(tang, axis=1, keepdims=True)
    tnorm = np.maximum(tnorm, 1e-12)
    tang = tang / tnorm

    # Rotate tangent by +90 degrees to get a normal
    normals = np.stack([-tang[:, 1], tang[:, 0]], axis=1)
    nnorm = np.linalg.norm(normals, axis=1, keepdims=True)
    nnorm = np.maximum(nnorm, 1e-12)
    normals = normals / nnorm
    return normals


def _project_points_inside_poly(points_xy: np.ndarray, occ_poly: Polygon) -> np.ndarray:
    """
    Hard projection: if outside, snap to nearest point on polygon boundary.
    This is conservative and keeps the hard-constraint behavior.
    """
    pts = np.asarray(points_xy, dtype=np.float64).copy()
    boundary = occ_poly.boundary
    for i in range(pts.shape[0]):
        p = Point(float(pts[i, 0]), float(pts[i, 1]))
        if occ_poly.contains(p) or occ_poly.touches(p):
            continue
        q = boundary.interpolate(boundary.project(p))
        pts[i, 0] = float(q.x)
        pts[i, 1] = float(q.y)
    return pts


def _warp_basis_values(t01: np.ndarray, k: int) -> np.ndarray:
    """
    Smooth basis in [0,1] that is zero at endpoints.
    Uses sine modes: sin(pi*m*t), m=1..k
    """
    t01 = np.asarray(t01, dtype=np.float64)
    B = np.zeros((t01.shape[0], k), dtype=np.float64)
    for m in range(1, k + 1):
        B[:, m - 1] = np.sin(math.pi * m * t01)
    return B


def warp_segment_inside(
    segment_xy: np.ndarray,
    *,
    pA: np.ndarray,
    pB: np.ndarray,
    occ_poly: Polygon,
    prepared_occ,
    rng: np.random.Generator,
    validate_inside_fn,
    is_simple_fn,
    k_modes: int = 6,
    amp_rel: float = 0.06,
    n_steps: int = 2,
    max_attempts: int = 10,
    decay: float = 0.65,
) -> np.ndarray:
    """
    Corridor warping.
    - Applies low-dim normal displacements along arclength.
    - Projects back inside after each step.
    - Keeps endpoints fixed.
    - Returns empty array if it cannot find a valid warp.
    """

    base = np.asarray(segment_xy, dtype=np.float64)
    if base.shape[0] < 4:
        out = base.copy()
        out[0] = pA
        out[-1] = pB
        return out

    # Scale warp amplitude to occluder size
    minx, miny, maxx, maxy = occ_poly.bounds
    scale = math.hypot(maxx - minx, maxy - miny)
    amp_px0 = float(amp_rel) * float(scale)

    s = _polyline_arclength(base)
    total = float(s[-1])
    if total <= 1e-12:
        out = base.copy()
        out[0] = pA
        out[-1] = pB
        return out

    t01 = s / total
    B = _warp_basis_values(t01, int(max(1, k_modes)))
    normals = _unit_normals(base)

    best: Optional[np.ndarray] = None
    amp_px = amp_px0

    for _ in range(int(max_attempts)):
        # Random coefficients. Symmetric distribution.
        coeff = rng.normal(0.0, 1.0, size=(B.shape[1],))
        # Normalize to keep typical amplitude stable
        coeff = coeff / max(1e-12, float(np.linalg.norm(coeff)))

        disp = (B @ coeff)[:, None] * normals  # (n,1)*(n,2)
        cand = base.copy()

        # Multi-step application, with projection each step
        for _step in range(int(max(1, n_steps))):
            cand[1:-1] = cand[1:-1] + (amp_px * disp[1:-1])
            cand[0] = pA
            cand[-1] = pB
            cand = _project_points_inside_poly(cand, occ_poly)
            cand[0] = pA
            cand[-1] = pB

        # Validate using your existing validators
        if not validate_inside_fn(cand, prepared_occ):
            amp_px *= float(decay)
            continue
        if not is_simple_fn(cand):
            amp_px *= float(decay)
            continue

        best = cand
        break

    if best is None:
        return np.zeros((0, 2), dtype=np.float64)

    return best.astype(np.float64, copy=False)
