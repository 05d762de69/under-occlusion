from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep


@dataclass
class DonorFitParams:
    n_samples: int
    enforce_simple: bool = True

    # If curve goes outside: shrink lateral deviation by gamma each iteration
    max_shrink_iters: int = 30
    shrink_gamma: float = 0.88

    # Optional light smoothing after fit (0/1 disables)
    smooth_win: int = 5

    # try reflection across chord (y -> -y) and pick best
    try_mirror: bool = True


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v * 0.0
    return v / n


def _moving_average_1d(y: np.ndarray, win: int) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    win = int(win)
    if win <= 1:
        return y
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones((win,), dtype=np.float64) / float(win)
    return np.convolve(yp, k, mode="valid")


def _resample_polyline_arclen(P: np.ndarray, n: int) -> np.ndarray:
    """
    Resample a polyline to n points by arc-length parameterization.
    """
    P = np.asarray(P, dtype=np.float64)
    if P.shape[0] < 2:
        return np.repeat(P[:1], n, axis=0)

    d = np.diff(P, axis=0)
    seglen = np.sqrt(np.sum(d * d, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])
    if total < 1e-12:
        return np.repeat(P[:1], n, axis=0)

    t = s / total
    tt = np.linspace(0.0, 1.0, int(n))

    x = np.interp(tt, t, P[:, 0])
    y = np.interp(tt, t, P[:, 1])
    return np.column_stack([x, y])


def _inside_fraction(prepared, pts: np.ndarray) -> float:
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        return 0.0
    inside = 0
    for p in pts:
        P = Point(float(p[0]), float(p[1]))
        if prepared.covers(P):
            inside += 1
    return inside / float(len(pts))


def _outside_count(prepared, pts: np.ndarray) -> int:
    pts = np.asarray(pts, dtype=np.float64)
    c = 0
    for p in pts:
        P = Point(float(p[0]), float(p[1]))
        if not prepared.covers(P):
            c += 1
    return c



def _simple(pts: np.ndarray) -> bool:
    return bool(LineString(pts.tolist()).is_simple)


def _similarity_map_segment(seg: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Map donor segment endpoints seg[0], seg[-1] to a,b using similarity transform.
    """
    seg = np.asarray(seg, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64).reshape(2,)
    b = np.asarray(b, dtype=np.float64).reshape(2,)

    p0 = seg[0]
    p1 = seg[-1]
    vd = p1 - p0
    vt = b - a

    nd = float(np.linalg.norm(vd))
    nt = float(np.linalg.norm(vt))
    if nd < 1e-12 or nt < 1e-12:
        return np.empty((0, 2), dtype=np.float64)

    ang_d = np.arctan2(vd[1], vd[0])
    ang_t = np.arctan2(vt[1], vt[0])
    ang = float(ang_t - ang_d)

    R = np.array(
        [[np.cos(ang), -np.sin(ang)],
         [np.sin(ang),  np.cos(ang)]],
        dtype=np.float64
    )
    s = nt / nd

    out = ((seg - p0) @ R.T) * s + a
    out[0] = a
    out[-1] = b
    return out


def _to_chord_frame(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Represent points in chord coordinates:
      base(t) = a + t*(b-a)
      y = signed lateral deviation along normal
    Returns: t (N,), y (N,), u (2,), n (2,)
    """
    a = np.asarray(a, dtype=np.float64).reshape(2,)
    b = np.asarray(b, dtype=np.float64).reshape(2,)
    P = np.asarray(P, dtype=np.float64)

    chord = b - a
    u = _normalize(chord)
    n = np.array([-u[1], u[0]], dtype=np.float64)

    rel = P - a
    x = rel @ u
    y = rel @ n

    L = float(np.linalg.norm(chord))
    if L < 1e-12:
        t = np.zeros((P.shape[0],), dtype=np.float64)
    else:
        t = x / L

    return t, y, u, n


def _from_chord_frame(t: np.ndarray, y: np.ndarray, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(2,)
    b = np.asarray(b, dtype=np.float64).reshape(2,)
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = np.asarray(n, dtype=np.float64).reshape(2,)

    chord = b - a
    base = a[None, :] + t[:, None] * chord[None, :]
    return base + y[:, None] * n[None, :]


def _monotone_lateral_profile_from_fit(base_fit: np.ndarray, a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the mapped donor segment into a stable representation:
    - throw away unstable donor longitudinal progression
    - keep only lateral deviation profile over a clean monotone tau in [0,1]
    """
    t_raw, y_raw, _, n = _to_chord_frame(base_fit, a, b)

    # Use arc-length parameter of the fitted donor as the stable independent variable
    d = np.diff(base_fit, axis=0)
    seglen = np.sqrt(np.sum(d * d, axis=1))
    s = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(s[-1])

    if total < 1e-12:
        tau_src = np.linspace(0.0, 1.0, len(base_fit))
    else:
        tau_src = s / total

    tau = np.linspace(0.0, 1.0, len(base_fit))
    y_tau = np.interp(tau, tau_src, y_raw)

    # exact endpoint anchoring
    y_tau[0] = 0.0
    y_tau[-1] = 0.0

    return tau, y_tau, n


def donor_occluder_fit(
    *,
    donor_segment: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    occluder: np.ndarray,
    rng: Optional[np.random.Generator] = None,
    params: DonorFitParams,
) -> np.ndarray:
    a = np.asarray(start_pt, dtype=np.float64).reshape(2,)
    b = np.asarray(end_pt, dtype=np.float64).reshape(2,)
    occ = np.asarray(occluder, dtype=np.float64)

    poly = Polygon(occ.tolist())
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty or poly.area <= 1e-9:
        return np.empty((0, 2), dtype=np.float64)

    poly_tol = poly.buffer(1.5)
    prepared = prep(poly_tol)

    seg = _resample_polyline_arclen(np.asarray(donor_segment, dtype=np.float64), int(params.n_samples))
    if seg.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    base_fit = _similarity_map_segment(seg, a, b)
    if base_fit.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    tau, y_base, n = _monotone_lateral_profile_from_fit(base_fit, a, b)

    candidates = [y_base]
    if params.try_mirror:
        candidates.append(-y_base)

    best = None
    best_score = -1e9

    for y0 in candidates:
        y_work = y0.copy()

        if int(params.smooth_win) > 1:
            y_work = _moving_average_1d(y_work, int(params.smooth_win))
            y_work[0] = 0.0
            y_work[-1] = 0.0

        gamma = float(params.shrink_gamma)
        pts = _from_chord_frame(tau, y_work, a, b, n)

        for _ in range(int(params.max_shrink_iters)):
            inside_frac = _inside_fraction(prepared, pts)
            outside = _outside_count(prepared, pts)

            score = inside_frac - 0.001 * outside
            if score > best_score:
                best_score = score
                best = pts.copy()

            if inside_frac >= 0.995:
                break

            y_work *= gamma
            y_work[0] = 0.0
            y_work[-1] = 0.0
            pts = _from_chord_frame(tau, y_work, a, b, n)

    if best is None:
        return np.empty((0, 2), dtype=np.float64)

    # fallback to straight chord if needed
    if _inside_fraction(prepared, best) < 0.995:
        tau = np.linspace(0.0, 1.0, int(params.n_samples))
        chord_pts = a[None, :] + tau[:, None] * (b - a)[None, :]
        if _inside_fraction(prepared, chord_pts) >= 0.995:
            best = chord_pts
        else:
            return np.empty((0, 2), dtype=np.float64)

    best[0] = a
    best[-1] = b

    if params.enforce_simple and (not _simple(best)):
        tau = np.linspace(0.0, 1.0, int(params.n_samples))
        chord_pts = a[None, :] + tau[:, None] * (b - a)[None, :]
        if _inside_fraction(prepared, chord_pts) >= 0.995 and _simple(chord_pts):
            best = chord_pts
        else:
            return np.empty((0, 2), dtype=np.float64)

    return best

