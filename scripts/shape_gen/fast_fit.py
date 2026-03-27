from __future__ import annotations

import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import nearest_points


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v * 0.0
    return v / n


def _signed_lateral_profile(seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert donor segment to a 1D deviation profile:
      t in [0,1] along chord from seg[0] to seg[-1]
      y = signed perpendicular displacement from chord
    """
    P = np.asarray(seg, dtype=np.float64)
    a = P[0]
    b = P[-1]
    chord = b - a
    u = _normalize(chord)
    n = np.array([-u[1], u[0]])

    rel = P - a
    x = rel @ u
    y = rel @ n

    # normalize x to [0,1]
    x0, x1 = x[0], x[-1]
    denom = (x1 - x0) if abs(x1 - x0) > 1e-12 else 1.0
    t = (x - x0) / denom
    return t, y


def _resample_profile(t: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    """
    Resample y(t) onto n evenly spaced points in [0,1] via linear interp.
    """
    tt = np.linspace(0.0, 1.0, n)
    # enforce monotonic t for interpolation
    order = np.argsort(t)
    t2 = t[order]
    y2 = y[order]
    # de-duplicate
    _, uniq = np.unique(t2, return_index=True)
    t2 = t2[uniq]
    y2 = y2[uniq]
    if t2.size < 2:
        return np.zeros((n,), dtype=np.float64)
    return np.interp(tt, t2, y2)


def _smooth_1d(y: np.ndarray, win: int = 11) -> np.ndarray:
    """
    Very cheap smoothing: moving average with odd window.
    """
    y = np.asarray(y, dtype=np.float64)
    win = int(win)
    if win < 3:
        return y
    if win % 2 == 0:
        win += 1
    pad = win // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    k = np.ones((win,), dtype=np.float64) / win
    return np.convolve(yp, k, mode="valid")


def _project_inside(poly: Polygon, pts: np.ndarray, inset_eps: float = 1e-6) -> np.ndarray:
    """
    For points outside polygon: snap to nearest boundary point and nudge slightly inward.
    """
    out = pts.copy()
    for i, p in enumerate(out):
        P = Point(float(p[0]), float(p[1]))
        if poly.contains(P) or poly.touches(P):
            continue
        # nearest boundary point
        q = nearest_points(poly.exterior, P)[0]
        qxy = np.array([q.x, q.y], dtype=np.float64)

        # inward nudge: move from boundary towards polygon representative point
        rep = poly.representative_point()
        v = np.array([rep.x, rep.y], dtype=np.float64) - qxy
        v = _normalize(v)
        out[i] = qxy + inset_eps * v
    return out


def fast_occluder_fit(
    donor_segment: np.ndarray,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    occluder: np.ndarray,
    *,
    n_samples: int,
    amp_scale: float = 0.8,
    smooth_win: int = 11,
    inset_eps: float = 1e-3,
    enforce_simple: bool = True,
) -> np.ndarray:
    """
    Construct a curve between start_pt and end_pt using donor's lateral deviation profile.

    Steps:
      1) extract y(t) from donor
      2) resample and scale amplitude
      3) build points along target chord + perpendicular * y
      4) project inside polygon
      5) smooth in 2D a bit
      6) validate simplicity (optional)
    """
    a = np.asarray(start_pt, dtype=np.float64).reshape(2,)
    b = np.asarray(end_pt, dtype=np.float64).reshape(2,)
    occ = np.asarray(occluder, dtype=np.float64)
    poly = Polygon(occ.tolist())

    chord = b - a
    u = _normalize(chord)
    n = np.array([-u[1], u[0]])

    t, y = _signed_lateral_profile(donor_segment)
    y_rs = _resample_profile(t, y, n_samples)

    # scale donor amplitude relative to chord length
    L = np.linalg.norm(chord)
    y_rs = y_rs - np.mean(y_rs)
    y_rs = amp_scale * y_rs
    # optionally damp very large excursions
    max_abs = np.max(np.abs(y_rs)) if y_rs.size else 0.0
    if max_abs > 0:
        y_rs = y_rs * min(1.0, 0.25 * L / max_abs)

    y_sm = _smooth_1d(y_rs, win=smooth_win)

    tt = np.linspace(0.0, 1.0, n_samples)
    base = a[None, :] + tt[:, None] * chord[None, :]
    pts = base + y_sm[:, None] * n[None, :]

    pts = _project_inside(poly, pts, inset_eps=inset_eps)

    # light 2D smoothing (apply to x and y separately)
    pts[:, 0] = _smooth_1d(pts[:, 0], win=max(5, smooth_win // 2))
    pts[:, 1] = _smooth_1d(pts[:, 1], win=max(5, smooth_win // 2))

    # enforce exact endpoints
    pts[0] = a
    pts[-1] = b

    if enforce_simple:
        ls = LineString(pts.tolist())
        if not ls.is_simple:
            # if it self-intersects, caller should resample donor
            return np.empty((0, 2), dtype=np.float64)

    return pts
