# File: shape_gen/noise_fit.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from shapely.geometry import Point, Polygon, LineString
from shapely.prepared import prep


@dataclass
class NoiseFitParams:
    n_samples: int
    # smoothness: larger -> smoother. Interpreted as low-pass cutoff in FFT domain.
    # cutoff is in (0, 0.5], relative to Nyquist.
    cutoff_min: float = 0.04
    cutoff_max: float = 0.20

    # amplitude relative to chord length before clearance capping
    amp_rel_min: float = 0.03
    amp_rel_max: float = 0.20

    # optional endpoint derivative softening
    endpoint_taper_frac: float = 0.15  # fraction of curve length to taper to 0 at both ends

    # rejection sampling
    max_tries: int = 200

    # geometry constraints
    enforce_simple: bool = True
    inset_eps: float = 1e-3           # nudge inside boundary for numerical stability
    clearance_scale: float = 0.85     # keep within this fraction of local clearance

    # smoothing in spatial domain after synthesis
    spatial_smooth_win: int = 9       # odd, 0 disables


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


def _taper_window(n: int, frac: float) -> np.ndarray:
    """
    Smoothly forces y(0)=y(1)=0 by tapering near endpoints.
    Uses a half-cosine ramp at both ends.
    """
    frac = float(frac)
    if frac <= 0:
        return np.ones((n,), dtype=np.float64)

    m = int(max(1, round(frac * (n - 1))))
    w = np.ones((n,), dtype=np.float64)

    # start ramp
    r = np.linspace(0.0, 1.0, m + 1)
    w[: m + 1] = 0.5 - 0.5 * np.cos(np.pi * r)

    # end ramp
    w[-(m + 1) :] = w[: m + 1][::-1]
    return w


def _bandlimited_noise_profile(
    rng: np.random.Generator,
    n: int,
    cutoff: float,
    taper_frac: float,
) -> np.ndarray:
    """
    Generate band-limited zero-mean noise y(t) using FFT low-pass.
    """
    n = int(n)
    cutoff = float(cutoff)

    # white noise
    y = rng.standard_normal(n).astype(np.float64)

    # FFT low-pass
    Y = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(n, d=1.0)  # 0..0.5

    mask = freqs <= cutoff
    Y[~mask] = 0.0
    y_lp = np.fft.irfft(Y, n=n).astype(np.float64)

    # taper endpoints to enforce y(0)=y(1)=0 smoothly
    w = _taper_window(n, taper_frac)
    y_lp = y_lp * w

    # re-center
    y_lp = y_lp - float(np.mean(y_lp))
    y_lp[0] = 0.0
    y_lp[-1] = 0.0
    return y_lp


def _points_inside(poly_prepared, pts: np.ndarray) -> bool:
    for p in pts:
        P = Point(float(p[0]), float(p[1]))
        if not (poly_prepared.contains(P) or poly_prepared.touches(P)):
            return False
    return True


def _is_simple(pts: np.ndarray) -> bool:
    ls = LineString(pts.tolist())
    return bool(ls.is_simple)


def _cap_by_clearance(
    poly: Polygon,
    a: np.ndarray,
    u: np.ndarray,
    nvec: np.ndarray,
    base: np.ndarray,
    y: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Cap |y| using distance to boundary at the candidate point.
    This is not a hard guarantee. It reduces rejection rate.
    """
    y2 = y.copy()
    scale = float(scale)

    for i in range(len(y2)):
        p = base[i] + y2[i] * nvec
        P = Point(float(p[0]), float(p[1]))
        d = float(poly.exterior.distance(P))
        cap = max(0.0, scale * d)
        if abs(y2[i]) > cap and cap > 0:
            y2[i] = np.sign(y2[i]) * cap

    # enforce endpoints again
    y2[0] = 0.0
    y2[-1] = 0.0
    return y2


def noise_occluder_fit(
    *,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    occluder: np.ndarray,
    rng: np.random.Generator,
    params: NoiseFitParams,
) -> np.ndarray:
    """
    Option 1: band-limited noise profile generator with rejection sampling.

    Returns:
      pts: (n_samples, 2) curve points connecting start_pt->end_pt

    Failure:
      returns empty array shape (0,2)
    """
    a = np.asarray(start_pt, dtype=np.float64).reshape(2,)
    b = np.asarray(end_pt, dtype=np.float64).reshape(2,)
    occ = np.asarray(occluder, dtype=np.float64)

    poly = Polygon(occ.tolist())
    poly_p = prep(poly)

    chord = b - a
    L = float(np.linalg.norm(chord))
    if L < 1e-9:
        return np.empty((0, 2), dtype=np.float64)

    u = _normalize(chord)
    nvec = np.array([-u[1], u[0]], dtype=np.float64)

    n = int(params.n_samples)
    tt = np.linspace(0.0, 1.0, n)
    base = a[None, :] + tt[:, None] * chord[None, :]

    for _ in range(int(params.max_tries)):
        cutoff = float(rng.uniform(params.cutoff_min, params.cutoff_max))
        y = _bandlimited_noise_profile(
            rng=rng,
            n=n,
            cutoff=cutoff,
            taper_frac=float(params.endpoint_taper_frac),
        )

        # scale amplitude to chord length
        amp_rel = float(rng.uniform(params.amp_rel_min, params.amp_rel_max))
        y = y / (np.max(np.abs(y)) + 1e-12)
        y = y * (amp_rel * L)

        # cap by local clearance to reduce rejections
        y = _cap_by_clearance(
            poly=poly,
            a=a,
            u=u,
            nvec=nvec,
            base=base,
            y=y,
            scale=float(params.clearance_scale),
        )

        # candidate curve
        pts = base + y[:, None] * nvec[None, :]

        # optional spatial smoothing
        if int(params.spatial_smooth_win) > 1:
            win = int(params.spatial_smooth_win)
            pts[:, 0] = _moving_average_1d(pts[:, 0], win)
            pts[:, 1] = _moving_average_1d(pts[:, 1], win)
            pts[0] = a
            pts[-1] = b

        # nudge endpoints slightly inside to avoid boundary numeric issues if needed
        if float(params.inset_eps) > 0:
            inset = float(params.inset_eps)
            rep = poly.representative_point()
            rep_xy = np.array([rep.x, rep.y], dtype=np.float64)

            for j in (0, -1):
                P = Point(float(pts[j, 0]), float(pts[j, 1]))
                if poly_p.touches(P) and not poly_p.contains(P):
                    v = _normalize(rep_xy - pts[j])
                    pts[j] = pts[j] + inset * v

        # validate
        if not _points_inside(poly_p, pts):
            continue
        if params.enforce_simple and (not _is_simple(pts)):
            continue

        return pts

    return np.empty((0, 2), dtype=np.float64)
