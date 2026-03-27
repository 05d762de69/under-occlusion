# =========================
# PATCH: shape_gen/generate_parallel.py
# =========================
# Apply these edits to your existing file.
# I’m showing the full generate_completions function with only the relevant changes.
# Everything else in your file stays the same.

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


from shape_gen.generate3 import (
    _choose_donor_index,
    _validate_inside,
    _is_simple_polyline,
    _flexible_refit_spline_inside,
    _nearest_index,
    _indices_between,
    _arc_points,
    CompletionMeta,
)



import json
import math
import numpy as np

from shapely.geometry import Point, Polygon, LineString, MultiPoint, GeometryCollection
from shapely.prepared import prep
from shapely.ops import unary_union, nearest_points

from shape_gen.segments import extract_random_segment
from shape_gen.render import draw_and_save
from shape_gen.io_mat import unit_to_pixel

from shape_gen.donor_fit import DonorFitParams, donor_occluder_fit

# ... keep the rest of your helpers as-is ...

def _nearest_index(curve_xy: np.ndarray, p: np.ndarray) -> int:
    curve_xy = np.asarray(curve_xy, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    d2 = np.sum((curve_xy - p[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def _indices_between(a: int, b: int, n: int) -> np.ndarray:
    if a <= b:
        return np.arange(a, b + 1)
    return np.concatenate([np.arange(a, n), np.arange(0, b + 1)])


def _arc_points(curve_xy: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    curve_xy = np.asarray(curve_xy, dtype=np.float64)
    return curve_xy[idxs, :]

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

    # NEW. Start index for completion_index and filename uniqueness in parallel runs.
    start_index: int = 1,

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
) -> tuple[List["CompletionMeta"], List[str], List[np.ndarray]]:
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

    # Use the endpoints provided by the caller. This avoids Shapely intersection degeneracies.
    pA_true = np.asarray(start_pt, dtype=np.float64).copy()
    pB_true = np.asarray(end_pt, dtype=np.float64).copy()

    # Construct visible arc by nearest silhouette indexing to pA/pB.
    # This is robust even when Shapely returns 1 intersection due to tangency/degeneracy.
    sil0 = sil
    n = sil0.shape[0]

    # These helpers come from generate3.py, same as before.
    idxA = _nearest_index(sil0, pA_true)
    idxB = _nearest_index(sil0, pB_true)

    arc_AtoB = _indices_between(idxA, idxB, n)
    arc_BtoA = _indices_between(idxB, idxA, n)

    arc1 = _arc_points(sil0, arc_AtoB)
    arc2 = _arc_points(sil0, arc_BtoA)

    # Visible arc should be the arc mostly OUTSIDE the occluder.
    inside1 = np.array(
        [prepared_occ.contains(Point(float(x), float(y))) for x, y in arc1],
        dtype=np.float64
    ).mean()
    inside2 = np.array(
        [prepared_occ.contains(Point(float(x), float(y))) for x, y in arc2],
        dtype=np.float64
    ).mean()

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

    # CHANGED. n_images now means "how many attempt indices to consume in this call".
    target_attempts = int(n_images)
    i = int(start_index)
    end_i = int(start_index) + target_attempts - 1

    # CHANGED. Loop over fixed index range so parallel calls do not collide.
    while i <= end_i:
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

            if not _validate_inside(aligned_segment, prepared_occ):
                continue
            if not _is_simple_polyline(aligned_segment):
                continue

            dA0 = np.sum((aligned_segment[0] - pA_true) ** 2)
            dA1 = np.sum((aligned_segment[-1] - pA_true) ** 2)
            if dA1 < dA0:
                aligned_segment = np.flipud(aligned_segment)

            aligned_segment = aligned_segment.copy()
            aligned_segment[0] = pA_true
            aligned_segment[-1] = pB_true

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

            new_shape = np.vstack([aligned_segment, visible_arc[1:]])

            if not np.allclose(new_shape[0], new_shape[-1]):
                new_shape = np.vstack([new_shape, new_shape[0]])

            poly = Polygon(new_shape.tolist())
            if (not poly.is_valid) or (poly.area <= 1e-6):
                continue

            # CHANGED. Filename uses the globally unique i.
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
                done = (i - int(start_index) + 1)
                print(f"Attempts logged {done}/{target_attempts}. Valid produced {produced}...")
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

        # CHANGED. Progress based on attempts, not produced.
        if ((i - int(start_index) + 1) % flush_every) == 0:
            done = (i - int(start_index) + 1)
            print(f"Attempts {done}/{target_attempts}. Valid produced {produced}. Last attempts={attempts}")

        i += 1

    return metas, out_files_xy, polygons_xy

# ============================================================
# Metadata I/O
# ============================================================

def save_metadata_jsonl(metas: List["CompletionMeta"], path: str | Path) -> None:
    """
    Save completion metadata as JSONL, one JSON object per line.
    Uses dataclasses.asdict so it stays stable across versions.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m)) + "\n")
