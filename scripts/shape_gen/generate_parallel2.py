from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import math
import numpy as np

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

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

from shape_gen.segments import extract_random_segment
from shape_gen.render import draw_and_save
from shape_gen.io_mat import unit_to_pixel
from shape_gen.donor_fit2 import DonorFitParams, donor_occluder_fit


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


def _make_valid_polygon(xy: np.ndarray) -> Polygon:
    poly = Polygon(np.asarray(xy, dtype=np.float64).tolist())
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _polyline_inside_occ_with_tolerance(
    polyline_xy: np.ndarray,
    occ_poly: Polygon,
    tol_px: float = 1.5,
    min_inside_frac: float = 0.97,
) -> bool:
    """
    Accept a polyline if almost all of it lies inside or on the occluder,
    allowing tiny numerical excursions near the boundary.
    """
    polyline_xy = np.asarray(polyline_xy, dtype=np.float64)
    if polyline_xy.ndim != 2 or polyline_xy.shape[1] != 2 or len(polyline_xy) < 2:
        return False

    occ_tol = occ_poly.buffer(float(tol_px))
    prepared_tol = prep(occ_tol)

    inside = np.array(
        [prepared_tol.covers(Point(float(x), float(y))) for x, y in polyline_xy],
        dtype=np.float64,
    )

    if inside.mean() >= float(min_inside_frac):
        return True

    return False


def _choose_visible_arc(
    sil0: np.ndarray,
    pA_true: np.ndarray,
    pB_true: np.ndarray,
    occ_poly: Polygon,
    tol_px: float = 1.5,
) -> np.ndarray:
    """
    Choose the arc from pB to pA that lies mostly outside the occluder.
    """
    n = sil0.shape[0]
    idxA = _nearest_index(sil0, pA_true)
    idxB = _nearest_index(sil0, pB_true)

    arc_AtoB_idx = _indices_between(idxA, idxB, n)
    arc_BtoA_idx = _indices_between(idxB, idxA, n)

    arc1 = _arc_points(sil0, arc_AtoB_idx)
    arc2 = _arc_points(sil0, arc_BtoA_idx)

    occ_tol = occ_poly.buffer(float(tol_px))
    prepared_occ_tol = prep(occ_tol)

    inside1 = np.array(
        [prepared_occ_tol.covers(Point(float(x), float(y))) for x, y in arc1],
        dtype=np.float64,
    ).mean()

    inside2 = np.array(
        [prepared_occ_tol.covers(Point(float(x), float(y))) for x, y in arc2],
        dtype=np.float64,
    ).mean()

    # Visible arc should be less inside the occluder
    if inside1 <= inside2:
        visible_arc = arc1[::-1].copy()  # pB -> pA
    else:
        visible_arc = arc2.copy()        # already pB -> pA

    visible_arc = np.asarray(visible_arc, dtype=np.float64)
    visible_arc[0] = pB_true
    visible_arc[-1] = pA_true
    return visible_arc


def _sample_fraction_band(target_fraction: float, rng: np.random.Generator) -> float:
    """
    Sample donor segment fraction from a range around the target.
    Wider at the extremes because those are harder.
    """
    tf = float(np.clip(target_fraction, 0.02, 0.90))

    if tf < 0.08:
        lo, hi = max(0.015, 0.50 * tf), min(0.16, 1.80 * tf)
    elif tf > 0.45:
        lo, hi = max(0.12, 0.75 * tf), min(0.90, 1.25 * tf)
    else:
        lo, hi = max(0.03, 0.75 * tf), min(0.70, 1.25 * tf)

    return float(rng.uniform(lo, hi))


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
    start_index: int = 1,
    fraction: float = 0.40,
    shrink_gamma: float = 0.88,
    max_shrink_iters: int = 30,
    smooth_win: int = 5,
    try_mirror: bool = True,
    final_n_samples_mode: str = "match_segment",
    supersample: int = 4,
    flush_every: int = 500,
    max_attempts_per_image: int = 50,
    require_valid: bool = True,
    save_invalid: bool = False,
    invalid_subdir: str = "_invalid",
    snap_intersections_to_vertices: bool = True,
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

    occ_poly = _make_valid_polygon(occ_xy)
    if occ_poly.is_empty or occ_poly.area <= 1e-9:
        raise RuntimeError("Occluder polygon is invalid or empty after repair.")

    prepared_occ = prep(occ_poly)

    pA_true = np.asarray(start_pt, dtype=np.float64).copy()
    pB_true = np.asarray(end_pt, dtype=np.float64).copy()

    visible_arc = _choose_visible_arc(
        sil0=sil,
        pA_true=pA_true,
        pB_true=pB_true,
        occ_poly=occ_poly,
        tol_px=1.5,
    )

    metas: List[CompletionMeta] = []
    out_files_xy: List[str] = []
    polygons_xy: List[np.ndarray] = []

    produced = 0
    fail_counts = {
        "donor_fit_empty": 0,
        "aligned_not_inside": 0,
        "aligned_not_simple": 0,
        "refit_empty": 0,
        "refit_rejected": 0,
        "final_poly_invalid": 0,
        "valid": 0,
    }

    target_attempts = int(n_images)
    i = int(start_index)
    end_i = int(start_index) + target_attempts - 1

    while i <= end_i:
        if ((i - int(start_index)) % 5) == 0:
            print(f"[generate] completion_index={i} / {end_i}")
        attempts = 0
        valid = False
        out_file: Optional[Path] = None
        new_shape_valid: Optional[np.ndarray] = None

        donor_idx: Optional[int] = None
        donor_class: Optional[str] = None
        donor_img_id: Optional[int] = None

        while attempts < max_attempts_per_image and (not valid):
            attempts += 1
            if (attempts % 25) == 0:
                print(f"[generate] i={i}, attempts={attempts}, fail_counts={fail_counts}")
            donor_idx, donor_class = _choose_donor_index(rng, records, classes, byClass, sil_class)
            donor = records[donor_idx]
            donor_img_id = getattr(donor, "img_id", None)

            alt_shape = unit_to_pixel(donor.contour_u, base_grid)

            donor_fraction = _sample_fraction_band(float(fraction), rng)
            random_segment, _ = extract_random_segment(
                alt_shape,
                fraction=donor_fraction,
                rng=rng,
            )

            n_samples = 100 if final_n_samples_mode == "matlab_100" else int(len(random_segment))
            n_samples = max(n_samples, 20)

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
                fail_counts["donor_fit_empty"] += 1
                continue

            aligned_segment = np.asarray(aligned_segment, dtype=np.float64)

            dA0 = np.sum((aligned_segment[0] - pA_true) ** 2)
            dA1 = np.sum((aligned_segment[-1] - pA_true) ** 2)
            if dA1 < dA0:
                aligned_segment = np.flipud(aligned_segment)

            aligned_segment = aligned_segment.copy()
            aligned_segment[0] = pA_true
            aligned_segment[-1] = pB_true

            if not _polyline_inside_occ_with_tolerance(
                aligned_segment,
                occ_poly=occ_poly,
                tol_px=1.5,
                min_inside_frac=0.97,
            ):
                fail_counts["aligned_not_inside"] += 1
                continue

            if not _is_simple_polyline(aligned_segment):
                fail_counts["aligned_not_simple"] += 1
                continue

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
                    fail_counts["refit_empty"] += 1
                    continue

                refined = np.asarray(refined, dtype=np.float64)
                refined[0] = pA_true
                refined[-1] = pB_true

                if _polyline_inside_occ_with_tolerance(
                    refined,
                    occ_poly=occ_poly,
                    tol_px=1.5,
                    min_inside_frac=0.97,
                ) and _is_simple_polyline(refined):
                    aligned_segment = refined
                else:
                    fail_counts["refit_rejected"] += 1
                    continue

            new_shape = np.vstack([aligned_segment, visible_arc[1:]])

            if not np.allclose(new_shape[0], new_shape[-1]):
                new_shape = np.vstack([new_shape, new_shape[0]])

            poly = Polygon(new_shape.tolist())
            if (not poly.is_valid) or (poly.area <= 1e-6):
                poly = poly.buffer(0)

            if poly.is_empty or (not poly.is_valid) or poly.area <= 1e-6:
                fail_counts["final_poly_invalid"] += 1
                continue

            new_shape = np.asarray(poly.exterior.coords, dtype=np.float64)

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

            fail_counts["valid"] += 1
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

        if ((i - int(start_index) + 1) % flush_every) == 0:
            done = (i - int(start_index) + 1)
            print(f"Attempts {done}/{target_attempts}. Valid produced {produced}. Last attempts={attempts}")

        i += 1
    print("FAIL COUNTS:", fail_counts)
    return metas, out_files_xy, polygons_xy, fail_counts


def save_metadata_jsonl(metas: List["CompletionMeta"], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m)) + "\n")
