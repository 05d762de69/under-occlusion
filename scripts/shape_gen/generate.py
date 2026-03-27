# File: shape_gen/generate.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from shape_gen.segments import extract_random_segment
from shape_gen.render import draw_and_save
from shape_gen.io_mat import unit_to_pixel

# NEW: donor-based fitting (natural shapes)
from shape_gen.donor_fit import DonorFitParams, donor_occluder_fit


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

    out_file: str
    attempts: int
    valid: bool


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


def _validate_inside(curve_xy: np.ndarray, prepared) -> bool:
    pts = np.asarray(curve_xy, dtype=np.float64)
    for p in pts:
        P = Point(float(p[0]), float(p[1]))
        if not (prepared.contains(P) or prepared.touches(P)):
            return False
    return True


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
) -> tuple[List[CompletionMeta], List[str], List[np.ndarray]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if save_invalid:
        (out_dir / invalid_subdir).mkdir(parents=True, exist_ok=True)

    if n_images <= 0:
        return [], [], []

    sil = np.asarray(silhouette, dtype=np.float64)
    occ_xy = np.asarray(occluder, dtype=np.float64)

    poly = Polygon(occ_xy.tolist())
    prepared = prep(poly)

    # Precompute silhouette arc outside occluder for stitching
    from matplotlib.path import Path as MplPath

    occ_path = MplPath(occ_xy)
    inside = occ_path.contains_points(sil, radius=1e-9)
    idxrm = np.where(inside)[0]
    if idxrm.size == 0:
        raise RuntimeError("No silhouette points inside occluder. Stitching would fail.")

    contour1 = np.vstack([sil[idxrm[-1] + 1 :, :], sil[: idxrm[0], :]])
    contour_endpoints = contour1[[0, -1], :]

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

            # empty => self-intersection or couldn't fit inside after shrinking
            if aligned_segment.size == 0:
                continue

            if not _validate_inside(aligned_segment, prepared):
                continue

            # Stitch, choose orientation to connect to visible arc endpoints
            rmse = np.sqrt(np.sum((aligned_segment[0, :] - contour_endpoints) ** 2, axis=1))
            if int(np.argmin(rmse)) == 0:
                aligned_segment = np.flipud(aligned_segment)

            new_shape = np.vstack([aligned_segment, contour1])

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

        # If we require valid, log failure but do not count toward produced
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
                    out_file=str(out_file) if out_file is not None else "",
                    attempts=int(attempts),
                    valid=False,
                )
            )
            if (len(metas) % flush_every) == 0:
                print(f"Attempts logged {len(metas)}. Valid produced {produced}/{target}...")
            i += 1
            continue

        # Count and store XY pack only if we saved a valid completion
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
                out_file=str(out_file) if out_file is not None else "",
                attempts=int(attempts),
                valid=bool(valid),
            )
        )

        if (produced % flush_every) == 0:
            print(f"Generated {produced}/{target} completions... (last attempts={attempts})")

        i += 1

    return metas, out_files_xy, polygons_xy


def save_metadata_jsonl(metas: List[CompletionMeta], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m)) + "\n")
