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
# --- ADD THESE IMPORTS NEAR THE TOP ---
from shapely.geometry import LineString, MultiPoint, GeometryCollection
from shapely.ops import unary_union

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

    pA_true, pB_true, visible_arc, prepared = _extract_intersections_AB_and_visible_arc(
        sil_xy=sil,
        occ_xy=occ_xy,
    )

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
            # orient donor segment so it runs pA -> pB
            dA0 = np.sum((aligned_segment[0] - pA_true) ** 2)
            dA1 = np.sum((aligned_segment[-1] - pA_true) ** 2)
            if dA1 < dA0:
                aligned_segment = np.flipud(aligned_segment)

            # force exact endpoints to eliminate closure chords
            aligned_segment = aligned_segment.copy()
            aligned_segment[0] = pA_true
            aligned_segment[-1] = pB_true

            # stitch polygon as: (inside completion) pA->pB, then (visible arc) pB->pA
            # drop the first point of visible_arc to avoid duplicating pB at the join
            new_shape = np.vstack([aligned_segment, visible_arc[1:]])

            # explicitly close polygon to avoid renderer closing with a stray chord
            if not np.allclose(new_shape[0], new_shape[-1]):
                new_shape = np.vstack([new_shape, new_shape[0]])

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

# --- ADD THESE HELPERS SOMEWHERE IN generate.py (e.g., above generate_completions) ---

def _pick_two_farthest_points_xy(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """xy: (M,2). returns (pA, pB) as (2,) arrays."""
    if xy.shape[0] < 2:
        raise RuntimeError("Need at least 2 intersection points.")
    if xy.shape[0] == 2:
        return xy[0], xy[1]
    # farthest pair
    d2 = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    return xy[i], xy[j]


def _nearest_index(curve_xy: np.ndarray, p: np.ndarray) -> int:
    d2 = np.sum((curve_xy - p[None, :]) ** 2, axis=1)
    return int(np.argmin(d2))


def _indices_between(a: int, b: int, n: int) -> np.ndarray:
    """Inclusive indices from a to b along circular polyline of length n."""
    if a <= b:
        return np.arange(a, b + 1)
    return np.concatenate([np.arange(a, n), np.arange(0, b + 1)])


def _arc_points(curve_xy: np.ndarray, idxs: np.ndarray) -> np.ndarray:
    return curve_xy[idxs, :]


def _extract_intersections_AB_and_visible_arc(
    sil_xy: np.ndarray,
    occ_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      pA, pB (2,) each.
      visible_arc_xy: polyline from pB -> pA (outside occluder).
      occ_poly_prepared: prepared polygon for point-in-occluder tests (returned as Shapely prepared).
    """
    sil = np.asarray(sil_xy, dtype=np.float64)
    occ = np.asarray(occ_xy, dtype=np.float64)

    # ensure closed for robust intersection
    sil_closed = sil
    if not np.allclose(sil_closed[0], sil_closed[-1]):
        sil_closed = np.vstack([sil_closed, sil_closed[0]])

    occ_poly = Polygon(occ.tolist())
    occ_boundary = occ_poly.boundary

    sil_line = LineString(sil_closed.tolist())
    inter = sil_line.intersection(occ_boundary)

    pts = []
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
        # If intersection is a LineString (overlap), fall back to endpoints of overlap.
        if inter.geom_type in ("LineString", "MultiLineString"):
            u = unary_union(inter)
            if u.geom_type == "LineString":
                coords = np.asarray(u.coords, dtype=np.float64)
                pts.append(coords[0].tolist())
                pts.append(coords[-1].tolist())
            elif u.geom_type == "MultiLineString":
                # collect all endpoints, then take farthest pair
                for ls in u.geoms:
                    coords = np.asarray(ls.coords, dtype=np.float64)
                    pts.append(coords[0].tolist())
                    pts.append(coords[-1].tolist())

    P = np.asarray(pts, dtype=np.float64)
    if P.shape[0] < 2:
        raise RuntimeError(f"Expected >=2 intersection points, got {P.shape[0]}.")

    pA, pB = _pick_two_farthest_points_xy(P)

    # snap A/B to nearest sampled silhouette vertex indices.
    # This keeps indexing stable while still using true geometric intersection to choose endpoints.
    n = sil.shape[0]
    idxA = _nearest_index(sil, pA)
    idxB = _nearest_index(sil, pB)
    pA = sil[idxA].copy()
    pB = sil[idxB].copy()

    # build both arcs between idxA and idxB on the sampled silhouette
    arc_AtoB = _indices_between(idxA, idxB, n)
    arc_BtoA = _indices_between(idxB, idxA, n)

    arc1 = _arc_points(sil, arc_AtoB)
    arc2 = _arc_points(sil, arc_BtoA)

    # decide which arc is OUTSIDE occluder (visible).
    # Using Shapely prepared polygon for robust interior test.
    prepared = prep(occ_poly)
    inside1 = np.array([prepared.contains(Point(float(x), float(y))) for x, y in arc1], dtype=np.float64).mean()
    inside2 = np.array([prepared.contains(Point(float(x), float(y))) for x, y in arc2], dtype=np.float64).mean()

    # occluded arc is the one with higher inside fraction.
    if inside1 >= inside2:
        occluded_arc = arc1
        visible_arc = arc2
        # arc2 runs idxB -> idxA, so it's already pB -> pA order (desired)
    else:
        occluded_arc = arc2
        visible_arc = arc1
        # arc1 runs idxA -> idxB, we want pB -> pA so reverse it
        visible_arc = visible_arc[::-1].copy()

    # enforce exact endpoints on visible arc
    visible_arc[0] = pB
    visible_arc[-1] = pA

    return pA, pB, visible_arc.astype(np.float64, copy=False), prepared
def save_metadata_jsonl(metas: List[CompletionMeta], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(asdict(m)) + "\n")
