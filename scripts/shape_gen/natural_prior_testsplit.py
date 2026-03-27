# File: shape_gen/natural_prior_testsplit.py
#
# Option 2 pipeline. Learn a natural-fragment prior from TEST split IDs only,
# using contours stored in master_lib.mat, then propose completion segments
# conditioned on endpoints + occluder.
#
# Dependencies: numpy, scipy, scikit-learn, shapely
#
# You WILL need to adapt `iter_master_records()` to your exact master_lib.mat structure.
# Everything else is generic.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import math
import re
import csv
import numpy as np
from scipy.io import loadmat

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from shapely.geometry import LineString, Polygon


# ============================================================
# I/O: parse test_split.csv -> set(img_id)
# ============================================================

_ID_RE = re.compile(r"(\d+)\.png$", re.IGNORECASE)


def load_test_ids(test_split_csv: str | Path) -> set[int]:
    test_split_csv = Path(test_split_csv)
    ids: set[int] = set()

    with test_split_csv.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            path = row[0].strip()
            m = _ID_RE.search(path)
            if not m:
                continue
            ids.add(int(m.group(1)))
    if not ids:
        raise RuntimeError(f"No ids parsed from {test_split_csv}")
    return ids


# ============================================================
# master_lib.mat adapter
# ============================================================

@dataclass
class MasterRecord:
    img_id: int
    category: str
    contour_xy: np.ndarray  # (N,2) in whichever coord system you store (normalized or pixel)


def iter_master_records(master_lib_mat: str | Path) -> Iterable[MasterRecord]:
    """
    v7.3 (HDF5) reader for master_lib.mat.

    You may need to adapt the dataset names below to match your file:
      - STRUCT_ARRAY_KEY: name of the top-level struct array
      - field names: img_id, category, contour_xy/shape_contour_xy/contour_u

    This implementation supports the common MATLAB v7.3 layout:
      - a 1xN struct array stored as an HDF5 group, fields are datasets of refs
      - strings stored as refs to char arrays
      - numeric arrays stored directly or via refs
    """
    from pathlib import Path
    import numpy as np
    import h5py

    master_lib_mat = Path(master_lib_mat)

    # Try likely top-level keys. If none match, inspect with the notebook cell above.
    CANDIDATE_STRUCT_KEYS = ["stim", "records", "master_lib", "lib"]

    def _decode_mat_string(f: h5py.File, obj) -> str:
        """
        Decode MATLAB v7.3 strings. Handles:
          - dataset of uint16 codes
          - object ref to such a dataset
          - numpy bytes
        """
        if isinstance(obj, h5py.Reference):
            if not obj:
                return ""
            obj = f[obj]

        # obj can be Dataset
        if isinstance(obj, h5py.Dataset):
            data = obj[()]
            # MATLAB chars often stored as uint16
            if data.dtype.kind in ("u", "i"):
                # flatten column-major-ish and convert to chars
                flat = np.array(data).reshape(-1)
                try:
                    return "".join(chr(int(c)) for c in flat if int(c) != 0).strip()
                except Exception:
                    return str(flat)
            if data.dtype.kind == "S":
                try:
                    return data.tobytes().decode("utf-8", errors="ignore").strip()
                except Exception:
                    return str(data)
            return str(data)

        # raw python bytes
        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="ignore").strip()

        return str(obj)

    def _read_ref_array(f: h5py.File, ref_arr) -> List[h5py.Reference]:
        """
        MATLAB often stores struct fields as (1,N) arrays of object references.
        """
        a = np.asarray(ref_arr)
        a = a.reshape(-1)
        return [r for r in a]

    def _read_numeric(f: h5py.File, obj) -> np.ndarray:
        """
        Read numeric matrix from Dataset or reference to Dataset.
        """
        if isinstance(obj, h5py.Reference):
            if not obj:
                return np.zeros((0, 0), dtype=np.float64)
            obj = f[obj]
        if isinstance(obj, h5py.Dataset):
            return np.asarray(obj[()], dtype=np.float64)
        return np.asarray(obj, dtype=np.float64)

    def _pick_key(f: h5py.File) -> str:
        for k in CANDIDATE_STRUCT_KEYS:
            if k in f:
                return k
        # fallback: if only one non-meta key exists, take it
        keys = [k for k in f.keys() if not k.startswith("#")]
        if len(keys) == 1:
            return keys[0]
        raise RuntimeError(
            f"Could not find any of {CANDIDATE_STRUCT_KEYS} at HDF5 root. "
            f"Root keys: {list(f.keys())}. Use the inspect cell to locate the struct array."
        )

    def _field_exists(g: h5py.Group, name: str) -> bool:
        return name in g

    with h5py.File(str(master_lib_mat), "r") as f:
        STRUCT_ARRAY_KEY = _pick_key(f)
        g = f[STRUCT_ARRAY_KEY]

        if not isinstance(g, h5py.Group):
            raise RuntimeError(f"Expected group at '{STRUCT_ARRAY_KEY}', got {type(g)}")

        # Determine N: from a field dataset shape
        # pick img_id/imgId field
        img_field = "img_id" if _field_exists(g, "img_id") else ("imgId" if _field_exists(g, "imgId") else None)
        if img_field is None:
            # no img field found, list fields
            raise RuntimeError(f"No img_id/imgId field in group '{STRUCT_ARRAY_KEY}'. Fields: {list(g.keys())}")

        img_ds = g[img_field]
        img_refs = _read_ref_array(f, img_ds[()])
        N = len(img_refs)

        # category field
        cat_field = "category" if _field_exists(g, "category") else ("sil_class" if _field_exists(g, "sil_class") else None)

        # contour field options
        contour_fields = [nm for nm in ("contour_xy", "shape_contour_xy", "contour_u", "contour") if _field_exists(g, nm)]
        if not contour_fields:
            raise RuntimeError(
                f"No contour field found in '{STRUCT_ARRAY_KEY}'. "
                f"Tried contour_xy/shape_contour_xy/contour_u/contour. Fields: {list(g.keys())}"
            )
        contour_field = contour_fields[0]

        # Pull ref arrays (common MATLAB pattern)
        cat_refs = None
        if cat_field is not None:
            cat_refs = _read_ref_array(f, g[cat_field][()])

        contour_refs = _read_ref_array(f, g[contour_field][()])

        for i in range(N):
            # img_id
            img_obj = img_refs[i]
            img_arr = _read_numeric(f, img_obj).reshape(-1)
            if img_arr.size == 0:
                continue
            img_id = int(img_arr[0])

            # category
            cat = ""
            if cat_refs is not None:
                cat = _decode_mat_string(f, cat_refs[i])

            # contour
            contour = _read_numeric(f, contour_refs[i])
            contour = np.asarray(contour, dtype=np.float64)

            # MATLAB sometimes stores as 2xN
            if contour.ndim == 2 and contour.shape[0] == 2 and contour.shape[1] != 2:
                contour = contour.T
            contour = contour.reshape(-1, 2)

            yield MasterRecord(img_id=img_id, category=str(cat), contour_xy=contour)



# ============================================================
# Fragment extraction + representation (tangent angle)
# ============================================================

def _close_curve(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] < 2:
        return xy
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0]])
    return xy


def _arclength(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64)
    if xy.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    d = np.linalg.norm(xy[1:] - xy[:-1], axis=1)
    return np.concatenate([[0.0], np.cumsum(d)])


def resample_polyline(xy: np.ndarray, n: int) -> np.ndarray:
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if xy.shape[0] == 0:
        return xy.reshape(0, 2)
    if xy.shape[0] == 1:
        return np.repeat(xy, n, axis=0)
    s = _arclength(xy)
    L = float(s[-1])
    if L <= 1e-12:
        return np.repeat(xy[:1], n, axis=0)
    t = np.linspace(0.0, L, n)
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
            out[i] = (1 - a) * p0 + a * p1
    return out


def sample_fragments_from_closed_contour(
    contour_xy: np.ndarray,
    *,
    rng: np.random.Generator,
    n_frags: int,
    n_points: int = 64,
    frac_range: Tuple[float, float] = (0.25, 0.55),
) -> List[np.ndarray]:
    """
    Sample fragments along a CLOSED contour (wraparound). Returns list of (n_points,2).
    """
    c = _close_curve(contour_xy)
    # remove duplicate last point for indexing
    c = c[:-1]
    M = c.shape[0]
    if M < 10:
        return []

    frags: List[np.ndarray] = []
    for _ in range(n_frags):
        i0 = int(rng.integers(0, M))
        frac = float(rng.uniform(frac_range[0], frac_range[1]))
        seg_len = max(5, int(round(frac * M)))

        idxs = (i0 + np.arange(seg_len)) % M
        seg = c[idxs]
        seg_rs = resample_polyline(seg, n_points)
        frags.append(seg_rs)
    return frags


def _canonicalize_endpoints(xy: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Translate start to origin, rotate so end lies on +x axis, scale so chord length=1.
    """
    xy = np.asarray(xy, dtype=np.float64)
    p0 = xy[0].copy()
    p1 = xy[-1].copy()

    v = p1 - p0
    L = float(np.linalg.norm(v))
    if L <= 1e-12:
        return xy * 0.0, {"scale": 1.0, "angle": 0.0}

    # translate
    z = xy - p0[None, :]
    # rotate
    ang = math.atan2(v[1], v[0])
    ca, sa = math.cos(-ang), math.sin(-ang)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    z = (R @ z.T).T
    # scale
    z /= L
    return z, {"scale": L, "angle": ang}


def tangent_angle_representation(xy: np.ndarray) -> np.ndarray:
    """
    Returns a vector of length (N-1) of unwrapped tangent angles along the fragment,
    after endpoint canonicalization.
    """
    z, _ = _canonicalize_endpoints(xy)
    d = z[1:] - z[:-1]
    ang = np.arctan2(d[:, 1], d[:, 0])
    ang = np.unwrap(ang)
    ang = ang - ang[0]  # remove global heading
    return ang.astype(np.float64)


def reconstruct_from_tangent_angles(
    theta: np.ndarray,
    *,
    n_points: int,
) -> np.ndarray:
    """
    Reconstruct a canonical polyline from tangent angles.
    We integrate unit steps, then rescale so endpoints match (0,0)->(1,0).
    """
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if theta.shape[0] != n_points - 1:
        raise ValueError("theta must have length n_points-1")

    steps = np.stack([np.cos(theta), np.sin(theta)], axis=1)  # (N-1,2)
    xy = np.zeros((n_points, 2), dtype=np.float64)
    xy[1:] = np.cumsum(steps, axis=0)

    # affine correction to make end exactly (1,0)
    end = xy[-1].copy()
    L = float(np.linalg.norm(end))
    if L > 1e-12:
        xy /= L
    # rotate end to (1,0)
    end = xy[-1].copy()
    ang = math.atan2(end[1], end[0])
    ca, sa = math.cos(-ang), math.sin(-ang)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)
    xy = (R @ xy.T).T
    # ensure exact endpoints
    xy[0] = 0.0
    xy[-1] = np.array([1.0, 0.0], dtype=np.float64)
    return xy


def map_canonical_to_endpoints(canon_xy: np.ndarray, pA: np.ndarray, pB: np.ndarray) -> np.ndarray:
    """
    Map canonical curve with endpoints (0,0)->(1,0) onto image-space endpoints pA->pB.
    Similarity transform: scale + rotate + translate.
    """
    pA = np.asarray(pA, dtype=np.float64).reshape(2)
    pB = np.asarray(pB, dtype=np.float64).reshape(2)
    v = pB - pA
    L = float(np.linalg.norm(v))
    if L <= 1e-12:
        out = canon_xy.copy()
        out[:] = pA[None, :]
        return out

    ang = math.atan2(v[1], v[0])
    ca, sa = math.cos(ang), math.sin(ang)
    R = np.array([[ca, -sa], [sa, ca]], dtype=np.float64)

    out = (R @ canon_xy.T).T * L + pA[None, :]
    out[0] = pA
    out[-1] = pB
    return out


# ============================================================
# Prior model: PCA (+ optional GMM)
# ============================================================

@dataclass
class FragmentPrior:
    n_points: int
    pca: PCA
    gmm: Optional[GaussianMixture] = None
    z_scale: Optional[np.ndarray] = None  # optional PCA std scaling


def fit_prior_from_fragments(
    fragments_xy: List[np.ndarray],
    *,
    n_points: int = 64,
    n_latent: int = 20,
    use_gmm: bool = False,
    gmm_components: int = 20,
    rng_seed: int = 0,
) -> FragmentPrior:
    V = []
    for frag in fragments_xy:
        if frag.shape[0] != n_points:
            continue
        V.append(tangent_angle_representation(frag))
    if not V:
        raise RuntimeError("No fragments available to fit prior.")

    V = np.vstack(V)  # (n_frags, n_points-1)
    pca = PCA(n_components=min(n_latent, V.shape[1]), random_state=rng_seed)
    Z = pca.fit_transform(V)

    gmm = None
    if use_gmm:
        gmm = GaussianMixture(
            n_components=int(gmm_components),
            covariance_type="full",
            random_state=rng_seed,
            reg_covar=1e-6,
            max_iter=500,
        )
        gmm.fit(Z)

    return FragmentPrior(n_points=n_points, pca=pca, gmm=gmm)


def sample_latent(prior: FragmentPrior, rng: np.random.Generator) -> np.ndarray:
    d = prior.pca.n_components_
    if prior.gmm is None:
        return rng.normal(0.0, 1.0, size=(d,))
    z, _ = prior.gmm.sample(1)
    return z.reshape(-1)


def decode_latent_to_theta(prior: FragmentPrior, z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64).reshape(1, -1)
    v = prior.pca.inverse_transform(z).reshape(-1)
    # v is tangent-angle sequence length (n_points-1)
    return v


# ============================================================
# Conditioning: keep natural. enforce inside occluder. avoid self-intersection
# ============================================================

def _is_simple(curve_xy: np.ndarray) -> bool:
    if curve_xy.shape[0] < 4:
        return True
    return bool(LineString(curve_xy.tolist()).is_simple)


def _inside_occluder(curve_xy: np.ndarray, occ_poly: Polygon) -> bool:
    return bool(occ_poly.covers(LineString(curve_xy.tolist())))


def propose_segment_conditioned(
    *,
    prior: FragmentPrior,
    pA: np.ndarray,
    pB: np.ndarray,
    occluder_xy: np.ndarray,
    rng: np.random.Generator,
    max_tries: int = 200,
    # latent random-walk conditioning
    rw_steps: int = 40,
    rw_sigma: float = 0.25,
    # validity checks
    require_simple: bool = True,
) -> np.ndarray:
    """
    Samples from learned prior (test-only), then conditions by doing a low-dim random-walk in z
    until the curve lies inside occluder and (optionally) is simple.

    This is gradient-free and fast. You can make it "more conditioning" by increasing rw_steps
    and/or max_tries.
    """
    occ_poly = Polygon(np.asarray(occluder_xy, dtype=np.float64).reshape(-1, 2).tolist())

    d = prior.pca.n_components_
    best_curve = None

    for _ in range(int(max_tries)):
        z = sample_latent(prior, rng)

        # small conditioning loop in latent space
        for _step in range(int(rw_steps)):
            theta = decode_latent_to_theta(prior, z)
            canon = reconstruct_from_tangent_angles(theta, n_points=prior.n_points)
            curve = map_canonical_to_endpoints(canon, pA, pB)

            ok_inside = _inside_occluder(curve, occ_poly)
            ok_simple = (not require_simple) or _is_simple(curve)

            if ok_inside and ok_simple:
                return curve.astype(np.float64, copy=False)

            # random-walk update. shrink step size slowly if failing
            step_scale = rw_sigma * (0.85 ** _step)
            z = z + rng.normal(0.0, step_scale, size=(d,))

        # keep last curve as fallback candidate (optional)
        best_curve = curve

    if best_curve is None:
        raise RuntimeError("Failed to propose any segment.")
    return best_curve.astype(np.float64, copy=False)


# ============================================================
# End-to-end: build prior from test IDs, then sample proposals
# ============================================================

def build_prior_from_test_split(
    *,
    master_lib_mat: str | Path,
    test_split_csv: str | Path,
    rng: np.random.Generator,
    # fragment dataset size controls
    frags_per_shape: int = 30,
    n_points: int = 64,
    frac_range: Tuple[float, float] = (0.25, 0.55),
    # model
    n_latent: int = 20,
    use_gmm: bool = False,
    gmm_components: int = 20,
    rng_seed: int = 0,
) -> FragmentPrior:
    test_ids = load_test_ids(test_split_csv)

    fragments: List[np.ndarray] = []
    n_used = 0

    for rec in iter_master_records(master_lib_mat):
        if rec.img_id not in test_ids:
            continue

        frags = sample_fragments_from_closed_contour(
            rec.contour_xy,
            rng=rng,
            n_frags=int(frags_per_shape),
            n_points=int(n_points),
            frac_range=frac_range,
        )
        if frags:
            fragments.extend(frags)
            n_used += 1

    if not fragments:
        raise RuntimeError("No fragments extracted. Check master_lib adapter and test_split ids.")

    prior = fit_prior_from_fragments(
        fragments,
        n_points=int(n_points),
        n_latent=int(n_latent),
        use_gmm=bool(use_gmm),
        gmm_components=int(gmm_components),
        rng_seed=int(rng_seed),
    )

    return prior


# ============================================================
# Example usage (you can delete this block)
# ============================================================

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    MASTER = "/Users/I743312/Documents/occlusion-study/data/mats/master_lib.mat"
    TESTCSV = "/Users/I743312/Documents/MATLAB/Discriminative Network/natural/test_split.csv"

    prior = build_prior_from_test_split(
        master_lib_mat=MASTER,
        test_split_csv=TESTCSV,
        rng=rng,
        frags_per_shape=25,
        n_points=64,
        n_latent=20,
        use_gmm=False,  # flip True if you want multimodal fragments
        gmm_components=25,
        rng_seed=0,
    )

    # Suppose you already computed pA, pB and have occluder polygon xy in same coordinate system:
    # pA = np.array([xA, yA]); pB = np.array([xB, yB])
    # occluder_xy = np.array([...], dtype=float)

    # curve = propose_segment_conditioned(
    #     prior=prior,
    #     pA=pA,
    #     pB=pB,
    #     occluder_xy=occluder_xy,
    #     rng=rng,
    #     max_tries=200,
    #     rw_steps=50,
    #     rw_sigma=0.25,
    #     require_simple=True,
    # )
    # print(curve.shape)
