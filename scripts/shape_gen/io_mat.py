from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class GeneratedCase:
    silhouette_u: np.ndarray  # (N,2) floats in [0,1]
    occluder_u: np.ndarray    # (M,2) floats in [0,1] or empty (0,2)
    base_grid: int
    sil_class: str


def _as_2col_array(x) -> np.ndarray:
    """
    Convert various MATLAB-loaded shapes into a (N,2) float array.
    """
    arr = np.asarray(x)

    # Squeeze singleton dims
    arr = np.squeeze(arr)

    # Common cases:
    # - (N,2)
    # - (2,N) from MATLAB transpose conventions
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for contour, got shape {arr.shape}")

    if arr.shape[1] == 2:
        out = arr
    elif arr.shape[0] == 2:
        out = arr.T
    else:
        raise ValueError(f"Contour must be Nx2 or 2xN. Got shape {arr.shape}")

    out = out.astype(np.float64, copy=False)
    return out


def _try_load_classic_mat(path: Path) -> Optional[dict]:
    try:
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError("scipy is required for classic .mat files") from e

    try:
        d = loadmat(path, squeeze_me=True, struct_as_record=False)
        return d
    except Exception:
        return None


def _try_load_v73_mat(path: Path) -> Optional["h5py.File"]:
    try:
        import h5py
    except Exception:
        return None

    try:
        f = h5py.File(path, "r")
        # v7.3 files are HDF5. Classic MAT files will fail to open here.
        return f
    except Exception:
        return None


def _extract_from_classic(d: dict) -> GeneratedCase:
    if "stim" not in d:
        raise ValueError("MAT file must contain variable 'stim'.")

    stim = d["stim"]

    # stim.natural.animal(1).c and .o
    try:
        natural = stim.natural
        animal = natural.animal
    except Exception as e:
        raise ValueError("Expected stim.natural.animal in MAT file.") from e

    # animal can be a single struct or array of structs
    if isinstance(animal, np.ndarray):
        animal0 = np.ravel(animal)[0]
    else:
        animal0 = animal

    if not hasattr(animal0, "c"):
        raise ValueError("Expected stim.natural.animal(1).c")
    sil_u = _as_2col_array(animal0.c)

    occ_u = np.zeros((0, 2), dtype=np.float64)
    if hasattr(animal0, "o"):
        try:
            if animal0.o is not None and np.size(animal0.o) > 0:
                occ_u = _as_2col_array(animal0.o)
        except Exception:
            occ_u = np.zeros((0, 2), dtype=np.float64)

    base_grid = 256
    sil_class = ""

    if hasattr(stim, "info"):
        info = stim.info
        if hasattr(info, "gridlen"):
            try:
                base_grid = int(np.squeeze(info.gridlen))
            except Exception:
                base_grid = 256
        if hasattr(info, "category"):
            try:
                sil_class = str(np.squeeze(info.category))
            except Exception:
                sil_class = ""

    return GeneratedCase(
        silhouette_u=sil_u,
        occluder_u=occ_u,
        base_grid=base_grid,
        sil_class=sil_class,
    )


def _read_h5_str(x) -> str:
    # h5py can store MATLAB strings in various ways
    import numpy as np

    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore")

    arr = np.array(x)
    arr = np.squeeze(arr)

    if arr.dtype.kind in {"S", "U"}:
        return str(arr)

    # Sometimes it's stored as uint16 char codes
    if arr.dtype.kind in {"u", "i"}:
        try:
            return "".join(chr(int(c)) for c in arr.flatten())
        except Exception:
            return ""

    return ""


def _extract_from_v73(f) -> GeneratedCase:
    """
    Minimal v7.3 reader for the expected fields. This handles common MATLAB HDF5 layouts,
    but if your file structure differs we can adapt quickly after you run Cell 2 and show errors.
    """
    import numpy as np

    if "stim" not in f:
        raise ValueError("MAT v7.3 file must contain group 'stim'.")

    stim = f["stim"]

    # Typical v7.3 layout is nested groups:
    # stim/natural/animal/c and stim/natural/animal/o
    # Animal can be an object array. We will try several common layouts.
    def get_dataset(path_candidates):
        for p in path_candidates:
            if p in f:
                return f[p]
            if p.startswith("stim/") and p in stim:
                return stim[p.replace("stim/", "")]
        return None

    # Try common paths
    c_ds = None
    o_ds = None

    # First attempt: stim/natural/animal/c
    if "natural" in stim:
        natural = stim["natural"]
        if "animal" in natural:
            animal = natural["animal"]
            # animal can be group with datasets c/o, or references
            if isinstance(animal, dict) or hasattr(animal, "keys"):
                if "c" in animal:
                    c_ds = animal["c"]
                if "o" in animal:
                    o_ds = animal["o"]

    # Fallback: direct paths
    if c_ds is None:
        c_ds = get_dataset(["stim/natural/animal/c", "stim/natural/animal/0/c"])
    if o_ds is None:
        o_ds = get_dataset(["stim/natural/animal/o", "stim/natural/animal/0/o"])

    if c_ds is None:
        raise ValueError("Could not locate stim.natural.animal.c in v7.3 file.")

    sil_u = np.array(c_ds)
    sil_u = _as_2col_array(sil_u)

    occ_u = np.zeros((0, 2), dtype=np.float64)
    if o_ds is not None:
        try:
            tmp = np.array(o_ds)
            if np.size(tmp) > 0:
                occ_u = _as_2col_array(tmp)
        except Exception:
            occ_u = np.zeros((0, 2), dtype=np.float64)

    base_grid = 256
    sil_class = ""

    if "info" in stim:
        info = stim["info"]
        if "gridlen" in info:
            try:
                base_grid = int(np.squeeze(np.array(info["gridlen"])))
            except Exception:
                base_grid = 256
        if "category" in info:
            try:
                sil_class = _read_h5_str(info["category"])
            except Exception:
                sil_class = ""

    return GeneratedCase(
        silhouette_u=sil_u,
        occluder_u=occ_u,
        base_grid=base_grid,
        sil_class=sil_class,
    )


def load_generated_case(mat_path: str | Path) -> GeneratedCase:
    """
    Loads your generated MAT file and returns unit-coordinate silhouette and occluder.
    Works with classic MAT and v7.3 MAT.
    """
    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"Not found: {mat_path}")

    d = _try_load_classic_mat(mat_path)
    if d is not None:
        return _extract_from_classic(d)

    f = _try_load_v73_mat(mat_path)
    if f is not None:
        try:
            return _extract_from_v73(f)
        finally:
            f.close()

    raise RuntimeError(
        "Could not read MAT file. Install scipy for classic MAT, and h5py for v7.3 MAT."
    )


def unit_to_pixel(u_xy: np.ndarray, base_grid: int) -> np.ndarray:
    """
    MATLAB-equivalent mapping:
    pix = round(u * (baseGrid - 1) + 1)
    Returns int32 array (N,2).
    """
    u_xy = np.asarray(u_xy, dtype=np.float64)
    pix = np.rint(u_xy * (base_grid - 1) + 1.0).astype(np.int32)
    return pix
