from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


def pack_xy(polygons: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Pack list of (Ni,2) polygons into:
      xy_all: (sum Ni, 2)
      offsets: (N+1,) with offsets[i]:offsets[i+1] selecting polygon i
    """
    if len(polygons) == 0:
        raise ValueError("No polygons to pack")

    lens = np.array([p.shape[0] for p in polygons], dtype=np.int64)
    if np.any(lens <= 0):
        raise ValueError("Found empty polygon")

    offsets = np.zeros((len(polygons) + 1,), dtype=np.int64)
    offsets[1:] = np.cumsum(lens)

    xy_all = np.vstack([np.asarray(p, dtype=np.float32) for p in polygons])
    if xy_all.ndim != 2 or xy_all.shape[1] != 2:
        raise ValueError(f"Packed xy_all must be (M,2), got {xy_all.shape}")

    return xy_all, offsets





def save_xy_npz(
    path: str | Path,
    *,
    out_files: List[str],
    polygons: List[np.ndarray],
    base_grid: int,
    matlab_1_indexed: bool = True,
) -> None:
    """
    Save completion polygons in a compact NPZ for heatmap construction.

    Parameters
    ----------
    path : output .npz path
    out_files : list of PNG paths (same order as inference paths)
    polygons : list of (Ni x 2) arrays, pixel coords
    base_grid : original grid size (e.g. 256)
    matlab_1_indexed "True" if coords are 1-based (MATLAB-style)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # store as object array (ragged polygons)
    polys = np.empty(len(polygons), dtype=object)
    for i, p in enumerate(polygons):
        polys[i] = np.asarray(p, dtype=np.float32)

    np.savez_compressed(
        path,
        polygons=polys,
        out_files=np.array(out_files, dtype=object),
        base_grid=int(base_grid),
        matlab_1_indexed=bool(matlab_1_indexed),
    )


def load_xy_npz(path: str | Path) -> tuple[list[str], np.ndarray, np.ndarray, int, bool]:
    """
    Returns out_files, xy_all, offsets, base_grid, matlab_1_indexed
    """
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    out_files = [str(x) for x in z["out_files"].tolist()]
    xy_all = z["xy"].astype(np.float32)
    offsets = z["offsets"].astype(np.int64)
    base_grid = int(z["base_grid"])
    matlab_1_indexed = bool(z["matlab_1_indexed"])
    return out_files, xy_all, offsets, base_grid, matlab_1_indexed


def unpack_one(xy_all: np.ndarray, offsets: np.ndarray, i: int) -> np.ndarray:
    a = int(offsets[i])
    b = int(offsets[i + 1])
    return xy_all[a:b].astype(np.float64)
