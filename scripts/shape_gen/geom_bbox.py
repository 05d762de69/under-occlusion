from __future__ import annotations

from typing import Iterable, Tuple
import numpy as np


def compute_bbox(polys: Iterable[np.ndarray], base_grid: int, margin: int = 5) -> Tuple[int, int, int, int]:
    """
    Matches MATLAB bbox logic on pixel coordinates:

    minX = floor(min(allX) - margin), maxX = ceil(max(allX) + margin)
    minY = floor(min(allY) - margin), maxY = ceil(max(allY) + margin)

    Clips to [1, base_grid].

    Returns: (minX, minY, wBB, hBB)
    """
    xs = []
    ys = []
    for p in polys:
        p = np.asarray(p)
        if p.size == 0:
            continue
        xs.append(p[:, 0])
        ys.append(p[:, 1])

    if not xs:
        raise ValueError("No non-empty polygons provided for bbox computation.")

    allX = np.concatenate(xs)
    allY = np.concatenate(ys)

    minX = int(np.floor(np.min(allX) - margin))
    maxX = int(np.ceil(np.max(allX) + margin))
    minY = int(np.floor(np.min(allY) - margin))
    maxY = int(np.ceil(np.max(allY) + margin))

    minX = max(1, minX)
    minY = max(1, minY)
    maxX = min(base_grid, maxX)
    maxY = min(base_grid, maxY)

    wBB = maxX - minX + 1
    hBB = maxY - minY + 1

    return minX, minY, wBB, hBB
