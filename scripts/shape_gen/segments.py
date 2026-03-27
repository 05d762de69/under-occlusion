from __future__ import annotations

from typing import Tuple
import numpy as np


def extract_random_segment(
    shape_xy: np.ndarray,
    fraction: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Python port of your MATLAB extract_random_segment.

    Inputs:
      - shape_xy: Nx2
      - fraction: (0,1]
      - rng: numpy Generator (for reproducibility)

    Returns:
      - segment: Kx2
      - segment_idx: indices into the original shape (int array)
    """
    if not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1].")

    shape = np.asarray(shape_xy, dtype=np.float64)
    if shape.ndim != 2 or shape.shape[1] != 2:
        raise ValueError(f"shape_xy must be Nx2. Got {shape.shape}")
    if shape.shape[0] < 2:
        return shape.copy(), np.arange(shape.shape[0], dtype=np.int64)

    d = np.diff(shape, axis=0)
    dists = np.sqrt(np.sum(d**2, axis=1))
    arc = np.concatenate([[0.0], np.cumsum(dists)])
    total = float(arc[-1])

    desired = fraction * total
    max_start = max(0.0, total - desired)

    start_dist = rng.random() * max_start
    end_dist = start_dist + desired

    # MATLAB: first index where arc_length >= target
    start_idx = int(np.searchsorted(arc, start_dist, side="left"))
    end_idx = int(np.searchsorted(arc, end_dist, side="left"))

    start_idx = min(start_idx, shape.shape[0] - 1)
    end_idx = min(max(end_idx, start_idx), shape.shape[0] - 1)

    idx = np.arange(start_idx, end_idx + 1, dtype=np.int64)
    seg = shape[idx, :]
    return seg, idx
