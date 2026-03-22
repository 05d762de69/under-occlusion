from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from captum.attr import Occlusion


def compute_stride(size: int, fraction: float = 0.5) -> int:
    return max(1, int(round(size * fraction)))


def summarize_attribution_map(attr_map: torch.Tensor) -> np.ndarray:
    attr_np = attr_map.squeeze(0).detach().cpu().numpy()
    if attr_np.ndim == 3:
        attr_2d = attr_np.mean(axis=0)
    else:
        attr_2d = attr_np
    return attr_2d.astype(np.float32)


def generate_positions(h: int, w: int, size: int, stride: int) -> List[Tuple[int, int]]:
    ys = list(range(0, max(1, h - size + 1), stride))
    xs = list(range(0, max(1, w - size + 1), stride))

    if ys[-1] != h - size:
        ys.append(h - size)
    if xs[-1] != w - size:
        xs.append(w - size)

    positions = []
    for y0 in ys:
        for x0 in xs:
            positions.append((y0, x0))
    return positions


def run_captum_occlusion(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_idx: int,
    occluder_size: int,
    stride_fraction: float,
    baseline_value: float,
) -> tuple[np.ndarray, int]:
    stride = compute_stride(occluder_size, stride_fraction)

    occlusion = Occlusion(model)
    attr = occlusion.attribute(
        input_tensor,
        target=target_idx,
        sliding_window_shapes=(3, occluder_size, occluder_size),
        strides=(3, stride, stride),
        baselines=baseline_value,
    )
    attr_2d = summarize_attribution_map(attr)
    return attr_2d, stride


def rank_positions_from_occlusion_map(
    heatmap_2d: np.ndarray,
    positions: Sequence[Tuple[int, int]],
    patch_size: int,
    allow_overlap: bool = False,
) -> List[Tuple[int, int]]:
    scored = []
    for y0, x0 in positions:
        score = float(np.nanmean(heatmap_2d[y0:y0+patch_size, x0:x0+patch_size]))
        scored.append((y0, x0, score))

    scored.sort(key=lambda t: (-t[2], t[0], t[1]))

    selected: List[Tuple[int, int]] = []

    def overlaps_existing(y: int, x: int) -> bool:
        for yy, xx in selected:
            if not (x + patch_size <= xx or xx + patch_size <= x or
                    y + patch_size <= yy or yy + patch_size <= y):
                return True
        return False

    for y0, x0, _ in scored:
        if allow_overlap or not overlaps_existing(y0, x0):
            selected.append((y0, x0))

    return selected


def apply_multiple_occluders(
    x: torch.Tensor,
    coords: Sequence[Tuple[int, int]],
    size: int,
    baseline_value: float = 0.0,
) -> torch.Tensor:
    x_occ = x.clone()
    for (y0, x0) in coords:
        x_occ[:, :, y0:y0+size, x0:x0+size] = baseline_value
    return x_occ


def sample_random_positions(
    all_positions: Sequence[Tuple[int, int]],
    k: int,
    patch_size: int,
    rng: np.random.Generator,
    allow_overlap: bool = False,
) -> List[Tuple[int, int]]:
    all_positions = list(all_positions)
    rng.shuffle(all_positions)

    chosen: List[Tuple[int, int]] = []

    def overlaps_existing(y: int, x: int) -> bool:
        for yy, xx in chosen:
            if not (x + patch_size <= xx or xx + patch_size <= x or
                    y + patch_size <= yy or yy + patch_size <= y):
                return True
        return False

    for y0, x0 in all_positions:
        if allow_overlap or not overlaps_existing(y0, x0):
            chosen.append((y0, x0))
        if len(chosen) >= k:
            break

    if len(chosen) < k:
        raise RuntimeError("Could not sample enough random positions.")

    return chosen