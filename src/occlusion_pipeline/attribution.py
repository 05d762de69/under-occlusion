from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from captum.attr import IntegratedGradients, Saliency

from .inference import run_torch_forward
from .modeling import ModelBundle


def compute_attribution_map(
    bundle: ModelBundle,
    x_torch: torch.Tensor,
    target_class: int,
    method: str = "integrated_gradients",
    ig_steps: int = 32,
) -> np.ndarray:
    x_torch = x_torch.clone().detach().requires_grad_(True)

    def forward_fn(inp: torch.Tensor) -> torch.Tensor:
        return run_torch_forward(bundle, inp)

    if method == "integrated_gradients":
        attr = IntegratedGradients(forward_fn).attribute(
            x_torch,
            target=target_class,
            n_steps=ig_steps,
        )
    elif method == "saliency":
        attr = Saliency(forward_fn).attribute(
            x_torch,
            target=target_class,
        )
    else:
        raise ValueError(f"Unknown attribution method: {method}")

    attr_np = attr.detach().cpu().numpy()[0]  # (C, H, W)
    # collapse channels to positive importance map
    attr_map = np.abs(attr_np).mean(axis=0)
    return attr_map.astype(np.float32)


def _window_score_map(attr_map: np.ndarray, patch_size: int) -> np.ndarray:
    h, w = attr_map.shape
    out_h = h - patch_size + 1
    out_w = w - patch_size + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("patch_size is larger than attribution map dimensions")

    scores = np.zeros((out_h, out_w), dtype=np.float32)
    for y in range(out_h):
        for x in range(out_w):
            scores[y, x] = float(attr_map[y:y+patch_size, x:x+patch_size].sum())
    return scores


def select_targeted_patch_positions(
    attr_map: np.ndarray,
    patch_size: int,
    k: int,
    percentile: float = 90.0,
    allow_overlap: bool = False,
) -> List[Tuple[int, int]]:
    scores = _window_score_map(attr_map, patch_size)
    threshold = np.percentile(scores, percentile)

    candidates = np.argwhere(scores >= threshold)
    if len(candidates) == 0:
        candidates = np.argwhere(np.ones_like(scores, dtype=bool))

    scored = [(int(y), int(x), float(scores[y, x])) for y, x in candidates]
    scored.sort(key=lambda t: (-t[2], t[0], t[1]))

    selected: List[Tuple[int, int]] = []

    def overlaps_existing(y: int, x: int) -> bool:
        for yy, xx in selected:
            if not (x + patch_size <= xx or xx + patch_size <= x or
                    y + patch_size <= yy or yy + patch_size <= y):
                return True
        return False

    for y, x, _ in scored:
        if allow_overlap or not overlaps_existing(y, x):
            selected.append((y, x))
        if len(selected) >= k:
            break

    return selected