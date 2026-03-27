from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image


@dataclass
class PreprocessConfig:
    size: Tuple[int, int] = (224, 224)  # (H, W)
    to_bgr: bool = False
    scale_01: bool = True
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)  # ImageNet default
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)   # ImageNet default
    channels_first: bool = True


def load_image_rgb(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def preprocess(img_u8: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """
    Returns float32 tensor either CHW or HWC.
    Default matches common ResNet50 PyTorch preprocessing.
    We will adapt to MATLAB once you share getActivations.m.
    """
    img = Image.fromarray(img_u8)
    img = img.resize((cfg.size[1], cfg.size[0]), resample=Image.BILINEAR)
    x = np.asarray(img, dtype=np.float32)

    if cfg.to_bgr:
        x = x[:, :, ::-1]

    if cfg.scale_01:
        x = x / 255.0

    mean = np.array(cfg.mean, dtype=np.float32)
    std = np.array(cfg.std, dtype=np.float32)
    x = (x - mean) / std

    if cfg.channels_first:
        x = np.transpose(x, (2, 0, 1))  # CHW

    return x.astype(np.float32)
