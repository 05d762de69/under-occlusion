from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


def build_preprocess(image_size: int, mean, std):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def rgb_uint8_to_torch(
    image_rgb: np.ndarray,
    image_size: int,
    mean,
    std,
    device: str = "cpu",
) -> torch.Tensor:
    pil_img = Image.fromarray(image_rgb)
    preprocess = build_preprocess(image_size=image_size, mean=mean, std=std)
    x = preprocess(pil_img).unsqueeze(0).to(device)
    return x