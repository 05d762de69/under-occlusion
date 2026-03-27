from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw


Color = Union[Sequence[float], Sequence[int]]


def _to_rgb255(c: Color) -> Tuple[int, int, int]:
    """
    Accepts either [0..1] floats or [0..255] ints and returns (R,G,B) in 0..255.
    """
    c = list(c)
    if len(c) != 3:
        raise ValueError("Color must have length 3 (RGB).")

    # If any component is > 1, assume already in 0..255
    if any(v > 1 for v in c):
        r, g, b = (int(round(v)) for v in c)
    else:
        r, g, b = (int(round(255 * float(v))) for v in c)

    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    return (r, g, b)


def draw_and_save(
    polygons: List[np.ndarray],
    colors: List[Color],
    minX: int,
    minY: int,
    wBB: int,
    hBB: int,
    out_w: int,
    out_h: int,
    out_file: str | Path,
    supersample: int = 4,
) -> None:
    """
    Draw polygons (pixel coords in baseGrid space) inside the same bounding box,
    then save a PNG of exact size [out_h x out_w].

    Equivalent intent to MATLAB drawAndSave, but faster and deterministic.

    polygons: list of (N,2) arrays in [x,y] pixel coordinates
    colors: list of RGB (either 0..1 floats or 0..255 ints)
    minX, minY, wBB, hBB: bbox in baseGrid pixel coordinates
    out_w, out_h: final output image size
    supersample: render at higher resolution then downsample for smoother edges
    """
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if len(polygons) != len(colors):
        raise ValueError("polygons and colors must have same length.")

    ss = int(max(1, supersample))
    canvas_w = int(wBB * ss)
    canvas_h = int(hBB * ss)

    # White background
    img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    for poly, col in zip(polygons, colors):
        if poly is None:
            continue
        poly = np.asarray(poly)
        if poly.size == 0:
            continue
        if poly.ndim != 2 or poly.shape[1] != 2:
            raise ValueError(f"Polygon must be Nx2. Got shape {poly.shape}")

        # Shift into bbox space like MATLAB knowing that MATLAB does coords(:,1)-minX
        shifted = poly.astype(np.float64)
        shifted[:, 0] = shifted[:, 0] - float(minX)
        shifted[:, 1] = shifted[:, 1] - float(minY)

        shifted[:, 1] = float(hBB) - shifted[:, 1]


        # Supersample scaling
        shifted *= float(ss)

        # Pillow expects list of (x,y) tuples
        pts = [(float(x), float(y)) for x, y in shifted]

        draw.polygon(pts, fill=_to_rgb255(col))

    # Downsample to exact output size
    if (canvas_w, canvas_h) != (out_w, out_h):
        img = img.resize((int(out_w), int(out_h)), resample=Image.Resampling.LANCZOS)

    img.save(out_file, format="PNG")
