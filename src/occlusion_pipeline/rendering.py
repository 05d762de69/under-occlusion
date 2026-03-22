from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


def render_contour_to_rgb(
    contour_xy: np.ndarray,
    image_size: int = 224,
    line_width: int = 1,
    invert_yaxis: bool = True,
) -> np.ndarray:
    """
    Render normalized contour coordinates into a filled black silhouette
    on a white RGB background.

    Parameters
    ----------
    contour_xy : np.ndarray
        Array of shape (N, 2) with normalized x/y coordinates.
    image_size : int
        Output image height/width in pixels.
    line_width : int
        Optional outline width. Keep small if you want a clean filled shape.
    invert_yaxis : bool
        Whether to invert y for plotting/image coordinates.

    Returns
    -------
    np.ndarray
        RGB uint8 image of shape (image_size, image_size, 3).
    """
    contour_xy = np.asarray(contour_xy, dtype=np.float32)
    if contour_xy.ndim != 2 or contour_xy.shape[1] != 2:
        raise ValueError("contour_xy must have shape (N, 2)")

    xy = contour_xy.copy()
    if invert_yaxis:
        xy[:, 1] = 1.0 - xy[:, 1]

    px = xy[:, 0] * (image_size - 1)
    py = xy[:, 1] * (image_size - 1)
    points = list(map(tuple, np.stack([px, py], axis=1)))

    # white background
    img = Image.new("RGB", (image_size, image_size), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # filled black silhouette
    if len(points) >= 3:
        draw.polygon(points, fill=(0, 0, 0), outline=(0, 0, 0))

    return np.asarray(img, dtype=np.uint8)


def save_rgb_image(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path)