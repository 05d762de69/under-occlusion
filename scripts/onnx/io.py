from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np


def list_images(folder: str | Path, exts: Sequence[str] = (".png", ".jpg", ".jpeg")) -> List[Path]:
    """
    Recursively list images under folder, sorted for deterministic ordering.
    """
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")

    paths: List[Path] = []
    for ext in exts:
        paths.extend(folder.rglob(f"*{ext}"))

    return sorted(paths)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_outputs(out_dir: str | Path, stem: str, paths: List[Path], logits: np.ndarray) -> None:
    """
    Save:
      - {stem}.npy (N x C)
      - {stem}_paths.txt (aligned file list)
    """
    out_dir = ensure_dir(out_dir)

    logits = np.asarray(logits)
    np.save(out_dir / f"{stem}.npy", logits.astype(np.float32))

    with (out_dir / f"{stem}_paths.txt").open("w", encoding="utf-8") as f:
        for p in paths:
            f.write(str(p) + "\n")
