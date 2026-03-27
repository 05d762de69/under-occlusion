# File: scripts/onnx/infer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from scripts.onnx.preprocess import PreprocessConfig, load_image_rgb, preprocess


@dataclass
class OnnxInferConfig:
    batch_size: int = 64
    use_softmax: bool = False  # applies ONLY to the first output (typically logits)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def _get_io_names(session) -> Tuple[str, List[str]]:
    in_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]
    return in_name, out_names


def infer_folder_onnx(
    model_path: Union[str, Path],
    image_paths: List[Path],
    *,
    preprocess_cfg: PreprocessConfig,
    infer_cfg: OnnxInferConfig,
    providers: Optional[List[str]] = None,
    output_names: Optional[Sequence[str]] = None,
    return_dict: bool = False,
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """
    Run ONNX inference over a folder/list of images.

    Backwards compatible behavior:
      - If output_names is None and return_dict is False:
          returns the FIRST output as an array (N x C) exactly like before.
      - Otherwise:
          returns a dict {output_name: array}, stacking batches.

    Notes:
      - infer_cfg.use_softmax is applied ONLY to the first requested output
        (usually logits). Features/embeddings should remain raw.
    """
    try:
        import onnxruntime as ort
    except Exception as e:
        raise RuntimeError("Missing dependency: onnxruntime. Install it in your env.") from e

    model_path = str(model_path)

    if providers is None:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, providers=providers)
    in_name, available_outs = _get_io_names(sess)

    # Decide which outputs to fetch
    if output_names is None:
        requested = [available_outs[0]]  # old behavior: first output only
    else:
        requested = list(output_names)
        missing = [n for n in requested if n not in available_outs]
        if missing:
            raise ValueError(
                f"Requested output(s) not found: {missing}. "
                f"Available outputs: {available_outs}"
            )

    n = len(image_paths)
    if n == 0:
        raise ValueError("No images found.")

    # Load first image to infer channel layout
    x0 = preprocess(load_image_rgb(str(image_paths[0])), preprocess_cfg)
    if preprocess_cfg.channels_first:
        c, h, w = x0.shape
        batch_shape = (int(infer_cfg.batch_size), c, h, w)
    else:
        h, w, c = x0.shape
        batch_shape = (int(infer_cfg.batch_size), h, w, c)

    bs = int(infer_cfg.batch_size)

    # Collect per-output batch lists
    out_lists: Dict[str, List[np.ndarray]] = {name: [] for name in requested}

    for start in range(0, n, bs):
        end = min(start + bs, n)
        cur_bs = end - start

        batch = np.zeros(batch_shape, dtype=np.float32)
        for i, p in enumerate(image_paths[start:end]):
            img = load_image_rgb(str(p))
            xi = preprocess(img, preprocess_cfg)
            batch[i] = xi

        batch = batch[:cur_bs]

        ys = sess.run(requested, {in_name: batch})

        for j, name in enumerate(requested):
            y = np.asarray(ys[j], dtype=np.float32)

            # Softmax only for the first requested output (usually logits)
            if infer_cfg.use_softmax and j == 0:
                if y.ndim != 2:
                    raise RuntimeError(f"Softmax expects 2D logits (N x C). Got {y.shape} for '{name}'")
                y = _softmax(y, axis=1)

            out_lists[name].append(y)

    # Stack batches for each output
    out_dict: Dict[str, np.ndarray] = {k: np.vstack(v) for k, v in out_lists.items()}

    # Return format
    if (output_names is None) and (not return_dict):
        # old behavior
        return out_dict[requested[0]]
    return out_dict
