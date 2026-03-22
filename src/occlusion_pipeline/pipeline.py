from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .config import PipelineConfig
from .inference import run_model
from .io_utils import append_rows_to_csv, iter_case_jsonl_paths, latest_jsonl_row
from .modeling import discover_model_paths, load_model_bundle
from .occlusion import apply_multiple_occluders, generate_positions, sample_random_positions
from .preprocess import rgb_uint8_to_torch
from .rendering import render_contour_to_rgb, save_rgb_image


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_class_names(class_names_jsonl: Path) -> list[str]:
    with class_names_jsonl.open("r", encoding="utf-8") as f:
        return [json.loads(line)["class_name"] for line in f if line.strip()]


def get_class_name(idx: int, classes: list[str]) -> str:
    if idx < 0 or idx >= len(classes):
        return f"idx_{idx}"
    return classes[idx]


def load_case_record(case_jsonl_path: Path) -> Dict:
    row = latest_jsonl_row(case_jsonl_path)
    case_id = case_jsonl_path.stem
    contour_xy = np.asarray(row["shape_contour_xy"], dtype=np.float32)
    invert_yaxis = bool(row.get("plotting", {}).get("invert_yaxis", True))

    return {
        "case_id": case_id,
        "category": row.get("category"),
        "img_id": row.get("img_id"),
        "contour_xy": contour_xy,
        "invert_yaxis": invert_yaxis,
    }


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def apply_single_occluder(
    x: torch.Tensor,
    y0: int,
    x0: int,
    size: int,
    baseline_value: float = 0.0,
) -> torch.Tensor:
    x_occ = x.clone()
    x_occ[:, :, y0:y0 + size, x0:x0 + size] = baseline_value
    return x_occ


def process_single_case(
    case_jsonl_path: Path,
    model_paths: Dict[str, Path],
    cfg: PipelineConfig,
    classes: list[str],
    device: str = "cpu",
    selected_model_types: Optional[List[str]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    case = load_case_record(case_jsonl_path)
    case_id = case["case_id"]
    ground_truth_class_name = str(case["category"])

    rows_out: List[Dict] = []
    occluder_rows_out: List[Dict] = []

    for model_type, model_path in model_paths.items():
        if selected_model_types is not None and model_type not in selected_model_types:
            continue

        bundle = load_model_bundle(model_type, model_path, device=device)
        input_h, input_w = bundle.input_hw
        if input_h != input_w:
            raise ValueError(f"Non-square model input not currently supported: {bundle.input_hw}")

        image_size = input_h

        image_rgb = render_contour_to_rgb(
            contour_xy=case["contour_xy"],
            image_size=image_size,
            line_width=cfg.line_width,
            invert_yaxis=case["invert_yaxis"],
        )

        case_art_dir = cfg.artifacts_dir / case_id / model_type
        case_art_dir.mkdir(parents=True, exist_ok=True)
        save_rgb_image(image_rgb, case_art_dir / "baseline_render.png")

        x = rgb_uint8_to_torch(
            image_rgb=image_rgb,
            image_size=image_size,
            mean=cfg.mean,
            std=cfg.std,
            device=device,
        )

        baseline = run_model(bundle, x)
        baseline_target_class = int(baseline["pred_class"])
        baseline_target_logit = float(baseline["logits"][0, baseline_target_class].item())
        baseline_confidence = float(baseline["pred_prob"])

        target_class = baseline_target_class
        target_class_name = get_class_name(target_class, classes)

        baseline_pred_class = int(baseline["pred_class"])
        baseline_pred_class_name = get_class_name(baseline_pred_class, classes)
        is_correct_baseline = int(baseline_pred_class_name == ground_truth_class_name)

        rng = np.random.default_rng(cfg.seed)

        for occluder_size in cfg.occluder_sizes:
            stride = max(1, int(round(occluder_size * cfg.stride_fraction)))
            positions = generate_positions(
                h=image_size,
                w=image_size,
                size=occluder_size,
                stride=stride,
            )

            position_rows = []

            for y0, x0 in positions:
                x_occ_single = apply_single_occluder(
                    x=x,
                    y0=y0,
                    x0=x0,
                    size=occluder_size,
                    baseline_value=cfg.baseline_value,
                )

                post_single = run_model(bundle, x_occ_single)
                post_target_logit_single = float(
                    post_single["logits"][0, baseline_target_class].item()
                )
                target_logit_drop_single = baseline_target_logit - post_target_logit_single

                position_rows.append({
                    "case_id": case_id,
                    "model_type": model_type,
                    "occluder_size": occluder_size,
                    "stride": stride,
                    "y0": y0,
                    "x0": x0,
                    "y_center": y0 + occluder_size / 2.0,
                    "x_center": x0 + occluder_size / 2.0,
                    "baseline_target_logit": baseline_target_logit,
                    "post_occlusion_target_logit": post_target_logit_single,
                    "target_logit_drop": target_logit_drop_single,
                })

            position_df = pd.DataFrame(position_rows)

            if cfg.save_position_tables:
                position_df.to_csv(
                    case_art_dir / f"position_scores_size_{occluder_size}.csv",
                    index=False,
                )

            ranked_df = position_df.sort_values(
                ["target_logit_drop", "y0", "x0"],
                ascending=[False, True, True],
            ).copy()

            ranked_targeted_positions = []

            def overlaps_existing(y: int, x_: int) -> bool:
                for yy, xx in ranked_targeted_positions:
                    if not (
                        x_ + occluder_size <= xx
                        or xx + occluder_size <= x_
                        or y + occluder_size <= yy
                        or yy + occluder_size <= y
                    ):
                        return True
                return False

            for _, r in ranked_df.iterrows():
                y0 = int(r["y0"])
                x0 = int(r["x0"])
                if cfg.allow_overlap_targeted or not overlaps_existing(y0, x0):
                    ranked_targeted_positions.append((y0, x0))

            for placement_type in ["targeted", "random"]:
                for num_occluders in range(1, cfg.max_num_occluders + 1):
                    if placement_type == "targeted":
                        chosen_positions = ranked_targeted_positions[:num_occluders]
                    else:
                        chosen_positions = sample_random_positions(
                            all_positions=positions,
                            k=num_occluders,
                            patch_size=occluder_size,
                            rng=rng,
                            allow_overlap=cfg.allow_overlap_random,
                        )

                    for occluder_id, (y0, x0) in enumerate(chosen_positions, start=1):
                        occluder_rows_out.append({
                            "case_id": case_id,
                            "model_type": model_type,
                            "placement_type": placement_type,
                            "num_occluders": num_occluders,
                            "occluder_size": occluder_size,
                            "occluder_id": occluder_id,
                            "x0": int(x0),
                            "y0": int(y0),
                            "width": int(occluder_size),
                            "height": int(occluder_size),
                            "size": int(occluder_size),
                        })

                    x_occ = apply_multiple_occluders(
                        x=x,
                        coords=chosen_positions,
                        size=occluder_size,
                        baseline_value=cfg.baseline_value,
                    )

                    post = run_model(bundle, x_occ)
                    post_pred_class = int(post["pred_class"])
                    post_pred_class_name = get_class_name(post_pred_class, classes)
                    post_target_logit = float(post["logits"][0, baseline_target_class].item())

                    is_correct_post_occlusion = int(post_pred_class_name == ground_truth_class_name)
                    prediction_changed = int(post_pred_class != baseline_pred_class)

                    row = {
                        "case_id": case_id,
                        "model_type": model_type,
                        "placement_type": placement_type,
                        "num_occluders": num_occluders,
                        "occluder_size": occluder_size,
                        "target_class": target_class,
                        "target_class_name": target_class_name,
                        "baseline_target_logit": baseline_target_logit,
                        "post_occlusion_target_logit": post_target_logit,
                        "target_logit_drop": baseline_target_logit - post_target_logit,
                        "is_correct_baseline": is_correct_baseline,
                        "is_correct_post_occlusion": is_correct_post_occlusion,
                        "prediction_changed": prediction_changed,
                    }
                    rows_out.append(row)

            save_json(
                case_art_dir / f"metadata_size_{occluder_size}.json",
                {
                    "case_id": case_id,
                    "model_type": model_type,
                    "baseline_target_class": baseline_target_class,
                    "baseline_target_class_name": target_class_name,
                    "baseline_target_logit": baseline_target_logit,
                    "baseline_confidence": baseline_confidence,
                    "ground_truth_class_name": ground_truth_class_name,
                    "occluder_size": occluder_size,
                    "stride": stride,
                },
            )

    return rows_out, occluder_rows_out


def process_all_cases(
    cfg: PipelineConfig,
    device: str = "cpu",
    selected_case_ids: Optional[List[str]] = None,
    selected_model_types: Optional[List[str]] = None,
    flush_every_n_cases: int = 1,
) -> None:
    cfg.ensure_dirs()
    set_all_seeds(cfg.seed)

    classes = load_class_names(cfg.class_names_jsonl)

    model_paths = discover_model_paths(cfg.models_root)
    if not model_paths:
        raise RuntimeError(f"No ONNX models found in {cfg.models_root}")

    all_case_paths = list(iter_case_jsonl_paths(cfg.cases_root))
    if selected_case_ids is not None:
        selected_case_ids = set(selected_case_ids)
        all_case_paths = [p for p in all_case_paths if p.stem in selected_case_ids]

    buffer_rows: List[Dict] = []
    buffer_occluder_rows: List[Dict] = []
    n_done = 0

    for case_jsonl_path in all_case_paths:
        rows, occluder_rows = process_single_case(
            case_jsonl_path=case_jsonl_path,
            model_paths=model_paths,
            cfg=cfg,
            classes=classes,
            device=device,
            selected_model_types=selected_model_types,
        )
        buffer_rows.extend(rows)
        buffer_occluder_rows.extend(occluder_rows)
        n_done += 1

        if n_done % flush_every_n_cases == 0:
            append_rows_to_csv(buffer_rows, cfg.long_csv_path)
            append_rows_to_csv(buffer_occluder_rows, cfg.occluder_positions_csv_path)
            buffer_rows = []
            buffer_occluder_rows = []

    if buffer_rows:
        append_rows_to_csv(buffer_rows, cfg.long_csv_path)

    if buffer_occluder_rows:
        append_rows_to_csv(buffer_occluder_rows, cfg.occluder_positions_csv_path)