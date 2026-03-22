from __future__ import annotations

from occlusion_pipeline.config import PipelineConfig
from occlusion_pipeline.pipeline import process_all_cases


def main() -> None:
    cfg = PipelineConfig(
        occluder_sizes=[8, 16, 32, 64],
        max_num_occluders=8,
        stride_fraction=0.5,
        baseline_value=0.0,
        allow_overlap_targeted=False,
        allow_overlap_random=False,
        save_variant_images=False,
        save_occlusion_arrays=False,
        save_position_tables=True,
    )

    process_all_cases(
        cfg=cfg,
        device="cpu",
        selected_case_ids=None,
        selected_model_types=None,
        flush_every_n_cases=1,
    )


if __name__ == "__main__":
    main()