from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass
class PipelineConfig:
    cases_root: Path = Path("/data/storage-occ-v2/repos/monte-carlo-selection/data/cases/occlusion")
    models_root: Path = Path("/data/storage-occ-v2/repos/monte-carlo-selection/data/models")
    output_root: Path = Path("/data/storage-occ-v2/repos/monte-carlo-selection/results/part1_occlusion_robustness")
    class_names_jsonl: Path = Path("/data/storage-occ-v2/repos/monte-carlo-selection/data/class_names.jsonl")

    seed: int = 123
    line_width: int = 3
    default_input_size: int = 224

    occluder_sizes: Sequence[int] = field(default_factory=lambda: [8, 16, 32, 64])
    max_num_occluders: int = 8

    stride_fraction: float = 0.5
    baseline_value: float = 0.0

    mean: Sequence[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: Sequence[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])

    allow_overlap_targeted: bool = False
    allow_overlap_random: bool = False

    save_variant_images: bool = False
    save_occlusion_arrays: bool = False
    save_position_tables: bool = True

    @property
    def artifacts_dir(self) -> Path:
        return self.output_root / "artifacts"

    @property
    def tables_dir(self) -> Path:
        return self.output_root / "tables"

    @property
    def long_csv_path(self) -> Path:
        return self.tables_dir / "part1_robustness_long.csv"

    @property
    def occluder_positions_csv_path(self) -> Path:
        return self.tables_dir / "part2_occluder_positions_long.csv"

    def ensure_dirs(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)