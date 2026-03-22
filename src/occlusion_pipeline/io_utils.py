from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd


def load_jsonl_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def latest_jsonl_row(path: Path) -> dict:
    rows = load_jsonl_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[-1]


def iter_case_jsonl_paths(cases_root: Path) -> Iterator[Path]:
    for case_dir in sorted(cases_root.iterdir()):
        if not case_dir.is_dir():
            continue
        jsonl_path = case_dir / "generated" / f"{case_dir.name}.jsonl"
        if jsonl_path.exists():
            yield jsonl_path


def append_rows_to_csv(rows: List[Dict], csv_path: Path) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    write_header = not csv_path.exists()
    df.to_csv(csv_path, mode="a", header=write_header, index=False)