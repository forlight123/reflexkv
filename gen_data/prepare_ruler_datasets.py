#!/usr/bin/env python3
"""Generate RULER synthetic test files for ReFlexKV experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_TASKS = ("niah_single_1", "niah_multikey_2", "niah_multikey_3")
DEFAULT_LENGTHS = (4096, 8192, 16384, 32768)


def ruler_length_label(max_seq_length: int) -> str:
    if max_seq_length % 1024 == 0:
        return f"{max_seq_length // 1024}k"
    return str(max_seq_length)


def build_ruler_prepare_command(
    *,
    ruler_root: Path,
    save_dir: Path,
    task: str,
    max_seq_length: int,
    num_samples: int,
    tokenizer_path: Path,
    tokenizer_type: str,
    random_seed: int,
) -> list[str]:
    return [
        "python",
        str(ruler_root / "scripts" / "data" / "prepare.py"),
        "--save_dir",
        str(save_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--tokenizer_path",
        str(tokenizer_path),
        "--tokenizer_type",
        tokenizer_type,
        "--max_seq_length",
        str(max_seq_length),
        "--model_template_type",
        "base",
        "--num_samples",
        str(num_samples),
        "--random_seed",
        str(random_seed),
        "--remove_newline_tab",
    ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def normalize_ruler_row(
    row: dict[str, Any],
    *,
    dataset: str,
    task: str,
    max_seq_length: int,
    row_index: int | None = None,
) -> dict[str, Any]:
    outputs = row.get("outputs", [])
    if not isinstance(outputs, list):
        outputs = [outputs]
    unique_index = row.get("index", 0) if row_index is None else row_index
    return {
        "problem": str(row["input"]),
        "answers": [str(item) for item in outputs],
        "source": "ruler",
        "unique_id": f"{dataset}/{unique_index}",
        "task": task,
        "max_seq_length": max_seq_length,
        "length": row.get("length"),
    }


def normalize_ruler_file(
    *,
    raw_path: Path,
    normalized_path: Path,
    dataset: str,
    task: str,
    max_seq_length: int,
) -> int:
    normalized = [
        normalize_ruler_row(
            row,
            dataset=dataset,
            task=task,
            max_seq_length=max_seq_length,
            row_index=index,
        )
        for index, row in enumerate(_read_jsonl(raw_path))
    ]
    _write_jsonl(normalized_path, normalized)
    return len(normalized)


def generate_ruler_dataset(
    *,
    ruler_root: Path,
    raw_root: Path,
    normalized_root: Path,
    task: str,
    max_seq_length: int,
    num_samples: int,
    tokenizer_path: Path,
    tokenizer_type: str,
    random_seed: int,
) -> dict[str, Any]:
    label = ruler_length_label(max_seq_length)
    dataset = f"ruler_{task}_{label}"
    save_dir = raw_root / dataset
    command = build_ruler_prepare_command(
        ruler_root=ruler_root,
        save_dir=save_dir,
        task=task,
        max_seq_length=max_seq_length,
        num_samples=num_samples,
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
        random_seed=random_seed,
    )
    subprocess.run(command, check=True)

    raw_path = save_dir / task / "validation.jsonl"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"RULER did not create expected raw file: {raw_path}"
        )
    normalized_path = normalized_root / f"{dataset}.jsonl"
    count = normalize_ruler_file(
        raw_path=raw_path,
        normalized_path=normalized_path,
        dataset=dataset,
        task=task,
        max_seq_length=max_seq_length,
    )
    return {
        "dataset": dataset,
        "task": task,
        "max_seq_length": max_seq_length,
        "samples": count,
        "raw_path": str(raw_path),
        "normalized_path": str(normalized_path),
        "command": command,
    }


def _parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate RULER synthetic raw and normalized JSONL files."
    )
    parser.add_argument("--ruler-root", default="data/RULER")
    parser.add_argument("--raw-root", default="data/ruler_raw")
    parser.add_argument("--normalized-root", default="data/ruler")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument(
        "--lengths",
        default=",".join(str(length) for length in DEFAULT_LENGTHS),
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument(
        "--tokenizer-path",
        default="/home/ytm/models/Llama-3.1-8B-Instruct",
    )
    parser.add_argument("--tokenizer-type", default="hf")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--summary-out", default="data/ruler/summary.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = []
    for task in _parse_csv(args.tasks):
        for length in _parse_csv_ints(args.lengths):
            record = generate_ruler_dataset(
                ruler_root=Path(args.ruler_root),
                raw_root=Path(args.raw_root),
                normalized_root=Path(args.normalized_root),
                task=task,
                max_seq_length=length,
                num_samples=args.num_samples,
                tokenizer_path=Path(args.tokenizer_path),
                tokenizer_type=args.tokenizer_type,
                random_seed=args.random_seed,
            )
            records.append(record)
            print(
                f"wrote {record['samples']} rows to {record['normalized_path']}"
            )
    _write_json(Path(args.summary_out), {"datasets": records})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
