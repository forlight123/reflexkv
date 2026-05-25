#!/usr/bin/env python3
"""Prepare reasoning datasets used by ReFlexKV paper experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MATH_PROMPT = (
    "Solve the problem step by step, and put your final answer within \\boxed{}.\n\n"
    "{problem}\n\n"
    "Please reason step by step and put your final answer within \\boxed{}."
)


DEFAULT_MAX_NEW_TOKENS = {
    "math500": 4096,
    "gsm8k": 1024,
    "aime24": 4096,
    "aime25": 4096,
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - pandas is a project dep here.
        raise RuntimeError("pandas is required to read parquet datasets") from exc

    try:
        frame = pd.read_parquet(path)
    except ImportError as exc:
        raise RuntimeError(
            "Reading parquet requires pyarrow or fastparquet. "
            "Install one of them, then rerun this script."
        ) from exc
    return frame.to_dict(orient="records")


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(row) for row in payload]
        raise ValueError(f"Expected a JSON list in {path}")
    if suffix == ".parquet":
        return _read_parquet(path)
    raise ValueError(f"Unsupported dataset format: {path}")


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    raise KeyError(f"Missing any of fields: {', '.join(keys)}")


def normalize_gsm8k_row(row: dict[str, Any], *, index: int) -> dict[str, Any]:
    question = str(_first_present(row, ("question", "problem", "input"))).strip()
    raw_answer = str(_first_present(row, ("answer", "target", "solution"))).strip()
    if "####" in raw_answer:
        solution, answer = raw_answer.rsplit("####", 1)
        solution = solution.strip()
        answer = answer.strip()
    else:
        solution = str(row.get("solution", "")).strip()
        answer = raw_answer
    return {
        "problem": question,
        "solution": solution,
        "answer": answer,
        "source": "gsm8k",
        "unique_id": str(row.get("unique_id") or row.get("id") or f"gsm8k/{index}"),
    }


def normalize_aime_row(
    row: dict[str, Any],
    *,
    dataset: str,
    index: int,
) -> dict[str, Any]:
    problem = str(_first_present(row, ("problem", "question", "input"))).strip()
    answer = str(_first_present(row, ("answer", "target", "solution"))).strip()
    return {
        "problem": problem,
        "answer": answer,
        "source": dataset,
        "unique_id": str(row.get("unique_id") or row.get("id") or f"{dataset}/{index}"),
    }


def convert_dataset(dataset: str, input_path: Path, output_path: Path) -> int:
    rows = _read_rows(input_path)
    if dataset == "gsm8k":
        normalized = [
            normalize_gsm8k_row(dict(row), index=index)
            for index, row in enumerate(rows)
        ]
    elif dataset in {"aime24", "aime25"}:
        normalized = [
            normalize_aime_row(dict(row), dataset=dataset, index=index)
            for index, row in enumerate(rows)
        ]
    else:
        raise ValueError(f"Unsupported reasoning dataset conversion: {dataset}")
    _write_jsonl(output_path, normalized)
    return len(normalized)


def update_reasoning_config(
    config_dir: str | Path,
    sample_counts: dict[str, int],
) -> None:
    config_path = Path(config_dir)
    existing_prompts = _read_json_if_exists(config_path / "reasoning_dataset2prompt.json")
    existing_maxlens = _read_json_if_exists(config_path / "reasoning_dataset2maxlen.json")
    existing_metrics = _read_json_if_exists(config_path / "reasoning_dataset2metric.json")
    existing_samples = _read_json_if_exists(config_path / "reasoning_dataset2samples.json")

    datasets = {
        "math500": 500,
        **{name: int(count) for name, count in sample_counts.items()},
    }
    prompts = {dataset: MATH_PROMPT for dataset in datasets}
    maxlens = {
        dataset: DEFAULT_MAX_NEW_TOKENS[dataset]
        for dataset in datasets
    }
    metrics = {dataset: "boxed_accuracy" for dataset in datasets}

    for dataset in sorted(existing_samples):
        if not dataset.startswith("ruler_"):
            continue
        if (
            dataset in existing_prompts
            and dataset in existing_maxlens
            and dataset in existing_metrics
        ):
            datasets[dataset] = int(existing_samples[dataset])
            prompts[dataset] = existing_prompts[dataset]
            maxlens[dataset] = int(existing_maxlens[dataset])
            metrics[dataset] = existing_metrics[dataset]

    _write_json(config_path / "reasoning_dataset2prompt.json", prompts)
    _write_json(config_path / "reasoning_dataset2maxlen.json", maxlens)
    _write_json(config_path / "reasoning_dataset2metric.json", metrics)
    _write_json(config_path / "reasoning_dataset2samples.json", datasets)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert GSM8K/AIME parquet/jsonl files into reasoning JSONL."
    )
    parser.add_argument("--data-dir", default="data/reasoning")
    parser.add_argument("--config-dir", default="eval/config")
    parser.add_argument("--gsm8k-input", default="gsm8k.parquet")
    parser.add_argument("--aime24-input", default="aime24.parquet")
    parser.add_argument("--aime25-input", default="aime25.jsonl")
    parser.add_argument("--skip-config", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)
    conversions = {
        "gsm8k": data_dir / args.gsm8k_input,
        "aime24": data_dir / args.aime24_input,
        "aime25": data_dir / args.aime25_input,
    }

    sample_counts: dict[str, int] = {}
    for dataset, input_path in conversions.items():
        output_path = data_dir / f"{dataset}.jsonl"
        count = convert_dataset(dataset, input_path, output_path)
        sample_counts[dataset] = count
        print(f"wrote {count} rows to {output_path}")

    if not args.skip_config:
        update_reasoning_config(args.config_dir, sample_counts)
        print(f"updated reasoning config in {args.config_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
