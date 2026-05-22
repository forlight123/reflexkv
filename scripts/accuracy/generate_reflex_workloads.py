#!/usr/bin/env python3
"""Generate an offline JSONL workload manifest for ReFlexKV experiments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.accuracy import run_pd_serving_mixed_accuracy as mixed


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _row_for_request(request: mixed.MixedRequest) -> dict[str, Any]:
    record = request.record
    answers = record.answers or []
    all_classes = record.all_classes or []
    return {
        "request_index": request.index,
        "task": request.task,
        "dataset": request.dataset,
        "source_index": request.source_index,
        "max_new_tokens": request.max_new_tokens,
        "slo_class": request.slo_class,
        "priority": request.priority,
        "prompt": record.prompt,
        "answers": list(answers),
        "all_classes": list(all_classes),
        "meta": dict(record.meta or {}),
    }


def write_workload_manifest(args: argparse.Namespace) -> dict[str, Any]:
    workload = mixed.load_mixed_workload(args, chat_formatter=None)
    rows = [_row_for_request(request) for request in workload.requests]
    output = Path(args.output)
    _write_jsonl(output, rows)

    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    summary = {
        "output": str(output),
        "total_requests": len(rows),
        "workload_mix_policy": getattr(args, "workload_mix_policy", "balanced"),
        "seed": int(getattr(args, "seed", 0)),
        "prompt_fit": workload.prompt_fit_summaries,
        "datasets": {
            dataset: {
                "requests": len(dataset_rows),
                "task": dataset_rows[0]["task"],
                "max_new_tokens": dataset_rows[0]["max_new_tokens"],
                "avg_prompt_chars": (
                    sum(len(str(row["prompt"])) for row in dataset_rows)
                    / len(dataset_rows)
                ),
            }
            for dataset, dataset_rows in sorted(by_dataset.items())
        },
    }
    summary_out = getattr(args, "summary_out", None)
    if summary_out:
        _write_json(Path(summary_out), summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a ReFlexKV mixed workload JSONL manifest."
    )
    parser.add_argument("--tasks", default="longbench,reasoning")
    parser.add_argument("--longbench-datasets", default="gov_report")
    parser.add_argument("--reasoning-datasets", default="math500")
    parser.add_argument("--model", default=mixed.DEFAULT_MODEL)
    parser.add_argument(
        "--longbench-data-dir",
        default=mixed.DEFAULT_LONGBENCH_DATA_DIR,
    )
    parser.add_argument(
        "--reasoning-data-dir",
        default=mixed.DEFAULT_REASONING_DATA_DIR,
    )
    parser.add_argument("--config-dir", default=mixed.DEFAULT_CONFIG_DIR)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--longbench-max-samples", type=int, default=20)
    parser.add_argument("--reasoning-max-samples", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--prompt-fit-policy",
        choices=["none", "skip", "truncate"],
        default="none",
    )
    parser.add_argument("--prompt-fit-token-margin", type=int, default=8)
    parser.add_argument(
        "--workload-mix-policy",
        choices=["balanced", "random"],
        default="balanced",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo-classes", default="high,normal,low")
    parser.add_argument("--slo-priorities", default="-1,0,1")
    parser.add_argument("--skip-chat-template", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = write_workload_manifest(parse_args(argv))
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
