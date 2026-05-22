#!/usr/bin/env python3
"""Summarize KV-cache dtype accuracy runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize KV accuracy outputs.")
    parser.add_argument("run_root", help="Output root created by run_kv_accuracy.py")
    parser.add_argument("--baseline-variant", default="auto")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _numeric_values(rows: list[dict], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mixed_request_trace_stats(
    run_dir: Path,
    dataset: str,
) -> dict[str, float | int | None]:
    trace_rows = [
        row
        for row in _read_jsonl(run_dir / "mixed_request_trace.jsonl")
        if row.get("dataset") == dataset
    ]
    max_new_tokens = _numeric_values(trace_rows, "max_new_tokens")
    prompt_chars = _numeric_values(trace_rows, "prompt_chars")
    prediction_chars = _numeric_values(trace_rows, "prediction_chars")
    original_prompt_tokens = _numeric_values(trace_rows, "prompt_original_tokens")
    final_prompt_tokens = _numeric_values(trace_rows, "prompt_final_tokens")
    prompt_truncated_total = sum(
        1 for row in trace_rows if row.get("prompt_truncated") is True
    )
    return {
        "max_new_tokens": int(max(max_new_tokens)) if max_new_tokens else None,
        "avg_prompt_chars": _mean(prompt_chars),
        "max_prompt_chars": int(max(prompt_chars)) if prompt_chars else None,
        "avg_prompt_original_tokens": _mean(original_prompt_tokens),
        "max_prompt_original_tokens": (
            int(max(original_prompt_tokens)) if original_prompt_tokens else None
        ),
        "avg_prompt_final_tokens": _mean(final_prompt_tokens),
        "max_prompt_final_tokens": (
            int(max(final_prompt_tokens)) if final_prompt_tokens else None
        ),
        "prompt_truncated_total": prompt_truncated_total,
        "avg_prediction_chars": _mean(prediction_chars),
        "max_prediction_chars": (
            int(max(prediction_chars)) if prediction_chars else None
        ),
    }


def _load_manifest(run_root: Path) -> dict[str, dict]:
    manifest_path = run_root / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = _read_json(manifest_path)
    return {
        item["run_name"]: item
        for item in manifest.get("runs", [])
        if "run_name" in item
    }


def _infer_task(run_name: str, run_config: dict, manifest_record: dict | None) -> str:
    if manifest_record and manifest_record.get("task"):
        return str(manifest_record["task"])
    if run_config.get("task"):
        return str(run_config["task"])
    if run_name.startswith("longbench_"):
        return "longbench"
    if run_name.startswith("reasoning_"):
        return "reasoning"
    data_dir = str(run_config.get("data_dir", "")).lower()
    if "longbench" in data_dir:
        return "longbench"
    if "reasoning" in data_dir:
        return "reasoning"
    return "unknown"


def _variant_from_config(run_config: dict, manifest_record: dict | None) -> str:
    if manifest_record and manifest_record.get("variant"):
        return str(manifest_record["variant"])
    if run_config.get("decode_kv_cache_dtype"):
        return str(run_config["decode_kv_cache_dtype"])
    return str(run_config.get("kv_cache_dtype", "unknown"))


def _prediction_match_rate(left: list[dict], right: list[dict]) -> float | None:
    total = min(len(left), len(right))
    if total == 0:
        return None
    matches = 0
    for left_row, right_row in zip(left[:total], right[:total]):
        if left_row.get("pred", "") == right_row.get("pred", ""):
            matches += 1
    return matches / total


def collect_rows(run_root: Path, baseline_variant: str) -> list[dict]:
    manifest = _load_manifest(run_root)
    rows = []
    predictions_by_key: dict[tuple[str, str, str], list[dict]] = {}

    for summary_path in sorted(run_root.glob("*/*/run_summary.json")):
        dataset_dir = summary_path.parent
        run_name = dataset_dir.parent.name
        dataset = dataset_dir.name
        result_path = dataset_dir / "result.json"
        config_path = dataset_dir / "run_config.json"
        pred_path = dataset_dir / "pred.jsonl"
        if not result_path.exists() or not config_path.exists():
            continue

        run_summary = _read_json(summary_path)
        result = _read_json(result_path)
        run_config = _read_json(config_path)
        manifest_record = manifest.get(run_name)
        task = _infer_task(run_name, run_config, manifest_record)
        variant = _variant_from_config(run_config, manifest_record)
        pred_rows = _read_jsonl(pred_path)
        empty_predictions = sum(1 for row in pred_rows if row.get("pred", "") == "")
        total_predictions = len(pred_rows)
        run_dir = dataset_dir.parent
        mixed_summary_path = run_dir / "mixed_summary.json"
        mixed_summary = (
            _read_json(mixed_summary_path)
            if mixed_summary_path.exists()
            else {}
        )
        serving_metrics = mixed_summary.get("serving_metrics", {})
        decode_metrics = (
            serving_metrics.get("decode", {})
            if isinstance(serving_metrics, dict)
            else {}
        )
        reflex_trace = mixed_summary.get("reflex_trace", {})
        mixed_dataset_summary = {}
        if isinstance(mixed_summary.get("datasets"), dict):
            candidate_summary = mixed_summary["datasets"].get(dataset, {})
            if isinstance(candidate_summary, dict):
                mixed_dataset_summary = candidate_summary
        if not isinstance(decode_metrics, dict):
            decode_metrics = {}
        if not isinstance(reflex_trace, dict):
            reflex_trace = {}
        request_trace_stats = _mixed_request_trace_stats(run_dir, dataset)
        if task == "unknown" and mixed_dataset_summary.get("task"):
            task = str(mixed_dataset_summary["task"])

        row = {
            "run_name": run_name,
            "task": task,
            "dataset": dataset,
            "variant": variant,
            "kv_cache_dtype": run_config.get(
                "kv_cache_dtype",
                run_config.get("decode_kv_cache_dtype", variant),
            ),
            "metric": result.get("metric", ""),
            "avg_score": float(result.get("avg_score", 0.0)),
            "total_samples": int(result.get("total_samples", 0)),
            "completed_predictions": int(run_summary.get("completed_predictions", 0)),
            "failed_predictions": int(run_summary.get("failed_predictions", 0)),
            "empty_prediction_rate": (
                empty_predictions / total_predictions if total_predictions else 0.0
            ),
            "max_new_tokens": request_trace_stats["max_new_tokens"],
            "avg_prompt_chars": request_trace_stats["avg_prompt_chars"],
            "max_prompt_chars": request_trace_stats["max_prompt_chars"],
            "avg_prompt_original_tokens": request_trace_stats[
                "avg_prompt_original_tokens"
            ],
            "max_prompt_original_tokens": request_trace_stats[
                "max_prompt_original_tokens"
            ],
            "avg_prompt_final_tokens": request_trace_stats[
                "avg_prompt_final_tokens"
            ],
            "max_prompt_final_tokens": request_trace_stats[
                "max_prompt_final_tokens"
            ],
            "prompt_truncated_total": request_trace_stats["prompt_truncated_total"],
            "avg_prediction_chars": request_trace_stats["avg_prediction_chars"],
            "max_prediction_chars": request_trace_stats["max_prediction_chars"],
            "avg_latency_seconds": float(run_summary.get("avg_latency_seconds", 0.0)),
            "max_latency_seconds": float(run_summary.get("max_latency_seconds", 0.0)),
            "p95_latency_seconds": float(run_summary.get("p95_latency_seconds", 0.0)),
            "decode_max_kv_cache_usage_pct": decode_metrics.get(
                "max_kv_cache_usage_pct",
            ),
            "decode_avg_kv_cache_usage_pct": decode_metrics.get(
                "avg_kv_cache_usage_pct",
            ),
            "decode_max_running": decode_metrics.get("max_running"),
            "decode_avg_running": decode_metrics.get("avg_running"),
            "decode_max_waiting": decode_metrics.get("max_waiting"),
            "decode_avg_waiting": decode_metrics.get("avg_waiting"),
            "demoted_pages_total": reflex_trace.get("demoted_pages_total"),
            "landing_materialized_pages_total": reflex_trace.get(
                "landing_materialized_pages_total",
            ),
            "recovery_plan_event_count": reflex_trace.get(
                "recovery_plan_event_count",
            ),
            "max_int4_ratio": reflex_trace.get("max_int4_ratio"),
            "duration_seconds": float(run_summary.get("duration_seconds", 0.0)),
            "pred_file": str(pred_path),
        }
        rows.append(row)
        predictions_by_key[(task, dataset, variant)] = pred_rows

    baseline_scores = {
        (row["task"], row["dataset"]): row["avg_score"]
        for row in rows
        if row["variant"] == baseline_variant
    }
    rows_by_dataset: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_dataset[(row["task"], row["dataset"])].append(row)

    for row in rows:
        key = (row["task"], row["dataset"])
        baseline_score = baseline_scores.get(key)
        row["score_delta_vs_baseline"] = (
            row["avg_score"] - baseline_score if baseline_score is not None else ""
        )
        baseline_preds = predictions_by_key.get((row["task"], row["dataset"], baseline_variant))
        current_preds = predictions_by_key.get((row["task"], row["dataset"], row["variant"]))
        match_rate = (
            _prediction_match_rate(current_preds, baseline_preds)
            if baseline_preds is not None and current_preds is not None
            else None
        )
        row["exact_pred_match_rate_vs_baseline"] = (
            "" if match_rate is None else match_rate
        )

    return sorted(rows, key=lambda item: (item["task"], item["dataset"], item["variant"]))


def write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "task",
        "dataset",
        "variant",
        "kv_cache_dtype",
        "metric",
        "avg_score",
        "score_delta_vs_baseline",
        "exact_pred_match_rate_vs_baseline",
        "total_samples",
        "completed_predictions",
        "failed_predictions",
        "empty_prediction_rate",
        "max_new_tokens",
        "avg_prompt_chars",
        "max_prompt_chars",
        "avg_prompt_original_tokens",
        "max_prompt_original_tokens",
        "avg_prompt_final_tokens",
        "max_prompt_final_tokens",
        "prompt_truncated_total",
        "avg_prediction_chars",
        "max_prediction_chars",
        "avg_latency_seconds",
        "max_latency_seconds",
        "p95_latency_seconds",
        "decode_max_kv_cache_usage_pct",
        "decode_avg_kv_cache_usage_pct",
        "decode_max_running",
        "decode_avg_running",
        "decode_max_waiting",
        "decode_avg_waiting",
        "demoted_pages_total",
        "landing_materialized_pages_total",
        "recovery_plan_event_count",
        "max_int4_ratio",
        "duration_seconds",
        "run_name",
        "pred_file",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    args = parse_args()
    run_root = Path(args.run_root)
    rows = collect_rows(run_root, args.baseline_variant)
    out_path = Path(args.out) if args.out else run_root / "accuracy_summary.csv"
    write_csv(out_path, rows)
    print(f"summary={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
