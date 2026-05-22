#!/usr/bin/env python3
"""Build answerable benchmark manifests with BurstGPT-shaped traffic."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.accuracy import run_pd_serving_mixed_accuracy as mixed


@dataclass(frozen=True)
class TraceRow:
    trace_index: int
    timestamp_sec: float
    model: str
    request_tokens: int
    response_tokens: int
    total_tokens: int
    log_type: str


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
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


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid numeric value in BurstGPT trace: {value!r}") from exc


def _parse_int(value: str) -> int:
    return int(round(_parse_float(value)))


def _read_burstgpt_rows(args: argparse.Namespace) -> list[TraceRow]:
    rows: list[TraceRow] = []
    skipped_before_start = max(0, int(getattr(args, "start_index", 0)))
    with Path(args.trace_csv).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for trace_index, row in enumerate(reader):
            if trace_index < skipped_before_start:
                continue
            response_tokens = _parse_int(row["Response tokens"])
            if response_tokens <= 0 and not getattr(args, "include_failed", False):
                continue
            rows.append(
                TraceRow(
                    trace_index=trace_index,
                    timestamp_sec=_parse_float(row["Timestamp"]),
                    model=str(row.get("Model", "")),
                    request_tokens=_parse_int(row["Request tokens"]),
                    response_tokens=response_tokens,
                    total_tokens=_parse_int(row["Total tokens"]),
                    log_type=str(row.get("Log Type", "")),
                )
            )
    return rows


def _target_count(total: int, ratio: float) -> int:
    if ratio <= 0:
        return 0
    return min(total, int(math.ceil(total * ratio)))


def _select_trace_rows(args: argparse.Namespace, rows: list[TraceRow]) -> list[TraceRow]:
    requested = int(args.num_requests)
    if requested <= 0:
        raise ValueError("--num-requests must be positive")
    if len(rows) < requested:
        raise ValueError(
            f"Trace only has {len(rows)} usable rows, fewer than --num-requests={requested}"
        )

    long_input_target = _target_count(requested, float(args.long_input_ratio))
    long_output_target = _target_count(requested, float(args.long_output_ratio))
    selected: set[int] = set()

    def count_long_input() -> int:
        return sum(rows[index].request_tokens >= args.long_input_tokens for index in selected)

    def count_long_output() -> int:
        return sum(rows[index].response_tokens >= args.long_output_tokens for index in selected)

    for index, row in enumerate(rows):
        if len(selected) >= requested or count_long_input() >= long_input_target:
            break
        if row.request_tokens >= args.long_input_tokens:
            selected.add(index)

    for index, row in enumerate(rows):
        if len(selected) >= requested or count_long_output() >= long_output_target:
            break
        if row.response_tokens >= args.long_output_tokens:
            selected.add(index)

    for index in range(len(rows)):
        if len(selected) >= requested:
            break
        selected.add(index)

    return sorted((rows[index] for index in selected), key=lambda row: row.timestamp_sec)


def _bucket(tokens: int, long_threshold: int) -> str:
    if tokens >= long_threshold:
        return "long"
    if tokens >= max(1, long_threshold // 4):
        return "medium"
    return "short"


def _arrival_offsets(
    rows: list[TraceRow],
    *,
    max_interarrival_sec: float | None,
) -> list[float]:
    if not rows:
        return []
    if max_interarrival_sec is None:
        first = rows[0].timestamp_sec
        return [max(0.0, row.timestamp_sec - first) for row in rows]

    offsets = [0.0]
    elapsed = 0.0
    previous = rows[0].timestamp_sec
    for row in rows[1:]:
        gap = max(0.0, row.timestamp_sec - previous)
        elapsed += min(gap, max_interarrival_sec)
        offsets.append(elapsed)
        previous = row.timestamp_sec
    return offsets


def _parse_bench_mix(value: str | None, datasets: list[str]) -> dict[str, float]:
    if not value:
        return {dataset: 1.0 for dataset in datasets}
    weights: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid --bench-mix entry: {item!r}")
        dataset, weight_text = item.split("=", 1)
        dataset = dataset.strip()
        if dataset not in datasets:
            raise ValueError(f"--bench-mix references unknown dataset {dataset!r}")
        weight = float(weight_text)
        if weight < 0:
            raise ValueError(f"--bench-mix weight must be non-negative: {item!r}")
        weights[dataset] = weight
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("--bench-mix must contain at least one positive weight")
    return weights


def _weighted_dataset_order(
    *,
    datasets: list[str],
    weights: dict[str, float],
    total: int,
) -> list[str]:
    weight_sum = sum(weights.get(dataset, 0.0) for dataset in datasets)
    raw = {
        dataset: (weights.get(dataset, 0.0) / weight_sum) * total
        for dataset in datasets
    }
    counts = {dataset: int(math.floor(value)) for dataset, value in raw.items()}
    remaining = total - sum(counts.values())
    remainders = sorted(
        datasets,
        key=lambda dataset: (raw[dataset] - counts[dataset], weights.get(dataset, 0.0)),
        reverse=True,
    )
    for dataset in remainders[:remaining]:
        counts[dataset] += 1

    order: list[str] = []
    while len(order) < total:
        progressed = False
        for dataset in datasets:
            if counts[dataset] > 0:
                order.append(dataset)
                counts[dataset] -= 1
                progressed = True
                if len(order) >= total:
                    break
        if not progressed:
            break
    return order


def _load_benchmark_pools(
    args: argparse.Namespace,
) -> dict[str, list[mixed.MixedRequest]]:
    workload = mixed.load_mixed_workload(args, chat_formatter=None)
    grouped: dict[str, list[mixed.MixedRequest]] = defaultdict(list)
    for request in workload.requests:
        grouped[request.dataset].append(request)
    for requests in grouped.values():
        requests.sort(key=lambda item: item.source_index)
    return dict(sorted(grouped.items()))


def _profile_row(
    trace: TraceRow,
    *,
    request_index: int,
    arrival_time_sec: float,
    scaled_arrival_time_sec: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "request_index": request_index,
        "trace_index": trace.trace_index,
        "trace_timestamp_sec": trace.timestamp_sec,
        "arrival_time_sec": arrival_time_sec,
        "scaled_arrival_time_sec": scaled_arrival_time_sec,
        "trace_model": trace.model,
        "trace_log_type": trace.log_type,
        "trace_request_tokens": trace.request_tokens,
        "trace_response_tokens": trace.response_tokens,
        "trace_total_tokens": trace.total_tokens,
        "input_bucket": _bucket(trace.request_tokens, args.long_input_tokens),
        "output_bucket": _bucket(trace.response_tokens, args.long_output_tokens),
    }


def write_trace_driven_manifest(args: argparse.Namespace) -> dict[str, Any]:
    trace_rows = _select_trace_rows(args, _read_burstgpt_rows(args))
    benchmark_pools = _load_benchmark_pools(args)
    datasets = list(benchmark_pools)
    dataset_order = _weighted_dataset_order(
        datasets=datasets,
        weights=_parse_bench_mix(getattr(args, "bench_mix", None), datasets),
        total=len(trace_rows),
    )
    counters: Counter[str] = Counter()
    arrivals = _arrival_offsets(
        trace_rows,
        max_interarrival_sec=getattr(args, "max_interarrival_sec", None),
    )

    profile_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for request_index, (trace, dataset, arrival_time_sec) in enumerate(
        zip(trace_rows, dataset_order, arrivals)
    ):
        pool = benchmark_pools[dataset]
        request = pool[counters[dataset] % len(pool)]
        counters[dataset] += 1
        scaled_arrival_time_sec = arrival_time_sec * float(args.time_scale)
        profile = _profile_row(
            trace,
            request_index=request_index,
            arrival_time_sec=arrival_time_sec,
            scaled_arrival_time_sec=scaled_arrival_time_sec,
            args=args,
        )
        profile_rows.append(profile)
        meta = dict(request.record.meta or {})
        meta.update(
            {
                "trace_index": trace.trace_index,
                "trace_request_tokens": trace.request_tokens,
                "trace_response_tokens": trace.response_tokens,
                "trace_total_tokens": trace.total_tokens,
                "input_bucket": profile["input_bucket"],
                "output_bucket": profile["output_bucket"],
            }
        )
        manifest_rows.append(
            {
                **profile,
                "task": request.task,
                "dataset": request.dataset,
                "source_index": request.source_index,
                "max_new_tokens": request.max_new_tokens,
                "slo_class": request.slo_class,
                "priority": request.priority,
                "prompt": request.record.prompt,
                "answers": list(request.record.answers or []),
                "all_classes": list(request.record.all_classes or []),
                "meta": meta,
            }
        )

    output_dir = Path(args.output_dir)
    prefix = str(args.output_prefix)
    profile_path = output_dir / f"{prefix}_trace_profile.jsonl"
    manifest_path = output_dir / f"{prefix}_manifest.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"
    _write_jsonl(profile_path, profile_rows)
    _write_jsonl(manifest_path, manifest_rows)

    dataset_counts = Counter(row["dataset"] for row in manifest_rows)
    summary = {
        "trace_csv": str(args.trace_csv),
        "trace_profile": str(profile_path),
        "manifest": str(manifest_path),
        "summary": str(summary_path),
        "total_requests": len(manifest_rows),
        "seed": int(getattr(args, "seed", 0)),
        "time_scale": float(args.time_scale),
        "trace": {
            "selected_rows": len(trace_rows),
            "selected_failed_response_rows": sum(
                1 for row in trace_rows if row.response_tokens <= 0
            ),
            "first_timestamp_sec": trace_rows[0].timestamp_sec if trace_rows else None,
            "last_timestamp_sec": trace_rows[-1].timestamp_sec if trace_rows else None,
            "duration_sec": arrivals[-1] if arrivals else 0.0,
            "scaled_duration_sec": (
                arrivals[-1] * float(args.time_scale) if arrivals else 0.0
            ),
            "long_input_tokens": int(args.long_input_tokens),
            "long_output_tokens": int(args.long_output_tokens),
            "long_input_requests": sum(
                row.request_tokens >= args.long_input_tokens for row in trace_rows
            ),
            "long_output_requests": sum(
                row.response_tokens >= args.long_output_tokens for row in trace_rows
            ),
        },
        "datasets": {
            dataset: {
                "requests": int(count),
                "task": next(
                    row["task"] for row in manifest_rows if row["dataset"] == dataset
                ),
            }
            for dataset, count in sorted(dataset_counts.items())
        },
    }
    _write_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a BurstGPT-shaped answerable benchmark manifest."
    )
    parser.add_argument("--trace-csv", default="data/burstgpt/data/BurstGPT_1.csv")
    parser.add_argument("--output-dir", default="gen_data/burstgpt_answerable_mix")
    parser.add_argument("--output-prefix", default="workload")
    parser.add_argument("--num-requests", type=int, default=500)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--max-interarrival-sec", type=float, default=None)
    parser.add_argument("--long-input-ratio", type=float, default=0.0)
    parser.add_argument("--long-output-ratio", type=float, default=0.0)
    parser.add_argument("--long-input-tokens", type=int, default=2048)
    parser.add_argument("--long-output-tokens", type=int, default=512)
    parser.add_argument("--bench-mix", default=None)

    parser.add_argument("--tasks", default="longbench,reasoning")
    parser.add_argument("--longbench-datasets", default="gov_report,qmsum,hotpotqa")
    parser.add_argument("--reasoning-datasets", default="math500")
    parser.add_argument("--longbench-data-dir", default=mixed.DEFAULT_LONGBENCH_DATA_DIR)
    parser.add_argument("--reasoning-data-dir", default=mixed.DEFAULT_REASONING_DATA_DIR)
    parser.add_argument("--config-dir", default=mixed.DEFAULT_CONFIG_DIR)
    parser.add_argument("--longbench-max-samples", type=int, default=200)
    parser.add_argument("--reasoning-max-samples", type=int, default=200)
    parser.add_argument("--model", default=mixed.DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument(
        "--prompt-fit-policy",
        choices=["none", "skip", "truncate"],
        default="truncate",
    )
    parser.add_argument("--prompt-fit-token-margin", type=int, default=8)
    parser.add_argument("--workload-mix-policy", default="balanced")
    parser.add_argument("--slo-classes", default="high,normal,low")
    parser.add_argument("--slo-priorities", default="-1,0,1")
    parser.add_argument("--skip-chat-template", action="store_true", default=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = write_trace_driven_manifest(parse_args(argv))
    print(json.dumps(_jsonable(summary), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
