#!/usr/bin/env python3
"""Build paper mixed manifests across reasoning, LongBench, and RULER."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gen_data import build_trace_driven_manifest as trace_builder
from scripts.accuracy import run_pd_serving_mixed_accuracy as mixed


DEFAULT_LONGBENCH_DATASETS = (
    "gov_report",
    "qasper",
    "passage_retrieval_en",
)
DEFAULT_REASONING_DATASETS = ("math500", "gsm8k")
DEFAULT_RULER_DATASETS = (
    "ruler_niah_single_1_4k",
    "ruler_niah_single_1_8k",
    "ruler_niah_single_1_16k",
)


@dataclass(frozen=True)
class Pool:
    group: str
    task: str
    dataset: str
    max_new_tokens: int
    records: list[Any]


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
        json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), ensure_ascii=False) + "\n")


def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_group_mix(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in _parse_csv(value):
        if "=" not in item:
            raise ValueError(f"Invalid --group-mix entry: {item!r}")
        group, weight_text = item.split("=", 1)
        group = group.strip()
        weight = float(weight_text)
        if weight < 0:
            raise ValueError(f"--group-mix weight must be non-negative: {item!r}")
        weights[group] = weight
    if not weights or sum(weights.values()) <= 0:
        raise ValueError("--group-mix must contain at least one positive weight")
    return weights


def _allocate_counts(
    labels: list[str],
    weights: dict[str, float],
    total: int,
) -> dict[str, int]:
    weight_sum = sum(weights.get(label, 0.0) for label in labels)
    if weight_sum <= 0:
        raise ValueError("At least one selected label must have positive weight")
    raw = {
        label: (weights.get(label, 0.0) / weight_sum) * total
        for label in labels
    }
    counts = {label: int(math.floor(value)) for label, value in raw.items()}
    remaining = total - sum(counts.values())
    order = sorted(
        labels,
        key=lambda label: (raw[label] - counts[label], weights.get(label, 0.0)),
        reverse=True,
    )
    for label in order[:remaining]:
        counts[label] += 1
    return counts


def _round_robin_from_counts(counts: dict[str, int]) -> list[str]:
    order: list[str] = []
    labels = [label for label, count in counts.items() if count > 0]
    while labels:
        next_labels: list[str] = []
        for label in labels:
            if counts[label] <= 0:
                continue
            order.append(label)
            counts[label] -= 1
            if counts[label] > 0:
                next_labels.append(label)
        labels = next_labels
    return order


def _dataset_args(
    args: argparse.Namespace,
    *,
    task: str,
    dataset: str,
    data_dir: str,
    max_samples: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        task=task,
        dataset=dataset,
        data_dir=data_dir,
        max_samples=max_samples,
        config_dir=args.config_dir,
        model=args.model,
        skip_chat_template=True,
    )


def _load_pool(
    args: argparse.Namespace,
    *,
    group: str,
    task: str,
    dataset: str,
    data_dir: str,
    max_samples: int,
    manifest_task: str | None = None,
) -> Pool:
    serving_dataset = mixed.load_serving_dataset(
        _dataset_args(
            args,
            task=task,
            dataset=dataset,
            data_dir=data_dir,
            max_samples=max_samples,
        ),
        chat_formatter=None,
    )
    return Pool(
        group=group,
        task=manifest_task or serving_dataset.task,
        dataset=serving_dataset.dataset,
        max_new_tokens=serving_dataset.max_new_tokens,
        records=serving_dataset.records,
    )


def _load_pools(args: argparse.Namespace) -> dict[str, Pool]:
    pools: dict[str, Pool] = {}
    for dataset in _parse_csv(args.longbench_datasets):
        pools[dataset] = _load_pool(
            args,
            group="longbench",
            task="longbench",
            dataset=dataset,
            data_dir=args.longbench_data_dir,
            max_samples=args.longbench_max_samples,
        )
    for dataset in _parse_csv(args.reasoning_datasets):
        pools[dataset] = _load_pool(
            args,
            group=dataset,
            task="reasoning",
            dataset=dataset,
            data_dir=args.reasoning_data_dir,
            max_samples=args.reasoning_max_samples,
        )
    for dataset in _parse_csv(args.ruler_datasets):
        pools[dataset] = _load_pool(
            args,
            group="ruler",
            task="reasoning",
            dataset=dataset,
            data_dir=args.ruler_data_dir,
            max_samples=args.ruler_max_samples,
            manifest_task="ruler",
        )
    return pools


def _build_dataset_order(args: argparse.Namespace, pools: dict[str, Pool]) -> list[str]:
    group_mix = _parse_group_mix(args.group_mix)
    groups = ["longbench", *list(_parse_csv(args.reasoning_datasets)), "ruler"]
    group_counts = _allocate_counts(groups, group_mix, int(args.num_requests))
    dataset_counts: dict[str, int] = {}
    for group, count in group_counts.items():
        if count <= 0:
            continue
        if group == "longbench":
            datasets = _parse_csv(args.longbench_datasets)
        elif group == "ruler":
            datasets = _parse_csv(args.ruler_datasets)
        else:
            datasets = [group]
        per_dataset = _allocate_counts(
            datasets,
            {dataset: 1.0 for dataset in datasets},
            count,
        )
        dataset_counts.update(per_dataset)
    missing = [dataset for dataset in dataset_counts if dataset not in pools]
    if missing:
        raise ValueError(f"Missing dataset pools: {', '.join(sorted(missing))}")
    return _round_robin_from_counts(dataset_counts)


def _row_for_request(
    *,
    request_index: int,
    trace: trace_builder.TraceRow,
    arrival_time_sec: float,
    dataset: str,
    pool: Pool,
    source_index: int,
    args: argparse.Namespace,
    slo: mixed.SLOClass,
) -> dict[str, Any]:
    record = pool.records[source_index % len(pool.records)]
    scaled_arrival_time_sec = arrival_time_sec * float(args.time_scale)
    profile = trace_builder._profile_row(
        trace,
        request_index=request_index,
        arrival_time_sec=arrival_time_sec,
        scaled_arrival_time_sec=scaled_arrival_time_sec,
        args=args,
    )
    meta = dict(record.meta or {})
    meta.update(
        {
            "group": pool.group,
            "trace_index": trace.trace_index,
            "trace_request_tokens": trace.request_tokens,
            "trace_response_tokens": trace.response_tokens,
            "trace_total_tokens": trace.total_tokens,
            "input_bucket": profile["input_bucket"],
            "output_bucket": profile["output_bucket"],
        }
    )
    return {
        **profile,
        "group": pool.group,
        "task": pool.task,
        "dataset": dataset,
        "source_index": source_index,
        "max_new_tokens": pool.max_new_tokens,
        "slo_class": slo.name,
        "priority": slo.priority,
        "prompt": record.prompt,
        "answers": list(record.answers or []),
        "all_classes": list(record.all_classes or []),
        "meta": meta,
    }


def write_paper_mixed_manifest(args: argparse.Namespace) -> dict[str, Any]:
    trace_rows = trace_builder._select_trace_rows(
        args,
        trace_builder._read_burstgpt_rows(args),
    )
    pools = _load_pools(args)
    dataset_order = _build_dataset_order(args, pools)
    if len(dataset_order) != len(trace_rows):
        raise ValueError("Dataset order length does not match selected trace rows")

    arrivals = trace_builder._arrival_offsets(
        trace_rows,
        max_interarrival_sec=getattr(args, "max_interarrival_sec", None),
    )
    rng = random.Random(args.seed)
    slo_classes = mixed.parse_slo_classes(args)
    counters: Counter[str] = Counter()
    manifest_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []

    for request_index, (trace, dataset, arrival_time_sec) in enumerate(
        zip(trace_rows, dataset_order, arrivals)
    ):
        pool = pools[dataset]
        source_index = counters[dataset]
        counters[dataset] += 1
        slo = rng.choice(slo_classes)
        row = _row_for_request(
            request_index=request_index,
            trace=trace,
            arrival_time_sec=arrival_time_sec,
            dataset=dataset,
            pool=pool,
            source_index=source_index,
            args=args,
            slo=slo,
        )
        manifest_rows.append(row)
        profile_rows.append(
            {
                key: row[key]
                for key in (
                    "request_index",
                    "trace_index",
                    "trace_timestamp_sec",
                    "arrival_time_sec",
                    "scaled_arrival_time_sec",
                    "trace_model",
                    "trace_log_type",
                    "trace_request_tokens",
                    "trace_response_tokens",
                    "trace_total_tokens",
                    "input_bucket",
                    "output_bucket",
                )
            }
        )

    output_dir = Path(args.output_dir)
    prefix = str(args.output_prefix)
    manifest_path = output_dir / f"{prefix}_manifest.jsonl"
    profile_path = output_dir / f"{prefix}_trace_profile.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"
    _write_jsonl(manifest_path, manifest_rows)
    _write_jsonl(profile_path, profile_rows)

    dataset_counts = Counter(row["dataset"] for row in manifest_rows)
    group_counts = Counter(row["group"] for row in manifest_rows)
    summary = {
        "trace_csv": str(args.trace_csv),
        "manifest": str(manifest_path),
        "trace_profile": str(profile_path),
        "summary": str(summary_path),
        "total_requests": len(manifest_rows),
        "seed": int(args.seed),
        "time_scale": float(args.time_scale),
        "group_mix": dict(sorted(group_counts.items())),
        "trace": {
            "selected_rows": len(trace_rows),
            "selected_failed_response_rows": sum(
                row.response_tokens <= 0 for row in trace_rows
            ),
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
                "group": pools[dataset].group,
                "task": pools[dataset].task,
                "max_new_tokens": pools[dataset].max_new_tokens,
            }
            for dataset, count in sorted(dataset_counts.items())
        },
    }
    _write_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a no-AIME paper mixed workload manifest."
    )
    parser.add_argument("--trace-csv", default="data/burstgpt/data/BurstGPT_1.csv")
    parser.add_argument("--output-dir", default="gen_data/paper_mixed_n100_seed0")
    parser.add_argument("--output-prefix", default="paper_mixed_n100_seed0")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--include-failed", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--time-scale", type=float, default=0.05)
    parser.add_argument("--max-interarrival-sec", type=float, default=60.0)
    parser.add_argument("--long-input-ratio", type=float, default=0.35)
    parser.add_argument("--long-output-ratio", type=float, default=0.25)
    parser.add_argument("--long-input-tokens", type=int, default=2048)
    parser.add_argument("--long-output-tokens", type=int, default=512)
    parser.add_argument(
        "--group-mix",
        default="longbench=0.25,math500=0.25,gsm8k=0.25,ruler=0.25",
    )
    parser.add_argument(
        "--longbench-datasets",
        default=",".join(DEFAULT_LONGBENCH_DATASETS),
    )
    parser.add_argument(
        "--reasoning-datasets",
        default=",".join(DEFAULT_REASONING_DATASETS),
    )
    parser.add_argument("--ruler-datasets", default=",".join(DEFAULT_RULER_DATASETS))
    parser.add_argument("--longbench-data-dir", default=mixed.DEFAULT_LONGBENCH_DATA_DIR)
    parser.add_argument("--reasoning-data-dir", default=mixed.DEFAULT_REASONING_DATA_DIR)
    parser.add_argument("--ruler-data-dir", default=str(REPO_ROOT / "data" / "ruler"))
    parser.add_argument("--config-dir", default=mixed.DEFAULT_CONFIG_DIR)
    parser.add_argument("--longbench-max-samples", type=int, default=200)
    parser.add_argument("--reasoning-max-samples", type=int, default=200)
    parser.add_argument("--ruler-max-samples", type=int, default=100)
    parser.add_argument("--model", default=mixed.DEFAULT_MODEL)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--slo-classes", default="high,normal,low")
    parser.add_argument("--slo-priorities", default="-1,0,1")
    parser.add_argument("--skip-chat-template", action="store_true", default=True)
    parser.add_argument("--prompt-fit-policy", default="none")
    parser.add_argument("--prompt-fit-token-margin", type=int, default=8)
    parser.add_argument("--workload-mix-policy", default="balanced")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    summary = write_paper_mixed_manifest(parse_args(argv))
    print(json.dumps(_jsonable(summary), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
