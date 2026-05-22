#!/usr/bin/env python3
"""Run mixed LongBench/Math500 accuracy through the ReFlexKV 1P1D path."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.backends.base import PromptRecord
from scripts.accuracy import run_pd_serving_accuracy as single_runner
from scripts.profiling import run_reflex_pd_1p1d as pd_runner
from scripts.profiling.summarize_reflex_pd_pressure import (
    _read_trace_events,
    _trace_stats,
)


ROOT = single_runner.ROOT
DEFAULT_MODEL = single_runner.DEFAULT_MODEL
DEFAULT_CONFIG_DIR = single_runner.DEFAULT_CONFIG_DIR
DEFAULT_LONGBENCH_DATA_DIR = single_runner.DEFAULT_LONGBENCH_DATA_DIR
DEFAULT_REASONING_DATA_DIR = single_runner.DEFAULT_REASONING_DATA_DIR

Role = single_runner.Role
ServingDataset = single_runner.ServingDataset
_async_completion_request = single_runner._async_completion_request
build_chat_formatter = single_runner.build_chat_formatter
build_proxy_cmd = single_runner.build_proxy_cmd
build_server_cmd = single_runner.build_server_cmd
build_server_env = single_runner.build_server_env
evaluate_longbench_file = single_runner.evaluate_longbench_file
evaluate_reasoning_file = single_runner.evaluate_reasoning_file
load_serving_dataset = single_runner.load_serving_dataset
fit_dataset_to_model_len = single_runner.fit_dataset_to_model_len


@dataclass(frozen=True)
class SLOClass:
    name: str
    priority: int


@dataclass(frozen=True)
class MixedRequest:
    index: int
    task: str
    dataset: str
    source_index: int
    record: PromptRecord
    max_new_tokens: int
    slo_class: str
    priority: int
    arrival_time_seconds: float | None = None
    trace_request_tokens: int | None = None
    trace_response_tokens: int | None = None
    trace_total_tokens: int | None = None
    trace_index: int | None = None
    input_bucket: str | None = None
    output_bucket: str | None = None


@dataclass(frozen=True)
class MixedWorkload:
    requests: list[MixedRequest]
    prompt_fit_summaries: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class MixedPrediction:
    request_index: int
    pred: str
    error: str | None
    latency_seconds: float
    request_id: str = ""
    queued_offset_seconds: float = 0.0
    start_offset_seconds: float = 0.0
    end_offset_seconds: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run mixed LongBench/Math500 accuracy through 1P1D serving."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefill-gpu", default="6")
    parser.add_argument("--decode-gpu", default="7")
    parser.add_argument("--prefill-port", type=int, default=8710)
    parser.add_argument("--decode-port", type=int, default=8720)
    parser.add_argument("--proxy-port", type=int, default=8730)
    parser.add_argument("--prefill-bootstrap-port", type=int, default=8998)
    parser.add_argument("--proxy-prefill-max-inflight", type=int, default=2)
    parser.add_argument(
        "--proxy-prefill-metadata-wait-timeout-sec",
        type=float,
        default=None,
    )
    parser.add_argument("--reflex-remote-chunk-tokens", type=int, default=512)
    parser.add_argument("--mooncake-protocol", default="rdma")
    parser.add_argument("--mooncake-num-workers", type=int, default=10)
    parser.add_argument("--reflex-keep-recent-blocks", type=int, default=16)
    parser.add_argument("--reflex-keep-initial-blocks", type=int, default=4)
    parser.add_argument("--reflex-max-int4-fraction-per-request", type=float, default=None)
    parser.add_argument("--reflex-survival-warmup-tokens", type=int, default=128)
    parser.add_argument("--reflex-risk-warmup-tokens", type=int, default=16)
    parser.add_argument("--reflex-short-admission-max-int4-fraction", type=float, default=0.03)
    parser.add_argument("--reflex-sparse-window-pages", type=int, default=32)
    parser.add_argument("--reflex-short-max-demote-per-window", type=int, default=1)
    parser.add_argument("--reflex-max-demote-per-window", type=int, default=2)
    parser.add_argument("--reflex-low-risk-score-fraction", type=float, default=0.25)
    parser.add_argument(
        "--reflex-page-selection-policy",
        choices=[
            "oldest",
            "distance",
            "random",
            "relevance",
            "relevance_sparse",
            "frontier_dual",
        ],
        default="relevance_sparse",
    )
    parser.add_argument(
        "--reflex-cold-admission-max-int4-fraction",
        type=float,
        default=0.10,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--reflex-cold-admission-emergency-free-ratio",
        type=float,
        default=0.05,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--reflex-decode-pressure-warmup-tokens", type=int, default=32)
    parser.add_argument("--reflex-decode-pressure-ramp-tokens", type=int, default=512)
    parser.add_argument("--reflex-short-prefill-pages", type=int, default=64)
    parser.add_argument("--reflex-long-prefill-pages", type=int, default=512)
    parser.add_argument("--reflex-global-evidence-min-prompt-pages", type=int, default=512)
    parser.add_argument("--reflex-global-evidence-min-decode-tokens", type=int, default=129)
    parser.add_argument(
        "--reflex-global-evidence-landing-max-int4-fraction",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--reflex-reasoning-prompt-protection-max-pages",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--reflex-reasoning-prompt-protection-min-decode-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument("--reflex-slo-pressure-step", type=float, default=0.25)
    parser.add_argument("--reflex-min-slo-pressure", type=float, default=0.5)
    parser.add_argument("--reflex-max-slo-pressure", type=float, default=1.5)
    parser.add_argument("--scheduling-policy", choices=["fcfs", "priority"], default="priority")

    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--prefill-kv-cache-dtype", default="auto")
    parser.add_argument("--decode-kv-cache-dtype", default="reflex_int4")
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--force-triton-attn", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--enable-reflex-trace", action="store_true")
    parser.add_argument(
        "--disable-reflex-prefill-page-metadata",
        action="store_true",
        help="Disable P-side ReFlexKV page-risk metadata for ablation runs.",
    )
    parser.add_argument("--reflex-int4-budget-fraction", type=float, default=None)
    parser.add_argument("--extra-serve-args", nargs=argparse.REMAINDER, default=[])

    parser.add_argument("--tasks", default="longbench,reasoning")
    parser.add_argument("--longbench-datasets", default="qasper,hotpotqa,multifieldqa_en")
    parser.add_argument("--reasoning-datasets", default="math500")
    parser.add_argument("--longbench-data-dir", default=DEFAULT_LONGBENCH_DATA_DIR)
    parser.add_argument("--reasoning-data-dir", default=DEFAULT_REASONING_DATA_DIR)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", default="outputs/accuracy/pd_serving_mixed")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--longbench-max-samples", type=int, default=16)
    parser.add_argument("--reasoning-max-samples", type=int, default=16)
    parser.add_argument("--skip-chat-template", action="store_true")
    parser.add_argument(
        "--prompt-fit-policy",
        choices=["none", "skip", "truncate"],
        default="truncate",
        help=(
            "How to handle prompts whose tokenized length plus max_new_tokens "
            "exceeds --max-model-len. truncate keeps the first and last tokens."
        ),
    )
    parser.add_argument(
        "--prompt-fit-token-margin",
        type=int,
        default=8,
        help="Safety margin reserved when fitting prompts to --max-model-len.",
    )

    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-rate", default="0.5")
    parser.add_argument(
        "--workload-manifest",
        default=None,
        help=(
            "Optional JSONL manifest generated by gen_data/"
            "build_trace_driven_manifest.py."
        ),
    )
    parser.add_argument(
        "--arrival-policy",
        choices=["poisson", "trace"],
        default="poisson",
        help="Use Poisson inter-arrivals or per-request manifest arrivals.",
    )
    parser.add_argument(
        "--trace-time-scale",
        type=float,
        default=1.0,
        help="Multiplier applied to manifest arrival offsets during trace replay.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo-classes", default="high,normal,low")
    parser.add_argument("--slo-priorities", default="-1,0,1")
    parser.add_argument(
        "--workload-mix-policy",
        choices=["balanced", "random"],
        default="balanced",
        help=(
            "How to order mixed samples before sending requests. balanced "
            "round-robins datasets so each concurrency window contains all "
            "available datasets; random preserves the old global shuffle."
        ),
    )
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--server-ready-timeout-sec", type=float, default=420.0)
    parser.add_argument("--request-timeout-sec", type=float, default=900.0)
    return parser.parse_args()


def _parse_str_csv(value: str, *, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    return items


def _parse_int_csv(value: str, *, name: str) -> list[int]:
    items = _parse_str_csv(value, name=name)
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-integer value: {item!r}") from exc
    return parsed


def parse_slo_classes(args: argparse.Namespace) -> list[SLOClass]:
    names = _parse_str_csv(args.slo_classes, name="slo_classes")
    priorities = _parse_int_csv(args.slo_priorities, name="slo_priorities")
    if len(names) != len(priorities):
        raise ValueError("--slo-classes and --slo-priorities must have the same length.")
    return [
        SLOClass(name=name, priority=priority)
        for name, priority in zip(names, priorities)
    ]


def _dataset_args(
    args: argparse.Namespace,
    *,
    task: str,
    dataset: str,
    data_dir: str,
    max_samples: int,
) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "task": task,
            "dataset": dataset,
            "data_dir": data_dir,
            "max_samples": max_samples,
        }
    )
    return argparse.Namespace(**values)


def _load_task_datasets(
    args: argparse.Namespace,
    chat_formatter: Callable[[str], str] | None,
) -> list[ServingDataset]:
    datasets: list[ServingDataset] = []
    tasks = set(_parse_str_csv(args.tasks, name="tasks"))
    unsupported = tasks - {"longbench", "reasoning"}
    if unsupported:
        raise ValueError(f"Unsupported tasks: {', '.join(sorted(unsupported))}")

    if "longbench" in tasks:
        for dataset in _parse_str_csv(
            args.longbench_datasets,
            name="longbench_datasets",
        ):
            datasets.append(
                load_serving_dataset(
                    _dataset_args(
                        args,
                        task="longbench",
                        dataset=dataset,
                        data_dir=args.longbench_data_dir,
                        max_samples=args.longbench_max_samples,
                    ),
                    chat_formatter,
                )
            )
    if "reasoning" in tasks:
        for dataset in _parse_str_csv(
            args.reasoning_datasets,
            name="reasoning_datasets",
        ):
            datasets.append(
                load_serving_dataset(
                    _dataset_args(
                        args,
                        task="reasoning",
                        dataset=dataset,
                        data_dir=args.reasoning_data_dir,
                        max_samples=args.reasoning_max_samples,
                    ),
                    chat_formatter,
                )
            )
    return datasets


def _reindex_requests(requests: list[MixedRequest]) -> list[MixedRequest]:
    return [
        replace(request, index=index)
        for index, request in enumerate(requests)
    ]


def _manifest_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _manifest_arrival_seconds(row: dict[str, Any]) -> float | None:
    if "scaled_arrival_time_sec" in row:
        return _optional_float(row.get("scaled_arrival_time_sec"))
    if "arrival_time_sec" in row:
        return _optional_float(row.get("arrival_time_sec"))
    return None


def load_manifest_workload(args: argparse.Namespace) -> MixedWorkload:
    manifest_path = Path(str(args.workload_manifest))
    requests: list[MixedRequest] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Manifest row {line_index} is not an object")
            meta = dict(row.get("meta") or {})
            for key in (
                "trace_index",
                "trace_request_tokens",
                "trace_response_tokens",
                "trace_total_tokens",
                "input_bucket",
                "output_bucket",
                "arrival_time_sec",
                "scaled_arrival_time_sec",
            ):
                if key in row and key not in meta:
                    meta[key] = row[key]
            dataset = str(row["dataset"])
            requests.append(
                MixedRequest(
                    index=int(row.get("request_index", len(requests))),
                    task=str(row["task"]),
                    dataset=dataset,
                    source_index=int(row.get("source_index", len(requests))),
                    record=PromptRecord(
                        dataset=dataset,
                        prompt=str(row["prompt"]),
                        answers=[str(item) for item in _manifest_list(row.get("answers"))],
                        all_classes=[
                            str(item) for item in _manifest_list(row.get("all_classes"))
                        ],
                        meta=meta,
                    ),
                    max_new_tokens=int(row["max_new_tokens"]),
                    slo_class=str(row.get("slo_class", "normal")),
                    priority=int(row.get("priority", 0)),
                    arrival_time_seconds=_manifest_arrival_seconds(row),
                    trace_request_tokens=_optional_int(row.get("trace_request_tokens")),
                    trace_response_tokens=_optional_int(row.get("trace_response_tokens")),
                    trace_total_tokens=_optional_int(row.get("trace_total_tokens")),
                    trace_index=_optional_int(row.get("trace_index")),
                    input_bucket=(
                        str(row["input_bucket"]) if row.get("input_bucket") is not None else None
                    ),
                    output_bucket=(
                        str(row["output_bucket"]) if row.get("output_bucket") is not None else None
                    ),
                )
            )
    if not requests:
        raise ValueError(f"Manifest contains no requests: {manifest_path}")
    return MixedWorkload(requests=requests, prompt_fit_summaries=[])


def _balanced_mix_requests(
    requests: list[MixedRequest],
    rng: random.Random,
) -> list[MixedRequest]:
    grouped: dict[tuple[str, str], list[MixedRequest]] = defaultdict(list)
    for request in requests:
        grouped[(request.task, request.dataset)].append(request)

    queues: dict[tuple[str, str], deque[MixedRequest]] = {}
    for key, group in grouped.items():
        rng.shuffle(group)
        queues[key] = deque(group)

    mixed: list[MixedRequest] = []
    active_keys = [key for key, queue in queues.items() if queue]
    while active_keys:
        round_keys = list(active_keys)
        rng.shuffle(round_keys)
        for key in round_keys:
            queue = queues[key]
            if queue:
                mixed.append(queue.popleft())
        active_keys = [key for key, queue in queues.items() if queue]
    return mixed


def load_mixed_workload(
    args: argparse.Namespace,
    chat_formatter: Callable[[str], str] | None,
) -> MixedWorkload:
    if getattr(args, "workload_manifest", None):
        return load_manifest_workload(args)

    rng = random.Random(args.seed)
    slo_classes = parse_slo_classes(args)
    requests: list[MixedRequest] = []
    prompt_fit_summaries: list[dict[str, Any]] = []

    for dataset in _load_task_datasets(args, chat_formatter):
        dataset, prompt_fit_summary = fit_dataset_to_model_len(args, dataset)
        prompt_fit_summaries.append(prompt_fit_summary)
        for source_index, record in enumerate(dataset.records):
            slo = rng.choice(slo_classes)
            requests.append(
                MixedRequest(
                    index=len(requests),
                    task=dataset.task,
                    dataset=dataset.dataset,
                    source_index=source_index,
                    record=record,
                    max_new_tokens=dataset.max_new_tokens,
                    slo_class=slo.name,
                    priority=slo.priority,
                )
            )

    mix_policy = getattr(args, "workload_mix_policy", "balanced")
    if mix_policy == "balanced":
        requests = _balanced_mix_requests(requests, rng)
    elif mix_policy == "random":
        rng.shuffle(requests)
    else:
        raise ValueError(f"Unsupported workload mix policy: {mix_policy!r}")
    return MixedWorkload(
        requests=_reindex_requests(requests),
        prompt_fit_summaries=prompt_fit_summaries,
    )


def _client_request_id(request: MixedRequest) -> str:
    return f"semantiq-mixed-{request.index:06d}"


async def run_mixed_serving_requests(
    *,
    args: argparse.Namespace,
    workload: MixedWorkload,
    base_url: str,
) -> list[MixedPrediction]:
    rng = random.Random(args.seed)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    request_rate = single_runner._parse_request_rate(str(args.request_rate))
    workload_start = time.perf_counter()
    request_timeout_sec = max(float(args.request_timeout_sec), 0.001)
    connection_timeout_sec = min(30.0, request_timeout_sec)
    timeout = httpx.Timeout(
        request_timeout_sec,
        connect=connection_timeout_sec,
        read=request_timeout_sec,
        write=connection_timeout_sec,
        pool=connection_timeout_sec,
    )
    limits = httpx.Limits(
        max_connections=max(4, args.max_concurrency * 2),
        max_keepalive_connections=max(1, args.max_concurrency),
    )

    async def run_one(
        client: httpx.AsyncClient,
        request: MixedRequest,
        queued_offset_seconds: float,
    ) -> MixedPrediction:
        async with semaphore:
            request_id = _client_request_id(request)
            start = time.perf_counter()
            try:
                priority = (
                    request.priority
                    if getattr(args, "scheduling_policy", "fcfs") == "priority"
                    else None
                )
                pred = await asyncio.wait_for(
                    _async_completion_request(
                        client=client,
                        base_url=base_url,
                        model=args.model,
                        record=request.record,
                        max_tokens=request.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        priority=priority,
                        request_id=request_id,
                    ),
                    timeout=request_timeout_sec,
                )
                error = None
            except (
                asyncio.TimeoutError,
                httpx.HTTPError,
                OSError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                pred = ""
                error = f"{type(exc).__name__}: {exc}"
            end = time.perf_counter()
            return MixedPrediction(
                request_index=request.index,
                pred=pred,
                error=error,
                latency_seconds=end - start,
                request_id=request_id,
                queued_offset_seconds=queued_offset_seconds,
                start_offset_seconds=start - workload_start,
                end_offset_seconds=end - workload_start,
            )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        tasks = []
        for request in workload.requests:
            if getattr(args, "arrival_policy", "poisson") == "trace":
                if request.arrival_time_seconds is None:
                    raise ValueError(
                        "--arrival-policy trace requires arrival_time_sec or "
                        "scaled_arrival_time_sec in every manifest row"
                    )
                target_offset_seconds = (
                    max(0.0, request.arrival_time_seconds)
                    * float(getattr(args, "trace_time_scale", 1.0))
                )
                current_offset_seconds = time.perf_counter() - workload_start
                delay = target_offset_seconds - current_offset_seconds
                if delay > 0:
                    await asyncio.sleep(delay)
            elif request.index > 0 and math.isfinite(request_rate):
                await asyncio.sleep(rng.expovariate(request_rate))
            queued_offset_seconds = time.perf_counter() - workload_start
            tasks.append(
                asyncio.create_task(
                    run_one(client, request, queued_offset_seconds)
                )
            )
        predictions = await asyncio.gather(*tasks)
    return sorted(predictions, key=lambda item: item.request_index)


def _latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {
            "avg_latency_seconds": 0.0,
            "max_latency_seconds": 0.0,
            "p95_latency_seconds": 0.0,
        }
    ordered = sorted(latencies)
    p95_index = min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)
    return {
        "avg_latency_seconds": mean(latencies),
        "max_latency_seconds": max(latencies),
        "p95_latency_seconds": ordered[p95_index],
    }


def _metric_values(metrics: dict[str, Any], prefix: str) -> list[float]:
    values: list[float] = []
    for key, value in metrics.items():
        if key.startswith(prefix) and isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _max_metric(metrics: dict[str, Any], prefix: str) -> float | None:
    values = _metric_values(metrics, prefix)
    return max(values) if values else None


def _mean_float(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _empty_role_metrics() -> dict[str, Any]:
    return {
        "samples": 0,
        "max_kv_cache_usage_pct": None,
        "avg_kv_cache_usage_pct": None,
        "max_running": None,
        "avg_running": None,
        "max_waiting": None,
        "avg_waiting": None,
    }


def _summarize_role_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    kv_values: list[float] = []
    running_values: list[float] = []
    waiting_values: list[float] = []

    for metrics in samples:
        kv_usage = _max_metric(metrics, "vllm:kv_cache_usage_perc")
        running = _max_metric(metrics, "vllm:num_requests_running")
        waiting = _max_metric(metrics, "vllm:num_requests_waiting")
        if kv_usage is not None:
            kv_values.append(kv_usage * 100)
        if running is not None:
            running_values.append(running)
        if waiting is not None:
            waiting_values.append(waiting)

    summary = _empty_role_metrics()
    summary.update(
        {
            "samples": len(samples),
            "max_kv_cache_usage_pct": max(kv_values) if kv_values else None,
            "avg_kv_cache_usage_pct": _mean_float(kv_values),
            "max_running": int(max(running_values)) if running_values else None,
            "avg_running": _mean_float(running_values),
            "max_waiting": int(max(waiting_values)) if waiting_values else None,
            "avg_waiting": _mean_float(waiting_values),
        }
    )
    return summary


def summarize_serving_metrics(run_dir: Path) -> dict[str, dict[str, Any]]:
    metrics_path = run_dir / "metrics_samples.jsonl"
    by_role: dict[str, list[dict[str, Any]]] = {"prefill": [], "decode": []}
    if not metrics_path.exists():
        return {
            "prefill": _empty_role_metrics(),
            "decode": _empty_role_metrics(),
        }

    with metrics_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(sample, dict):
                continue
            metrics_by_role = sample.get("vllm_metrics")
            if not isinstance(metrics_by_role, dict):
                continue
            for role in ("prefill", "decode"):
                metrics = metrics_by_role.get(role)
                if isinstance(metrics, dict):
                    by_role[role].append(metrics)

    return {
        role: _summarize_role_metrics(samples)
        for role, samples in by_role.items()
    }


def summarize_reflex_trace(run_dir: Path) -> dict[str, Any]:
    return _trace_stats(_read_trace_events(run_dir))


def _group_requests_by_dataset(
    workload: MixedWorkload,
) -> dict[str, list[MixedRequest]]:
    grouped: dict[str, list[MixedRequest]] = {}
    for request in workload.requests:
        grouped.setdefault(request.dataset, []).append(request)
    for requests in grouped.values():
        requests.sort(key=lambda item: item.source_index)
    return grouped


def _write_request_manifest(run_dir: Path, workload: MixedWorkload) -> None:
    rows = [
        {
            "request_index": request.index,
            "task": request.task,
            "dataset": request.dataset,
            "source_index": request.source_index,
                "max_new_tokens": request.max_new_tokens,
                "slo_class": request.slo_class,
                "priority": request.priority,
                "arrival_time_seconds": request.arrival_time_seconds,
                "trace_index": request.trace_index,
                "trace_request_tokens": request.trace_request_tokens,
                "trace_response_tokens": request.trace_response_tokens,
                "trace_total_tokens": request.trace_total_tokens,
                "input_bucket": request.input_bucket,
                "output_bucket": request.output_bucket,
            }
        for request in sorted(workload.requests, key=lambda item: item.index)
    ]
    single_runner._append_jsonl(run_dir / "mixed_requests.jsonl", rows)


def _write_request_trace(
    run_dir: Path,
    workload: MixedWorkload,
    predictions: list[MixedPrediction],
) -> None:
    prediction_by_index = {
        prediction.request_index: prediction for prediction in predictions
    }
    rows = []
    for request in sorted(workload.requests, key=lambda item: item.index):
        prediction = prediction_by_index[request.index]
        request_id = prediction.request_id or _client_request_id(request)
        rows.append(
            {
                "request_index": request.index,
                "request_id": request_id,
                "vllm_request_id_prefix": f"cmpl-{request_id}",
                "task": request.task,
                "dataset": request.dataset,
                "source_index": request.source_index,
                "max_new_tokens": request.max_new_tokens,
                "slo_class": request.slo_class,
                "priority": request.priority,
                "arrival_time_seconds": request.arrival_time_seconds,
                "trace_index": request.trace_index,
                "trace_request_tokens": request.trace_request_tokens,
                "trace_response_tokens": request.trace_response_tokens,
                "trace_total_tokens": request.trace_total_tokens,
                "input_bucket": request.input_bucket,
                "output_bucket": request.output_bucket,
                "prompt_chars": len(request.record.prompt),
                "answer_count": len(request.record.answers),
                "prediction_chars": len(prediction.pred),
                "prompt_original_tokens": (
                    (request.record.meta or {}).get("prompt_original_tokens")
                ),
                "prompt_final_tokens": (
                    (request.record.meta or {}).get("prompt_final_tokens")
                ),
                "prompt_truncated": (
                    (request.record.meta or {}).get("prompt_truncated")
                ),
                "queued_offset_seconds": prediction.queued_offset_seconds,
                "start_offset_seconds": prediction.start_offset_seconds,
                "end_offset_seconds": prediction.end_offset_seconds,
                "latency_seconds": prediction.latency_seconds,
                "error": prediction.error,
            }
        )
    single_runner._append_jsonl(run_dir / "mixed_request_trace.jsonl", rows)


def write_mixed_predictions_and_scores(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    workload: MixedWorkload,
    predictions: list[MixedPrediction],
    duration_seconds: float,
) -> dict[str, Any]:
    prediction_by_index = {
        prediction.request_index: prediction for prediction in predictions
    }
    _write_request_manifest(run_dir, workload)
    _write_request_trace(run_dir, workload, predictions)
    summaries: dict[str, dict[str, Any]] = {}
    slo_latencies: dict[str, list[float]] = {}

    for dataset_name, requests in _group_requests_by_dataset(workload).items():
        dataset_dir = run_dir / dataset_name
        pred_file = dataset_dir / "pred.jsonl"
        rows = []
        latencies = []
        failed_predictions = 0
        task = requests[0].task

        for request in requests:
            prediction = prediction_by_index[request.index]
            if prediction.error is not None or prediction.pred == "":
                failed_predictions += 1
            latencies.append(prediction.latency_seconds)
            slo_latencies.setdefault(request.slo_class, []).append(
                prediction.latency_seconds
            )
            rows.append(
                {
                    "pred": prediction.pred,
                    "answers": request.record.answers,
                    "all_classes": request.record.all_classes,
                    "meta": {
                        **(request.record.meta or {}),
                        "task": request.task,
                        "dataset": request.dataset,
                        "source_index": request.source_index,
                        "request_index": request.index,
                        "max_new_tokens": request.max_new_tokens,
                        "slo_class": request.slo_class,
                        "priority": request.priority,
                        "latency_seconds": prediction.latency_seconds,
                        "error": prediction.error,
                    },
                }
            )

        single_runner._append_jsonl(pred_file, rows)
        single_runner._write_json(dataset_dir / "run_config.json", vars(args))
        if task == "longbench":
            score = float(
                evaluate_longbench_file(str(pred_file), dataset_name, args.config_dir)
            )
        else:
            score = float(
                evaluate_reasoning_file(str(pred_file), dataset_name, args.config_dir)
            )

        summary = {
            "dataset": dataset_name,
            "task": task,
            "requested_samples": len(requests),
            "completed_predictions": len(requests),
            "failed_predictions": failed_predictions,
            "avg_score": score,
            "duration_seconds": round(duration_seconds, 4),
            **_latency_summary(latencies),
            "request_rate": args.request_rate,
            "max_concurrency": args.max_concurrency,
        }
        single_runner._write_json(dataset_dir / "run_summary.json", summary)
        summaries[dataset_name] = summary

    mixed_summary = {
        "duration_seconds": round(duration_seconds, 4),
        "total_requests": len(workload.requests),
        "workload_mix_policy": getattr(args, "workload_mix_policy", "balanced"),
        "workload_manifest": getattr(args, "workload_manifest", None),
        "arrival_policy": getattr(args, "arrival_policy", "poisson"),
        "trace_time_scale": getattr(args, "trace_time_scale", 1.0),
        "prompt_fit": workload.prompt_fit_summaries,
        "datasets": summaries,
        "slo_latency": {
            slo_class: _latency_summary(latencies)
            for slo_class, latencies in sorted(slo_latencies.items())
        },
        "serving_metrics": summarize_serving_metrics(run_dir),
        "reflex_trace": summarize_reflex_trace(run_dir),
    }
    single_runner._write_json(run_dir / "mixed_summary.json", mixed_summary)
    return mixed_summary


def _sanitize_label(value: str) -> str:
    return (
        value.replace(",", "+")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace(" ", "_")
    )


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rate = _sanitize_label(str(args.request_rate))
        run_name = (
            f"{stamp}_pdserv_mixed_{_sanitize_label(args.tasks)}_"
            f"kv-{_sanitize_label(args.decode_kv_cache_dtype)}_"
            f"c{args.max_concurrency}_ln{args.longbench_max_samples}_"
            f"mn{args.reasoning_max_samples}_r{rate}"
        )
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> int:
    args = parse_args()
    pd_runner.validate_args(args)
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")
    parse_slo_classes(args)

    run_dir = make_run_dir(args)
    prefill_url = f"http://{args.host}:{args.prefill_port}"
    decode_url = f"http://{args.host}:{args.decode_port}"
    proxy_url = f"http://{args.host}:{args.proxy_port}"

    prefill_cmd = build_server_cmd(args, Role.PREFILL)
    decode_cmd = build_server_cmd(args, Role.DECODE)
    proxy_cmd = build_proxy_cmd(args)
    single_runner._write_json(run_dir / "config.json", vars(args))
    (run_dir / "prefill_server_cmd.txt").write_text(
        " ".join(prefill_cmd) + "\n", encoding="utf-8"
    )
    (run_dir / "decode_server_cmd.txt").write_text(
        " ".join(decode_cmd) + "\n", encoding="utf-8"
    )
    (run_dir / "proxy_cmd.txt").write_text(
        " ".join(proxy_cmd) + "\n", encoding="utf-8"
    )

    chat_formatter = build_chat_formatter(args)
    workload = load_mixed_workload(args, chat_formatter)
    single_runner._write_json(
        run_dir / "prompt_fit_summary.json",
        workload.prompt_fit_summaries,
    )

    procs = []
    stop_event = threading.Event()
    sampler_thread: threading.Thread | None = None
    start = time.perf_counter()
    try:
        prefill_proc = pd_runner.launch_process(
            prefill_cmd,
            env=build_server_env(args, Role.PREFILL),
            log_path=run_dir / "prefill_server.log",
        )
        procs.append(prefill_proc)
        decode_proc = pd_runner.launch_process(
            decode_cmd,
            env=build_server_env(args, Role.DECODE),
            log_path=run_dir / "decode_server.log",
        )
        procs.append(decode_proc)

        pd_runner.wait_for_http_ok(
            f"{prefill_url}/v1/models",
            args.server_ready_timeout_sec,
        )
        pd_runner.wait_for_http_ok(
            f"{decode_url}/v1/models",
            args.server_ready_timeout_sec,
        )

        proxy_proc = pd_runner.launch_process(
            proxy_cmd,
            env=pd_runner.build_base_env(),
            log_path=run_dir / "proxy.log",
        )
        procs.append(proxy_proc)
        pd_runner.wait_for_tcp_port(
            args.host,
            args.proxy_port,
            args.server_ready_timeout_sec,
        )

        sampler_thread = threading.Thread(
            target=pd_runner.sampler,
            args=(
                {"prefill": prefill_url, "decode": decode_url},
                args.sample_interval_sec,
                run_dir / "metrics_samples.jsonl",
                stop_event,
            ),
            daemon=True,
        )
        sampler_thread.start()

        predictions = asyncio.run(
            run_mixed_serving_requests(args=args, workload=workload, base_url=proxy_url)
        )
        stop_event.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=5)
            sampler_thread = None
        summary = write_mixed_predictions_and_scores(
            args=args,
            run_dir=run_dir,
            workload=workload,
            predictions=predictions,
            duration_seconds=time.perf_counter() - start,
        )
        print(f"run_dir={run_dir}")
        print(json.dumps(summary["datasets"], ensure_ascii=False, sort_keys=True))
        return 0
    finally:
        stop_event.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=5)
        for proc in reversed(procs):
            pd_runner.terminate_process_group(proc)
        for proc in procs:
            pd_runner.close_process_log(proc)


if __name__ == "__main__":
    raise SystemExit(main())
