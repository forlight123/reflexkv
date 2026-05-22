#!/usr/bin/env python3
"""Summarize vLLM Full-KV pressure profiling runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _metric_max(metrics: dict[str, float], prefix: str) -> float | None:
    vals = [value for key, value in metrics.items() if key.startswith(prefix)]
    return max(vals) if vals else None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _request_tpots_ms(bench: dict[str, Any]) -> list[float]:
    tpots = []
    for itls in bench.get("itls", []):
        if itls:
            tpots.append(sum(itls) / len(itls) * 1000)
    return tpots


def _rate(values: list[float], threshold: float) -> float | None:
    if not values:
        return None
    return sum(value > threshold for value in values) / len(values)


def _safe_get(bench: dict[str, Any], key: str, default: Any = None) -> Any:
    return bench.get(key, default)


def summarize_run(
    run_dir: Path,
    *,
    slo_ttft_ms: float,
    slo_tpot_ms: float,
    demotion_threshold_pct: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bench_path = run_dir / "bench_result.json"
    metrics_path = run_dir / "metrics_samples.jsonl"
    bench = json.loads(bench_path.read_text(encoding="utf-8"))

    timeline: list[dict[str, Any]] = []
    max_kv = 0.0
    max_running = 0.0
    max_waiting = 0.0
    first_time: float | None = None

    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                sample = json.loads(line)
                timestamp = float(sample["time"])
                if first_time is None:
                    first_time = timestamp
                metrics = sample.get("vllm_metrics", {})
                kv_usage = _metric_max(metrics, "vllm:kv_cache_usage_perc")
                running = _metric_max(metrics, "vllm:num_requests_running")
                waiting = _metric_max(metrics, "vllm:num_requests_waiting")
                kv_pct = None if kv_usage is None else kv_usage * 100
                max_kv = max(max_kv, kv_pct or 0.0)
                max_running = max(max_running, running or 0.0)
                max_waiting = max(max_waiting, waiting or 0.0)
                pressure_action = "none"
                if kv_pct is not None and kv_pct >= demotion_threshold_pct:
                    pressure_action = "demotion_would_trigger"
                timeline.append(
                    {
                        "run": run_dir.name,
                        "elapsed_s": timestamp - first_time,
                        "kv_cache_usage_pct": kv_pct,
                        "num_requests_running": running,
                        "num_requests_waiting": waiting,
                        "fullkv_precision_state": "FP16",
                        "pressure_action": pressure_action,
                    }
                )

    ttfts_ms = [value * 1000 for value in bench.get("ttfts", [])]
    tpots_ms = _request_tpots_ms(bench)
    input_lens = [float(value) for value in bench.get("input_lens", [])]

    summary = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "num_prompts": _safe_get(bench, "num_prompts"),
        "completed": _safe_get(bench, "completed"),
        "failed": _safe_get(bench, "failed"),
        "request_rate": _safe_get(bench, "request_rate"),
        "max_concurrency": _safe_get(bench, "max_concurrency"),
        "duration_s": _safe_get(bench, "duration"),
        "total_input_tokens": _safe_get(bench, "total_input_tokens"),
        "total_output_tokens": _safe_get(bench, "total_output_tokens"),
        "avg_input_len": _mean(input_lens),
        "min_input_len": min(input_lens) if input_lens else None,
        "max_input_len": max(input_lens) if input_lens else None,
        "request_throughput": _safe_get(bench, "request_throughput"),
        "output_throughput": _safe_get(bench, "output_throughput"),
        "total_token_throughput": _safe_get(bench, "total_token_throughput"),
        "mean_ttft_ms": _safe_get(bench, "mean_ttft_ms"),
        "p95_ttft_ms": _safe_get(bench, "p95_ttft_ms"),
        "p99_ttft_ms": _safe_get(bench, "p99_ttft_ms"),
        "mean_tpot_ms": _safe_get(bench, "mean_tpot_ms"),
        "p95_tpot_ms": _safe_get(bench, "p95_tpot_ms"),
        "p99_tpot_ms": _safe_get(bench, "p99_tpot_ms"),
        "mean_e2el_ms": _safe_get(bench, "mean_e2el_ms"),
        "p95_e2el_ms": _safe_get(bench, "p95_e2el_ms"),
        "p99_e2el_ms": _safe_get(bench, "p99_e2el_ms"),
        "slo_ttft_ms": slo_ttft_ms,
        "slo_tpot_ms": slo_tpot_ms,
        "ttft_slo_violation_rate": _rate(ttfts_ms, slo_ttft_ms),
        "tpot_slo_violation_rate": _rate(tpots_ms, slo_tpot_ms),
        "max_kv_cache_usage_pct": max_kv,
        "max_requests_running": int(max_running),
        "max_requests_waiting": int(max_waiting),
        "demotion_threshold_pct": demotion_threshold_pct,
        "demotion_would_trigger": max_kv >= demotion_threshold_pct,
        "timeline_samples": len(timeline),
    }
    return summary, timeline


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeline-out", type=Path, default=None)
    parser.add_argument("--slo-ttft-ms", type=float, default=30_000)
    parser.add_argument("--slo-tpot-ms", type=float, default=50)
    parser.add_argument("--demotion-threshold-pct", type=float, default=85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summaries = []
    timelines = []
    for run_dir in args.run_dirs:
        summary, timeline = summarize_run(
            run_dir,
            slo_ttft_ms=args.slo_ttft_ms,
            slo_tpot_ms=args.slo_tpot_ms,
            demotion_threshold_pct=args.demotion_threshold_pct,
        )
        summaries.append(summary)
        timelines.extend(timeline)
    _write_csv(args.out, summaries)
    if args.timeline_out is not None:
        _write_csv(args.timeline_out, timelines)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
