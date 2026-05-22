#!/usr/bin/env python3
"""Summarize ReFlexKV 1P1D pressure sweep profiling runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable


SUMMARY_FIELDS = [
    "run",
    "run_dir",
    "decode_kv_cache_dtype",
    "input_len",
    "output_len",
    "num_prompts",
    "max_concurrency",
    "num_gpu_blocks_override",
    "completed",
    "failed",
    "req/s",
    "total_token_throughput",
    "mean_tpot_ms",
    "p95_tpot_ms",
    "p99_tpot_ms",
    "mean_ttft_ms",
    "p95_ttft_ms",
    "p99_ttft_ms",
    "mean_e2el_ms",
    "p95_e2el_ms",
    "p99_e2el_ms",
    "max_decode_running",
    "avg_decode_running",
    "max_decode_waiting",
    "avg_decode_waiting",
    "max_decode_kv_usage_pct",
    "demotion_event_count",
    "demoted_pages_total",
    "released_bf16_blocks_total",
    "actual_release_blocks_total",
    "planned_bf16_blocks_total",
    "target_release_blocks_total",
    "precision_budget_event_count",
    "precision_budget_max_int4_pages_total",
    "precision_budget_release_budget_total",
    "precision_budget_priority_total",
    "admission_control_event_count",
    "admission_requested_release_total",
    "admission_candidate_release_capacity_total",
    "admission_feasible_release_total",
    "admission_planned_release_total",
    "admission_actual_release_total",
    "admission_success_after_demote_total",
    "admission_blocked_total",
    "admission_infeasible_total",
    "admission_wait_reduction_total",
    "admission_mixed_landing_feasible_total",
    "admission_required_int4_landing_total",
    "admission_eligible_int4_landing_total",
    "admission_planned_int4_landing_total",
    "admission_residual_bf16_deficit_total",
    "landing_contract_event_count",
    "landing_contract_persisted_pages_total",
    "landing_contract_direct_pages_total",
    "landing_materialize_event_count",
    "landing_materialized_pages_total",
    "landing_materialize_layer_copies_total",
    "landing_materialize_gpu_ms_total",
    "landing_commit_event_count",
    "landing_committed_pages_total",
    "landing_policy_event_count",
    "landing_fallback_event_count",
    "landing_fallback_pages_total",
    "landing_fallback_unmaterialized_total",
    "landing_fallback_unmaterialized_ratio",
    "page_metadata_produce_event_count",
    "page_metadata_produced_requests_total",
    "page_metadata_produced_pages_total",
    "page_metadata_receive_event_count",
    "page_metadata_received_requests_total",
    "page_metadata_received_pages_total",
    "page_metadata_plan_event_count",
    "page_metadata_plan_real_risk_requests_total",
    "page_metadata_plan_real_risk_pages_total",
    "page_metadata_plan_compressible_requests_total",
    "page_metadata_plan_compressible_pages_total",
    "page_metadata_plan_shadow_requests_total",
    "page_metadata_plan_shadow_pages_total",
    "page_metadata_plan_synthetic_requests_total",
    "page_metadata_plan_synthetic_pages_total",
    "page_metadata_real_risk_coverage_ratio",
    "landing_metadata_source_counts",
    "landing_real_risk_pages_total",
    "landing_explicit_compressible_pages_total",
    "landing_synthetic_pages_total",
    "recovery_plan_event_count",
    "background_promoted_pages_total",
    "recovery_exec_event_count",
    "recovery_exec_pages_total",
    "recovery_exec_layer_copies_total",
    "recovery_exec_cpu_ms_total",
    "candidate_breakdown_event_count",
    "candidate_raw_bf16_pages_total",
    "candidate_open_bf16_pages_total",
    "candidate_remote_inflight_bf16_pages_total",
    "candidate_open_tail_bf16_pages_total",
    "candidate_request_protected_bf16_pages_total",
    "candidate_shared_bf16_pages_total",
    "candidate_prompt_protected_bf16_pages_total",
    "candidate_copy_on_demote_pages_total",
    "candidate_eligible_full_unshared_pages_total",
    "candidate_after_initial_recent_protection_total",
    "candidate_after_low_risk_filter_total",
    "candidate_after_request_budget_cap_total",
    "candidate_after_sparse_window_quota_total",
    "candidate_after_frontier_optimizer_total",
    "candidate_after_int4_pool_limit_total",
    "candidate_selected_actual_total",
    "demotion_gpu_ms_total",
    "bf16_capacity_blocks",
    "int4_capacity_blocks",
    "bf16_budget_bytes",
    "int4_budget_bytes",
    "int4_budget_fraction",
    "mean_int4_ratio",
    "max_int4_ratio",
    "mean_forward_gpu_ms",
    "max_forward_gpu_ms",
    "attention_trace_event_count",
    "attention_gpu_ms_total",
    "timeline_samples",
    "trace_events",
]

TIMELINE_FIELDS = [
    "run",
    "run_dir",
    "elapsed_s",
    "decode_kv_cache_usage_pct",
    "decode_running",
    "decode_waiting",
]

_PLANNED_RE = re.compile(
    r"ReFlexKV planned (?P<released>\d+)/(?P<planned>\d+) "
    r"BF16->INT4 KV block demotions for (?P<reason>[^;]+); (?P<rest>.*)"
)
_PLANNED_KV_RE = re.compile(
    r"ReFlexKV planned BF16->INT4 KV block demotions for "
    r"(?P<reason>[^;]+); (?P<rest>.*)"
)
_TRACE_RE = re.compile(r"ReFlexKV trace (?P<event>[A-Za-z_]+)(?P<inline>=[^ ]+)?(?: (?P<rest>.*))?")
_KEY_VALUE_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[^ ]+)")

FRONTIER_LEVEL_NAMES = (
    "pinned",
    "protected",
    "candidate",
    "eager_compressible",
    "low_precision",
)
REJECTION_REASON_NAMES = (
    "shared_or_open",
    "recent_or_initial",
    "high_risk",
    "request_fraction_cap",
    "quality_debt_cap",
    "request_release_budget",
    "short_decode_protection",
    "reasoning_prompt_protection",
    "request_budget",
    "sparse_quota",
    "frontier_optimizer",
    "int4_pool_full",
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _number(value: str) -> Any:
    value = value.rstrip(".")
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _parse_key_values(text: str) -> dict[str, Any]:
    return {
        match.group("key"): _number(match.group("value"))
        for match in _KEY_VALUE_RE.finditer(text)
    }


def _empty_named_counts(names: tuple[str, ...]) -> dict[str, int]:
    return {name: 0 for name in names}


def parse_trace_colon_counts(value: Any) -> dict[str, int]:
    if not isinstance(value, str) or not value or value == "none":
        return {}
    counts: dict[str, int] = {}
    for item in value.rstrip(".").split(","):
        if not item or ":" not in item:
            continue
        name, raw_count = item.split(":", 1)
        name = name.strip()
        if not name:
            continue
        try:
            counts[name] = int(float(raw_count))
        except ValueError:
            continue
    return counts


def sum_trace_colon_counts(
    events: Iterable[dict[str, Any]],
    key: str,
    *,
    names: tuple[str, ...],
) -> dict[str, int]:
    totals = _empty_named_counts(names)
    for event in events:
        for name, count in parse_trace_colon_counts(event.get(key)).items():
            totals[name] = totals.get(name, 0) + count
    return totals


def count_trace_field_values(
    events: Iterable[dict[str, Any]],
    key: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        value = event.get(key)
        if value is None or value == "" or value == "none":
            continue
        name = str(value).rstrip(".")
        counts[name] = counts.get(name, 0) + 1
    return counts


def parse_reflex_trace_events(lines: Iterable[str], *, run: str = "") -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        planned = _PLANNED_RE.search(line)
        if planned is not None:
            rest = _parse_key_values(planned.group("rest"))
            events.append(
                {
                    "run": run,
                    "line_no": line_no,
                    "event": "planned",
                    "released_bf16_blocks": int(planned.group("released")),
                    "planned_bf16_blocks": int(planned.group("planned")),
                    "actual_release_blocks": int(planned.group("released")),
                    "target_release_blocks": int(planned.group("planned")),
                    "reason": planned.group("reason").strip(),
                    **rest,
                }
            )
            continue

        planned_kv = _PLANNED_KV_RE.search(line)
        if planned_kv is not None:
            rest = _parse_key_values(planned_kv.group("rest"))
            events.append(
                {
                    "run": run,
                    "line_no": line_no,
                    "event": "planned",
                    "released_bf16_blocks": int(rest.get("actual_release", 0)),
                    "planned_bf16_blocks": int(rest.get("target_release", 0)),
                    "actual_release_blocks": int(rest.get("actual_release", 0)),
                    "target_release_blocks": int(rest.get("target_release", 0)),
                    "reason": planned_kv.group("reason").strip(),
                    **rest,
                }
            )
            continue

        trace = _TRACE_RE.search(line)
        if trace is None:
            continue
        event_name = trace.group("event")
        event = {
            "run": run,
            "line_no": line_no,
            "event": event_name,
        }
        inline = trace.group("inline")
        rest = trace.group("rest")
        if inline:
            event.update(_parse_key_values(f"{event_name}{inline}"))
        if rest:
            event.update(_parse_key_values(rest))
        events.append(event)
    return events


def _metric_max(metrics: dict[str, Any], prefix: str) -> float | None:
    vals = [
        float(value)
        for key, value in metrics.items()
        if key.startswith(prefix) and isinstance(value, (int, float))
    ]
    return max(vals) if vals else None


def _decode_metrics(sample: dict[str, Any]) -> dict[str, Any]:
    metrics = sample.get("vllm_metrics", {})
    if not isinstance(metrics, dict):
        return {}
    decode = metrics.get("decode")
    if isinstance(decode, dict):
        return decode
    return metrics


def _read_timeline(run_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    metrics_path = run_dir / "metrics_samples.jsonl"
    timeline: list[dict[str, Any]] = []
    kv_values: list[float] = []
    running_values: list[float] = []
    waiting_values: list[float] = []
    first_time: float | None = None

    if not metrics_path.exists():
        stats = {
            "max_decode_kv_usage_pct": None,
            "max_decode_running": None,
            "avg_decode_running": None,
            "max_decode_waiting": None,
            "avg_decode_waiting": None,
            "timeline_samples": 0,
        }
        return timeline, stats

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
            try:
                timestamp = float(sample["time"])
            except (KeyError, TypeError, ValueError):
                continue
            if first_time is None:
                first_time = timestamp

            metrics = _decode_metrics(sample)
            kv_usage = _metric_max(metrics, "vllm:kv_cache_usage_perc")
            running = _metric_max(metrics, "vllm:num_requests_running")
            waiting = _metric_max(metrics, "vllm:num_requests_waiting")
            kv_pct = None if kv_usage is None else kv_usage * 100

            if kv_pct is not None:
                kv_values.append(kv_pct)
            if running is not None:
                running_values.append(running)
            if waiting is not None:
                waiting_values.append(waiting)

            timeline.append(
                {
                    "run": run_dir.name,
                    "run_dir": str(run_dir),
                    "elapsed_s": timestamp - first_time,
                    "decode_kv_cache_usage_pct": kv_pct,
                    "decode_running": running,
                    "decode_waiting": waiting,
                }
            )

    stats = {
        "max_decode_kv_usage_pct": max(kv_values) if kv_values else None,
        "max_decode_running": int(max(running_values)) if running_values else None,
        "avg_decode_running": _mean(running_values),
        "max_decode_waiting": int(max(waiting_values)) if waiting_values else None,
        "avg_decode_waiting": _mean(waiting_values),
        "timeline_samples": len(timeline),
    }
    return timeline, stats


def _read_trace_events(run_dir: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for log_name in ("prefill_server.log", "decode_server.log", "proxy.log"):
        log_path = run_dir / log_name
        if not log_path.exists():
            continue
        try:
            with log_path.open(encoding="utf-8", errors="replace") as f:
                parsed = parse_reflex_trace_events(f, run=run_dir.name)
        except OSError:
            continue
        for event in parsed:
            event["log"] = log_name
        events.extend(parsed)
    return events


def _trace_stats(events: list[dict[str, Any]]) -> dict[str, Any]:
    demote_events = [event for event in events if event.get("event") == "demote_exec"]
    planned_events = [event for event in events if event.get("event") == "planned"]
    admission_events = [
        event for event in events if event.get("event") == "admission_control"
    ]
    blocked_admission_events = [
        event for event in admission_events if event.get("admission_blocked") is True
    ]
    precision_budget_events = [
        event for event in events if event.get("event") == "precision_budget"
    ]
    candidate_breakdown_events = [
        event for event in events if event.get("event") == "candidate_breakdown"
    ]
    landing_materialize_events = [
        event for event in events if event.get("event") == "landing_materialize"
    ]
    landing_contract_events = [
        event for event in events if event.get("event") == "landing_contract"
    ]
    landing_commit_events = [
        event for event in events if event.get("event") == "landing_commit"
    ]
    landing_policy_events = [
        event for event in events if event.get("event") == "landing_policy"
    ]
    landing_fallback_events = [
        event
        for event in landing_policy_events
        if str(event.get("outcome", "")).startswith("fallback")
    ]
    page_metadata_produce_events = [
        event for event in events if event.get("event") == "page_metadata_produce"
    ]
    page_metadata_receive_events = [
        event for event in events if event.get("event") == "page_metadata_receive"
    ]
    page_metadata_plan_events = [
        event for event in events if event.get("event") == "page_metadata_plan"
    ]
    recovery_plan_events = [
        event for event in events if event.get("event") == "recovery_plan"
    ]
    recovery_exec_events = [
        event for event in events if event.get("event") == "recovery_exec"
    ]
    step_events = [event for event in events if event.get("event") == "step"]
    attention_events = [event for event in events if event.get("event") == "attention"]

    int4_ratios = [
        float(event["kv_int4_ratio"])
        for event in step_events
        if isinstance(event.get("kv_int4_ratio"), (int, float))
    ]
    forward_gpu_ms = [
        float(event["forward_gpu_ms"])
        for event in step_events
        if isinstance(event.get("forward_gpu_ms"), (int, float))
    ]
    latest_step = step_events[-1] if step_events else {}

    actual_release_blocks_total = sum(
        int(
            event.get(
                "actual_release_blocks",
                event.get("released_bf16_blocks", 0),
            )
        )
        for event in planned_events
        if isinstance(
            event.get(
                "actual_release_blocks",
                event.get("released_bf16_blocks", 0),
            ),
            (int, float),
        )
    )
    target_release_blocks_total = sum(
        int(
            event.get(
                "target_release_blocks",
                event.get("planned_bf16_blocks", 0),
            )
        )
        for event in planned_events
        if isinstance(
            event.get(
                "target_release_blocks",
                event.get("planned_bf16_blocks", 0),
            ),
            (int, float),
        )
    )
    candidate_frontier_level_totals = sum_trace_colon_counts(
        candidate_breakdown_events,
        "frontier_levels",
        names=FRONTIER_LEVEL_NAMES,
    )
    candidate_rejection_reason_totals = sum_trace_colon_counts(
        candidate_breakdown_events,
        "rejection_reasons",
        names=REJECTION_REASON_NAMES,
    )
    admission_frontier_rejection_reason_totals = sum_trace_colon_counts(
        admission_events,
        "frontier_rejection_reasons",
        names=REJECTION_REASON_NAMES,
    )
    admission_blocked_frontier_rejection_reason_totals = sum_trace_colon_counts(
        blocked_admission_events,
        "frontier_rejection_reasons",
        names=REJECTION_REASON_NAMES,
    )
    admission_blocked_reason_counts = count_trace_field_values(
        admission_events,
        "blocked_reason",
    )
    admission_outcome_counts = count_trace_field_values(
        admission_events,
        "outcome",
    )
    landing_metadata_source_counts = count_trace_field_values(
        admission_events,
        "landing_metadata_source",
    )
    landing_materialized_pages_total = sum(
        int(event.get("pages", 0))
        for event in landing_materialize_events
        if isinstance(event.get("pages", 0), (int, float))
    )
    landing_contract_persisted_pages_total = sum(
        int(event.get("pages", 0))
        for event in landing_contract_events
        if isinstance(event.get("pages", 0), (int, float))
    )
    landing_fallback_pages_total = sum(
        int(event.get("planned_pages", 0))
        for event in landing_fallback_events
        if isinstance(event.get("planned_pages", 0), (int, float))
    )
    page_metadata_plan_real_risk_pages_total = sum(
        int(event.get("real_risk_pages", 0))
        for event in page_metadata_plan_events
        if isinstance(event.get("real_risk_pages", 0), (int, float))
    )
    page_metadata_plan_synthetic_pages_total = sum(
        int(event.get("synthetic_pages", 0))
        for event in page_metadata_plan_events
        if isinstance(event.get("synthetic_pages", 0), (int, float))
    )
    page_metadata_received_pages_total = sum(
        int(event.get("pages", 0))
        for event in page_metadata_receive_events
        if isinstance(event.get("pages", 0), (int, float))
    )

    return {
        "demotion_event_count": len(demote_events),
        "demoted_pages_total": sum(
            int(event.get("pages", 0))
            for event in demote_events
            if isinstance(event.get("pages", 0), (int, float))
        ),
        "released_bf16_blocks_total": actual_release_blocks_total,
        "actual_release_blocks_total": actual_release_blocks_total,
        "planned_bf16_blocks_total": target_release_blocks_total,
        "target_release_blocks_total": target_release_blocks_total,
        "precision_budget_event_count": len(precision_budget_events),
        "precision_budget_max_int4_pages_total": sum(
            int(event.get("max_int4_pages", 0))
            for event in precision_budget_events
            if isinstance(event.get("max_int4_pages", 0), (int, float))
        ),
        "precision_budget_release_budget_total": sum(
            int(event.get("release_budget_blocks", 0))
            for event in precision_budget_events
            if isinstance(event.get("release_budget_blocks", 0), (int, float))
        ),
        "precision_budget_priority_total": sum(
            float(event.get("priority", 0.0))
            for event in precision_budget_events
            if isinstance(event.get("priority", 0.0), (int, float))
        ),
        "admission_control_event_count": len(admission_events),
        "admission_requested_release_total": sum(
            int(event.get("requested_release", 0))
            for event in admission_events
            if isinstance(event.get("requested_release", 0), (int, float))
        ),
        "admission_candidate_release_capacity_total": sum(
            int(event.get("candidate_release_capacity", 0))
            for event in admission_events
            if isinstance(
                event.get("candidate_release_capacity", 0),
                (int, float),
            )
        ),
        "admission_feasible_release_total": sum(
            int(event.get("feasible_release", 0))
            for event in admission_events
            if isinstance(event.get("feasible_release", 0), (int, float))
        ),
        "admission_planned_release_total": sum(
            int(event.get("planned_release", 0))
            for event in admission_events
            if isinstance(event.get("planned_release", 0), (int, float))
        ),
        "admission_actual_release_total": sum(
            int(event.get("actual_release", 0))
            for event in admission_events
            if isinstance(event.get("actual_release", 0), (int, float))
        ),
        "admission_success_after_demote_total": sum(
            1
            for event in admission_events
            if event.get("admission_success_after_demote") is True
        ),
        "admission_blocked_total": sum(
            1 for event in admission_events if event.get("admission_blocked") is True
        ),
        "admission_infeasible_total": sum(
            1
            for event in admission_events
            if event.get("admission_infeasible") is True
        ),
        "admission_wait_reduction_total": sum(
            int(event.get("admission_wait_reduction", 0))
            for event in admission_events
            if isinstance(event.get("admission_wait_reduction", 0), (int, float))
        ),
        "admission_mixed_landing_feasible_total": sum(
            1
            for event in admission_events
            if event.get("landing_mixed_feasible") is True
        ),
        "admission_required_int4_landing_total": sum(
            int(event.get("landing_required_int4_blocks", 0))
            for event in admission_events
            if isinstance(event.get("landing_required_int4_blocks", 0), (int, float))
        ),
        "admission_eligible_int4_landing_total": sum(
            int(event.get("landing_eligible_int4_blocks", 0))
            for event in admission_events
            if isinstance(event.get("landing_eligible_int4_blocks", 0), (int, float))
        ),
        "admission_planned_int4_landing_total": sum(
            int(event.get("landing_planned_int4_blocks", 0))
            for event in admission_events
            if isinstance(event.get("landing_planned_int4_blocks", 0), (int, float))
        ),
        "admission_residual_bf16_deficit_total": sum(
            int(event.get("landing_residual_bf16_deficit", 0))
            for event in admission_events
            if isinstance(
                event.get("landing_residual_bf16_deficit", 0),
                (int, float),
            )
        ),
        "landing_contract_event_count": len(landing_contract_events),
        "landing_contract_persisted_pages_total": (
            landing_contract_persisted_pages_total
        ),
        "landing_contract_direct_pages_total": sum(
            int(event.get("pages", 0))
            for event in landing_contract_events
            if event.get("direct") is True
            and isinstance(event.get("pages", 0), (int, float))
        ),
        "landing_materialize_event_count": len(landing_materialize_events),
        "landing_materialized_pages_total": landing_materialized_pages_total,
        "landing_materialize_layer_copies_total": sum(
            int(event.get("layer_copies", 0))
            for event in landing_materialize_events
            if isinstance(event.get("layer_copies", 0), (int, float))
        ),
        "landing_materialize_gpu_ms_total": sum(
            float(event.get("gpu_ms", 0.0))
            for event in landing_materialize_events
            if isinstance(event.get("gpu_ms", 0.0), (int, float))
        ),
        "landing_commit_event_count": len(landing_commit_events),
        "landing_committed_pages_total": sum(
            int(event.get("committed", event.get("pages", 0)))
            for event in landing_commit_events
            if isinstance(
                event.get("committed", event.get("pages", 0)),
                (int, float),
            )
        ),
        "landing_policy_event_count": len(landing_policy_events),
        "landing_fallback_event_count": len(landing_fallback_events),
        "landing_fallback_pages_total": landing_fallback_pages_total,
        "landing_fallback_unmaterialized_total": sum(
            1
            for event in landing_fallback_events
            if event.get("outcome") == "fallback_unmaterialized"
        ),
        "landing_fallback_unmaterialized_ratio": _ratio(
            landing_fallback_pages_total,
            landing_materialized_pages_total,
        ),
        "page_metadata_produce_event_count": len(page_metadata_produce_events),
        "page_metadata_produced_requests_total": sum(
            int(event.get("requests", 0))
            for event in page_metadata_produce_events
            if isinstance(event.get("requests", 0), (int, float))
        ),
        "page_metadata_produced_pages_total": sum(
            int(event.get("pages", 0))
            for event in page_metadata_produce_events
            if isinstance(event.get("pages", 0), (int, float))
        ),
        "page_metadata_receive_event_count": len(page_metadata_receive_events),
        "page_metadata_received_requests_total": sum(
            int(event.get("requests", 0))
            for event in page_metadata_receive_events
            if isinstance(event.get("requests", 0), (int, float))
        ),
        "page_metadata_received_pages_total": page_metadata_received_pages_total,
        "page_metadata_plan_event_count": len(page_metadata_plan_events),
        "page_metadata_plan_real_risk_requests_total": sum(
            int(event.get("real_risk_requests", 0))
            for event in page_metadata_plan_events
            if isinstance(event.get("real_risk_requests", 0), (int, float))
        ),
        "page_metadata_plan_real_risk_pages_total": (
            page_metadata_plan_real_risk_pages_total
        ),
        "page_metadata_plan_compressible_requests_total": sum(
            int(event.get("compressible_requests", 0))
            for event in page_metadata_plan_events
            if isinstance(event.get("compressible_requests", 0), (int, float))
        ),
        "page_metadata_plan_compressible_pages_total": sum(
            int(event.get("compressible_pages", 0))
            for event in page_metadata_plan_events
            if isinstance(event.get("compressible_pages", 0), (int, float))
        ),
        "page_metadata_plan_shadow_requests_total": sum(
            int(event.get("shadow_requests", 0))
            for event in page_metadata_plan_events
            if isinstance(event.get("shadow_requests", 0), (int, float))
        ),
        "page_metadata_plan_shadow_pages_total": sum(
            int(event.get("shadow_pages", 0))
            for event in page_metadata_plan_events
            if isinstance(event.get("shadow_pages", 0), (int, float))
        ),
        "page_metadata_plan_synthetic_requests_total": sum(
            int(event.get("synthetic_requests", 0))
            for event in page_metadata_plan_events
            if isinstance(event.get("synthetic_requests", 0), (int, float))
        ),
        "page_metadata_plan_synthetic_pages_total": (
            page_metadata_plan_synthetic_pages_total
        ),
        "page_metadata_real_risk_coverage_ratio": _ratio(
            page_metadata_plan_real_risk_pages_total,
            max(
                page_metadata_received_pages_total,
                page_metadata_plan_real_risk_pages_total
                + page_metadata_plan_synthetic_pages_total,
            ),
        ),
        "landing_metadata_source_counts": landing_metadata_source_counts,
        "landing_real_risk_pages_total": sum(
            int(event.get("landing_real_risk_pages", 0))
            for event in admission_events
            if isinstance(event.get("landing_real_risk_pages", 0), (int, float))
        ),
        "landing_explicit_compressible_pages_total": sum(
            int(event.get("landing_explicit_compressible_pages", 0))
            for event in admission_events
            if isinstance(
                event.get("landing_explicit_compressible_pages", 0),
                (int, float),
            )
        ),
        "landing_synthetic_pages_total": sum(
            int(event.get("landing_synthetic_pages", 0))
            for event in admission_events
            if isinstance(event.get("landing_synthetic_pages", 0), (int, float))
        ),
        "recovery_plan_event_count": len(recovery_plan_events),
        "background_promoted_pages_total": sum(
            int(event.get("promoted_pages", 0))
            for event in recovery_plan_events
            if event.get("reason") == "background_promotion"
            and isinstance(event.get("promoted_pages", 0), (int, float))
        ),
        "recovery_exec_event_count": len(recovery_exec_events),
        "recovery_exec_pages_total": sum(
            int(event.get("pages", 0))
            for event in recovery_exec_events
            if isinstance(event.get("pages", 0), (int, float))
        ),
        "recovery_exec_layer_copies_total": sum(
            int(event.get("layer_copies", 0))
            for event in recovery_exec_events
            if isinstance(event.get("layer_copies", 0), (int, float))
        ),
        "recovery_exec_cpu_ms_total": sum(
            float(event.get("cpu_ms", 0.0))
            for event in recovery_exec_events
            if isinstance(event.get("cpu_ms", 0.0), (int, float))
        ),
        "candidate_breakdown_event_count": len(candidate_breakdown_events),
        "candidate_raw_bf16_pages_total": sum(
            int(event.get("raw_bf16_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("raw_bf16_pages", 0), (int, float))
        ),
        "candidate_open_bf16_pages_total": sum(
            int(event.get("open_bf16_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("open_bf16_pages", 0), (int, float))
        ),
        "candidate_remote_inflight_bf16_pages_total": sum(
            int(event.get("remote_inflight_bf16_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(
                event.get("remote_inflight_bf16_pages", 0),
                (int, float),
            )
        ),
        "candidate_open_tail_bf16_pages_total": sum(
            int(event.get("open_tail_bf16_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("open_tail_bf16_pages", 0), (int, float))
        ),
        "candidate_request_protected_bf16_pages_total": sum(
            int(event.get("request_protected_bf16_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(
                event.get("request_protected_bf16_pages", 0),
                (int, float),
            )
        ),
        "candidate_shared_bf16_pages_total": sum(
            int(event.get("shared_bf16_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("shared_bf16_pages", 0), (int, float))
        ),
        "candidate_prompt_protected_bf16_pages_total": sum(
            int(event.get("prompt_protected_bf16_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(
                event.get("prompt_protected_bf16_pages", 0),
                (int, float),
            )
        ),
        "candidate_copy_on_demote_pages_total": sum(
            int(event.get("copy_on_demote_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(
                event.get("copy_on_demote_pages", 0),
                (int, float),
            )
        ),
        "candidate_eligible_full_unshared_pages_total": sum(
            int(event.get("eligible_full_unshared_pages", 0))
            for event in candidate_breakdown_events
            if isinstance(
                event.get("eligible_full_unshared_pages", 0),
                (int, float),
            )
        ),
        "candidate_after_initial_recent_protection_total": sum(
            int(event.get("after_initial_recent_protection", 0))
            for event in candidate_breakdown_events
            if isinstance(
                event.get("after_initial_recent_protection", 0),
                (int, float),
            )
        ),
        "candidate_after_low_risk_filter_total": sum(
            int(event.get("after_low_risk_filter", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("after_low_risk_filter", 0), (int, float))
        ),
        "candidate_after_request_budget_cap_total": sum(
            int(event.get("after_request_budget_cap", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("after_request_budget_cap", 0), (int, float))
        ),
        "candidate_after_sparse_window_quota_total": sum(
            int(event.get("after_sparse_window_quota", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("after_sparse_window_quota", 0), (int, float))
        ),
        "candidate_after_frontier_optimizer_total": sum(
            int(
                event.get(
                    "after_frontier_optimizer",
                    event.get("after_sparse_window_quota", 0),
                )
            )
            for event in candidate_breakdown_events
            if isinstance(
                event.get(
                    "after_frontier_optimizer",
                    event.get("after_sparse_window_quota", 0),
                ),
                (int, float),
            )
        ),
        "candidate_after_int4_pool_limit_total": sum(
            int(event.get("after_int4_pool_limit", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("after_int4_pool_limit", 0), (int, float))
        ),
        "candidate_selected_actual_total": sum(
            int(event.get("selected_actual", 0))
            for event in candidate_breakdown_events
            if isinstance(event.get("selected_actual", 0), (int, float))
        ),
        "candidate_frontier_level_totals": candidate_frontier_level_totals,
        "candidate_rejection_reason_totals": candidate_rejection_reason_totals,
        "admission_blocked_reason_counts": admission_blocked_reason_counts,
        "admission_outcome_counts": admission_outcome_counts,
        "admission_frontier_rejection_reason_totals": (
            admission_frontier_rejection_reason_totals
        ),
        "admission_blocked_frontier_rejection_reason_totals": (
            admission_blocked_frontier_rejection_reason_totals
        ),
        "demotion_gpu_ms_total": sum(
            float(event.get("gpu_ms", 0.0))
            for event in demote_events
            if isinstance(event.get("gpu_ms", 0.0), (int, float))
        ),
        "bf16_capacity_blocks": latest_step.get("bf16_capacity_blocks"),
        "int4_capacity_blocks": latest_step.get("int4_capacity_blocks"),
        "bf16_budget_bytes": latest_step.get("bf16_budget_bytes"),
        "int4_budget_bytes": latest_step.get("int4_budget_bytes"),
        "int4_budget_fraction": latest_step.get("int4_budget_fraction"),
        "mean_int4_ratio": _mean(int4_ratios),
        "max_int4_ratio": max(int4_ratios) if int4_ratios else None,
        "mean_forward_gpu_ms": _mean(forward_gpu_ms),
        "max_forward_gpu_ms": max(forward_gpu_ms) if forward_gpu_ms else None,
        "attention_trace_event_count": len(attention_events),
        "attention_gpu_ms_total": sum(
            float(event.get("gpu_ms", 0.0))
            for event in attention_events
            if isinstance(event.get("gpu_ms", 0.0), (int, float))
        ),
        "trace_events": len(events),
    }


def summarize_run(
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not run_dir.exists() or not run_dir.is_dir():
        raise FileNotFoundError(f"missing run dir: {run_dir}")

    config = _read_json(run_dir / "config.json")
    bench = _read_json(run_dir / "bench_result.json")
    timeline, timeline_stats = _read_timeline(run_dir)
    trace_events = _read_trace_events(run_dir)
    trace_stats = _trace_stats(trace_events)

    summary = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "decode_kv_cache_dtype": config.get("decode_kv_cache_dtype"),
        "input_len": config.get("input_len"),
        "output_len": config.get("output_len"),
        "num_prompts": config.get("num_prompts", bench.get("num_prompts")),
        "max_concurrency": config.get("max_concurrency", bench.get("max_concurrency")),
        "num_gpu_blocks_override": config.get("num_gpu_blocks_override"),
        "completed": bench.get("completed"),
        "failed": bench.get("failed"),
        "req/s": bench.get("request_throughput"),
        "total_token_throughput": bench.get("total_token_throughput"),
        "mean_tpot_ms": bench.get("mean_tpot_ms"),
        "p95_tpot_ms": bench.get("p95_tpot_ms"),
        "p99_tpot_ms": bench.get("p99_tpot_ms"),
        "mean_ttft_ms": bench.get("mean_ttft_ms"),
        "p95_ttft_ms": bench.get("p95_ttft_ms"),
        "p99_ttft_ms": bench.get("p99_ttft_ms"),
        "mean_e2el_ms": bench.get("mean_e2el_ms"),
        "p95_e2el_ms": bench.get("p95_e2el_ms"),
        "p99_e2el_ms": bench.get("p99_e2el_ms"),
        **timeline_stats,
        **trace_stats,
    }
    return summary, timeline, trace_events


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    extras = sorted({key for row in rows for key in row} - set(fieldnames))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames + extras)
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".jsonl":
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
        return
    fieldnames = ["run", "line_no", "event"]
    _write_csv(path, rows, fieldnames)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize ReFlexKV 1P1D pressure sweep run directories."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="Summary CSV path.")
    parser.add_argument(
        "--timeline-out",
        type=Path,
        default=None,
        help="Optional sampled timeline CSV path.",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Optional parsed ReFlex trace output. Use .jsonl for JSONL; any other suffix writes CSV.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = []
    timelines = []
    traces = []
    for run_dir in args.run_dirs:
        summary, timeline, trace_events = summarize_run(run_dir)
        summaries.append(summary)
        timelines.extend(timeline)
        traces.extend(trace_events)
    _write_csv(args.out, summaries, SUMMARY_FIELDS)
    if args.timeline_out is not None:
        _write_csv(args.timeline_out, timelines, TIMELINE_FIELDS)
    if args.trace_out is not None:
        _write_trace(args.trace_out, traces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
