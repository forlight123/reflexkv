#!/usr/bin/env python3
"""Run or print a small ReFlexKV mixed-workload ablation matrix."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIXED_RUN_SCRIPT = ROOT / "scripts" / "accuracy" / "run_pd_serving_mixed_accuracy.py"
DEFAULT_MODEL = "/home/ytm/models/Llama-3.1-8B-Instruct"


@dataclass(frozen=True)
class AblationCase:
    name: str
    decode_kv_cache_dtype: str = "reflex_int4"
    page_selection_policy: str = "frontier_dual"
    env: dict[str, str] = field(default_factory=dict)
    disable_prefill_page_metadata: bool = False
    proxy_prefill_max_inflight: int | None = None
    proxy_decode_backpressure_policy: str | None = None
    proxy_decode_backpressure_waiting_policy: str | None = None


CASE_PRESETS: dict[str, AblationCase] = {
    "bf16_baseline": AblationCase(
        name="bf16_baseline",
        decode_kv_cache_dtype="auto",
        page_selection_policy="relevance_sparse",
        env={
            "SEMANTIQ_REFLEX_ENABLE_DIRECT_INT4_LANDING": "0",
            "SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA": "0",
        },
        disable_prefill_page_metadata=True,
        proxy_decode_backpressure_policy="metrics",
        proxy_decode_backpressure_waiting_policy="fixed",
    ),
    "heuristic_reflex": AblationCase(
        name="heuristic_reflex",
        page_selection_policy="relevance_sparse",
        proxy_decode_backpressure_policy="metrics",
        proxy_decode_backpressure_waiting_policy="adaptive",
    ),
    "frontier_dual_reflex": AblationCase(
        name="frontier_dual_reflex",
        page_selection_policy="frontier_dual",
        proxy_decode_backpressure_policy="metrics",
        proxy_decode_backpressure_waiting_policy="adaptive",
    ),
    "direct_landing_off": AblationCase(
        name="direct_landing_off",
        page_selection_policy="frontier_dual",
        env={"SEMANTIQ_REFLEX_ENABLE_DIRECT_INT4_LANDING": "0"},
        proxy_decode_backpressure_policy="metrics",
        proxy_decode_backpressure_waiting_policy="adaptive",
    ),
    "direct_landing_on": AblationCase(
        name="direct_landing_on",
        page_selection_policy="frontier_dual",
        env={"SEMANTIQ_REFLEX_ENABLE_DIRECT_INT4_LANDING": "1"},
        proxy_decode_backpressure_policy="metrics",
        proxy_decode_backpressure_waiting_policy="adaptive",
    ),
    "p_side_risk_off": AblationCase(
        name="p_side_risk_off",
        page_selection_policy="frontier_dual",
        env={"SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA": "0"},
        disable_prefill_page_metadata=True,
        proxy_decode_backpressure_policy="metrics",
        proxy_decode_backpressure_waiting_policy="adaptive",
    ),
}


def _parse_csv(value: str, *, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one case.")
    return items


def selected_cases(args: argparse.Namespace) -> list[AblationCase]:
    cases: list[AblationCase] = []
    for name in _parse_csv(args.cases, name="cases"):
        if name not in CASE_PRESETS:
            raise ValueError(f"unknown ablation case {name!r}")
        cases.append(CASE_PRESETS[name])
    if args.limit is not None:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        cases = cases[: args.limit]
    return cases


def _append(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def _ports(args: argparse.Namespace, index: int) -> dict[str, int]:
    if args.port_stride < 4:
        raise ValueError("--port-stride must be at least 4")
    base = args.base_port + index * args.port_stride
    return {
        "prefill_port": base,
        "decode_port": base + 1,
        "proxy_port": base + 2,
        "prefill_bootstrap_port": base + 3,
    }


def _metadata_wait_timeout(args: argparse.Namespace, case: AblationCase) -> float:
    configured = getattr(args, "proxy_prefill_metadata_wait_timeout_sec", None)
    if configured is not None:
        return max(0.0, float(configured))
    if (
        case.decode_kv_cache_dtype == "reflex_int4"
        and not case.disable_prefill_page_metadata
        and case.env.get("SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA") != "0"
    ):
        return 5.0
    return 0.0


def build_command(
    args: argparse.Namespace,
    case: AblationCase,
    *,
    index: int,
) -> list[str]:
    ports = _ports(args, index)
    command = [
        sys.executable,
        str(MIXED_RUN_SCRIPT),
        "--model",
        args.model,
        "--host",
        args.host,
        "--prefill-gpu",
        args.prefill_gpu,
        "--decode-gpu",
        args.decode_gpu,
        "--prefill-port",
        str(ports["prefill_port"]),
        "--decode-port",
        str(ports["decode_port"]),
        "--proxy-port",
        str(ports["proxy_port"]),
        "--prefill-bootstrap-port",
        str(ports["prefill_bootstrap_port"]),
        "--run-name",
        f"ablation_{index:02d}_{case.name}",
        "--output-root",
        args.output_root,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--block-size",
        str(args.block_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--prefill-kv-cache-dtype",
        "auto",
        "--decode-kv-cache-dtype",
        case.decode_kv_cache_dtype,
        "--proxy-prefill-max-inflight",
        str(
            case.proxy_prefill_max_inflight
            if case.proxy_prefill_max_inflight is not None
            else args.proxy_prefill_max_inflight
        ),
        "--proxy-prefill-metadata-wait-timeout-sec",
        str(_metadata_wait_timeout(args, case)),
        "--proxy-decode-backpressure-policy",
        str(
            case.proxy_decode_backpressure_policy
            if case.proxy_decode_backpressure_policy is not None
            else args.proxy_decode_backpressure_policy
        ),
        "--proxy-decode-backpressure-max-kv-usage",
        str(args.proxy_decode_backpressure_max_kv_usage),
        "--proxy-decode-backpressure-max-waiting",
        str(args.proxy_decode_backpressure_max_waiting),
        "--proxy-decode-backpressure-waiting-policy",
        str(
            case.proxy_decode_backpressure_waiting_policy
            if case.proxy_decode_backpressure_waiting_policy is not None
            else args.proxy_decode_backpressure_waiting_policy
        ),
        "--proxy-decode-backpressure-adaptive-max-waiting",
        str(args.proxy_decode_backpressure_adaptive_max_waiting),
        "--proxy-decode-backpressure-adaptive-kv-headroom-per-waiting",
        str(args.proxy_decode_backpressure_adaptive_kv_headroom_per_waiting),
        "--proxy-decode-backpressure-poll-interval-sec",
        str(args.proxy_decode_backpressure_poll_interval_sec),
        "--proxy-decode-backpressure-timeout-sec",
        str(args.proxy_decode_backpressure_timeout_sec),
        "--proxy-decode-backpressure-admission-settle-sec",
        str(args.proxy_decode_backpressure_admission_settle_sec),
        "--reflex-remote-chunk-tokens",
        str(args.reflex_remote_chunk_tokens),
        "--reflex-page-selection-policy",
        case.page_selection_policy,
        "--tasks",
        args.tasks,
        "--longbench-datasets",
        args.longbench_datasets,
        "--reasoning-datasets",
        args.reasoning_datasets,
        "--longbench-max-samples",
        str(args.longbench_max_samples),
        "--reasoning-max-samples",
        str(args.reasoning_max_samples),
        "--workload-mix-policy",
        args.workload_mix_policy,
        "--max-concurrency",
        str(args.max_concurrency),
        "--request-rate",
        str(args.request_rate),
    ]
    _append(command, "--workload-manifest", args.workload_manifest)
    if args.workload_manifest is not None or args.arrival_policy != "poisson":
        command.extend(["--arrival-policy", args.arrival_policy])
    if args.workload_manifest is not None or float(args.trace_time_scale) != 1.0:
        command.extend(["--trace-time-scale", str(args.trace_time_scale)])
    _append(command, "--num-gpu-blocks-override", args.num_gpu_blocks_override)
    if args.enable_reflex_trace:
        command.append("--enable-reflex-trace")
    if args.force_triton_attn:
        command.append("--force-triton-attn")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.skip_chat_template:
        command.append("--skip-chat-template")
    if case.disable_prefill_page_metadata:
        command.append("--disable-reflex-prefill-page-metadata")
    return command


def _record(
    args: argparse.Namespace,
    case: AblationCase,
    *,
    index: int,
) -> dict[str, object]:
    return {
        "case": asdict(case),
        "env": dict(case.env),
        "command": build_command(args, case, index=index),
    }


def _write_commands(path: str | None, records: list[dict[str, object]]) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def run_matrix(args: argparse.Namespace) -> int:
    records = [
        _record(args, case, index=index)
        for index, case in enumerate(selected_cases(args))
    ]
    _write_commands(args.commands_out, records)
    if args.dry_run:
        for record in records:
            env = record["env"]
            prefix = " ".join(
                f"{key}={shlex.quote(str(value))}"
                for key, value in sorted(env.items())  # type: ignore[union-attr]
            )
            command = shlex.join(record["command"])  # type: ignore[arg-type]
            print(f"{prefix} {command}".strip())
        return 0

    for record in records:
        env = os.environ.copy()
        env.update(record["env"])  # type: ignore[arg-type]
        proc = subprocess.run(
            record["command"],  # type: ignore[arg-type]
            cwd=ROOT,
            env=env,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return int(proc.returncode)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small ReFlexKV mixed-workload ablation matrix."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--base-port", type=int, default=8710)
    parser.add_argument("--port-stride", type=int, default=20)
    parser.add_argument("--output-root", default="outputs/accuracy/reflex_ablation")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--proxy-prefill-max-inflight", type=int, default=0)
    parser.add_argument(
        "--proxy-prefill-metadata-wait-timeout-sec",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-policy",
        choices=["off", "metrics"],
        default="metrics",
    )
    parser.add_argument(
        "--proxy-decode-backpressure-max-kv-usage",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-max-waiting",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-waiting-policy",
        choices=["fixed", "adaptive"],
        default="fixed",
    )
    parser.add_argument(
        "--proxy-decode-backpressure-adaptive-max-waiting",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-adaptive-kv-headroom-per-waiting",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-poll-interval-sec",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-timeout-sec",
        type=float,
        default=300.0,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-admission-settle-sec",
        type=float,
        default=1.0,
    )
    parser.add_argument("--reflex-remote-chunk-tokens", type=int, default=512)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--workload-manifest", default=None)
    parser.add_argument(
        "--arrival-policy",
        choices=["poisson", "trace"],
        default="poisson",
    )
    parser.add_argument("--trace-time-scale", type=float, default=1.0)
    parser.add_argument("--longbench-max-samples", type=int, default=20)
    parser.add_argument("--reasoning-max-samples", type=int, default=20)
    parser.add_argument("--longbench-datasets", default="gov_report")
    parser.add_argument("--reasoning-datasets", default="math500")
    parser.add_argument("--tasks", default="longbench,reasoning")
    parser.add_argument("--workload-mix-policy", default="balanced")
    parser.add_argument("--enable-reflex-trace", action="store_true", default=True)
    parser.add_argument("--force-triton-attn", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-chat-template", action="store_true")
    parser.add_argument(
        "--cases",
        default=(
            "bf16_baseline,heuristic_reflex,frontier_dual_reflex,"
            "direct_landing_off,direct_landing_on,p_side_risk_off"
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commands-out", default=None)
    dry = parser.add_mutually_exclusive_group()
    dry.add_argument("--dry-run", dest="dry_run", action="store_true")
    dry.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run_matrix(parse_args(argv))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
