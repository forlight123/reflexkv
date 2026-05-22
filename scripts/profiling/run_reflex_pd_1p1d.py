#!/usr/bin/env python3
"""Run a 1P1D Mooncake disaggregated-prefill experiment for ReFlexKV.

The prefill instance keeps the normal BF16/auto KV path, while the decode
instance can switch between full BF16/auto and ``reflex_int4``. ReFlexKV uses a
Mooncake-based connector as the main handoff path; the older P2P/NCCL connector
is intentionally not used here because it stages large tensors through host
memory on this machine.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from enum import Enum
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/ytm/models/Llama-3.1-8B-Instruct"
PROXY_SCRIPT = (
    ROOT
    / "vllm"
    / "examples"
    / "online_serving"
    / "disaggregated_serving"
    / "mooncake_connector"
    / "mooncake_connector_proxy.py"
)
METRIC_KEEP_RE = re.compile(
    r"(kv|cache|running|waiting|request|queue|scheduler|token|latency|tpot|ttft)",
    re.IGNORECASE,
)


class Role(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--prefill-port", type=int, default=8310)
    parser.add_argument("--decode-port", type=int, default=8410)
    parser.add_argument("--proxy-port", type=int, default=8510)
    parser.add_argument("--prefill-bootstrap-port", type=int, default=8998)
    parser.add_argument(
        "--proxy-prefill-max-inflight",
        type=int,
        default=2,
        help=(
            "Maximum end-to-end requests the proxy lets hold P-side prefill "
            "KV concurrently. Use 0 to disable proxy-side prefill backpressure."
        ),
    )
    parser.add_argument(
        "--proxy-prefill-metadata-wait-timeout-sec",
        type=float,
        default=None,
        help=(
            "Seconds the proxy may wait for prefill-returned ReFlexKV metadata "
            "before starting decode. Defaults to 5s for reflex_int4 with "
            "P-side metadata enabled, otherwise 0s."
        ),
    )
    parser.add_argument(
        "--reflex-remote-chunk-tokens",
        type=int,
        default=512,
        help="Remote KV chunk size used by the ReFlexKV P/D connector.",
    )
    parser.add_argument("--mooncake-protocol", default="rdma")
    parser.add_argument("--mooncake-num-workers", type=int, default=10)
    parser.add_argument("--reflex-keep-recent-blocks", type=int, default=16)
    parser.add_argument(
        "--reflex-keep-initial-blocks",
        type=int,
        default=4,
        help="Initial logical KV blocks protected from ReFlexKV demotion.",
    )
    parser.add_argument(
        "--reflex-max-int4-fraction-per-request",
        type=float,
        default=None,
        help=(
            "Maximum fraction of a request's logical KV blocks that ReFlexKV "
            "may demote to INT4. Defaults to scheduler internal default."
        ),
    )
    parser.add_argument(
        "--reflex-survival-warmup-tokens",
        type=int,
        default=128,
        help=(
            "Generated decode tokens required before a request may receive "
            "background ReFlexKV demotion budget. Admission pressure can still "
            "assign a scaled budget before this warmup."
        ),
    )
    parser.add_argument(
        "--reflex-risk-warmup-tokens",
        type=int,
        default=16,
        help="Generated decode tokens required before any admission demotion.",
    )
    parser.add_argument(
        "--reflex-short-admission-max-int4-fraction",
        type=float,
        default=0.03,
        help="Admission-only INT4 cap for short-output requests after risk warmup.",
    )
    parser.add_argument(
        "--reflex-sparse-window-pages",
        type=int,
        default=32,
        help="Page window size used by sparse low-risk demotion selection.",
    )
    parser.add_argument(
        "--reflex-short-max-demote-per-window",
        type=int,
        default=1,
        help="Max low-risk pages demoted per window for short-output requests.",
    )
    parser.add_argument(
        "--reflex-max-demote-per-window",
        type=int,
        default=2,
        help="Default max low-risk pages demoted per window.",
    )
    parser.add_argument(
        "--reflex-low-risk-score-fraction",
        type=float,
        default=0.25,
        help="Fraction of lowest-risk page scores treated as compressible.",
    )
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
        help="ReFlexKV page selection ablation policy.",
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
    parser.add_argument(
        "--reflex-decode-pressure-warmup-tokens",
        type=int,
        default=32,
        help="Generated decode tokens below which demotion pressure stays minimal.",
    )
    parser.add_argument(
        "--reflex-decode-pressure-ramp-tokens",
        type=int,
        default=512,
        help="Generated decode tokens at which demotion pressure reaches its high value.",
    )
    parser.add_argument(
        "--reflex-short-prefill-pages",
        type=int,
        default=64,
        help="Prompt page count treated as short for ReFlexKV demotion pressure.",
    )
    parser.add_argument(
        "--reflex-long-prefill-pages",
        type=int,
        default=512,
        help="Prompt page count treated as long for ReFlexKV demotion pressure.",
    )
    parser.add_argument(
        "--reflex-global-evidence-min-prompt-pages",
        type=int,
        default=512,
        help="Prompt page threshold for global-evidence landing protection.",
    )
    parser.add_argument(
        "--reflex-global-evidence-min-decode-tokens",
        type=int,
        default=129,
        help="Remaining decode threshold for global-evidence landing protection.",
    )
    parser.add_argument(
        "--reflex-global-evidence-landing-max-int4-fraction",
        type=float,
        default=0.08,
        help="Max INT4 landing fraction for long-prompt, long-output requests.",
    )
    parser.add_argument(
        "--reflex-reasoning-prompt-protection-max-pages",
        type=int,
        default=64,
        help="Pin prompt pages for short-prompt, long-decode reasoning-like requests.",
    )
    parser.add_argument(
        "--reflex-reasoning-prompt-protection-min-decode-tokens",
        type=int,
        default=1024,
        help="Decode budget threshold for reasoning-like prompt page protection.",
    )
    parser.add_argument(
        "--reflex-slo-pressure-step",
        type=float,
        default=0.25,
        help="Per-priority-step multiplier delta for SLO-aware ReFlexKV demotion.",
    )
    parser.add_argument(
        "--reflex-min-slo-pressure",
        type=float,
        default=0.5,
        help="Lower clamp for priority-derived ReFlexKV demotion pressure.",
    )
    parser.add_argument(
        "--reflex-max-slo-pressure",
        type=float,
        default=1.5,
        help="Upper clamp for priority-derived ReFlexKV demotion pressure.",
    )
    parser.add_argument(
        "--scheduling-policy",
        choices=["fcfs", "priority"],
        default="fcfs",
        help="vLLM scheduler policy for serving requests.",
    )
    # Deprecated P2pNcclConnector knobs. They are accepted for old command
    # lines but are not used by the Mooncake/ReFlexMooncake path.
    parser.add_argument("--kv-proxy-port", type=int, default=30011, help=argparse.SUPPRESS)
    parser.add_argument("--prefill-kv-port", type=int, default=14579, help=argparse.SUPPRESS)
    parser.add_argument("--decode-kv-port", type=int, default=14580, help=argparse.SUPPRESS)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="outputs/profiling/reflex_pd_1p1d")

    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--prefill-kv-cache-dtype", default="auto")
    parser.add_argument("--decode-kv-cache-dtype", default="reflex_int4")
    parser.add_argument(
        "--reflex-int4-budget-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of the decode KV byte budget reserved for ReFlexKV INT4 "
            "blocks. Defaults to vLLM/ReFlexKV's internal default."
        ),
    )
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--force-triton-attn", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-reflex-trace", action="store_true")
    parser.add_argument(
        "--disable-reflex-prefill-page-metadata",
        action="store_true",
        help=(
            "Disable P-side ReFlexKV page-risk metadata even when the decode "
            "instance uses reflex_int4. This is used by risk-estimator ablations."
        ),
    )
    parser.add_argument("--p2p-kv-chunk-blocks", type=int, default=32, help=argparse.SUPPRESS)
    parser.add_argument(
        "--p2p-max-staged-bytes",
        type=int,
        default=64 * 1024 * 1024,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--extra-serve-args", nargs=argparse.REMAINDER, default=[])

    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--input-len", type=int, default=16000)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--num-prompts", type=int, default=2)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int, default=2)
    parser.add_argument("--temperature", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-chat-template", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--server-ready-timeout-sec", type=float, default=420.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.prefill_gpu == args.decode_gpu:
        raise ValueError("1P1D requires different GPUs for prefill and decode.")
    ports = [
        args.prefill_port,
        args.decode_port,
        args.proxy_port,
        args.prefill_bootstrap_port,
    ]
    if len(set(ports)) != len(ports):
        raise ValueError(f"1P1D ports must be distinct, got {ports}.")


def make_run_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.run_name or (
        f"pd1p1d_{args.dataset_name}_i{args.input_len}_o{args.output_len}_"
        f"np{args.num_prompts}_c{args.max_concurrency}_"
        f"dtype{args.decode_kv_cache_dtype}"
    )
    run_dir = ROOT / args.output_root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_base_env() -> dict[str, str]:
    env = os.environ.copy()
    py_path = f"{ROOT}:{ROOT / 'vllm'}"
    env["PYTHONPATH"] = f"{py_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else py_path
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    no_proxy = env.get("NO_PROXY") or env.get("no_proxy") or ""
    localhost_entries = ["127.0.0.1", "localhost", "::1"]
    merged_no_proxy = [
        entry.strip() for entry in no_proxy.split(",") if entry.strip()
    ]
    for entry in localhost_entries:
        if entry not in merged_no_proxy:
            merged_no_proxy.append(entry)
    env["NO_PROXY"] = ",".join(merged_no_proxy)
    env["no_proxy"] = env["NO_PROXY"]
    return env


def build_server_env(args: argparse.Namespace, role: Role) -> dict[str, str]:
    env = build_base_env()
    env["CUDA_VISIBLE_DEVICES"] = (
        args.prefill_gpu if role is Role.PREFILL else args.decode_gpu
    )
    env["VLLM_HOST_IP"] = args.host
    if role is Role.PREFILL:
        env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] = str(args.prefill_bootstrap_port)
        if (
            getattr(args, "decode_kv_cache_dtype", None) == "reflex_int4"
            and not getattr(args, "disable_reflex_prefill_page_metadata", False)
        ):
            env["SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA"] = "1"
        elif getattr(args, "disable_reflex_prefill_page_metadata", False):
            env["SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA"] = "0"
    if role is Role.DECODE and args.enable_reflex_trace:
        env["SEMANTIQ_REFLEX_TRACE"] = "1"
    reflex_int4_budget_fraction = getattr(args, "reflex_int4_budget_fraction", None)
    if role is Role.DECODE and reflex_int4_budget_fraction is not None:
        env["SEMANTIQ_REFLEX_INT4_BUDGET_FRACTION"] = str(
            reflex_int4_budget_fraction
        )
    if role is Role.DECODE:
        env["SEMANTIQ_REFLEX_KEEP_RECENT_PAGES"] = str(
            args.reflex_keep_recent_blocks
        )
        env["SEMANTIQ_REFLEX_KEEP_INITIAL_PAGES"] = str(
            args.reflex_keep_initial_blocks
        )
        max_int4_fraction = getattr(
            args,
            "reflex_max_int4_fraction_per_request",
            None,
        )
        if max_int4_fraction is not None:
            env["SEMANTIQ_REFLEX_MAX_INT4_FRACTION_PER_REQUEST"] = str(
                max_int4_fraction
            )
        env["SEMANTIQ_REFLEX_SURVIVAL_WARMUP_TOKENS"] = str(
            args.reflex_survival_warmup_tokens
        )
        env["SEMANTIQ_REFLEX_RISK_WARMUP_TOKENS"] = str(
            getattr(args, "reflex_risk_warmup_tokens", 16)
        )
        env["SEMANTIQ_REFLEX_SHORT_ADMISSION_MAX_INT4_FRACTION"] = str(
            getattr(args, "reflex_short_admission_max_int4_fraction", 0.03)
        )
        env["SEMANTIQ_REFLEX_COLD_ADMISSION_MAX_INT4_FRACTION"] = str(
            getattr(args, "reflex_cold_admission_max_int4_fraction", 0.10)
        )
        env["SEMANTIQ_REFLEX_COLD_ADMISSION_EMERGENCY_FREE_RATIO"] = str(
            getattr(args, "reflex_cold_admission_emergency_free_ratio", 0.05)
        )
        env["SEMANTIQ_REFLEX_SPARSE_WINDOW_PAGES"] = str(
            getattr(args, "reflex_sparse_window_pages", 32)
        )
        env["SEMANTIQ_REFLEX_SHORT_MAX_DEMOTE_PER_WINDOW"] = str(
            getattr(args, "reflex_short_max_demote_per_window", 1)
        )
        env["SEMANTIQ_REFLEX_MAX_DEMOTE_PER_WINDOW"] = str(
            getattr(args, "reflex_max_demote_per_window", 2)
        )
        env["SEMANTIQ_REFLEX_LOW_RISK_SCORE_FRACTION"] = str(
            getattr(args, "reflex_low_risk_score_fraction", 0.25)
        )
        env["SEMANTIQ_REFLEX_PAGE_SELECTION_POLICY"] = str(
            getattr(args, "reflex_page_selection_policy", "relevance_sparse")
        )
        env["SEMANTIQ_REFLEX_SLO_PRESSURE_STEP"] = str(
            getattr(args, "reflex_slo_pressure_step", 0.25)
        )
        env["SEMANTIQ_REFLEX_MIN_SLO_PRESSURE"] = str(
            getattr(args, "reflex_min_slo_pressure", 0.5)
        )
        env["SEMANTIQ_REFLEX_MAX_SLO_PRESSURE"] = str(
            getattr(args, "reflex_max_slo_pressure", 1.5)
        )
        env["SEMANTIQ_REFLEX_DECODE_PRESSURE_WARMUP_TOKENS"] = str(
            getattr(args, "reflex_decode_pressure_warmup_tokens", 32)
        )
        env["SEMANTIQ_REFLEX_DECODE_PRESSURE_RAMP_TOKENS"] = str(
            getattr(args, "reflex_decode_pressure_ramp_tokens", 512)
        )
        env["SEMANTIQ_REFLEX_SHORT_PREFILL_PAGES"] = str(
            getattr(args, "reflex_short_prefill_pages", 64)
        )
        env["SEMANTIQ_REFLEX_LONG_PREFILL_PAGES"] = str(
            getattr(args, "reflex_long_prefill_pages", 512)
        )
        env["SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_MIN_PROMPT_PAGES"] = str(
            getattr(args, "reflex_global_evidence_min_prompt_pages", 512)
        )
        env["SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_MIN_DECODE_TOKENS"] = str(
            getattr(args, "reflex_global_evidence_min_decode_tokens", 129)
        )
        env["SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_LANDING_MAX_INT4_FRACTION"] = str(
            getattr(
                args,
                "reflex_global_evidence_landing_max_int4_fraction",
                0.08,
            )
        )
        env["SEMANTIQ_REFLEX_REASONING_PROMPT_PROTECTION_MAX_PAGES"] = str(
            getattr(args, "reflex_reasoning_prompt_protection_max_pages", 64)
        )
        env[
            "SEMANTIQ_REFLEX_REASONING_PROMPT_PROTECTION_MIN_DECODE_TOKENS"
        ] = str(
            getattr(
                args,
                "reflex_reasoning_prompt_protection_min_decode_tokens",
                1024,
            )
        )
    return env


def build_kv_transfer_config(args: argparse.Namespace, role: Role) -> str:
    is_prefill = role is Role.PREFILL
    config = {
        "kv_connector": "ReFlexMooncakeConnector",
        "kv_role": "kv_producer" if is_prefill else "kv_consumer",
        "kv_connector_extra_config": {
            "mooncake_protocol": args.mooncake_protocol,
            "num_workers": args.mooncake_num_workers,
            "reflex_keep_recent_blocks": args.reflex_keep_recent_blocks,
        },
    }
    return json.dumps(config, separators=(",", ":"))


def build_server_cmd(args: argparse.Namespace, role: Role) -> list[str]:
    is_decode = role is Role.DECODE
    cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.decode_port if is_decode else args.prefill_port),
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        str(args.max_model_len),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-dtype",
        args.decode_kv_cache_dtype if is_decode else args.prefill_kv_cache_dtype,
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--trust-remote-code",
        "--disable-uvicorn-access-log",
        "--kv-transfer-config",
        build_kv_transfer_config(args, role),
    ]
    if is_decode and args.num_gpu_blocks_override is not None:
        cmd.extend(["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)])
    if args.force_triton_attn or args.decode_kv_cache_dtype in {"int4", "reflex_int4"}:
        cmd.extend(["--attention-backend", "TRITON_ATTN"])
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    scheduling_policy = getattr(args, "scheduling_policy", None)
    if scheduling_policy:
        cmd.extend(["--scheduling-policy", str(scheduling_policy)])
    cmd.extend(args.extra_serve_args)
    return cmd


def build_proxy_cmd(args: argparse.Namespace) -> list[str]:
    metadata_wait_timeout = getattr(
        args,
        "proxy_prefill_metadata_wait_timeout_sec",
        None,
    )
    if metadata_wait_timeout is None:
        metadata_wait_timeout = (
            5.0
            if getattr(args, "decode_kv_cache_dtype", None) == "reflex_int4"
            and not getattr(args, "disable_reflex_prefill_page_metadata", False)
            else 0.0
        )
    metadata_wait_timeout = max(0.0, float(metadata_wait_timeout))
    return [
        sys.executable,
        str(PROXY_SCRIPT),
        "--port",
        str(args.proxy_port),
        "--prefill",
        f"http://{args.host}:{args.prefill_port}",
        str(args.prefill_bootstrap_port),
        "--decode",
        f"http://{args.host}:{args.decode_port}",
        "--prefill-max-inflight",
        str(max(0, int(getattr(args, "proxy_prefill_max_inflight", 0) or 0))),
        "--prefill-metadata-wait-timeout-sec",
        str(metadata_wait_timeout),
        "--reflex-remote-chunk-tokens",
        str(max(1, int(getattr(args, "reflex_remote_chunk_tokens", 512) or 512))),
    ]


def build_bench_cmd(args: argparse.Namespace, run_dir: Path) -> list[str]:
    bench_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        f"http://{args.host}:{args.proxy_port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--dataset-name",
        args.dataset_name,
        "--num-prompts",
        str(args.num_prompts),
        "--request-rate",
        str(args.request_rate),
        "--max-concurrency",
        str(args.max_concurrency),
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,95,99",
        "--ignore-eos",
        "--temperature",
        str(args.temperature),
        "--seed",
        str(args.seed),
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(run_dir),
        "--result-filename",
        "bench_result.json",
    ]
    if args.skip_chat_template:
        bench_cmd.append("--skip-chat-template")
    if args.no_stream:
        bench_cmd.append("--no-stream")
    if args.dataset_name == "random":
        bench_cmd.extend(
            [
                "--random-input-len",
                str(args.input_len),
                "--random-output-len",
                str(args.output_len),
                "--random-range-ratio",
                "0",
            ]
        )
    elif args.dataset_path:
        bench_cmd.extend(["--dataset-path", args.dataset_path])
        if args.output_len > 0 and args.dataset_name == "sharegpt":
            bench_cmd.extend(["--sharegpt-output-len", str(args.output_len)])
        elif args.output_len > 0 and args.dataset_name == "custom":
            bench_cmd.extend(["--custom-output-len", str(args.output_len)])
    return bench_cmd


def wait_for_http_ok(url: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"endpoint did not become ready at {url}: {last_error}")


def wait_for_tcp_port(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                return
        except OSError as exc:
            last_error = repr(exc)
        time.sleep(1)
    raise TimeoutError(f"port did not become ready at {host}:{port}: {last_error}")


def fetch_text(url: str, timeout: float = 1.5) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#") or not METRIC_KEEP_RE.search(line):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def sample_nvidia_smi() -> list[dict[str, str]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 4:
            rows.append(
                {
                    "gpu": parts[0],
                    "memory_used_mib": parts[1],
                    "memory_free_mib": parts[2],
                    "utilization_gpu_pct": parts[3],
                }
            )
    return rows


def sampler(
    base_urls: dict[str, str],
    interval_sec: float,
    out_path: Path,
    stop_event: threading.Event,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        while not stop_event.is_set():
            sample: dict[str, object] = {
                "time": time.time(),
                "nvidia_smi": sample_nvidia_smi(),
                "vllm_metrics": {},
            }
            metrics_by_role = {}
            for role, base_url in base_urls.items():
                try:
                    metrics_by_role[role] = parse_prometheus_metrics(
                        fetch_text(f"{base_url}/metrics")
                    )
                except Exception as exc:  # noqa: BLE001 - sampler must not kill run.
                    metrics_by_role[role] = {"metrics_error": repr(exc)}
            sample["vllm_metrics"] = metrics_by_role
            f.write(json.dumps(sample, sort_keys=True) + "\n")
            f.flush()
            stop_event.wait(interval_sec)


def terminate_process_group(proc: subprocess.Popen[str], timeout: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGINT)
        proc.wait(timeout=timeout)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=timeout)


def launch_process(
    cmd: list[str],
    *,
    env: dict[str, str],
    log_path: Path,
) -> subprocess.Popen[str]:
    log_file = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    proc._semantiq_log_file = log_file  # type: ignore[attr-defined]
    return proc


def close_process_log(proc: subprocess.Popen[str] | None) -> None:
    if proc is None:
        return
    log_file = getattr(proc, "_semantiq_log_file", None)
    if log_file is not None:
        log_file.close()


def main() -> int:
    args = parse_args()
    validate_args(args)
    run_dir = make_run_dir(args)
    prefill_url = f"http://{args.host}:{args.prefill_port}"
    decode_url = f"http://{args.host}:{args.decode_port}"
    proxy_url = f"http://{args.host}:{args.proxy_port}"

    prefill_cmd = build_server_cmd(args, Role.PREFILL)
    decode_cmd = build_server_cmd(args, Role.DECODE)
    proxy_cmd = build_proxy_cmd(args)
    bench_cmd = build_bench_cmd(args, run_dir)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args), indent=2),
        encoding="utf-8",
    )
    (run_dir / "prefill_server_cmd.txt").write_text(
        " ".join(prefill_cmd) + "\n", encoding="utf-8"
    )
    (run_dir / "decode_server_cmd.txt").write_text(
        " ".join(decode_cmd) + "\n", encoding="utf-8"
    )
    (run_dir / "proxy_cmd.txt").write_text(
        " ".join(proxy_cmd) + "\n", encoding="utf-8"
    )
    (run_dir / "bench_cmd.txt").write_text(
        " ".join(bench_cmd) + "\n", encoding="utf-8"
    )

    procs: list[subprocess.Popen[str]] = []
    stop_event = threading.Event()
    sampler_thread: threading.Thread | None = None
    try:
        prefill_proc = launch_process(
            prefill_cmd,
            env=build_server_env(args, Role.PREFILL),
            log_path=run_dir / "prefill_server.log",
        )
        procs.append(prefill_proc)
        decode_proc = launch_process(
            decode_cmd,
            env=build_server_env(args, Role.DECODE),
            log_path=run_dir / "decode_server.log",
        )
        procs.append(decode_proc)

        wait_for_http_ok(f"{prefill_url}/v1/models", args.server_ready_timeout_sec)
        wait_for_http_ok(f"{decode_url}/v1/models", args.server_ready_timeout_sec)

        proxy_proc = launch_process(
            proxy_cmd,
            env=build_base_env(),
            log_path=run_dir / "proxy.log",
        )
        procs.append(proxy_proc)
        wait_for_tcp_port(args.host, args.proxy_port, args.server_ready_timeout_sec)

        sampler_thread = threading.Thread(
            target=sampler,
            args=(
                {"prefill": prefill_url, "decode": decode_url},
                args.sample_interval_sec,
                run_dir / "metrics_samples.jsonl",
                stop_event,
            ),
            daemon=True,
        )
        sampler_thread.start()

        with (run_dir / "bench.log").open("w", encoding="utf-8") as bench_log:
            bench_proc = subprocess.run(
                bench_cmd,
                cwd=ROOT,
                env=build_base_env(),
                stdout=bench_log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        print(f"run_dir={run_dir}")
        print(f"bench_exit_code={bench_proc.returncode}")
        return bench_proc.returncode
    finally:
        stop_event.set()
        if sampler_thread is not None:
            sampler_thread.join(timeout=5)
        for proc in reversed(procs):
            terminate_process_group(proc)
        for proc in procs:
            close_process_log(proc)


if __name__ == "__main__":
    raise SystemExit(main())
