#!/usr/bin/env python3
"""Run a bounded 1P1D accuracy regression matrix.

This script is an orchestration layer around ``run_pd_serving_accuracy.py``.
It keeps PD serving setup in one place while making the core comparison matrix
repeatable after connector, cleanup, or ReFlexKV policy changes.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PD_ACCURACY_SCRIPT = ROOT / "scripts" / "accuracy" / "run_pd_serving_accuracy.py"
PD_MIXED_ACCURACY_SCRIPT = (
    ROOT / "scripts" / "accuracy" / "run_pd_serving_mixed_accuracy.py"
)
DEFAULT_MODEL = "/home/ytm/models/Llama-3.1-8B-Instruct"
DEFAULT_CONFIG_DIR = str(ROOT / "eval" / "config")
DEFAULT_LONGBENCH_DATA_DIR = str(ROOT / "data" / "longbench")
DEFAULT_REASONING_DATA_DIR = str(ROOT / "data" / "reasoning")


@dataclass(frozen=True)
class RegressionPoint:
    index: int
    variant: str
    max_concurrency: int
    longbench_max_samples: int
    reasoning_max_samples: int


def parse_str_csv(value: str, *, name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    return items


def parse_int_csv(value: str, *, name: str) -> list[int]:
    items = parse_str_csv(value, name=name)
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-integer value: {item!r}") from exc
    return parsed


def _sanitize_label(value: str) -> str:
    return (
        value.replace(",", "+")
        .replace("/", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace(" ", "_")
    )


def enumerate_points(args: argparse.Namespace) -> list[RegressionPoint]:
    tasks = parse_str_csv(args.tasks, name="tasks")
    unsupported = set(tasks) - {"longbench", "reasoning"}
    if unsupported:
        raise ValueError(f"Unsupported tasks: {', '.join(sorted(unsupported))}")
    variants = parse_str_csv(args.variants, name="variants")
    concurrencies = parse_int_csv(args.concurrencies, name="concurrencies")
    points: list[RegressionPoint] = []
    for variant in variants:
        for max_concurrency in concurrencies:
            points.append(
                RegressionPoint(
                    index=len(points),
                    variant=variant,
                    max_concurrency=max_concurrency,
                    longbench_max_samples=args.longbench_max_samples,
                    reasoning_max_samples=args.reasoning_max_samples,
                )
            )
    return points


def selected_points(args: argparse.Namespace) -> list[RegressionPoint]:
    points = enumerate_points(args)
    if args.limit is None:
        return points
    if args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    return points[: args.limit]


def ports_for_point(args: argparse.Namespace, point: RegressionPoint) -> dict[str, int]:
    if args.port_stride < 4:
        raise ValueError("--port-stride must be at least 4 to avoid port collisions.")
    base = args.base_port + point.index * args.port_stride
    ports = {
        "prefill_port": base,
        "decode_port": base + 1,
        "proxy_port": base + 2,
        "prefill_bootstrap_port": base + 3,
    }
    if len(set(ports.values())) != len(ports):
        raise ValueError(f"derived ports must be distinct, got {ports}.")
    return ports


def run_name_for_point(args: argparse.Namespace, point: RegressionPoint) -> str:
    rate = _sanitize_label(str(args.request_rate))
    tasks_label = _sanitize_label("+".join(parse_str_csv(args.tasks, name="tasks")))
    datasets_label = _sanitize_label(
        "+".join(
            parse_str_csv(args.longbench_datasets, name="longbench_datasets")
            + parse_str_csv(args.reasoning_datasets, name="reasoning_datasets")
        )
    )
    return (
        f"pdacc_mixed_{tasks_label}_{datasets_label}_"
        f"kv-{_sanitize_label(point.variant)}_c{point.max_concurrency}_"
        f"ln{point.longbench_max_samples}_mn{point.reasoning_max_samples}_r{rate}"
    )


def _append_flag(command: list[str], flag: str, enabled: bool) -> None:
    if enabled:
        command.append(flag)


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_command(args: argparse.Namespace, point: RegressionPoint) -> list[str]:
    ports = ports_for_point(args, point)
    prefill_dtype = "auto" if point.variant == "reflex_int4" else point.variant
    prompt_fit_policy = getattr(args, "prompt_fit_policy", "truncate")
    prompt_fit_token_margin = getattr(args, "prompt_fit_token_margin", 8)
    command = [
        sys.executable,
        str(PD_MIXED_ACCURACY_SCRIPT),
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
        "--proxy-prefill-max-inflight",
        str(args.proxy_prefill_max_inflight),
        "--mooncake-protocol",
        args.mooncake_protocol,
        "--mooncake-num-workers",
        str(args.mooncake_num_workers),
        "--tasks",
        args.tasks,
        "--longbench-datasets",
        args.longbench_datasets,
        "--reasoning-datasets",
        args.reasoning_datasets,
        "--longbench-data-dir",
        args.longbench_data_dir,
        "--reasoning-data-dir",
        args.reasoning_data_dir,
        "--config-dir",
        args.config_dir,
        "--output-root",
        args.output_root,
        "--run-name",
        run_name_for_point(args, point),
        "--longbench-max-samples",
        str(point.longbench_max_samples),
        "--reasoning-max-samples",
        str(point.reasoning_max_samples),
        "--prompt-fit-policy",
        prompt_fit_policy,
        "--prompt-fit-token-margin",
        str(prompt_fit_token_margin),
        "--max-concurrency",
        str(point.max_concurrency),
        "--request-rate",
        str(args.request_rate),
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
        prefill_dtype,
        "--decode-kv-cache-dtype",
        point.variant,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--seed",
        str(args.seed),
        "--sample-interval-sec",
        str(args.sample_interval_sec),
        "--server-ready-timeout-sec",
        str(args.server_ready_timeout_sec),
        "--request-timeout-sec",
        str(args.request_timeout_sec),
        "--scheduling-policy",
        args.scheduling_policy,
        "--slo-classes",
        args.slo_classes,
        f"--slo-priorities={args.slo_priorities}",
        "--workload-mix-policy",
        args.workload_mix_policy,
    ]

    _append_flag(command, "--force-triton-attn", args.force_triton_attn)
    _append_flag(command, "--enforce-eager", args.enforce_eager)
    _append_flag(command, "--skip-chat-template", args.skip_chat_template)

    if point.variant == "reflex_int4":
        _append_flag(command, "--enable-reflex-trace", args.enable_reflex_trace)
        _append_optional(
            command,
            "--reflex-int4-budget-fraction",
            args.reflex_int4_budget_fraction,
        )
        command.extend(
            [
                "--reflex-keep-initial-blocks",
                str(args.reflex_keep_initial_blocks),
                "--reflex-keep-recent-blocks",
                str(args.reflex_keep_recent_blocks),
            ]
        )
        _append_optional(
            command,
            "--reflex-max-int4-fraction-per-request",
            args.reflex_max_int4_fraction_per_request,
        )
        command.extend(
            [
                "--reflex-survival-warmup-tokens",
                str(args.reflex_survival_warmup_tokens),
                "--reflex-risk-warmup-tokens",
                str(args.reflex_risk_warmup_tokens),
                "--reflex-short-admission-max-int4-fraction",
                str(args.reflex_short_admission_max_int4_fraction),
                "--reflex-sparse-window-pages",
                str(args.reflex_sparse_window_pages),
                "--reflex-short-max-demote-per-window",
                str(args.reflex_short_max_demote_per_window),
                "--reflex-max-demote-per-window",
                str(args.reflex_max_demote_per_window),
                "--reflex-low-risk-score-fraction",
                str(args.reflex_low_risk_score_fraction),
                "--reflex-page-selection-policy",
                str(
                    getattr(
                        args,
                        "reflex_page_selection_policy",
                        "relevance_sparse",
                    )
                ),
                "--reflex-decode-pressure-warmup-tokens",
                str(args.reflex_decode_pressure_warmup_tokens),
                "--reflex-decode-pressure-ramp-tokens",
                str(args.reflex_decode_pressure_ramp_tokens),
                "--reflex-short-prefill-pages",
                str(args.reflex_short_prefill_pages),
                "--reflex-long-prefill-pages",
                str(args.reflex_long_prefill_pages),
                "--reflex-slo-pressure-step",
                str(args.reflex_slo_pressure_step),
                "--reflex-min-slo-pressure",
                str(args.reflex_min_slo_pressure),
                "--reflex-max-slo-pressure",
                str(args.reflex_max_slo_pressure),
            ]
        )
    return command


def command_record(
    args: argparse.Namespace,
    point: RegressionPoint,
    command: list[str],
) -> dict[str, object]:
    return {
        "point": asdict(point),
        "run_name": run_name_for_point(args, point),
        "command": command,
    }


def write_commands(path: str | None, records: list[dict[str, object]]) -> None:
    if path is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_manifest(
    output_root: Path,
    args: argparse.Namespace,
    records: list[dict[str, object]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "args": vars(args),
        "runs": records,
    }
    (output_root / "regression_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_regression(args: argparse.Namespace) -> int:
    records = [
        command_record(args, point, build_command(args, point))
        for point in selected_points(args)
    ]
    write_commands(args.commands_out, records)
    write_manifest(Path(args.output_root), args, records)

    if args.dry_run:
        for record in records:
            print(shlex.join(record["command"]))  # type: ignore[arg-type]
        return 0

    for record in records:
        command = record["command"]
        proc = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            check=False,
        )
        record["returncode"] = proc.returncode
        write_manifest(Path(args.output_root), args, records)
        if proc.returncode != 0 and not args.continue_on_error:
            return proc.returncode
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded 1P1D accuracy regression matrix.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefill-gpu", default="6")
    parser.add_argument("--decode-gpu", default="7")
    parser.add_argument("--base-port", type=int, default=8810)
    parser.add_argument("--port-stride", type=int, default=10)
    parser.add_argument("--output-root", default="outputs/accuracy/pd_serving_regression")

    parser.add_argument("--tasks", default="longbench,reasoning")
    parser.add_argument("--longbench-datasets", default="qasper,hotpotqa,multifieldqa_en")
    parser.add_argument("--reasoning-datasets", default="math500")
    parser.add_argument("--longbench-data-dir", default=DEFAULT_LONGBENCH_DATA_DIR)
    parser.add_argument("--reasoning-data-dir", default=DEFAULT_REASONING_DATA_DIR)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--longbench-max-samples", type=int, default=16)
    parser.add_argument("--reasoning-max-samples", type=int, default=16)
    parser.add_argument(
        "--prompt-fit-policy",
        choices=["none", "skip", "truncate"],
        default="truncate",
    )
    parser.add_argument("--prompt-fit-token-margin", type=int, default=8)
    parser.add_argument("--variants", default="auto,fp8,int4,reflex_int4")
    parser.add_argument("--concurrencies", default="1,8")
    parser.add_argument("--request-rate", default="0.5")

    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--mooncake-protocol", default="rdma")
    parser.add_argument("--mooncake-num-workers", type=int, default=10)
    parser.add_argument("--force-triton-attn", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true", default=True)
    parser.add_argument("--enable-reflex-trace", action="store_true", default=True)
    parser.add_argument("--reflex-int4-budget-fraction", type=float, default=0.25)
    parser.add_argument("--reflex-keep-initial-blocks", type=int, default=4)
    parser.add_argument("--reflex-keep-recent-blocks", type=int, default=16)
    parser.add_argument("--reflex-max-int4-fraction-per-request", type=float, default=0.5)
    parser.add_argument("--reflex-survival-warmup-tokens", type=int, default=128)
    parser.add_argument("--reflex-risk-warmup-tokens", type=int, default=16)
    parser.add_argument("--reflex-short-admission-max-int4-fraction", type=float, default=0.03)
    parser.add_argument("--reflex-sparse-window-pages", type=int, default=32)
    parser.add_argument("--reflex-short-max-demote-per-window", type=int, default=1)
    parser.add_argument("--reflex-max-demote-per-window", type=int, default=2)
    parser.add_argument("--reflex-low-risk-score-fraction", type=float, default=0.25)
    parser.add_argument(
        "--reflex-page-selection-policy",
        choices=["oldest", "distance", "random", "relevance", "relevance_sparse"],
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
    parser.add_argument("--proxy-prefill-max-inflight", type=int, default=2)
    parser.add_argument("--reflex-decode-pressure-warmup-tokens", type=int, default=32)
    parser.add_argument("--reflex-decode-pressure-ramp-tokens", type=int, default=512)
    parser.add_argument("--reflex-short-prefill-pages", type=int, default=64)
    parser.add_argument("--reflex-long-prefill-pages", type=int, default=512)
    parser.add_argument("--reflex-slo-pressure-step", type=float, default=0.25)
    parser.add_argument("--reflex-min-slo-pressure", type=float, default=0.5)
    parser.add_argument("--reflex-max-slo-pressure", type=float, default=1.5)
    parser.add_argument("--scheduling-policy", choices=["fcfs", "priority"], default="priority")

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--slo-classes", default="high,normal,low")
    parser.add_argument("--slo-priorities", default="-1,0,1")
    parser.add_argument(
        "--workload-mix-policy",
        choices=["balanced", "random"],
        default="balanced",
    )
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--server-ready-timeout-sec", type=float, default=420.0)
    parser.add_argument("--request-timeout-sec", type=float, default=900.0)
    parser.add_argument("--skip-chat-template", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commands-out", default=None)
    parser.add_argument("--continue-on-error", action="store_true")
    dry_run = parser.add_mutually_exclusive_group()
    dry_run.add_argument("--dry-run", dest="dry_run", action="store_true")
    dry_run.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    return parser.parse_args()


def main() -> int:
    try:
        return run_regression(parse_args())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
