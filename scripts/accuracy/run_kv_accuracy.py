#!/usr/bin/env python3
"""Run KV-cache dtype accuracy evaluations.

This script is intentionally accuracy-first. It reuses the existing LongBench
and reasoning runners so each KV dtype is evaluated with identical prompts,
metrics, and decoding parameters.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/ytm/models/Llama-3.1-8B-Instruct"
DEFAULT_LONGBENCH_DATA_DIR = str(ROOT / "data" / "longbench")
DEFAULT_REASONING_DATA_DIR = str(ROOT / "data" / "reasoning")
DEFAULT_OUTPUT_ROOT = str(ROOT / "outputs" / "accuracy" / "kv_cache")


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KV-cache accuracy sweeps.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpus", default="6")
    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--variants", default="auto,fp8,int4,reflex_int4")
    parser.add_argument("--tasks", default="longbench,reasoning")

    parser.add_argument("--longbench-datasets", default="qasper,hotpotqa,multifieldqa_en")
    parser.add_argument("--longbench-data-dir", default=DEFAULT_LONGBENCH_DATA_DIR)
    parser.add_argument("--longbench-max-samples", type=int, default=16)

    parser.add_argument("--reasoning-datasets", default="math500")
    parser.add_argument("--reasoning-data-dir", default=DEFAULT_REASONING_DATA_DIR)
    parser.add_argument("--reasoning-max-samples", type=int, default=16)

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--reflex-int4-budget-fraction",
        type=float,
        default=None,
        help="Fraction of KV byte budget reserved for ReFlexKV INT4 blocks.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    reflex_int4_budget_fraction = getattr(args, "reflex_int4_budget_fraction", None)
    if reflex_int4_budget_fraction is not None:
        env["SEMANTIQ_REFLEX_INT4_BUDGET_FRACTION"] = str(
            reflex_int4_budget_fraction
        )
    py_path = f"{ROOT}:{ROOT / 'vllm'}"
    env["PYTHONPATH"] = f"{py_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else py_path
    return env


def _effective_tp(args: argparse.Namespace) -> int:
    if args.tensor_parallel_size is not None:
        return args.tensor_parallel_size
    return len(_split_csv(args.gpus))


def _sanitize_label(value: str) -> str:
    return (
        value.replace(",", "+")
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def _variant_engine_args(variant: str) -> list[str]:
    args = ["--kv-cache-dtype", variant]
    if variant in {"int4", "reflex_int4"}:
        args.extend(["--attention-backend", "TRITON_ATTN", "--enforce-eager"])
    return args


def _dataset_items(value: str) -> list[str]:
    value = value.strip()
    if value == "all":
        return ["all"]
    items = _split_csv(value)
    if not items:
        raise ValueError("dataset list must include at least one dataset")
    return items


def build_eval_command(
    *,
    args: argparse.Namespace,
    task: str,
    datasets: str,
    data_dir: str,
    max_samples: int,
    variant: str,
    run_root: Path,
    run_name: str,
) -> list[str]:
    if task == "longbench":
        module = "eval.bench.longbench"
    elif task == "reasoning":
        module = "eval.bench.reasoning"
    else:
        raise ValueError(f"Unsupported task: {task}")

    cmd = [
        sys.executable,
        "-m",
        module,
        "--backend",
        "vllm",
        "--dataset",
        datasets,
        "--data-dir",
        data_dir,
        "--output-dir",
        str(run_root),
        "--run-name",
        run_name,
        "--max-samples",
        str(max_samples),
        "--batch-size",
        str(args.batch_size),
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(_effective_tp(args)),
        "--max-model-len",
        str(args.max_model_len),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--seed",
        str(args.seed),
        "--no-enable-prefix-caching",
    ]
    if args.resume:
        cmd.append("--resume")
    cmd.extend(_variant_engine_args(variant))
    return cmd


def _task_specs(args: argparse.Namespace) -> list[dict[str, str | int]]:
    specs = []
    tasks = set(_split_csv(args.tasks))
    if "longbench" in tasks:
        for dataset in _dataset_items(args.longbench_datasets):
            specs.append(
                {
                    "task": "longbench",
                    "datasets": dataset,
                    "data_dir": args.longbench_data_dir,
                    "max_samples": args.longbench_max_samples,
                }
            )
    if "reasoning" in tasks:
        for dataset in _dataset_items(args.reasoning_datasets):
            specs.append(
                {
                    "task": "reasoning",
                    "datasets": dataset,
                    "data_dir": args.reasoning_data_dir,
                    "max_samples": args.reasoning_max_samples,
                }
            )
    unsupported = tasks - {"longbench", "reasoning"}
    if unsupported:
        raise ValueError(f"Unsupported tasks: {sorted(unsupported)}")
    return specs


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    args = parse_args()
    variants = _split_csv(args.variants)
    if not variants:
        raise ValueError("--variants must include at least one KV cache dtype")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_root = Path(args.output_root) / run_tag
    run_root.mkdir(parents=True, exist_ok=True)
    env = build_env(args)

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "run_root": str(run_root),
        "args": vars(args),
        "runs": [],
    }

    for variant in variants:
        for spec in _task_specs(args):
            task = str(spec["task"])
            datasets = str(spec["datasets"])
            run_name = (
                f"{task}_{_sanitize_label(datasets)}_kv-{_sanitize_label(variant)}"
            )
            cmd = build_eval_command(
                args=args,
                task=task,
                datasets=datasets,
                data_dir=str(spec["data_dir"]),
                max_samples=int(spec["max_samples"]),
                variant=variant,
                run_root=run_root,
                run_name=run_name,
            )
            record = {
                "task": task,
                "datasets": datasets,
                "variant": variant,
                "run_name": run_name,
                "command": cmd,
                "returncode": None,
                "duration_seconds": None,
            }
            manifest["runs"].append(record)
            _write_json(run_root / "manifest.json", manifest)

            if args.dry_run:
                record["returncode"] = 0
                continue

            start = time.time()
            proc = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                text=True,
                check=False,
            )
            record["returncode"] = proc.returncode
            record["duration_seconds"] = round(time.time() - start, 4)
            _write_json(run_root / "manifest.json", manifest)
            if proc.returncode != 0 and not args.continue_on_error:
                return proc.returncode

    _write_json(run_root / "manifest.json", manifest)
    print(f"run_root={run_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
