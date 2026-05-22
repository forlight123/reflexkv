#!/usr/bin/env python3
"""Run a vLLM Full-KV serving pressure experiment.

The important vLLM signal is KV-pool occupancy, not total GPU memory from
nvidia-smi. vLLM reserves the KV pool at server startup, then request KV growth
appears as increased cache usage inside that pool.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = "/home/ytm/models/Llama-3.1-8B-Instruct"
METRIC_KEEP_RE = re.compile(
    r"(kv|cache|running|waiting|request|queue|scheduler|token|latency|tpot|ttft)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--port", type=int, default=8104)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--output-root", default="outputs/profiling/fullkv_pressure")

    parser.add_argument("--tensor-parallel-size", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-dtype", default="auto")
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)

    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--input-len", type=int, default=16000)
    parser.add_argument("--output-len", type=int, default=256)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--request-rate", default="inf")
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--temperature", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-chat-template", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--server-ready-timeout-sec", type=float, default=420.0)
    parser.add_argument("--enable-reflex-trace", action="store_true")
    parser.add_argument("--enable-reflex-attention-trace", action="store_true")
    parser.add_argument("--force-triton-attn", action="store_true")
    return parser.parse_args()


def make_run_dir(args: argparse.Namespace) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = args.run_name or (
        f"fullkv_{args.dataset_name}_i{args.input_len}_o{args.output_len}_"
        f"np{args.num_prompts}_c{args.max_concurrency}_rr{args.request_rate}"
    )
    run_dir = ROOT / args.output_root / f"{stamp}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def build_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    py_path = f"{ROOT}:{ROOT / 'vllm'}"
    env["PYTHONPATH"] = f"{py_path}:{env['PYTHONPATH']}" if env.get("PYTHONPATH") else py_path
    if args.enable_reflex_trace:
        env["SEMANTIQ_REFLEX_TRACE"] = "1"
    if args.enable_reflex_attention_trace:
        env["SEMANTIQ_REFLEX_TRACE"] = "1"
        env["SEMANTIQ_REFLEX_TRACE_ATTENTION"] = "1"
    return env


def wait_for_server(base_url: str, timeout_sec: float) -> None:
    deadline = time.time() + timeout_sec
    last_error = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=2) as resp:
                if resp.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = repr(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready at {base_url}: {last_error}")


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
    base_url: str,
    interval_sec: float,
    out_path: Path,
    stop_event: threading.Event,
) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        while not stop_event.is_set():
            sample = {
                "time": time.time(),
                "nvidia_smi": sample_nvidia_smi(),
                "vllm_metrics": {},
            }
            try:
                sample["vllm_metrics"] = parse_prometheus_metrics(
                    fetch_text(f"{base_url}/metrics")
                )
            except Exception as exc:  # noqa: BLE001 - sampler must not kill run.
                sample["metrics_error"] = repr(exc)
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


def main() -> int:
    args = parse_args()
    run_dir = make_run_dir(args)
    base_url = f"http://127.0.0.1:{args.port}"
    tp = args.tensor_parallel_size or len([gpu for gpu in args.gpus.split(",") if gpu])
    env = build_env(args)

    (run_dir / "config.json").write_text(
        json.dumps(vars(args) | {"tensor_parallel_size_effective": tp}, indent=2),
        encoding="utf-8",
    )

    server_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--tensor-parallel-size",
        str(tp),
        "--max-model-len",
        str(args.max_model_len),
        "--block-size",
        str(args.block_size),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--no-enable-prefix-caching",
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--disable-uvicorn-access-log",
    ]
    if args.num_gpu_blocks_override is not None:
        server_cmd.extend(
            ["--num-gpu-blocks-override", str(args.num_gpu_blocks_override)]
        )
    if args.force_triton_attn or args.kv_cache_dtype in {"int4", "reflex_int4"}:
        server_cmd.extend(["--attention-backend", "TRITON_ATTN", "--enforce-eager"])

    bench_result = "bench_result.json"
    bench_cmd = [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--base-url",
        base_url,
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
        bench_result,
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

    (run_dir / "server_cmd.txt").write_text(" ".join(server_cmd) + "\n", encoding="utf-8")
    (run_dir / "bench_cmd.txt").write_text(" ".join(bench_cmd) + "\n", encoding="utf-8")

    stop_event = threading.Event()
    server_log = (run_dir / "server.log").open("w", encoding="utf-8")
    bench_log = (run_dir / "bench.log").open("w", encoding="utf-8")
    server_proc: subprocess.Popen[str] | None = None
    sampler_thread: threading.Thread | None = None

    try:
        server_proc = subprocess.Popen(
            server_cmd,
            cwd=ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        wait_for_server(base_url, args.server_ready_timeout_sec)
        sampler_thread = threading.Thread(
            target=sampler,
            args=(base_url, args.sample_interval_sec, run_dir / "metrics_samples.jsonl", stop_event),
            daemon=True,
        )
        sampler_thread.start()
        bench_proc = subprocess.run(
            bench_cmd,
            cwd=ROOT,
            env=env,
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
        if server_proc is not None:
            terminate_process_group(server_proc)
        server_log.close()
        bench_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
