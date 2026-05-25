#!/usr/bin/env python3
"""Run accuracy evaluation through the ReFlexKV 1P1D serving path."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import sys
import threading
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Callable

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.backends.base import PromptRecord
from eval.utils.eval_longbench import evaluate_file as evaluate_longbench_file
from eval.utils.eval_reasoning import evaluate_file as evaluate_reasoning_file
from eval.utils.longbench import (
    build_prompt_records,
    load_config as load_longbench_config,
    load_jsonl_rows,
    resolve_dataset_max_samples,
    should_use_chat_format,
)
from eval.utils.reasoning import (
    build_reasoning_prompt_records,
    load_config as load_reasoning_config,
    load_reasoning_rows,
    resolve_reasoning_max_samples,
)
from scripts.profiling import run_reflex_pd_1p1d as pd_runner


ROOT = pd_runner.ROOT
DEFAULT_MODEL = pd_runner.DEFAULT_MODEL
DEFAULT_CONFIG_DIR = str(ROOT / "eval" / "config")
DEFAULT_LONGBENCH_DATA_DIR = str(ROOT / "data" / "longbench")
DEFAULT_REASONING_DATA_DIR = str(ROOT / "data" / "reasoning")

Role = pd_runner.Role
build_server_cmd = pd_runner.build_server_cmd
build_server_env = pd_runner.build_server_env
build_proxy_cmd = pd_runner.build_proxy_cmd


@dataclass(frozen=True)
class ServingDataset:
    task: str
    dataset: str
    max_new_tokens: int
    records: list[PromptRecord]


@dataclass(frozen=True)
class ServingPrediction:
    index: int
    pred: str
    error: str | None
    latency_seconds: float


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LongBench/Math500 accuracy through 1P1D serving."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefill-gpu", default="6")
    parser.add_argument("--decode-gpu", default="7")
    parser.add_argument("--prefill-port", type=int, default=8710)
    parser.add_argument("--decode-port", type=int, default=8720)
    parser.add_argument("--proxy-port", type=int, default=8730)
    parser.add_argument("--prefill-bootstrap-port", type=int, default=8998)
    parser.add_argument("--proxy-prefill-max-inflight", type=int, default=0)
    parser.add_argument(
        "--proxy-prefill-metadata-wait-timeout-sec",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--proxy-decode-backpressure-policy",
        choices=["off", "metrics"],
        default="off",
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
    parser.add_argument("--scheduling-policy", choices=["fcfs", "priority"], default="fcfs")

    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--prefill-kv-cache-dtype", default="auto")
    parser.add_argument("--decode-kv-cache-dtype", default="reflex_int4")
    parser.add_argument("--num-gpu-blocks-override", type=int, default=None)
    parser.add_argument("--force-triton-attn", action="store_true", default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--enable-reflex-trace", action="store_true")
    parser.add_argument(
        "--disable-reflex-prefill-page-metadata",
        action="store_true",
        help="Disable P-side ReFlexKV page-risk metadata for ablation runs.",
    )
    parser.add_argument("--reflex-int4-budget-fraction", type=float, default=None)
    parser.add_argument("--extra-serve-args", nargs=argparse.REMAINDER, default=[])

    parser.add_argument("--task", choices=["longbench", "reasoning"], required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-root", default="outputs/accuracy/pd_serving")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-samples", type=int, default=8)
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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-interval-sec", type=float, default=1.0)
    parser.add_argument("--server-ready-timeout-sec", type=float, default=420.0)
    parser.add_argument("--request-timeout-sec", type=float, default=900.0)
    return parser.parse_args(argv)


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    return str(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_data_dir(args: argparse.Namespace) -> str:
    if args.data_dir is not None:
        return args.data_dir
    if args.task == "longbench":
        return DEFAULT_LONGBENCH_DATA_DIR
    return DEFAULT_REASONING_DATA_DIR


def build_chat_formatter(args: argparse.Namespace) -> Callable[[str], str] | None:
    if args.skip_chat_template:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if not hasattr(tokenizer, "apply_chat_template"):
        return None

    def format_prompt(prompt: str) -> str:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    return format_prompt


def load_serving_dataset(
    args: argparse.Namespace,
    chat_formatter: Callable[[str], str] | None,
) -> ServingDataset:
    data_dir = resolve_data_dir(args)
    data_file = Path(data_dir) / f"{args.dataset}.jsonl"
    if not data_file.exists():
        raise FileNotFoundError(f"Missing dataset file: {data_file}")

    if args.task == "longbench":
        prompt_config = load_longbench_config(args.config_dir, "dataset2prompt")
        maxlen_config = load_longbench_config(args.config_dir, "dataset2maxlen")
        max_samples = resolve_dataset_max_samples(
            args.dataset,
            args.config_dir,
            args.max_samples,
        )
        formatter = chat_formatter if should_use_chat_format(args.dataset) else None
        payloads = build_prompt_records(
            load_jsonl_rows(str(data_file))[:max_samples],
            prompt_config[args.dataset],
            chat_formatter=formatter,
        )
        records = [
            PromptRecord(
                dataset=args.dataset,
                prompt=payload["prompt"],
                answers=payload["answers"],
                all_classes=payload["all_classes"],
            )
            for payload in payloads
        ]
        return ServingDataset(
            task=args.task,
            dataset=args.dataset,
            max_new_tokens=int(maxlen_config[args.dataset]),
            records=records,
        )

    prompt_config = load_reasoning_config(args.config_dir, "reasoning_dataset2prompt")
    maxlen_config = load_reasoning_config(args.config_dir, "reasoning_dataset2maxlen")
    max_samples = resolve_reasoning_max_samples(
        args.dataset,
        args.config_dir,
        args.max_samples,
    )
    payloads = build_reasoning_prompt_records(
        args.dataset,
        load_reasoning_rows(str(data_file))[:max_samples],
        prompt_config[args.dataset],
        chat_formatter=chat_formatter,
    )
    records = [
        PromptRecord(
            dataset=args.dataset,
            prompt=payload["prompt"],
            answers=payload["answers"],
            all_classes=payload["all_classes"],
            meta=payload.get("meta"),
        )
        for payload in payloads
    ]
    return ServingDataset(
        task=args.task,
        dataset=args.dataset,
        max_new_tokens=int(maxlen_config[args.dataset]),
        records=records,
    )


def _tokenizer_encode(tokenizer: Any, prompt: str) -> list[int]:
    try:
        return list(tokenizer.encode(prompt, add_special_tokens=False))
    except TypeError:
        return list(tokenizer.encode(prompt))


def _tokenizer_decode(tokenizer: Any, token_ids: list[int]) -> str:
    try:
        return str(tokenizer.decode(token_ids, skip_special_tokens=False))
    except TypeError:
        return str(tokenizer.decode(token_ids))


def _load_prompt_fit_tokenizer(args: argparse.Namespace) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)


def _truncate_middle(token_ids: list[int], limit: int) -> list[int]:
    if len(token_ids) <= limit:
        return token_ids
    left = limit // 2
    right = limit - left
    if right <= 0:
        return token_ids[:limit]
    return token_ids[:left] + token_ids[-right:]


def fit_dataset_to_model_len(
    args: argparse.Namespace,
    dataset: ServingDataset,
    *,
    tokenizer: Any | None = None,
) -> tuple[ServingDataset, dict[str, Any]]:
    policy = getattr(args, "prompt_fit_policy", "truncate")
    margin = int(getattr(args, "prompt_fit_token_margin", 8))
    if margin < 0:
        raise ValueError("--prompt-fit-token-margin must be non-negative")

    token_limit = int(args.max_model_len) - int(dataset.max_new_tokens) - margin
    if token_limit <= 0:
        raise ValueError(
            "--max-model-len must exceed dataset max_new_tokens plus "
            "--prompt-fit-token-margin"
        )

    summary: dict[str, Any] = {
        "dataset": dataset.dataset,
        "task": dataset.task,
        "policy": policy,
        "max_model_len": int(args.max_model_len),
        "max_new_tokens": int(dataset.max_new_tokens),
        "token_margin": margin,
        "prompt_token_limit": token_limit,
        "original_records": len(dataset.records),
        "kept_records": len(dataset.records),
        "skipped_records": 0,
        "truncated_records": 0,
        "max_original_prompt_tokens": None,
        "max_final_prompt_tokens": None,
    }
    if policy == "none":
        return dataset, summary

    tokenizer = tokenizer or _load_prompt_fit_tokenizer(args)
    fitted_records: list[PromptRecord] = []
    original_lengths: list[int] = []
    final_lengths: list[int] = []
    skipped = 0
    truncated = 0

    for record in dataset.records:
        original_ids = _tokenizer_encode(tokenizer, record.prompt)
        original_len = len(original_ids)
        original_lengths.append(original_len)
        final_prompt = record.prompt
        final_len = original_len
        was_truncated = False

        if original_len > token_limit:
            if policy == "skip":
                skipped += 1
                continue
            truncated_ids = _truncate_middle(original_ids, token_limit)
            final_prompt = _tokenizer_decode(tokenizer, truncated_ids)
            final_len = len(truncated_ids)
            truncated += 1
            was_truncated = True

        meta = dict(record.meta or {})
        meta.update(
            {
                "prompt_original_tokens": original_len,
                "prompt_final_tokens": final_len,
                "prompt_truncated": was_truncated,
                "prompt_fit_limit_tokens": token_limit,
            }
        )
        final_lengths.append(final_len)
        fitted_records.append(replace(record, prompt=final_prompt, meta=meta))

    summary.update(
        {
            "kept_records": len(fitted_records),
            "skipped_records": skipped,
            "truncated_records": truncated,
            "max_original_prompt_tokens": (
                max(original_lengths) if original_lengths else None
            ),
            "max_final_prompt_tokens": max(final_lengths) if final_lengths else None,
        }
    )
    return replace(dataset, records=fitted_records), summary


async def _async_completion_request(
    *,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    record: PromptRecord,
    max_tokens: int,
    temperature: float,
    top_p: float,
    priority: int | None = None,
    request_id: str | None = None,
) -> str:
    payload = {
        "model": model,
        "prompt": record.prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stream": False,
    }
    if priority is not None:
        payload["priority"] = priority
    if request_id is not None:
        payload["request_id"] = request_id
    response = await client.post(f"{base_url}/v1/completions", json=payload)
    response.raise_for_status()
    body = response.json()
    choices = body.get("choices") or []
    if not choices:
        return ""
    return str(choices[0].get("text", ""))


def _parse_request_rate(value: str) -> float:
    if value.lower() in {"inf", "infinity"}:
        return math.inf
    rate = float(value)
    if rate <= 0:
        raise ValueError("--request-rate must be positive or inf")
    return rate


async def run_serving_requests(
    *,
    args: argparse.Namespace,
    dataset: ServingDataset,
    base_url: str,
) -> list[ServingPrediction]:
    rng = random.Random(args.seed)
    semaphore = asyncio.Semaphore(args.max_concurrency)
    request_rate = _parse_request_rate(str(args.request_rate))
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
        index: int,
        record: PromptRecord,
    ) -> ServingPrediction:
        async with semaphore:
            start = time.perf_counter()
            try:
                pred = await asyncio.wait_for(
                    _async_completion_request(
                        client=client,
                        base_url=base_url,
                        model=args.model,
                        record=record,
                        max_tokens=dataset.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
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
            return ServingPrediction(
                index=index,
                pred=pred,
                error=error,
                latency_seconds=time.perf_counter() - start,
            )

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        trust_env=False,
    ) as client:
        tasks = []
        for index, record in enumerate(dataset.records):
            if index > 0 and math.isfinite(request_rate):
                await asyncio.sleep(rng.expovariate(request_rate))
            tasks.append(asyncio.create_task(run_one(client, index, record)))
        predictions = await asyncio.gather(*tasks)
    return sorted(predictions, key=lambda item: item.index)


def write_predictions_and_score(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset: ServingDataset,
    predictions: list[ServingPrediction],
    duration_seconds: float,
) -> float:
    dataset_dir = run_dir / dataset.dataset
    pred_file = dataset_dir / "pred.jsonl"
    rows = []
    failed_predictions = 0
    latencies = []
    for prediction in predictions:
        record = dataset.records[prediction.index]
        if prediction.error is not None or prediction.pred == "":
            failed_predictions += 1
        latencies.append(prediction.latency_seconds)
        rows.append(
            {
                "pred": prediction.pred,
                "answers": record.answers,
                "all_classes": record.all_classes,
                "meta": {
                    **(record.meta or {}),
                    "request_index": prediction.index,
                    "latency_seconds": prediction.latency_seconds,
                    "error": prediction.error,
                },
            }
        )
    _append_jsonl(pred_file, rows)
    _write_json(dataset_dir / "run_config.json", vars(args))

    if dataset.task == "longbench":
        score = float(evaluate_longbench_file(str(pred_file), dataset.dataset, args.config_dir))
    else:
        score = float(evaluate_reasoning_file(str(pred_file), dataset.dataset, args.config_dir))

    _write_json(
        dataset_dir / "run_summary.json",
        {
            "dataset": dataset.dataset,
            "task": dataset.task,
            "requested_samples": len(dataset.records),
            "completed_predictions": len(predictions),
            "failed_predictions": failed_predictions,
            "avg_score": score,
            "duration_seconds": round(duration_seconds, 4),
            "avg_latency_seconds": mean(latencies) if latencies else 0.0,
            "max_latency_seconds": max(latencies) if latencies else 0.0,
            "request_rate": args.request_rate,
            "max_concurrency": args.max_concurrency,
        },
    )
    return score


def make_run_dir(args: argparse.Namespace) -> Path:
    run_name = args.run_name
    if not run_name:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        rate = str(args.request_rate).replace(".", "p")
        run_name = (
            f"{stamp}_pdserv_{args.task}_{args.dataset}_"
            f"kv-{args.decode_kv_cache_dtype}_c{args.max_concurrency}_"
            f"n{args.max_samples}_r{rate}"
        )
    run_dir = Path(args.output_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main() -> int:
    args = parse_args()
    args.data_dir = resolve_data_dir(args)
    pd_runner.validate_args(args)
    if args.max_concurrency <= 0:
        raise ValueError("--max-concurrency must be positive")

    run_dir = make_run_dir(args)
    prefill_url = f"http://{args.host}:{args.prefill_port}"
    decode_url = f"http://{args.host}:{args.decode_port}"
    proxy_url = f"http://{args.host}:{args.proxy_port}"

    prefill_cmd = build_server_cmd(args, Role.PREFILL)
    decode_cmd = build_server_cmd(args, Role.DECODE)
    proxy_cmd = build_proxy_cmd(args)
    _write_json(run_dir / "config.json", vars(args))
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
    dataset = load_serving_dataset(args, chat_formatter)
    dataset, prompt_fit_summary = fit_dataset_to_model_len(args, dataset)
    _write_json(run_dir / "prompt_fit_summary.json", prompt_fit_summary)

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
            run_serving_requests(args=args, dataset=dataset, base_url=proxy_url)
        )
        score = write_predictions_and_score(
            args=args,
            run_dir=run_dir,
            dataset=dataset,
            predictions=predictions,
            duration_seconds=time.perf_counter() - start,
        )
        print(f"run_dir={run_dir}")
        print(f"score={score}")
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
