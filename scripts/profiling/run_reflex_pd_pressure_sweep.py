#!/usr/bin/env python3
"""Enumerate and run ReFlexKV 1P1D pressure sweep experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SINGLE_RUN_SCRIPT = ROOT / "scripts" / "profiling" / "run_reflex_pd_1p1d.py"
DEFAULT_MODEL = "/home/ytm/models/Llama-3.1-8B-Instruct"

DEFAULT_INPUT_LENS = "4096,8192,12288"
DEFAULT_OUTPUT_LENS = "128,512,1024"
DEFAULT_REQUEST_RATES = "0.25,0.5,1.0"
DEFAULT_CONCURRENCIES = "4,8,16"
DEFAULT_NUM_PROMPTS_LIST = "16"
DEFAULT_DECODE_DTYPES = "auto,reflex_int4"


@dataclass(frozen=True)
class SweepPoint:
    index: int
    decode_dtype: str
    input_len: int
    output_len: int
    request_rate: str
    max_concurrency: int
    num_prompts: int


def parse_int_csv(value: str, *, name: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one integer.")
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError as exc:
            raise ValueError(f"{name} contains a non-integer value: {item!r}") from exc
    return parsed


def parse_str_csv(value: str, *, name: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError(f"{name} must contain at least one value.")
    return parsed


def enumerate_points(args: argparse.Namespace) -> list[SweepPoint]:
    decode_dtypes = parse_str_csv(args.decode_dtypes, name="decode_dtypes")
    input_lens = parse_int_csv(args.input_lens, name="input_lens")
    output_lens = parse_int_csv(args.output_lens, name="output_lens")
    request_rates = parse_str_csv(args.request_rates, name="request_rates")
    concurrencies = parse_int_csv(args.concurrencies, name="concurrencies")
    num_prompts_list = parse_int_csv(
        args.num_prompts_list,
        name="num_prompts_list",
    )

    points: list[SweepPoint] = []
    for decode_dtype in decode_dtypes:
        for input_len in input_lens:
            for output_len in output_lens:
                for request_rate in request_rates:
                    for concurrency in concurrencies:
                        for num_prompts in num_prompts_list:
                            points.append(
                                SweepPoint(
                                    index=len(points),
                                    decode_dtype=decode_dtype,
                                    input_len=input_len,
                                    output_len=output_len,
                                    request_rate=request_rate,
                                    max_concurrency=concurrency,
                                    num_prompts=num_prompts,
                                )
                            )
    return points


def selected_points(args: argparse.Namespace) -> list[SweepPoint]:
    points = enumerate_points(args)
    if args.limit is None:
        return points
    if args.limit < 0:
        raise ValueError("--limit must be non-negative.")
    return points[: args.limit]


def ports_for_point(args: argparse.Namespace, point: SweepPoint) -> dict[str, int]:
    if args.port_stride < 6:
        raise ValueError(
            "--port-stride (port_stride) must be at least 6 to avoid port collisions."
        )
    base = args.base_port + point.index * args.port_stride
    ports = {
        "prefill_port": base,
        "decode_port": base + 1,
        "proxy_port": base + 2,
        "kv_proxy_port": base + 3,
        "prefill_kv_port": base + 4,
        "decode_kv_port": base + 5,
    }
    if len(set(ports.values())) != len(ports):
        raise ValueError(f"derived ports must be distinct, got {ports}.")
    return ports


def run_name_for_point(point: SweepPoint) -> str:
    dtype = point.decode_dtype.replace("/", "_").replace(":", "_")
    request_rate = point.request_rate.replace(".", "p").replace("/", "_")
    return (
        f"pd1p1d_decode-{dtype}_"
        f"i{point.input_len}_o{point.output_len}_"
        f"c{point.max_concurrency}_np{point.num_prompts}_"
        f"r{request_rate}"
    )


def _append_optional(command: list[str], flag: str, value: object | None) -> None:
    if value is not None:
        command.extend([flag, str(value)])


def build_command(args: argparse.Namespace, point: SweepPoint) -> list[str]:
    ports = ports_for_point(args, point)
    command = [
        sys.executable,
        str(SINGLE_RUN_SCRIPT),
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
        "--kv-proxy-port",
        str(ports["kv_proxy_port"]),
        "--prefill-kv-port",
        str(ports["prefill_kv_port"]),
        "--decode-kv-port",
        str(ports["decode_kv_port"]),
        "--run-name",
        run_name_for_point(point),
        "--output-root",
        args.output_root,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--decode-kv-cache-dtype",
        point.decode_dtype,
        "--dataset-name",
        args.dataset_name,
        "--input-len",
        str(point.input_len),
        "--output-len",
        str(point.output_len),
        "--num-prompts",
        str(point.num_prompts),
        "--request-rate",
        point.request_rate,
        "--max-concurrency",
        str(point.max_concurrency),
        "--seed",
        str(args.seed),
        "--p2p-kv-chunk-blocks",
        str(args.p2p_kv_chunk_blocks),
        "--p2p-max-staged-bytes",
        str(args.p2p_max_staged_bytes),
    ]
    reflex_int4_budget_fraction = getattr(args, "reflex_int4_budget_fraction", None)
    if reflex_int4_budget_fraction is not None:
        command.extend(
            [
                "--reflex-int4-budget-fraction",
                str(reflex_int4_budget_fraction),
            ]
        )
    _append_optional(command, "--dataset-path", args.dataset_path)
    if args.enable_reflex_trace:
        command.append("--enable-reflex-trace")
    if args.force_triton_attn:
        command.append("--force-triton-attn")
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.skip_chat_template:
        command.append("--skip-chat-template")
    if args.no_stream:
        command.append("--no-stream")
    return command


def command_record(point: SweepPoint, command: list[str]) -> dict[str, object]:
    return {
        "point": asdict(point),
        "run_name": run_name_for_point(point),
        "command": command,
    }


def write_commands(path: str | None, records: list[dict[str, object]]) -> None:
    if path is None:
        return
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, sort_keys=True) + "\n")


def run_sweep(args: argparse.Namespace) -> int:
    points = selected_points(args)
    records = [
        command_record(point, build_command(args, point))
        for point in points
    ]
    write_commands(args.commands_out, records)

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
        if proc.returncode != 0:
            return proc.returncode
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded ReFlexKV 1P1D pressure sweep.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--prefill-gpu", default="0")
    parser.add_argument("--decode-gpu", default="1")
    parser.add_argument("--base-port", type=int, default=8310)
    parser.add_argument("--port-stride", type=int, default=10)
    parser.add_argument("--output-root", default="outputs/profiling/reflex_pd_pressure")

    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--dataset-name", default="random")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--enable-reflex-trace", action="store_true")
    parser.add_argument("--force-triton-attn", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--skip-chat-template", action="store_true")
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--p2p-kv-chunk-blocks", type=int, default=32)
    parser.add_argument("--p2p-max-staged-bytes", type=int, default=64 * 1024 * 1024)

    parser.add_argument("--input-lens", default=DEFAULT_INPUT_LENS)
    parser.add_argument("--output-lens", default=DEFAULT_OUTPUT_LENS)
    parser.add_argument("--request-rates", default=DEFAULT_REQUEST_RATES)
    parser.add_argument("--concurrencies", default=DEFAULT_CONCURRENCIES)
    parser.add_argument("--num-prompts-list", default=DEFAULT_NUM_PROMPTS_LIST)
    parser.add_argument("--decode-dtypes", default=DEFAULT_DECODE_DTYPES)
    parser.add_argument("--reflex-int4-budget-fraction", type=float, default=None)

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--commands-out", default=None)
    dry_run = parser.add_mutually_exclusive_group()
    dry_run.add_argument("--dry-run", dest="dry_run", action="store_true")
    dry_run.add_argument("--no-dry-run", dest="dry_run", action="store_false")
    parser.set_defaults(dry_run=True)
    return parser.parse_args()


def main() -> int:
    try:
        return run_sweep(parse_args())
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
