from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch
from eval.utils.reasoning import build_reasoning_prompt_context, load_reasoning_rows
from eval.utils.semantiq_prior_calibration import (
    build_prior_artifact,
    mean_token_delta_nll,
    select_k_base_bits,
)
from vllm.semantiq.prior import resolve_semantiq_rank
from vllm.semantiq.query_segments import (
    ENV_ENABLE,
    ENV_FAKE_QUANT_ENABLE,
    ENV_FORCE_BIT_WIDTH,
    ENV_FORCE_HEAD_ID,
    ENV_FORCE_LAYER_NAME,
    ENV_FORCE_RANK,
    ENV_FORCE_SIDE,
    ENV_OUTPUT,
    ENV_PAGE_SIZE,
    ENV_PRIOR_PATH,
    ENV_THRESHOLD,
    get_query_segment_runtime,
    reset_query_segment_runtime,
)

_FORCED_TARGET_ENV_KEYS = (
    ENV_FORCE_LAYER_NAME,
    ENV_FORCE_RANK,
    ENV_FORCE_HEAD_ID,
    ENV_FORCE_SIDE,
    ENV_FORCE_BIT_WIDTH,
)
_CALIBRATION_ENV_KEYS = (
    ENV_ENABLE,
    ENV_PAGE_SIZE,
    ENV_THRESHOLD,
    ENV_OUTPUT,
    ENV_PRIOR_PATH,
    ENV_FAKE_QUANT_ENABLE,
    *_FORCED_TARGET_ENV_KEYS,
)
_SEMANTIQ_WORKER_EXTENSION_CLS = (
    "vllm.semantiq.query_segments.SemantiqWorkerExtension"
)


def _import_vllm():
    from vllm import LLM, SamplingParams

    return LLM, SamplingParams


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a standalone offline SemantiQ prior artifact."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--k-tau-2", type=float, required=True)
    parser.add_argument("--k-tau-4", type=float, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--semantiq-segment-page-size", type=int, default=16)
    parser.add_argument(
        "--semantiq-segment-similarity-threshold",
        type=float,
        default=0.8,
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def _ensure_file_exists(path: str, description: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {description}: {path}")


def _extract_prompt(row: dict[str, object]) -> str:
    return str(build_reasoning_prompt_context(row)["prompt"])


def _load_prompts(data_file: str, max_samples: int | None) -> list[str]:
    rows = load_reasoning_rows(data_file)
    if max_samples is not None:
        rows = rows[:max_samples]
    prompts = [_extract_prompt(row) for row in rows]
    if not prompts:
        raise ValueError("Calibration dataset must contain at least one prompt")
    return prompts


def _extract_token_nlls(request_output) -> list[float]:
    token_ids = list(request_output.prompt_token_ids or [])
    prompt_logprobs = list(request_output.prompt_logprobs or [])
    nlls: list[float] = []
    for position in range(1, min(len(token_ids), len(prompt_logprobs))):
        reference = prompt_logprobs[position]
        if not reference:
            continue
        token_id = token_ids[position]
        token_logprob = reference.get(token_id)
        if token_logprob is None:
            continue
        nlls.append(-float(token_logprob.logprob))
    if not nlls:
        raise ValueError("No prompt-side token logprobs were available for calibration")
    return nlls


def _run_prompt_nlls(llm, prompts: list[str], sampling_params) -> list[float]:
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise ValueError(
            f"vLLM returned {len(outputs)} outputs for {len(prompts)} prompts"
        )
    token_nlls: list[float] = []
    for output in outputs:
        token_nlls.extend(_extract_token_nlls(output))
    return token_nlls


def _collect_target_heads(snapshot: dict[str, object]) -> list[tuple[str, str, int]]:
    raw_rank = snapshot.get("rank")
    rank = resolve_semantiq_rank() if raw_rank is None else str(raw_rank)
    layers_to_heads: dict[str, set[int]] = {}
    requests = snapshot.get("requests", {})
    if not isinstance(requests, dict):
        return []
    for request_state in requests.values():
        if not isinstance(request_state, dict):
            continue
        layers = request_state.get("layers", {})
        if not isinstance(layers, dict):
            continue
        for layer_name, layer_state in layers.items():
            if not isinstance(layer_name, str) or not isinstance(layer_state, dict):
                continue
            segments_by_head = layer_state.get("segments_by_head", {})
            if not isinstance(segments_by_head, dict):
                continue
            for raw_head_id in segments_by_head:
                try:
                    head_id = int(raw_head_id)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid SemantiQ calibration head id {raw_head_id!r} "
                        f"for layer {layer_name!r}"
                    ) from exc
                layers_to_heads.setdefault(layer_name, set()).add(head_id)
    return [
        (rank, layer_name, head_id)
        for layer_name in sorted(layers_to_heads)
        for head_id in sorted(layers_to_heads[layer_name])
    ]


def _collect_target_heads_from_snapshots(
    snapshots: list[dict[str, object]],
) -> list[tuple[str, str, int]]:
    targets = {
        target
        for snapshot in snapshots
        for target in _collect_target_heads(snapshot)
    }
    return sorted(targets)


def _initialize_base_maps(
    targets: list[tuple[str, str, int]],
) -> dict[str, dict[str, list[int]]]:
    max_head_by_target: dict[tuple[str, str], int] = {}
    for rank, layer_name, head_id in targets:
        target_key = (layer_name, rank)
        max_head_by_target[target_key] = max(
            head_id,
            max_head_by_target.get(target_key, -1),
        )
    k_base_bits: dict[str, dict[str, list[int]]] = {}
    for (layer_name, rank), max_head_id in sorted(max_head_by_target.items()):
        k_base_bits.setdefault(layer_name, {})[rank] = [8] * (max_head_id + 1)
    return k_base_bits


def _clear_forced_target_env() -> None:
    for key in _FORCED_TARGET_ENV_KEYS:
        os.environ.pop(key, None)


def _refresh_query_segment_runtime() -> None:
    reset_query_segment_runtime()
    get_query_segment_runtime()


def _build_runtime_overrides(
    args,
    *,
    fake_quant_enabled: bool,
    rank: str | None = None,
    layer_name: str | None = None,
    head_id: int | None = None,
    side: str | None = None,
    bit_width: int | None = None,
    output_path: str | None = None,
) -> dict[str, object]:
    return {
        "enabled": True,
        "page_size": args.semantiq_segment_page_size,
        "similarity_threshold": args.semantiq_segment_similarity_threshold,
        "output_path": output_path,
        "prior_path": None,
        "fake_quant_enabled": fake_quant_enabled,
        "force_rank": rank,
        "force_layer_name": layer_name,
        "force_head_id": head_id,
        "force_side": side,
        "force_bit_width": bit_width,
    }


def _apply_local_runtime_overrides(overrides: dict[str, object]) -> None:
    os.environ[ENV_ENABLE] = "1" if bool(overrides["enabled"]) else "0"
    os.environ[ENV_PAGE_SIZE] = str(overrides["page_size"])
    os.environ[ENV_THRESHOLD] = str(overrides["similarity_threshold"])
    output_path = overrides.get("output_path")
    if output_path is None:
        os.environ.pop(ENV_OUTPUT, None)
    else:
        os.environ[ENV_OUTPUT] = str(output_path)
    prior_path = overrides.get("prior_path")
    if prior_path is None:
        os.environ.pop(ENV_PRIOR_PATH, None)
    else:
        os.environ[ENV_PRIOR_PATH] = str(prior_path)
    os.environ[ENV_FAKE_QUANT_ENABLE] = (
        "1" if bool(overrides["fake_quant_enabled"]) else "0"
    )
    _clear_forced_target_env()
    if overrides.get("force_layer_name") is not None:
        os.environ[ENV_FORCE_LAYER_NAME] = str(overrides["force_layer_name"])
    if overrides.get("force_rank") is not None:
        os.environ[ENV_FORCE_RANK] = str(overrides["force_rank"])
    if overrides.get("force_head_id") is not None:
        os.environ[ENV_FORCE_HEAD_ID] = str(overrides["force_head_id"])
    if overrides.get("force_side") is not None:
        os.environ[ENV_FORCE_SIDE] = str(overrides["force_side"])
    if overrides.get("force_bit_width") is not None:
        os.environ[ENV_FORCE_BIT_WIDTH] = str(overrides["force_bit_width"])
    _refresh_query_segment_runtime()


def _configure_runtime(llm, overrides: dict[str, object]) -> None:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        collective_rpc(
            "semantiq_configure_query_segment_runtime",
            args=(overrides,),
        )
        return
    _apply_local_runtime_overrides(overrides)


def _restore_env(original_env: dict[str, str | None]) -> None:
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _build_llm(args, llm_cls):
    return llm_cls(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        worker_extension_cls=_SEMANTIQ_WORKER_EXTENSION_CLS,
    )


def _shutdown_llm(llm) -> None:
    if llm is None:
        return
    engine = getattr(llm, "llm_engine", None)
    renderer = getattr(engine, "renderer", None)
    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()
    if renderer is not None and hasattr(renderer, "shutdown"):
        renderer.shutdown()
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _runtime_dump_paths(base_path: str) -> list[Path]:
    path = Path(base_path)
    suffix = path.suffix or ".json"
    stem = path.stem if path.suffix else path.name
    return sorted(path.parent.glob(f"{stem}.rank*.pid*{suffix}"))


def _clear_runtime_dumps(base_path: str) -> None:
    for path in _runtime_dump_paths(base_path):
        path.unlink(missing_ok=True)


def _load_runtime_dumps(base_path: str) -> list[dict[str, object]]:
    snapshots: list[dict[str, object]] = []
    for path in _runtime_dump_paths(base_path):
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            snapshots.append(payload)
    return snapshots


def _collect_runtime_snapshots(llm) -> list[dict[str, object]]:
    collective_rpc = getattr(llm, "collective_rpc", None)
    if callable(collective_rpc):
        try:
            snapshots = collective_rpc("semantiq_snapshot_query_segment_runtime")
        except RuntimeError as exc:
            if "not implemented" not in str(exc).lower():
                raise
            snapshots = []
        if snapshots:
            return [snapshot for snapshot in snapshots if isinstance(snapshot, dict)]
        return []
    return [get_query_segment_runtime().snapshot()]


def main(argv=None):
    args = parse_args(argv)
    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    if args.max_model_len is not None and args.max_model_len <= 0:
        raise ValueError("--max-model-len must be positive")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("--gpu-memory-utilization must be in (0, 1]")
    if args.semantiq_segment_page_size <= 0:
        raise ValueError("--semantiq-segment-page-size must be positive")

    _ensure_file_exists(args.data_file, "calibration dataset file")
    prompts = _load_prompts(args.data_file, args.max_samples)

    llm_cls, sampling_params_cls = _import_vllm()
    sampling_params = sampling_params_cls(
        max_tokens=1,
        temperature=0.0,
        prompt_logprobs=1,
    )

    original_env = {key: os.environ.get(key) for key in _CALIBRATION_ENV_KEYS}
    try:
        llm = _build_llm(args, llm_cls)
        try:
            baseline_overrides = _build_runtime_overrides(
                args,
                fake_quant_enabled=False,
            )
            _configure_runtime(llm, baseline_overrides)
            baseline_token_nlls = _run_prompt_nlls(llm, prompts, sampling_params)
            runtime_snapshots = _collect_runtime_snapshots(llm)

            targets = _collect_target_heads_from_snapshots(runtime_snapshots)
            if not targets:
                raise ValueError(
                    "No SemantiQ KV-head targets were observed in the baseline run"
                )

            deltas: dict[tuple[str, str, int, int], float] = {}
            for rank, layer_name, head_id in targets:
                for bit_width in (2, 4):
                    _configure_runtime(
                        llm,
                        _build_runtime_overrides(
                            args,
                            fake_quant_enabled=True,
                            rank=rank,
                            layer_name=layer_name,
                            head_id=head_id,
                            side="key",
                            bit_width=bit_width,
                        ),
                    )
                    perturbed_token_nlls = _run_prompt_nlls(
                        llm, prompts, sampling_params
                    )
                    deltas[(rank, layer_name, head_id, bit_width)] = (
                        mean_token_delta_nll(
                            baseline_token_nlls,
                            perturbed_token_nlls,
                        )
                    )
        finally:
            _shutdown_llm(llm)

        k_base_bits = _initialize_base_maps(targets)
        for rank, layer_name, head_id in targets:
            k_base_bit = select_k_base_bits(
                delta_at_2=deltas[(rank, layer_name, head_id, 2)],
                delta_at_4=deltas[(rank, layer_name, head_id, 4)],
                tau_2=args.k_tau_2,
                tau_4=args.k_tau_4,
            )
            k_base_bits[layer_name][rank][head_id] = k_base_bit

        meta = {
            "model": args.model,
            "metric": "delta_nll",
            "granularity": "kv_head",
            "page_size": args.semantiq_segment_page_size,
            "default_k_base_bits": 4,
            "aggregation": "mean_token_delta_nll",
            "k_thresholds": {"2": args.k_tau_2, "4": args.k_tau_4},
        }
        artifact = build_prior_artifact(
            k_base_bits=k_base_bits,
            meta=meta,
        )

        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    finally:
        _restore_env(original_env)
        _refresh_query_segment_runtime()


if __name__ == "__main__":
    main()
