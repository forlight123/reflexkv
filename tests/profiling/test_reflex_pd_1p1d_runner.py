import argparse
import json
import sys
from pathlib import Path

import pytest

from scripts.profiling import run_reflex_pd_1p1d as runner


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/llama",
        host="127.0.0.1",
        prefill_gpu="0",
        decode_gpu="1",
        prefill_port=8310,
        decode_port=8410,
        proxy_port=8510,
        prefill_bootstrap_port=8998,
        proxy_prefill_max_inflight=2,
        proxy_prefill_metadata_wait_timeout_sec=None,
        proxy_decode_backpressure_policy="metrics",
        proxy_decode_backpressure_max_kv_usage=0.88,
        proxy_decode_backpressure_max_waiting=1,
        proxy_decode_backpressure_waiting_policy="adaptive",
        proxy_decode_backpressure_adaptive_max_waiting=6,
        proxy_decode_backpressure_adaptive_kv_headroom_per_waiting=0.03,
        proxy_decode_backpressure_poll_interval_sec=0.05,
        proxy_decode_backpressure_timeout_sec=30.0,
        proxy_decode_backpressure_admission_settle_sec=1.5,
        mooncake_protocol="rdma",
        mooncake_num_workers=10,
        reflex_keep_recent_blocks=4,
        reflex_keep_initial_blocks=32,
        reflex_max_int4_fraction_per_request=0.5,
        reflex_survival_warmup_tokens=128,
        reflex_risk_warmup_tokens=16,
        reflex_short_admission_max_int4_fraction=0.03,
        reflex_sparse_window_pages=32,
        reflex_short_max_demote_per_window=1,
        reflex_max_demote_per_window=2,
        reflex_low_risk_score_fraction=0.25,
        reflex_page_selection_policy="relevance_sparse",
        reflex_cold_admission_max_int4_fraction=0.1,
        reflex_cold_admission_emergency_free_ratio=0.05,
        reflex_decode_pressure_warmup_tokens=32,
        reflex_decode_pressure_ramp_tokens=512,
        reflex_short_prefill_pages=64,
        reflex_long_prefill_pages=512,
        reflex_global_evidence_min_prompt_pages=512,
        reflex_global_evidence_min_decode_tokens=129,
        reflex_global_evidence_landing_max_int4_fraction=0.08,
        reflex_slo_pressure_step=0.25,
        reflex_min_slo_pressure=0.5,
        reflex_max_slo_pressure=1.5,
        scheduling_policy="priority",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        block_size=16,
        max_num_seqs=8,
        max_num_batched_tokens=8192,
        prefill_kv_cache_dtype="auto",
        decode_kv_cache_dtype="reflex_int4",
        num_gpu_blocks_override=None,
        enforce_eager=True,
        force_triton_attn=True,
        enable_reflex_trace=True,
        extra_serve_args=[],
        output_root=str(tmp_path),
        run_name="unit",
        dataset_name="random",
        dataset_path=None,
        input_len=128,
        output_len=8,
        num_prompts=2,
        request_rate="inf",
        max_concurrency=2,
        temperature="0",
        seed=0,
        skip_chat_template=False,
        no_stream=True,
    )


def _kv_config_from_cmd(cmd: list[str]) -> dict:
    idx = cmd.index("--kv-transfer-config")
    return json.loads(cmd[idx + 1])


def test_1p1d_server_commands_keep_prefill_bf16_and_decode_reflex_int4(tmp_path):
    args = _args(tmp_path)

    prefill_cmd = runner.build_server_cmd(args, runner.Role.PREFILL)
    decode_cmd = runner.build_server_cmd(args, runner.Role.DECODE)

    assert prefill_cmd[:4] == [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
    ]
    assert prefill_cmd[prefill_cmd.index("--port") + 1] == "8310"
    assert prefill_cmd[prefill_cmd.index("--kv-cache-dtype") + 1] == "auto"
    assert prefill_cmd[prefill_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert "--enforce-eager" in prefill_cmd

    prefill_kv_config = _kv_config_from_cmd(prefill_cmd)
    assert prefill_kv_config["kv_connector"] == "ReFlexMooncakeConnector"
    assert prefill_kv_config["kv_role"] == "kv_producer"
    assert "kv_rank" not in prefill_kv_config
    assert "kv_parallel_size" not in prefill_kv_config
    assert "kv_port" not in prefill_kv_config
    assert prefill_kv_config["kv_connector_extra_config"] == {
        "mooncake_protocol": "rdma",
        "num_workers": 10,
        "reflex_keep_recent_blocks": 4,
    }

    assert decode_cmd[decode_cmd.index("--port") + 1] == "8410"
    assert decode_cmd[decode_cmd.index("--kv-cache-dtype") + 1] == "reflex_int4"
    assert decode_cmd[decode_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert "--enforce-eager" in decode_cmd
    assert decode_cmd[decode_cmd.index("--scheduling-policy") + 1] == "priority"
    assert "--num-gpu-blocks-override" not in decode_cmd

    decode_kv_config = _kv_config_from_cmd(decode_cmd)
    assert decode_kv_config["kv_connector"] == "ReFlexMooncakeConnector"
    assert decode_kv_config["kv_role"] == "kv_consumer"
    assert "kv_rank" not in decode_kv_config
    assert "kv_parallel_size" not in decode_kv_config
    assert "kv_port" not in decode_kv_config


def test_1p1d_reflex_decode_forces_triton_on_prefill_and_decode(tmp_path):
    args = _args(tmp_path)
    args.force_triton_attn = False
    args.decode_kv_cache_dtype = "reflex_int4"

    prefill_cmd = runner.build_server_cmd(args, runner.Role.PREFILL)
    decode_cmd = runner.build_server_cmd(args, runner.Role.DECODE)

    assert prefill_cmd[prefill_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert decode_cmd[decode_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"


def test_1p1d_env_and_proxy_benchmark_target_the_expected_processes(tmp_path):
    args = _args(tmp_path)

    prefill_env = runner.build_server_env(args, runner.Role.PREFILL)
    decode_env = runner.build_server_env(args, runner.Role.DECODE)
    proxy_cmd = runner.build_proxy_cmd(args)
    bench_cmd = runner.build_bench_cmd(args, tmp_path)

    assert prefill_env["CUDA_VISIBLE_DEVICES"] == "0"
    assert decode_env["CUDA_VISIBLE_DEVICES"] == "1"
    assert prefill_env["VLLM_MOONCAKE_BOOTSTRAP_PORT"] == "8998"
    assert prefill_env["SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA"] == "1"
    assert "VLLM_MOONCAKE_BOOTSTRAP_PORT" not in decode_env
    assert "SEMANTIQ_P2P_KV_CHUNK_BLOCKS" not in prefill_env
    assert "SEMANTIQ_P2P_KV_CHUNK_BLOCKS" not in decode_env
    assert "SEMANTIQ_REFLEX_TRACE" not in prefill_env
    assert decode_env["SEMANTIQ_REFLEX_TRACE"] == "1"
    assert decode_env["SEMANTIQ_REFLEX_KEEP_RECENT_PAGES"] == "4"
    assert decode_env["SEMANTIQ_REFLEX_KEEP_INITIAL_PAGES"] == "32"
    assert decode_env["SEMANTIQ_REFLEX_MAX_INT4_FRACTION_PER_REQUEST"] == "0.5"
    assert decode_env["SEMANTIQ_REFLEX_SURVIVAL_WARMUP_TOKENS"] == "128"
    assert decode_env["SEMANTIQ_REFLEX_RISK_WARMUP_TOKENS"] == "16"
    assert decode_env["SEMANTIQ_REFLEX_SHORT_ADMISSION_MAX_INT4_FRACTION"] == "0.03"
    assert decode_env["SEMANTIQ_REFLEX_SPARSE_WINDOW_PAGES"] == "32"
    assert decode_env["SEMANTIQ_REFLEX_SHORT_MAX_DEMOTE_PER_WINDOW"] == "1"
    assert decode_env["SEMANTIQ_REFLEX_MAX_DEMOTE_PER_WINDOW"] == "2"
    assert decode_env["SEMANTIQ_REFLEX_LOW_RISK_SCORE_FRACTION"] == "0.25"
    assert decode_env["SEMANTIQ_REFLEX_PAGE_SELECTION_POLICY"] == "relevance_sparse"
    assert decode_env["SEMANTIQ_REFLEX_COLD_ADMISSION_MAX_INT4_FRACTION"] == "0.1"
    assert decode_env["SEMANTIQ_REFLEX_COLD_ADMISSION_EMERGENCY_FREE_RATIO"] == "0.05"
    assert decode_env["SEMANTIQ_REFLEX_DECODE_PRESSURE_WARMUP_TOKENS"] == "32"
    assert decode_env["SEMANTIQ_REFLEX_DECODE_PRESSURE_RAMP_TOKENS"] == "512"
    assert decode_env["SEMANTIQ_REFLEX_SHORT_PREFILL_PAGES"] == "64"
    assert decode_env["SEMANTIQ_REFLEX_LONG_PREFILL_PAGES"] == "512"
    assert decode_env["SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_MIN_PROMPT_PAGES"] == "512"
    assert decode_env["SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_MIN_DECODE_TOKENS"] == "129"
    assert (
        decode_env["SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_LANDING_MAX_INT4_FRACTION"]
        == "0.08"
    )
    assert decode_env["SEMANTIQ_REFLEX_SLO_PRESSURE_STEP"] == "0.25"
    assert decode_env["SEMANTIQ_REFLEX_MIN_SLO_PRESSURE"] == "0.5"
    assert decode_env["SEMANTIQ_REFLEX_MAX_SLO_PRESSURE"] == "1.5"
    assert str(runner.ROOT / "vllm") in decode_env["PYTHONPATH"]

    assert str(runner.PROXY_SCRIPT) in proxy_cmd
    assert proxy_cmd[proxy_cmd.index("--prefill") + 1] == "http://127.0.0.1:8310"
    assert proxy_cmd[proxy_cmd.index("--prefill") + 2] == "8998"
    assert proxy_cmd[proxy_cmd.index("--decode") + 1] == "http://127.0.0.1:8410"
    assert proxy_cmd[proxy_cmd.index("--port") + 1] == "8510"
    assert proxy_cmd[proxy_cmd.index("--prefill-max-inflight") + 1] == "2"
    assert (
        proxy_cmd[proxy_cmd.index("--prefill-metadata-wait-timeout-sec") + 1]
        == "5.0"
    )
    assert (
        proxy_cmd[proxy_cmd.index("--decode-backpressure-policy") + 1]
        == "metrics"
    )
    assert (
        proxy_cmd[proxy_cmd.index("--decode-backpressure-max-kv-usage") + 1]
        == "0.88"
    )
    assert (
        proxy_cmd[proxy_cmd.index("--decode-backpressure-max-waiting") + 1]
        == "1"
    )
    assert (
        proxy_cmd[proxy_cmd.index("--decode-backpressure-waiting-policy") + 1]
        == "adaptive"
    )
    assert (
        proxy_cmd[proxy_cmd.index("--decode-backpressure-adaptive-max-waiting") + 1]
        == "6"
    )
    assert (
        proxy_cmd[
            proxy_cmd.index("--decode-backpressure-adaptive-kv-headroom-per-waiting")
            + 1
        ]
        == "0.03"
    )
    assert (
        proxy_cmd[proxy_cmd.index("--decode-backpressure-admission-settle-sec") + 1]
        == "1.5"
    )

    assert bench_cmd[bench_cmd.index("--base-url") + 1] == "http://127.0.0.1:8510"
    assert bench_cmd[bench_cmd.index("--endpoint") + 1] == "/v1/completions"
    assert "--no-stream" in bench_cmd


def test_1p1d_rejects_same_gpu_for_prefill_and_decode(tmp_path):
    args = _args(tmp_path)
    args.decode_gpu = args.prefill_gpu

    with pytest.raises(ValueError, match="different GPUs"):
        runner.validate_args(args)


def test_1p1d_proxy_metadata_wait_defaults_to_zero_without_reflex_metadata(tmp_path):
    args = _args(tmp_path)
    args.decode_kv_cache_dtype = "auto"

    proxy_cmd = runner.build_proxy_cmd(args)

    assert (
        proxy_cmd[proxy_cmd.index("--prefill-metadata-wait-timeout-sec") + 1]
        == "0.0"
    )

    args.decode_kv_cache_dtype = "reflex_int4"
    args.disable_reflex_prefill_page_metadata = True
    proxy_cmd = runner.build_proxy_cmd(args)

    assert (
        proxy_cmd[proxy_cmd.index("--prefill-metadata-wait-timeout-sec") + 1]
        == "0.0"
    )
