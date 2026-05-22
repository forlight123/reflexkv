import argparse
import json
import sys
from pathlib import Path

from scripts.accuracy import run_pd_serving_regression as regression


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/llama",
        host="127.0.0.1",
        prefill_gpu="6",
        decode_gpu="7",
        base_port=9100,
        port_stride=10,
        output_root=str(tmp_path / "runs"),
        tasks="longbench,reasoning",
        longbench_datasets="qasper,hotpotqa",
        reasoning_datasets="math500",
        longbench_data_dir=str(tmp_path / "longbench"),
        reasoning_data_dir=str(tmp_path / "reasoning"),
        config_dir=str(tmp_path / "config"),
        longbench_max_samples=8,
        reasoning_max_samples=4,
        variants="auto,fp8,int4,reflex_int4",
        concurrencies="1,8",
        request_rate="0.5",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        block_size=16,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        mooncake_protocol="rdma",
        mooncake_num_workers=10,
        force_triton_attn=True,
        enforce_eager=True,
        enable_reflex_trace=True,
        reflex_int4_budget_fraction=0.25,
        reflex_keep_initial_blocks=4,
        reflex_keep_recent_blocks=4,
        reflex_max_int4_fraction_per_request=0.5,
        reflex_survival_warmup_tokens=128,
        reflex_risk_warmup_tokens=16,
        reflex_short_admission_max_int4_fraction=0.03,
        reflex_sparse_window_pages=32,
        reflex_short_max_demote_per_window=1,
        reflex_max_demote_per_window=2,
        reflex_low_risk_score_fraction=0.25,
        reflex_page_selection_policy="relevance_sparse",
        reflex_cold_admission_max_int4_fraction=0.10,
        reflex_cold_admission_emergency_free_ratio=0.05,
        proxy_prefill_max_inflight=2,
        reflex_decode_pressure_warmup_tokens=32,
        reflex_decode_pressure_ramp_tokens=512,
        reflex_short_prefill_pages=64,
        reflex_long_prefill_pages=512,
        reflex_slo_pressure_step=0.25,
        reflex_min_slo_pressure=0.5,
        reflex_max_slo_pressure=1.5,
        scheduling_policy="priority",
        temperature=0.0,
        top_p=1.0,
        seed=0,
        slo_classes="high,normal,low",
        slo_priorities="-1,0,1",
        workload_mix_policy="balanced",
        sample_interval_sec=1.0,
        server_ready_timeout_sec=420.0,
        request_timeout_sec=900.0,
        skip_chat_template=False,
        limit=None,
        dry_run=True,
        commands_out=None,
        continue_on_error=False,
    )


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_enumerates_accuracy_matrix_in_deterministic_order(tmp_path):
    args = _args(tmp_path)

    points = regression.enumerate_points(args)

    assert len(points) == 8
    assert points[0] == regression.RegressionPoint(
        index=0,
        variant="auto",
        max_concurrency=1,
        longbench_max_samples=8,
        reasoning_max_samples=4,
    )
    assert points[1].max_concurrency == 8
    assert points[2].variant == "fp8"
    assert points[-1].variant == "reflex_int4"
    assert points[-1].max_concurrency == 8


def test_build_command_contains_pd_accuracy_args_and_reflex_guards(tmp_path):
    args = _args(tmp_path)
    point = [
        point
        for point in regression.enumerate_points(args)
        if point.variant == "reflex_int4"
        and point.max_concurrency == 8
    ][0]

    command = regression.build_command(args, point)

    assert command[:2] == [sys.executable, str(regression.PD_MIXED_ACCURACY_SCRIPT)]
    assert _value_after(command, "--tasks") == "longbench,reasoning"
    assert _value_after(command, "--longbench-datasets") == "qasper,hotpotqa"
    assert _value_after(command, "--reasoning-datasets") == "math500"
    assert _value_after(command, "--decode-kv-cache-dtype") == "reflex_int4"
    assert _value_after(command, "--longbench-max-samples") == "8"
    assert _value_after(command, "--reasoning-max-samples") == "4"
    assert _value_after(command, "--max-concurrency") == "8"
    assert _value_after(command, "--request-rate") == "0.5"
    assert _value_after(command, "--prefill-port") == "9170"
    assert _value_after(command, "--decode-port") == "9171"
    assert _value_after(command, "--proxy-port") == "9172"
    assert _value_after(command, "--prefill-bootstrap-port") == "9173"
    assert _value_after(command, "--reflex-int4-budget-fraction") == "0.25"
    assert _value_after(command, "--reflex-keep-initial-blocks") == "4"
    assert _value_after(command, "--reflex-keep-recent-blocks") == "4"
    assert _value_after(command, "--reflex-max-int4-fraction-per-request") == "0.5"
    assert _value_after(command, "--reflex-survival-warmup-tokens") == "128"
    assert _value_after(command, "--reflex-risk-warmup-tokens") == "16"
    assert _value_after(command, "--reflex-short-admission-max-int4-fraction") == "0.03"
    assert _value_after(command, "--reflex-sparse-window-pages") == "32"
    assert _value_after(command, "--reflex-short-max-demote-per-window") == "1"
    assert _value_after(command, "--reflex-max-demote-per-window") == "2"
    assert _value_after(command, "--reflex-low-risk-score-fraction") == "0.25"
    assert _value_after(command, "--reflex-page-selection-policy") == "relevance_sparse"
    assert "--reflex-cold-admission-max-int4-fraction" not in command
    assert "--reflex-cold-admission-emergency-free-ratio" not in command
    assert _value_after(command, "--proxy-prefill-max-inflight") == "2"
    assert _value_after(command, "--reflex-decode-pressure-warmup-tokens") == "32"
    assert _value_after(command, "--reflex-decode-pressure-ramp-tokens") == "512"
    assert _value_after(command, "--reflex-short-prefill-pages") == "64"
    assert _value_after(command, "--reflex-long-prefill-pages") == "512"
    assert _value_after(command, "--reflex-slo-pressure-step") == "0.25"
    assert _value_after(command, "--scheduling-policy") == "priority"
    assert _value_after(command, "--workload-mix-policy") == "balanced"
    assert "--slo-priorities" not in command
    assert "--slo-priorities=-1,0,1" in command
    assert "--enable-reflex-trace" in command
    assert "--force-triton-attn" in command
    assert "--enforce-eager" in command
    assert _value_after(command, "--run-name") == (
        "pdacc_mixed_longbench+reasoning_qasper+hotpotqa+math500_"
        "kv-reflex_int4_c8_ln8_mn4_r0p5"
    )


def test_build_command_omits_reflex_specific_flags_for_auto(tmp_path):
    args = _args(tmp_path)
    point = regression.enumerate_points(args)[0]

    command = regression.build_command(args, point)

    assert _value_after(command, "--decode-kv-cache-dtype") == "auto"
    assert "--reflex-int4-budget-fraction" not in command
    assert "--reflex-max-int4-fraction-per-request" not in command
    assert "--enable-reflex-trace" not in command


def test_build_command_uses_matching_prefill_dtype_for_non_reflex_variants(tmp_path):
    args = _args(tmp_path)
    point = [
        point
        for point in regression.enumerate_points(args)
        if point.variant == "fp8" and point.max_concurrency == 1
    ][0]

    command = regression.build_command(args, point)

    assert _value_after(command, "--prefill-kv-cache-dtype") == "fp8"
    assert _value_after(command, "--decode-kv-cache-dtype") == "fp8"


def test_build_command_keeps_reflex_prefill_on_auto(tmp_path):
    args = _args(tmp_path)
    point = [
        point
        for point in regression.enumerate_points(args)
        if point.variant == "reflex_int4" and point.max_concurrency == 1
    ][0]

    command = regression.build_command(args, point)

    assert _value_after(command, "--prefill-kv-cache-dtype") == "auto"
    assert _value_after(command, "--decode-kv-cache-dtype") == "reflex_int4"


def test_dry_run_writes_command_manifest_without_subprocess(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path)
    args.limit = 2
    args.commands_out = str(tmp_path / "commands.jsonl")
    calls = []
    monkeypatch.setattr(regression.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))

    exit_code = regression.run_regression(args)

    assert exit_code == 0
    assert calls == []
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert len(output_lines) == 2
    records = [
        json.loads(line)
        for line in Path(args.commands_out).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["point"]["index"] for record in records] == [0, 1]
    assert records[0]["command"] == regression.build_command(
        args,
        regression.enumerate_points(args)[0],
    )


def test_limit_truncates_executed_points(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.limit = 3
    args.dry_run = False
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(regression.subprocess, "run", fake_run)

    exit_code = regression.run_regression(args)

    assert exit_code == 0
    assert len(calls) == 3
    assert [call[0] for call in calls] == [
        regression.build_command(args, point)
        for point in regression.enumerate_points(args)[:3]
    ]
    assert all(call[1]["cwd"] == regression.ROOT for call in calls)
