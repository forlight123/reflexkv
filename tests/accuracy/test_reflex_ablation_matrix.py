import argparse
import json
import sys
from pathlib import Path

from scripts.accuracy import run_reflex_ablation_matrix as matrix


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/llama",
        host="127.0.0.1",
        prefill_gpu="0",
        decode_gpu="1",
        base_port=9100,
        port_stride=20,
        output_root=str(tmp_path / "runs"),
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        block_size=16,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        num_gpu_blocks_override=736,
        proxy_prefill_max_inflight=4,
        proxy_prefill_metadata_wait_timeout_sec=None,
        proxy_decode_backpressure_policy="off",
        proxy_decode_backpressure_max_kv_usage=0.90,
        proxy_decode_backpressure_max_waiting=0,
        proxy_decode_backpressure_waiting_policy="fixed",
        proxy_decode_backpressure_adaptive_max_waiting=4,
        proxy_decode_backpressure_adaptive_kv_headroom_per_waiting=0.04,
        proxy_decode_backpressure_poll_interval_sec=0.05,
        proxy_decode_backpressure_timeout_sec=300.0,
        proxy_decode_backpressure_admission_settle_sec=1.0,
        reflex_remote_chunk_tokens=512,
        max_concurrency=8,
        request_rate="inf",
        workload_manifest=None,
        arrival_policy="poisson",
        trace_time_scale=1.0,
        longbench_max_samples=10,
        reasoning_max_samples=10,
        longbench_datasets="gov_report",
        reasoning_datasets="math500",
        tasks="longbench,reasoning",
        workload_mix_policy="balanced",
        enable_reflex_trace=True,
        force_triton_attn=True,
        enforce_eager=False,
        skip_chat_template=False,
        cases="bf16_baseline,heuristic_reflex,frontier_dual_reflex,direct_landing_off,p_side_risk_off",
        limit=None,
        commands_out=None,
        dry_run=True,
    )


def _value_after(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_ablation_matrix_defaults_to_unbounded_proxy_prefill_inflight():
    args = matrix.parse_args([])

    assert args.proxy_prefill_max_inflight == 0


def test_ablation_matrix_builds_expected_cases_and_env(tmp_path):
    args = _args(tmp_path)

    cases = matrix.selected_cases(args)

    assert [case.name for case in cases] == [
        "bf16_baseline",
        "heuristic_reflex",
        "frontier_dual_reflex",
        "direct_landing_off",
        "p_side_risk_off",
    ]
    assert cases[0].decode_kv_cache_dtype == "auto"
    assert cases[2].page_selection_policy == "frontier_dual"
    assert cases[3].env["SEMANTIQ_REFLEX_ENABLE_DIRECT_INT4_LANDING"] == "0"
    assert cases[4].disable_prefill_page_metadata is True


def test_ablation_command_contains_mixed_accuracy_args_and_case_flags(tmp_path):
    args = _args(tmp_path)
    case = matrix.selected_cases(args)[4]

    command = matrix.build_command(args, case, index=4)

    assert command[:2] == [sys.executable, str(matrix.MIXED_RUN_SCRIPT)]
    assert _value_after(command, "--decode-kv-cache-dtype") == "reflex_int4"
    assert _value_after(command, "--reflex-page-selection-policy") == "frontier_dual"
    assert "--disable-reflex-prefill-page-metadata" in command
    assert _value_after(command, "--run-name").endswith("p_side_risk_off")
    assert _value_after(command, "--longbench-max-samples") == "10"
    assert _value_after(command, "--reasoning-max-samples") == "10"
    assert _value_after(command, "--num-gpu-blocks-override") == "736"
    assert _value_after(command, "--proxy-prefill-metadata-wait-timeout-sec") == "0.0"


def test_bf16_baseline_uses_same_inflight_with_decode_backpressure(tmp_path):
    args = _args(tmp_path)
    case = matrix.selected_cases(args)[0]

    command = matrix.build_command(args, case, index=0)

    assert _value_after(command, "--proxy-prefill-max-inflight") == str(
        args.proxy_prefill_max_inflight
    )
    assert _value_after(command, "--proxy-decode-backpressure-policy") == "metrics"
    assert _value_after(command, "--proxy-decode-backpressure-waiting-policy") == "fixed"
    assert _value_after(command, "--proxy-decode-backpressure-admission-settle-sec") == "1.0"


def test_reflex_cases_enable_adaptive_decode_waiting_backpressure(tmp_path):
    args = _args(tmp_path)
    case = matrix.selected_cases(args)[2]

    command = matrix.build_command(args, case, index=2)

    assert _value_after(command, "--decode-kv-cache-dtype") == "reflex_int4"
    assert _value_after(command, "--proxy-decode-backpressure-policy") == "metrics"
    assert _value_after(command, "--proxy-decode-backpressure-waiting-policy") == "adaptive"
    assert _value_after(command, "--proxy-decode-backpressure-adaptive-max-waiting") == "4"
    assert (
        _value_after(
            command,
            "--proxy-decode-backpressure-adaptive-kv-headroom-per-waiting",
        )
        == "0.04"
    )


def test_ablation_command_passes_fixed_manifest_replay_args(tmp_path):
    args = _args(tmp_path)
    args.workload_manifest = str(tmp_path / "manifest.jsonl")
    args.arrival_policy = "trace"
    args.trace_time_scale = 0.25
    case = matrix.selected_cases(args)[2]

    command = matrix.build_command(args, case, index=2)

    assert _value_after(command, "--workload-manifest") == str(tmp_path / "manifest.jsonl")
    assert _value_after(command, "--arrival-policy") == "trace"
    assert _value_after(command, "--trace-time-scale") == "0.25"


def test_ablation_dry_run_writes_command_records_without_running(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path)
    args.limit = 2
    args.commands_out = str(tmp_path / "commands.jsonl")
    calls = []
    monkeypatch.setattr(matrix.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))

    rc = matrix.run_matrix(args)

    assert rc == 0
    assert calls == []
    assert len(capsys.readouterr().out.strip().splitlines()) == 2
    records = [
        json.loads(line)
        for line in Path(args.commands_out).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["case"]["name"] for record in records] == [
        "bf16_baseline",
        "heuristic_reflex",
    ]
    assert "env" in records[0]


def test_ablation_execution_passes_case_environment(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.cases = "direct_landing_off"
    args.dry_run = False
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(matrix.subprocess, "run", fake_run)

    rc = matrix.run_matrix(args)

    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1]["cwd"] == matrix.ROOT
    assert calls[0][1]["env"]["SEMANTIQ_REFLEX_ENABLE_DIRECT_INT4_LANDING"] == "0"
