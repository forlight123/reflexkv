import argparse
import json
import sys
from pathlib import Path

from scripts.profiling import run_reflex_pd_pressure_sweep as sweep


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/llama",
        host="127.0.0.1",
        prefill_gpu="0",
        decode_gpu="1",
        base_port=9000,
        port_stride=20,
        output_root=str(tmp_path),
        max_model_len=4096,
        gpu_memory_utilization=0.8,
        max_num_batched_tokens=4096,
        max_num_seqs=99,
        request_rates="0.25,0.5",
        dataset_name="random",
        dataset_path=None,
        seed=123,
        enable_reflex_trace=True,
        force_triton_attn=True,
        enforce_eager=True,
        skip_chat_template=True,
        no_stream=True,
        p2p_kv_chunk_blocks=16,
        p2p_max_staged_bytes=1048576,
        input_lens="2048,4096",
        output_lens="32",
        concurrencies="4",
        num_prompts_list="16,32",
        decode_dtypes="auto,reflex_int4",
        limit=None,
        dry_run=True,
        commands_out=None,
    )


def _value_after(cmd: list[str], flag: str) -> str:
    return cmd[cmd.index(flag) + 1]


def test_enumerates_reduced_matrix_in_deterministic_order(tmp_path):
    args = _args(tmp_path)

    points = sweep.enumerate_points(args)

    assert len(points) == 16
    assert points[0] == sweep.SweepPoint(
        index=0,
        decode_dtype="auto",
        input_len=2048,
        output_len=32,
        request_rate="0.25",
        max_concurrency=4,
        num_prompts=16,
    )
    assert points[1].num_prompts == 32
    assert points[2].request_rate == "0.5"
    assert points[8].decode_dtype == "reflex_int4"


def test_build_command_contains_expected_single_run_args_and_run_name(tmp_path):
    args = _args(tmp_path)
    point = sweep.enumerate_points(args)[8]

    command = sweep.build_command(args, point)

    assert command[:2] == [sys.executable, str(sweep.SINGLE_RUN_SCRIPT)]
    assert _value_after(command, "--model") == "/models/llama"
    assert _value_after(command, "--prefill-gpu") == "0"
    assert _value_after(command, "--decode-gpu") == "1"
    assert _value_after(command, "--output-root") == str(tmp_path)
    assert _value_after(command, "--decode-kv-cache-dtype") == "reflex_int4"
    assert "--num-gpu-blocks-override" not in command
    assert _value_after(command, "--input-len") == "2048"
    assert _value_after(command, "--output-len") == "32"
    assert _value_after(command, "--request-rate") == "0.25"
    assert _value_after(command, "--max-concurrency") == "4"
    assert _value_after(command, "--num-prompts") == "16"
    assert _value_after(command, "--max-num-seqs") == "99"
    assert _value_after(command, "--run-name") == (
        "pd1p1d_decode-reflex_int4_i2048_o32_c4_np16_r0p25"
    )
    assert "--enable-reflex-trace" in command
    assert "--force-triton-attn" in command
    assert "--enforce-eager" in command
    assert "--skip-chat-template" in command
    assert "--no-stream" in command


def test_build_command_offsets_all_ports_by_point_index(tmp_path):
    args = _args(tmp_path)
    point = sweep.enumerate_points(args)[3]

    command = sweep.build_command(args, point)

    assert _value_after(command, "--prefill-port") == "9060"
    assert _value_after(command, "--decode-port") == "9061"
    assert _value_after(command, "--proxy-port") == "9062"
    assert _value_after(command, "--kv-proxy-port") == "9063"
    assert _value_after(command, "--prefill-kv-port") == "9064"
    assert _value_after(command, "--decode-kv-port") == "9065"


def test_rejects_port_stride_that_can_collide_between_points(tmp_path):
    args = _args(tmp_path)
    args.port_stride = 5

    try:
        sweep.build_command(args, sweep.enumerate_points(args)[0])
    except ValueError as exc:
        assert "port_stride" in str(exc)
    else:
        raise AssertionError("expected port_stride validation to fail")


def test_dry_run_prints_and_writes_commands_without_subprocess(tmp_path, monkeypatch, capsys):
    args = _args(tmp_path)
    args.limit = 2
    args.commands_out = str(tmp_path / "commands.jsonl")
    calls = []
    monkeypatch.setattr(sweep.subprocess, "run", lambda *a, **kw: calls.append((a, kw)))

    exit_code = sweep.run_sweep(args)

    assert exit_code == 0
    assert calls == []
    output_lines = capsys.readouterr().out.strip().splitlines()
    assert len(output_lines) == 2
    records = [
        json.loads(line)
        for line in Path(args.commands_out).read_text(encoding="utf-8").splitlines()
    ]
    assert [record["point"]["index"] for record in records] == [0, 1]
    assert records[0]["command"] == sweep.build_command(args, sweep.enumerate_points(args)[0])


def test_limit_truncates_executed_points(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.limit = 3
    args.dry_run = False
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return argparse.Namespace(returncode=0)

    monkeypatch.setattr(sweep.subprocess, "run", fake_run)

    exit_code = sweep.run_sweep(args)

    assert exit_code == 0
    assert len(calls) == 3
    assert [call[0] for call in calls] == [
        sweep.build_command(args, point) for point in sweep.enumerate_points(args)[:3]
    ]
    assert all(call[1]["cwd"] == sweep.ROOT for call in calls)
