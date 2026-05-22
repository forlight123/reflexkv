import argparse
import csv
import json
from pathlib import Path

from gen_data import build_trace_driven_manifest as builder


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_burstgpt_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "Timestamp",
                "Model",
                "Request tokens",
                "Response tokens",
                "Total tokens",
                "Log Type",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Timestamp": "10",
                    "Model": "ChatGPT",
                    "Request tokens": "100",
                    "Response tokens": "20",
                    "Total tokens": "120",
                    "Log Type": "API log",
                },
                {
                    "Timestamp": "11",
                    "Model": "GPT-4",
                    "Request tokens": "3000",
                    "Response tokens": "900",
                    "Total tokens": "3900",
                    "Log Type": "Conversation log",
                },
                {
                    "Timestamp": "12",
                    "Model": "ChatGPT",
                    "Request tokens": "50",
                    "Response tokens": "0",
                    "Total tokens": "50",
                    "Log Type": "API log",
                },
                {
                    "Timestamp": "14",
                    "Model": "ChatGPT",
                    "Request tokens": "7000",
                    "Response tokens": "100",
                    "Total tokens": "7100",
                    "Log Type": "API log",
                },
                {
                    "Timestamp": "18",
                    "Model": "GPT-4",
                    "Request tokens": "1000",
                    "Response tokens": "2500",
                    "Total tokens": "3500",
                    "Log Type": "API log",
                },
                {
                    "Timestamp": "20",
                    "Model": "ChatGPT",
                    "Request tokens": "200",
                    "Response tokens": "50",
                    "Total tokens": "250",
                    "Log Type": "API log",
                },
            ]
        )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        trace_csv=str(tmp_path / "BurstGPT_1.csv"),
        output_dir=str(tmp_path / "out"),
        output_prefix="paper_mix",
        num_requests=4,
        start_index=0,
        include_failed=False,
        seed=13,
        time_scale=0.1,
        max_interarrival_sec=None,
        long_input_ratio=0.5,
        long_output_ratio=0.5,
        long_input_tokens=2048,
        long_output_tokens=512,
        tasks="longbench,reasoning",
        longbench_datasets="qasper",
        reasoning_datasets="math500",
        longbench_data_dir=str(tmp_path / "longbench"),
        reasoning_data_dir=str(tmp_path / "reasoning"),
        config_dir=str(tmp_path / "config"),
        longbench_max_samples=2,
        reasoning_max_samples=2,
        bench_mix="qasper=0.5,math500=0.5",
        model="/models/unit-test",
        max_model_len=32768,
        prompt_fit_policy="none",
        prompt_fit_token_margin=8,
        workload_mix_policy="balanced",
        slo_classes="high,normal",
        slo_priorities="-1,0",
        skip_chat_template=True,
    )


def _write_fixture_benchmarks(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir)
    _write_json(config_dir / "dataset2samples.json", {"qasper": 2})
    _write_json(config_dir / "dataset2prompt.json", {"qasper": "Q: {input}\nC: {context}\nA:"})
    _write_json(config_dir / "dataset2maxlen.json", {"qasper": 16})
    _write_json(config_dir / "dataset2metric.json", {"qasper": "qa_f1"})
    _write_json(config_dir / "reasoning_dataset2samples.json", {"math500": 2})
    _write_json(
        config_dir / "reasoning_dataset2prompt.json",
        {"math500": "Solve: {problem}\nAnswer:"},
    )
    _write_json(config_dir / "reasoning_dataset2maxlen.json", {"math500": 64})
    _write_json(config_dir / "reasoning_dataset2metric.json", {"math500": "boxed_accuracy"})
    _write_jsonl(
        Path(args.longbench_data_dir) / "qasper.jsonl",
        [
            {"input": "q0", "context": "ctx0", "answers": ["a0"], "all_classes": []},
            {"input": "q1", "context": "ctx1", "answers": ["a1"], "all_classes": []},
        ],
    )
    _write_jsonl(
        Path(args.reasoning_data_dir) / "math500.jsonl",
        [
            {"problem": "1+1", "answer": "2", "id": "m0"},
            {"problem": "2+2", "answer": "4", "id": "m1"},
        ],
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_write_trace_driven_manifest_binds_burstgpt_shape_to_answerable_tasks(tmp_path):
    args = _args(tmp_path)
    _write_burstgpt_csv(Path(args.trace_csv))
    _write_fixture_benchmarks(args)

    summary = builder.write_trace_driven_manifest(args)

    out_dir = Path(args.output_dir)
    profile_rows = _read_jsonl(out_dir / "paper_mix_trace_profile.jsonl")
    manifest_rows = _read_jsonl(out_dir / "paper_mix_manifest.jsonl")
    persisted_summary = json.loads(
        (out_dir / "paper_mix_summary.json").read_text(encoding="utf-8")
    )

    assert summary == persisted_summary
    assert len(profile_rows) == 4
    assert len(manifest_rows) == 4
    assert [row["request_index"] for row in manifest_rows] == [0, 1, 2, 3]
    assert manifest_rows[0]["arrival_time_sec"] == 0.0
    assert manifest_rows[1]["scaled_arrival_time_sec"] == 0.1
    assert all(row["trace_response_tokens"] > 0 for row in manifest_rows)
    assert sum(row["trace_request_tokens"] >= 2048 for row in manifest_rows) >= 2
    assert sum(row["trace_response_tokens"] >= 512 for row in manifest_rows) >= 2
    assert {row["dataset"] for row in manifest_rows} == {"qasper", "math500"}
    assert {row["task"] for row in manifest_rows} == {"longbench", "reasoning"}
    assert all(row["prompt"] for row in manifest_rows)
    assert all(isinstance(row["answers"], list) for row in manifest_rows)
    assert summary["total_requests"] == 4
    assert summary["trace"]["selected_failed_response_rows"] == 0
    assert summary["trace"]["long_input_requests"] >= 2
    assert summary["trace"]["long_output_requests"] >= 2
    assert summary["datasets"]["qasper"]["requests"] == 2
    assert summary["datasets"]["math500"]["requests"] == 2
