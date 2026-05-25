import argparse
import csv
import json
from pathlib import Path

from gen_data import build_paper_mixed_manifest as builder


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_burstgpt_csv(path: Path, rows: int = 8) -> None:
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
        for index in range(rows):
            writer.writerow(
                {
                    "Timestamp": str(10 + index),
                    "Model": "ChatGPT",
                    "Request tokens": str(100 + index),
                    "Response tokens": str(20 + index),
                    "Total tokens": str(120 + index),
                    "Log Type": "API log",
                }
            )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        trace_csv=str(tmp_path / "BurstGPT_1.csv"),
        output_dir=str(tmp_path / "out"),
        output_prefix="paper_mix",
        num_requests=8,
        start_index=0,
        include_failed=False,
        seed=3,
        time_scale=0.1,
        max_interarrival_sec=None,
        long_input_ratio=0.0,
        long_output_ratio=0.0,
        long_input_tokens=2048,
        long_output_tokens=512,
        group_mix="longbench=0.25,math500=0.25,gsm8k=0.25,ruler=0.25",
        longbench_datasets="qasper",
        reasoning_datasets="math500,gsm8k",
        ruler_datasets="ruler_niah_single_1_4k",
        longbench_data_dir=str(tmp_path / "longbench"),
        reasoning_data_dir=str(tmp_path / "reasoning"),
        ruler_data_dir=str(tmp_path / "ruler"),
        config_dir=str(tmp_path / "config"),
        longbench_max_samples=2,
        reasoning_max_samples=2,
        ruler_max_samples=2,
        model="/models/unit-test",
        max_model_len=32768,
        slo_classes="high,normal",
        slo_priorities="-1,0",
        skip_chat_template=True,
        prompt_fit_policy="none",
        prompt_fit_token_margin=8,
        workload_mix_policy="balanced",
    )


def _write_fixture_data(args: argparse.Namespace) -> None:
    config = Path(args.config_dir)
    _write_json(config / "dataset2samples.json", {"qasper": 2})
    _write_json(config / "dataset2prompt.json", {"qasper": "Q: {input}\nC: {context}\nA:"})
    _write_json(config / "dataset2maxlen.json", {"qasper": 16})
    _write_json(config / "dataset2metric.json", {"qasper": "qa_f1"})
    _write_json(
        config / "reasoning_dataset2samples.json",
        {"math500": 2, "gsm8k": 2, "ruler_niah_single_1_4k": 2},
    )
    _write_json(
        config / "reasoning_dataset2prompt.json",
        {
            "math500": "Solve: {problem}",
            "gsm8k": "Solve: {problem}",
            "ruler_niah_single_1_4k": "{problem}",
        },
    )
    _write_json(
        config / "reasoning_dataset2maxlen.json",
        {"math500": 64, "gsm8k": 64, "ruler_niah_single_1_4k": 16},
    )
    _write_json(
        config / "reasoning_dataset2metric.json",
        {
            "math500": "boxed_accuracy",
            "gsm8k": "boxed_accuracy",
            "ruler_niah_single_1_4k": "ruler_string_match",
        },
    )
    _write_jsonl(
        Path(args.longbench_data_dir) / "qasper.jsonl",
        [
            {"input": "q0", "context": "ctx0", "answers": ["a0"], "all_classes": []},
            {"input": "q1", "context": "ctx1", "answers": ["a1"], "all_classes": []},
        ],
    )
    for dataset in ("math500", "gsm8k"):
        _write_jsonl(
            Path(args.reasoning_data_dir) / f"{dataset}.jsonl",
            [
                {"problem": f"{dataset} p0", "answer": "0", "id": f"{dataset}/0"},
                {"problem": f"{dataset} p1", "answer": "1", "id": f"{dataset}/1"},
            ],
        )
    _write_jsonl(
        Path(args.ruler_data_dir) / "ruler_niah_single_1_4k.jsonl",
        [
            {
                "problem": "Find hidden number 123.",
                "answers": ["123"],
                "unique_id": "r0",
            },
            {
                "problem": "Find hidden number 456.",
                "answers": ["456"],
                "unique_id": "r1",
            },
        ],
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_write_paper_mixed_manifest_balances_no_aime_groups(tmp_path):
    args = _args(tmp_path)
    _write_burstgpt_csv(Path(args.trace_csv))
    _write_fixture_data(args)

    summary = builder.write_paper_mixed_manifest(args)
    rows = _read_jsonl(Path(args.output_dir) / "paper_mix_manifest.jsonl")

    assert summary["total_requests"] == 8
    assert summary["group_mix"] == {
        "gsm8k": 2,
        "longbench": 2,
        "math500": 2,
        "ruler": 2,
    }
    assert {row["dataset"] for row in rows} == {
        "qasper",
        "math500",
        "gsm8k",
        "ruler_niah_single_1_4k",
    }
    assert {row["dataset"] for row in rows if row["task"] == "ruler"} == {
        "ruler_niah_single_1_4k"
    }
    assert "aime24" not in {row["dataset"] for row in rows}
    assert "aime25" not in {row["dataset"] for row in rows}
