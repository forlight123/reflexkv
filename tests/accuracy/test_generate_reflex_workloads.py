import argparse
import json
from pathlib import Path

from scripts.accuracy import generate_reflex_workloads as generator


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        tasks="longbench,reasoning",
        longbench_datasets="gov_report",
        reasoning_datasets="math500",
        longbench_data_dir=str(tmp_path / "longbench"),
        reasoning_data_dir=str(tmp_path / "reasoning"),
        config_dir=str(tmp_path / "config"),
        output=str(tmp_path / "workload.jsonl"),
        summary_out=str(tmp_path / "workload_summary.json"),
        longbench_max_samples=2,
        reasoning_max_samples=2,
        max_model_len=32768,
        prompt_fit_policy="none",
        prompt_fit_token_margin=8,
        workload_mix_policy="balanced",
        seed=3,
        slo_classes="high,normal",
        slo_priorities="-1,0",
        skip_chat_template=True,
    )


def _write_fixture_data(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir)
    _write_json(config_dir / "dataset2samples.json", {"gov_report": 2})
    _write_json(config_dir / "dataset2prompt.json", {"gov_report": "Doc: {context}\nSummary:"})
    _write_json(config_dir / "dataset2maxlen.json", {"gov_report": 16})
    _write_json(config_dir / "dataset2metric.json", {"gov_report": "rouge"})
    _write_json(config_dir / "reasoning_dataset2samples.json", {"math500": 2})
    _write_json(
        config_dir / "reasoning_dataset2prompt.json",
        {"math500": "Solve: {problem}\nAnswer:"},
    )
    _write_json(config_dir / "reasoning_dataset2maxlen.json", {"math500": 32})
    _write_json(config_dir / "reasoning_dataset2metric.json", {"math500": "boxed_accuracy"})
    _write_jsonl(
        Path(args.longbench_data_dir) / "gov_report.jsonl",
        [
            {"context": "report-a", "answers": ["summary-a"], "all_classes": None},
            {"context": "report-b", "answers": ["summary-b"], "all_classes": []},
        ],
    )
    _write_jsonl(
        Path(args.reasoning_data_dir) / "math500.jsonl",
        [
            {"problem": "1+1", "answer": "2", "id": "m0"},
            {"problem": "2+2", "answer": "4", "id": "m1"},
        ],
    )


def test_generate_reflex_workload_manifest_writes_interleaved_jsonl_and_summary(tmp_path):
    args = _args(tmp_path)
    _write_fixture_data(args)

    summary = generator.write_workload_manifest(args)

    rows = [
        json.loads(line)
        for line in Path(args.output).read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 4
    assert {row["dataset"] for row in rows} == {"gov_report", "math500"}
    assert {row["task"] for row in rows} == {"longbench", "reasoning"}
    assert all("prompt" in row and row["prompt"] for row in rows)
    assert all("answers" in row for row in rows)
    assert rows[0]["request_index"] == 0
    assert summary["total_requests"] == 4
    assert summary["datasets"]["gov_report"]["requests"] == 2
    assert summary["datasets"]["math500"]["max_new_tokens"] == 32
    persisted = json.loads(Path(args.summary_out).read_text(encoding="utf-8"))
    assert persisted == summary


def test_generate_reflex_workload_cli_supports_prompt_fit_tokenizer_model(
    tmp_path,
    monkeypatch,
):
    class FakeTokenizer:
        def encode(self, prompt, **_kwargs):
            return [ord(ch) for ch in prompt]

        def decode(self, token_ids, **_kwargs):
            return "".join(chr(token_id) for token_id in token_ids)

    args = generator.parse_args(
        [
            "--tasks",
            "longbench,reasoning",
            "--longbench-datasets",
            "gov_report",
            "--reasoning-datasets",
            "math500",
            "--longbench-data-dir",
            str(tmp_path / "longbench"),
            "--reasoning-data-dir",
            str(tmp_path / "reasoning"),
            "--config-dir",
            str(tmp_path / "config"),
            "--output",
            str(tmp_path / "workload.jsonl"),
            "--summary-out",
            str(tmp_path / "workload_summary.json"),
            "--longbench-max-samples",
            "2",
            "--reasoning-max-samples",
            "2",
            "--prompt-fit-policy",
            "truncate",
            "--max-model-len",
            "64",
            "--model",
            "/models/unit-test",
        ]
    )
    _write_fixture_data(args)

    def fake_loader(loader_args):
        assert loader_args.model == "/models/unit-test"
        return FakeTokenizer()

    monkeypatch.setattr(
        generator.mixed.single_runner,
        "_load_prompt_fit_tokenizer",
        fake_loader,
    )

    summary = generator.write_workload_manifest(args)

    assert summary["total_requests"] == 4
