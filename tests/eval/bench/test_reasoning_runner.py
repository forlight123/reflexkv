import json

import pytest

from eval.backends import register_backend
from eval.backends.base import BaseBackend
from eval.bench.reasoning import main as run_main
from eval.bench.reasoning import parse_args


class FakeBackend(BaseBackend):
    name = "fake-reasoning"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--fake-flag", default="fake-default")
        return parser

    def build(self, args):
        self.fake_flag = args.fake_flag

    def get_prompt_formatter(self):
        return lambda prompt: f"FMT::{prompt}"

    def generate(self, records, gen_config):
        assert gen_config.max_new_tokens == 128
        outputs = []
        for record in records:
            outputs.append(
                {
                    "pred": f"work\\boxed{{{record.answers[0]}}}",
                    "meta": {
                        "fake_flag": self.fake_flag,
                        "source_meta": record.meta,
                    },
                }
            )
        return outputs


class ShortBatchBackend(FakeBackend):
    name = "fake-reasoning-short"

    def generate(self, records, gen_config):
        outputs = super().generate(records, gen_config)
        if len(records) > 1:
            return outputs[:-1]
        return outputs


class OverlongBatchBackend(FakeBackend):
    name = "fake-reasoning-overlong"

    def generate(self, records, gen_config):
        outputs = super().generate(records, gen_config)
        if len(records) > 1:
            return outputs + [outputs[-1]]
        return outputs


class CountingBackend(FakeBackend):
    name = "fake-reasoning-counting"
    build_calls = 0

    def build(self, args):
        type(self).build_calls += 1
        super().build(args)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _prepare_fixture(tmp_path):
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    config_dir.mkdir()
    data_dir.mkdir()

    _write_json(
        config_dir / "reasoning_dataset2prompt.json",
        {"math500": "{problem}\n\nPlease reason step by step and put your final answer within \\boxed{}."},
    )
    _write_json(config_dir / "reasoning_dataset2maxlen.json", {"math500": 128})
    _write_json(config_dir / "reasoning_dataset2samples.json", {"math500": 2})
    _write_json(config_dir / "reasoning_dataset2metric.json", {"math500": "boxed_accuracy"})
    _write_jsonl(
        data_dir / "math500.jsonl",
        [
            {"problem": "2+2", "answer": "4", "unique_id": "m1"},
            {"problem": "3+3", "answer": "6", "unique_id": "m2"},
        ],
    )
    return config_dir, data_dir, output_dir


def test_parse_args_accepts_backend_specific_flags():
    register_backend(FakeBackend)

    args = parse_args(
        [
            "--backend",
            "fake-reasoning",
            "--dataset",
            "math500",
            "--data-dir",
            "/tmp/data",
            "--output-dir",
            "/tmp/out",
            "--fake-flag",
            "x",
        ]
    )

    assert args.backend == "fake-reasoning"
    assert args.fake_flag == "x"


def test_runner_writes_reasoning_outputs(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    run_main(
        [
            "--backend",
            "fake-reasoning",
            "--dataset",
            "math500",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "smoke",
            "--fake-flag",
            "flagged",
        ]
    )

    save_dir = output_dir / "smoke" / "math500"
    pred_file = save_dir / "pred.jsonl"
    result_file = save_dir / "result.json"
    run_config_file = save_dir / "run_config.json"
    run_summary_file = save_dir / "run_summary.json"
    badcases_file = save_dir / "badcases.jsonl"

    assert pred_file.exists()
    assert result_file.exists()
    assert run_config_file.exists()
    assert run_summary_file.exists()
    assert badcases_file.exists()

    predictions = _read_jsonl(pred_file)
    assert len(predictions) == 2
    assert predictions[0]["pred"] == "work\\boxed{4}"
    assert predictions[0]["answers"] == ["4"]
    assert predictions[0]["meta"]["fake_flag"] == "flagged"
    assert predictions[0]["meta"]["source_meta"]["unique_id"] == "m1"

    result_payload = json.loads(result_file.read_text(encoding="utf-8"))
    run_config_payload = json.loads(run_config_file.read_text(encoding="utf-8"))
    run_summary_payload = json.loads(run_summary_file.read_text(encoding="utf-8"))
    badcases_rows = _read_jsonl(badcases_file)

    assert result_payload["avg_score"] == 1.0
    assert run_config_payload["backend"] == "fake-reasoning"
    assert run_config_payload["fake_flag"] == "flagged"
    assert run_summary_payload["completed_predictions"] == 2
    assert run_summary_payload["failed_predictions"] == 0
    assert run_summary_payload["avg_score"] == 1.0
    assert badcases_rows == []


def test_runner_resume_skips_completed_rows_and_truncates_invalid_tail(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)
    save_dir = output_dir / "resume" / "math500"
    save_dir.mkdir(parents=True)
    pred_file = save_dir / "pred.jsonl"
    pred_file.write_text(
        json.dumps({"pred": "work\\boxed{4}", "answers": ["4"], "all_classes": [], "meta": {}})
        + "\n"
        + "{",
        encoding="utf-8",
    )

    run_main(
        [
            "--backend",
            "fake-reasoning",
            "--dataset",
            "math500",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "resume",
            "--resume",
        ]
    )

    predictions = _read_jsonl(pred_file)
    assert len(predictions) == 2
    assert predictions[0]["pred"] == "work\\boxed{4}"
    assert predictions[1]["pred"] == "work\\boxed{6}"


def test_runner_resume_truncates_stale_predictions_beyond_requested_samples(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)
    save_dir = output_dir / "resume-shorter" / "math500"
    save_dir.mkdir(parents=True)
    pred_file = save_dir / "pred.jsonl"
    _write_jsonl(
        pred_file,
        [
            {"pred": "work\\boxed{4}", "answers": ["4"], "all_classes": [], "meta": {}},
            {"pred": "work\\boxed{6}", "answers": ["6"], "all_classes": [], "meta": {}},
        ],
    )

    run_main(
        [
            "--backend",
            "fake-reasoning",
            "--dataset",
            "math500",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
            "--run-name",
            "resume-shorter",
            "--resume",
            "--max-samples",
            "1",
        ]
    )

    predictions = _read_jsonl(pred_file)
    run_summary = json.loads((save_dir / "run_summary.json").read_text(encoding="utf-8"))
    result_payload = json.loads((save_dir / "result.json").read_text(encoding="utf-8"))

    assert len(predictions) == 1
    assert predictions[0]["pred"] == "work\\boxed{4}"
    assert run_summary["requested_samples"] == 1
    assert run_summary["completed_predictions"] == 1
    assert run_summary["resumed_predictions"] == 1
    assert result_payload["total_samples"] == 1


def test_runner_falls_back_when_batch_returns_too_few_outputs(tmp_path):
    register_backend(ShortBatchBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    run_main(
        [
            "--backend",
            "fake-reasoning-short",
            "--dataset",
            "math500",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
        ]
    )

    predictions = _read_jsonl(output_dir / "math500" / "pred.jsonl")
    assert [item["pred"] for item in predictions] == ["work\\boxed{4}", "work\\boxed{6}"]


def test_runner_falls_back_when_batch_returns_too_many_outputs(tmp_path):
    register_backend(OverlongBatchBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    run_main(
        [
            "--backend",
            "fake-reasoning-overlong",
            "--dataset",
            "math500",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
            "--batch-size",
            "2",
        ]
    )

    predictions = _read_jsonl(output_dir / "math500" / "pred.jsonl")
    assert [item["pred"] for item in predictions] == ["work\\boxed{4}", "work\\boxed{6}"]


def test_main_rejects_non_positive_batch_size(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    with pytest.raises(ValueError, match="--batch-size must be positive"):
        run_main(
            [
                "--backend",
                "fake-reasoning",
                "--dataset",
                "math500",
                "--data-dir",
                str(data_dir),
                "--config-dir",
                str(config_dir),
                "--output-dir",
                str(output_dir),
                "--batch-size",
                "0",
            ]
        )


def test_runner_defaults_answers_when_backend_omits_them(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    run_main(
        [
            "--backend",
            "fake-reasoning",
            "--dataset",
            "math500",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    predictions = _read_jsonl(output_dir / "math500" / "pred.jsonl")
    assert predictions[0]["answers"] == ["4"]
    assert predictions[1]["answers"] == ["6"]


def test_main_fails_fast_when_metric_mapping_is_missing(tmp_path):
    register_backend(CountingBackend)
    CountingBackend.build_calls = 0
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)
    _write_json(config_dir / "reasoning_dataset2metric.json", {"other": "boxed_accuracy"})

    with pytest.raises(KeyError, match="math500"):
        run_main(
            [
                "--backend",
                "fake-reasoning-counting",
                "--dataset",
                "math500",
                "--data-dir",
                str(data_dir),
                "--config-dir",
                str(config_dir),
                "--output-dir",
                str(output_dir),
            ]
        )

    assert CountingBackend.build_calls == 0
