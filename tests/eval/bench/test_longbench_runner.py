import json
from enum import Enum

from eval.backends import register_backend
from eval.backends.base import BaseBackend
from eval.bench.longbench import main as run_main
from eval.bench.longbench import parse_args


class FakeBackend(BaseBackend):
    name = "fake"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--fake-flag", default="fake-default")
        return parser

    def build(self, args):
        self.fake_flag = args.fake_flag

    def get_prompt_formatter(self):
        return lambda prompt: f"FMT::{prompt}"

    def generate(self, records, gen_config):
        del gen_config
        outputs = []
        for record in records:
            pred = record.answers[0] if record.prompt.startswith("FMT::") else "wrong"
            outputs.append(
                {
                    "pred": pred,
                    "answers": record.answers,
                    "all_classes": record.all_classes,
                    "meta": {"fake_flag": self.fake_flag},
                }
            )
        return outputs


class SeededBackend(BaseBackend):
    name = "seeded"

    @classmethod
    def add_cli_args(cls, parser):
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--seeded-flag", default="seeded-default")
        return parser

    def build(self, args):
        self.seed = args.seed
        self.seeded_flag = args.seeded_flag

    def generate(self, records, gen_config):
        del records, gen_config
        return []


class ExampleBackendEnum(Enum):
    MAMBA = "mamba"


class EnumBackend(FakeBackend):
    name = "enumy"

    @classmethod
    def add_cli_args(cls, parser):
        parser = super().add_cli_args(parser)
        parser.add_argument(
            "--enum-flag",
            type=lambda value: ExampleBackendEnum(value),
            default=ExampleBackendEnum.MAMBA,
        )
        return parser

    def build(self, args):
        super().build(args)
        self.enum_flag = args.enum_flag


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prepare_fixture(tmp_path):
    config_dir = tmp_path / "config"
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    config_dir.mkdir()
    data_dir.mkdir()

    _write_json(config_dir / "dataset2prompt.json", {"2wikimqa": "Question: {input}\nContext: {context}"})
    _write_json(config_dir / "dataset2maxlen.json", {"2wikimqa": 32})
    _write_json(config_dir / "dataset2samples.json", {"2wikimqa": 2})
    _write_json(config_dir / "dataset2metric.json", {"2wikimqa": "qa_f1"})
    _write_jsonl(
        data_dir / "2wikimqa.jsonl",
        [
            {"input": "Who?", "context": "Alice went home.", "answers": ["Alice"]},
            {"input": "Where?", "context": "Bob stayed in Paris.", "answers": ["Paris"]},
        ],
    )
    return config_dir, data_dir, output_dir


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_runner_writes_pred_result_and_run_metadata(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    run_main(
        [
            "--backend",
            "fake",
            "--dataset",
            "2wikimqa",
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

    save_dir = output_dir / "smoke" / "2wikimqa"
    pred_file = save_dir / "pred.jsonl"
    result_file = save_dir / "result.json"
    run_config_file = save_dir / "run_config.json"
    run_summary_file = save_dir / "run_summary.json"

    assert pred_file.exists()
    assert result_file.exists()
    assert run_config_file.exists()
    assert run_summary_file.exists()

    predictions = _read_jsonl(pred_file)
    assert len(predictions) == 2
    assert predictions[0]["pred"] == "Alice"

    result_payload = json.loads(result_file.read_text(encoding="utf-8"))
    run_config_payload = json.loads(run_config_file.read_text(encoding="utf-8"))
    run_summary_payload = json.loads(run_summary_file.read_text(encoding="utf-8"))

    assert result_payload["avg_score"] == 1.0
    assert run_config_payload["backend"] == "fake"
    assert run_config_payload["fake_flag"] == "flagged"
    assert run_summary_payload["completed_predictions"] == 2
    assert run_summary_payload["failed_predictions"] == 0


def test_runner_resume_skips_completed_rows_and_truncates_invalid_tail(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)
    save_dir = output_dir / "resume" / "2wikimqa"
    save_dir.mkdir(parents=True)
    pred_file = save_dir / "pred.jsonl"
    pred_file.write_text(
        json.dumps({"pred": "Alice", "answers": ["Alice"], "all_classes": []}) + "\n" + "{",
        encoding="utf-8",
    )

    run_main(
        [
            "--backend",
            "fake",
            "--dataset",
            "2wikimqa",
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
    assert predictions[0]["pred"] == "Alice"
    assert predictions[1]["pred"] == "Paris"


def test_parse_args_accepts_backend_specific_flags():
    register_backend(FakeBackend)

    args = parse_args(
        [
            "--backend",
            "fake",
            "--dataset",
            "2wikimqa",
            "--data-dir",
            "/tmp/data",
            "--output-dir",
            "/tmp/out",
            "--run-name",
            "smoke",
            "--fake-flag",
            "x",
        ]
    )

    assert args.backend == "fake"
    assert args.fake_flag == "x"


def test_parse_args_allows_omitting_run_name():
    register_backend(FakeBackend)

    args = parse_args(
        [
            "--backend",
            "fake",
            "--dataset",
            "2wikimqa",
            "--data-dir",
            "/tmp/data",
            "--output-dir",
            "/tmp/out",
            "--fake-flag",
            "x",
        ]
    )

    assert args.backend == "fake"
    assert args.run_name is None
    assert args.fake_flag == "x"


def test_runner_writes_to_dataset_dir_when_run_name_omitted(tmp_path):
    register_backend(FakeBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    run_main(
        [
            "--backend",
            "fake",
            "--dataset",
            "2wikimqa",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    save_dir = output_dir / "2wikimqa"
    assert (save_dir / "pred.jsonl").exists()
    assert (save_dir / "result.json").exists()
    assert (save_dir / "run_config.json").exists()
    assert (save_dir / "run_summary.json").exists()


def test_runner_serializes_enum_args_in_run_config(tmp_path):
    register_backend(EnumBackend)
    config_dir, data_dir, output_dir = _prepare_fixture(tmp_path)

    run_main(
        [
            "--backend",
            "enumy",
            "--dataset",
            "2wikimqa",
            "--data-dir",
            str(data_dir),
            "--config-dir",
            str(config_dir),
            "--output-dir",
            str(output_dir),
        ]
    )

    run_config_payload = json.loads(
        (output_dir / "2wikimqa" / "run_config.json").read_text(encoding="utf-8")
    )
    assert run_config_payload["enum_flag"] == "mamba"


def test_parse_args_keeps_backend_owned_seed_flag():
    register_backend(SeededBackend)

    args = parse_args(
        [
            "--backend",
            "seeded",
            "--dataset",
            "2wikimqa",
            "--data-dir",
            "/tmp/data",
            "--output-dir",
            "/tmp/out",
            "--run-name",
            "smoke",
            "--seed",
            "13",
            "--seeded-flag",
            "y",
        ]
    )

    assert args.backend == "seeded"
    assert args.seed == 13
    assert args.seeded_flag == "y"
