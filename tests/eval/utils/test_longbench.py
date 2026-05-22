import json

import pytest

from eval.utils.longbench import (
    build_prompt_records,
    resolve_dataset_max_samples,
    resolve_longbench_datasets,
    should_use_chat_format,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_resolve_longbench_datasets_returns_supported_intersection(tmp_path):
    _write_json(
        tmp_path / "dataset2prompt.json",
        {"narrativeqa": "prompt", "qasper": "prompt", "unused": "prompt"},
    )
    _write_json(
        tmp_path / "dataset2maxlen.json",
        {"narrativeqa": 32, "qasper": 64},
    )
    _write_json(
        tmp_path / "dataset2samples.json",
        {"narrativeqa": 3, "qasper": 2, "orphan": 1},
    )

    assert resolve_longbench_datasets("all", str(tmp_path)) == [
        "narrativeqa",
        "qasper",
    ]


def test_resolve_longbench_datasets_rejects_unknown_dataset(tmp_path):
    _write_json(tmp_path / "dataset2prompt.json", {"qasper": "prompt"})
    _write_json(tmp_path / "dataset2maxlen.json", {"qasper": 32})
    _write_json(tmp_path / "dataset2samples.json", {"qasper": 1})

    with pytest.raises(ValueError, match="Unsupported LongBench dataset"):
        resolve_longbench_datasets("gov_report", str(tmp_path))


def test_build_prompt_records_formats_template_fields():
    rows = build_prompt_records(
        [{"context": "ctx", "input": "question", "answers": ["answer"]}],
        "Question: {input}\nContext: {context}",
        chat_formatter=None,
    )

    assert rows == [
        {
            "prompt": "Question: question\nContext: ctx",
            "answers": ["answer"],
            "all_classes": [],
        }
    ]


def test_build_prompt_records_uses_chat_formatter_when_present():
    rows = build_prompt_records(
        [{"context": "ctx", "input": "question", "answers": ["answer"]}],
        "Question: {input}\nContext: {context}",
        chat_formatter=lambda prompt: f"CHAT::{prompt}",
    )

    assert rows[0]["prompt"] == "CHAT::Question: question\nContext: ctx"


def test_resolve_dataset_max_samples_applies_cli_limit(tmp_path):
    _write_json(tmp_path / "dataset2samples.json", {"2wikimqa": 200})

    assert resolve_dataset_max_samples("2wikimqa", str(tmp_path), max_samples=7) == 7


def test_resolve_dataset_max_samples_uses_dataset_default_when_cli_limit_missing(tmp_path):
    _write_json(tmp_path / "dataset2samples.json", {"2wikimqa": 200})

    assert resolve_dataset_max_samples("2wikimqa", str(tmp_path), max_samples=None) == 200


def test_should_use_chat_format_disables_known_non_chat_datasets():
    assert should_use_chat_format("trec") is False
    assert should_use_chat_format("samsum") is False
    assert should_use_chat_format("2wikimqa") is True
