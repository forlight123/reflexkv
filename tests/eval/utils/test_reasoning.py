import json
from pathlib import Path

import pytest

from eval.utils.reasoning import (
    build_reasoning_prompt_records,
    build_reasoning_prompt_context,
    load_reasoning_rows,
    resolve_reasoning_datasets,
    resolve_reasoning_max_samples,
)


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_resolve_reasoning_datasets_returns_supported_intersection(tmp_path):
    _write_json(
        tmp_path / "reasoning_dataset2prompt.json",
        {"math500": "Solve:\n{problem}"},
    )
    _write_json(tmp_path / "reasoning_dataset2maxlen.json", {"math500": 4096})
    _write_json(
        tmp_path / "reasoning_dataset2samples.json",
        {"math500": 500, "orphan": 10},
    )

    assert resolve_reasoning_datasets("all", str(tmp_path)) == ["math500"]


def test_resolve_reasoning_datasets_rejects_unknown_dataset(tmp_path):
    _write_json(
        tmp_path / "reasoning_dataset2prompt.json",
        {"math500": "Solve:\n{problem}"},
    )
    _write_json(tmp_path / "reasoning_dataset2maxlen.json", {"math500": 4096})
    _write_json(tmp_path / "reasoning_dataset2samples.json", {"math500": 500})

    with pytest.raises(ValueError, match="Unsupported reasoning dataset"):
        resolve_reasoning_datasets("aime2025", str(tmp_path))


def test_load_reasoning_rows_reads_jsonl_records(tmp_path):
    data_file = tmp_path / "math500.jsonl"
    _write_jsonl(data_file, [{"problem": "2+2", "answer": "4"}])

    rows = load_reasoning_rows(str(data_file))

    assert rows == [{"problem": "2+2", "answer": "4"}]


def test_build_reasoning_prompt_records_uses_problem_field_and_boxed_instruction():
    rows = [{"problem": "2+2", "answer": "4", "unique_id": "m1"}]

    records = build_reasoning_prompt_records(
        dataset_name="math500",
        rows=rows,
        prompt_template=(
            "{problem}\n\nPlease reason step by step and put your final answer within \\boxed{}."
        ),
        chat_formatter=None,
    )

    assert records[0]["prompt"].endswith("within \\boxed{}.")
    assert records[0]["answers"] == ["4"]
    assert records[0]["meta"]["unique_id"] == "m1"


def test_build_reasoning_prompt_records_normalizes_plural_answers_and_preserves_metadata():
    rows = [
        {
            "problem": "2+2",
            "answers": ["4", 4],
            "id": "row-1",
            "subject": "math",
            "level": "easy",
        }
    ]

    records = build_reasoning_prompt_records(
        dataset_name="math500",
        rows=rows,
        prompt_template="{problem}",
        chat_formatter=None,
    )

    assert records[0]["answers"] == ["4", "4"]
    assert records[0]["meta"] == {
        "id": "row-1",
        "subject": "math",
        "level": "easy",
    }


def test_build_reasoning_prompt_records_uses_chat_formatter_when_present():
    rows = [{"question": "2+2", "answer": "4"}]

    records = build_reasoning_prompt_records(
        dataset_name="math500",
        rows=rows,
        prompt_template="{problem}",
        chat_formatter=lambda prompt: f"CHAT::{prompt}",
    )

    assert records[0]["prompt"] == "CHAT::2+2"


def test_build_reasoning_prompt_records_prefers_higher_priority_prompt_field():
    rows = [{"problem": "2+2", "question": "wrong", "answer": "4"}]

    records = build_reasoning_prompt_records(
        dataset_name="math500",
        rows=rows,
        prompt_template="{question}",
        chat_formatter=None,
    )

    assert records[0]["prompt"] == "2+2"


@pytest.mark.parametrize(
    ("row", "expected_prompt"),
    [
        (
            {
                "problem": "problem-value",
                "question": "question-value",
                "input": "input-value",
                "prompt": "prompt-value",
                "text": "text-value",
            },
            "problem-value",
        ),
        (
            {
                "question": "question-value",
                "input": "input-value",
                "prompt": "prompt-value",
                "text": "text-value",
            },
            "question-value",
        ),
        (
            {
                "input": "input-value",
                "prompt": "prompt-value",
                "text": "text-value",
            },
            "input-value",
        ),
        (
            {
                "prompt": "prompt-value",
                "text": "text-value",
            },
            "prompt-value",
        ),
        (
            {
                "text": "text-value",
            },
            "text-value",
        ),
    ],
)
def test_build_reasoning_prompt_context_uses_shared_prompt_field_precedence(
    row,
    expected_prompt,
):
    context = build_reasoning_prompt_context(row)

    assert context["problem"] == expected_prompt
    assert context["question"] == expected_prompt
    assert context["input"] == expected_prompt
    assert context["prompt"] == expected_prompt
    assert context["text"] == expected_prompt


def test_resolve_reasoning_max_samples_uses_dataset_default_when_cli_limit_missing(tmp_path):
    _write_json(tmp_path / "reasoning_dataset2samples.json", {"math500": 500})

    assert resolve_reasoning_max_samples("math500", str(tmp_path), max_samples=None) == 500


def test_resolve_reasoning_max_samples_uses_cli_limit_when_provided(tmp_path):
    _write_json(tmp_path / "reasoning_dataset2samples.json", {"math500": 500})

    assert resolve_reasoning_max_samples("math500", str(tmp_path), max_samples=7) == 7


def test_reasoning_configs_register_math500_values():
    config_dir = "eval/config"

    assert _read_json(f"{config_dir}/reasoning_dataset2prompt.json") == {
        "math500": (
            "Solve the problem step by step, and put your final answer within \\boxed{}.\n\n"
            "{problem}\n\nPlease reason step by step and put your final answer within \\boxed{}."
        )
    }
    assert _read_json(f"{config_dir}/reasoning_dataset2maxlen.json") == {"math500": 4096}
    assert _read_json(f"{config_dir}/reasoning_dataset2metric.json") == {"math500": "boxed_accuracy"}
    assert _read_json(f"{config_dir}/reasoning_dataset2samples.json") == {"math500": 500}
