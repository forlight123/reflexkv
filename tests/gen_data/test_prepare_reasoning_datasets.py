import json
from pathlib import Path

from gen_data.prepare_reasoning_datasets import (
    normalize_aime_row,
    normalize_gsm8k_row,
    update_reasoning_config,
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def test_normalize_gsm8k_row_extracts_question_and_hash_answer():
    row = {
        "question": "Jan has 3 apples and buys 4 more. How many apples?",
        "answer": "Jan has 3+4=7 apples. #### 7",
    }

    normalized = normalize_gsm8k_row(row, index=12)

    assert normalized == {
        "problem": "Jan has 3 apples and buys 4 more. How many apples?",
        "solution": "Jan has 3+4=7 apples.",
        "answer": "7",
        "source": "gsm8k",
        "unique_id": "gsm8k/12",
    }


def test_normalize_aime_row_accepts_problem_answer_schema():
    row = {"problem": "Find x.", "answer": 42, "id": "aime24-0"}

    normalized = normalize_aime_row(row, dataset="aime24", index=0)

    assert normalized == {
        "problem": "Find x.",
        "answer": "42",
        "source": "aime24",
        "unique_id": "aime24-0",
    }


def test_update_reasoning_config_registers_paper_reasoning_datasets(tmp_path):
    update_reasoning_config(tmp_path, {"gsm8k": 1319, "aime24": 30, "aime25": 30})

    prompts = _read_json(tmp_path / "reasoning_dataset2prompt.json")
    maxlens = _read_json(tmp_path / "reasoning_dataset2maxlen.json")
    metrics = _read_json(tmp_path / "reasoning_dataset2metric.json")
    samples = _read_json(tmp_path / "reasoning_dataset2samples.json")

    assert set(prompts) == {"math500", "gsm8k", "aime24", "aime25"}
    assert maxlens == {
        "math500": 4096,
        "gsm8k": 1024,
        "aime24": 4096,
        "aime25": 4096,
    }
    assert metrics == {
        "math500": "boxed_accuracy",
        "gsm8k": "boxed_accuracy",
        "aime24": "boxed_accuracy",
        "aime25": "boxed_accuracy",
    }
    assert samples == {
        "math500": 500,
        "gsm8k": 1319,
        "aime24": 30,
        "aime25": 30,
    }


def test_update_reasoning_config_preserves_existing_ruler_entries(tmp_path):
    (tmp_path / "reasoning_dataset2prompt.json").write_text(
        json.dumps({"ruler_niah_single_1_4k": "{problem}"}),
        encoding="utf-8",
    )
    (tmp_path / "reasoning_dataset2maxlen.json").write_text(
        json.dumps({"ruler_niah_single_1_4k": 64}),
        encoding="utf-8",
    )
    (tmp_path / "reasoning_dataset2metric.json").write_text(
        json.dumps({"ruler_niah_single_1_4k": "ruler_string_match"}),
        encoding="utf-8",
    )
    (tmp_path / "reasoning_dataset2samples.json").write_text(
        json.dumps({"ruler_niah_single_1_4k": 100}),
        encoding="utf-8",
    )

    update_reasoning_config(tmp_path, {"gsm8k": 1319})

    assert _read_json(tmp_path / "reasoning_dataset2prompt.json")[
        "ruler_niah_single_1_4k"
    ] == "{problem}"
    assert _read_json(tmp_path / "reasoning_dataset2maxlen.json")[
        "ruler_niah_single_1_4k"
    ] == 64
    assert _read_json(tmp_path / "reasoning_dataset2metric.json")[
        "ruler_niah_single_1_4k"
    ] == "ruler_string_match"
    assert _read_json(tmp_path / "reasoning_dataset2samples.json")[
        "ruler_niah_single_1_4k"
    ] == 100
