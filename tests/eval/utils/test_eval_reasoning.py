import json

import pytest

from eval.utils.eval_reasoning import (
    clean_latex_answer,
    evaluate_file,
    extract_last_boxed_content,
    is_boxed_answer_correct,
    is_ruler_string_match_correct,
)


def test_extract_last_boxed_content_returns_last_balanced_match():
    pred = "First \\boxed{1} then final \\boxed{\\frac{3}{4}}"
    assert extract_last_boxed_content(pred) == "\\frac{3}{4}"


def test_extract_last_boxed_content_handles_nested_braces():
    pred = "Answer: \\boxed{\\left(\\frac{1}{2}\\right)}"
    assert extract_last_boxed_content(pred) == "\\left(\\frac{1}{2}\\right)"


def test_is_boxed_answer_correct_falls_back_to_cleaned_prediction():
    assert is_boxed_answer_correct("  42  ", "42") is True


def test_clean_latex_answer_removes_common_wrappers():
    assert clean_latex_answer("\\left( \\frac{1}{2} \\right)") == "(\\frac{1}{2})"


def test_clean_latex_answer_strips_wrapped_command_content():
    assert clean_latex_answer(r"\mathrm{4}") == "4"
    assert clean_latex_answer(r"\text{abc}") == "abc"


def test_clean_latex_answer_preserves_longer_command_tokens():
    assert clean_latex_answer(r"\textbf{abc}") == r"\textbf{abc}"
    assert clean_latex_answer(r"\mathrmx") == r"\mathrmx"


def test_is_boxed_answer_correct_handles_wrapped_boxed_prediction():
    assert is_boxed_answer_correct(r"Answer: \boxed{\mathrm{4}}", "4") is True


def test_is_boxed_answer_correct_uses_numeric_float_fallback():
    assert is_boxed_answer_correct(r"Answer: \boxed{4.0}", "4") is True


def test_is_ruler_string_match_correct_requires_all_references():
    assert is_ruler_string_match_correct("The values are 123 and 456.", ["123", "456"])
    assert not is_ruler_string_match_correct("The value is 123.", ["123", "456"])


@pytest.mark.parametrize(
    ("pred", "gold"),
    [
        (
            r"\boxed{\frac{270}{7}\text{ degrees}}",
            r"\frac{270}7\text{ degrees}",
        ),
        (
            r"\boxed{864 \text{ inches}^2}",
            r"864 \mbox{ inches}^2",
        ),
        (
            r"\boxed{\frac{17}{50}}",
            r"\dfrac{17}{50}",
        ),
    ],
)
def test_is_boxed_answer_correct_normalizes_math500_unit_forms(pred, gold):
    assert is_boxed_answer_correct(pred, gold) is True


def test_evaluate_file_writes_result_and_badcases(tmp_path):
    pred_file = tmp_path / "pred.jsonl"
    pred_file.write_text(
        json.dumps({"pred": "Answer is \\boxed{4}", "answers": ["4"], "meta": {"unique_id": "ok"}})
        + "\n"
        + json.dumps({"pred": "Answer is \\boxed{5}", "answers": ["4"], "meta": {"unique_id": "bad"}})
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "reasoning_dataset2metric.json").write_text(
        json.dumps({"math500": "boxed_accuracy"}),
        encoding="utf-8",
    )

    score = evaluate_file(str(pred_file), "math500", str(tmp_path))

    assert score == 0.5
    result_payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result_payload == {
        "dataset": "math500",
        "metric": "boxed_accuracy",
        "total_samples": 2,
        "avg_score": 0.5,
    }
    badcases = [
        json.loads(line)
        for line in (tmp_path / "badcases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert badcases == [
        {
            "meta": {"unique_id": "bad"},
            "passed": False,
            "extracted": "5",
            "gold_clean": "4",
            "original_pred": "Answer is \\boxed{5}",
        }
    ]


def test_evaluate_file_supports_ruler_string_match(tmp_path):
    pred_file = tmp_path / "pred.jsonl"
    pred_file.write_text(
        json.dumps(
            {
                "pred": "The hidden values are 123 and 456.",
                "answers": ["123", "456"],
                "meta": {"unique_id": "ok"},
            }
        )
        + "\n"
        + json.dumps(
            {
                "pred": "The hidden value is 123.",
                "answers": ["123", "456"],
                "meta": {"unique_id": "bad"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "reasoning_dataset2metric.json").write_text(
        json.dumps({"ruler_niah_multikey_2_4k": "ruler_string_match"}),
        encoding="utf-8",
    )

    score = evaluate_file(str(pred_file), "ruler_niah_multikey_2_4k", str(tmp_path))

    assert score == 0.5
    result_payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result_payload["metric"] == "ruler_string_match"
