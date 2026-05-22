import json
import os
from pathlib import Path

from eval.utils.reasoning import load_config


_BRACED_WRAPPERS = ("\\mathrm", "\\text", "\\mbox")
_SIMPLE_WRAPPERS = ("\\left", "\\right", "\\displaystyle", "\\limits")


def extract_last_boxed_content(text: str) -> str | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    index = start + len(marker)
    depth = 1
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start + len(marker) : index]
        index += 1
    return None


def clean_latex_answer(text: str) -> str:
    cleaned = text.replace("$", "")
    cleaned = cleaned.replace("\\dfrac", "\\frac")
    cleaned = _strip_latex_wrappers(cleaned)
    cleaned = _normalize_frac_forms(cleaned)
    cleaned = cleaned.replace("\n", "").replace("\r", "").replace("\t", "")
    cleaned = "".join(cleaned.split())
    return cleaned


def _extract_balanced_braces(text: str, start_index: int):
    if start_index >= len(text) or text[start_index] != "{":
        return None, start_index

    depth = 0
    index = start_index
    while index < len(text):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start_index + 1 : index], index
        index += 1
    return None, start_index


def _strip_latex_wrappers(text: str) -> str:
    parts = []
    index = 0
    while index < len(text):
        if text[index] == "\\":
            matched = None
            for wrapper in _BRACED_WRAPPERS + _SIMPLE_WRAPPERS:
                if text.startswith(wrapper, index):
                    matched = wrapper
                    break

            if matched is not None:
                boundary_index = index + len(matched)
                if boundary_index < len(text) and text[boundary_index].isalpha():
                    parts.append(text[index])
                    index += 1
                    continue
                if matched in _BRACED_WRAPPERS:
                    brace_index = boundary_index
                    while brace_index < len(text) and text[brace_index].isspace():
                        brace_index += 1
                    if brace_index < len(text) and text[brace_index] == "{":
                        content, end_index = _extract_balanced_braces(text, brace_index)
                        if content is not None:
                            parts.append(content)
                            index = end_index + 1
                            continue
                index += len(matched)
                continue

        parts.append(text[index])
        index += 1
    return "".join(parts)


def _normalize_frac_forms(text: str) -> str:
    parts = []
    index = 0
    while index < len(text):
        if text.startswith("\\frac", index):
            frac_start = index
            index += len("\\frac")
            numerator, numerator_end = _extract_balanced_braces(text, index)
            if numerator is None:
                parts.append(text[frac_start])
                index = frac_start + 1
                continue

            index = numerator_end + 1
            if index >= len(text):
                parts.append(text[frac_start : numerator_end + 1])
                continue

            if text[index] == "{":
                denominator, denominator_end = _extract_balanced_braces(text, index)
                if denominator is None:
                    parts.append(text[frac_start])
                    index = frac_start + 1
                    continue
                index = denominator_end + 1
            else:
                denominator_start = index
                while index < len(text) and (text[index].isalnum() or text[index] in ".-"):
                    index += 1
                denominator = text[denominator_start:index]
                if not denominator:
                    parts.append(text[frac_start])
                    index = frac_start + 1
                    continue

            parts.append(f"\\frac{{{numerator}}}{{{denominator}}}")
            continue

        parts.append(text[index])
        index += 1
    return "".join(parts)


def _normalize_prediction(pred_str: str) -> str:
    extracted = extract_last_boxed_content(pred_str)
    if extracted is None:
        return clean_latex_answer(pred_str)
    return clean_latex_answer(extracted)


def _to_float(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_boxed_answer_correct(pred_str: str, gold_str: str) -> bool:
    pred_clean = _normalize_prediction(pred_str)
    gold_clean = clean_latex_answer(gold_str)

    if pred_clean == gold_clean:
        return True

    pred_num = _to_float(pred_clean)
    gold_num = _to_float(gold_clean)
    if pred_num is None or gold_num is None:
        return False
    return pred_num == gold_num


def _load_predictions(file_path: str):
    rows = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_answers(answers):
    if isinstance(answers, list):
        return [str(answer) for answer in answers]
    return [str(answers)]


def evaluate_file(file_path: str, dataset_name: str, config_dir: str) -> float:
    metric_name = load_config(config_dir, "reasoning_dataset2metric")[dataset_name]
    rows = _load_predictions(file_path)

    scores = []
    badcases = []
    for row in rows:
        pred_str = row["pred"]
        answers = _normalize_answers(row.get("answers", []))
        meta = row.get("meta", {})

        pred_clean = _normalize_prediction(pred_str)
        passed = any(is_boxed_answer_correct(pred_str, answer) for answer in answers)
        score = 1.0 if passed else 0.0
        scores.append(score)

        if not passed:
            gold_clean = clean_latex_answer(answers[0] if answers else "")
            badcases.append(
                {
                    "meta": meta,
                    "passed": passed,
                    "extracted": pred_clean,
                    "gold_clean": gold_clean,
                    "original_pred": pred_str,
                }
            )

    total_samples = len(scores)
    avg_score = sum(scores) / total_samples if total_samples else 0.0

    result_payload = {
        "dataset": dataset_name,
        "metric": metric_name,
        "total_samples": total_samples,
        "avg_score": avg_score,
    }

    output_dir = Path(os.path.dirname(file_path))
    result_path = output_dir / "result.json"
    badcases_path = output_dir / "badcases.jsonl"

    with open(result_path, "w", encoding="utf-8") as handle:
        json.dump(result_payload, handle, indent=2, ensure_ascii=False)

    with open(badcases_path, "w", encoding="utf-8") as handle:
        for badcase in badcases:
            handle.write(json.dumps(badcase, ensure_ascii=False) + "\n")

    return avg_score
