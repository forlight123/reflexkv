import json
import os

import numpy as np

from eval.utils.metrics import (
    classification_score,
    code_sim_score,
    count_score,
    qa_f1_score,
    qa_f1_zh_score,
    retrieval_score,
    retrieval_zh_score,
    rouge_score,
    rouge_zh_score,
)


METRIC_MAP = {
    "qa_f1": qa_f1_score,
    "qa_f1_zh": qa_f1_zh_score,
    "rouge": rouge_score,
    "rouge_zh": rouge_zh_score,
    "classification": classification_score,
    "retrieval": retrieval_score,
    "retrieval_zh": retrieval_zh_score,
    "count": count_score,
    "code_sim": code_sim_score,
}


def postprocess_prediction_for_dataset(prediction, dataset_name):
    if dataset_name != "samsum":
        return prediction

    markers = [
        "\nDialogue:",
        "\r\nDialogue:",
        "Dialogue:",
        "\nCorrected Summary:",
        "\r\nCorrected Summary:",
        "Corrected Summary:",
        "\nSummary:",
        "\r\nSummary:",
        "Summary:",
    ]
    cutoff = len(prediction)
    for marker in markers:
        idx = prediction.find(marker)
        if idx != -1:
            cutoff = min(cutoff, idx)
    return prediction[:cutoff].strip()


def load_config(config_dir, config_name):
    config_path = os.path.join(config_dir, f"{config_name}.json")
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_file(prediction_file, dataset_name, config_dir):
    metric_name = load_config(config_dir, "dataset2metric")[dataset_name]
    metric_fn = METRIC_MAP[metric_name]
    scores = []

    with open(prediction_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            data = json.loads(line)
            pred = postprocess_prediction_for_dataset(data["pred"], dataset_name)
            answers = data["answers"]
            all_classes = data.get("all_classes", [])
            if isinstance(answers, list):
                current_scores = [
                    metric_fn(pred, answer, all_classes=all_classes)
                    for answer in answers
                ]
                score = max(current_scores) if current_scores else 0.0
            else:
                score = metric_fn(pred, answers, all_classes=all_classes)
            scores.append(score)

    avg_score = np.mean(scores) if scores else 0.0
    stats = {
        "dataset": dataset_name,
        "avg_score": avg_score,
        "total_samples": len(scores),
        "metric": metric_name,
    }
    result_file = os.path.join(os.path.dirname(prediction_file), "result.json")
    with open(result_file, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, ensure_ascii=False)
    return avg_score
