import json

from eval.utils.eval_longbench import evaluate_file


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_evaluate_file_trims_samsum_prediction_before_dialogue_marker(tmp_path):
    pred_file = tmp_path / "pred.jsonl"
    pred_file.write_text(
        json.dumps(
            {
                "pred": "Molly and Anna will go to the Muse concert in Cracow.\nDialogue: stray tail",
                "answers": ["Molly and Anna will go to the Muse concert in Cracow."],
                "all_classes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / "dataset2metric.json", {"samsum": "rouge"})

    score = evaluate_file(str(pred_file), "samsum", str(tmp_path))

    assert score > 0.99


def test_evaluate_file_writes_result_json(tmp_path):
    pred_file = tmp_path / "pred.jsonl"
    pred_file.write_text(
        json.dumps(
            {
                "pred": "Paragraph 7",
                "answers": ["Paragraph 7"],
                "all_classes": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(tmp_path / "dataset2metric.json", {"passage_retrieval_en": "retrieval"})

    evaluate_file(str(pred_file), "passage_retrieval_en", str(tmp_path))

    result_payload = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result_payload["dataset"] == "passage_retrieval_en"
    assert result_payload["total_samples"] == 1
