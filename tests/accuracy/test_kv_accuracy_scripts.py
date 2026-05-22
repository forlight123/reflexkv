import json
import sys

from scripts.accuracy.run_kv_accuracy import main as run_accuracy_main
from scripts.accuracy.summarize_kv_accuracy import collect_rows


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_dry_run_manifest_includes_low_bit_engine_switches(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kv_accuracy.py",
            "--output-root",
            str(tmp_path),
            "--run-tag",
            "dry",
            "--variants",
            "auto,fp8,int4,reflex_int4",
            "--tasks",
            "longbench",
            "--longbench-datasets",
            "qasper",
            "--longbench-max-samples",
            "2",
            "--gpus",
            "6",
            "--dry-run",
        ],
    )

    assert run_accuracy_main() == 0

    manifest = json.loads((tmp_path / "dry" / "manifest.json").read_text())
    commands = {run["variant"]: run["command"] for run in manifest["runs"]}

    assert commands["auto"].count("--kv-cache-dtype") == 1
    assert commands["auto"][commands["auto"].index("--kv-cache-dtype") + 1] == "auto"
    assert "--enforce-eager" not in commands["auto"]

    fp8_cmd = commands["fp8"]
    assert fp8_cmd[fp8_cmd.index("--kv-cache-dtype") + 1] == "fp8"
    assert "--enforce-eager" not in fp8_cmd

    int4_cmd = commands["int4"]
    assert int4_cmd[int4_cmd.index("--kv-cache-dtype") + 1] == "int4"
    assert int4_cmd[int4_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert "--enforce-eager" in int4_cmd

    reflex_cmd = commands["reflex_int4"]
    assert reflex_cmd[reflex_cmd.index("--kv-cache-dtype") + 1] == "reflex_int4"
    assert reflex_cmd[reflex_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert "--enforce-eager" in reflex_cmd


def test_dry_run_defaults_cover_requested_accuracy_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_kv_accuracy.py",
            "--output-root",
            str(tmp_path),
            "--run-tag",
            "defaults",
            "--tasks",
            "longbench",
            "--longbench-max-samples",
            "1",
            "--dry-run",
        ],
    )

    assert run_accuracy_main() == 0

    manifest = json.loads((tmp_path / "defaults" / "manifest.json").read_text())
    assert len(manifest["runs"]) == 12
    assert {run["variant"] for run in manifest["runs"]} == {
        "auto",
        "fp8",
        "int4",
        "reflex_int4",
    }
    assert {run["datasets"] for run in manifest["runs"]} == {
        "qasper",
        "hotpotqa",
        "multifieldqa_en",
    }
    for run in manifest["runs"]:
        command = run["command"]
        assert command[command.index("--data-dir") + 1].endswith("data/longbench")


def test_summarizer_computes_score_delta_and_prediction_match(tmp_path):
    run_root = tmp_path / "runs"
    _write_json(
        run_root / "manifest.json",
        {
            "runs": [
                {
                    "task": "longbench",
                    "variant": "auto",
                    "run_name": "longbench_qasper_kv-auto",
                },
                {
                    "task": "longbench",
                    "variant": "int4",
                    "run_name": "longbench_qasper_kv-int4",
                },
            ]
        },
    )
    for run_name, variant, score, preds in [
        ("longbench_qasper_kv-auto", "auto", 0.75, ["a", "b"]),
        ("longbench_qasper_kv-int4", "int4", 0.50, ["a", "c"]),
    ]:
        dataset_dir = run_root / run_name / "qasper"
        _write_json(
            dataset_dir / "result.json",
            {
                "dataset": "qasper",
                "metric": "qa_f1",
                "avg_score": score,
                "total_samples": 2,
            },
        )
        _write_json(
            dataset_dir / "run_summary.json",
            {
                "completed_predictions": 2,
                "failed_predictions": 0,
                "duration_seconds": 1.0,
            },
        )
        _write_json(
            dataset_dir / "run_config.json",
            {
                "backend": "vllm",
                "kv_cache_dtype": variant,
                "data_dir": "/home/ytm/datasets/LongBench/data",
            },
        )
        _write_jsonl(
            dataset_dir / "pred.jsonl",
            [{"pred": pred, "answers": [pred], "all_classes": []} for pred in preds],
        )

    rows = collect_rows(run_root, baseline_variant="auto")
    int4_row = next(row for row in rows if row["variant"] == "int4")

    assert int4_row["score_delta_vs_baseline"] == -0.25
    assert int4_row["exact_pred_match_rate_vs_baseline"] == 0.5


def test_summarizer_accepts_pd_serving_run_config(tmp_path):
    run_root = tmp_path / "pd_serving"
    dataset_dir = run_root / "qasper_reflex_c8_n8" / "qasper"
    _write_json(
        dataset_dir / "result.json",
        {
            "dataset": "qasper",
            "metric": "qa_f1",
            "avg_score": 0.25,
            "total_samples": 2,
        },
    )
    _write_json(
        dataset_dir / "run_summary.json",
        {
            "completed_predictions": 2,
            "failed_predictions": 0,
            "duration_seconds": 1.0,
        },
    )
    _write_json(
        dataset_dir / "run_config.json",
        {
            "task": "longbench",
            "decode_kv_cache_dtype": "reflex_int4",
            "data_dir": "data/longbench",
        },
    )

    rows = collect_rows(run_root, baseline_variant="auto")

    assert rows[0]["task"] == "longbench"
    assert rows[0]["variant"] == "reflex_int4"
    assert rows[0]["kv_cache_dtype"] == "reflex_int4"


def test_summarizer_includes_mixed_serving_pressure_metrics(tmp_path):
    run_root = tmp_path / "pd_serving_mixed"
    run_dir = run_root / "pdserv_mixed_real_c8"
    dataset_dir = run_dir / "math500"
    _write_json(
        dataset_dir / "result.json",
        {
            "dataset": "math500",
            "metric": "boxed_accuracy",
            "avg_score": 0.5,
            "total_samples": 2,
        },
    )
    _write_json(
        dataset_dir / "run_summary.json",
        {
            "completed_predictions": 2,
            "failed_predictions": 0,
            "duration_seconds": 12.0,
            "avg_latency_seconds": 5.0,
            "p95_latency_seconds": 6.0,
        },
    )
    _write_json(
        dataset_dir / "run_config.json",
        {
            "decode_kv_cache_dtype": "reflex_int4",
            "data_dir": "data/mixed",
        },
    )
    _write_jsonl(
        dataset_dir / "pred.jsonl",
        [
            {"pred": "one", "answers": ["1"], "all_classes": []},
            {"pred": "two two", "answers": ["2"], "all_classes": []},
        ],
    )
    _write_json(
        run_dir / "mixed_summary.json",
        {
            "serving_metrics": {
                "decode": {
                    "max_kv_cache_usage_pct": 91.5,
                    "avg_kv_cache_usage_pct": 72.25,
                    "max_running": 8,
                    "avg_running": 5.5,
                    "max_waiting": 6,
                    "avg_waiting": 2.25,
                }
            },
            "datasets": {
                "math500": {
                    "task": "reasoning",
                },
            },
            "reflex_trace": {
                "demoted_pages_total": 32,
                "landing_materialized_pages_total": 16,
                "max_int4_ratio": 0.625,
            },
        },
    )
    _write_jsonl(
        run_dir / "mixed_request_trace.jsonl",
        [
            {
                "dataset": "math500",
                "max_new_tokens": 4096,
                "prompt_chars": 1000,
                "prompt_original_tokens": 900,
                "prompt_final_tokens": 900,
                "prompt_truncated": False,
                "prediction_chars": 200,
            },
            {
                "dataset": "math500",
                "max_new_tokens": 4096,
                "prompt_chars": 1500,
                "prompt_original_tokens": 17000,
                "prompt_final_tokens": 12280,
                "prompt_truncated": True,
                "prediction_chars": 600,
            },
        ],
    )

    rows = collect_rows(run_root, baseline_variant="auto")

    row = rows[0]
    assert row["task"] == "reasoning"
    assert row["max_new_tokens"] == 4096
    assert row["avg_prompt_chars"] == 1250.0
    assert row["max_prompt_chars"] == 1500
    assert row["avg_prompt_original_tokens"] == 8950.0
    assert row["max_prompt_original_tokens"] == 17000
    assert row["avg_prompt_final_tokens"] == 6590.0
    assert row["max_prompt_final_tokens"] == 12280
    assert row["prompt_truncated_total"] == 1
    assert row["avg_prediction_chars"] == 400.0
    assert row["max_prediction_chars"] == 600
    assert row["avg_latency_seconds"] == 5.0
    assert row["p95_latency_seconds"] == 6.0
    assert row["decode_max_running"] == 8
    assert row["decode_avg_running"] == 5.5
    assert row["decode_max_waiting"] == 6
    assert row["decode_avg_waiting"] == 2.25
    assert row["decode_max_kv_cache_usage_pct"] == 91.5
    assert row["demoted_pages_total"] == 32
    assert row["landing_materialized_pages_total"] == 16
    assert row["max_int4_ratio"] == 0.625
