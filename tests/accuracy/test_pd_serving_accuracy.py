import argparse
import asyncio
import json
from pathlib import Path

from scripts.accuracy import run_pd_serving_accuracy as runner


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model="/models/llama",
        host="127.0.0.1",
        prefill_gpu="6",
        decode_gpu="7",
        prefill_port=8710,
        decode_port=8720,
        proxy_port=8730,
        prefill_bootstrap_port=8998,
        proxy_prefill_max_inflight=2,
        mooncake_protocol="rdma",
        mooncake_num_workers=10,
        reflex_keep_recent_blocks=4,
        reflex_keep_initial_blocks=32,
        reflex_max_int4_fraction_per_request=0.5,
        reflex_survival_warmup_tokens=128,
        reflex_risk_warmup_tokens=16,
        reflex_short_admission_max_int4_fraction=0.03,
        reflex_sparse_window_pages=32,
        reflex_short_max_demote_per_window=1,
        reflex_max_demote_per_window=2,
        reflex_low_risk_score_fraction=0.25,
        reflex_page_selection_policy="relevance_sparse",
        max_model_len=32768,
        gpu_memory_utilization=0.85,
        block_size=16,
        max_num_seqs=16,
        max_num_batched_tokens=8192,
        prefill_kv_cache_dtype="auto",
        decode_kv_cache_dtype="reflex_int4",
        num_gpu_blocks_override=None,
        force_triton_attn=True,
        enforce_eager=True,
        enable_reflex_trace=True,
        reflex_int4_budget_fraction=0.5,
        extra_serve_args=[],
        output_root=str(tmp_path),
        run_name="unit",
        task="longbench",
        dataset="qasper",
        data_dir=str(tmp_path / "data"),
        config_dir=str(tmp_path / "config"),
        max_samples=2,
        max_concurrency=8,
        request_rate="0.5",
        temperature=0.0,
        top_p=1.0,
        seed=0,
        sample_interval_sec=1.0,
        server_ready_timeout_sec=30.0,
        request_timeout_sec=120.0,
    )


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_server_commands_use_pd_reflex_defaults(tmp_path):
    args = _args(tmp_path)

    prefill_cmd = runner.build_server_cmd(args, runner.Role.PREFILL)
    decode_cmd = runner.build_server_cmd(args, runner.Role.DECODE)
    prefill_env = runner.build_server_env(args, runner.Role.PREFILL)
    decode_env = runner.build_server_env(args, runner.Role.DECODE)

    assert prefill_cmd[prefill_cmd.index("--kv-cache-dtype") + 1] == "auto"
    assert decode_cmd[decode_cmd.index("--kv-cache-dtype") + 1] == "reflex_int4"
    assert prefill_cmd[prefill_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert decode_cmd[decode_cmd.index("--attention-backend") + 1] == "TRITON_ATTN"
    assert "--enforce-eager" in prefill_cmd
    assert "--enforce-eager" in decode_cmd
    assert decode_env["CUDA_VISIBLE_DEVICES"] == "7"
    assert prefill_env["SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA"] == "1"
    assert decode_env["SEMANTIQ_REFLEX_TRACE"] == "1"
    assert decode_env["SEMANTIQ_REFLEX_INT4_BUDGET_FRACTION"] == "0.5"
    assert decode_env["SEMANTIQ_REFLEX_KEEP_RECENT_PAGES"] == "4"
    assert decode_env["SEMANTIQ_REFLEX_KEEP_INITIAL_PAGES"] == "32"
    assert decode_env["SEMANTIQ_REFLEX_MAX_INT4_FRACTION_PER_REQUEST"] == "0.5"
    assert decode_env["SEMANTIQ_REFLEX_SURVIVAL_WARMUP_TOKENS"] == "128"
    assert decode_env["SEMANTIQ_REFLEX_RISK_WARMUP_TOKENS"] == "16"
    assert decode_env["SEMANTIQ_REFLEX_SHORT_ADMISSION_MAX_INT4_FRACTION"] == "0.03"
    assert decode_env["SEMANTIQ_REFLEX_SPARSE_WINDOW_PAGES"] == "32"
    assert decode_env["SEMANTIQ_REFLEX_SHORT_MAX_DEMOTE_PER_WINDOW"] == "1"
    assert decode_env["SEMANTIQ_REFLEX_MAX_DEMOTE_PER_WINDOW"] == "2"
    assert decode_env["SEMANTIQ_REFLEX_LOW_RISK_SCORE_FRACTION"] == "0.25"
    assert decode_env["SEMANTIQ_REFLEX_PAGE_SELECTION_POLICY"] == "relevance_sparse"
    assert decode_env["SEMANTIQ_REFLEX_DECODE_PRESSURE_WARMUP_TOKENS"] == "32"
    assert decode_env["SEMANTIQ_REFLEX_DECODE_PRESSURE_RAMP_TOKENS"] == "512"
    assert decode_env["SEMANTIQ_REFLEX_SHORT_PREFILL_PAGES"] == "64"
    assert decode_env["SEMANTIQ_REFLEX_LONG_PREFILL_PAGES"] == "512"


def test_load_longbench_records_builds_scored_prompts(tmp_path):
    args = _args(tmp_path)
    _write_json(args.config_dir and Path(args.config_dir) / "dataset2samples.json", {"qasper": 2})
    _write_json(Path(args.config_dir) / "dataset2prompt.json", {"qasper": "Q: {input}\nC: {context}\nA:"})
    _write_json(Path(args.config_dir) / "dataset2maxlen.json", {"qasper": 8})
    _write_json(Path(args.config_dir) / "dataset2metric.json", {"qasper": "qa_f1"})
    _write_jsonl(
        Path(args.data_dir) / "qasper.jsonl",
        [
            {
                "input": "question",
                "context": "context",
                "answers": ["answer"],
                "all_classes": [],
            },
            {
                "input": "question2",
                "context": "context2",
                "answers": ["answer2"],
                "all_classes": [],
            },
        ],
    )

    dataset = runner.load_serving_dataset(args, chat_formatter=None)

    assert dataset.max_new_tokens == 8
    assert [record.answers for record in dataset.records] == [["answer"], ["answer2"]]
    assert dataset.records[0].prompt == "Q: question\nC: context\nA:"


def test_write_predictions_scores_reasoning_outputs(tmp_path):
    args = _args(tmp_path)
    args.task = "reasoning"
    args.dataset = "math500"
    _write_json(Path(args.config_dir) / "reasoning_dataset2metric.json", {"math500": "boxed_accuracy"})
    dataset = runner.ServingDataset(
        task="reasoning",
        dataset="math500",
        max_new_tokens=16,
        records=[
            runner.PromptRecord(
                dataset="math500",
                prompt="problem",
                answers=["42"],
                all_classes=[],
                meta={"id": "sample-0"},
            )
        ],
    )
    predictions = [
        runner.ServingPrediction(
            index=0,
            pred="The answer is \\boxed{42}.",
            error=None,
            latency_seconds=0.25,
        )
    ]
    run_dir = tmp_path / "runs" / "unit"

    score = runner.write_predictions_and_score(
        args=args,
        run_dir=run_dir,
        dataset=dataset,
        predictions=predictions,
        duration_seconds=1.5,
    )

    dataset_dir = run_dir / "math500"
    assert score == 1.0
    assert json.loads((dataset_dir / "result.json").read_text())["avg_score"] == 1.0
    summary = json.loads((dataset_dir / "run_summary.json").read_text())
    assert summary["completed_predictions"] == 1
    assert summary["failed_predictions"] == 0
    assert summary["avg_latency_seconds"] == 0.25


def test_run_serving_requests_times_out_stuck_pd_request(monkeypatch, tmp_path):
    args = _args(tmp_path)
    args.max_concurrency = 1
    args.request_rate = "inf"
    args.request_timeout_sec = 0.01
    dataset = runner.ServingDataset(
        task="longbench",
        dataset="qasper",
        max_new_tokens=8,
        records=[
            runner.PromptRecord(
                dataset="qasper",
                prompt="prompt",
                answers=["answer"],
                all_classes=[],
            )
        ],
    )

    async def stuck_request(**_kwargs):
        await asyncio.sleep(60)
        return "unreachable"

    monkeypatch.setattr(runner, "_async_completion_request", stuck_request)

    predictions = asyncio.run(
        runner.run_serving_requests(
            args=args,
            dataset=dataset,
            base_url="http://127.0.0.1:9",
        )
    )

    assert len(predictions) == 1
    assert predictions[0].pred == ""
    assert predictions[0].error is not None
    assert "Timeout" in predictions[0].error


def test_completion_request_includes_priority_when_supplied():
    captured = {}

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"text": "ok"}]}

    class _Client:
        async def post(self, url, json):
            captured["url"] = url
            captured["payload"] = json
            return _Response()

    pred = asyncio.run(
        runner._async_completion_request(
            client=_Client(),
            base_url="http://127.0.0.1:8730",
            model="/models/llama",
            record=runner.PromptRecord(
                dataset="qasper",
                prompt="prompt",
                answers=["answer"],
            ),
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            priority=-1,
            request_id="semantiq-unit-000001",
        )
    )

    assert pred == "ok"
    assert captured["url"] == "http://127.0.0.1:8730/v1/completions"
    assert captured["payload"]["priority"] == -1
    assert captured["payload"]["request_id"] == "semantiq-unit-000001"
