import argparse
import asyncio
import json
from pathlib import Path

from scripts.accuracy import run_pd_serving_mixed_accuracy as mixed


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
        reflex_cold_admission_max_int4_fraction=0.1,
        reflex_cold_admission_emergency_free_ratio=0.05,
        reflex_slo_pressure_step=0.25,
        reflex_min_slo_pressure=0.5,
        reflex_max_slo_pressure=1.5,
        scheduling_policy="priority",
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
        tasks="longbench,reasoning",
        longbench_datasets="qasper",
        reasoning_datasets="math500",
        longbench_data_dir=str(tmp_path / "longbench"),
        reasoning_data_dir=str(tmp_path / "reasoning"),
        config_dir=str(tmp_path / "config"),
        output_root=str(tmp_path / "runs"),
        run_name="unit",
        longbench_max_samples=2,
        reasoning_max_samples=2,
        skip_chat_template=True,
        prompt_fit_policy="none",
        prompt_fit_token_margin=8,
        max_concurrency=8,
        request_rate="inf",
        workload_manifest=None,
        arrival_policy="poisson",
        trace_time_scale=1.0,
        temperature=0.0,
        top_p=1.0,
        seed=7,
        slo_classes="high,normal,low",
        slo_priorities="-1,0,1",
        workload_mix_policy="balanced",
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


def _write_fixture_data(args: argparse.Namespace) -> None:
    config_dir = Path(args.config_dir)
    _write_json(config_dir / "dataset2samples.json", {"qasper": 2})
    _write_json(config_dir / "dataset2prompt.json", {"qasper": "Q: {input}\nC: {context}\nA:"})
    _write_json(config_dir / "dataset2maxlen.json", {"qasper": 8})
    _write_json(config_dir / "dataset2metric.json", {"qasper": "qa_f1"})
    _write_json(config_dir / "reasoning_dataset2samples.json", {"math500": 2})
    _write_json(
        config_dir / "reasoning_dataset2prompt.json",
        {"math500": "Solve: {problem}\nAnswer in \\boxed{}."},
    )
    _write_json(config_dir / "reasoning_dataset2maxlen.json", {"math500": 32})
    _write_json(config_dir / "reasoning_dataset2metric.json", {"math500": "boxed_accuracy"})
    _write_jsonl(
        Path(args.longbench_data_dir) / "qasper.jsonl",
        [
            {"input": "q0", "context": "c0", "answers": ["a0"], "all_classes": []},
            {"input": "q1", "context": "c1", "answers": ["a1"], "all_classes": []},
        ],
    )
    _write_jsonl(
        Path(args.reasoning_data_dir) / "math500.jsonl",
        [
            {"problem": "1+1", "answer": "2", "id": "m0"},
            {"problem": "2+2", "answer": "4", "id": "m1"},
        ],
    )


def test_load_mixed_workload_assigns_slo_and_shuffles_across_tasks(tmp_path):
    args = _args(tmp_path)
    _write_fixture_data(args)

    workload = mixed.load_mixed_workload(args, chat_formatter=None)

    assert len(workload.requests) == 4
    assert {request.dataset for request in workload.requests} == {"qasper", "math500"}
    assert {request.task for request in workload.requests} == {"longbench", "reasoning"}
    assert {request.max_new_tokens for request in workload.requests} == {8, 32}
    assert {request.slo_class for request in workload.requests} <= {
        "high",
        "normal",
        "low",
    }
    assert {request.priority for request in workload.requests} <= {-1, 0, 1}
    grouped_order = [
        ("longbench", "qasper", 0),
        ("longbench", "qasper", 1),
        ("reasoning", "math500", 0),
        ("reasoning", "math500", 1),
    ]
    actual_order = [
        (request.task, request.dataset, request.source_index)
        for request in workload.requests
    ]
    assert actual_order != grouped_order


def test_load_mixed_workload_balanced_interleaves_dataset_rounds(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    args.longbench_max_samples = 3
    args.reasoning_max_samples = 3
    args.seed = 11

    datasets = []
    for task, dataset, max_tokens in [
        ("longbench", "qasper", 128),
        ("longbench", "hotpotqa", 32),
        ("longbench", "multifieldqa_en", 64),
        ("reasoning", "math500", 4096),
    ]:
        datasets.append(
            mixed.ServingDataset(
                task=task,
                dataset=dataset,
                max_new_tokens=max_tokens,
                records=[
                    mixed.PromptRecord(
                        dataset=dataset,
                        prompt=f"{dataset}-{index}",
                        answers=[str(index)],
                        all_classes=[],
                    )
                    for index in range(3)
                ],
            )
        )
    monkeypatch.setattr(mixed, "_load_task_datasets", lambda *_args: datasets)

    workload = mixed.load_mixed_workload(args, chat_formatter=None)

    assert len(workload.requests) == 12
    for offset in range(0, len(workload.requests), 4):
        round_requests = workload.requests[offset: offset + 4]
        assert {request.dataset for request in round_requests} == {
            "qasper",
            "hotpotqa",
            "multifieldqa_en",
            "math500",
        }
    assert [request.index for request in workload.requests] == list(range(12))


def test_load_mixed_workload_from_manifest_preserves_trace_metadata(tmp_path):
    args = _args(tmp_path)
    args.workload_manifest = str(tmp_path / "manifest.jsonl")
    _write_jsonl(
        Path(args.workload_manifest),
        [
            {
                "request_index": 0,
                "arrival_time_sec": 0.0,
                "scaled_arrival_time_sec": 0.0,
                "trace_index": 10,
                "trace_request_tokens": 3000,
                "trace_response_tokens": 900,
                "trace_total_tokens": 3900,
                "input_bucket": "long",
                "output_bucket": "long",
                "task": "longbench",
                "dataset": "qasper",
                "source_index": 1,
                "max_new_tokens": 16,
                "slo_class": "high",
                "priority": -1,
                "prompt": "Q: q1\nC: ctx1\nA:",
                "answers": ["a1"],
                "all_classes": [],
                "meta": {"custom": "kept"},
            },
            {
                "request_index": 1,
                "arrival_time_sec": 3.0,
                "scaled_arrival_time_sec": 0.3,
                "trace_index": 11,
                "trace_request_tokens": 120,
                "trace_response_tokens": 40,
                "trace_total_tokens": 160,
                "input_bucket": "short",
                "output_bucket": "short",
                "task": "reasoning",
                "dataset": "math500",
                "source_index": 0,
                "max_new_tokens": 64,
                "slo_class": "normal",
                "priority": 0,
                "prompt": "Solve: 1+1\nAnswer:",
                "answers": ["2"],
                "all_classes": [],
                "meta": {},
            },
        ],
    )

    workload = mixed.load_mixed_workload(args, chat_formatter=None)

    assert len(workload.requests) == 2
    assert workload.prompt_fit_summaries == []
    first = workload.requests[0]
    assert first.index == 0
    assert first.task == "longbench"
    assert first.dataset == "qasper"
    assert first.arrival_time_seconds == 0.0
    assert first.trace_request_tokens == 3000
    assert first.trace_response_tokens == 900
    assert first.record.prompt == "Q: q1\nC: ctx1\nA:"
    assert first.record.answers == ["a1"]
    assert first.record.meta["custom"] == "kept"
    assert first.record.meta["trace_index"] == 10
    assert workload.requests[1].arrival_time_seconds == 0.3
    assert workload.requests[1].max_new_tokens == 64


def test_prompt_fit_truncates_overlong_records_without_dropping_samples(tmp_path):
    class FakeTokenizer:
        def encode(self, prompt, **_kwargs):
            return [ord(ch) for ch in prompt]

        def decode(self, token_ids, **_kwargs):
            return "".join(chr(token_id) for token_id in token_ids)

    args = _args(tmp_path)
    args.prompt_fit_policy = "truncate"
    args.max_model_len = 13
    args.prompt_fit_token_margin = 1
    dataset = mixed.ServingDataset(
        task="longbench",
        dataset="qasper",
        max_new_tokens=4,
        records=[
            mixed.PromptRecord(
                dataset="qasper",
                prompt="abcdefghij",
                answers=["a"],
                all_classes=[],
            ),
            mixed.PromptRecord(
                dataset="qasper",
                prompt="abc",
                answers=["b"],
                all_classes=[],
            ),
        ],
    )

    fitted, summary = mixed.fit_dataset_to_model_len(
        args,
        dataset,
        tokenizer=FakeTokenizer(),
    )

    assert len(fitted.records) == 2
    assert fitted.records[0].prompt == "abcdghij"
    assert fitted.records[0].meta["prompt_original_tokens"] == 10
    assert fitted.records[0].meta["prompt_final_tokens"] == 8
    assert fitted.records[0].meta["prompt_truncated"] is True
    assert fitted.records[1].prompt == "abc"
    assert summary["truncated_records"] == 1
    assert summary["skipped_records"] == 0
    assert summary["max_original_prompt_tokens"] == 10
    assert summary["max_final_prompt_tokens"] == 8


def test_run_mixed_serving_requests_uses_per_request_max_tokens_and_priority(
    tmp_path,
    monkeypatch,
):
    args = _args(tmp_path)
    args.request_rate = "inf"
    workload = mixed.MixedWorkload(
        requests=[
            mixed.MixedRequest(
                index=0,
                task="longbench",
                dataset="qasper",
                source_index=0,
                record=mixed.PromptRecord(
                    dataset="qasper",
                    prompt="prompt-a",
                    answers=["a"],
                    all_classes=[],
                ),
                max_new_tokens=8,
                slo_class="high",
                priority=-1,
            ),
            mixed.MixedRequest(
                index=1,
                task="reasoning",
                dataset="math500",
                source_index=0,
                record=mixed.PromptRecord(
                    dataset="math500",
                    prompt="prompt-b",
                    answers=["2"],
                    all_classes=[],
                ),
                max_new_tokens=32,
                slo_class="low",
                priority=1,
            ),
        ]
    )
    calls = []

    async def fake_request(**kwargs):
        calls.append(kwargs)
        return f"pred-{kwargs['record'].prompt}"

    monkeypatch.setattr(mixed, "_async_completion_request", fake_request)

    predictions = asyncio.run(
        mixed.run_mixed_serving_requests(
            args=args,
            workload=workload,
            base_url="http://127.0.0.1:9",
        )
    )

    assert [call["max_tokens"] for call in calls] == [8, 32]
    assert [call["priority"] for call in calls] == [-1, 1]
    assert [call["request_id"] for call in calls] == [
        "semantiq-mixed-000000",
        "semantiq-mixed-000001",
    ]
    assert [prediction.request_index for prediction in predictions] == [0, 1]
    assert [prediction.request_id for prediction in predictions] == [
        "semantiq-mixed-000000",
        "semantiq-mixed-000001",
    ]
    assert all(prediction.end_offset_seconds >= prediction.start_offset_seconds
               for prediction in predictions)
    assert [prediction.pred for prediction in predictions] == [
        "pred-prompt-a",
        "pred-prompt-b",
    ]


def test_run_mixed_serving_requests_replays_trace_arrivals(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.arrival_policy = "trace"
    args.trace_time_scale = 2.0
    workload = mixed.MixedWorkload(
        requests=[
            mixed.MixedRequest(
                index=0,
                task="longbench",
                dataset="qasper",
                source_index=0,
                record=mixed.PromptRecord(
                    dataset="qasper",
                    prompt="prompt-a",
                    answers=["a"],
                    all_classes=[],
                ),
                max_new_tokens=8,
                slo_class="high",
                priority=-1,
                arrival_time_seconds=0.0,
            ),
            mixed.MixedRequest(
                index=1,
                task="reasoning",
                dataset="math500",
                source_index=0,
                record=mixed.PromptRecord(
                    dataset="math500",
                    prompt="prompt-b",
                    answers=["2"],
                    all_classes=[],
                ),
                max_new_tokens=32,
                slo_class="normal",
                priority=0,
                arrival_time_seconds=0.25,
            ),
        ]
    )
    sleeps = []
    clock = {"now": 0.0}

    def fake_perf_counter():
        return clock["now"]

    async def fake_sleep(delay):
        sleeps.append(delay)
        clock["now"] += delay

    async def fake_request(**kwargs):
        return kwargs["record"].prompt

    monkeypatch.setattr(mixed.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(mixed.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(mixed, "_async_completion_request", fake_request)

    predictions = asyncio.run(
        mixed.run_mixed_serving_requests(
            args=args,
            workload=workload,
            base_url="http://127.0.0.1:9",
        )
    )

    assert sleeps == [0.5]
    assert [prediction.queued_offset_seconds for prediction in predictions] == [0.0, 0.5]
    assert [prediction.request_index for prediction in predictions] == [0, 1]


def test_write_mixed_predictions_scores_each_dataset_separately(tmp_path):
    args = _args(tmp_path)
    _write_fixture_data(args)
    workload = mixed.load_mixed_workload(args, chat_formatter=None)
    predictions = [
        mixed.MixedPrediction(
            request_index=request.index,
            pred="a0" if request.dataset == "qasper" else "The answer is \\boxed{2}.",
            error=None,
            latency_seconds=0.5 + request.index,
        )
        for request in workload.requests
    ]
    run_dir = tmp_path / "runs" / "unit"

    summary = mixed.write_mixed_predictions_and_scores(
        args=args,
        run_dir=run_dir,
        workload=workload,
        predictions=predictions,
        duration_seconds=3.0,
    )

    assert set(summary["datasets"]) == {"qasper", "math500"}
    qasper_summary = json.loads((run_dir / "qasper" / "run_summary.json").read_text())
    math_summary = json.loads((run_dir / "math500" / "run_summary.json").read_text())
    assert qasper_summary["task"] == "longbench"
    assert math_summary["task"] == "reasoning"
    assert qasper_summary["requested_samples"] == 2
    assert math_summary["requested_samples"] == 2
    first_qasper = json.loads(
        (run_dir / "qasper" / "pred.jsonl").read_text().splitlines()[0]
    )
    assert "slo_class" in first_qasper["meta"]
    assert "priority" in first_qasper["meta"]
    trace_rows = [
        json.loads(line)
        for line in (run_dir / "mixed_request_trace.jsonl").read_text().splitlines()
    ]
    assert len(trace_rows) == len(workload.requests)
    first_trace = trace_rows[0]
    assert first_trace["request_id"] == "semantiq-mixed-000000"
    assert first_trace["dataset"] in {"qasper", "math500"}
    assert "prompt_chars" in first_trace
    assert "prediction_chars" in first_trace
    assert "latency_seconds" in first_trace


def test_write_mixed_predictions_request_trace_preserves_timings(tmp_path):
    args = _args(tmp_path)
    _write_fixture_data(args)
    workload = mixed.load_mixed_workload(args, chat_formatter=None)
    predictions = [
        mixed.MixedPrediction(
            request_index=request.index,
            request_id=f"rid-{request.index}",
            pred=f"prediction-{request.index}",
            error=None,
            latency_seconds=0.25 + request.index,
            queued_offset_seconds=0.1 + request.index,
            start_offset_seconds=0.2 + request.index,
            end_offset_seconds=0.45 + request.index,
        )
        for request in workload.requests
    ]
    run_dir = tmp_path / "runs" / "unit"

    mixed.write_mixed_predictions_and_scores(
        args=args,
        run_dir=run_dir,
        workload=workload,
        predictions=predictions,
        duration_seconds=3.0,
    )

    rows = [
        json.loads(line)
        for line in (run_dir / "mixed_request_trace.jsonl").read_text().splitlines()
    ]
    assert [row["request_index"] for row in rows] == [
        request.index for request in sorted(workload.requests, key=lambda item: item.index)
    ]
    assert rows[0]["request_id"] == "rid-0"
    assert rows[0]["queued_offset_seconds"] == 0.1
    assert rows[0]["start_offset_seconds"] == 0.2
    assert rows[0]["end_offset_seconds"] == 0.45


def test_write_mixed_predictions_includes_pd_pressure_metrics(tmp_path):
    args = _args(tmp_path)
    _write_fixture_data(args)
    workload = mixed.load_mixed_workload(args, chat_formatter=None)
    predictions = [
        mixed.MixedPrediction(
            request_index=request.index,
            pred="a0" if request.dataset == "qasper" else "The answer is \\boxed{2}.",
            error=None,
            latency_seconds=1.0,
        )
        for request in workload.requests
    ]
    run_dir = tmp_path / "runs" / "unit"
    run_dir.mkdir(parents=True)
    _write_jsonl(
        run_dir / "metrics_samples.jsonl",
        [
            {
                "time": 10.0,
                "vllm_metrics": {
                    "prefill": {
                        'vllm:kv_cache_usage_perc{engine="0"}': 0.98,
                        'vllm:num_requests_running{engine="0"}': 0,
                        'vllm:num_requests_waiting{engine="0"}': 3,
                    },
                    "decode": {
                        'vllm:kv_cache_usage_perc{engine="0"}': 0.4,
                        'vllm:num_requests_running{engine="0"}': 1,
                        'vllm:num_requests_waiting{engine="0"}': 2,
                    },
                },
            },
            {
                "time": 11.0,
                "vllm_metrics": {
                    "prefill": {
                        'vllm:kv_cache_usage_perc{engine="0"}': 0.5,
                        'vllm:num_requests_running{engine="0"}': 1,
                        'vllm:num_requests_waiting{engine="0"}': 0,
                    },
                    "decode": {
                        'vllm:kv_cache_usage_perc{engine="0"}': 0.75,
                        'vllm:num_requests_running{engine="0"}': 2,
                        'vllm:num_requests_waiting{engine="0"}': 1,
                    },
                },
            },
        ],
    )
    (run_dir / "decode_server.log").write_text(
        "\n".join(
            [
                "ReFlexKV planned BF16->INT4 KV block demotions for admission_waiting; target_release=10 actual_release=5 skipped_pages=5 bf16_free=1/16 bf16_free_before=1 plan_ms=0.5.",
                "ReFlexKV trace precision_budget request=req-0 max_int4_pages=8 priority=1.5 max_int4_fraction=0.5 release_budget_blocks=4 max_demote_per_window=2 request_priority=1 generated_decode_tokens=64 remaining_decode_tokens=192 prompt_pages=32.",
                "ReFlexKV trace candidate_breakdown reason=admission_waiting selection_policy=relevance_sparse raw_bf16_pages=16 eligible_full_unshared_pages=14 after_initial_recent_protection=12 after_low_risk_filter=8 after_request_budget_cap=6 after_sparse_window_quota=5 after_int4_pool_limit=5 selected_actual=5 frontier_levels=pinned:4,protected:4,candidate:8,eager_compressible:5,low_precision:0 rejection_reasons=shared_or_open:2,recent_or_initial:2,high_risk:4,request_budget:2,sparse_quota:1,frontier_optimizer:0,int4_pool_full:0.",
                "ReFlexKV trace page_metadata_plan reason=admission_waiting real_risk_requests=1 real_risk_pages=3 compressible_requests=1 compressible_pages=5 shadow_requests=1 shadow_pages=2 synthetic_requests=1 synthetic_pages=4.",
                "ReFlexKV trace admission_control request=req-0 requested_release=10 candidate_release_capacity=6 feasible_release=6 planned_release=0 actual_release=0 admission_success_after_demote=False admission_blocked=True admission_infeasible=True admission_wait_reduction=0 free_before=1 free_after_estimated=1 needed_blocks=16 reserve_blocks=2 landing_mixed_feasible=True landing_required_int4_blocks=11 landing_eligible_int4_blocks=12 landing_planned_int4_blocks=11 landing_residual_bf16_deficit=11 landing_reason=mixed_landing_feasible landing_metadata_source=real_risk landing_real_risk_pages=3 landing_explicit_compressible_pages=5 landing_synthetic_pages=0 blocked_reason=request_budget frontier_age=0 frontier_levels=pinned:4,protected:4,candidate:8,eager_compressible:5,low_precision:0 frontier_rejection_reasons=shared_or_open:2,recent_or_initial:2,high_risk:4,request_budget:2,sparse_quota:1,frontier_optimizer:0,int4_pool_full:0.",
                "ReFlexKV trace admission_control request=req-2 outcome=defer_full_sequence_reserve reason=no_progress_after_full_sequence_reserve blocked_reason=full_sequence_reserve frontier_age=2 frontier_rejection_reasons=request_budget:3,sparse_quota:2.",
                "ReFlexKV trace admission_control request=req-2 outcome=skip_admission_ticket blocked_reason=full_sequence_reserve required_blocks=512 next_retry_step=72 current_step=64.",
                "ReFlexKV trace landing_policy request=req-1 outcome=fallback_unmaterialized planned_pages=2 materialized=False reason=no_materialized_signal.",
                "ReFlexKV trace landing_materialize request=req-0 pages=11 layer_copies=352 kernel_launches=32 cpu_ms=2.0 gpu_ms=1.0.",
                "ReFlexKV trace landing_commit request=req-0 pages=11 committed=11.",
                "ReFlexKV trace demote_exec pages=12 layer_copies=384 kernel_launches=32 cpu_ms=1.0 gpu_ms=0.5.",
                "ReFlexKV trace step=7 phase=decode reqs=2 scheduled_tokens=2 max_query_len=1 max_seq_len=4096 demotions=1 kv_blocks_total=16 kv_blocks_bf16=12 kv_blocks_int4=4 bf16_capacity_blocks=16 int4_capacity_blocks=32 int4_budget_fraction=0.25 kv_int4_ratio=0.25 forward_gpu_ms=12.5.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "prefill_server.log").write_text(
        "ReFlexKV trace page_metadata_produce requests=1 pages=3 source=prefill_recorder.\n",
        encoding="utf-8",
    )
    (run_dir / "proxy.log").write_text(
        "ReFlexKV trace page_metadata_receive requests=1 pages=3 source=mooncake_worker_meta.\n",
        encoding="utf-8",
    )

    summary = mixed.write_mixed_predictions_and_scores(
        args=args,
        run_dir=run_dir,
        workload=workload,
        predictions=predictions,
        duration_seconds=3.0,
    )

    assert summary["serving_metrics"]["prefill"]["max_kv_cache_usage_pct"] == 98.0
    assert summary["serving_metrics"]["prefill"]["max_waiting"] == 3
    assert summary["serving_metrics"]["prefill"]["avg_running"] == 0.5
    assert summary["serving_metrics"]["decode"]["max_kv_cache_usage_pct"] == 75.0
    assert summary["serving_metrics"]["decode"]["max_running"] == 2
    assert summary["reflex_trace"]["demotion_event_count"] == 1
    assert summary["reflex_trace"]["demoted_pages_total"] == 12
    assert summary["reflex_trace"]["released_bf16_blocks_total"] == 5
    assert summary["reflex_trace"]["actual_release_blocks_total"] == 5
    assert summary["reflex_trace"]["planned_bf16_blocks_total"] == 10
    assert summary["reflex_trace"]["target_release_blocks_total"] == 10
    assert summary["reflex_trace"]["precision_budget_event_count"] == 1
    assert summary["reflex_trace"]["precision_budget_max_int4_pages_total"] == 8
    assert summary["reflex_trace"]["precision_budget_release_budget_total"] == 4
    assert summary["reflex_trace"]["precision_budget_priority_total"] == 1.5
    assert summary["reflex_trace"]["admission_control_event_count"] == 3
    assert summary["reflex_trace"]["admission_feasible_release_total"] == 6
    assert summary["reflex_trace"]["admission_planned_release_total"] == 0
    assert summary["reflex_trace"]["admission_infeasible_total"] == 1
    assert summary["reflex_trace"]["admission_mixed_landing_feasible_total"] == 1
    assert summary["reflex_trace"]["admission_required_int4_landing_total"] == 11
    assert summary["reflex_trace"]["admission_eligible_int4_landing_total"] == 12
    assert summary["reflex_trace"]["admission_planned_int4_landing_total"] == 11
    assert summary["reflex_trace"]["admission_residual_bf16_deficit_total"] == 11
    assert summary["reflex_trace"]["landing_materialize_event_count"] == 1
    assert summary["reflex_trace"]["landing_materialized_pages_total"] == 11
    assert summary["reflex_trace"]["landing_materialize_layer_copies_total"] == 352
    assert summary["reflex_trace"]["landing_materialize_gpu_ms_total"] == 1.0
    assert summary["reflex_trace"]["landing_commit_event_count"] == 1
    assert summary["reflex_trace"]["landing_committed_pages_total"] == 11
    assert summary["reflex_trace"]["landing_policy_event_count"] == 1
    assert summary["reflex_trace"]["landing_fallback_event_count"] == 1
    assert summary["reflex_trace"]["landing_fallback_pages_total"] == 2
    assert summary["reflex_trace"]["landing_fallback_unmaterialized_total"] == 1
    assert summary["reflex_trace"]["landing_fallback_unmaterialized_ratio"] == 2 / 11
    assert summary["reflex_trace"]["page_metadata_produce_event_count"] == 1
    assert summary["reflex_trace"]["page_metadata_produced_pages_total"] == 3
    assert summary["reflex_trace"]["page_metadata_receive_event_count"] == 1
    assert summary["reflex_trace"]["page_metadata_received_pages_total"] == 3
    assert summary["reflex_trace"]["page_metadata_plan_event_count"] == 1
    assert summary["reflex_trace"]["page_metadata_plan_real_risk_pages_total"] == 3
    assert summary["reflex_trace"]["page_metadata_plan_compressible_pages_total"] == 5
    assert summary["reflex_trace"]["page_metadata_plan_shadow_pages_total"] == 2
    assert summary["reflex_trace"]["page_metadata_plan_synthetic_pages_total"] == 4
    assert summary["reflex_trace"]["page_metadata_real_risk_coverage_ratio"] == 3 / 7
    assert summary["reflex_trace"]["landing_metadata_source_counts"] == {
        "real_risk": 1
    }
    assert summary["reflex_trace"]["landing_real_risk_pages_total"] == 3
    assert summary["reflex_trace"]["landing_explicit_compressible_pages_total"] == 5
    assert summary["reflex_trace"]["landing_synthetic_pages_total"] == 0
    assert summary["reflex_trace"]["candidate_breakdown_event_count"] == 1
    assert summary["reflex_trace"]["candidate_after_low_risk_filter_total"] == 8
    assert summary["reflex_trace"]["candidate_after_sparse_window_quota_total"] == 5
    assert summary["reflex_trace"]["candidate_selected_actual_total"] == 5
    assert summary["reflex_trace"]["candidate_frontier_level_totals"] == {
        "pinned": 4,
        "protected": 4,
        "candidate": 8,
        "eager_compressible": 5,
        "low_precision": 0,
    }
    assert summary["reflex_trace"]["candidate_rejection_reason_totals"] == {
        "shared_or_open": 2,
        "recent_or_initial": 2,
        "high_risk": 4,
        "request_fraction_cap": 0,
        "quality_debt_cap": 0,
        "request_release_budget": 0,
        "short_decode_protection": 0,
        "reasoning_prompt_protection": 0,
        "request_budget": 2,
        "sparse_quota": 1,
        "frontier_optimizer": 0,
        "int4_pool_full": 0,
    }
    assert summary["reflex_trace"]["admission_blocked_reason_counts"] == {
        "request_budget": 1,
        "full_sequence_reserve": 2,
    }
    assert summary["reflex_trace"]["admission_outcome_counts"] == {
        "defer_full_sequence_reserve": 1,
        "skip_admission_ticket": 1,
    }
    assert summary["reflex_trace"]["admission_frontier_rejection_reason_totals"] == {
        "shared_or_open": 2,
        "recent_or_initial": 2,
        "high_risk": 4,
        "request_fraction_cap": 0,
        "quality_debt_cap": 0,
        "request_release_budget": 0,
        "short_decode_protection": 0,
        "reasoning_prompt_protection": 0,
        "request_budget": 5,
        "sparse_quota": 3,
        "frontier_optimizer": 0,
        "int4_pool_full": 0,
    }
    assert summary["reflex_trace"]["max_int4_ratio"] == 0.25
