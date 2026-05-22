import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.profiling.summarize_reflex_pd_pressure import (
    parse_reflex_trace_events,
    summarize_run,
)


class SummarizeReFlexPDPressureTest(unittest.TestCase):
    def test_summarize_run_computes_trace_summary_and_timeline_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_a"
            run_dir.mkdir()
            (run_dir / "config.json").write_text(
                json.dumps(
                    {
                        "decode_kv_cache_dtype": "reflex_int4",
                        "input_len": 4096,
                        "output_len": 16,
                        "num_prompts": 2,
                        "max_concurrency": 2,
                        "num_gpu_blocks_override": 512,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "bench_result.json").write_text(
                json.dumps(
                    {
                        "completed": 2,
                        "failed": 0,
                        "request_throughput": 0.25,
                        "total_token_throughput": 1028.0,
                        "mean_tpot_ms": 11.0,
                        "p95_tpot_ms": 12.0,
                        "p99_tpot_ms": 13.0,
                        "mean_ttft_ms": 21.0,
                        "p95_ttft_ms": 22.0,
                        "p99_ttft_ms": 23.0,
                        "mean_e2el_ms": 31.0,
                        "p95_e2el_ms": 32.0,
                        "p99_e2el_ms": 33.0,
                    }
                ),
                encoding="utf-8",
            )
            samples = [
                {
                    "time": 10.0,
                    "vllm_metrics": {
                        "decode": {
                            'vllm:kv_cache_usage_perc{engine="0"}': 0.25,
                            'vllm:num_requests_running{engine="0"}': 1,
                            'vllm:num_requests_waiting{engine="0"}': 0,
                        }
                    },
                },
                {
                    "time": 11.5,
                    "vllm_metrics": {
                        "decode": {
                            'vllm:kv_cache_usage_perc{engine="0"}': 0.75,
                            'vllm:num_requests_running{engine="0"}': 2,
                            'vllm:num_requests_waiting{engine="0"}': 3,
                        }
                    },
                },
            ]
            (run_dir / "metrics_samples.jsonl").write_text(
                "\n".join(json.dumps(sample) for sample in samples) + "\n",
                encoding="utf-8",
            )
            (run_dir / "decode_server.log").write_text(
                "\n".join(
                    [
                        "ReFlexKV planned 258/258 BF16->INT4 KV block demotions for admission_waiting; bf16_free=0/512 bf16_free_before=0 plan_ms=1.836.",
                        "ReFlexKV planned BF16->INT4 KV block demotions for admission_waiting; target_release=128 actual_release=64 skipped_pages=64 bf16_free=0/512 bf16_free_before=0 plan_ms=1.500.",
                        "ReFlexKV trace precision_budget request=req-0 max_int4_pages=32 priority=2.5 max_int4_fraction=0.5 release_budget_blocks=16 max_demote_per_window=2 request_priority=1 generated_decode_tokens=64 remaining_decode_tokens=192 prompt_pages=256.",
                        "ReFlexKV trace candidate_breakdown reason=admission_waiting selection_policy=relevance_sparse raw_bf16_pages=512 open_bf16_pages=12 remote_inflight_bf16_pages=3 open_tail_bf16_pages=4 request_protected_bf16_pages=5 shared_bf16_pages=2 prompt_protected_bf16_pages=10 copy_on_demote_pages=1 eligible_full_unshared_pages=500 after_initial_recent_protection=480 after_low_risk_filter=120 after_request_budget_cap=96 after_sparse_window_quota=64 after_int4_pool_limit=64 selected_actual=64.",
                        "ReFlexKV trace admission_control request=req-0 requested_release=128 candidate_release_capacity=96 feasible_release=96 planned_release=0 actual_release=64 admission_success_after_demote=False admission_blocked=True admission_infeasible=True admission_wait_reduction=64 free_before=0 free_after_estimated=64 needed_blocks=256 reserve_blocks=32 landing_mixed_feasible=True landing_required_int4_blocks=128 landing_eligible_int4_blocks=160 landing_planned_int4_blocks=128 landing_residual_bf16_deficit=128 landing_reason=mixed_landing_feasible.",
                        "ReFlexKV trace admission_control request=req-ok requested_release=8 candidate_release_capacity=64 feasible_release=64 planned_release=8 actual_release=8 admission_success_after_demote=True admission_blocked=False admission_infeasible=False admission_wait_reduction=8 free_before=16 free_after_estimated=24 needed_blocks=16 reserve_blocks=0 landing_mixed_feasible=False landing_required_int4_blocks=0 landing_eligible_int4_blocks=0 landing_planned_int4_blocks=0 landing_residual_bf16_deficit=0 landing_reason=none blocked_reason=none frontier_rejection_reasons=request_budget:99,sparse_quota:88.",
                        "ReFlexKV trace landing_contract request=req-0 pages=128 direct=True required_blocks=128 planned_blocks=128 reason=mixed_landing_feasible.",
                        "ReFlexKV trace landing_policy request=req-1 outcome=fallback_unmaterialized planned_pages=4 materialized=False reason=no_materialized_signal.",
                        "ReFlexKV trace page_metadata_produce requests=2 pages=512 source=prefill_recorder.",
                        "ReFlexKV trace page_metadata_receive requests=2 pages=512 source=mooncake_worker_meta.",
                        "ReFlexKV trace page_metadata_plan reason=admission_waiting real_risk_requests=2 real_risk_pages=512 compressible_requests=2 compressible_pages=128 shadow_requests=2 shadow_pages=4 synthetic_requests=0 synthetic_pages=0.",
                        "ReFlexKV trace landing_materialize request=req-0 pages=128 layer_copies=4096 kernel_launches=32 cpu_ms=2.0 gpu_ms=1.25.",
                        "ReFlexKV trace landing_commit request=req-0 pages=128 committed=128.",
                        "ReFlexKV trace recovery_plan reason=background_promotion promoted_pages=1 free_ratio=0.6500.",
                        "ReFlexKV trace recovery_exec pages=1 layer_copies=32 cpu_ms=0.25.",
                        "ReFlexKV trace demote_exec pages=258 layer_copies=8256 kernel_launches=32 cpu_ms=1.908 gpu_ms=1.857.",
                        "ReFlexKV trace step=4 phase=decode reqs=2 scheduled_tokens=2 max_query_len=1 max_seq_len=4098 demotions=0 kv_blocks_total=514 kv_blocks_bf16=511 kv_blocks_int4=3 kv_int4_ratio=0.0058 preprocess_cpu_ms=2.585 forward_cpu_ms=781.198 forward_gpu_ms=781.145 postprocess_cpu_ms=1.346 postprocess_gpu_ms=1.311.",
                        "ReFlexKV trace step=5 phase=decode reqs=2 scheduled_tokens=2 max_query_len=1 max_seq_len=4099 demotions=1 kv_blocks_total=514 kv_blocks_bf16=256 kv_blocks_int4=258 kv_int4_ratio=0.5019 preprocess_cpu_ms=1.0 forward_cpu_ms=700.0 forward_gpu_ms=699.5 postprocess_cpu_ms=1.0 postprocess_gpu_ms=1.0.",
                        "ReFlexKV trace attention kernel=3d use_reflex_int4=True pages=258 gpu_ms=2.5.",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary, timeline, traces = summarize_run(run_dir)

            self.assertEqual(summary["run"], "run_a")
            self.assertEqual(summary["decode_kv_cache_dtype"], "reflex_int4")
            self.assertEqual(summary["input_len"], 4096)
            self.assertEqual(summary["output_len"], 16)
            self.assertEqual(summary["num_prompts"], 2)
            self.assertEqual(summary["max_concurrency"], 2)
            self.assertEqual(summary["num_gpu_blocks_override"], 512)
            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["req/s"], 0.25)
            self.assertEqual(summary["total_token_throughput"], 1028.0)
            self.assertEqual(summary["max_decode_running"], 2)
            self.assertEqual(summary["avg_decode_running"], 1.5)
            self.assertEqual(summary["max_decode_waiting"], 3)
            self.assertEqual(summary["avg_decode_waiting"], 1.5)
            self.assertEqual(summary["max_decode_kv_usage_pct"], 75.0)
            self.assertEqual(summary["demotion_event_count"], 1)
            self.assertEqual(summary["demoted_pages_total"], 258)
            self.assertEqual(summary["released_bf16_blocks_total"], 322)
            self.assertEqual(summary["actual_release_blocks_total"], 322)
            self.assertEqual(summary["planned_bf16_blocks_total"], 386)
            self.assertEqual(summary["target_release_blocks_total"], 386)
            self.assertEqual(summary["precision_budget_event_count"], 1)
            self.assertEqual(summary["precision_budget_max_int4_pages_total"], 32)
            self.assertEqual(summary["precision_budget_release_budget_total"], 16)
            self.assertEqual(summary["precision_budget_priority_total"], 2.5)
            self.assertEqual(summary["admission_control_event_count"], 2)
            self.assertEqual(summary["admission_requested_release_total"], 136)
            self.assertEqual(summary["admission_candidate_release_capacity_total"], 160)
            self.assertEqual(summary["admission_feasible_release_total"], 160)
            self.assertEqual(summary["admission_planned_release_total"], 8)
            self.assertEqual(summary["admission_actual_release_total"], 72)
            self.assertEqual(summary["admission_success_after_demote_total"], 1)
            self.assertEqual(summary["admission_blocked_total"], 1)
            self.assertEqual(summary["admission_infeasible_total"], 1)
            self.assertEqual(summary["admission_wait_reduction_total"], 72)
            self.assertEqual(summary["admission_mixed_landing_feasible_total"], 1)
            self.assertEqual(summary["admission_required_int4_landing_total"], 128)
            self.assertEqual(summary["admission_eligible_int4_landing_total"], 160)
            self.assertEqual(summary["admission_planned_int4_landing_total"], 128)
            self.assertEqual(summary["admission_residual_bf16_deficit_total"], 128)
            self.assertEqual(summary["landing_contract_event_count"], 1)
            self.assertEqual(summary["landing_contract_persisted_pages_total"], 128)
            self.assertEqual(summary["landing_contract_direct_pages_total"], 128)
            self.assertEqual(summary["landing_materialize_event_count"], 1)
            self.assertEqual(summary["landing_materialized_pages_total"], 128)
            self.assertEqual(summary["landing_materialize_layer_copies_total"], 4096)
            self.assertEqual(summary["landing_materialize_gpu_ms_total"], 1.25)
            self.assertEqual(summary["landing_commit_event_count"], 1)
            self.assertEqual(summary["landing_committed_pages_total"], 128)
            self.assertEqual(summary["landing_policy_event_count"], 1)
            self.assertEqual(summary["landing_fallback_event_count"], 1)
            self.assertEqual(summary["landing_fallback_pages_total"], 4)
            self.assertEqual(summary["landing_fallback_unmaterialized_total"], 1)
            self.assertEqual(
                summary["admission_blocked_frontier_rejection_reason_totals"],
                {
                    "shared_or_open": 0,
                    "recent_or_initial": 0,
                    "high_risk": 0,
                    "request_fraction_cap": 0,
                    "quality_debt_cap": 0,
                    "request_release_budget": 0,
                    "short_decode_protection": 0,
                    "reasoning_prompt_protection": 0,
                    "request_budget": 0,
                    "sparse_quota": 0,
                    "frontier_optimizer": 0,
                    "int4_pool_full": 0,
                },
            )
            self.assertEqual(summary["page_metadata_produce_event_count"], 1)
            self.assertEqual(summary["page_metadata_produced_requests_total"], 2)
            self.assertEqual(summary["page_metadata_produced_pages_total"], 512)
            self.assertEqual(summary["page_metadata_receive_event_count"], 1)
            self.assertEqual(summary["page_metadata_received_requests_total"], 2)
            self.assertEqual(summary["page_metadata_received_pages_total"], 512)
            self.assertEqual(summary["page_metadata_plan_event_count"], 1)
            self.assertEqual(summary["page_metadata_plan_real_risk_requests_total"], 2)
            self.assertEqual(summary["page_metadata_plan_real_risk_pages_total"], 512)
            self.assertEqual(summary["page_metadata_plan_compressible_pages_total"], 128)
            self.assertEqual(summary["page_metadata_plan_shadow_pages_total"], 4)
            self.assertEqual(summary["page_metadata_plan_synthetic_pages_total"], 0)
            self.assertEqual(summary["page_metadata_real_risk_coverage_ratio"], 1.0)
            self.assertEqual(summary["landing_fallback_unmaterialized_ratio"], 4 / 128)
            self.assertEqual(summary["recovery_plan_event_count"], 1)
            self.assertEqual(summary["background_promoted_pages_total"], 1)
            self.assertEqual(summary["recovery_exec_event_count"], 1)
            self.assertEqual(summary["recovery_exec_pages_total"], 1)
            self.assertEqual(summary["recovery_exec_layer_copies_total"], 32)
            self.assertEqual(summary["recovery_exec_cpu_ms_total"], 0.25)
            self.assertEqual(summary["candidate_breakdown_event_count"], 1)
            self.assertEqual(summary["candidate_raw_bf16_pages_total"], 512)
            self.assertEqual(summary["candidate_open_bf16_pages_total"], 12)
            self.assertEqual(
                summary["candidate_remote_inflight_bf16_pages_total"], 3
            )
            self.assertEqual(summary["candidate_open_tail_bf16_pages_total"], 4)
            self.assertEqual(
                summary["candidate_request_protected_bf16_pages_total"], 5
            )
            self.assertEqual(summary["candidate_shared_bf16_pages_total"], 2)
            self.assertEqual(
                summary["candidate_prompt_protected_bf16_pages_total"], 10
            )
            self.assertEqual(summary["candidate_copy_on_demote_pages_total"], 1)
            self.assertEqual(
                summary["candidate_eligible_full_unshared_pages_total"], 500
            )
            self.assertEqual(
                summary["candidate_after_initial_recent_protection_total"], 480
            )
            self.assertEqual(summary["candidate_after_low_risk_filter_total"], 120)
            self.assertEqual(summary["candidate_after_request_budget_cap_total"], 96)
            self.assertEqual(summary["candidate_after_sparse_window_quota_total"], 64)
            self.assertEqual(summary["candidate_after_int4_pool_limit_total"], 64)
            self.assertEqual(summary["candidate_selected_actual_total"], 64)
            self.assertEqual(summary["demotion_gpu_ms_total"], 1.857)
            self.assertEqual(summary["mean_int4_ratio"], (0.0058 + 0.5019) / 2)
            self.assertEqual(summary["max_int4_ratio"], 0.5019)
            self.assertEqual(summary["mean_forward_gpu_ms"], (781.145 + 699.5) / 2)
            self.assertEqual(summary["max_forward_gpu_ms"], 781.145)
            self.assertEqual(summary["attention_trace_event_count"], 1)
            self.assertEqual(summary["attention_gpu_ms_total"], 2.5)

            self.assertEqual(timeline[0]["elapsed_s"], 0.0)
            self.assertEqual(timeline[1]["elapsed_s"], 1.5)
            self.assertEqual(timeline[1]["decode_kv_cache_usage_pct"], 75.0)
            self.assertEqual(timeline[1]["decode_running"], 2)
            self.assertEqual(timeline[1]["decode_waiting"], 3)

            self.assertEqual([event["event"] for event in traces], ["planned", "planned", "precision_budget", "candidate_breakdown", "admission_control", "admission_control", "landing_contract", "landing_policy", "page_metadata_produce", "page_metadata_receive", "page_metadata_plan", "landing_materialize", "landing_commit", "recovery_plan", "recovery_exec", "demote_exec", "step", "step", "attention"])
            self.assertEqual(traces[0]["released_bf16_blocks"], 258)
            self.assertEqual(traces[1]["released_bf16_blocks"], 64)
            self.assertEqual(traces[1]["skipped_pages"], 64)
            self.assertEqual(traces[2]["release_budget_blocks"], 16)
            self.assertEqual(traces[3]["after_low_risk_filter"], 120)
            self.assertEqual(traces[4]["candidate_release_capacity"], 96)
            self.assertEqual(traces[4]["feasible_release"], 96)
            self.assertEqual(traces[4]["admission_infeasible"], True)
            self.assertEqual(traces[6]["pages"], 128)
            self.assertEqual(traces[6]["direct"], True)
            self.assertEqual(traces[7]["outcome"], "fallback_unmaterialized")
            self.assertEqual(traces[7]["planned_pages"], 4)
            self.assertEqual(traces[11]["pages"], 128)
            self.assertEqual(traces[12]["committed"], 128)
            self.assertEqual(traces[14]["pages"], 1)
            self.assertEqual(traces[15]["pages"], 258)
            self.assertEqual(traces[16]["kv_int4_ratio"], 0.0058)

    def test_missing_optional_files_yield_blank_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_missing"
            run_dir.mkdir()

            summary, timeline, traces = summarize_run(run_dir)

            self.assertEqual(summary["run"], "run_missing")
            self.assertIsNone(summary["completed"])
            self.assertIsNone(summary["max_decode_kv_usage_pct"])
            self.assertEqual(summary["demotion_event_count"], 0)
            self.assertEqual(summary["precision_budget_event_count"], 0)
            self.assertEqual(summary["admission_control_event_count"], 0)
            self.assertEqual(summary["landing_materialize_event_count"], 0)
            self.assertEqual(summary["landing_contract_event_count"], 0)
            self.assertEqual(summary["landing_contract_persisted_pages_total"], 0)
            self.assertEqual(summary["landing_commit_event_count"], 0)
            self.assertEqual(summary["landing_committed_pages_total"], 0)
            self.assertEqual(summary["landing_policy_event_count"], 0)
            self.assertEqual(summary["landing_fallback_event_count"], 0)
            self.assertEqual(summary["landing_fallback_pages_total"], 0)
            self.assertEqual(summary["page_metadata_produce_event_count"], 0)
            self.assertEqual(summary["page_metadata_receive_event_count"], 0)
            self.assertEqual(summary["page_metadata_plan_event_count"], 0)
            self.assertEqual(summary["recovery_plan_event_count"], 0)
            self.assertEqual(summary["recovery_exec_event_count"], 0)
            self.assertEqual(summary["attention_trace_event_count"], 0)
            self.assertEqual(timeline, [])
            self.assertEqual(traces, [])

    def test_parse_reflex_trace_events_ignores_unrelated_lines(self):
        events = parse_reflex_trace_events(
            [
                "ordinary server log line",
                "ReFlexKV trace attention kernel=3d use_reflex_int4=True gpu_ms=4.25.",
            ],
            run="run_b",
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["run"], "run_b")
        self.assertEqual(events[0]["event"], "attention")
        self.assertEqual(events[0]["gpu_ms"], 4.25)

    def test_write_trace_csv_from_cli(self):
        from scripts.profiling.summarize_reflex_pd_pressure import main

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run_cli"
            run_dir.mkdir()
            (run_dir / "decode_server.log").write_text(
                "ReFlexKV trace demote_exec pages=3 layer_copies=96 kernel_launches=2 cpu_ms=0.5 gpu_ms=0.25.\n",
                encoding="utf-8",
            )

            out = root / "summary.csv"
            trace_out = root / "trace.csv"
            rc = main([str(run_dir), "--out", str(out), "--trace-out", str(trace_out)])

            self.assertEqual(rc, 0)
            with trace_out.open(encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(rows[0]["event"], "demote_exec")
            self.assertEqual(rows[0]["pages"], "3")


if __name__ == "__main__":
    unittest.main()
