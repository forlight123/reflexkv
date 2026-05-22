import json
import tempfile
import unittest
from pathlib import Path

from scripts.profiling.summarize_fullkv_pressure import summarize_run


class SummarizeFullKVPressureTest(unittest.TestCase):
    def test_summarize_run_computes_slo_and_timeline_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "bench_result.json").write_text(
                json.dumps(
                    {
                        "num_prompts": 2,
                        "completed": 2,
                        "failed": 0,
                        "duration": 10.0,
                        "request_rate": "inf",
                        "max_concurrency": 2,
                        "request_throughput": 0.2,
                        "output_throughput": 20.0,
                        "total_token_throughput": 120.0,
                        "total_input_tokens": 3000,
                        "total_output_tokens": 200,
                        "input_lens": [1000, 2000],
                        "output_lens": [100, 100],
                        "ttfts": [1.0, 40.0],
                        "itls": [[0.02, 0.03], [0.06, 0.07]],
                        "mean_ttft_ms": 20500.0,
                        "p95_ttft_ms": 39000.0,
                        "p99_ttft_ms": 39800.0,
                        "mean_tpot_ms": 45.0,
                        "p95_tpot_ms": 65.0,
                        "p99_tpot_ms": 68.0,
                        "mean_e2el_ms": 1000.0,
                        "p95_e2el_ms": 2000.0,
                        "p99_e2el_ms": 3000.0,
                    }
                ),
                encoding="utf-8",
            )
            samples = [
                {
                    "time": 100.0,
                    "vllm_metrics": {
                        'vllm:kv_cache_usage_perc{engine="0"}': 0.1,
                        'vllm:num_requests_running{engine="0"}': 1,
                        'vllm:num_requests_waiting{engine="0"}': 0,
                    },
                },
                {
                    "time": 101.0,
                    "vllm_metrics": {
                        'vllm:kv_cache_usage_perc{engine="0"}': 0.9,
                        'vllm:num_requests_running{engine="0"}': 2,
                        'vllm:num_requests_waiting{engine="0"}': 3,
                    },
                },
            ]
            (run_dir / "metrics_samples.jsonl").write_text(
                "\n".join(json.dumps(s) for s in samples) + "\n",
                encoding="utf-8",
            )

            summary, timeline = summarize_run(
                run_dir,
                slo_ttft_ms=30_000,
                slo_tpot_ms=50,
                demotion_threshold_pct=85,
            )

            self.assertEqual(summary["completed"], 2)
            self.assertEqual(summary["failed"], 0)
            self.assertEqual(summary["avg_input_len"], 1500.0)
            self.assertEqual(summary["ttft_slo_violation_rate"], 0.5)
            self.assertEqual(summary["tpot_slo_violation_rate"], 0.5)
            self.assertEqual(summary["max_kv_cache_usage_pct"], 90.0)
            self.assertEqual(summary["max_requests_waiting"], 3)
            self.assertEqual(timeline[1]["pressure_action"], "demotion_would_trigger")


if __name__ == "__main__":
    unittest.main()
