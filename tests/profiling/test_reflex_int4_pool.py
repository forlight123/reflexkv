import unittest

from vllm.v1.core.precision_kv.demotion_planner import (
    DistanceDemotionPlanner,
    RequestBudgetCandidate,
    RequestPrecisionBudget,
    allocate_request_release_budgets,
)
from vllm.v1.core.precision_kv.types import (
    Int4BlockPool,
    PrecisionState,
    ReflexPageMeta,
    decode_block_table_entry,
    encode_bf16_block_id,
    encode_int4_block_id,
)


class ReflexInt4PoolTest(unittest.TestCase):

    def test_int4_pool_reuses_freed_blocks_in_fifo_order(self):
        pool = Int4BlockPool(num_blocks=3)

        self.assertEqual(pool.allocate(), 0)
        self.assertEqual(pool.allocate(), 1)
        pool.free(0)

        self.assertEqual(pool.allocate(), 0)
        self.assertEqual(pool.allocate(), 2)
        self.assertIsNone(pool.allocate())

    def test_int4_pool_can_reserve_specific_handoff_blocks(self):
        pool = Int4BlockPool(num_blocks=4)

        pool.reserve(2)

        self.assertEqual(pool.num_allocated_blocks, 1)
        self.assertEqual(pool.num_free_blocks, 3)
        self.assertEqual(pool.allocate(), 0)
        self.assertEqual(pool.allocate(), 1)
        self.assertEqual(pool.allocate(), 3)
        self.assertIsNone(pool.allocate())

        with self.assertRaises(ValueError):
            pool.reserve(2)

    def test_int4_pool_rejects_double_free_and_invalid_free(self):
        pool = Int4BlockPool(num_blocks=2)
        block_id = pool.allocate()
        assert block_id is not None

        pool.free(block_id)

        with self.assertRaises(ValueError):
            pool.free(block_id)
        with self.assertRaises(ValueError):
            pool.free(3)

    def test_block_table_encoding_preserves_bf16_and_marks_int4_negative(self):
        self.assertEqual(encode_bf16_block_id(7), 7)
        self.assertEqual(encode_int4_block_id(2), -3)
        self.assertEqual(decode_block_table_entry(7), (PrecisionState.BF16, 7))
        self.assertEqual(decode_block_table_entry(-3), (PrecisionState.INT4, 2))

    def test_distance_planner_demotes_earliest_full_unshared_pages(self):
        pool = Int4BlockPool(num_blocks=4)
        pages = {
            "req-a": [
                ReflexPageMeta("req-a", 0, PrecisionState.BF16, 10, None),
                ReflexPageMeta("req-a", 1, PrecisionState.BF16, 11, None),
                ReflexPageMeta("req-a", 2, PrecisionState.BF16, 12, None),
                ReflexPageMeta("req-a", 3, PrecisionState.BF16, 13, None),
                ReflexPageMeta("req-a", 4, PrecisionState.BF16, 14, None),
            ]
        }

        plan = DistanceDemotionPlanner(keep_recent_pages=2).plan(
            pages, target_bf16_blocks=3, int4_pool=pool
        )

        self.assertEqual([item.page_idx for item in plan.items], [0, 1, 2])
        self.assertEqual([item.bf16_block_id for item in plan.items], [10, 11, 12])
        self.assertEqual([item.int4_block_id for item in plan.items], [0, 1, 2])
        self.assertEqual([item.encoded_block_table_id for item in plan.items],
                         [-1, -2, -3])

    def test_distance_planner_skips_shared_partial_and_already_int4_pages(self):
        pool = Int4BlockPool(num_blocks=4)
        pages = {
            "req-a": [
                ReflexPageMeta("req-a", 0, PrecisionState.BF16, 20, None,
                               is_shared=True),
                ReflexPageMeta("req-a", 1, PrecisionState.INT4, None, 0),
                ReflexPageMeta("req-a", 2, PrecisionState.BF16, 22, None),
                ReflexPageMeta("req-a", 3, PrecisionState.BF16, 23, None,
                               is_full=False),
                ReflexPageMeta("req-a", 4, PrecisionState.BF16, 24, None),
                ReflexPageMeta("req-a", 5, PrecisionState.BF16, 25, None),
            ]
        }

        plan = DistanceDemotionPlanner(keep_recent_pages=1).plan(
            pages, target_bf16_blocks=3, int4_pool=pool
        )

        self.assertEqual([item.page_idx for item in plan.items], [2, 4])
        self.assertEqual(pool.num_free_blocks, 2)

    def test_distance_planner_respects_request_precision_budgets(self):
        pool = Int4BlockPool(num_blocks=6)
        pages = {
            "short": [
                ReflexPageMeta("short", idx, PrecisionState.BF16, 100 + idx, None)
                for idx in range(4)
            ],
            "long": [
                ReflexPageMeta("long", idx, PrecisionState.BF16, 200 + idx, None)
                for idx in range(8)
            ],
        }

        plan = DistanceDemotionPlanner(keep_recent_pages=0).plan(
            pages,
            target_bf16_blocks=4,
            int4_pool=pool,
            request_precision_budgets={
                "short": RequestPrecisionBudget(
                    max_int4_pages=0,
                    priority=0.0,
                    max_int4_fraction=0.0,
                ),
                "long": RequestPrecisionBudget(
                    max_int4_pages=4,
                    priority=10.0,
                    max_int4_fraction=0.5,
                ),
            },
        )

        self.assertEqual([item.request_id for item in plan.items], ["long"] * 4)
        self.assertEqual([item.page_idx for item in plan.items], [0, 1, 2, 3])

    def test_global_budget_allocator_splits_bf16_release_blocks(self):
        budgets = allocate_request_release_budgets(
            [
                RequestBudgetCandidate(
                    request_id="warm-long",
                    capacity_blocks=4,
                    utility=10.0,
                ),
                RequestBudgetCandidate(
                    request_id="warm-medium",
                    capacity_blocks=2,
                    utility=5.0,
                ),
                RequestBudgetCandidate(
                    request_id="cold-large",
                    capacity_blocks=0,
                    utility=100.0,
                ),
            ],
            target_bf16_blocks=5,
        )

        self.assertEqual(budgets, {
            "warm-long": 3,
            "warm-medium": 2,
        })

    def test_global_budget_allocator_redistributes_after_capacity_saturates(self):
        budgets = allocate_request_release_budgets(
            [
                RequestBudgetCandidate(
                    request_id="tiny-high-priority",
                    capacity_blocks=1,
                    utility=100.0,
                ),
                RequestBudgetCandidate(
                    request_id="large-low-priority",
                    capacity_blocks=16,
                    utility=1.0,
                ),
            ],
            target_bf16_blocks=6,
        )

        self.assertEqual(budgets, {
            "tiny-high-priority": 1,
            "large-low-priority": 5,
        })

    def test_distance_planner_respects_per_request_release_budgets(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta("req-a", idx, PrecisionState.BF16, 100 + idx, None)
                for idx in range(4)
            ],
            "req-b": [
                ReflexPageMeta("req-b", idx, PrecisionState.BF16, 200 + idx, None)
                for idx in range(4)
            ],
        }

        plan = DistanceDemotionPlanner(keep_recent_pages=0).plan(
            pages,
            target_bf16_blocks=6,
            int4_pool=pool,
            request_precision_budgets={
                "req-a": RequestPrecisionBudget(
                    max_int4_pages=4,
                    priority=10.0,
                    release_budget_blocks=1,
                ),
                "req-b": RequestPrecisionBudget(
                    max_int4_pages=4,
                    priority=1.0,
                    release_budget_blocks=2,
                ),
            },
        )

        self.assertEqual(
            [(item.request_id, item.page_idx) for item in plan.items],
            [("req-a", 0), ("req-b", 0), ("req-b", 1)],
        )

    def test_prefill_guided_planner_only_demotes_low_risk_pages(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    idx,
                    PrecisionState.BF16,
                    100 + idx,
                    None,
                    compressible=(idx in {6, 3}),
                    prefill_risk={
                        6: 0.05,
                        3: 0.10,
                    }.get(idx),
                )
                for idx in range(10)
            ]
        }

        plan = DistanceDemotionPlanner(
            keep_initial_pages=2,
            keep_recent_pages=2,
            low_risk_only=True,
        ).plan(
            pages,
            target_bf16_blocks=4,
            int4_pool=pool,
        )

        self.assertEqual([item.page_idx for item in plan.items], [6, 3])

    def test_prefill_guided_planner_selects_sparsely_per_window(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    idx,
                    PrecisionState.BF16,
                    100 + idx,
                    None,
                    compressible=True,
                    prefill_risk=idx / 100.0,
                )
                for idx in range(16)
            ]
        }

        plan = DistanceDemotionPlanner(
            keep_recent_pages=0,
            low_risk_only=True,
            sparse_window_pages=8,
            max_demote_per_window=1,
        ).plan(
            pages,
            target_bf16_blocks=4,
            int4_pool=pool,
        )

        self.assertEqual([item.page_idx for item in plan.items], [0, 8])

    def test_relevance_planner_prefers_decode_pages_without_prompt_risk(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    idx,
                    PrecisionState.BF16,
                    100 + idx,
                    None,
                    compressible=True,
                    prefill_risk=0.01 + idx / 100.0,
                    is_prompt_page=True,
                )
                for idx in range(4)
            ]
            + [
                ReflexPageMeta(
                    "req-a",
                    idx,
                    PrecisionState.BF16,
                    100 + idx,
                    None,
                    compressible=True,
                    prefill_risk=None,
                    is_prompt_page=False,
                )
                for idx in range(4, 8)
            ],
        }

        plan = DistanceDemotionPlanner(
            keep_recent_pages=0,
            low_risk_only=True,
            selection_policy="relevance_sparse",
        ).plan(
            pages,
            target_bf16_blocks=4,
            int4_pool=pool,
        )

        self.assertEqual([item.page_idx for item in plan.items], [4, 5, 6, 7])
        self.assertTrue(all(not item.is_prompt_page for item in plan.items))

    def test_planner_reports_candidate_capacity_after_page_filters(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    idx,
                    PrecisionState.BF16,
                    100 + idx,
                    None,
                    compressible=(idx % 2 == 0),
                    prefill_risk=float(idx),
                )
                for idx in range(10)
            ]
        }

        plan = DistanceDemotionPlanner(
            keep_initial_pages=1,
            keep_recent_pages=1,
            low_risk_only=True,
        ).plan(
            pages,
            target_bf16_blocks=2,
            int4_pool=pool,
        )

        self.assertEqual([item.page_idx for item in plan.items], [2, 4])
        self.assertEqual(plan.candidate_bf16_blocks, 4)

    def test_planner_dry_run_reports_frontier_without_allocating_int4(self):
        pool = Int4BlockPool(num_blocks=2)
        pages = {
            "req-a": [
                ReflexPageMeta("req-a", idx, PrecisionState.BF16, 100 + idx, None)
                for idx in range(4)
            ]
        }

        plan = DistanceDemotionPlanner(keep_recent_pages=0).plan(
            pages,
            target_bf16_blocks=3,
            int4_pool=pool,
            dry_run=True,
        )

        self.assertEqual(plan.items, [])
        self.assertEqual(plan.candidate_bf16_blocks, 2)
        self.assertEqual(plan.released_bf16_blocks, 2)
        self.assertEqual(plan.candidate_breakdown.selected_actual, 2)
        self.assertEqual(pool.num_free_blocks, 2)

    def test_planner_reports_candidate_breakdown_by_filter_stage(self):
        pool = Int4BlockPool(num_blocks=1)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    0,
                    PrecisionState.BF16,
                    100,
                    None,
                    compressible=True,
                    prefill_risk=0.0,
                ),
                ReflexPageMeta("req-a", 1, PrecisionState.INT4, None, 1),
                ReflexPageMeta(
                    "req-a",
                    2,
                    PrecisionState.BF16,
                    102,
                    None,
                    is_shared=True,
                    compressible=True,
                    prefill_risk=0.0,
                ),
                ReflexPageMeta(
                    "req-a",
                    3,
                    PrecisionState.BF16,
                    103,
                    None,
                    is_full=False,
                    compressible=True,
                    prefill_risk=0.0,
                ),
                ReflexPageMeta(
                    "req-a",
                    4,
                    PrecisionState.BF16,
                    104,
                    None,
                    compressible=True,
                    prefill_risk=0.4,
                ),
                ReflexPageMeta(
                    "req-a",
                    5,
                    PrecisionState.BF16,
                    105,
                    None,
                    compressible=True,
                    prefill_risk=0.1,
                ),
                ReflexPageMeta(
                    "req-a",
                    6,
                    PrecisionState.BF16,
                    106,
                    None,
                    compressible=False,
                    prefill_risk=0.0,
                ),
                ReflexPageMeta(
                    "req-a",
                    7,
                    PrecisionState.BF16,
                    107,
                    None,
                    compressible=True,
                    prefill_risk=0.2,
                ),
                ReflexPageMeta(
                    "req-a",
                    8,
                    PrecisionState.BF16,
                    108,
                    None,
                    compressible=True,
                    prefill_risk=0.3,
                ),
                ReflexPageMeta(
                    "req-a",
                    9,
                    PrecisionState.BF16,
                    109,
                    None,
                    compressible=True,
                    prefill_risk=0.0,
                ),
            ]
        }

        plan = DistanceDemotionPlanner(
            keep_initial_pages=1,
            keep_recent_pages=1,
            low_risk_only=True,
            sparse_window_pages=4,
            max_demote_per_window=1,
            selection_policy="relevance_sparse",
        ).plan(
            pages,
            target_bf16_blocks=2,
            int4_pool=pool,
            request_precision_budgets={
                "req-a": RequestPrecisionBudget(
                    max_int4_pages=4,
                    release_budget_blocks=3,
                ),
            },
        )

        self.assertEqual([item.page_idx for item in plan.items], [5])
        self.assertEqual(plan.candidate_bf16_blocks, 1)
        self.assertEqual(plan.candidate_breakdown.raw_bf16_pages, 9)
        self.assertEqual(
            plan.candidate_breakdown.eligible_full_unshared_pages, 7
        )
        self.assertEqual(
            plan.candidate_breakdown.after_initial_recent_protection, 5
        )
        self.assertEqual(plan.candidate_breakdown.after_low_risk_filter, 4)
        self.assertEqual(plan.candidate_breakdown.after_request_budget_cap, 3)
        self.assertEqual(plan.candidate_breakdown.after_sparse_window_quota, 2)
        self.assertEqual(plan.candidate_breakdown.after_int4_pool_limit, 1)
        self.assertEqual(plan.candidate_breakdown.selected_actual, 1)

    def test_planner_reports_lifecycle_blocker_breakdown(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta("req-a", 0, PrecisionState.BF16, 100, None),
                ReflexPageMeta(
                    "req-a",
                    1,
                    PrecisionState.BF16,
                    101,
                    None,
                    is_full=False,
                    is_remote_inflight=True,
                ),
                ReflexPageMeta(
                    "req-a",
                    2,
                    PrecisionState.BF16,
                    102,
                    None,
                    is_full=False,
                ),
                ReflexPageMeta(
                    "req-a",
                    3,
                    PrecisionState.BF16,
                    103,
                    None,
                    is_shared=True,
                ),
                ReflexPageMeta(
                    "req-a",
                    4,
                    PrecisionState.BF16,
                    104,
                    None,
                    is_prompt_protected=True,
                ),
                ReflexPageMeta(
                    "req-a",
                    5,
                    PrecisionState.BF16,
                    105,
                    None,
                    is_full=False,
                    is_request_protected=True,
                ),
                ReflexPageMeta(
                    "req-a",
                    6,
                    PrecisionState.BF16,
                    106,
                    None,
                    is_shared=True,
                    copy_on_demote=True,
                ),
            ]
        }

        plan = DistanceDemotionPlanner(keep_recent_pages=0).plan(
            pages,
            target_bf16_blocks=8,
            int4_pool=pool,
            dry_run=True,
        )

        breakdown = plan.candidate_breakdown
        self.assertEqual(breakdown.raw_bf16_pages, 7)
        self.assertEqual(breakdown.open_bf16_pages, 3)
        self.assertEqual(breakdown.remote_inflight_bf16_pages, 1)
        self.assertEqual(breakdown.open_tail_bf16_pages, 1)
        self.assertEqual(breakdown.request_protected_bf16_pages, 1)
        self.assertEqual(breakdown.shared_bf16_pages, 1)
        self.assertEqual(breakdown.prompt_protected_bf16_pages, 1)
        self.assertEqual(breakdown.copy_on_demote_pages, 1)
        self.assertEqual(breakdown.eligible_full_unshared_pages, 2)

    def test_planner_excludes_request_protected_full_pages(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    0,
                    PrecisionState.BF16,
                    100,
                    None,
                    is_request_protected=True,
                ),
                ReflexPageMeta("req-a", 1, PrecisionState.BF16, 101, None),
            ]
        }

        plan = DistanceDemotionPlanner(keep_recent_pages=0).plan(
            pages,
            target_bf16_blocks=2,
            int4_pool=pool,
        )

        self.assertEqual([item.page_idx for item in plan.items], [1])
        self.assertEqual(plan.candidate_breakdown.request_protected_bf16_pages, 1)
        self.assertEqual(plan.candidate_breakdown.eligible_full_unshared_pages, 1)

    def test_planner_can_run_distance_only_ablation_with_relevance_scores(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    idx,
                    PrecisionState.BF16,
                    100 + idx,
                    None,
                    compressible=(idx in {1, 4}),
                    prefill_risk={1: 0.9, 4: 0.1}.get(idx),
                )
                for idx in range(6)
            ]
        }

        plan = DistanceDemotionPlanner(
            keep_recent_pages=0,
            low_risk_only=False,
            selection_policy="distance",
        ).plan(
            pages,
            target_bf16_blocks=3,
            int4_pool=pool,
        )

        self.assertEqual([item.page_idx for item in plan.items], [0, 1, 2])

    def test_planner_can_run_relevance_without_sparse_window_ablation(self):
        pool = Int4BlockPool(num_blocks=8)
        pages = {
            "req-a": [
                ReflexPageMeta(
                    "req-a",
                    idx,
                    PrecisionState.BF16,
                    100 + idx,
                    None,
                    compressible=True,
                    prefill_risk=float(idx),
                )
                for idx in range(8)
            ]
        }

        plan = DistanceDemotionPlanner(
            keep_recent_pages=0,
            low_risk_only=True,
            sparse_window_pages=4,
            max_demote_per_window=1,
            selection_policy="relevance",
        ).plan(
            pages,
            target_bf16_blocks=3,
            int4_pool=pool,
        )

        self.assertEqual([item.page_idx for item in plan.items], [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
