from collections import defaultdict
from types import SimpleNamespace

import pytest

from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.core.precision_kv.frontier import (
    AdmissionTicket,
    FeasibleFrontierCache,
    FeasibleFrontierSummary,
    PageCompressibilityLevel,
    RejectionReason,
)
from vllm.v1.core.precision_kv.demotion_planner import ReflexCandidateBreakdown
from vllm.v1.core.precision_kv.demotion_planner import DistanceDemotionPlanner
from vllm.v1.core.precision_kv.types import (
    Int4BlockPool,
    PrecisionState,
    ReflexPageMeta,
)


def test_frontier_summary_derives_page_levels_and_rejection_reasons():
    breakdown = ReflexCandidateBreakdown(
        raw_bf16_pages=100,
        eligible_full_unshared_pages=80,
        after_initial_recent_protection=60,
        after_low_risk_filter=30,
        after_request_budget_cap=10,
        after_sparse_window_quota=4,
        after_int4_pool_limit=2,
        selected_actual=2,
    )

    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=17,
        reason="admission_waiting",
        target_release=64,
        feasible_release=2,
        candidate_breakdown=breakdown,
    )

    assert summary.eligible_by_level[PageCompressibilityLevel.PINNED] == 40
    assert summary.eligible_by_level[PageCompressibilityLevel.PROTECTED] == 30
    assert summary.eligible_by_level[PageCompressibilityLevel.CANDIDATE] == 30
    assert (
        summary.eligible_by_level[
            PageCompressibilityLevel.EAGER_COMPRESSIBLE
        ]
        == 2
    )
    assert summary.eligible_by_level[PageCompressibilityLevel.LOW_PRECISION] == 0
    assert summary.blocked_by_reason[RejectionReason.SHARED_OR_OPEN] == 20
    assert summary.blocked_by_reason[RejectionReason.RECENT_OR_INITIAL] == 20
    assert summary.blocked_by_reason[RejectionReason.HIGH_RISK] == 30
    assert summary.blocked_by_reason[RejectionReason.REQUEST_BUDGET] == 20
    assert summary.blocked_by_reason[RejectionReason.SPARSE_QUOTA] == 6
    assert summary.blocked_by_reason[RejectionReason.INT4_POOL_FULL] == 2
    assert summary.cached_frontier_age(current_step=21) == 4


def test_frontier_summary_reports_optimizer_limit_separately_from_sparse_quota():
    breakdown = ReflexCandidateBreakdown(
        raw_bf16_pages=12,
        eligible_full_unshared_pages=12,
        after_initial_recent_protection=12,
        after_low_risk_filter=12,
        after_request_budget_cap=10,
        after_sparse_window_quota=6,
        after_frontier_optimizer=2,
        after_int4_pool_limit=2,
        selected_actual=2,
    )

    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=1,
        reason="allocation_failure",
        target_release=8,
        feasible_release=2,
        candidate_breakdown=breakdown,
    )

    assert summary.blocked_by_reason[RejectionReason.REQUEST_BUDGET] == 2
    assert summary.blocked_by_reason[RejectionReason.SPARSE_QUOTA] == 4
    assert summary.blocked_by_reason[RejectionReason.FRONTIER_OPTIMIZER] == 4
    assert summary.blocked_by_reason[RejectionReason.INT4_POOL_FULL] == 0
    assert summary.candidate_release_capacity == 2


def test_frontier_summary_splits_request_budget_rejection_reasons():
    breakdown = ReflexCandidateBreakdown(
        raw_bf16_pages=20,
        eligible_full_unshared_pages=20,
        after_initial_recent_protection=20,
        after_low_risk_filter=20,
        after_request_fraction_cap=12,
        after_quality_debt_cap=9,
        after_request_budget_cap=5,
        after_sparse_window_quota=5,
        after_frontier_optimizer=5,
        after_int4_pool_limit=5,
        selected_actual=5,
    )

    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=1,
        reason="admission_waiting",
        target_release=8,
        feasible_release=5,
        candidate_breakdown=breakdown,
    )

    assert summary.blocked_by_reason[RejectionReason.REQUEST_FRACTION_CAP] == 8
    assert summary.blocked_by_reason[RejectionReason.QUALITY_DEBT_CAP] == 3
    assert summary.blocked_by_reason[RejectionReason.REQUEST_RELEASE_BUDGET] == 4
    assert summary.blocked_by_reason[RejectionReason.REQUEST_BUDGET] == 15


def test_frontier_summary_attributes_zero_optimizer_budget_to_int4_pool():
    breakdown = ReflexCandidateBreakdown(
        raw_bf16_pages=12,
        eligible_full_unshared_pages=12,
        after_initial_recent_protection=12,
        after_low_risk_filter=12,
        after_request_budget_cap=10,
        after_sparse_window_quota=6,
        after_frontier_optimizer=0,
        after_int4_pool_limit=0,
        selected_actual=0,
        int4_free_blocks=0,
        frontier_optimizer_budget=0,
    )

    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=1,
        reason="allocation_failure",
        target_release=8,
        feasible_release=0,
        candidate_breakdown=breakdown,
    )

    assert summary.blocked_by_reason[RejectionReason.SPARSE_QUOTA] == 4
    assert summary.blocked_by_reason[RejectionReason.FRONTIER_OPTIMIZER] == 0
    assert summary.blocked_by_reason[RejectionReason.INT4_POOL_FULL] == 6


def test_frontier_cache_returns_only_fresh_matching_summary():
    breakdown = ReflexCandidateBreakdown(after_int4_pool_limit=7)
    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=10,
        reason="admission_waiting",
        target_release=64,
        feasible_release=7,
        candidate_breakdown=breakdown,
    )
    cache = FeasibleFrontierCache(max_age_steps=4)
    cache.update(summary)

    assert cache.get(
        reason="admission_waiting",
        target_release=64,
        current_step=13,
    ) is summary
    assert cache.get(
        reason="allocation_failure",
        target_release=64,
        current_step=13,
    ) is None
    assert cache.get(
        reason="admission_waiting",
        target_release=32,
        current_step=13,
    ) is summary
    assert cache.get(
        reason="admission_waiting",
        target_release=128,
        current_step=13,
    ) is None
    assert cache.get(
        reason="admission_waiting",
        target_release=64,
        current_step=15,
    ) is None


def test_frontier_cache_reuses_larger_fresh_target_summary():
    breakdown = ReflexCandidateBreakdown(after_int4_pool_limit=12)
    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=3,
        reason="admission_waiting",
        target_release=16,
        feasible_release=12,
        candidate_breakdown=breakdown,
    )
    cache = FeasibleFrontierCache(max_age_steps=4)
    cache.update(summary)

    smaller = cache.get(
        reason="admission_waiting",
        target_release=8,
        current_step=4,
    )
    larger = cache.get(
        reason="admission_waiting",
        target_release=24,
        current_step=4,
    )

    assert smaller is summary
    assert larger is None


def test_admission_ticket_retries_on_due_step_or_frontier_event():
    ticket = AdmissionTicket(
        request_id="waiting-0",
        required_blocks=512,
        blocked_reason="full_sequence_reserve",
        created_step=10,
        last_retry_step=10,
        next_retry_step=18,
        retry_on_events=frozenset({"bf16_freed"}),
    )

    assert ticket.should_retry(current_step=17, events=frozenset()) is False
    assert ticket.should_retry(
        current_step=17,
        events=frozenset({"bf16_freed"}),
    ) is True
    assert ticket.should_retry(current_step=18, events=frozenset()) is True


def test_demotion_dry_run_exports_commit_frontier_pages():
    pages = [
        ReflexPageMeta(
            request_id="req-0",
            page_idx=idx,
            precision=PrecisionState.BF16,
            bf16_block_id=100 + idx,
            int4_block_id=None,
            is_full=True,
            compressible=True,
        )
        for idx in range(8)
    ]
    planner = DistanceDemotionPlanner(
        keep_recent_pages=1,
        keep_initial_pages=1,
        selection_policy="relevance_sparse",
        sparse_window_pages=4,
        max_demote_per_window=2,
    )

    plan = planner.plan(
        {"req-0": pages},
        target_bf16_blocks=4,
        int4_pool=Int4BlockPool(8),
        dry_run=True,
    )

    assert plan.released_bf16_blocks == 4
    assert [page.page_idx for page in plan.candidate_pages] == [1, 2, 4, 5]
    assert plan.candidate_breakdown.after_sparse_window_quota == 4


class _ConcreteCacheManager(SingleTypeKVCacheManager):

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        use_eagle,
        alignment_tokens,
    ):
        return [], 0


def _make_cacheable_reflex_manager() -> SingleTypeKVCacheManager:
    manager = object.__new__(_ConcreteCacheManager)
    manager.block_size = 16
    manager.kv_cache_group_id = 0
    manager.reflex_int4_pool = Int4BlockPool(8)
    manager._null_block = SimpleNamespace(block_id=-1, block_hash=None, ref_cnt=0)
    manager.block_pool = SimpleNamespace(evict_blocks=lambda _blocks: None)
    blocks = [
        SimpleNamespace(block_id=100 + idx, block_hash=None, ref_cnt=1)
        for idx in range(8)
    ]
    manager.req_to_blocks = defaultdict(list, {"req-0": blocks})
    manager.req_to_reflex_block_ids = defaultdict(
        list,
        {"req-0": [block.block_id for block in blocks]},
    )
    manager.req_to_reflex_recovery_artifacts = defaultdict(dict)
    manager._pending_reflex_int4_demotions = []
    manager._pending_reflex_int4_recoveries = []
    manager._pending_reflex_bf16_releases = []
    manager._pending_reflex_bf16_release_count_by_request = defaultdict(int)
    manager._pending_reflex_bf16_release_pages_by_request = defaultdict(set)
    manager._reflex_recovered_pages_by_request = defaultdict(set)
    manager._last_reflex_int4_candidate_capacity = 0
    manager._last_reflex_int4_candidate_breakdown = ReflexCandidateBreakdown()
    manager._clear_reflex_int4_cached_frontier()
    return manager


def test_demotion_commit_reuses_cached_frontier_from_dry_run(monkeypatch):
    manager = _make_cacheable_reflex_manager()
    common_kwargs = dict(
        target_bf16_blocks=4,
        keep_recent_pages=1,
        keep_initial_pages=1,
        sparse_window_pages=4,
        max_demote_per_window=2,
        selection_policy="relevance_sparse",
        computed_tokens_by_request={"req-0": 128},
        prompt_tokens_by_request={"req-0": 0},
    )

    dry_run_release = manager.plan_reflex_int4_demotions(
        **common_kwargs,
        dry_run=True,
    )
    assert dry_run_release == 4

    monkeypatch.setattr(
        manager,
        "_build_reflex_page_metadata",
        lambda **_kwargs: pytest.fail("commit should use cached frontier"),
    )
    actual_release = manager.plan_reflex_int4_demotions(
        **common_kwargs,
        dry_run=False,
    )

    assert actual_release == 4
    assert [d.page_idx for d in manager._pending_reflex_int4_demotions] == [
        1,
        2,
        4,
        5,
    ]


def test_demotion_commit_reuses_cached_frontier_for_smaller_target(monkeypatch):
    manager = _make_cacheable_reflex_manager()
    common_kwargs = dict(
        keep_recent_pages=1,
        keep_initial_pages=1,
        sparse_window_pages=4,
        max_demote_per_window=2,
        selection_policy="relevance_sparse",
        computed_tokens_by_request={"req-0": 128},
        prompt_tokens_by_request={"req-0": 0},
    )

    dry_run_release = manager.plan_reflex_int4_demotions(
        target_bf16_blocks=4,
        **common_kwargs,
        dry_run=True,
    )
    assert dry_run_release == 4

    monkeypatch.setattr(
        manager,
        "_build_reflex_page_metadata",
        lambda **_kwargs: pytest.fail(
            "smaller commit should use cached frontier prefix"
        ),
    )
    actual_release = manager.plan_reflex_int4_demotions(
        target_bf16_blocks=2,
        **common_kwargs,
        dry_run=False,
    )

    assert actual_release == 2
    assert [d.page_idx for d in manager._pending_reflex_int4_demotions] == [1, 2]
