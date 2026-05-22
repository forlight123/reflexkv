from vllm.v1.core.precision_kv.frontier import (
    AdmissionTicket,
    FeasibleFrontierCache,
    FeasibleFrontierSummary,
    PageCompressibilityLevel,
    RejectionReason,
)
from vllm.v1.core.precision_kv.demotion_planner import ReflexCandidateBreakdown


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
    ) is None
    assert cache.get(
        reason="admission_waiting",
        target_release=64,
        current_step=15,
    ) is None


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
