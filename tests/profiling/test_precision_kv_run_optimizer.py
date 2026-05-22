from vllm.v1.core.precision_kv.demotion_planner import (
    DistanceDemotionPlanner,
    RequestPrecisionBudget,
)
from vllm.v1.core.precision_kv.run_optimizer import (
    DualPriceState,
    DualRunOptimizer,
    RunCandidate,
)
from vllm.v1.core.precision_kv.types import (
    Int4BlockPool,
    PrecisionState,
    ReflexPageMeta,
)


def test_dual_price_state_updates_from_pressure_signals():
    state = DualPriceState()

    updated = state.updated(
        kv_usage=0.95,
        kv_target=0.80,
        waiting_requests=8,
        waiting_target=2,
        migration_backlog=4,
        migration_target=1,
        eta=0.5,
    )

    assert updated.memory_price > state.memory_price
    assert updated.admission_price > state.admission_price
    assert updated.migration_price > state.migration_price
    assert updated.quality_price == state.quality_price


def test_dual_run_optimizer_prefers_lower_risk_run_when_quality_price_is_high():
    optimizer = DualRunOptimizer(
        dual_prices=DualPriceState(
            memory_price=1.0,
            admission_price=0.0,
            quality_price=10.0,
            migration_price=0.1,
            slo_price=0.0,
        ),
        sparse_window_pages=4,
        max_pages_per_window=2,
    )
    low_risk = RunCandidate(
        request_id="req",
        start_page=0,
        end_page=0,
        pages=(_page(0, risk=0.05),),
        saving_blocks=1,
        quality_risk=0.05,
        migration_cost=1.0,
        admission_benefit=1.0,
        slo_risk=0.0,
        is_prompt_run=False,
    )
    high_risk = RunCandidate(
        request_id="req",
        start_page=1,
        end_page=1,
        pages=(_page(1, risk=0.9),),
        saving_blocks=1,
        quality_risk=0.9,
        migration_cost=1.0,
        admission_benefit=1.0,
        slo_risk=0.0,
        is_prompt_run=False,
    )

    selected = optimizer.select_runs(
        [high_risk, low_risk],
        target_release=1,
        int4_capacity_blocks=2,
    )

    assert selected == [low_risk]


def test_frontier_dual_planner_selects_runs_and_respects_window_quota():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=False,
        sparse_window_pages=4,
        max_demote_per_window=2,
        selection_policy="frontier_dual",
    )
    pages = [
        _page(0, risk=0.05),
        _page(1, risk=0.05),
        _page(2, risk=0.05),
    ]
    pool = Int4BlockPool(num_blocks=8)

    plan = planner.plan(
        {"req": pages},
        target_bf16_blocks=3,
        int4_pool=pool,
    )

    assert [item.page_idx for item in plan.items] == [0, 1]
    assert plan.candidate_breakdown.after_sparse_window_quota == 2
    assert plan.candidate_breakdown.selected_actual == 2


def test_dual_run_optimizer_splits_oversized_run_for_small_release_target():
    optimizer = DualRunOptimizer(
        dual_prices=DualPriceState(),
        sparse_window_pages=32,
        max_pages_per_window=8,
        max_run_pages=8,
    )
    pages = [_page(idx, risk=0.05) for idx in range(4)]

    selected = optimizer.select_pages(
        pages,
        target_release=1,
        int4_capacity_blocks=8,
    )

    assert [page.page_idx for page in selected] == [0]


def test_dual_run_optimizer_builds_window_level_run_metadata():
    optimizer = DualRunOptimizer(
        dual_prices=DualPriceState(),
        sparse_window_pages=4,
    )

    runs = optimizer.build_runs([_page(idx, risk=0.05) for idx in range(4, 7)])

    assert len(runs) == 1
    assert runs[0].window_id == 1
    assert runs[0].num_pages == 3
    assert runs[0].is_decode_run is True
    assert runs[0].backlog_cost == 0.0
    assert runs[0].constraint_signature == "decode:window=1"


def test_frontier_dual_emergency_release_bypasses_step_budget_and_window_quota():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=False,
        sparse_window_pages=32,
        max_demote_per_window=1,
        selection_policy="frontier_dual",
        emergency_release=True,
    )
    pages = [_page(idx, risk=0.05) for idx in range(16)]
    pool = Int4BlockPool(num_blocks=16)

    plan = planner.plan(
        {"req": pages},
        target_bf16_blocks=4,
        int4_pool=pool,
        request_precision_budgets={
            "req": RequestPrecisionBudget(
                max_int4_pages=16,
                release_budget_blocks=1,
                max_demote_per_window=1,
            ),
        },
    )

    assert [item.page_idx for item in plan.items] == [0, 1, 2, 3]
    assert plan.candidate_breakdown.after_request_budget_cap == 16
    assert plan.candidate_breakdown.after_sparse_window_quota == 16
    assert plan.candidate_breakdown.after_frontier_optimizer == 4
    assert plan.candidate_breakdown.selected_actual == 4


def test_frontier_dual_emergency_release_bypasses_request_fraction_cap():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=False,
        sparse_window_pages=32,
        max_demote_per_window=1,
        selection_policy="frontier_dual",
        emergency_release=True,
    )
    pages = [_page(idx, risk=0.05) for idx in range(8)]
    pool = Int4BlockPool(num_blocks=8)

    plan = planner.plan(
        {"req": pages},
        target_bf16_blocks=2,
        int4_pool=pool,
        request_precision_budgets={
            "req": RequestPrecisionBudget(
                max_int4_pages=0,
                release_budget_blocks=0,
                max_demote_per_window=0,
            ),
        },
    )

    assert [item.page_idx for item in plan.items] == [0, 1]
    assert plan.candidate_breakdown.after_request_fraction_cap == 8
    assert plan.candidate_breakdown.after_request_budget_cap == 8


def test_frontier_dual_planner_selects_global_frontier_across_requests():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=False,
        sparse_window_pages=32,
        max_demote_per_window=8,
        selection_policy="frontier_dual",
        dual_price_state=DualPriceState(quality_price=10.0),
    )
    high_risk_page = _page(0, risk=0.9)
    low_risk_page = ReflexPageMeta(
        request_id="low",
        page_idx=0,
        precision=PrecisionState.BF16,
        bf16_block_id=200,
        int4_block_id=None,
        compressible=True,
        prefill_risk=0.01,
    )

    plan = planner.plan(
        {"req": [high_risk_page], "low": [low_risk_page]},
        target_bf16_blocks=1,
        int4_pool=Int4BlockPool(num_blocks=8),
    )

    assert [(item.request_id, item.page_idx) for item in plan.items] == [
        ("low", 0)
    ]
    assert plan.candidate_breakdown.after_request_budget_cap == 2
    assert plan.candidate_breakdown.after_sparse_window_quota == 2
    assert plan.candidate_breakdown.after_frontier_optimizer == 1


def test_frontier_dual_breakdown_separates_frontier_from_optimizer_selection():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=False,
        sparse_window_pages=32,
        max_demote_per_window=8,
        selection_policy="frontier_dual",
    )
    pages = [_page(idx, risk=0.05) for idx in range(4)]

    plan = planner.plan(
        {"req": pages},
        target_bf16_blocks=1,
        int4_pool=Int4BlockPool(num_blocks=8),
        dry_run=True,
    )

    assert plan.candidate_breakdown.after_request_budget_cap == 4
    assert plan.candidate_breakdown.after_sparse_window_quota == 4
    assert plan.candidate_breakdown.after_frontier_optimizer == 1
    assert plan.candidate_breakdown.after_int4_pool_limit == 1
    assert plan.candidate_breakdown.selected_actual == 1


def test_frontier_dual_breakdown_reports_empty_int4_pool_as_zero_budget():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=False,
        sparse_window_pages=32,
        max_demote_per_window=8,
        selection_policy="frontier_dual",
        emergency_release=True,
    )
    pages = [_page(idx, risk=0.05) for idx in range(4)]

    plan = planner.plan(
        {"req": pages},
        target_bf16_blocks=1,
        int4_pool=Int4BlockPool(num_blocks=0),
        dry_run=True,
    )

    assert plan.candidate_breakdown.after_sparse_window_quota == 4
    assert plan.candidate_breakdown.after_frontier_optimizer == 0
    assert plan.candidate_breakdown.after_int4_pool_limit == 0
    assert plan.candidate_breakdown.int4_free_blocks == 0
    assert plan.candidate_breakdown.frontier_optimizer_budget == 0


def test_request_budget_separates_prompt_and_decode_caps():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=False,
        sparse_window_pages=0,
        max_demote_per_window=0,
        selection_policy="oldest",
    )
    pages = [
        _page(0, risk=0.05, is_prompt_page=True),
        _page(1, risk=0.05, is_prompt_page=True),
        _page(2, risk=0.05, is_prompt_page=False),
        _page(3, risk=0.05, is_prompt_page=False),
    ]

    plan = planner.plan(
        {"req": pages},
        target_bf16_blocks=4,
        int4_pool=Int4BlockPool(num_blocks=8),
        request_precision_budgets={
            "req": RequestPrecisionBudget(
                max_int4_pages=4,
                max_prompt_int4_pages=1,
                max_decode_int4_pages=2,
                release_budget_blocks=4,
            ),
        },
    )

    assert [item.page_idx for item in plan.items] == [0, 2, 3]
    assert plan.candidate_breakdown.after_request_fraction_cap == 3
    assert plan.candidate_breakdown.after_request_budget_cap == 3


def test_planner_excludes_page_protected_decode_pages():
    planner = DistanceDemotionPlanner(
        keep_recent_pages=0,
        keep_initial_pages=0,
        max_int4_fraction_per_request=1.0,
        low_risk_only=True,
        sparse_window_pages=0,
        max_demote_per_window=0,
        selection_policy="relevance",
    )
    pages = [
        ReflexPageMeta(
            request_id="req",
            page_idx=idx,
            precision=PrecisionState.BF16,
            bf16_block_id=100 + idx,
            int4_block_id=None,
            is_full=True,
            compressible=True,
            is_page_protected=idx in {2, 3},
        )
        for idx in range(8)
    ]

    plan = planner.plan(
        {"req": pages},
        target_bf16_blocks=8,
        int4_pool=Int4BlockPool(num_blocks=8),
    )

    assert plan.released_bf16_blocks == 6
    assert {item.page_idx for item in plan.items} == {0, 1, 4, 5, 6, 7}


def _page(
    page_idx: int,
    *,
    risk: float,
    is_prompt_page: bool = False,
) -> ReflexPageMeta:
    return ReflexPageMeta(
        request_id="req",
        page_idx=page_idx,
        precision=PrecisionState.BF16,
        bf16_block_id=100 + page_idx,
        int4_block_id=None,
        compressible=risk <= 0.5,
        prefill_risk=risk,
        is_prompt_page=is_prompt_page,
    )
