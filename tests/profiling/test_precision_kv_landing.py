from vllm.v1.core.precision_kv.landing import (
    PrecisionLandingPlanner,
    PrecisionLandingState,
)


def test_precision_landing_planner_uses_no_int4_when_bf16_fits():
    planner = PrecisionLandingPlanner()

    decision = planner.plan_landing(
        PrecisionLandingState(
            request_id="waiting-0",
            needed_blocks=128,
            reserve_blocks=16,
            free_bf16_blocks=160,
            running_feasible_release=0,
            eligible_int4_landing_blocks=64,
        )
    )

    assert decision.required_blocks == 144
    assert decision.bf16_deficit_blocks == 0
    assert decision.residual_deficit_after_running == 0
    assert decision.planned_int4_landing_blocks == 0
    assert decision.admission_feasible_with_landing is True
    assert decision.mixed_landing_required is False
    assert decision.reason == "bf16_fit"


def test_precision_landing_planner_uses_running_frontier_before_landing():
    planner = PrecisionLandingPlanner()

    decision = planner.plan_landing(
        PrecisionLandingState(
            request_id="waiting-1",
            needed_blocks=512,
            reserve_blocks=32,
            free_bf16_blocks=500,
            running_feasible_release=64,
            eligible_int4_landing_blocks=128,
        )
    )

    assert decision.required_blocks == 544
    assert decision.bf16_deficit_blocks == 44
    assert decision.residual_deficit_after_running == 0
    assert decision.planned_int4_landing_blocks == 0
    assert decision.admission_feasible_with_landing is True
    assert decision.mixed_landing_required is False
    assert decision.reason == "running_frontier_can_admit"


def test_precision_landing_planner_marks_mixed_landing_feasible():
    planner = PrecisionLandingPlanner()

    decision = planner.plan_landing(
        PrecisionLandingState(
            request_id="waiting-2",
            needed_blocks=1024,
            reserve_blocks=32,
            free_bf16_blocks=900,
            running_feasible_release=20,
            eligible_int4_landing_blocks=192,
        )
    )

    assert decision.required_blocks == 1056
    assert decision.bf16_deficit_blocks == 156
    assert decision.residual_deficit_after_running == 136
    assert decision.eligible_int4_landing_blocks == 192
    assert decision.planned_int4_landing_blocks == 136
    assert decision.admission_feasible_with_landing is True
    assert decision.mixed_landing_required is True
    assert decision.reason == "mixed_landing_feasible"


def test_precision_landing_planner_returns_page_level_landing_plan():
    planner = PrecisionLandingPlanner()

    decision = planner.plan_landing(
        PrecisionLandingState(
            request_id="waiting-page-plan",
            needed_blocks=1024,
            reserve_blocks=32,
            free_bf16_blocks=900,
            running_feasible_release=20,
            eligible_int4_landing_blocks=192,
            eligible_int4_landing_pages=tuple(range(200, 392)),
        )
    )

    assert decision.residual_deficit_after_running == 136
    assert decision.planned_int4_landing_blocks == 136
    assert decision.planned_int4_landing_pages == tuple(range(200, 336))
    assert decision.reason == "mixed_landing_feasible"


def test_precision_landing_planner_exposes_insufficient_landing_frontier():
    planner = PrecisionLandingPlanner()

    decision = planner.plan_landing(
        PrecisionLandingState(
            request_id="waiting-3",
            needed_blocks=1024,
            reserve_blocks=32,
            free_bf16_blocks=900,
            running_feasible_release=20,
            eligible_int4_landing_blocks=64,
        )
    )

    assert decision.required_blocks == 1056
    assert decision.residual_deficit_after_running == 136
    assert decision.planned_int4_landing_blocks == 0
    assert decision.admission_feasible_with_landing is False
    assert decision.mixed_landing_required is True
    assert decision.reason == "int4_landing_frontier_insufficient"


def test_precision_landing_planner_can_relax_reserve_for_direct_chunk_landing():
    planner = PrecisionLandingPlanner()

    decision = planner.plan_landing(
        PrecisionLandingState(
            request_id="remote-chunk",
            needed_blocks=32,
            reserve_blocks=32,
            free_bf16_blocks=0,
            running_feasible_release=0,
            eligible_int4_landing_blocks=32,
            eligible_int4_landing_pages=tuple(range(32)),
            allow_reserve_relaxation=True,
        )
    )

    assert decision.required_blocks == 64
    assert decision.bf16_deficit_blocks == 64
    assert decision.residual_deficit_after_running == 64
    assert decision.needed_deficit_after_running == 32
    assert decision.planned_int4_landing_blocks == 32
    assert decision.planned_int4_landing_pages == tuple(range(32))
    assert decision.admission_feasible_with_landing is True
    assert decision.mixed_landing_required is True
    assert decision.reserve_relaxed is True
    assert decision.reason == "mixed_landing_relaxed_reserve_feasible"
