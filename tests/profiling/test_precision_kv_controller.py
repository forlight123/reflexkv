from vllm.v1.core.precision_kv.controller import (
    PrecisionAdmissionController,
    PrecisionAdmissionState,
)
from vllm.v1.core.precision_kv.landing import PrecisionLandingDecision


def test_precision_controller_plans_partial_release_for_infeasible_frontier():
    controller = PrecisionAdmissionController()

    decision = controller.plan_admission(
        PrecisionAdmissionState(
            request_id="waiting-0",
            needed_blocks=256,
            reserve_blocks=32,
            free_bf16_blocks=0,
            requested_release=128,
            feasible_release=96,
        )
    )

    assert decision.required_blocks == 288
    assert decision.feasible_release == 96
    assert decision.planned_release == 96
    assert decision.free_after_planned == 96
    assert decision.admission_infeasible is True
    assert decision.admission_success_after_planned is False


def test_precision_controller_allows_release_when_frontier_can_admit():
    controller = PrecisionAdmissionController()

    decision = controller.plan_admission(
        PrecisionAdmissionState(
            request_id="waiting-1",
            needed_blocks=512,
            reserve_blocks=32,
            free_bf16_blocks=500,
            requested_release=44,
            feasible_release=64,
        )
    )

    assert decision.required_blocks == 544
    assert decision.feasible_release == 44
    assert decision.planned_release == 44
    assert decision.admission_infeasible is False
    assert decision.admission_success_after_planned is True


def test_precision_controller_reduces_release_when_mixed_landing_closes_gap():
    controller = PrecisionAdmissionController()
    admission = controller.plan_admission(
        PrecisionAdmissionState(
            request_id="mixed-0",
            needed_blocks=512,
            reserve_blocks=32,
            free_bf16_blocks=400,
            requested_release=144,
            feasible_release=144,
        )
    )
    landing = PrecisionLandingDecision(
        request_id="mixed-0",
        required_blocks=544,
        bf16_deficit_blocks=144,
        residual_deficit_after_running=120,
        eligible_int4_landing_blocks=120,
        planned_int4_landing_blocks=120,
        planned_int4_landing_pages=tuple(range(120)),
        admission_feasible_with_landing=True,
        mixed_landing_required=True,
        reason="mixed_landing_feasible",
    )

    plan = controller.plan_precision_admission(
        admission_decision=admission,
        landing_decision=landing,
    )

    assert plan.planned_release == 24
    assert plan.mixed_landing_release_gap == 24
    assert plan.admission_infeasible is False
    assert plan.reason == "mixed_landing_reduces_release"


def test_precision_controller_marks_full_mixed_landing_feasible_without_release():
    controller = PrecisionAdmissionController()
    admission = controller.plan_admission(
        PrecisionAdmissionState(
            request_id="mixed-1",
            needed_blocks=512,
            reserve_blocks=32,
            free_bf16_blocks=400,
            requested_release=144,
            feasible_release=0,
        )
    )
    landing = PrecisionLandingDecision(
        request_id="mixed-1",
        required_blocks=544,
        bf16_deficit_blocks=144,
        residual_deficit_after_running=144,
        eligible_int4_landing_blocks=144,
        planned_int4_landing_blocks=144,
        planned_int4_landing_pages=tuple(range(144)),
        admission_feasible_with_landing=True,
        mixed_landing_required=True,
        reason="emergency_mixed_landing_feasible",
    )

    plan = controller.plan_precision_admission(
        admission_decision=admission,
        landing_decision=landing,
    )

    assert plan.planned_release == 0
    assert plan.mixed_landing_release_gap == 0
    assert plan.admission_infeasible is False
    assert plan.reason == "mixed_landing_closes_deficit"
