# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Closed-loop planning primitives for precision-elastic KV cache.

This module is intentionally independent from INT4 kernels and block-table
mutation. It owns the scheduler-facing decision about whether a precision
transition can actually change admission state.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.core.precision_kv.landing import PrecisionLandingDecision


@dataclass(frozen=True)
class PrecisionAdmissionState:
    """KV-centered state needed to decide an admission-triggered transition."""

    request_id: str
    needed_blocks: int
    reserve_blocks: int
    free_bf16_blocks: int
    requested_release: int
    feasible_release: int


@dataclass(frozen=True)
class PrecisionAdmissionDecision:
    """Decision returned by the precision controller for one waiting request."""

    request_id: str
    required_blocks: int
    requested_release: int
    feasible_release: int
    planned_release: int
    free_after_planned: int
    admission_success_after_planned: bool
    admission_infeasible: bool


@dataclass(frozen=True)
class PrecisionAdmissionPlan:
    """Controller-owned admission plan consumed by the scheduler."""

    request_id: str
    admission_decision: PrecisionAdmissionDecision
    landing_decision: PrecisionLandingDecision | None
    planned_release: int
    admission_infeasible: bool
    mixed_landing_release_gap: int
    reason: str


class PrecisionAdmissionController:
    """Plan whether demotion can satisfy an admission requirement.

    If the frontier can satisfy admission, release exactly the requested
    deficit. If it cannot, still release the feasible frontier so a large
    waiting request can make monotonic progress across scheduling rounds.
    """

    def plan_admission(
        self, state: PrecisionAdmissionState
    ) -> PrecisionAdmissionDecision:
        required_blocks = max(0, state.needed_blocks) + max(0, state.reserve_blocks)
        requested_release = max(0, state.requested_release)
        free_bf16_blocks = max(0, state.free_bf16_blocks)
        feasible_release = min(max(0, state.feasible_release), requested_release)

        if requested_release == 0 or free_bf16_blocks >= required_blocks:
            planned_release = 0
            admission_success = True
            admission_infeasible = False
        else:
            frontier_can_admit = free_bf16_blocks + feasible_release >= required_blocks
            if feasible_release > 0:
                planned_release = feasible_release
                free_after_planned = free_bf16_blocks + planned_release
                admission_success = free_after_planned >= required_blocks
            else:
                planned_release = 0
                admission_success = False
            admission_infeasible = not frontier_can_admit

        free_after_planned = free_bf16_blocks + planned_release
        return PrecisionAdmissionDecision(
            request_id=state.request_id,
            required_blocks=required_blocks,
            requested_release=requested_release,
            feasible_release=feasible_release,
            planned_release=planned_release,
            free_after_planned=free_after_planned,
            admission_success_after_planned=admission_success,
            admission_infeasible=admission_infeasible,
        )

    def plan_precision_admission(
        self,
        *,
        admission_decision: PrecisionAdmissionDecision,
        landing_decision: PrecisionLandingDecision | None = None,
    ) -> PrecisionAdmissionPlan:
        planned_release = admission_decision.planned_release
        admission_infeasible = admission_decision.admission_infeasible
        mixed_landing_release_gap = 0
        reason = "admission_release"

        if (
            landing_decision is not None
            and landing_decision.admission_feasible_with_landing
            and (
                landing_decision.mixed_landing_required
                or landing_decision.reserve_relaxed
            )
        ):
            landing_gap_basis = (
                landing_decision.needed_deficit_after_running
                if landing_decision.reserve_relaxed
                else landing_decision.bf16_deficit_blocks
            )
            mixed_landing_release_gap = max(
                0,
                landing_gap_basis - landing_decision.planned_int4_landing_blocks,
            )
            if mixed_landing_release_gap == 0:
                planned_release = 0
                admission_infeasible = False
                reason = (
                    "mixed_landing_relaxes_reserve"
                    if landing_decision.reserve_relaxed
                    else "mixed_landing_closes_deficit"
                )
            elif admission_decision.feasible_release >= mixed_landing_release_gap and (
                planned_release == 0 or mixed_landing_release_gap < planned_release
            ):
                planned_release = mixed_landing_release_gap
                admission_infeasible = False
                reason = "mixed_landing_reduces_release"
        elif admission_infeasible:
            reason = "admission_frontier_infeasible"

        return PrecisionAdmissionPlan(
            request_id=admission_decision.request_id,
            admission_decision=admission_decision,
            landing_decision=landing_decision,
            planned_release=planned_release,
            admission_infeasible=admission_infeasible,
            mixed_landing_release_gap=mixed_landing_release_gap,
            reason=reason,
        )
