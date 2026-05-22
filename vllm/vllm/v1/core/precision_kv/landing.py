# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission planning for precision-elastic KV landing.

The landing planner is deliberately data-plane agnostic. It answers whether a
waiting request could fit if part of its incoming KV pages landed directly into
a lower precision tier, after accounting for BF16 free space and the running
request demotion frontier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PrecisionLandingState:
    """KV-centered state for mixed-precision admission of one request."""

    request_id: str
    needed_blocks: int
    reserve_blocks: int
    free_bf16_blocks: int
    running_feasible_release: int
    eligible_int4_landing_blocks: int
    eligible_int4_landing_pages: tuple[int, ...] = ()
    allow_reserve_relaxation: bool = False


@dataclass(frozen=True)
class PrecisionLandingDecision:
    """Decision for whether mixed landing can satisfy admission."""

    request_id: str
    required_blocks: int
    bf16_deficit_blocks: int
    residual_deficit_after_running: int
    eligible_int4_landing_blocks: int
    planned_int4_landing_blocks: int
    admission_feasible_with_landing: bool
    mixed_landing_required: bool
    reason: str
    planned_int4_landing_pages: tuple[int, ...] = ()
    needed_deficit_after_running: int = 0
    reserve_relaxed: bool = False


class PrecisionLandingPlanner:
    """Plan whether request-local INT4 landing can close an admission gap."""

    def plan_landing(
        self, state: PrecisionLandingState
    ) -> PrecisionLandingDecision:
        needed_blocks = max(0, state.needed_blocks)
        reserve_blocks = max(0, state.reserve_blocks)
        required_blocks = needed_blocks + reserve_blocks
        free_bf16_blocks = max(0, state.free_bf16_blocks)
        running_feasible_release = max(0, state.running_feasible_release)
        eligible_int4_landing_blocks = max(
            0, state.eligible_int4_landing_blocks
        )
        if state.eligible_int4_landing_pages:
            eligible_int4_landing_blocks = min(
                eligible_int4_landing_blocks,
                len(state.eligible_int4_landing_pages),
            )

        bf16_deficit_blocks = max(0, required_blocks - free_bf16_blocks)
        residual_deficit_after_running = max(
            0,
            required_blocks - free_bf16_blocks - running_feasible_release,
        )
        needed_deficit_after_running = max(
            0,
            needed_blocks - free_bf16_blocks - running_feasible_release,
        )
        reserve_relaxed = False

        if bf16_deficit_blocks == 0:
            planned_int4_landing_blocks = 0
            planned_int4_landing_pages = ()
            feasible = True
            mixed_required = False
            reason = "bf16_fit"
        elif residual_deficit_after_running == 0:
            planned_int4_landing_blocks = 0
            planned_int4_landing_pages = ()
            feasible = True
            mixed_required = False
            reason = "running_frontier_can_admit"
        elif eligible_int4_landing_blocks >= residual_deficit_after_running:
            planned_int4_landing_blocks = residual_deficit_after_running
            planned_int4_landing_pages = state.eligible_int4_landing_pages[
                :planned_int4_landing_blocks
            ]
            feasible = True
            mixed_required = True
            reason = "mixed_landing_feasible"
        elif (
            state.allow_reserve_relaxation
            and needed_deficit_after_running == 0
        ):
            planned_int4_landing_blocks = 0
            planned_int4_landing_pages = ()
            feasible = True
            mixed_required = False
            reserve_relaxed = True
            reason = "reserve_relaxed_needed_fit"
        elif (
            state.allow_reserve_relaxation
            and eligible_int4_landing_blocks >= needed_deficit_after_running
        ):
            planned_int4_landing_blocks = needed_deficit_after_running
            planned_int4_landing_pages = state.eligible_int4_landing_pages[
                :planned_int4_landing_blocks
            ]
            feasible = True
            mixed_required = planned_int4_landing_blocks > 0
            reserve_relaxed = True
            reason = "mixed_landing_relaxed_reserve_feasible"
        else:
            planned_int4_landing_blocks = 0
            planned_int4_landing_pages = ()
            feasible = False
            mixed_required = residual_deficit_after_running > 0
            reason = "int4_landing_frontier_insufficient"

        return PrecisionLandingDecision(
            request_id=state.request_id,
            required_blocks=required_blocks,
            bf16_deficit_blocks=bf16_deficit_blocks,
            residual_deficit_after_running=residual_deficit_after_running,
            needed_deficit_after_running=needed_deficit_after_running,
            eligible_int4_landing_blocks=eligible_int4_landing_blocks,
            planned_int4_landing_blocks=planned_int4_landing_blocks,
            admission_feasible_with_landing=feasible,
            mixed_landing_required=mixed_required,
            reason=reason,
            planned_int4_landing_pages=planned_int4_landing_pages,
            reserve_relaxed=reserve_relaxed,
        )
