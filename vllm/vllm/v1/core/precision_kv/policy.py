# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pressure-aware policy primitives for precision-elastic KV cache."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CandidateFunnelSnapshot:
    """Compact view of the previous demotion candidate funnel."""

    after_initial_recent_protection: int = 0
    after_low_risk_filter: int = 0
    after_request_budget_cap: int = 0
    after_sparse_window_quota: int = 0
    after_frontier_optimizer: int = 0
    after_int4_pool_limit: int = 0

    @classmethod
    def from_object(cls, value: object | None) -> CandidateFunnelSnapshot:
        if value is None:
            return cls()
        after_sparse_window_quota = max(
            0,
            int(getattr(value, "after_sparse_window_quota", 0) or 0),
        )
        after_int4_pool_limit = max(
            0,
            int(getattr(value, "after_int4_pool_limit", 0) or 0),
        )
        after_frontier_optimizer = max(
            0,
            int(getattr(value, "after_frontier_optimizer", 0) or 0),
        )
        if after_frontier_optimizer <= 0 and after_int4_pool_limit > 0:
            after_frontier_optimizer = after_sparse_window_quota
        return cls(
            after_initial_recent_protection=max(
                0,
                int(getattr(value, "after_initial_recent_protection", 0) or 0),
            ),
            after_low_risk_filter=max(
                0,
                int(getattr(value, "after_low_risk_filter", 0) or 0),
            ),
            after_request_budget_cap=max(
                0,
                int(getattr(value, "after_request_budget_cap", 0) or 0),
            ),
            after_sparse_window_quota=after_sparse_window_quota,
            after_frontier_optimizer=after_frontier_optimizer,
            after_int4_pool_limit=after_int4_pool_limit,
        )


@dataclass(frozen=True)
class PrecisionPressureState:
    """Global pressure inputs for one policy decision."""

    reason: str
    target_bf16_blocks: int
    free_bf16_blocks: int
    total_bf16_blocks: int
    waiting_requests: int = 0
    skipped_waiting_requests: int = 0
    candidate_funnel: CandidateFunnelSnapshot = field(
        default_factory=CandidateFunnelSnapshot
    )
    base_low_risk_score_fraction: float = 0.25


@dataclass(frozen=True)
class PrecisionPressureDecision:
    """Global pressure outputs consumed by scheduler-side planning."""

    pressure_active: bool
    target_release_blocks: int
    request_release_budget_multiplier: float = 1.0
    max_demote_per_window_multiplier: float = 1.0
    low_risk_score_fraction: float = 0.25
    policy_reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class RequestPolicyState:
    """Request-local inputs for a pressure-aware budget decision."""

    request_id: str
    page_count: int
    prompt_pages: int
    protected_prompt_pages: int
    generated_decode_tokens: int
    remaining_decode_tokens: int
    priority: int
    base_max_int4_fraction: float
    is_short_decode: bool


@dataclass(frozen=True)
class RequestPolicyDecision:
    request_id: str
    max_int4_pages: int
    priority: float
    max_int4_fraction: float
    protected_prompt_pages: int


class PrecisionKVPolicy:
    """Small closed-loop policy for ReFlexKV pressure response.

    This class deliberately stays independent from the vLLM scheduler and cache
    manager. It only turns observable pressure and the previous candidate funnel
    into budget/eligibility multipliers.
    """

    _ADMISSION_REASONS = {
        "admission_waiting",
        "allocation_failure",
        "full_sequence_reserve",
    }

    @staticmethod
    def _severity_multiplier(
        *,
        before: int,
        after: int,
        trigger_fraction: float,
        base_multiplier: float,
        max_multiplier: float,
    ) -> float:
        if before <= 0:
            return 1.0
        trigger_after = max(1, math.ceil(trigger_fraction * before))
        if after >= trigger_after:
            return 1.0
        if after <= 0:
            return max_multiplier
        severity = trigger_after / max(1, after)
        return min(max_multiplier, max(base_multiplier, severity))

    def plan_pressure(
        self,
        state: PrecisionPressureState,
    ) -> PrecisionPressureDecision:
        total_blocks = max(1, int(state.total_bf16_blocks))
        free_ratio = max(0, int(state.free_bf16_blocks)) / total_blocks
        target_blocks = max(0, int(state.target_bf16_blocks))
        pressure_active = (
            state.reason in self._ADMISSION_REASONS
            or state.waiting_requests > 0
            or state.skipped_waiting_requests > 0
            or (target_blocks > 0 and free_ratio <= 0.10)
        )

        low_risk_fraction = min(
            1.0,
            max(0.0, float(state.base_low_risk_score_fraction)),
        )
        if not pressure_active:
            return PrecisionPressureDecision(
                pressure_active=False,
                target_release_blocks=target_blocks,
                low_risk_score_fraction=low_risk_fraction,
            )

        funnel = state.candidate_funnel
        release_multiplier = 1.0
        window_multiplier = 1.0
        reasons: list[str] = []

        if (
            funnel.after_low_risk_filter > 0
            and funnel.after_request_budget_cap
            < math.ceil(0.25 * funnel.after_low_risk_filter)
        ):
            release_multiplier = max(
                release_multiplier,
                self._severity_multiplier(
                    before=funnel.after_low_risk_filter,
                    after=funnel.after_request_budget_cap,
                    trigger_fraction=0.25,
                    base_multiplier=2.0,
                    max_multiplier=8.0,
                ),
            )
            reasons.append("request_budget_cap")

        if (
            funnel.after_request_budget_cap > 0
            and funnel.after_sparse_window_quota
            < math.ceil(0.75 * funnel.after_request_budget_cap)
        ):
            window_multiplier = max(
                window_multiplier,
                self._severity_multiplier(
                    before=funnel.after_request_budget_cap,
                    after=funnel.after_sparse_window_quota,
                    trigger_fraction=0.75,
                    base_multiplier=2.0,
                    max_multiplier=8.0,
                ),
            )
            reasons.append("sparse_window_quota")

        if (
            funnel.after_sparse_window_quota > 0
            and funnel.after_frontier_optimizer
            < math.ceil(0.75 * funnel.after_sparse_window_quota)
        ):
            release_multiplier = max(
                release_multiplier,
                self._severity_multiplier(
                    before=funnel.after_sparse_window_quota,
                    after=funnel.after_frontier_optimizer,
                    trigger_fraction=0.75,
                    base_multiplier=1.5,
                    max_multiplier=4.0,
                ),
            )
            reasons.append("frontier_optimizer")

        if (
            funnel.after_initial_recent_protection > 0
            and funnel.after_low_risk_filter
            < math.ceil(0.25 * funnel.after_initial_recent_protection)
        ):
            low_risk_fraction = min(1.0, max(low_risk_fraction * 2.0, 0.50))
            reasons.append("low_risk_filter")

        target_release_blocks = math.ceil(target_blocks * release_multiplier)
        return PrecisionPressureDecision(
            pressure_active=True,
            target_release_blocks=target_release_blocks,
            request_release_budget_multiplier=release_multiplier,
            max_demote_per_window_multiplier=window_multiplier,
            low_risk_score_fraction=low_risk_fraction,
            policy_reasons=tuple(reasons),
        )

    def plan_request_budget(
        self,
        pressure: PrecisionPressureState,
        request: RequestPolicyState,
    ) -> RequestPolicyDecision:
        pressure_decision = self.plan_pressure(pressure)
        page_count = max(0, int(request.page_count))
        protected_prompt_pages = max(0, int(request.protected_prompt_pages))
        demotable_pages = max(0, page_count - protected_prompt_pages)

        base_fraction = min(
            1.0,
            max(0.0, float(request.base_max_int4_fraction)),
        )
        if request.is_short_decode:
            base_fraction *= 0.5
        if pressure_decision.pressure_active and request.prompt_pages >= 64:
            base_fraction = min(1.0, base_fraction * 1.25)

        max_int4_pages = min(demotable_pages, int(page_count * base_fraction))
        remaining_score = min(
            1.0,
            max(0, int(request.remaining_decode_tokens)) / 512,
        )
        generated_score = min(
            1.0,
            max(0, int(request.generated_decode_tokens)) / 512,
        )
        slo_pressure = max(0.5, min(1.5, 1.0 + 0.25 * int(request.priority)))
        priority = (
            max_int4_pages
            * (0.5 + remaining_score + 0.5 * generated_score)
            * slo_pressure
        )

        return RequestPolicyDecision(
            request_id=request.request_id,
            max_int4_pages=max_int4_pages,
            priority=priority,
            max_int4_fraction=base_fraction,
            protected_prompt_pages=protected_prompt_pages,
        )
