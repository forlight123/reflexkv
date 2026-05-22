# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Admission-frontier state for precision-elastic KV scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PageCompressibilityLevel(str, Enum):
    """Coarse page classes used by the admission controller.

    These levels intentionally describe scheduling eligibility rather than the
    physical storage format.  The current ReFlexKV planner reports aggregate
    funnel counters, so a frontier summary is an explainable approximation of
    the candidate frontier.
    """

    PINNED = "pinned"
    PROTECTED = "protected"
    CANDIDATE = "candidate"
    EAGER_COMPRESSIBLE = "eager_compressible"
    LOW_PRECISION = "low_precision"


class RejectionReason(str, Enum):
    SHARED_OR_OPEN = "shared_or_open"
    RECENT_OR_INITIAL = "recent_or_initial"
    HIGH_RISK = "high_risk"
    REQUEST_FRACTION_CAP = "request_fraction_cap"
    QUALITY_DEBT_CAP = "quality_debt_cap"
    REQUEST_RELEASE_BUDGET = "request_release_budget"
    SHORT_DECODE_PROTECTION = "short_decode_protection"
    REASONING_PROMPT_PROTECTION = "reasoning_prompt_protection"
    REQUEST_BUDGET = "request_budget"
    SPARSE_QUOTA = "sparse_quota"
    FRONTIER_OPTIMIZER = "frontier_optimizer"
    INT4_POOL_FULL = "int4_pool_full"


def _nonnegative_delta(before: int, after: int) -> int:
    return max(0, int(before) - int(after))


def _all_level_counts() -> dict[PageCompressibilityLevel, int]:
    return {level: 0 for level in PageCompressibilityLevel}


def _all_rejection_counts() -> dict[RejectionReason, int]:
    return {reason: 0 for reason in RejectionReason}


@dataclass(frozen=True)
class FeasibleFrontierSummary:
    """Structured view of why a demotion frontier can or cannot admit work."""

    scheduler_step: int
    reason: str
    target_release: int
    feasible_release: int
    candidate_breakdown: Any | None = None
    eligible_by_level: dict[PageCompressibilityLevel, int] = field(
        default_factory=_all_level_counts
    )
    blocked_by_reason: dict[RejectionReason, int] = field(
        default_factory=_all_rejection_counts
    )

    @classmethod
    def from_candidate_breakdown(
        cls,
        *,
        scheduler_step: int,
        reason: str,
        target_release: int,
        feasible_release: int,
        candidate_breakdown: Any | None,
    ) -> "FeasibleFrontierSummary":
        if candidate_breakdown is None:
            return cls(
                scheduler_step=scheduler_step,
                reason=reason,
                target_release=target_release,
                feasible_release=feasible_release,
                candidate_breakdown=None,
            )

        raw_bf16_pages = int(getattr(candidate_breakdown, "raw_bf16_pages", 0))
        eligible_full_unshared_pages = int(
            getattr(
                candidate_breakdown,
                "eligible_full_unshared_pages",
                0,
            )
        )
        after_initial_recent_protection = int(
            getattr(
                candidate_breakdown,
                "after_initial_recent_protection",
                0,
            )
        )
        after_low_risk_filter = int(
            getattr(candidate_breakdown, "after_low_risk_filter", 0)
        )
        raw_after_request_fraction_cap = getattr(
            candidate_breakdown,
            "after_request_fraction_cap",
            None,
        )
        has_request_fraction_cap = raw_after_request_fraction_cap is not None
        after_request_fraction_cap = (
            after_low_risk_filter
            if raw_after_request_fraction_cap is None
            else int(raw_after_request_fraction_cap)
        )
        raw_after_quality_debt_cap = getattr(
            candidate_breakdown,
            "after_quality_debt_cap",
            None,
        )
        has_quality_debt_cap = raw_after_quality_debt_cap is not None
        after_quality_debt_cap = (
            after_request_fraction_cap
            if raw_after_quality_debt_cap is None
            else int(raw_after_quality_debt_cap)
        )
        after_request_budget_cap = int(
            getattr(candidate_breakdown, "after_request_budget_cap", 0)
        )
        after_sparse_window_quota = int(
            getattr(candidate_breakdown, "after_sparse_window_quota", 0)
        )
        after_int4_pool_limit = int(
            getattr(candidate_breakdown, "after_int4_pool_limit", 0)
        )
        after_frontier_optimizer = int(
            getattr(candidate_breakdown, "after_frontier_optimizer", 0)
        )
        int4_free_blocks = int(
            getattr(candidate_breakdown, "int4_free_blocks", -1)
        )
        frontier_optimizer_budget = int(
            getattr(candidate_breakdown, "frontier_optimizer_budget", -1)
        )
        if after_frontier_optimizer <= 0 and after_int4_pool_limit > 0:
            after_frontier_optimizer = after_sparse_window_quota
        int4_pool_exhausted_before_optimizer = (
            after_sparse_window_quota > 0
            and after_frontier_optimizer == 0
            and after_int4_pool_limit == 0
            and int4_free_blocks == 0
            and frontier_optimizer_budget == 0
        )

        eligible_by_level = _all_level_counts()
        eligible_by_level[PageCompressibilityLevel.PINNED] = _nonnegative_delta(
            raw_bf16_pages,
            after_initial_recent_protection,
        )
        eligible_by_level[PageCompressibilityLevel.PROTECTED] = (
            _nonnegative_delta(
                after_initial_recent_protection,
                after_low_risk_filter,
            )
        )
        eligible_by_level[PageCompressibilityLevel.CANDIDATE] = max(
            0,
            after_low_risk_filter,
        )
        eligible_by_level[PageCompressibilityLevel.EAGER_COMPRESSIBLE] = max(
            0,
            after_int4_pool_limit,
        )

        blocked_by_reason = _all_rejection_counts()
        blocked_by_reason[RejectionReason.SHARED_OR_OPEN] = _nonnegative_delta(
            raw_bf16_pages,
            eligible_full_unshared_pages,
        )
        blocked_by_reason[RejectionReason.RECENT_OR_INITIAL] = (
            _nonnegative_delta(
                eligible_full_unshared_pages,
                after_initial_recent_protection,
            )
        )
        blocked_by_reason[RejectionReason.HIGH_RISK] = _nonnegative_delta(
            after_initial_recent_protection,
            after_low_risk_filter,
        )
        blocked_by_reason[RejectionReason.REQUEST_FRACTION_CAP] = (
            _nonnegative_delta(
                after_low_risk_filter,
                after_request_fraction_cap,
            )
            if has_request_fraction_cap
            else 0
        )
        blocked_by_reason[RejectionReason.QUALITY_DEBT_CAP] = (
            _nonnegative_delta(
                after_request_fraction_cap,
                after_quality_debt_cap,
            )
            if has_quality_debt_cap
            else 0
        )
        blocked_by_reason[RejectionReason.REQUEST_RELEASE_BUDGET] = (
            _nonnegative_delta(
                after_quality_debt_cap,
                after_request_budget_cap,
            )
            if (has_request_fraction_cap or has_quality_debt_cap)
            else 0
        )
        blocked_by_reason[RejectionReason.REQUEST_BUDGET] = _nonnegative_delta(
            after_low_risk_filter,
            after_request_budget_cap,
        )
        blocked_by_reason[RejectionReason.SPARSE_QUOTA] = _nonnegative_delta(
            after_request_budget_cap,
            after_sparse_window_quota,
        )
        if int4_pool_exhausted_before_optimizer:
            blocked_by_reason[RejectionReason.FRONTIER_OPTIMIZER] = 0
            blocked_by_reason[RejectionReason.INT4_POOL_FULL] = max(
                0,
                after_sparse_window_quota,
            )
        else:
            blocked_by_reason[RejectionReason.FRONTIER_OPTIMIZER] = (
                _nonnegative_delta(
                    after_sparse_window_quota,
                    after_frontier_optimizer,
                )
            )
            blocked_by_reason[RejectionReason.INT4_POOL_FULL] = (
                _nonnegative_delta(
                    after_frontier_optimizer,
                    after_int4_pool_limit,
                )
            )

        return cls(
            scheduler_step=scheduler_step,
            reason=reason,
            target_release=target_release,
            feasible_release=feasible_release,
            candidate_breakdown=candidate_breakdown,
            eligible_by_level=eligible_by_level,
            blocked_by_reason=blocked_by_reason,
        )

    @property
    def candidate_release_capacity(self) -> int:
        if self.candidate_breakdown is None:
            return max(0, self.feasible_release)
        return max(
            0,
            int(
                getattr(
                    self.candidate_breakdown,
                    "after_int4_pool_limit",
                    self.feasible_release,
                )
            ),
        )

    def cached_frontier_age(self, *, current_step: int) -> int:
        return max(0, int(current_step) - int(self.scheduler_step))

    def format_levels(self) -> str:
        return ",".join(
            f"{level.value}:{self.eligible_by_level.get(level, 0)}"
            for level in PageCompressibilityLevel
        )

    def format_rejection_reasons(self) -> str:
        return ",".join(
            f"{reason.value}:{self.blocked_by_reason.get(reason, 0)}"
            for reason in RejectionReason
        )


class FeasibleFrontierCache:
    """Small exact-match cache for recent demotion dry-run frontiers."""

    def __init__(self, *, max_age_steps: int = 2) -> None:
        self.max_age_steps = max(0, int(max_age_steps))
        self._summaries: dict[tuple[str, int], FeasibleFrontierSummary] = {}
        self._latest_summary: FeasibleFrontierSummary | None = None

    def update(self, summary: FeasibleFrontierSummary) -> None:
        key = (summary.reason, summary.target_release)
        self._summaries[key] = summary
        self._latest_summary = summary

    def get(
        self,
        *,
        reason: str,
        target_release: int,
        current_step: int,
    ) -> FeasibleFrontierSummary | None:
        summary = self._summaries.get((reason, target_release))
        if summary is None:
            return None
        if (
            summary.cached_frontier_age(current_step=current_step)
            > self.max_age_steps
        ):
            return None
        return summary

    def latest(self) -> FeasibleFrontierSummary | None:
        return self._latest_summary

    def invalidate(self) -> None:
        self._summaries.clear()
        self._latest_summary = None


@dataclass(frozen=True)
class AdmissionTicket:
    """Defers repeated waiting admission work until a useful retry point."""

    request_id: str
    required_blocks: int
    blocked_reason: str
    created_step: int
    last_retry_step: int
    next_retry_step: int
    retry_count: int = 0
    retry_on_events: frozenset[str] = frozenset()
    cached_frontier_summary: FeasibleFrontierSummary | None = None

    def should_retry(
        self,
        *,
        current_step: int,
        events: frozenset[str] | set[str] | None = None,
    ) -> bool:
        if current_step >= self.next_retry_step:
            return True
        if not events:
            return False
        return bool(self.retry_on_events.intersection(events))
