# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Online demotion planner for precision-aware KV memory."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from vllm.v1.core.precision_kv.run_optimizer import (
    DualPriceState,
    DualRunOptimizer,
)
from vllm.v1.core.precision_kv.types import (
    Int4BlockPool,
    PrecisionState,
    ReflexDemotion,
    ReflexPageMeta,
    encode_int4_block_id,
)


def _sum_optional_count(left: int | None, right: int | None) -> int | None:
    if left is None and right is None:
        return None
    return int(left or 0) + int(right or 0)


@dataclass(frozen=True)
class ReflexCandidateBreakdown:
    raw_bf16_pages: int = 0
    open_bf16_pages: int = 0
    remote_inflight_bf16_pages: int = 0
    open_tail_bf16_pages: int = 0
    request_protected_bf16_pages: int = 0
    shared_bf16_pages: int = 0
    prompt_protected_bf16_pages: int = 0
    copy_on_demote_pages: int = 0
    eligible_full_unshared_pages: int = 0
    after_initial_recent_protection: int = 0
    after_low_risk_filter: int = 0
    after_request_fraction_cap: int | None = None
    after_quality_debt_cap: int | None = None
    after_request_budget_cap: int = 0
    after_sparse_window_quota: int = 0
    after_frontier_optimizer: int = 0
    after_int4_pool_limit: int = 0
    selected_actual: int = 0
    int4_free_blocks: int = 0
    frontier_optimizer_budget: int = 0

    def __add__(self, other: ReflexCandidateBreakdown):
        return ReflexCandidateBreakdown(
            raw_bf16_pages=(self.raw_bf16_pages + other.raw_bf16_pages),
            open_bf16_pages=(self.open_bf16_pages + other.open_bf16_pages),
            remote_inflight_bf16_pages=(
                self.remote_inflight_bf16_pages + other.remote_inflight_bf16_pages
            ),
            open_tail_bf16_pages=(
                self.open_tail_bf16_pages + other.open_tail_bf16_pages
            ),
            request_protected_bf16_pages=(
                self.request_protected_bf16_pages + other.request_protected_bf16_pages
            ),
            shared_bf16_pages=(self.shared_bf16_pages + other.shared_bf16_pages),
            prompt_protected_bf16_pages=(
                self.prompt_protected_bf16_pages + other.prompt_protected_bf16_pages
            ),
            copy_on_demote_pages=(
                self.copy_on_demote_pages + other.copy_on_demote_pages
            ),
            eligible_full_unshared_pages=(
                self.eligible_full_unshared_pages + other.eligible_full_unshared_pages
            ),
            after_initial_recent_protection=(
                self.after_initial_recent_protection
                + other.after_initial_recent_protection
            ),
            after_low_risk_filter=(
                self.after_low_risk_filter + other.after_low_risk_filter
            ),
            after_request_fraction_cap=_sum_optional_count(
                self.after_request_fraction_cap,
                other.after_request_fraction_cap,
            ),
            after_quality_debt_cap=_sum_optional_count(
                self.after_quality_debt_cap,
                other.after_quality_debt_cap,
            ),
            after_request_budget_cap=(
                self.after_request_budget_cap + other.after_request_budget_cap
            ),
            after_sparse_window_quota=(
                self.after_sparse_window_quota + other.after_sparse_window_quota
            ),
            after_frontier_optimizer=(
                self.after_frontier_optimizer + other.after_frontier_optimizer
            ),
            after_int4_pool_limit=(
                self.after_int4_pool_limit + other.after_int4_pool_limit
            ),
            selected_actual=self.selected_actual + other.selected_actual,
            int4_free_blocks=self.int4_free_blocks + other.int4_free_blocks,
            frontier_optimizer_budget=(
                self.frontier_optimizer_budget + other.frontier_optimizer_budget
            ),
        )


@dataclass(frozen=True)
class ReflexDemotionPlan:
    items: list[ReflexDemotion]
    candidate_bf16_blocks: int = 0
    planned_bf16_blocks: int | None = None
    candidate_breakdown: ReflexCandidateBreakdown = field(
        default_factory=ReflexCandidateBreakdown
    )

    @property
    def released_bf16_blocks(self) -> int:
        if self.planned_bf16_blocks is not None:
            return self.planned_bf16_blocks
        return len(self.items)


@dataclass(frozen=True)
class RequestPrecisionBudget:
    """Per-request cap used by the runtime demotion controller."""

    max_int4_pages: int
    priority: float = 0.0
    max_int4_fraction: float | None = None
    release_budget_blocks: int | None = None
    max_demote_per_window: int | None = None
    max_prompt_int4_pages: int | None = None
    max_decode_int4_pages: int | None = None
    quality_debt_budget_pages: int | None = None


@dataclass(frozen=True)
class RequestBudgetCandidate:
    """Candidate request for global BF16-release budget allocation."""

    request_id: str
    capacity_blocks: int
    utility: float


def allocate_request_release_budgets(
    candidates: Sequence[RequestBudgetCandidate],
    *,
    target_bf16_blocks: int,
) -> dict[str, int]:
    """Allocate this-step BF16-release budgets across requests.

    Budgets are expressed in BF16-equivalent blocks. The allocator is
    deliberately small and deterministic: first assign proportional floor
    shares, then give leftover blocks to requests with the largest fractional
    remainder while respecting each request's capacity.
    """
    if target_bf16_blocks <= 0:
        return {}

    active = [
        candidate
        for candidate in candidates
        if candidate.capacity_blocks > 0 and candidate.utility > 0
    ]
    if not active:
        return {}

    target = min(
        target_bf16_blocks,
        sum(candidate.capacity_blocks for candidate in active),
    )
    total_utility = sum(candidate.utility for candidate in active)
    allocations: dict[str, int] = {}
    remainders: list[tuple[float, float, int, str]] = []

    for order, candidate in enumerate(active):
        raw_share = target * candidate.utility / total_utility
        base_share = min(candidate.capacity_blocks, int(raw_share))
        if base_share > 0:
            allocations[candidate.request_id] = base_share
        remainders.append(
            (
                raw_share - int(raw_share),
                candidate.utility,
                -order,
                candidate.request_id,
            )
        )

    remaining = target - sum(allocations.values())
    if remaining <= 0:
        return allocations

    capacity_by_request = {
        candidate.request_id: candidate.capacity_blocks for candidate in active
    }
    sorted_remainders = sorted(
        remainders,
        reverse=True,
    )
    while remaining > 0:
        made_progress = False
        for _fraction, _utility, _order, request_id in sorted_remainders:
            if remaining <= 0:
                break
            current = allocations.get(request_id, 0)
            if current >= capacity_by_request[request_id]:
                continue
            allocations[request_id] = current + 1
            remaining -= 1
            made_progress = True
        if not made_progress:
            break

    return allocations


class DistanceDemotionPlanner:
    """Select sealed BF16 pages using distance-based protection bands."""

    def __init__(
        self,
        keep_recent_pages: int = 4,
        keep_initial_pages: int = 0,
        max_int4_fraction_per_request: float = 1.0,
        low_risk_only: bool = False,
        sparse_window_pages: int = 0,
        max_demote_per_window: int = 0,
        selection_policy: str = "relevance_sparse",
        dual_price_state: DualPriceState | None = None,
        emergency_release: bool = False,
    ) -> None:
        if keep_recent_pages < 0:
            raise ValueError(
                f"keep_recent_pages must be non-negative, got {keep_recent_pages}."
            )
        if keep_initial_pages < 0:
            raise ValueError(
                f"keep_initial_pages must be non-negative, got {keep_initial_pages}."
            )
        if not 0.0 <= max_int4_fraction_per_request <= 1.0:
            raise ValueError(
                "max_int4_fraction_per_request must be in [0, 1], got "
                f"{max_int4_fraction_per_request}."
            )
        if sparse_window_pages < 0:
            raise ValueError(
                f"sparse_window_pages must be non-negative, got {sparse_window_pages}."
            )
        if max_demote_per_window < 0:
            raise ValueError(
                "max_demote_per_window must be non-negative, got "
                f"{max_demote_per_window}."
            )
        valid_policies = {
            "oldest",
            "distance",
            "random",
            "relevance",
            "relevance_sparse",
            "frontier_dual",
        }
        if selection_policy not in valid_policies:
            raise ValueError(
                "selection_policy must be one of "
                f"{sorted(valid_policies)}, got {selection_policy!r}."
            )
        self.keep_recent_pages = keep_recent_pages
        self.keep_initial_pages = keep_initial_pages
        self.max_int4_fraction_per_request = max_int4_fraction_per_request
        self.low_risk_only = low_risk_only
        self.sparse_window_pages = sparse_window_pages
        self.max_demote_per_window = max_demote_per_window
        self.selection_policy = selection_policy
        self.dual_price_state = dual_price_state
        self.emergency_release = emergency_release

    def plan(
        self,
        request_pages: Mapping[str, Sequence[ReflexPageMeta]],
        *,
        target_bf16_blocks: int,
        int4_pool: Int4BlockPool,
        request_precision_budgets: (Mapping[str, RequestPrecisionBudget] | None) = None,
        dry_run: bool = False,
    ) -> ReflexDemotionPlan:
        if target_bf16_blocks <= 0:
            return ReflexDemotionPlan([])

        selected: list[ReflexDemotion] = []
        selectable_pages: list[ReflexPageMeta] = []
        frontier_candidate_pages: list[ReflexPageMeta] = []
        raw_bf16_pages = 0
        open_bf16_pages = 0
        remote_inflight_bf16_pages = 0
        open_tail_bf16_pages = 0
        request_protected_bf16_pages = 0
        shared_bf16_pages = 0
        prompt_protected_bf16_pages = 0
        copy_on_demote_pages = 0
        eligible_full_unshared_pages = 0
        after_initial_recent_protection = 0
        after_low_risk_filter = 0
        after_request_fraction_cap = 0
        after_quality_debt_cap = 0
        after_request_budget_cap = 0
        after_sparse_window_quota = 0
        after_frontier_optimizer = 0
        int4_capacity_blocks = int4_pool.num_free_blocks
        frontier_optimizer_budget = min(
            target_bf16_blocks,
            int4_capacity_blocks,
        )
        prompt_pages_by_request: dict[str, int] = {}
        request_precision_budgets = request_precision_budgets or {}
        request_items = list(request_pages.items())
        if request_precision_budgets:
            request_order = {
                request_id: order for order, (request_id, _) in enumerate(request_items)
            }
            request_items.sort(
                key=lambda item: (
                    -request_precision_budgets.get(
                        item[0], RequestPrecisionBudget(max_int4_pages=0)
                    ).priority,
                    request_order[item[0]],
                )
            )

        for request_id, pages in request_items:
            num_pages = len(pages)
            prompt_pages_by_request[request_id] = sum(
                1 for page in pages if page.is_prompt_page
            )
            protect_from = max(0, num_pages - self.keep_recent_pages)
            existing_int4_pages = sum(
                1 for page in pages if page.precision == PrecisionState.INT4
            )
            existing_prompt_int4_pages = sum(
                1
                for page in pages
                if (page.precision == PrecisionState.INT4 and page.is_prompt_page)
            )
            existing_decode_int4_pages = max(
                0,
                existing_int4_pages - existing_prompt_int4_pages,
            )
            request_budget = request_precision_budgets.get(request_id)
            if request_budget is None:
                max_int4_pages = int(num_pages * self.max_int4_fraction_per_request)
                release_budget_blocks = None
                max_prompt_int4_pages = None
                max_decode_int4_pages = None
                quality_debt_budget_pages = None
            else:
                max_int4_pages = max(0, request_budget.max_int4_pages)
                release_budget_blocks = request_budget.release_budget_blocks
                max_prompt_int4_pages = request_budget.max_prompt_int4_pages
                max_decode_int4_pages = request_budget.max_decode_int4_pages
                quality_debt_budget_pages = request_budget.quality_debt_budget_pages

            raw_pages = [
                page
                for page in pages
                if (
                    page.precision == PrecisionState.BF16
                    and page.bf16_block_id is not None
                    and page.int4_block_id is None
                )
            ]
            eligible_pages = [
                page
                for page in raw_pages
                if (
                    page.is_full
                    and (not page.is_shared or page.copy_on_demote)
                    and not page.is_request_protected
                    and not page.is_page_protected
                    and not page.is_prompt_protected
                )
            ]
            protected_pages: list[ReflexPageMeta] = []
            for page in pages:
                if page.request_id != request_id:
                    raise ValueError(
                        "Page metadata request_id does not match mapping key: "
                        f"{page.request_id!r} != {request_id!r}."
                    )
            for page in eligible_pages:
                if page.page_idx < self.keep_initial_pages:
                    continue
                if page.page_idx >= protect_from:
                    continue
                protected_pages.append(page)

            if self.low_risk_only:
                risk_filtered_pages = [
                    page for page in protected_pages if page.compressible is True
                ]
            else:
                risk_filtered_pages = protected_pages

            risk_filtered_pages.sort(key=self._candidate_sort_key)
            max_new_int4_pages = max(0, max_int4_pages - existing_int4_pages)
            if self.emergency_release:
                max_new_int4_pages = len(risk_filtered_pages)
                max_prompt_int4_pages = None
                max_decode_int4_pages = None
            fraction_capped_pages = self._cap_pages_by_request_precision_budget(
                risk_filtered_pages,
                max_new_int4_pages=max_new_int4_pages,
                max_prompt_int4_pages=max_prompt_int4_pages,
                existing_prompt_int4_pages=existing_prompt_int4_pages,
                max_decode_int4_pages=max_decode_int4_pages,
                existing_decode_int4_pages=existing_decode_int4_pages,
            )
            if quality_debt_budget_pages is not None and not self.emergency_release:
                quality_debt_remaining = max(
                    0,
                    int(quality_debt_budget_pages) - existing_int4_pages,
                )
                quality_capped_pages = fraction_capped_pages[:quality_debt_remaining]
            else:
                quality_capped_pages = fraction_capped_pages
            if release_budget_blocks is not None and not self.emergency_release:
                capped_pages = quality_capped_pages[: max(0, release_budget_blocks)]
            else:
                capped_pages = quality_capped_pages

            per_window_limit = self.max_demote_per_window
            if request_budget and request_budget.max_demote_per_window is not None:
                per_window_limit = request_budget.max_demote_per_window
            sparse_window_enabled = (
                not self.emergency_release
                and self.selection_policy in {"relevance_sparse", "frontier_dual"}
                and self.sparse_window_pages > 0
                and per_window_limit > 0
            )

            if self.selection_policy == "frontier_dual":
                candidate_pages = capped_pages
                if sparse_window_enabled:
                    candidate_pages = []
                    candidate_by_window: dict[tuple[str, int], int] = {}
                    for page in capped_pages:
                        window_id = page.page_idx // self.sparse_window_pages
                        window_key = (request_id, window_id)
                        if candidate_by_window.get(window_key, 0) >= per_window_limit:
                            continue
                        candidate_by_window[window_key] = (
                            candidate_by_window.get(window_key, 0) + 1
                        )
                        candidate_pages.append(page)
                sparse_pages = candidate_pages
                frontier_candidate_pages.extend(candidate_pages)
            else:
                sparse_pages = []
                candidate_by_window: dict[tuple[str, int], int] = {}
                for page in capped_pages:
                    if sparse_window_enabled:
                        window_id = page.page_idx // self.sparse_window_pages
                        window_key = (request_id, window_id)
                        if candidate_by_window.get(window_key, 0) >= per_window_limit:
                            continue
                        candidate_by_window[window_key] = (
                            candidate_by_window.get(window_key, 0) + 1
                        )
                    sparse_pages.append(page)

            raw_bf16_pages += len(raw_pages)
            open_bf16_pages += sum(1 for page in raw_pages if not page.is_full)
            remote_inflight_bf16_pages += sum(
                1
                for page in raw_pages
                if (
                    not page.is_full
                    and page.is_remote_inflight
                    and not page.is_request_protected
                )
            )
            open_tail_bf16_pages += sum(
                1
                for page in raw_pages
                if (
                    not page.is_full
                    and not page.is_remote_inflight
                    and not page.is_request_protected
                )
            )
            request_protected_bf16_pages += sum(
                1 for page in raw_pages if page.is_request_protected
            )
            shared_bf16_pages += sum(
                1
                for page in raw_pages
                if page.is_full and page.is_shared and not page.copy_on_demote
            )
            prompt_protected_bf16_pages += sum(
                1
                for page in raw_pages
                if (
                    page.is_full
                    and page.is_prompt_protected
                    and (not page.is_shared or page.copy_on_demote)
                )
            )
            copy_on_demote_pages += sum(
                1
                for page in raw_pages
                if page.is_full and page.is_shared and page.copy_on_demote
            )
            eligible_full_unshared_pages += len(eligible_pages)
            after_initial_recent_protection += len(protected_pages)
            after_low_risk_filter += len(risk_filtered_pages)
            after_request_fraction_cap += len(fraction_capped_pages)
            after_quality_debt_cap += len(quality_capped_pages)
            after_request_budget_cap += len(capped_pages)
            after_sparse_window_quota += len(sparse_pages)
            if self.selection_policy != "frontier_dual":
                selectable_pages.extend(sparse_pages)

        if self.selection_policy == "frontier_dual":
            selectable_pages = DualRunOptimizer(
                dual_prices=self.dual_price_state,
                sparse_window_pages=self.sparse_window_pages,
                max_pages_per_window=0,
                max_run_pages=None,
            ).select_pages(
                frontier_candidate_pages,
                target_release=frontier_optimizer_budget,
                int4_capacity_blocks=int4_capacity_blocks,
            )
            after_frontier_optimizer = len(selectable_pages)
        else:
            after_frontier_optimizer = after_sparse_window_quota

        candidate_bf16_blocks = min(
            after_frontier_optimizer,
            int4_capacity_blocks,
        )
        planned_bf16_blocks = min(target_bf16_blocks, candidate_bf16_blocks)
        if dry_run:
            candidate_breakdown = ReflexCandidateBreakdown(
                raw_bf16_pages=raw_bf16_pages,
                open_bf16_pages=open_bf16_pages,
                remote_inflight_bf16_pages=remote_inflight_bf16_pages,
                open_tail_bf16_pages=open_tail_bf16_pages,
                request_protected_bf16_pages=request_protected_bf16_pages,
                shared_bf16_pages=shared_bf16_pages,
                prompt_protected_bf16_pages=prompt_protected_bf16_pages,
                copy_on_demote_pages=copy_on_demote_pages,
                eligible_full_unshared_pages=eligible_full_unshared_pages,
                after_initial_recent_protection=after_initial_recent_protection,
                after_low_risk_filter=after_low_risk_filter,
                after_request_fraction_cap=after_request_fraction_cap,
                after_quality_debt_cap=after_quality_debt_cap,
                after_request_budget_cap=after_request_budget_cap,
                after_sparse_window_quota=after_sparse_window_quota,
                after_frontier_optimizer=after_frontier_optimizer,
                after_int4_pool_limit=candidate_bf16_blocks,
                selected_actual=planned_bf16_blocks,
                int4_free_blocks=int4_capacity_blocks,
                frontier_optimizer_budget=frontier_optimizer_budget,
            )
            return ReflexDemotionPlan(
                [],
                candidate_bf16_blocks=candidate_bf16_blocks,
                planned_bf16_blocks=planned_bf16_blocks,
                candidate_breakdown=candidate_breakdown,
            )

        for page in selectable_pages[:candidate_bf16_blocks]:
            if len(selected) >= target_bf16_blocks:
                break
            int4_block_id = int4_pool.allocate()
            if int4_block_id is None:
                candidate_bf16_blocks = len(selected)
                break
            selected.append(
                ReflexDemotion(
                    request_id=page.request_id,
                    page_idx=page.page_idx,
                    bf16_block_id=page.bf16_block_id,
                    int4_block_id=int4_block_id,
                    encoded_block_table_id=encode_int4_block_id(int4_block_id),
                    is_prompt_page=page.is_prompt_page,
                    prompt_pages=prompt_pages_by_request.get(
                        page.request_id,
                    ),
                    risk_score=page.prefill_risk,
                    is_low_risk=page.compressible is True,
                    copy_on_demote=page.copy_on_demote,
                )
            )

        candidate_breakdown = ReflexCandidateBreakdown(
            raw_bf16_pages=raw_bf16_pages,
            open_bf16_pages=open_bf16_pages,
            remote_inflight_bf16_pages=remote_inflight_bf16_pages,
            open_tail_bf16_pages=open_tail_bf16_pages,
            request_protected_bf16_pages=request_protected_bf16_pages,
            shared_bf16_pages=shared_bf16_pages,
            prompt_protected_bf16_pages=prompt_protected_bf16_pages,
            copy_on_demote_pages=copy_on_demote_pages,
            eligible_full_unshared_pages=eligible_full_unshared_pages,
            after_initial_recent_protection=after_initial_recent_protection,
            after_low_risk_filter=after_low_risk_filter,
            after_request_fraction_cap=after_request_fraction_cap,
            after_quality_debt_cap=after_quality_debt_cap,
            after_request_budget_cap=after_request_budget_cap,
            after_sparse_window_quota=after_sparse_window_quota,
            after_frontier_optimizer=after_frontier_optimizer,
            after_int4_pool_limit=candidate_bf16_blocks,
            selected_actual=len(selected),
            int4_free_blocks=int4_capacity_blocks,
            frontier_optimizer_budget=frontier_optimizer_budget,
        )
        return ReflexDemotionPlan(
            selected,
            candidate_bf16_blocks=candidate_bf16_blocks,
            candidate_breakdown=candidate_breakdown,
        )

    @staticmethod
    def _cap_pages_by_request_precision_budget(
        pages: Sequence[ReflexPageMeta],
        *,
        max_new_int4_pages: int,
        max_prompt_int4_pages: int | None,
        existing_prompt_int4_pages: int,
        max_decode_int4_pages: int | None,
        existing_decode_int4_pages: int,
    ) -> list[ReflexPageMeta]:
        remaining_total = max(0, int(max_new_int4_pages))
        if remaining_total <= 0:
            return []
        remaining_prompt = (
            None
            if max_prompt_int4_pages is None
            else max(0, int(max_prompt_int4_pages) - existing_prompt_int4_pages)
        )
        remaining_decode = (
            None
            if max_decode_int4_pages is None
            else max(0, int(max_decode_int4_pages) - existing_decode_int4_pages)
        )
        capped_pages: list[ReflexPageMeta] = []
        for page in pages:
            if len(capped_pages) >= remaining_total:
                break
            if page.is_prompt_page:
                if remaining_prompt is not None and remaining_prompt <= 0:
                    continue
                if remaining_prompt is not None:
                    remaining_prompt -= 1
            else:
                if remaining_decode is not None and remaining_decode <= 0:
                    continue
                if remaining_decode is not None:
                    remaining_decode -= 1
            capped_pages.append(page)
        return capped_pages

    def _candidate_sort_key(self, page: ReflexPageMeta):
        if self.selection_policy in {"oldest", "distance"}:
            return (page.page_idx,)
        if self.selection_policy == "random":
            digest = hashlib.blake2b(
                f"{page.request_id}:{page.page_idx}".encode(),
                digest_size=8,
            ).digest()
            return (int.from_bytes(digest, "big"), page.page_idx)
        if not page.is_prompt_page and page.prefill_risk is None:
            return (0, page.page_idx)
        return (
            1,
            float("inf") if page.prefill_risk is None else page.prefill_risk,
            page.page_idx,
        )
