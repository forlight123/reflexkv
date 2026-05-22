# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lightweight dual run optimizer for precision-aware KV demotion."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from vllm.v1.core.precision_kv.types import ReflexPageMeta


@dataclass(frozen=True)
class RunCandidate:
    request_id: str
    start_page: int
    end_page: int
    pages: tuple[ReflexPageMeta, ...]
    saving_blocks: int
    quality_risk: float
    migration_cost: float
    admission_benefit: float
    slo_risk: float
    is_prompt_run: bool
    window_id: int = 0
    backlog_cost: float = 0.0
    is_decode_run: bool = True
    constraint_signature: str = ""

    @property
    def num_pages(self) -> int:
        return self.saving_blocks


@dataclass(frozen=True)
class DualPriceState:
    memory_price: float = 1.0
    admission_price: float = 0.5
    quality_price: float = 2.0
    migration_price: float = 0.1
    slo_price: float = 0.0
    min_price: float = 0.0
    max_price: float = 16.0

    def updated(
        self,
        *,
        kv_usage: float,
        kv_target: float,
        waiting_requests: int,
        waiting_target: int,
        migration_backlog: int,
        migration_target: int,
        eta: float,
    ) -> DualPriceState:
        return DualPriceState(
            memory_price=self._clip(self.memory_price + eta * (kv_usage - kv_target)),
            admission_price=self._clip(
                self.admission_price + eta * (waiting_requests - waiting_target)
            ),
            quality_price=self.quality_price,
            migration_price=self._clip(
                self.migration_price + eta * (migration_backlog - migration_target)
            ),
            slo_price=self.slo_price,
            min_price=self.min_price,
            max_price=self.max_price,
        )

    def _clip(self, value: float) -> float:
        return min(self.max_price, max(self.min_price, value))


class DualRunOptimizer:
    """Greedy primal-dual selector over window-level page runs."""

    def __init__(
        self,
        *,
        dual_prices: DualPriceState | None = None,
        sparse_window_pages: int = 0,
        max_pages_per_window: int = 0,
        max_run_pages: int | None = None,
    ) -> None:
        self.dual_prices = dual_prices or DualPriceState()
        self.sparse_window_pages = max(0, sparse_window_pages)
        self.max_pages_per_window = max(0, max_pages_per_window)
        self.max_run_pages = (
            max(1, int(max_run_pages))
            if max_run_pages is not None and max_run_pages > 0
            else None
        )

    def build_runs(
        self,
        pages: Iterable[ReflexPageMeta],
    ) -> list[RunCandidate]:
        ordered_pages = sorted(pages, key=lambda page: page.page_idx)
        runs: list[RunCandidate] = []
        current: list[ReflexPageMeta] = []

        for page in ordered_pages:
            if not current or self._can_extend_run(current[-1], page, current):
                current.append(page)
            else:
                runs.append(self._make_run(current))
                current = [page]
        if current:
            runs.append(self._make_run(current))
        return self.prune_dominated_runs(runs)

    def prune_dominated_runs(
        self,
        runs: Iterable[RunCandidate],
    ) -> list[RunCandidate]:
        run_list = list(runs)
        kept: list[RunCandidate] = []
        for idx, run in enumerate(run_list):
            if any(
                self._dominates(other, run)
                for other_idx, other in enumerate(run_list)
                if other_idx != idx
            ):
                continue
            kept.append(run)
        return kept

    def select_pages(
        self,
        pages: Iterable[ReflexPageMeta],
        *,
        target_release: int,
        int4_capacity_blocks: int,
    ) -> list[ReflexPageMeta]:
        selected_runs = self.select_runs(
            self.build_runs(pages),
            target_release=target_release,
            int4_capacity_blocks=int4_capacity_blocks,
        )
        selected_pages: list[ReflexPageMeta] = []
        for run in selected_runs:
            selected_pages.extend(run.pages)
        return selected_pages

    def select_runs(
        self,
        runs: Iterable[RunCandidate],
        *,
        target_release: int,
        int4_capacity_blocks: int,
    ) -> list[RunCandidate]:
        budget = min(max(0, target_release), max(0, int4_capacity_blocks))
        if budget <= 0:
            return []
        selected: list[RunCandidate] = []
        selected_pages = 0
        selected_by_window: dict[tuple[str, int], int] = defaultdict(int)
        ranked_runs = sorted(
            runs,
            key=lambda run: (
                self._score_density(run),
                self._score(run),
                -run.start_page,
            ),
            reverse=True,
        )
        for run in ranked_runs:
            if selected_pages >= budget:
                break
            remaining_budget = budget - selected_pages
            window_key = (run.request_id, self._window_id(run.start_page))
            window_remaining = remaining_budget
            if self.max_pages_per_window > 0:
                window_remaining = min(
                    window_remaining,
                    self.max_pages_per_window - selected_by_window[window_key],
                )
            if window_remaining <= 0:
                continue
            selected_run = run
            if run.saving_blocks > window_remaining:
                selected_run = self._slice_run(run, int(window_remaining))
            selected.append(selected_run)
            selected_pages += selected_run.saving_blocks
            selected_by_window[window_key] += selected_run.saving_blocks
        return selected

    def _can_extend_run(
        self,
        previous: ReflexPageMeta,
        page: ReflexPageMeta,
        current_run: list[ReflexPageMeta],
    ) -> bool:
        if page.request_id != previous.request_id:
            return False
        if page.page_idx != previous.page_idx + 1:
            return False
        if page.is_prompt_page != previous.is_prompt_page:
            return False
        if self._window_id(page.page_idx) != self._window_id(previous.page_idx):
            return False
        if self.max_run_pages is not None and len(current_run) >= self.max_run_pages:
            return False
        return True

    def _make_run(self, pages: list[ReflexPageMeta]) -> RunCandidate:
        quality_risk = sum(self._page_risk(page) for page in pages)
        saving_blocks = len(pages)
        window_id = self._window_id(pages[0].page_idx)
        is_prompt_run = pages[0].is_prompt_page
        run_kind = "prompt" if is_prompt_run else "decode"
        return RunCandidate(
            request_id=pages[0].request_id,
            start_page=pages[0].page_idx,
            end_page=pages[-1].page_idx,
            pages=tuple(pages),
            saving_blocks=saving_blocks,
            quality_risk=quality_risk,
            migration_cost=float(saving_blocks),
            admission_benefit=float(saving_blocks),
            slo_risk=0.0,
            is_prompt_run=is_prompt_run,
            window_id=window_id,
            backlog_cost=0.0,
            is_decode_run=not is_prompt_run,
            constraint_signature=f"{run_kind}:window={window_id}",
        )

    def _slice_run(self, run: RunCandidate, num_pages: int) -> RunCandidate:
        if num_pages <= 0:
            raise ValueError("num_pages must be positive when slicing a run.")
        if num_pages >= run.saving_blocks:
            return run
        return self._make_run(list(run.pages[:num_pages]))

    def _dominates(
        self,
        candidate: RunCandidate,
        other: RunCandidate,
    ) -> bool:
        if candidate.request_id != other.request_id:
            return False
        if self._window_id(candidate.start_page) != self._window_id(other.start_page):
            return False
        no_worse = (
            candidate.saving_blocks >= other.saving_blocks
            and candidate.quality_risk <= other.quality_risk
            and candidate.migration_cost <= other.migration_cost
        )
        strictly_better = (
            candidate.saving_blocks > other.saving_blocks
            or candidate.quality_risk < other.quality_risk
            or candidate.migration_cost < other.migration_cost
        )
        return no_worse and strictly_better

    def _score_density(self, run: RunCandidate) -> float:
        return self._score(run) / max(1, run.saving_blocks)

    def _score(self, run: RunCandidate) -> float:
        prices = self.dual_prices
        return (
            prices.memory_price * run.saving_blocks
            + prices.admission_price * run.admission_benefit
            - prices.quality_price * run.quality_risk
            - prices.migration_price * run.migration_cost
            - prices.migration_price * run.backlog_cost
            - prices.slo_price * run.slo_risk
        )

    def _window_id(self, page_idx: int) -> int:
        if self.sparse_window_pages <= 0:
            return 0
        return page_idx // self.sparse_window_pages

    @staticmethod
    def _page_risk(page: ReflexPageMeta) -> float:
        if page.prefill_risk is not None:
            return max(0.0, float(page.prefill_risk))
        if page.is_prompt_page:
            return 1.0
        if page.compressible is False:
            return 1.0
        return 0.0
