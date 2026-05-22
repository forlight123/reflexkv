# SPDX-License-Identifier: Apache-2.0
"""Lightweight prefill-side page scoring for ReFlexKV demotion.

This module intentionally records only page-level summaries. It never
materializes an attention matrix and is enabled only when
SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA is set.
"""

from __future__ import annotations

import math
import os
import re
import threading
from collections import defaultdict
from collections.abc import Iterable
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch

from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.core.precision_kv.risk import (
    PageRiskSummary,
    PrefillRiskEstimator,
)


_TRUE_VALUES = {"1", "true", "yes", "on"}
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def reflex_prefill_metadata_enabled() -> bool:
    return (
        os.environ.get("SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA", "").lower()
        in _TRUE_VALUES
    )


def _read_positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _selected_layer(layer_name: str) -> bool:
    raw_ids = os.environ.get("SEMANTIQ_REFLEX_PREFILL_SCORE_LAYER_IDS")
    match = _LAYER_RE.search(layer_name)
    layer_idx = int(match.group(1)) if match else None
    if raw_ids:
        allowed = set[int]()
        for item in raw_ids.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                allowed.add(int(item))
            except ValueError:
                continue
        return layer_idx in allowed if layer_idx is not None else False

    if layer_idx is None:
        return False
    period = _read_positive_int_env(
        "SEMANTIQ_REFLEX_PREFILL_SCORE_LAYER_PERIOD",
        8,
    )
    max_layers = _read_positive_int_env(
        "SEMANTIQ_REFLEX_PREFILL_SCORE_MAX_LAYERS",
        4,
    )
    return layer_idx % period == 0 and layer_idx // period < max_layers


@dataclass(frozen=True)
class ReflexPrefillRequestInfo:
    request_id: str
    query_start: int
    query_end: int
    prompt_start: int
    prompt_end: int
    prompt_tokens: int


@dataclass
class _LayerPageStats:
    q_tail_sum: torch.Tensor | None = None
    q_tail_count: int = 0
    q_all_sum: torch.Tensor | None = None
    q_all_count: int = 0
    page_sums: dict[int, torch.Tensor] = field(default_factory=dict)
    page_counts: dict[int, int] = field(default_factory=dict)

    def add_q(self, q_sum: torch.Tensor, q_count: int, *, tail: bool) -> None:
        if q_count <= 0:
            return
        attr_sum = "q_tail_sum" if tail else "q_all_sum"
        attr_count = "q_tail_count" if tail else "q_all_count"
        current = getattr(self, attr_sum)
        setattr(self, attr_sum, q_sum if current is None else current + q_sum)
        setattr(self, attr_count, getattr(self, attr_count) + q_count)

    def add_page(self, page_idx: int, k_sum: torch.Tensor, k_count: int) -> None:
        if k_count <= 0:
            return
        if page_idx in self.page_sums:
            self.page_sums[page_idx] += k_sum
            self.page_counts[page_idx] += k_count
        else:
            self.page_sums[page_idx] = k_sum
            self.page_counts[page_idx] = k_count


class ReflexPrefillMetadataRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._batch_infos: list[ReflexPrefillRequestInfo] = []
        self._block_size = 16
        self._stats: dict[str, dict[str, _LayerPageStats]] = defaultdict(dict)
        self._prompt_tokens_by_request: dict[str, int] = {}
        self._completed_requests: set[str] = set()

    @contextmanager
    def batch(
        self,
        infos: list[ReflexPrefillRequestInfo],
        *,
        block_size: int,
    ):
        if not reflex_prefill_metadata_enabled() or not infos:
            yield
            return
        with self._lock:
            self._batch_infos = infos
            self._block_size = max(1, int(block_size))
            for info in infos:
                self._prompt_tokens_by_request[info.request_id] = info.prompt_tokens
                if info.prompt_end >= info.prompt_tokens:
                    self._completed_requests.add(info.request_id)
        try:
            yield
        finally:
            with self._lock:
                self._batch_infos = []

    def record_layer(
        self,
        layer_name: str,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> None:
        if not reflex_prefill_metadata_enabled() or not _selected_layer(layer_name):
            return
        with self._lock:
            infos = list(self._batch_infos)
            block_size = self._block_size
        if not infos:
            return

        tail_tokens = _read_positive_int_env(
            "SEMANTIQ_REFLEX_PREFILL_Q_TAIL_TOKENS",
            256,
        )
        with torch.no_grad():
            for info in infos:
                if info.query_end <= info.query_start:
                    continue
                q_slice = query[info.query_start : info.query_end]
                k_slice = key[info.query_start : info.query_end]
                if q_slice.numel() == 0 or k_slice.numel() == 0:
                    continue

                layer_stats = self._stats[info.request_id].setdefault(
                    layer_name,
                    _LayerPageStats(),
                )
                q_sum = q_slice.float().sum(dim=(0, 1)).detach().cpu()
                layer_stats.add_q(
                    q_sum,
                    q_slice.shape[0] * q_slice.shape[1],
                    tail=False,
                )

                tail_start = max(0, info.prompt_tokens - tail_tokens)
                tail_abs_start = max(info.prompt_start, tail_start)
                tail_abs_end = info.prompt_end
                if tail_abs_end > tail_abs_start:
                    local_start = tail_abs_start - info.prompt_start
                    local_end = tail_abs_end - info.prompt_start
                    q_tail = q_slice[local_start:local_end]
                    if q_tail.numel() > 0:
                        layer_stats.add_q(
                            q_tail.float().sum(dim=(0, 1)).detach().cpu(),
                            q_tail.shape[0] * q_tail.shape[1],
                            tail=True,
                        )

                abs_pos = info.prompt_start
                while abs_pos < info.prompt_end:
                    page_idx = abs_pos // block_size
                    page_end = min(info.prompt_end, (page_idx + 1) * block_size)
                    local_start = abs_pos - info.prompt_start
                    local_end = page_end - info.prompt_start
                    k_page = k_slice[local_start:local_end]
                    if k_page.numel() > 0:
                        layer_stats.add_page(
                            page_idx,
                            k_page.float().sum(dim=(0, 1)).detach().cpu(),
                            k_page.shape[0] * k_page.shape[1],
                        )
                    abs_pos = page_end

    def finalize_request(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        block_size: int,
    ) -> list[float] | None:
        summaries = self.finalize_request_summaries(
            request_id,
            prompt_tokens=prompt_tokens,
            block_size=block_size,
        )
        if not summaries:
            return None
        return [summary.risk_score for summary in summaries]

    def finalize_request_summaries(
        self,
        request_id: str,
        *,
        prompt_tokens: int,
        block_size: int,
    ) -> list[PageRiskSummary] | None:
        page_count = (max(0, prompt_tokens) + max(1, block_size) - 1) // max(
            1,
            block_size,
        )
        if page_count <= 0:
            return None
        with self._lock:
            layers = self._stats.pop(request_id, None)
            self._prompt_tokens_by_request.pop(request_id, None)
        if not layers:
            return None

        page_scores = [float("-inf")] * page_count
        q_anchor_sum: torch.Tensor | None = None
        q_anchor_layers = 0
        page_anchor_sums: dict[int, torch.Tensor] = {}
        page_anchor_counts: dict[int, int] = {}
        for layer_stats in layers.values():
            q_sum = (
                layer_stats.q_tail_sum
                if layer_stats.q_tail_count > 0
                else layer_stats.q_all_sum
            )
            q_count = (
                layer_stats.q_tail_count
                if layer_stats.q_tail_count > 0
                else layer_stats.q_all_count
            )
            if q_sum is None or q_count <= 0:
                continue
            q_anchor = q_sum / float(q_count)
            if q_anchor_sum is None:
                q_anchor_sum = q_anchor.clone()
                q_anchor_layers = 1
            elif q_anchor_sum.numel() == q_anchor.numel():
                q_anchor_sum += q_anchor
                q_anchor_layers += 1
            scale = math.sqrt(max(1, q_anchor.numel()))
            for page_idx, k_sum in layer_stats.page_sums.items():
                if page_idx >= page_count:
                    continue
                k_count = layer_stats.page_counts.get(page_idx, 0)
                if k_count <= 0:
                    continue
                k_anchor = k_sum / float(k_count)
                if page_idx not in page_anchor_sums:
                    page_anchor_sums[page_idx] = k_anchor.clone()
                    page_anchor_counts[page_idx] = 1
                elif page_anchor_sums[page_idx].numel() == k_anchor.numel():
                    page_anchor_sums[page_idx] += k_anchor
                    page_anchor_counts[page_idx] += 1
                score = float(torch.dot(q_anchor, k_anchor).item() / scale)
                page_scores[page_idx] = max(page_scores[page_idx], score)

        finite_scores = [score for score in page_scores if math.isfinite(score)]
        if not finite_scores:
            return None
        if q_anchor_sum is not None and q_anchor_layers > 0:
            q_anchor = q_anchor_sum / float(q_anchor_layers)
            fallback_anchor = torch.zeros_like(q_anchor)
            page_anchors = [
                (
                    page_anchor_sums[page_idx]
                    / float(page_anchor_counts[page_idx])
                    if page_idx in page_anchor_sums
                    else fallback_anchor
                )
                for page_idx in range(page_count)
            ]
            return PrefillRiskEstimator().estimate_from_anchors(
                request_id=request_id,
                q_tail_anchor=q_anchor,
                page_key_anchors=page_anchors,
                block_size=block_size,
            )
        fallback = max(finite_scores) + 1.0
        scores = [
            score if math.isfinite(score) else fallback for score in page_scores
        ]
        compressible_pages = set(
            sorted(range(len(scores)), key=lambda idx: (scores[idx], idx))[
                : max(1, int(len(scores) * 0.25))
            ]
        )
        return [
            PageRiskSummary(
                request_id=request_id,
                page_idx=page_idx,
                token_start=page_idx * max(1, block_size),
                token_end=(page_idx + 1) * max(1, block_size),
                risk_score=scores[page_idx],
                semantic_hash="0" * 16,
                compressible=page_idx in compressible_pages,
            )
            for page_idx in range(page_count)
        ]

    def finalize_requests(
        self,
        request_ids: Iterable[str],
        *,
        prompt_tokens_by_request: dict[str, int],
        block_size: int,
    ) -> dict[str, list[float]]:
        results: dict[str, list[float]] = {}
        for request_id in request_ids:
            prompt_tokens = prompt_tokens_by_request.get(request_id)
            if prompt_tokens is None:
                with self._lock:
                    prompt_tokens = self._prompt_tokens_by_request.get(request_id, 0)
            scores = self.finalize_request(
                request_id,
                prompt_tokens=prompt_tokens,
                block_size=block_size,
            )
            if scores is not None:
                results[request_id] = scores
        return results

    def drain_completed_requests(self, *, block_size: int) -> dict[str, list[float]]:
        with self._lock:
            request_ids = set(self._completed_requests)
            self._completed_requests.clear()
        return self.finalize_requests(
            request_ids,
            prompt_tokens_by_request={},
            block_size=block_size,
        )


_RECORDER = ReflexPrefillMetadataRecorder()


def get_reflex_prefill_metadata_recorder() -> ReflexPrefillMetadataRecorder:
    return _RECORDER


def _record_reflex_prefill_metadata_op(
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> None:
    _RECORDER.record_layer(layer_name, query, key)


def _record_reflex_prefill_metadata_fake(
    query: torch.Tensor,
    key: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="record_reflex_prefill_metadata",
    op_func=_record_reflex_prefill_metadata_op,
    mutates_args=["query", "key"],
    fake_impl=_record_reflex_prefill_metadata_fake,
    dispatch_key="CompositeExplicitAutograd",
)


def maybe_record_reflex_prefill_layer(
    layer_name: str,
    query: torch.Tensor,
    key: torch.Tensor | None,
) -> None:
    if key is None:
        return
    torch.ops.vllm.record_reflex_prefill_metadata(query, key, layer_name)
