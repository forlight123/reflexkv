# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Risk estimation and page selection helpers for precision-elastic KV."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PageRiskSummary:
    """Compact prefill-side semantic/risk summary for one KV page."""

    request_id: str
    page_idx: int
    token_start: int
    token_end: int
    risk_score: float
    semantic_hash: str
    compressible: bool


def _clamp_fraction(value: float | None, default: float = 1.0) -> float:
    if value is None:
        value = default
    return min(1.0, max(0.0, float(value)))


def derive_compressible_pages_from_risks(
    page_risks: Sequence[float],
    *,
    fraction: float,
    explicit_pages: set[int] | None = None,
) -> set[int]:
    """Return the low-risk page set used by planner/admission paths."""

    compressible_pages = set(explicit_pages or set())
    risk_count = len(page_risks)
    if risk_count <= 0:
        return compressible_pages
    normalized_fraction = _clamp_fraction(fraction)
    candidate_count = int(risk_count * normalized_fraction)
    if normalized_fraction > 0.0 and candidate_count == 0:
        candidate_count = 1
    if candidate_count <= 0:
        return compressible_pages
    ordered = sorted(
        range(risk_count),
        key=lambda idx: (float(page_risks[idx]), idx),
    )
    compressible_pages.update(ordered[:candidate_count])
    return compressible_pages


def select_bf16_shadow_pages(
    page_risks: Sequence[float],
    *,
    max_pages: int,
    min_risk: float | None = None,
) -> set[int]:
    """Pick high-risk pages that should keep a BF16 recovery shadow."""

    if max_pages <= 0 or not page_risks:
        return set()
    threshold = -math.inf if min_risk is None else float(min_risk)
    candidates = [
        (float(score), -page_idx, page_idx)
        for page_idx, score in enumerate(page_risks)
        if float(score) >= threshold
    ]
    candidates.sort(reverse=True)
    return {page_idx for _score, _neg_idx, page_idx in candidates[:max_pages]}


def synthesize_remote_chunk_landing_pages(
    *,
    page_start: int,
    page_end: int,
    page_count: int,
    keep_initial_pages: int,
    keep_recent_pages: int,
    protected_prompt_pages: int,
    max_int4_fraction: float,
    short_prefill_pages: int,
) -> tuple[int, ...]:
    """Build conservative direct-landing candidates for the current P/D chunk.

    This is a fallback for the real system race where decode admission can run
    before producer-side page-risk metadata has reached the decode scheduler.
    It only exposes pages from the currently transferred chunk.
    """

    page_count = max(0, int(page_count))
    if page_count <= 0 or page_count <= max(0, int(short_prefill_pages)):
        return ()
    page_start = max(0, int(page_start))
    page_end = min(page_count, max(page_start, int(page_end)))
    if page_end <= page_start:
        return ()

    keep_initial_pages = max(0, int(keep_initial_pages))
    keep_recent_pages = max(0, int(keep_recent_pages))
    protected_prompt_pages = max(0, int(protected_prompt_pages))
    recent_start = max(0, page_count - keep_recent_pages)
    candidates = [
        page_idx
        for page_idx in range(page_start, page_end)
        if page_idx >= keep_initial_pages
        and page_idx >= protected_prompt_pages
        and page_idx < recent_start
    ]
    if not candidates:
        return ()

    max_int4_pages = int(page_count * _clamp_fraction(max_int4_fraction))
    if max_int4_pages <= 0:
        return ()
    return tuple(candidates[:max_int4_pages])


class PrefillRiskEstimator:
    """Semantic anchor estimator used by P-side page-risk generation."""

    def __init__(
        self,
        *,
        compressible_fraction: float = 0.25,
    ) -> None:
        self.compressible_fraction = _clamp_fraction(compressible_fraction)

    def estimate_from_anchors(
        self,
        *,
        request_id: str,
        q_tail_anchor: torch.Tensor,
        page_key_anchors: Sequence[torch.Tensor],
        block_size: int,
    ) -> list[PageRiskSummary]:
        if not page_key_anchors:
            return []
        risks = [
            self._normalized_cosine_risk(q_tail_anchor, page_anchor)
            for page_anchor in page_key_anchors
        ]
        compressible_pages = derive_compressible_pages_from_risks(
            risks,
            fraction=self.compressible_fraction,
        )
        block_size = max(1, int(block_size))
        return [
            PageRiskSummary(
                request_id=request_id,
                page_idx=page_idx,
                token_start=page_idx * block_size,
                token_end=(page_idx + 1) * block_size,
                risk_score=risks[page_idx],
                semantic_hash=self._semantic_hash(page_key_anchors[page_idx]),
                compressible=page_idx in compressible_pages,
            )
            for page_idx in range(len(page_key_anchors))
        ]

    @staticmethod
    def _normalized_cosine_risk(
        q_anchor: torch.Tensor,
        page_anchor: torch.Tensor,
    ) -> float:
        q = q_anchor.detach().float().reshape(-1)
        k = page_anchor.detach().float().reshape(-1)
        if q.numel() != k.numel() or q.numel() == 0:
            return 0.5
        q_norm = torch.linalg.vector_norm(q)
        k_norm = torch.linalg.vector_norm(k)
        if float(q_norm.item()) <= 0.0 or float(k_norm.item()) <= 0.0:
            return 0.5
        cosine = float(torch.dot(q, k).item() / (q_norm.item() * k_norm.item()))
        return _clamp_fraction((cosine + 1.0) * 0.5)

    @staticmethod
    def _semantic_hash(anchor: torch.Tensor) -> str:
        values = anchor.detach().float().reshape(-1).cpu()
        if values.numel() == 0:
            return "0" * 16
        max_abs = float(torch.max(torch.abs(values)).item())
        if max_abs <= 0.0:
            quantized = torch.zeros_like(values, dtype=torch.int8)
        else:
            quantized = torch.clamp(torch.round(values / max_abs * 31), -32, 31)
            quantized = quantized.to(torch.int8)
        digest = hashlib.blake2b(
            quantized.numpy().tobytes(),
            digest_size=8,
        ).hexdigest()
        return digest
