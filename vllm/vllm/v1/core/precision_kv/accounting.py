# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Telemetry and accounting helpers for precision-aware KV state."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ReflexBlockTableStats:
    total_blocks: int
    bf16_blocks: int
    int4_blocks: int


def summarize_reflex_block_table(
    block_table: Sequence[Sequence[int]],
    num_blocks_per_row: Sequence[int],
    *,
    num_rows: int,
) -> ReflexBlockTableStats:
    total_blocks = 0
    int4_blocks = 0
    for row_idx in range(num_rows):
        row_blocks = int(num_blocks_per_row[row_idx])
        total_blocks += row_blocks
        for entry in block_table[row_idx][:row_blocks]:
            if int(entry) < 0:
                int4_blocks += 1
    return ReflexBlockTableStats(
        total_blocks=total_blocks,
        bf16_blocks=total_blocks - int4_blocks,
        int4_blocks=int4_blocks,
    )
