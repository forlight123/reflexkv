# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Materialization helpers for the dynamic ReFlexKV INT4 prototype."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.reflex_int4_codec import get_reflex_int4_codec
from vllm.v1.core.precision_kv.types import (
    PrecisionState,
    decode_block_table_entry,
)


@dataclass(frozen=True)
class MaterializedReflexInt4KVCache:
    kv_cache: torch.Tensor
    block_table: torch.Tensor


def materialize_reflex_int4_kv_cache(
    *,
    bf16_cache: torch.Tensor,
    int4_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    head_size: int,
    dtype: torch.dtype,
) -> MaterializedReflexInt4KVCache:
    """Gather active BF16/INT4 pages into a compact BF16 KV cache.

    The worker-facing block table uses non-negative ids for BF16 blocks and
    negative ids for INT4 blocks, where INT4 id N is encoded as -(N + 1).
    """

    active_entries = _active_block_table_entries(block_table, seq_lens, block_size)
    if not active_entries:
        empty_shape = (0, *bf16_cache.shape[1:])
        return MaterializedReflexInt4KVCache(
            kv_cache=torch.empty(empty_shape, dtype=dtype, device=bf16_cache.device),
            block_table=torch.zeros_like(block_table),
        )

    workspace = torch.empty(
        (len(active_entries), *bf16_cache.shape[1:]),
        dtype=dtype,
        device=bf16_cache.device,
    )
    entry_to_compact = {entry: idx for idx, entry in enumerate(active_entries)}

    int4_workspace_indices: list[int] = []
    int4_block_ids: list[int] = []
    for workspace_idx, entry in enumerate(active_entries):
        precision, block_id = decode_block_table_entry(entry)
        if precision == PrecisionState.BF16:
            if block_id >= bf16_cache.shape[0]:
                raise ValueError(
                    f"BF16 block id {block_id} exceeds cache size "
                    f"{bf16_cache.shape[0]}."
                )
            workspace[workspace_idx].copy_(bf16_cache[block_id])
        else:
            if block_id >= int4_cache.shape[0]:
                raise ValueError(
                    f"INT4 block id {block_id} exceeds cache size "
                    f"{int4_cache.shape[0]}."
                )
            int4_workspace_indices.append(workspace_idx)
            int4_block_ids.append(block_id)

    if int4_block_ids:
        int4_ids = torch.tensor(
            int4_block_ids, dtype=torch.long, device=int4_cache.device
        )
        compact_int4_cache = int4_cache.index_select(0, int4_ids)
        dequantized = get_reflex_int4_codec().dequantize_kv_cache(
            compact_int4_cache,
            head_size=head_size,
            dtype=dtype,
        )
        workspace_indices = torch.tensor(
            int4_workspace_indices, dtype=torch.long, device=workspace.device
        )
        workspace.index_copy_(0, workspace_indices, dequantized)

    compact_table = _compact_block_table(
        block_table=block_table,
        seq_lens=seq_lens,
        block_size=block_size,
        entry_to_compact=entry_to_compact,
    )
    return MaterializedReflexInt4KVCache(
        kv_cache=workspace,
        block_table=compact_table,
    )


def _active_block_table_entries(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
) -> list[int]:
    block_table_cpu = block_table.detach().to("cpu")
    seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
    seen: set[int] = set()
    entries: list[int] = []
    for req_idx, seq_len_raw in enumerate(seq_lens_cpu):
        total_pages = (int(seq_len_raw) + block_size - 1) // block_size
        row = block_table_cpu[req_idx]
        for page_idx in range(min(total_pages, row.numel())):
            entry = int(row[page_idx].item())
            if entry not in seen:
                seen.add(entry)
                entries.append(entry)
    return entries


def _compact_block_table(
    *,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    block_size: int,
    entry_to_compact: dict[int, int],
) -> torch.Tensor:
    block_table_cpu = block_table.detach().to("cpu")
    seq_lens_cpu = seq_lens.detach().to("cpu").tolist()
    rows: list[list[int]] = []
    for req_idx, seq_len_raw in enumerate(seq_lens_cpu):
        total_pages = (int(seq_len_raw) + block_size - 1) // block_size
        row = block_table_cpu[req_idx]
        remapped: list[int] = []
        for page_idx in range(row.numel()):
            if page_idx < total_pages:
                remapped.append(entry_to_compact[int(row[page_idx].item())])
            else:
                remapped.append(0)
        rows.append(remapped)
    return torch.tensor(rows, dtype=block_table.dtype, device=block_table.device)
