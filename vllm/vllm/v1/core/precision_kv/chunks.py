# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Chunk-level contracts for precision-aware remote KV transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


REFLEX_REMOTE_CHUNK_CONTRACT_KEYS = (
    "reflex_remote_chunk_enabled",
    "reflex_remote_chunk_tokens",
    "reflex_remote_chunk_role",
    "reflex_remote_chunk_id",
    "reflex_remote_chunk_token_start",
    "reflex_remote_chunk_token_end",
    "reflex_remote_chunk_page_start",
    "reflex_remote_chunk_page_end",
    "reflex_remote_chunk_is_last",
)


@dataclass(frozen=True)
class RemoteKVChunk:
    request_id: str
    chunk_id: int
    token_start: int
    token_end: int
    page_start: int
    page_end: int
    block_size: int
    is_last_chunk: bool

    @property
    def num_tokens(self) -> int:
        return max(0, self.token_end - self.token_start)

    @property
    def num_pages(self) -> int:
        return max(0, self.page_end - self.page_start)


def remote_chunking_enabled(params: dict[str, Any] | None) -> bool:
    return bool(params and params.get("reflex_remote_chunk_enabled"))


def normalize_remote_chunk_tokens(chunk_tokens: int, block_size: int) -> int:
    block_size = max(1, int(block_size))
    chunk_tokens = max(block_size, int(chunk_tokens))
    return max(block_size, (chunk_tokens // block_size) * block_size)


def plan_remote_kv_chunk(
    *,
    request_id: str,
    prompt_token_count: int,
    num_computed_tokens: int,
    block_size: int,
    chunk_tokens: int,
) -> RemoteKVChunk | None:
    prompt_token_count = max(0, int(prompt_token_count))
    num_computed_tokens = max(0, int(num_computed_tokens))
    if num_computed_tokens >= prompt_token_count:
        return None

    block_size = max(1, int(block_size))
    normalized_chunk_tokens = normalize_remote_chunk_tokens(
        chunk_tokens,
        block_size,
    )
    token_start = num_computed_tokens
    remaining = prompt_token_count - token_start
    token_end = (
        prompt_token_count
        if remaining <= normalized_chunk_tokens
        else token_start + normalized_chunk_tokens
    )
    page_start = token_start // block_size
    page_end = (token_end + block_size - 1) // block_size
    return RemoteKVChunk(
        request_id=str(request_id),
        chunk_id=token_start // normalized_chunk_tokens,
        token_start=token_start,
        token_end=token_end,
        page_start=page_start,
        page_end=page_end,
        block_size=block_size,
        is_last_chunk=token_end >= prompt_token_count,
    )


def write_remote_kv_chunk_contract(
    params: dict[str, Any],
    chunk: RemoteKVChunk,
    *,
    role: Literal["prefill", "decode"],
) -> None:
    params["reflex_remote_chunk_enabled"] = True
    params["reflex_remote_chunk_role"] = role
    params["reflex_remote_chunk_id"] = int(chunk.chunk_id)
    params["reflex_remote_chunk_token_start"] = int(chunk.token_start)
    params["reflex_remote_chunk_token_end"] = int(chunk.token_end)
    params["reflex_remote_chunk_page_start"] = int(chunk.page_start)
    params["reflex_remote_chunk_page_end"] = int(chunk.page_end)
    params["reflex_remote_chunk_is_last"] = bool(chunk.is_last_chunk)


def clear_remote_kv_chunk_contract(params: dict[str, Any] | None) -> None:
    if not isinstance(params, dict):
        return
    for key in REFLEX_REMOTE_CHUNK_CONTRACT_KEYS:
        if key != "reflex_remote_chunk_enabled":
            params.pop(key, None)
