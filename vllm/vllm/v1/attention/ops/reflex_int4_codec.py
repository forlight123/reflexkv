# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization codec boundary for ReFlexKV INT4 KV pages."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.v1.attention.ops.int4_kv_cache import (
    int4_dequantize_kv_cache,
    int4_packed_head_size_bytes,
    int4_quantize_and_cache,
    int4_quantize_blocks_to_cache,
)


@dataclass(frozen=True)
class ReflexInt4Codec:
    """Thin codec facade over the current token-local INT4 implementation."""

    name: str = "token_channel_int4"

    def packed_head_size_bytes(self, head_size: int) -> int:
        return int4_packed_head_size_bytes(head_size)

    def quantize_and_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        int4_quantize_and_cache(key, value, kv_cache, slot_mapping)

    def quantize_blocks_to_cache(
        self,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
    ) -> None:
        int4_quantize_blocks_to_cache(
            src_cache,
            dst_cache,
            src_block_ids,
            dst_block_ids,
        )

    def dequantize_kv_cache(
        self,
        kv_cache: torch.Tensor,
        *,
        head_size: int,
        dtype: torch.dtype,
        num_blocks: int | None = None,
    ) -> torch.Tensor:
        return int4_dequantize_kv_cache(
            kv_cache,
            head_size=head_size,
            dtype=dtype,
            num_blocks=num_blocks,
        )


_DEFAULT_REFLEX_INT4_CODEC = ReflexInt4Codec()


def get_reflex_int4_codec() -> ReflexInt4Codec:
    return _DEFAULT_REFLEX_INT4_CODEC
