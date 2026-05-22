# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental packed INT4 KV cache helpers.

This is a static all-INT4 prototype path for research experiments. Each
token/head vector stores two signed INT4 values per byte, followed by one byte
containing a power-of-two scale exponent biased by 127.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def int4_packed_head_size_bytes(head_size: int) -> int:
    if head_size % 2 != 0:
        raise ValueError("INT4 KV cache requires an even head_size.")
    return head_size // 2 + 1


@triton.jit
def _round_to_nearest(x):
    return tl.where(x >= 0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _quantize_and_cache_kernel(
    key_ptr,
    value_ptr,
    cache_ptr,
    slot_mapping_ptr,
    key_stride_token: tl.constexpr,
    key_stride_head: tl.constexpr,
    key_stride_dim: tl.constexpr,
    value_stride_token: tl.constexpr,
    value_stride_head: tl.constexpr,
    value_stride_dim: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_kv: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_byte: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    block_head_half: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_idx = tl.program_id(2)
    slot_idx = tl.load(slot_mapping_ptr + token_idx)

    half_offsets = tl.arange(0, block_head_half)
    mask = half_offsets < head_size // 2
    dim0 = half_offsets * 2
    dim1 = dim0 + 1

    src_ptr = tl.where(kv_idx == 0, key_ptr, value_ptr)
    stride_token = tl.where(kv_idx == 0, key_stride_token, value_stride_token)
    stride_head = tl.where(kv_idx == 0, key_stride_head, value_stride_head)
    stride_dim = tl.where(kv_idx == 0, key_stride_dim, value_stride_dim)
    src_base = src_ptr + token_idx * stride_token + head_idx * stride_head
    val0 = tl.load(src_base + dim0 * stride_dim, mask=mask, other=0.0).to(tl.float32)
    val1 = tl.load(src_base + dim1 * stride_dim, mask=mask, other=0.0).to(tl.float32)

    amax = tl.max(tl.maximum(tl.abs(val0), tl.abs(val1)), axis=0)
    scale = tl.maximum(amax / 7.0, 1.0e-6)
    scale_exp = tl.ceil(tl.log2(scale))
    scale = tl.exp2(scale_exp)
    scale_byte = tl.minimum(tl.maximum(scale_exp + 127.0, 0.0), 255.0).to(tl.uint8)

    q0 = _round_to_nearest(val0 / scale)
    q1 = _round_to_nearest(val1 / scale)
    q0 = tl.minimum(tl.maximum(q0, -8.0), 7.0).to(tl.int32)
    q1 = tl.minimum(tl.maximum(q1, -8.0), 7.0).to(tl.int32)
    packed = ((q0 & 0xF) | ((q1 & 0xF) << 4)).to(tl.uint8)

    block_idx = slot_idx // block_size
    block_offset = slot_idx - block_idx * block_size
    cache_base = (
        cache_ptr
        + block_idx * cache_stride_block
        + kv_idx * cache_stride_kv
        + block_offset * cache_stride_token
        + head_idx * cache_stride_head
    )
    valid_slot = slot_idx >= 0
    tl.store(
        cache_base + half_offsets * cache_stride_byte,
        packed,
        mask=mask & valid_slot,
    )
    tl.store(
        cache_base + (head_size // 2) * cache_stride_byte,
        scale_byte,
        mask=valid_slot,
    )


@triton.jit
def _dequantize_cache_kernel(
    cache_ptr,
    out_ptr,
    cache_stride_block: tl.constexpr,
    cache_stride_kv: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_head: tl.constexpr,
    cache_stride_byte: tl.constexpr,
    out_stride_block: tl.constexpr,
    out_stride_kv: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_head: tl.constexpr,
    out_stride_dim: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    block_head: tl.constexpr,
):
    token_flat = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_idx = tl.program_id(2)
    block_idx = token_flat // block_size
    block_offset = token_flat - block_idx * block_size

    offsets = tl.arange(0, block_head)
    mask = offsets < head_size
    packed_offsets = offsets // 2
    use_high = offsets % 2 == 1

    cache_base = (
        cache_ptr
        + block_idx * cache_stride_block
        + kv_idx * cache_stride_kv
        + block_offset * cache_stride_token
        + head_idx * cache_stride_head
    )
    packed = tl.load(
        cache_base + packed_offsets * cache_stride_byte, mask=mask, other=0
    )
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    nibble = tl.where(use_high, high, low).to(tl.int32)
    signed = tl.where(nibble >= 8, nibble - 16, nibble).to(tl.float32)

    scale_byte = tl.load(cache_base + (head_size // 2) * cache_stride_byte)
    scale = tl.exp2(scale_byte.to(tl.float32) - 127.0)
    dequant = signed * scale

    out_base = (
        out_ptr
        + block_idx * out_stride_block
        + kv_idx * out_stride_kv
        + block_offset * out_stride_token
        + head_idx * out_stride_head
    )
    tl.store(out_base + offsets * out_stride_dim, dequant, mask=mask)


@triton.jit
def _quantize_blocks_to_cache_kernel(
    src_cache_ptr,
    dst_cache_ptr,
    src_block_ids_ptr,
    dst_block_ids_ptr,
    src_stride_block: tl.constexpr,
    src_stride_kv: tl.constexpr,
    src_stride_token: tl.constexpr,
    src_stride_head: tl.constexpr,
    src_stride_dim: tl.constexpr,
    dst_stride_block: tl.constexpr,
    dst_stride_kv: tl.constexpr,
    dst_stride_token: tl.constexpr,
    dst_stride_head: tl.constexpr,
    dst_stride_byte: tl.constexpr,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    block_head_half: tl.constexpr,
):
    token_flat = tl.program_id(0)
    head_idx = tl.program_id(1)
    kv_idx = tl.program_id(2)
    page_idx = token_flat // block_size
    block_offset = token_flat - page_idx * block_size
    src_block_idx = tl.load(src_block_ids_ptr + page_idx)
    dst_block_idx = tl.load(dst_block_ids_ptr + page_idx)

    half_offsets = tl.arange(0, block_head_half)
    mask = half_offsets < head_size // 2
    dim0 = half_offsets * 2
    dim1 = dim0 + 1

    src_base = (
        src_cache_ptr
        + src_block_idx * src_stride_block
        + kv_idx * src_stride_kv
        + block_offset * src_stride_token
        + head_idx * src_stride_head
    )
    val0 = tl.load(src_base + dim0 * src_stride_dim, mask=mask, other=0.0).to(
        tl.float32
    )
    val1 = tl.load(src_base + dim1 * src_stride_dim, mask=mask, other=0.0).to(
        tl.float32
    )

    amax = tl.max(tl.maximum(tl.abs(val0), tl.abs(val1)), axis=0)
    scale = tl.maximum(amax / 7.0, 1.0e-6)
    scale_exp = tl.ceil(tl.log2(scale))
    scale = tl.exp2(scale_exp)
    scale_byte = tl.minimum(tl.maximum(scale_exp + 127.0, 0.0), 255.0).to(tl.uint8)

    q0 = _round_to_nearest(val0 / scale)
    q1 = _round_to_nearest(val1 / scale)
    q0 = tl.minimum(tl.maximum(q0, -8.0), 7.0).to(tl.int32)
    q1 = tl.minimum(tl.maximum(q1, -8.0), 7.0).to(tl.int32)
    packed = ((q0 & 0xF) | ((q1 & 0xF) << 4)).to(tl.uint8)

    dst_base = (
        dst_cache_ptr
        + dst_block_idx * dst_stride_block
        + kv_idx * dst_stride_kv
        + block_offset * dst_stride_token
        + head_idx * dst_stride_head
    )
    tl.store(dst_base + half_offsets * dst_stride_byte, packed, mask=mask)
    tl.store(dst_base + (head_size // 2) * dst_stride_byte, scale_byte)


def int4_quantize_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    num_tokens, num_kv_heads, head_size = key.shape
    if value.shape != key.shape:
        raise ValueError("INT4 KV cache prototype requires K and V shapes to match.")
    packed_head_size = int4_packed_head_size_bytes(head_size)
    if kv_cache.shape[-1] != packed_head_size:
        raise ValueError(
            f"INT4 KV cache packed width mismatch: expected {packed_head_size}, "
            f"got {kv_cache.shape[-1]}."
        )

    grid = (num_tokens, num_kv_heads, 2)
    _quantize_and_cache_kernel[grid](
        key,
        value,
        kv_cache,
        slot_mapping,
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        kv_cache.shape[2],
        head_size,
        triton.next_power_of_2(head_size // 2),
    )


def int4_quantize_blocks_to_cache(
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
    src_block_ids: torch.Tensor,
    dst_block_ids: torch.Tensor,
) -> None:
    if src_cache.ndim != 5:
        raise ValueError("src_cache must have shape [blocks, 2, block, heads, dim].")
    if dst_cache.ndim != 5:
        raise ValueError("dst_cache must have shape [blocks, 2, block, heads, packed].")
    if src_block_ids.shape != dst_block_ids.shape:
        raise ValueError("src_block_ids and dst_block_ids must have the same shape.")
    if src_block_ids.ndim != 1:
        raise ValueError("src_block_ids and dst_block_ids must be 1-D tensors.")
    if src_block_ids.numel() == 0:
        return
    if not src_block_ids.is_cuda or not dst_block_ids.is_cuda:
        raise ValueError("block id tensors must be CUDA tensors.")
    if (
        src_block_ids.device != src_cache.device
        or dst_block_ids.device != dst_cache.device
    ):
        raise ValueError("block id tensors must live on the same devices as caches.")
    if src_cache.shape[1] != 2 or dst_cache.shape[1] != 2:
        raise ValueError("INT4 KV cache expects a K/V dimension of size 2.")
    if (
        src_cache.shape[2] != dst_cache.shape[2]
        or src_cache.shape[3] != dst_cache.shape[3]
    ):
        raise ValueError("src_cache and dst_cache block/head dimensions must match.")

    block_size = src_cache.shape[2]
    num_kv_heads = src_cache.shape[3]
    head_size = src_cache.shape[4]
    packed_head_size = int4_packed_head_size_bytes(head_size)
    if dst_cache.shape[-1] != packed_head_size:
        raise ValueError(
            f"INT4 KV cache packed width mismatch: expected {packed_head_size}, "
            f"got {dst_cache.shape[-1]}."
        )

    grid = (src_block_ids.numel() * block_size, num_kv_heads, 2)
    _quantize_blocks_to_cache_kernel[grid](
        src_cache,
        dst_cache,
        src_block_ids,
        dst_block_ids,
        src_cache.stride(0),
        src_cache.stride(1),
        src_cache.stride(2),
        src_cache.stride(3),
        src_cache.stride(4),
        dst_cache.stride(0),
        dst_cache.stride(1),
        dst_cache.stride(2),
        dst_cache.stride(3),
        dst_cache.stride(4),
        block_size,
        head_size,
        triton.next_power_of_2(head_size // 2),
    )


def int4_dequantize_kv_cache(
    kv_cache: torch.Tensor,
    *,
    head_size: int,
    dtype: torch.dtype,
    num_blocks: int | None = None,
) -> torch.Tensor:
    cache_num_blocks, _, block_size, num_kv_heads, packed_head_size = kv_cache.shape
    if num_blocks is None:
        num_blocks = cache_num_blocks
    else:
        num_blocks = min(num_blocks, cache_num_blocks)
    expected_packed_head_size = int4_packed_head_size_bytes(head_size)
    if packed_head_size != expected_packed_head_size:
        raise ValueError(
            f"INT4 KV cache packed width mismatch: expected "
            f"{expected_packed_head_size}, got {packed_head_size}."
        )
    out = torch.empty(
        (num_blocks, 2, block_size, num_kv_heads, head_size),
        dtype=dtype,
        device=kv_cache.device,
    )
    kv_cache = kv_cache[:num_blocks]
    grid = (num_blocks * block_size, num_kv_heads, 2)
    _dequantize_cache_kernel[grid](
        kv_cache,
        out,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.stride(3),
        kv_cache.stride(4),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        out.stride(4),
        block_size,
        head_size,
        triton.next_power_of_2(head_size),
    )
    return out
