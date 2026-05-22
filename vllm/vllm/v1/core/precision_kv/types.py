# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision-aware KV state substrate types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PrecisionState(str, Enum):
    BF16 = "bf16"
    INT4 = "int4"


class MemoryTier(str, Enum):
    GPU = "gpu"
    CPU = "cpu"


class RecoveryClass(str, Enum):
    NONE = "none"
    BF16_SHADOW = "bf16_shadow"
    FP8_SHADOW = "fp8_shadow"
    RESIDUAL_CAPSULE = "residual_capsule"


class KVPageLifecycle(str, Enum):
    BF16_ACTIVE = "bf16_active"
    INT4_LANDING = "int4_landing"
    INT4_ACTIVE = "int4_active"
    INT4_ACTIVE_RECOVERABLE = "int4_active_recoverable"
    RECOVERING = "recovering"
    BF16_RECOVERED = "bf16_recovered"
    RELEASE_PENDING = "release_pending"


def encode_bf16_block_id(block_id: int) -> int:
    if block_id < 0:
        raise ValueError(f"BF16 block id must be non-negative, got {block_id}.")
    return block_id


def encode_int4_block_id(block_id: int) -> int:
    if block_id < 0:
        raise ValueError(f"INT4 block id must be non-negative, got {block_id}.")
    return -(block_id + 1)


def decode_block_table_entry(entry: int) -> tuple[PrecisionState, int]:
    if entry >= 0:
        return PrecisionState.BF16, entry
    return PrecisionState.INT4, -entry - 1


class Int4BlockPool:
    """Small free-list allocator for packed INT4 KV blocks."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}.")
        self._free_blocks = list(range(num_blocks))
        self._allocated_blocks: set[int] = set()
        self.num_blocks = num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    @property
    def num_allocated_blocks(self) -> int:
        return len(self._allocated_blocks)

    def is_allocated(self, block_id: int) -> bool:
        return block_id in self._allocated_blocks

    def allocate(self) -> int | None:
        if not self._free_blocks:
            return None
        block_id = self._free_blocks.pop(0)
        self._allocated_blocks.add(block_id)
        return block_id

    def reserve(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid INT4 block id {block_id}.")
        if block_id in self._allocated_blocks:
            raise ValueError(f"INT4 block {block_id} is already allocated.")
        try:
            self._free_blocks.remove(block_id)
        except ValueError as exc:
            raise ValueError(f"INT4 block {block_id} is not free.") from exc
        self._allocated_blocks.add(block_id)

    def free(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_blocks:
            raise ValueError(f"Invalid INT4 block id {block_id}.")
        if block_id not in self._allocated_blocks:
            raise ValueError(f"INT4 block {block_id} is not allocated.")
        self._allocated_blocks.remove(block_id)
        self._free_blocks.insert(0, block_id)


@dataclass(frozen=True)
class ReflexPageMeta:
    request_id: str
    page_idx: int
    precision: PrecisionState
    bf16_block_id: int | None
    int4_block_id: int | None
    is_full: bool = True
    is_shared: bool = False
    compressible: bool | None = None
    prefill_risk: float | None = None
    is_prompt_page: bool = False
    is_prompt_protected: bool = False
    is_page_protected: bool = False
    copy_on_demote: bool = False
    is_remote_inflight: bool = False
    is_request_protected: bool = False


@dataclass(frozen=True)
class KVPageRuntimeDescriptor:
    request_id: str
    page_idx: int
    precision: PrecisionState
    tier: MemoryTier
    lifecycle: KVPageLifecycle
    physical_block_id: int
    bf16_block_id: int | None = None
    int4_block_id: int | None = None
    planned_precision: PrecisionState | None = None
    landing_int4_block_id: int | None = None
    risk_score: float | None = None
    is_low_risk: bool = False
    is_full: bool = True
    is_shared: bool = False
    is_initial_protected: bool = False
    is_recent_protected: bool = False
    bf16_release_pending: bool = False
    quality_debt: float = 0.0
    recovery_class: RecoveryClass = RecoveryClass.NONE
    has_recovery_artifact: bool = False
    recovery_tier: MemoryTier | None = None
    recovery_source_bf16_block_id: int | None = None


@dataclass(frozen=True)
class ReflexDemotion:
    request_id: str
    page_idx: int
    bf16_block_id: int
    int4_block_id: int
    encoded_block_table_id: int
    kv_cache_group_id: int = 0
    recovery_class: RecoveryClass = RecoveryClass.NONE
    recovery_shadow_bytes: int = 0
    is_prompt_page: bool = False
    prompt_pages: int | None = None
    risk_score: float | None = None
    is_low_risk: bool = False
    copy_on_demote: bool = False


@dataclass(frozen=True)
class ReflexRecoveryArtifact:
    request_id: str
    page_idx: int
    source_bf16_block_id: int
    int4_block_id: int
    recovery_class: RecoveryClass
    kv_cache_group_id: int = 0
    tier: MemoryTier = MemoryTier.CPU
    shadow_bytes: int = 0


@dataclass(frozen=True)
class ReflexRecovery:
    request_id: str
    page_idx: int
    int4_block_id: int
    bf16_block_id: int
    encoded_block_table_id: int
    recovery_class: RecoveryClass
    kv_cache_group_id: int = 0
