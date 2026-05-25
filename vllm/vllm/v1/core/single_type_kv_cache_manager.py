# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHashList,
    BlockHashWithGroupId,
    KVCacheBlock,
)
from vllm.v1.core.precision_kv.demotion_planner import (
    DistanceDemotionPlanner,
    ReflexCandidateBreakdown,
    ReflexDemotionPlan,
    RequestPrecisionBudget,
)
from vllm.v1.core.precision_kv.run_optimizer import DualPriceState
from vllm.v1.core.precision_kv.types import (
    Int4BlockPool,
    KVPageLifecycle,
    KVPageRuntimeDescriptor,
    MemoryTier,
    PrecisionState,
    RecoveryClass,
    ReflexDemotion,
    ReflexPageMeta,
    ReflexRecovery,
    ReflexRecoveryArtifact,
    decode_block_table_entry,
    encode_int4_block_id,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CrossAttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SinkFullAttentionSpec,
    SlidingWindowSpec,
)
from vllm.v1.request import Request

logger = init_logger(__name__)


class SingleTypeKVCacheManager(ABC):
    """
    An abstract base class for a manager that handle the kv cache management
    logic of one specific type of attention layer.
    """

    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        reflex_int4_num_blocks: int | None = None,
    ) -> None:
        """
        Initializes the SingleTypeKVCacheManager.
        Args:
            kv_cache_spec: The kv_cache_spec for this manager.
            block_pool: The block pool.
            kv_cache_group_id: The id of the kv cache group of this manager.
        """
        self.block_size = kv_cache_spec.block_size
        self.dcp_world_size = dcp_world_size
        self.pcp_world_size = pcp_world_size
        if dcp_world_size * pcp_world_size > 1:
            self.block_size *= dcp_world_size * pcp_world_size
        self.kv_cache_spec = kv_cache_spec
        self.block_pool = block_pool
        self.enable_caching = enable_caching
        self.new_block_ids: list[int] = []

        # Mapping from request ID to blocks to track the blocks allocated
        # for each request, so that we can free the blocks when the request
        # is finished.
        self.req_to_blocks: defaultdict[str, list[KVCacheBlock]] = defaultdict(list)
        int4_num_blocks = (
            block_pool.num_gpu_blocks
            if reflex_int4_num_blocks is None
            else reflex_int4_num_blocks
        )
        self.reflex_int4_pool: Int4BlockPool | None = (
            Int4BlockPool(int4_num_blocks)
            if getattr(kv_cache_spec, "cache_dtype_str", None) == "reflex_int4"
            else None
        )
        self.req_to_reflex_block_ids: defaultdict[str, list[int]] = defaultdict(list)
        self.req_to_reflex_landing_int4_ids: defaultdict[str, list[int]] = defaultdict(
            list
        )
        self.req_to_reflex_landing_page_indices: defaultdict[str, list[int]] = (
            defaultdict(list)
        )
        self.req_to_reflex_direct_landing_page_indices: defaultdict[str, set[int]] = (
            defaultdict(set)
        )
        self.req_to_reflex_recovery_artifacts: defaultdict[
            str, dict[int, ReflexRecoveryArtifact]
        ] = defaultdict(dict)
        self._pending_reflex_int4_demotions: list[ReflexDemotion] = []
        self._pending_reflex_int4_recoveries: list[ReflexRecovery] = []
        self._pending_reflex_bf16_releases: list[KVCacheBlock] = []
        self._pending_reflex_bf16_release_count_by_request: defaultdict[str, int] = (
            defaultdict(int)
        )
        self._pending_reflex_bf16_release_pages_by_request: defaultdict[
            str, set[int]
        ] = defaultdict(set)
        self._reflex_recovered_pages_by_request: defaultdict[str, set[int]] = (
            defaultdict(set)
        )
        self._last_reflex_int4_candidate_capacity = 0
        self._last_reflex_int4_candidate_breakdown = ReflexCandidateBreakdown()
        self._reflex_int4_cached_frontier_key: tuple | None = None
        self._reflex_int4_cached_frontier_reuse_key: tuple | None = None
        self._reflex_int4_cached_frontier_pages: tuple[ReflexPageMeta, ...] = ()
        self._reflex_int4_cached_frontier_candidate_capacity = 0
        self._reflex_int4_cached_frontier_breakdown = ReflexCandidateBreakdown()

        # {req_id: The number of cached blocks for this given request}
        # This is used to track the number of cached blocks for each request.
        # This is only used to track the RUNNING requests, we do not track the
        # data for preempted ones.
        self.num_cached_block: dict[str, int] = {}

        self.kv_cache_group_id = kv_cache_group_id
        self._null_block = block_pool.null_block

    @classmethod
    def _get_num_evictable_blocks(cls, blocks: Sequence[KVCacheBlock]):
        return sum(blk.ref_cnt == 0 and not blk.is_null for blk in blocks)

    def _reflex_direct_landing_int4_block_id(
        self,
        request_id: str,
        page_idx: int,
    ) -> int | None:
        if self.reflex_int4_pool is None:
            return None
        if page_idx not in self.req_to_reflex_direct_landing_page_indices.get(
            request_id,
            set(),
        ):
            return None
        landing_pages = self.req_to_reflex_landing_page_indices.get(
            request_id,
            [],
        )
        landing_ids = self.req_to_reflex_landing_int4_ids.get(request_id, [])
        for idx, landing_page_idx in enumerate(landing_pages):
            if int(landing_page_idx) != page_idx or idx >= len(landing_ids):
                continue
            int4_block_id = int(landing_ids[idx])
            if self.reflex_int4_pool.is_allocated(int4_block_id):
                return int4_block_id
        return None

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        """
        Get the number of blocks needed to be allocated for the request.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            new_computed_blocks: The new computed blocks just hitting the
                prefix caching.
            total_computed_tokens: Include both local and external computed
                tokens.
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.

        Returns:
            The number of blocks to allocate.
        """

        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_req_blocks = len(self.req_to_blocks.get(request_id, ()))

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            assert len(new_computed_blocks) == 0
            # NOTE: With speculative decoding, request's blocks may be allocated
            # for draft tokens which are later rejected. In this case,
            # num_required_blocks may be smaller than num_req_blocks.
            num_new_blocks = max(num_required_blocks - num_req_blocks, 0)
            if self.reflex_int4_pool is not None and num_new_blocks > 0:
                num_direct_landing_pages = sum(
                    1
                    for page_idx in range(num_req_blocks, num_required_blocks)
                    if self._reflex_direct_landing_int4_block_id(
                        request_id,
                        page_idx,
                    )
                    is not None
                )
                num_new_blocks = max(
                    0,
                    num_new_blocks - num_direct_landing_pages,
                )
            return num_new_blocks

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        num_local_computed_blocks = len(new_computed_blocks) + num_req_blocks
        # Number of whole blocks that are skipped by the attention window.
        # If nothing is skipped, this is 0.
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # We need blocks for the non-skipped suffix. If there are still
        # local-computed blocks inside the window, they contribute to the
        # required capacity; otherwise, skipped blocks dominate.
        num_new_blocks = max(
            num_required_blocks - max(num_skipped_blocks, num_local_computed_blocks),
            0,
        )
        if self.reflex_int4_pool is not None and num_new_blocks > 0:
            first_new_page = num_required_blocks - num_new_blocks
            num_direct_landing_pages = sum(
                1
                for page_idx in range(first_new_page, num_required_blocks)
                if self._reflex_direct_landing_int4_block_id(
                    request_id,
                    page_idx,
                )
                is not None
            )
            num_new_blocks = max(0, num_new_blocks - num_direct_landing_pages)
        # Among the `new_computed_blocks`, the first `num_skipped_blocks` worth
        # of blocks are skipped; `num_req_blocks` of those may already be in
        # `req_to_blocks`, so only skip the remainder from `new_computed_blocks`.
        num_skipped_new_computed_blocks = max(0, num_skipped_blocks - num_req_blocks)

        # If a computed block is an eviction candidate (in the free queue and
        # ref_cnt == 0), it will be removed from the free queue when touched by
        # the allocated request, so we must count it in the free-capacity check.
        num_evictable_blocks = self._get_num_evictable_blocks(
            new_computed_blocks[num_skipped_new_computed_blocks:]
        )
        return num_new_blocks + num_evictable_blocks

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        """
        Add the new computed blocks to the request. This involves three steps:
        1. Touch the computed blocks to make sure they won't be evicted.
        1.5. (Optional) For sliding window, skip blocks are padded with null blocks.
        2. Add the remaining computed blocks.
        3. (Optional) For KV connectors, allocate new blocks for external computed
            tokens (if any).

        Args:
            request_id: The request ID.
            new_computed_blocks: The new computed blocks just hitting the
                prefix cache.
            num_local_computed_tokens: The number of local computed tokens.
            num_external_computed_tokens: The number of external computed tokens.
        """

        if request_id in self.num_cached_block:
            # Fast-path: a running request won't have any new prefix-cache hits.
            # It should not have any new computed blocks.
            assert len(new_computed_blocks) == 0
            return

        # A new request.
        req_blocks = self.req_to_blocks[request_id]
        assert len(req_blocks) == 0
        num_total_computed_tokens = (
            num_local_computed_tokens + num_external_computed_tokens
        )
        num_skipped_tokens = self.get_num_skipped_tokens(num_total_computed_tokens)
        num_skipped_blocks = num_skipped_tokens // self.block_size
        if num_skipped_blocks > 0:
            # It is possible that all new computed blocks are skipped when
            # num_skipped_blocks > len(new_computed_blocks).
            new_computed_blocks = new_computed_blocks[num_skipped_blocks:]
            # Some external computed tokens may be skipped too.
            num_external_computed_tokens = min(
                num_total_computed_tokens - num_skipped_tokens,
                num_external_computed_tokens,
            )

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_blocks)
        else:
            assert not any(new_computed_blocks), (
                "Computed blocks should be empty when prefix caching is disabled"
            )

        # Skip blocks are padded with null blocks.
        req_blocks.extend([self._null_block] * num_skipped_blocks)
        # Add the remaining computed blocks.
        req_blocks.extend(new_computed_blocks)
        if self.reflex_int4_pool is not None:
            self._ensure_reflex_block_ids(request_id)
        # All cached hits (including skipped nulls) are already cached; mark
        # them so cache_blocks() will not try to re-cache blocks that already
        # have a block_hash set.
        self.num_cached_block[request_id] = len(req_blocks)

        if num_external_computed_tokens > 0:
            target_num_blocks = cdiv(num_total_computed_tokens, self.block_size)
            new_page_indices = list(range(len(req_blocks), target_num_blocks))
            direct_landing_by_page = {
                page_idx: int4_block_id
                for page_idx in new_page_indices
                if (
                    int4_block_id := self._reflex_direct_landing_int4_block_id(
                        request_id,
                        page_idx,
                    )
                )
                is not None
            }
            allocated_blocks = self.block_pool.get_new_blocks(
                len(new_page_indices) - len(direct_landing_by_page)
            )
            allocated_iter = iter(allocated_blocks)
            for page_idx in new_page_indices:
                int4_block_id = direct_landing_by_page.get(page_idx)
                if int4_block_id is not None:
                    req_blocks.append(self._null_block)
                    self.req_to_reflex_block_ids[request_id].append(
                        encode_int4_block_id(int4_block_id)
                    )
                else:
                    block = next(allocated_iter)
                    req_blocks.append(block)
                    if self.reflex_int4_pool is not None:
                        self.req_to_reflex_block_ids[request_id].append(block.block_id)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in allocated_blocks)

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        """
        Allocate new blocks for the request to give it at least `num_tokens`
        token slots.

        Args:
            request_id: The request ID.
            num_tokens: The total number of tokens that need a slot (including
                tokens that are already allocated).
            num_tokens_main_model: The number of tokens for the main model (aka target
                model in spec decode). w/o spec decode, it is num_tokens;
                with spec decode, it is num_tokens - num_lookahead_tokens.
        Returns:
            The new allocated blocks.
        """
        req_blocks = self.req_to_blocks[request_id]
        num_required_blocks = cdiv(num_tokens, self.block_size)
        num_new_blocks = num_required_blocks - len(req_blocks)
        if num_new_blocks <= 0:
            return []
        else:
            if self.reflex_int4_pool is not None:
                self._ensure_reflex_block_ids(request_id)
            new_page_indices = list(
                range(len(req_blocks), len(req_blocks) + num_new_blocks)
            )
            direct_landing_by_page = {
                page_idx: int4_block_id
                for page_idx in new_page_indices
                if (
                    int4_block_id := self._reflex_direct_landing_int4_block_id(
                        request_id,
                        page_idx,
                    )
                )
                is not None
            }
            new_blocks = self.block_pool.get_new_blocks(
                num_new_blocks - len(direct_landing_by_page)
            )
            new_block_iter = iter(new_blocks)
            for page_idx in new_page_indices:
                int4_block_id = direct_landing_by_page.get(page_idx)
                if int4_block_id is not None:
                    req_blocks.append(self._null_block)
                    self.req_to_reflex_block_ids[request_id].append(
                        encode_int4_block_id(int4_block_id)
                    )
                    continue
                block = next(new_block_iter)
                req_blocks.append(block)
                if self.reflex_int4_pool is not None:
                    self.req_to_reflex_block_ids[request_id].append(block.block_id)
            if type(self.kv_cache_spec) is FullAttentionSpec:
                self.new_block_ids.extend(b.block_id for b in new_blocks)
            return new_blocks

    def take_new_block_ids(self) -> list[int]:
        """Drain and return block IDs allocated since the last call."""
        ids = self.new_block_ids
        self.new_block_ids = []
        return ids

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        """
        Cache the blocks for the request.

        Args:
            request: The request.
            num_tokens: The total number of tokens that need to be cached
                (including tokens that are already cached).
        """
        num_cached_blocks = self.num_cached_block.get(request.request_id, 0)
        num_full_blocks = num_tokens // self.block_size

        if num_cached_blocks >= num_full_blocks:
            return

        self.block_pool.cache_full_blocks(
            request=request,
            blocks=self.req_to_blocks[request.request_id],
            num_cached_blocks=num_cached_blocks,
            num_full_blocks=num_full_blocks,
            block_size=self.block_size,
            kv_cache_group_id=self.kv_cache_group_id,
        )

        self.num_cached_block[request.request_id] = num_full_blocks

    def free(self, request_id: str) -> None:
        """
        Free the blocks for the request.

        Args:
            request_id: The request ID.
        """
        # Default to [] in case a request is freed (aborted) before alloc.
        req_blocks = self.req_to_blocks.pop(request_id, [])
        self._free_reflex_int4_blocks(request_id)
        self._free_reflex_int4_landing_blocks(request_id)

        # Free blocks in reverse order so that the tail blocks are
        # freed first.
        ordered_blocks = [
            block for block in reversed(req_blocks) if block != self._null_block
        ]

        self.block_pool.free_blocks(ordered_blocks)
        self.num_cached_block.pop(request_id, None)
        self.req_to_reflex_block_ids.pop(request_id, None)
        self.req_to_reflex_landing_int4_ids.pop(request_id, None)
        self.req_to_reflex_landing_page_indices.pop(request_id, None)
        self.req_to_reflex_direct_landing_page_indices.pop(request_id, None)
        self.req_to_reflex_recovery_artifacts.pop(request_id, None)
        self._pending_reflex_int4_recoveries = [
            recovery
            for recovery in self._pending_reflex_int4_recoveries
            if recovery.request_id != request_id
        ]
        self._reflex_recovered_pages_by_request.pop(request_id, None)
        self._pending_reflex_bf16_release_count_by_request.pop(request_id, None)
        self._pending_reflex_bf16_release_pages_by_request.pop(request_id, None)

    def get_reflex_block_ids(self, request_id: str) -> tuple[int, ...]:
        if self.reflex_int4_pool is None:
            return tuple(block.block_id for block in self.req_to_blocks[request_id])
        self._ensure_reflex_block_ids(request_id)
        return tuple(self.req_to_reflex_block_ids[request_id])

    def has_reflex_int4_blocks(self, request_id: str) -> bool:
        if self.reflex_int4_pool is None or request_id not in self.req_to_blocks:
            return False
        self._ensure_reflex_block_ids(request_id)
        return any(
            encoded_id < 0 for encoded_id in self.req_to_reflex_block_ids[request_id]
        )

    def reserve_reflex_int4_landing_blocks(
        self,
        request_id: str,
        count: int,
        page_indices: Sequence[int] | None = None,
    ) -> list[int]:
        if self.reflex_int4_pool is None or count <= 0:
            return []

        current_ids = self.req_to_reflex_landing_int4_ids.get(request_id, [])
        if len(current_ids) == count and all(
            self.reflex_int4_pool.is_allocated(block_id) for block_id in current_ids
        ):
            if page_indices is not None:
                self.record_reflex_int4_landing_pages(request_id, page_indices)
            return list(current_ids)

        self._free_reflex_int4_landing_blocks(request_id)
        if self.reflex_int4_pool.num_free_blocks < count:
            return []

        reserved_ids: list[int] = []
        for _ in range(count):
            block_id = self.reflex_int4_pool.allocate()
            assert block_id is not None
            reserved_ids.append(block_id)
        self.req_to_reflex_landing_int4_ids[request_id] = reserved_ids
        if page_indices is not None:
            self.record_reflex_int4_landing_pages(request_id, page_indices)
        return list(reserved_ids)

    def record_reflex_int4_landing_pages(
        self,
        request_id: str,
        page_indices: Sequence[int],
    ) -> None:
        if self.reflex_int4_pool is None:
            return
        landing_ids = self.req_to_reflex_landing_int4_ids.get(request_id, [])
        if landing_ids and len(page_indices) != len(landing_ids):
            raise ValueError(
                "ReFlexKV landing page contract must match reserved INT4 "
                f"blocks for {request_id}: {len(page_indices)} pages, "
                f"{len(landing_ids)} block ids."
            )
        if len(set(page_indices)) != len(page_indices):
            raise ValueError(
                f"ReFlexKV landing page contract has duplicate pages for {request_id}."
            )
        self.req_to_reflex_landing_page_indices[request_id] = [
            int(page_idx) for page_idx in page_indices
        ]

    def mark_reflex_int4_direct_landing_pages(
        self,
        request_id: str,
        page_indices: Sequence[int],
        int4_block_ids: Sequence[int] | None = None,
    ) -> None:
        if self.reflex_int4_pool is None:
            return
        if int4_block_ids is not None and len(page_indices) != len(int4_block_ids):
            raise ValueError(
                "ReFlexKV direct landing requires one INT4 block id per page: "
                f"{len(page_indices)} pages, {len(int4_block_ids)} block ids."
            )
        if int4_block_ids is not None:
            reserved = self.req_to_reflex_landing_int4_ids.get(request_id, [])
            for block_id in int4_block_ids:
                if int(block_id) not in reserved:
                    raise RuntimeError(
                        "Cannot mark ReFlexKV direct landing because INT4 block "
                        f"{block_id} is not reserved for request {request_id}."
                    )
        self.record_reflex_int4_landing_pages(request_id, page_indices)
        self.req_to_reflex_direct_landing_page_indices[request_id] = {
            int(page_idx) for page_idx in page_indices
        }

    def release_reflex_int4_landing_blocks(self, request_id: str) -> None:
        self._free_reflex_int4_landing_blocks(request_id)

    def commit_reflex_int4_landing_pages(
        self,
        request_id: str,
        page_indices: Sequence[int],
        int4_block_ids: Sequence[int],
    ) -> int:
        if self.reflex_int4_pool is None:
            return 0
        if request_id not in self.req_to_blocks:
            return 0
        if len(page_indices) != len(int4_block_ids):
            raise ValueError(
                "ReFlexKV landing commit requires one INT4 block id per page: "
                f"{len(page_indices)} pages, {len(int4_block_ids)} block ids."
            )
        if len(set(page_indices)) != len(page_indices):
            raise ValueError(
                f"ReFlexKV landing commit has duplicate pages for {request_id}."
            )
        if len(set(int4_block_ids)) != len(int4_block_ids):
            raise ValueError(
                "ReFlexKV landing commit has duplicate INT4 block ids for "
                f"{request_id}."
            )

        self._ensure_reflex_block_ids(request_id)
        blocks = self.req_to_blocks[request_id]
        encoded_ids = self.req_to_reflex_block_ids[request_id]
        landing_ids = self.req_to_reflex_landing_int4_ids[request_id]
        committed = 0

        for page_idx, int4_block_id in zip(page_indices, int4_block_ids):
            if page_idx < 0 or page_idx >= len(blocks):
                raise RuntimeError(
                    "Cannot commit stale ReFlexKV landing page: "
                    f"request={request_id}, page={page_idx}, "
                    f"allocated_pages={len(blocks)}."
                )
            precision, physical_id = decode_block_table_entry(encoded_ids[page_idx])
            if precision == PrecisionState.INT4:
                if physical_id != int4_block_id:
                    raise RuntimeError(
                        "Cannot overwrite ReFlexKV INT4 page with a different "
                        f"landing block: request={request_id}, page={page_idx}, "
                        f"current={physical_id}, new={int4_block_id}."
                    )
                if int4_block_id in landing_ids:
                    landing_ids.remove(int4_block_id)
                landing_pages = self.req_to_reflex_landing_page_indices.get(
                    request_id,
                    [],
                )
                if page_idx in landing_pages:
                    landing_pages.remove(page_idx)
                self.req_to_reflex_direct_landing_page_indices.get(
                    request_id,
                    set(),
                ).discard(page_idx)
                committed += 1
                continue
            if int4_block_id not in landing_ids:
                raise RuntimeError(
                    "Cannot commit ReFlexKV landing page because INT4 block "
                    f"{int4_block_id} is not reserved for request {request_id}."
                )
            if not self.reflex_int4_pool.is_allocated(int4_block_id):
                raise RuntimeError(
                    "Cannot commit ReFlexKV landing page because INT4 block "
                    f"{int4_block_id} is not allocated."
                )

            block = blocks[page_idx]
            if block != self._null_block and block.block_id != physical_id:
                raise RuntimeError(
                    "Cannot commit stale ReFlexKV landing page: "
                    f"request={request_id}, page={page_idx}, "
                    f"expected BF16 block={physical_id}."
                )

            if block != self._null_block:
                self._pending_reflex_bf16_releases.append(block)
                self._pending_reflex_bf16_release_count_by_request[request_id] += 1
                self._pending_reflex_bf16_release_pages_by_request[request_id].add(
                    page_idx
                )
                blocks[page_idx] = self._null_block
            encoded_ids[page_idx] = encode_int4_block_id(int4_block_id)
            landing_ids.remove(int4_block_id)
            landing_pages = self.req_to_reflex_landing_page_indices.get(
                request_id,
                [],
            )
            if page_idx in landing_pages:
                landing_pages.remove(page_idx)
            self.req_to_reflex_direct_landing_page_indices.get(
                request_id,
                set(),
            ).discard(page_idx)
            committed += 1

        return committed

    def recover_reflex_int4_pages(
        self,
        request_id: str,
        page_indices: Sequence[int],
        *,
        target_precision: PrecisionState = PrecisionState.BF16,
    ) -> int:
        if self.reflex_int4_pool is None:
            return 0
        if target_precision != PrecisionState.BF16:
            raise ValueError("ReFlexKV phase-1 recovery only supports BF16 promotion.")
        if request_id not in self.req_to_blocks:
            return 0

        self._ensure_reflex_block_ids(request_id)
        blocks = self.req_to_blocks[request_id]
        encoded_ids = self.req_to_reflex_block_ids[request_id]
        artifacts = self.req_to_reflex_recovery_artifacts.get(request_id, {})
        pending_pages = self._pending_reflex_bf16_release_pages_by_request.get(
            request_id,
            set(),
        )

        eligible_pages: list[tuple[int, int, ReflexRecoveryArtifact]] = []
        seen: set[int] = set()
        for raw_page_idx in page_indices:
            page_idx = int(raw_page_idx)
            if page_idx in seen:
                continue
            seen.add(page_idx)
            if page_idx < 0 or page_idx >= len(encoded_ids):
                continue
            if page_idx in pending_pages:
                continue
            precision, physical_id = decode_block_table_entry(encoded_ids[page_idx])
            if precision != PrecisionState.INT4:
                continue
            artifact = artifacts.get(page_idx)
            if artifact is None or artifact.recovery_class == RecoveryClass.NONE:
                continue
            if artifact.int4_block_id != physical_id:
                raise RuntimeError(
                    "Cannot recover ReFlexKV page because its recovery "
                    "artifact points at a stale INT4 block: "
                    f"request={request_id}, page={page_idx}, "
                    f"artifact_int4={artifact.int4_block_id}, "
                    f"current_int4={physical_id}."
                )
            eligible_pages.append((page_idx, physical_id, artifact))

        if not eligible_pages:
            return 0
        if self.block_pool.get_num_free_blocks() < len(eligible_pages):
            return 0

        recovered_blocks = self.block_pool.get_new_blocks(len(eligible_pages))
        recovered_pages = self._reflex_recovered_pages_by_request[request_id]
        for (page_idx, int4_block_id, artifact), block in zip(
            eligible_pages,
            recovered_blocks,
        ):
            blocks[page_idx] = block
            encoded_ids[page_idx] = block.block_id
            artifacts.pop(page_idx, None)
            recovered_pages.add(page_idx)
            if self.reflex_int4_pool.is_allocated(int4_block_id):
                self.reflex_int4_pool.free(int4_block_id)
            self._pending_reflex_int4_recoveries.append(
                ReflexRecovery(
                    request_id=request_id,
                    page_idx=page_idx,
                    int4_block_id=int4_block_id,
                    bf16_block_id=block.block_id,
                    encoded_block_table_id=block.block_id,
                    recovery_class=artifact.recovery_class,
                    kv_cache_group_id=self.kv_cache_group_id,
                )
            )
        if not artifacts:
            self.req_to_reflex_recovery_artifacts.pop(request_id, None)
        return len(recovered_blocks)

    def promote_reflex_recoverable_pages(
        self,
        *,
        max_pages: int,
        prefill_page_risks_by_request: dict[str, Sequence[float]] | None = None,
        remaining_decode_tokens_by_request: dict[str, int] | None = None,
        min_remaining_decode_tokens: int = 16,
    ) -> int:
        if self.reflex_int4_pool is None or max_pages <= 0:
            return 0
        limit = min(max_pages, self.block_pool.get_num_free_blocks())
        if limit <= 0:
            return 0
        prefill_page_risks_by_request = prefill_page_risks_by_request or {}
        remaining_decode_tokens_by_request = remaining_decode_tokens_by_request or {}
        min_remaining_decode_tokens = max(0, int(min_remaining_decode_tokens))

        candidates: list[tuple[float, int, str, int]] = []
        for request_id, artifacts in self.req_to_reflex_recovery_artifacts.items():
            if request_id not in self.req_to_blocks:
                continue
            remaining_decode = remaining_decode_tokens_by_request.get(
                request_id,
                min_remaining_decode_tokens + 1,
            )
            if remaining_decode < min_remaining_decode_tokens:
                continue
            page_risks = prefill_page_risks_by_request.get(request_id, ())
            for page_idx, artifact in artifacts.items():
                if artifact.recovery_class == RecoveryClass.NONE:
                    continue
                if not self._is_reflex_recoverable_page(request_id, page_idx):
                    continue
                risk = (
                    0.5 if page_idx >= len(page_risks) else float(page_risks[page_idx])
                )
                candidates.append((-risk, -remaining_decode, request_id, page_idx))

        if not candidates:
            return 0
        candidates.sort()
        pages_by_request: defaultdict[str, list[int]] = defaultdict(list)
        for _neg_risk, _neg_remaining, request_id, page_idx in candidates[:limit]:
            pages_by_request[request_id].append(page_idx)

        recovered = 0
        for request_id, page_indices in pages_by_request.items():
            recovered += self.recover_reflex_int4_pages(
                request_id,
                page_indices,
            )
        return recovered

    def _is_reflex_shared_bf16_block(self, block: KVCacheBlock) -> bool:
        # A block hash means the page is cacheable, not necessarily shared.
        # ReFlexKV evicts the hash before demotion; live sharing is ref_cnt > 1.
        return block == self._null_block or block.ref_cnt > 1

    def get_reflex_page_runtime_descriptors(
        self,
        request_id: str,
        *,
        computed_tokens_by_request: dict[str, int] | None = None,
        prompt_tokens_by_request: dict[str, int] | None = None,
        sealed_pages_by_request: dict[str, int] | None = None,
        prefill_page_risks_by_request: dict[str, Sequence[float]] | None = None,
        compressible_pages_by_request: dict[str, set[int]] | None = None,
        protected_request_ids: set[str] | None = None,
        allow_partial_prefill_demotion_request_ids: set[str] | None = None,
        keep_recent_pages: int = 0,
        keep_initial_pages: int = 0,
    ) -> list[KVPageRuntimeDescriptor]:
        if self.reflex_int4_pool is None or request_id not in self.req_to_blocks:
            return []
        computed_tokens_by_request = computed_tokens_by_request or {}
        prompt_tokens_by_request = prompt_tokens_by_request or {}
        sealed_pages_by_request = sealed_pages_by_request or {}
        prefill_page_risks_by_request = prefill_page_risks_by_request or {}
        compressible_pages_by_request = compressible_pages_by_request or {}
        protected_request_ids = protected_request_ids or set()
        allow_partial_prefill_demotion_request_ids = (
            allow_partial_prefill_demotion_request_ids or set()
        )

        self._ensure_reflex_block_ids(request_id)
        blocks = self.req_to_blocks[request_id]
        encoded_ids = self.req_to_reflex_block_ids[request_id]
        computed_tokens = computed_tokens_by_request.get(request_id, 0)
        prompt_tokens = prompt_tokens_by_request.get(request_id)
        explicit_sealed_pages = max(
            0,
            int(sealed_pages_by_request.get(request_id, 0) or 0),
        )
        page_risks = prefill_page_risks_by_request.get(request_id)
        compressible_pages = compressible_pages_by_request.get(request_id, set())
        prompt_pages = (
            None if prompt_tokens is None else cdiv(prompt_tokens, self.block_size)
        )
        allow_partial_prefill_demotion = (
            request_id in allow_partial_prefill_demotion_request_ids
        )
        if request_id in protected_request_ids or (
            prompt_tokens is not None
            and computed_tokens < prompt_tokens
            and not allow_partial_prefill_demotion
            and explicit_sealed_pages <= 0
        ):
            num_sealed_pages = 0
        else:
            # A completed prefill chunk is already a closed KV prefix even
            # before the whole prompt is loaded. Only the partial tail page is
            # kept open by the floor division below. This is only allowed for
            # requests that explicitly opt into chunk-level prefill demotion.
            num_sealed_pages = max(
                computed_tokens // self.block_size,
                explicit_sealed_pages,
            )
        num_sealed_pages = min(len(blocks), num_sealed_pages)

        landing_pages = self.req_to_reflex_landing_page_indices.get(request_id, [])
        landing_ids = self.req_to_reflex_landing_int4_ids.get(request_id, [])
        landing_by_page = {
            page_idx: landing_ids[idx]
            for idx, page_idx in enumerate(landing_pages)
            if idx < len(landing_ids)
        }
        recovery_artifacts = self.req_to_reflex_recovery_artifacts.get(
            request_id,
            {},
        )
        recovered_pages = self._reflex_recovered_pages_by_request.get(
            request_id,
            set(),
        )
        pending_pages = self._pending_reflex_bf16_release_pages_by_request.get(
            request_id,
            set(),
        )
        protect_from = max(0, len(blocks) - keep_recent_pages)
        descriptors: list[KVPageRuntimeDescriptor] = []

        for page_idx, encoded_id in enumerate(encoded_ids):
            precision, physical_id = decode_block_table_entry(encoded_id)
            risk_score = (
                None
                if page_risks is None or page_idx >= len(page_risks)
                else float(page_risks[page_idx])
            )
            is_full = page_idx < num_sealed_pages
            is_shared = self._is_reflex_shared_bf16_block(blocks[page_idx])
            is_initial_protected = page_idx < keep_initial_pages
            is_recent_protected = keep_recent_pages > 0 and page_idx >= protect_from
            if precision == PrecisionState.INT4:
                release_pending = page_idx in pending_pages
                artifact = recovery_artifacts.get(page_idx)
                if page_idx in landing_by_page:
                    lifecycle = KVPageLifecycle.INT4_LANDING
                elif release_pending:
                    lifecycle = KVPageLifecycle.RELEASE_PENDING
                elif artifact is not None:
                    lifecycle = KVPageLifecycle.INT4_ACTIVE_RECOVERABLE
                else:
                    lifecycle = KVPageLifecycle.INT4_ACTIVE
                descriptors.append(
                    KVPageRuntimeDescriptor(
                        request_id=request_id,
                        page_idx=page_idx,
                        precision=PrecisionState.INT4,
                        tier=MemoryTier.GPU,
                        lifecycle=lifecycle,
                        physical_block_id=physical_id,
                        int4_block_id=physical_id,
                        planned_precision=PrecisionState.INT4,
                        risk_score=risk_score,
                        is_low_risk=page_idx in compressible_pages,
                        is_full=is_full,
                        is_shared=is_shared,
                        is_initial_protected=is_initial_protected,
                        is_recent_protected=is_recent_protected,
                        bf16_release_pending=release_pending,
                        quality_debt=1.0,
                        recovery_class=(
                            artifact.recovery_class
                            if artifact is not None
                            else RecoveryClass.NONE
                        ),
                        has_recovery_artifact=artifact is not None,
                        recovery_tier=(artifact.tier if artifact is not None else None),
                        recovery_source_bf16_block_id=(
                            artifact.source_bf16_block_id
                            if artifact is not None
                            else None
                        ),
                    )
                )
                continue

            landing_int4_block_id = landing_by_page.get(page_idx)
            if landing_int4_block_id is not None:
                lifecycle = KVPageLifecycle.INT4_LANDING
            elif page_idx in recovered_pages:
                lifecycle = KVPageLifecycle.BF16_RECOVERED
            else:
                lifecycle = KVPageLifecycle.BF16_ACTIVE
            descriptors.append(
                KVPageRuntimeDescriptor(
                    request_id=request_id,
                    page_idx=page_idx,
                    precision=PrecisionState.BF16,
                    tier=MemoryTier.GPU,
                    lifecycle=lifecycle,
                    physical_block_id=physical_id,
                    bf16_block_id=physical_id,
                    planned_precision=(
                        PrecisionState.INT4
                        if landing_int4_block_id is not None
                        else PrecisionState.BF16
                    ),
                    landing_int4_block_id=landing_int4_block_id,
                    risk_score=risk_score,
                    is_low_risk=page_idx in compressible_pages,
                    is_full=is_full,
                    is_shared=is_shared,
                    is_initial_protected=is_initial_protected,
                    is_recent_protected=is_recent_protected,
                )
            )
        return descriptors

    def get_reflex_precision_state_counts(
        self,
        request_id: str | None = None,
    ) -> dict[str, int]:
        counts = {
            "BF16_ACTIVE": 0,
            "INT4_ACTIVE": 0,
            "INT4_RECOVERABLE": 0,
            "BF16_RECOVERED": 0,
            "RELEASE_PENDING": 0,
            "LANDING_RESERVED": 0,
        }
        if self.reflex_int4_pool is None:
            return counts

        request_ids = (
            [request_id] if request_id is not None else list(self.req_to_blocks.keys())
        )
        for req_id in request_ids:
            if req_id not in self.req_to_blocks:
                continue
            self._ensure_reflex_block_ids(req_id)
            blocks = self.req_to_blocks[req_id]
            encoded_ids = self.req_to_reflex_block_ids[req_id]
            recovery_artifacts = self.req_to_reflex_recovery_artifacts.get(
                req_id,
                {},
            )
            recovered_pages = self._reflex_recovered_pages_by_request.get(
                req_id,
                set(),
            )
            for page_idx, (block, encoded_id) in enumerate(zip(blocks, encoded_ids)):
                precision, _ = decode_block_table_entry(encoded_id)
                if precision == PrecisionState.INT4:
                    counts["INT4_ACTIVE"] += 1
                    if page_idx in recovery_artifacts:
                        counts["INT4_RECOVERABLE"] += 1
                elif block != self._null_block:
                    counts["BF16_ACTIVE"] += 1
                    if page_idx in recovered_pages:
                        counts["BF16_RECOVERED"] += 1
            counts["LANDING_RESERVED"] += len(
                self.req_to_reflex_landing_page_indices.get(
                    req_id,
                    self.req_to_reflex_landing_int4_ids.get(req_id, ()),
                )
            )
            counts["RELEASE_PENDING"] += len(
                self._pending_reflex_bf16_release_pages_by_request.get(
                    req_id,
                    set(),
                )
            )
        return counts

    def check_reflex_int4_invariants(
        self,
        request_id: str | None = None,
    ) -> list[str]:
        if self.reflex_int4_pool is None:
            return []

        request_ids = (
            [request_id] if request_id is not None else list(self.req_to_blocks.keys())
        )
        violations: list[str] = []
        for req_id in request_ids:
            if req_id not in self.req_to_blocks:
                continue
            try:
                self._ensure_reflex_block_ids(req_id)
            except RuntimeError as exc:
                violations.append(str(exc))
                continue

            blocks = self.req_to_blocks[req_id]
            encoded_ids = self.req_to_reflex_block_ids[req_id]
            landing_pages = set(self.req_to_reflex_landing_page_indices.get(req_id, ()))
            if len(blocks) != len(encoded_ids):
                violations.append(
                    "ReFlexKV invariant violation: "
                    f"request={req_id} has {len(blocks)} BF16 block slots but "
                    f"{len(encoded_ids)} encoded entries."
                )
                continue

            for page_idx, (block, encoded_id) in enumerate(zip(blocks, encoded_ids)):
                precision, physical_id = decode_block_table_entry(encoded_id)
                if precision == PrecisionState.INT4:
                    if block != self._null_block:
                        violations.append(
                            "ReFlexKV invariant violation: INT4 page still has "
                            f"a BF16 block slot: request={req_id}, page={page_idx}."
                        )
                    if not self.reflex_int4_pool.is_allocated(physical_id):
                        violations.append(
                            "ReFlexKV invariant violation: INT4 page references "
                            f"unallocated block {physical_id}: request={req_id}, "
                            f"page={page_idx}."
                        )
                    continue

                if block == self._null_block and page_idx not in landing_pages:
                    violations.append(
                        "ReFlexKV invariant violation: BF16 page has a null "
                        f"block slot: request={req_id}, page={page_idx}, "
                        f"block_id={physical_id}."
                    )
                elif block != self._null_block and block.block_id != physical_id:
                    violations.append(
                        "ReFlexKV invariant violation: BF16 block table mismatch: "
                        f"request={req_id}, page={page_idx}, "
                        f"encoded={physical_id}, actual={block.block_id}."
                    )

            landing_ids = self.req_to_reflex_landing_int4_ids.get(req_id, ())
            if len(set(landing_ids)) != len(landing_ids):
                violations.append(
                    "ReFlexKV invariant violation: duplicate landing INT4 blocks "
                    f"for request={req_id}."
                )
            for int4_block_id in landing_ids:
                if not self.reflex_int4_pool.is_allocated(int4_block_id):
                    violations.append(
                        "ReFlexKV invariant violation: landing INT4 block "
                        f"{int4_block_id} is not allocated for request={req_id}."
                    )
            recovery_artifacts = self.req_to_reflex_recovery_artifacts.get(
                req_id,
                {},
            )
            for page_idx, artifact in recovery_artifacts.items():
                if page_idx < 0 or page_idx >= len(encoded_ids):
                    violations.append(
                        "ReFlexKV invariant violation: recovery artifact points "
                        f"outside request={req_id}: page={page_idx}."
                    )
                    continue
                precision, physical_id = decode_block_table_entry(encoded_ids[page_idx])
                if precision != PrecisionState.INT4:
                    violations.append(
                        "ReFlexKV invariant violation: recovery artifact is "
                        f"attached to a non-INT4 page: request={req_id}, "
                        f"page={page_idx}."
                    )
                    continue
                if artifact.int4_block_id != physical_id:
                    violations.append(
                        "ReFlexKV invariant violation: recovery artifact INT4 "
                        f"block mismatch: request={req_id}, page={page_idx}, "
                        f"artifact={artifact.int4_block_id}, current={physical_id}."
                    )
        return violations

    @staticmethod
    def _reflex_signature_mapping(mapping) -> tuple:
        if not mapping:
            return ()
        items = []
        for key, value in sorted(mapping.items(), key=lambda item: str(item[0])):
            if isinstance(value, set | frozenset):
                normalized = tuple(sorted(value))
            elif isinstance(value, dict):
                normalized = SingleTypeKVCacheManager._reflex_signature_mapping(value)
            elif isinstance(value, (list, tuple)):
                normalized = tuple(value)
            else:
                normalized = value
            items.append((key, normalized))
        return tuple(items)

    @staticmethod
    def _reflex_page_risk_signature(mapping) -> tuple:
        if not mapping:
            return ()
        items = []
        for request_id, risks in sorted(mapping.items(), key=lambda item: str(item[0])):
            rounded = tuple(round(float(risk), 6) for risk in risks)
            items.append((request_id, rounded))
        return tuple(items)

    @staticmethod
    def _reflex_budget_signature(
        budgets: dict[str, RequestPrecisionBudget] | None,
    ) -> tuple:
        if not budgets:
            return ()
        return tuple(
            (
                request_id,
                budget.max_int4_pages,
                budget.priority,
                budget.max_int4_fraction,
                budget.release_budget_blocks,
                budget.max_demote_per_window,
                budget.max_prompt_int4_pages,
                budget.max_decode_int4_pages,
                budget.quality_debt_budget_pages,
            )
            for request_id, budget in sorted(budgets.items())
        )

    @staticmethod
    def _reflex_budget_reuse_signature(
        budgets: dict[str, RequestPrecisionBudget] | None,
    ) -> tuple:
        if not budgets:
            return ()
        return tuple(
            (
                request_id,
                budget.max_int4_pages,
                round(float(budget.max_int4_fraction), 6),
                budget.max_demote_per_window,
                budget.max_prompt_int4_pages,
                budget.max_decode_int4_pages,
                budget.quality_debt_budget_pages,
            )
            for request_id, budget in sorted(budgets.items())
        )

    def _make_reflex_int4_frontier_cache_key(
        self,
        *,
        cache_scope: str | None,
        target_bf16_blocks: int,
        keep_recent_pages: int,
        keep_initial_pages: int,
        max_int4_fraction_per_request: float,
        low_risk_only: bool,
        sparse_window_pages: int,
        max_demote_per_window: int,
        selection_policy: str,
        dual_price_state: DualPriceState | None,
        emergency_release: bool,
        request_precision_budgets: dict[str, RequestPrecisionBudget] | None,
        computed_tokens_by_request: dict[str, int] | None,
        prompt_tokens_by_request: dict[str, int] | None,
        protected_request_ids: set[str] | None,
        allow_partial_prefill_demotion_request_ids: set[str] | None,
        protected_prompt_pages_by_request: dict[str, int] | None,
        protected_pages_by_request: dict[str, set[int]] | None,
        sealed_pages_by_request: dict[str, int] | None,
        remote_inflight_pages_by_request: dict[str, set[int]] | None,
        prefill_page_risks_by_request: dict[str, Sequence[float]] | None,
        compressible_pages_by_request: dict[str, set[int]] | None,
        copy_on_demote_pages_by_request: dict[str, set[int]] | None,
    ) -> tuple:
        dual_signature = None
        if dual_price_state is not None:
            dual_signature = (
                round(float(dual_price_state.memory_price), 6),
                round(float(dual_price_state.admission_price), 6),
                round(float(dual_price_state.quality_price), 6),
                round(float(dual_price_state.migration_price), 6),
                round(float(dual_price_state.slo_price), 6),
            )
        int4_free_blocks = (
            self.reflex_int4_pool.num_free_blocks
            if self.reflex_int4_pool is not None
            else 0
        )
        return (
            cache_scope,
            target_bf16_blocks,
            keep_recent_pages,
            keep_initial_pages,
            round(float(max_int4_fraction_per_request), 6),
            bool(low_risk_only),
            sparse_window_pages,
            max_demote_per_window,
            selection_policy,
            dual_signature,
            bool(emergency_release),
            int4_free_blocks,
            self._reflex_budget_signature(request_precision_budgets),
            self._reflex_signature_mapping(computed_tokens_by_request),
            self._reflex_signature_mapping(prompt_tokens_by_request),
            tuple(sorted(protected_request_ids or ())),
            tuple(sorted(allow_partial_prefill_demotion_request_ids or ())),
            self._reflex_signature_mapping(protected_prompt_pages_by_request),
            self._reflex_signature_mapping(protected_pages_by_request),
            self._reflex_signature_mapping(sealed_pages_by_request),
            self._reflex_signature_mapping(remote_inflight_pages_by_request),
            self._reflex_page_risk_signature(prefill_page_risks_by_request),
            self._reflex_signature_mapping(compressible_pages_by_request),
            self._reflex_signature_mapping(copy_on_demote_pages_by_request),
        )

    def _make_reflex_int4_frontier_reuse_key(
        self,
        *,
        cache_scope: str | None,
        keep_recent_pages: int,
        keep_initial_pages: int,
        max_int4_fraction_per_request: float,
        low_risk_only: bool,
        sparse_window_pages: int,
        max_demote_per_window: int,
        selection_policy: str,
        emergency_release: bool,
        request_precision_budgets: dict[str, RequestPrecisionBudget] | None,
        computed_tokens_by_request: dict[str, int] | None,
        prompt_tokens_by_request: dict[str, int] | None,
        protected_request_ids: set[str] | None,
        allow_partial_prefill_demotion_request_ids: set[str] | None,
        protected_prompt_pages_by_request: dict[str, int] | None,
        protected_pages_by_request: dict[str, set[int]] | None,
        sealed_pages_by_request: dict[str, int] | None,
        remote_inflight_pages_by_request: dict[str, set[int]] | None,
        prefill_page_risks_by_request: dict[str, Sequence[float]] | None,
        compressible_pages_by_request: dict[str, set[int]] | None,
        copy_on_demote_pages_by_request: dict[str, set[int]] | None,
    ) -> tuple:
        return (
            cache_scope,
            keep_recent_pages,
            keep_initial_pages,
            round(float(max_int4_fraction_per_request), 6),
            bool(low_risk_only),
            sparse_window_pages,
            max_demote_per_window,
            selection_policy,
            bool(emergency_release),
            self._reflex_budget_reuse_signature(request_precision_budgets),
            self._reflex_signature_mapping(computed_tokens_by_request),
            self._reflex_signature_mapping(prompt_tokens_by_request),
            tuple(sorted(protected_request_ids or ())),
            tuple(sorted(allow_partial_prefill_demotion_request_ids or ())),
            self._reflex_signature_mapping(protected_prompt_pages_by_request),
            self._reflex_signature_mapping(protected_pages_by_request),
            self._reflex_signature_mapping(sealed_pages_by_request),
            self._reflex_signature_mapping(remote_inflight_pages_by_request),
            self._reflex_page_risk_signature(prefill_page_risks_by_request),
            self._reflex_signature_mapping(compressible_pages_by_request),
            self._reflex_signature_mapping(copy_on_demote_pages_by_request),
        )

    def _clear_reflex_int4_cached_frontier(self) -> None:
        self._reflex_int4_cached_frontier_key = None
        self._reflex_int4_cached_frontier_reuse_key = None
        self._reflex_int4_cached_frontier_pages = ()
        self._reflex_int4_cached_frontier_candidate_capacity = 0
        self._reflex_int4_cached_frontier_breakdown = ReflexCandidateBreakdown()

    def _store_reflex_int4_cached_frontier(
        self,
        *,
        cache_key: tuple,
        reuse_key: tuple,
        plan: ReflexDemotionPlan,
    ) -> None:
        self._reflex_int4_cached_frontier_key = cache_key
        self._reflex_int4_cached_frontier_reuse_key = reuse_key
        self._reflex_int4_cached_frontier_pages = tuple(plan.candidate_pages)
        self._reflex_int4_cached_frontier_candidate_capacity = (
            plan.candidate_bf16_blocks
        )
        self._reflex_int4_cached_frontier_breakdown = plan.candidate_breakdown

    def _cached_reflex_page_is_current(self, page: ReflexPageMeta) -> bool:
        if page.request_id not in self.req_to_blocks:
            return False
        blocks = self.req_to_blocks[page.request_id]
        if page.page_idx < 0 or page.page_idx >= len(blocks):
            return False
        self._ensure_reflex_block_ids(page.request_id)
        encoded_ids = self.req_to_reflex_block_ids[page.request_id]
        if page.page_idx >= len(encoded_ids):
            return False
        precision, physical_id = decode_block_table_entry(encoded_ids[page.page_idx])
        if precision != PrecisionState.BF16 or physical_id != page.bf16_block_id:
            return False
        block = blocks[page.page_idx]
        return block != self._null_block and block.block_id == page.bf16_block_id

    def _plan_reflex_int4_demotions_from_cached_frontier(
        self,
        *,
        cache_key: tuple,
        reuse_key: tuple,
        target_bf16_blocks: int,
        prompt_tokens_by_request: dict[str, int] | None,
    ) -> ReflexDemotionPlan | None:
        if self.reflex_int4_pool is None:
            return None
        exact_match = cache_key == self._reflex_int4_cached_frontier_key
        reuse_match = reuse_key == self._reflex_int4_cached_frontier_reuse_key
        if not exact_match and not reuse_match:
            return None
        if (
            not exact_match
            and target_bf16_blocks
            > self._reflex_int4_cached_frontier_candidate_capacity
        ):
            return None
        cached_pages = self._reflex_int4_cached_frontier_pages
        if not cached_pages:
            return ReflexDemotionPlan(
                [],
                candidate_bf16_blocks=0,
                candidate_breakdown=replace(
                    self._reflex_int4_cached_frontier_breakdown,
                    selected_actual=0,
                ),
            )

        prompt_tokens_by_request = prompt_tokens_by_request or {}
        prompt_pages_by_request = {
            request_id: cdiv(prompt_tokens, self.block_size)
            for request_id, prompt_tokens in prompt_tokens_by_request.items()
        }
        selected: list[ReflexDemotion] = []
        for page in cached_pages:
            if len(selected) >= target_bf16_blocks:
                break
            if not self._cached_reflex_page_is_current(page):
                self._clear_reflex_int4_cached_frontier()
                return None
            int4_block_id = self.reflex_int4_pool.allocate()
            if int4_block_id is None:
                break
            selected.append(
                ReflexDemotion(
                    request_id=page.request_id,
                    page_idx=page.page_idx,
                    bf16_block_id=page.bf16_block_id,
                    int4_block_id=int4_block_id,
                    encoded_block_table_id=encode_int4_block_id(int4_block_id),
                    is_prompt_page=page.is_prompt_page,
                    prompt_pages=prompt_pages_by_request.get(page.request_id),
                    risk_score=page.prefill_risk,
                    is_low_risk=page.compressible is True,
                    copy_on_demote=page.copy_on_demote,
                )
            )

        cached_breakdown = replace(
            self._reflex_int4_cached_frontier_breakdown,
            selected_actual=len(selected),
        )
        candidate_capacity = self._reflex_int4_cached_frontier_candidate_capacity
        self._clear_reflex_int4_cached_frontier()
        logger.info(
            "ReFlexKV trace frontier_commit_cache outcome=hit "
            "match=%s target_release=%d cached_pages=%d selected_actual=%d "
            "candidate_release_capacity=%d.",
            "exact" if exact_match else "reuse",
            target_bf16_blocks,
            len(cached_pages),
            len(selected),
            candidate_capacity,
        )
        return ReflexDemotionPlan(
            selected,
            candidate_bf16_blocks=candidate_capacity,
            candidate_pages=tuple(cached_pages[: len(selected)]),
            candidate_breakdown=cached_breakdown,
        )

    def plan_reflex_int4_demotions(
        self,
        *,
        target_bf16_blocks: int,
        keep_recent_pages: int = 4,
        keep_initial_pages: int = 0,
        max_int4_fraction_per_request: float = 1.0,
        low_risk_only: bool = False,
        sparse_window_pages: int = 0,
        max_demote_per_window: int = 0,
        selection_policy: str = "relevance_sparse",
        dual_price_state: DualPriceState | None = None,
        emergency_release: bool = False,
        cache_scope: str | None = None,
        request_precision_budgets: dict[str, RequestPrecisionBudget] | None = None,
        computed_tokens_by_request: dict[str, int] | None = None,
        prompt_tokens_by_request: dict[str, int] | None = None,
        protected_request_ids: set[str] | None = None,
        allow_partial_prefill_demotion_request_ids: set[str] | None = None,
        protected_prompt_pages_by_request: dict[str, int] | None = None,
        protected_pages_by_request: dict[str, set[int]] | None = None,
        sealed_pages_by_request: dict[str, int] | None = None,
        remote_inflight_pages_by_request: dict[str, set[int]] | None = None,
        prefill_page_risks_by_request: dict[str, Sequence[float]] | None = None,
        compressible_pages_by_request: dict[str, set[int]] | None = None,
        copy_on_demote_pages_by_request: dict[str, set[int]] | None = None,
        recovery_shadow_pages_by_request: dict[str, set[int]] | None = None,
        recovery_shadow_pages_per_request: int = 0,
        dry_run: bool = False,
    ) -> int:
        if self.reflex_int4_pool is None or target_bf16_blocks <= 0:
            self._last_reflex_int4_candidate_capacity = 0
            self._last_reflex_int4_candidate_breakdown = ReflexCandidateBreakdown()
            return 0

        cache_key = self._make_reflex_int4_frontier_cache_key(
            cache_scope=cache_scope,
            target_bf16_blocks=target_bf16_blocks,
            keep_recent_pages=keep_recent_pages,
            keep_initial_pages=keep_initial_pages,
            max_int4_fraction_per_request=max_int4_fraction_per_request,
            low_risk_only=low_risk_only,
            sparse_window_pages=sparse_window_pages,
            max_demote_per_window=max_demote_per_window,
            selection_policy=selection_policy,
            dual_price_state=dual_price_state,
            emergency_release=emergency_release,
            request_precision_budgets=request_precision_budgets,
            computed_tokens_by_request=computed_tokens_by_request,
            prompt_tokens_by_request=prompt_tokens_by_request,
            protected_request_ids=protected_request_ids,
            allow_partial_prefill_demotion_request_ids=(
                allow_partial_prefill_demotion_request_ids
            ),
            protected_prompt_pages_by_request=protected_prompt_pages_by_request,
            protected_pages_by_request=protected_pages_by_request,
            sealed_pages_by_request=sealed_pages_by_request,
            remote_inflight_pages_by_request=remote_inflight_pages_by_request,
            prefill_page_risks_by_request=prefill_page_risks_by_request,
            compressible_pages_by_request=compressible_pages_by_request,
            copy_on_demote_pages_by_request=copy_on_demote_pages_by_request,
        )
        reuse_key = self._make_reflex_int4_frontier_reuse_key(
            cache_scope=cache_scope,
            keep_recent_pages=keep_recent_pages,
            keep_initial_pages=keep_initial_pages,
            max_int4_fraction_per_request=max_int4_fraction_per_request,
            low_risk_only=low_risk_only,
            sparse_window_pages=sparse_window_pages,
            max_demote_per_window=max_demote_per_window,
            selection_policy=selection_policy,
            emergency_release=emergency_release,
            request_precision_budgets=request_precision_budgets,
            computed_tokens_by_request=computed_tokens_by_request,
            prompt_tokens_by_request=prompt_tokens_by_request,
            protected_request_ids=protected_request_ids,
            allow_partial_prefill_demotion_request_ids=(
                allow_partial_prefill_demotion_request_ids
            ),
            protected_prompt_pages_by_request=protected_prompt_pages_by_request,
            protected_pages_by_request=protected_pages_by_request,
            sealed_pages_by_request=sealed_pages_by_request,
            remote_inflight_pages_by_request=remote_inflight_pages_by_request,
            prefill_page_risks_by_request=prefill_page_risks_by_request,
            compressible_pages_by_request=compressible_pages_by_request,
            copy_on_demote_pages_by_request=copy_on_demote_pages_by_request,
        )
        plan: ReflexDemotionPlan | None = None
        if not dry_run:
            plan = self._plan_reflex_int4_demotions_from_cached_frontier(
                cache_key=cache_key,
                reuse_key=reuse_key,
                target_bf16_blocks=target_bf16_blocks,
                prompt_tokens_by_request=prompt_tokens_by_request,
            )

        if plan is None:
            request_pages = self._build_reflex_page_metadata(
                computed_tokens_by_request=computed_tokens_by_request,
                prompt_tokens_by_request=prompt_tokens_by_request,
                protected_request_ids=protected_request_ids,
                allow_partial_prefill_demotion_request_ids=(
                    allow_partial_prefill_demotion_request_ids
                ),
                protected_prompt_pages_by_request=protected_prompt_pages_by_request,
                protected_pages_by_request=protected_pages_by_request,
                sealed_pages_by_request=sealed_pages_by_request,
                remote_inflight_pages_by_request=remote_inflight_pages_by_request,
                prefill_page_risks_by_request=prefill_page_risks_by_request,
                compressible_pages_by_request=compressible_pages_by_request,
                copy_on_demote_pages_by_request=copy_on_demote_pages_by_request,
            )
            planner = DistanceDemotionPlanner(
                keep_recent_pages=keep_recent_pages,
                keep_initial_pages=keep_initial_pages,
                max_int4_fraction_per_request=max_int4_fraction_per_request,
                low_risk_only=low_risk_only,
                sparse_window_pages=sparse_window_pages,
                max_demote_per_window=max_demote_per_window,
                selection_policy=selection_policy,
                dual_price_state=dual_price_state,
                emergency_release=emergency_release,
            )
            plan = planner.plan(
                request_pages,
                target_bf16_blocks=target_bf16_blocks,
                int4_pool=self.reflex_int4_pool,
                request_precision_budgets=request_precision_budgets,
                dry_run=dry_run,
            )
        self._last_reflex_int4_candidate_capacity = plan.candidate_bf16_blocks
        self._last_reflex_int4_candidate_breakdown = plan.candidate_breakdown
        if dry_run:
            self._store_reflex_int4_cached_frontier(
                cache_key=cache_key,
                reuse_key=reuse_key,
                plan=plan,
            )
            return plan.released_bf16_blocks
        recovery_shadow_pages_by_request = recovery_shadow_pages_by_request or {}
        recovery_shadow_pages_per_request = max(
            0,
            int(recovery_shadow_pages_per_request),
        )
        recovery_shadow_counts_by_request: defaultdict[str, int] = defaultdict(int)
        for demotion in plan.items:
            shadow_pages = recovery_shadow_pages_by_request.get(
                demotion.request_id,
                set(),
            )
            should_store_shadow = demotion.page_idx in shadow_pages
            if (
                not should_store_shadow
                and recovery_shadow_counts_by_request[demotion.request_id]
                < recovery_shadow_pages_per_request
            ):
                should_store_shadow = True
            if should_store_shadow:
                recovery_shadow_counts_by_request[demotion.request_id] += 1
            recovery_class = (
                RecoveryClass.BF16_SHADOW if should_store_shadow else RecoveryClass.NONE
            )
            demotion = ReflexDemotion(
                request_id=demotion.request_id,
                page_idx=demotion.page_idx,
                bf16_block_id=demotion.bf16_block_id,
                int4_block_id=demotion.int4_block_id,
                encoded_block_table_id=demotion.encoded_block_table_id,
                kv_cache_group_id=self.kv_cache_group_id,
                recovery_class=recovery_class,
                recovery_shadow_bytes=demotion.recovery_shadow_bytes,
                is_prompt_page=demotion.is_prompt_page,
                prompt_pages=demotion.prompt_pages,
                risk_score=demotion.risk_score,
                is_low_risk=demotion.is_low_risk,
                copy_on_demote=demotion.copy_on_demote,
            )
            logger.info(
                "ReFlexKV trace demote_page request=%s page_idx=%d "
                "is_prompt_page=%s prompt_pages=%s bf16_block=%d "
                "int4_block=%d risk_score=%s is_low_risk=%s "
                "copy_on_demote=%s recovery_class=%s.",
                demotion.request_id,
                demotion.page_idx,
                demotion.is_prompt_page,
                ("none" if demotion.prompt_pages is None else demotion.prompt_pages),
                demotion.bf16_block_id,
                demotion.int4_block_id,
                (
                    "none"
                    if demotion.risk_score is None
                    else f"{demotion.risk_score:.6f}"
                ),
                demotion.is_low_risk,
                demotion.copy_on_demote,
                demotion.recovery_class.value,
            )
            self._apply_reflex_int4_demotion(demotion)
            self._pending_reflex_int4_demotions.append(demotion)
        return plan.released_bf16_blocks

    def get_last_reflex_int4_candidate_capacity(self) -> int:
        return self._last_reflex_int4_candidate_capacity

    def get_last_reflex_int4_candidate_breakdown(
        self,
    ) -> ReflexCandidateBreakdown:
        return self._last_reflex_int4_candidate_breakdown

    def take_reflex_int4_demotions(self) -> list[ReflexDemotion]:
        demotions = self._pending_reflex_int4_demotions
        self._pending_reflex_int4_demotions = []
        return demotions

    def take_reflex_int4_recoveries(self) -> list[ReflexRecovery]:
        recoveries = self._pending_reflex_int4_recoveries
        self._pending_reflex_int4_recoveries = []
        return recoveries

    def _is_reflex_recoverable_page(
        self,
        request_id: str,
        page_idx: int,
    ) -> bool:
        if request_id not in self.req_to_blocks:
            return False
        if page_idx < 0:
            return False
        self._ensure_reflex_block_ids(request_id)
        encoded_ids = self.req_to_reflex_block_ids[request_id]
        if page_idx >= len(encoded_ids):
            return False
        if page_idx in self._pending_reflex_bf16_release_pages_by_request.get(
            request_id,
            set(),
        ):
            return False
        precision, physical_id = decode_block_table_entry(encoded_ids[page_idx])
        if precision != PrecisionState.INT4:
            return False
        artifact = self.req_to_reflex_recovery_artifacts.get(
            request_id,
            {},
        ).get(page_idx)
        return (
            artifact is not None
            and artifact.recovery_class != RecoveryClass.NONE
            and artifact.int4_block_id == physical_id
        )

    def _ensure_reflex_block_ids(self, request_id: str) -> None:
        if self.reflex_int4_pool is None:
            return
        encoded = self.req_to_reflex_block_ids[request_id]
        blocks = self.req_to_blocks[request_id]
        if len(encoded) == len(blocks):
            return
        if encoded:
            raise RuntimeError(
                "ReFlexKV block id table is out of sync for request "
                f"{request_id}: {len(encoded)} encoded ids for {len(blocks)} blocks."
            )
        encoded.extend(block.block_id for block in blocks)

    def _build_reflex_page_metadata(
        self,
        *,
        computed_tokens_by_request: dict[str, int] | None = None,
        prompt_tokens_by_request: dict[str, int] | None = None,
        protected_request_ids: set[str] | None = None,
        allow_partial_prefill_demotion_request_ids: set[str] | None = None,
        protected_prompt_pages_by_request: dict[str, int] | None = None,
        protected_pages_by_request: dict[str, set[int]] | None = None,
        sealed_pages_by_request: dict[str, int] | None = None,
        remote_inflight_pages_by_request: dict[str, set[int]] | None = None,
        prefill_page_risks_by_request: dict[str, Sequence[float]] | None = None,
        compressible_pages_by_request: dict[str, set[int]] | None = None,
        copy_on_demote_pages_by_request: dict[str, set[int]] | None = None,
    ) -> dict[str, list[ReflexPageMeta]]:
        assert self.reflex_int4_pool is not None
        computed_tokens_by_request = computed_tokens_by_request or {}
        prompt_tokens_by_request = prompt_tokens_by_request or {}
        protected_request_ids = protected_request_ids or set()
        allow_partial_prefill_demotion_request_ids = (
            allow_partial_prefill_demotion_request_ids or set()
        )
        protected_prompt_pages_by_request = protected_prompt_pages_by_request or {}
        protected_pages_by_request = protected_pages_by_request or {}
        sealed_pages_by_request = sealed_pages_by_request or {}
        remote_inflight_pages_by_request = remote_inflight_pages_by_request or {}
        prefill_page_risks_by_request = prefill_page_risks_by_request or {}
        compressible_pages_by_request = compressible_pages_by_request or {}
        copy_on_demote_pages_by_request = copy_on_demote_pages_by_request or {}
        request_pages: dict[str, list[ReflexPageMeta]] = {}
        for request_id, blocks in self.req_to_blocks.items():
            self._ensure_reflex_block_ids(request_id)
            encoded_ids = self.req_to_reflex_block_ids[request_id]
            computed_tokens = computed_tokens_by_request.get(request_id, 0)
            prompt_tokens = prompt_tokens_by_request.get(request_id)
            explicit_sealed_pages = max(
                0,
                int(sealed_pages_by_request.get(request_id, 0) or 0),
            )
            prompt_pages = (
                None if prompt_tokens is None else cdiv(prompt_tokens, self.block_size)
            )
            page_risks = prefill_page_risks_by_request.get(request_id)
            compressible_pages = compressible_pages_by_request.get(request_id)
            copy_on_demote_pages = copy_on_demote_pages_by_request.get(
                request_id,
                set(),
            )
            remote_inflight_pages = remote_inflight_pages_by_request.get(
                request_id,
                set(),
            )
            protected_prompt_pages = max(
                0,
                int(protected_prompt_pages_by_request.get(request_id, 0) or 0),
            )
            protected_pages = protected_pages_by_request.get(request_id, set())
            allow_partial_prefill_demotion = (
                request_id in allow_partial_prefill_demotion_request_ids
            )
            request_protected = request_id in protected_request_ids
            if request_protected or (
                prompt_tokens is not None
                and computed_tokens < prompt_tokens
                and not allow_partial_prefill_demotion
                and explicit_sealed_pages <= 0
            ):
                num_sealed_pages = 0
            else:
                # A completed prefill chunk is already a closed KV prefix even
                # before the whole prompt is loaded. Only the partial tail page
                # remains open by construction. This path is reserved for
                # chunked remote prefill where decode already owns that chunk.
                num_sealed_pages = max(
                    computed_tokens // self.block_size,
                    explicit_sealed_pages,
                )
            num_sealed_pages = min(len(blocks), num_sealed_pages)
            pages: list[ReflexPageMeta] = []
            for page_idx, encoded_id in enumerate(encoded_ids):
                precision, physical_id = decode_block_table_entry(encoded_id)
                is_prompt_page = prompt_pages is not None and page_idx < prompt_pages
                if precision == PrecisionState.INT4:
                    pages.append(
                        ReflexPageMeta(
                            request_id=request_id,
                            page_idx=page_idx,
                            precision=PrecisionState.INT4,
                            bf16_block_id=None,
                            int4_block_id=physical_id,
                            is_prompt_page=is_prompt_page,
                        )
                    )
                    continue
                block = blocks[page_idx]
                if prompt_pages is not None and page_idx >= prompt_pages:
                    compressible = True
                else:
                    compressible = (
                        None
                        if compressible_pages is None
                        else page_idx in compressible_pages
                    )
                prefill_risk = (
                    None
                    if page_risks is None or page_idx >= len(page_risks)
                    else float(page_risks[page_idx])
                )
                pages.append(
                    ReflexPageMeta(
                        request_id=request_id,
                        page_idx=page_idx,
                        precision=PrecisionState.BF16,
                        bf16_block_id=physical_id,
                        int4_block_id=None,
                        is_full=page_idx < num_sealed_pages,
                        is_shared=self._is_reflex_shared_bf16_block(block),
                        compressible=compressible,
                        prefill_risk=prefill_risk,
                        is_prompt_page=is_prompt_page,
                        is_prompt_protected=(
                            is_prompt_page
                            and (
                                page_idx < protected_prompt_pages
                                or page_idx in protected_pages
                            )
                        ),
                        is_page_protected=page_idx in protected_pages,
                        copy_on_demote=page_idx in copy_on_demote_pages,
                        is_remote_inflight=page_idx in remote_inflight_pages,
                        is_request_protected=request_protected,
                    )
                )
            request_pages[request_id] = pages
        return request_pages

    def _apply_reflex_int4_demotion(self, demotion: ReflexDemotion) -> None:
        blocks = self.req_to_blocks[demotion.request_id]
        encoded_ids = self.req_to_reflex_block_ids[demotion.request_id]
        block = blocks[demotion.page_idx]
        if block == self._null_block or block.block_id != demotion.bf16_block_id:
            raise RuntimeError(
                "Cannot demote stale ReFlexKV page: "
                f"request={demotion.request_id}, page={demotion.page_idx}, "
                f"expected BF16 block={demotion.bf16_block_id}."
            )
        if block.block_hash is not None:
            self.block_pool.evict_blocks({block.block_id})
        if demotion.recovery_class != RecoveryClass.NONE:
            self.req_to_reflex_recovery_artifacts[demotion.request_id][
                demotion.page_idx
            ] = ReflexRecoveryArtifact(
                request_id=demotion.request_id,
                page_idx=demotion.page_idx,
                source_bf16_block_id=demotion.bf16_block_id,
                int4_block_id=demotion.int4_block_id,
                recovery_class=demotion.recovery_class,
                kv_cache_group_id=demotion.kv_cache_group_id,
                shadow_bytes=demotion.recovery_shadow_bytes,
            )
        else:
            self.req_to_reflex_recovery_artifacts.get(
                demotion.request_id,
                {},
            ).pop(demotion.page_idx, None)
        self._reflex_recovered_pages_by_request.get(
            demotion.request_id,
            set(),
        ).discard(demotion.page_idx)
        self._pending_reflex_bf16_releases.append(block)
        self._pending_reflex_bf16_release_count_by_request[demotion.request_id] += 1
        self._pending_reflex_bf16_release_pages_by_request[demotion.request_id].add(
            demotion.page_idx
        )
        blocks[demotion.page_idx] = self._null_block
        encoded_ids[demotion.page_idx] = demotion.encoded_block_table_id

    def _free_reflex_int4_blocks(self, request_id: str) -> None:
        if self.reflex_int4_pool is None:
            return
        for encoded_id in self.req_to_reflex_block_ids.get(request_id, ()):
            precision, physical_id = decode_block_table_entry(encoded_id)
            if precision == PrecisionState.INT4 and self.reflex_int4_pool.is_allocated(
                physical_id
            ):
                self.reflex_int4_pool.free(physical_id)

    def _free_reflex_int4_landing_blocks(self, request_id: str) -> None:
        if self.reflex_int4_pool is None:
            return
        for block_id in self.req_to_reflex_landing_int4_ids.pop(request_id, ()):
            if self.reflex_int4_pool.is_allocated(block_id):
                self.reflex_int4_pool.free(block_id)
        self.req_to_reflex_landing_page_indices.pop(request_id, None)
        self.req_to_reflex_direct_landing_page_indices.pop(request_id, None)

    @abstractmethod
    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        Get the number of common prefix blocks for all requests with allocated
        KV cache.

        Args:
            running_request_id: The request ID.

        Returns:
            The number of common prefix blocks for all requests with allocated
            KV cache.
        """

        raise NotImplementedError

    @classmethod
    @abstractmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        Get the longest cache hit prefix of the blocks that is not longer than
        `max_length`. The prefix should be a common prefix hit for all the
        kv cache groups in `kv_cache_group_ids`. If no cache hit is found,
        return an empty list.
        If eagle is enabled, drop the last matched block to force recompute the
        last block to get the required hidden states for eagle drafting head.
        Need to be customized for each attention type.

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens). By default, it should
                be set to the block_size.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.

        Returns:
            A list of cached blocks with skipped blocks replaced by null block
            for each kv cache group in `kv_cache_group_ids`.
            Return a list of length `len(kv_cache_group_ids)`, where the i-th
            element is a list of cached blocks for the i-th kv cache group
            in `kv_cache_group_ids`.
            For example, sliding window manager should return a list like
            ([NULL, NULL, KVCacheBlock(7), KVCacheBlock(8)]) for block size 4
            and sliding window 8 and len(kv_cache_group_ids) = 1.
        """

        raise NotImplementedError

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """
        Remove and free the blocks that are no longer needed for attention computation.
        The removed blocks should be replaced by null_block.

        This function depends on `get_num_skipped_tokens`, which need to be implemented
        differently for each attention type.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        # Remove the blocks that will be skipped during attention computation.
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            # This indicates that ALL tokens are inside attention window.
            # Thus we do not need to free any blocks outside attention window.
            # A typical case is full attention that we never free any token
            # before the request is finished.
            return
        blocks = self.req_to_blocks[request_id]
        num_skipped_blocks = num_skipped_tokens // self.block_size
        # `num_skipped_tokens` may include tokens that haven't been allocated yet
        # (e.g., when the attention window moves into the external computed tokens
        # range), so we must cap to the number of blocks that currently exist for
        # this request.
        num_skipped_blocks = min(num_skipped_blocks, len(blocks))
        removed_blocks: list[KVCacheBlock] = []
        # Because the block starts from index 0, the num_skipped_block-th block
        # corresponds to index num_skipped_blocks - 1.
        for i in range(num_skipped_blocks - 1, -1, -1):
            if blocks[i] == self._null_block:
                # If the block is already a null block, the blocks before it
                # should also have been set to null blocks by the previous calls
                # to this function.
                break
            removed_blocks.append(blocks[i])
            blocks[i] = self._null_block
        self.block_pool.free_blocks(removed_blocks)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        # The default behavior is to not skip any tokens.
        return 0

    def new_step_starts(self) -> None:
        if self._pending_reflex_bf16_releases:
            self.block_pool.free_blocks(self._pending_reflex_bf16_releases)
            self._pending_reflex_bf16_releases = []
            self._pending_reflex_bf16_release_count_by_request.clear()
            self._pending_reflex_bf16_release_pages_by_request.clear()
        return None


class FullAttentionManager(SingleTypeKVCacheManager):
    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(
            kv_cache_spec, FullAttentionSpec | ChunkedLocalAttentionSpec
        ), (
            "FullAttentionManager can only be used for full attention "
            "and chunked local attention groups"
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        if dcp_world_size * pcp_world_size > 1:
            block_size *= dcp_world_size * pcp_world_size
        max_num_blocks = max_length // block_size
        for block_hash in itertools.islice(block_hashes, max_num_blocks):
            # block_hashes is a chain of block hashes. If a block hash is not
            # in the cached_block_hash_to_id, the following block hashes are
            # not computed yet for sure.
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        if use_eagle and computed_blocks[0]:
            # Need to drop the last matched block if eagle is enabled.
            for computed in computed_blocks:
                computed.pop()
        while (
            block_size != alignment_tokens  # Faster for common case.
            and len(computed_blocks[0]) * block_size % alignment_tokens != 0
        ):
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        blocks = self.req_to_blocks[running_request_id]
        num_common_blocks = 0
        for block in blocks:
            if block.ref_cnt == len(self.req_to_blocks):
                num_common_blocks += 1
            else:
                break
        return num_common_blocks


class SlidingWindowManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: SlidingWindowSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.sliding_window = kv_cache_spec.sliding_window

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, SlidingWindowSpec), (
            "SlidingWindowManager can only be used for sliding window groups"
        )
        assert dcp_world_size == 1, "DCP not support sliding window attn now."
        assert pcp_world_size == 1, "PCP not support sliding window attn now."

        # The number of contiguous blocks needed for prefix cache hit.
        # -1 since the input token itself is also included in the window
        sliding_window_contiguous_blocks = cdiv(
            kv_cache_spec.sliding_window - 1, kv_cache_spec.block_size
        )
        if use_eagle:
            # Need to drop the last matched block if eagle is enabled. For
            # sliding window layer, we achieve this by increasing the number of
            # contiguous blocks needed for prefix cache hit by one and dropping
            # the last matched block.
            sliding_window_contiguous_blocks += 1

        # TODO: reduce i by sliding_window_contiguous_blocks when cache miss, to
        # optimize the time complexity from O(max_num_blocks) to
        # O(max_num_blocks / sliding_window_contiguous_blocks +
        # sliding_window_contiguous_blocks),
        # which is good for low cache hit rate scenarios.
        max_num_blocks = max_length // kv_cache_spec.block_size
        computed_blocks = tuple(
            [block_pool.null_block] * max_num_blocks
            for _ in range(len(kv_cache_group_ids))
        )
        block_size = kv_cache_spec.block_size
        num_contiguous_blocks = 0
        match_found = False
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # Skip prefix matching check if the block is not aligned with
                # `alignment_tokens`.
                if (
                    num_contiguous_blocks == 0
                    and block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                # Add the cached block to the computed blocks.
                for computed, cached in zip(computed_blocks, cached_block):
                    computed[i] = cached
                num_contiguous_blocks += 1
                if num_contiguous_blocks >= sliding_window_contiguous_blocks:
                    # Trim the trailing blocks.
                    # E.g., [NULL, NULL, 8, 3, NULL, 9] -> [NULL, NULL, 8, 3]
                    # when sliding_window_contiguous_blocks=2.
                    for computed in computed_blocks:
                        del computed[i + num_contiguous_blocks :]
                    match_found = True
                    break
            else:
                num_contiguous_blocks = 0
        if not match_found:
            # The first `num_contiguous_blocks` is a cache hit even if
            # `num_contiguous_blocks < sliding_window_contiguous_blocks`.
            for computed in computed_blocks:
                del computed[num_contiguous_blocks:]
            while (
                block_size != alignment_tokens  # Faster for common case.
                and len(computed_blocks[0]) * block_size % alignment_tokens != 0
            ):
                for computed in computed_blocks:
                    computed.pop()
        if use_eagle and computed_blocks[0]:
            assert kv_cache_spec.block_size == alignment_tokens, (
                "aligned_length is not compatible with eagle now"
            )
            for computed in computed_blocks:
                computed.pop()
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For sliding window, this corresponds to the tokens that are prior to
        the current sliding window.

        Example:
        sliding_window=4, num_computed_tokens=7

        Tokens:   [ 0  1  2  3  4  5  6  7 ]
                  | ---- computed -----|
                                         ^ next token to be computed
                               |-----------| sliding window for next token
                  |--skipped---|

        The current window contains tokens 4~7. Tokens 0~3 will be skipped for
        attention computation since they are outside the sliding window.
        Thus, get_num_skipped_tokens(7) == 4.

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        return max(0, num_computed_tokens - self.sliding_window + 1)

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        NOTE(Chen): The prefix blocks are null blocks for sliding window layers.
        So it's not correct to count ref_cnt like FullAttentionManager. Return
        0 here for correctness. Need to support cascade attention + sliding
        window in the future.
        """
        return 0


class ChunkedLocalAttentionManager(SingleTypeKVCacheManager):
    def __init__(self, kv_cache_spec: ChunkedLocalAttentionSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self.attention_chunk_size = kv_cache_spec.attention_chunk_size

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        """
        For chunked local attention, we need to find the longest cache hit
        prefix of the blocks that is not longer than `max_length`. The prefix
        should be a common prefix hit for all the kv cache groups in
        `kv_cache_group_ids`. If no cache hit is found, return an empty list.
        note we mark as computed if the whole block is outside of the local
        window, and set the block as null. Examples:

        1. Attention chunk size of 8, block size of 4, max length of 15
        for next token at 15th (zero-indexed), 8th - 14th tokens are in
        the window(needs lookup), 0th - 7th are not in the window,
        so they are already marked as computed. We check the complete
        block3 (8th - 11th tokens), Assume block 3 is hit, we will return
        [null, null, block 3], otherwise, we return [null, null]

        2. Attention chunk size of 8, block size of 4, max length of 16
        for next token at 16th (zero-indexed), 0th - 15th tokens are not
        in the window, so they are already marked as computed.
        we return 4 blocks[null, null, null, null]

        Args:
            block_hashes: The block hashes of the request.
            max_length: The maximum length of the cache hit prefix.
            kv_cache_group_ids: The ids of the kv cache groups.
            block_pool: The block pool.
            kv_cache_spec: The kv cache spec.
            use_eagle: Whether to use eagle.
            dcp_world_size: The world size of decode context parallelism.
            pcp_world_size: The world size of prefill context parallelism.
            alignment_tokens: The returned cache hit length (in tokens) should
                be a multiple of this value (in tokens).

        Returns:
            A list of cached blocks
        """
        assert isinstance(kv_cache_spec, ChunkedLocalAttentionSpec), (
            "ChunkedLocalAttentionManager can only be used for "
            "chunked local attention groups"
        )
        assert use_eagle is False, (
            "Hybrid KV cache is not supported for " + "eagle + chunked local attention."
        )
        assert dcp_world_size == 1, "DCP not support chunked local attn now."
        assert pcp_world_size == 1, "PCP not support chunked local attn now."
        assert kv_cache_spec.block_size == alignment_tokens, (
            "KV cache groups with different block sizes are not compatible with "
            "chunked local attention now"
        )
        max_num_blocks = max_length // kv_cache_spec.block_size
        if max_length > 0:
            local_attention_start_idx = (
                max_length
                // kv_cache_spec.attention_chunk_size
                * kv_cache_spec.attention_chunk_size
            )
        else:
            local_attention_start_idx = 0
        # we marked blocks out of window as computed
        # with null blocks, and blocks inside window based on cache lookup
        # result [null] [null] ... [null] [hit block 1 (1st block contain
        # last window)] [hit block 2] ... [hit block x]
        local_attention_start_block_idx = (
            local_attention_start_idx // kv_cache_spec.block_size
        )
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [block_pool.null_block] * local_attention_start_block_idx
            for _ in range(len(kv_cache_group_ids))
        )
        for i in range(local_attention_start_block_idx, max_num_blocks):
            block_hash = block_hashes[i]
            if cached_block := block_pool.get_cached_block(
                block_hash, kv_cache_group_ids
            ):
                for computed, cached in zip(computed_blocks, cached_block):
                    computed.append(cached)
            else:
                break
        return computed_blocks

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens that will be skipped for attention computation.

        For chunked local attention, this corresponds to the tokens that are on
        the left side of the current chunk.

        Example 1:
        chunk size = 8, num_computed_tokens = 13
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | ----- computed ---------------|
                                                  ^^ next token to be computed
                                   |----------------| <-- attention window for
                                                          next token
                 |--- skipped -----|
        Output: get_num_skipped_tokens(13) == 8

        Example 2:
        chunk size = 8, num_computed_tokens = 8
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 | --- computed ---|
                                     ^ next token to be computed
                                   |--| <-- attention window for next token
                 | --- skipped ----|
        Output: get_num_skipped_tokens(8) == 8

        Example 3:
        chunk size = 8, num_computed_tokens = 7
        Tokens:  [ 0 1 2 3 4 5 6 7 | 8 9 10 11 12 13 14 15 ] ...
                 |---computed---|
                                 ^ next token to be computed
                 |-----------------| <-- attention window for next token
                 no token should be skipped.
        Output: get_num_skipped_tokens(7) == 0

        Args:
            num_computed_tokens: The number of tokens that have been computed.

        Returns:
            The number of tokens that will be skipped for attention computation.
        """
        num_skipped_tokens = (
            num_computed_tokens // self.attention_chunk_size
        ) * self.attention_chunk_size
        return num_skipped_tokens

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by chunked local attention.
        """
        return 0


class MambaManager(SingleTypeKVCacheManager):
    def __init__(
        self, kv_cache_spec: MambaSpec, block_pool: BlockPool, **kwargs
    ) -> None:
        super().__init__(kv_cache_spec, block_pool, **kwargs)
        self.cached_blocks_this_step: set[BlockHashWithGroupId] = set()
        self.mamba_cache_mode = kv_cache_spec.mamba_cache_mode
        self.num_speculative_blocks: int = kv_cache_spec.num_speculative_blocks
        if self.mamba_cache_mode == "align":
            # Mapping from request ID to the index of the block
            # allocated in the previous step
            self.last_state_block_idx: dict[str, int] = {}
            # The set of the requests that have been allocated blocks
            self._allocated_block_reqs: set[str] = set()

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, MambaSpec), (
            "MambaManager can only be used for mamba groups"
        )
        assert dcp_world_size == 1, "DCP not support mamba now."
        assert pcp_world_size == 1, "PCP not support mamba now."
        computed_blocks: tuple[list[KVCacheBlock], ...] = tuple(
            [] for _ in range(len(kv_cache_group_ids))
        )

        block_size = kv_cache_spec.block_size
        max_num_blocks = max_length // block_size
        # Search from right to left and early stop when a match is found.
        for i in range(max_num_blocks - 1, -1, -1):
            if cached_block := block_pool.get_cached_block(
                block_hashes[i], kv_cache_group_ids
            ):
                # When enable Mamba prefix caching, `block_size` will be aligned
                # across full attention layers and Mamba layers to ensure the
                # prefix hit length aligned at block
                if (
                    block_size != alignment_tokens  # Faster for common case.
                    and (i + 1) * block_size % alignment_tokens != 0
                ):
                    continue
                for computed, cached in zip(computed_blocks, cached_block):
                    # the hit length logic later assumes:
                    #  hit_length = len(hit_blocks_other_attn[0])
                    #               * self.other_block_size
                    # so we insert dummy blocks at the beginning:
                    computed.extend([block_pool.null_block] * i)
                    computed.append(cached)
                break  # we just need the last match - early stopping

        return computed_blocks

    def remove_skipped_blocks(self, request_id: str, num_computed_tokens: int) -> None:
        assert isinstance(self.kv_cache_spec, MambaSpec)

        # NOTE (tdoublep) with async scheduling, the num_computed_tokens can contain
        # draft tokens from the previous step that may or may not be rejected later.
        # This can make us think we are further ahead in the sequence than we actually
        # are, so let's assume that all tokens are rejected so we don't free blocks
        # that we might actually need.
        num_computed_tokens = max(0, num_computed_tokens - self.num_speculative_blocks)

        super().remove_skipped_blocks(request_id, num_computed_tokens)
        if self.mamba_cache_mode == "align":
            # `last_state_block_idx` refers to the block index allocated two steps ago.
            # The block allocated in the previous step is used to copy Mamba states
            # into the block allocated in the current step; the earlier block is
            # no longer needed and should be freed here.
            last_state_block_idx = self.last_state_block_idx.get(request_id)
            # Blocks allocated during prefill may be non-contiguous. Use
            # `last_state_block_idx` to free the appropriate block and replace it
            # with a null block.
            if (
                last_state_block_idx is not None
                and last_state_block_idx
                < cdiv(num_computed_tokens, self.block_size) - 1
            ):
                blocks = self.req_to_blocks[request_id]
                if blocks[last_state_block_idx] != self._null_block:
                    self.block_pool.free_blocks([blocks[last_state_block_idx]])
                    blocks[last_state_block_idx] = self._null_block

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        """
        cascade attention is not supported by mamba
        """
        return 0

    def get_num_blocks_to_allocate(
        self,
        request_id: str,
        num_tokens: int,
        new_computed_blocks: Sequence[KVCacheBlock],
        total_computed_tokens: int,
        num_tokens_main_model: int,
    ) -> int:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if (
            len(new_computed_blocks) > 0
            and new_computed_blocks[-1].block_hash in self.cached_blocks_this_step
        ):
            # Mamba can't rely on blocks generated by other requests in the current step
            # To put it in the next step, we return num_gpu_blocks + 1 so
            # that kv_cache_manager will think there is no enough blocks to allocate now
            # and don't schedule it in the current step.
            return self.block_pool.num_gpu_blocks + 1
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += (
                    self.kv_cache_spec.block_size * self.num_speculative_blocks
                )
            return super().get_num_blocks_to_allocate(
                request_id,
                num_tokens,
                new_computed_blocks,
                total_computed_tokens,
                num_tokens_main_model,
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model

            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            num_new_blocks = (
                num_required_blocks
                - len(new_computed_blocks)
                - len(self.req_to_blocks[request_id])
            )
            if num_new_blocks > 0:
                if request_id in self._allocated_block_reqs:
                    # Old request. Needs at most 1 more blocks as we can reuse the
                    # speculative blocks in previous step.
                    num_new_blocks = 1
                else:
                    # First prefill. Allocate 1 block for running state and the
                    # speculative blocks.
                    num_new_blocks = 1 + self.num_speculative_blocks

            num_evictable_computed_blocks = self._get_num_evictable_blocks(
                new_computed_blocks
            )
            return num_new_blocks + num_evictable_computed_blocks

    def allocate_new_blocks(
        self, request_id: str, num_tokens: int, num_tokens_main_model: int
    ) -> list[KVCacheBlock]:
        assert isinstance(self.kv_cache_spec, MambaSpec)
        if self.mamba_cache_mode != "align":
            # Allocate extra `num_speculative_blocks` blocks for
            # speculative decoding (MTP/EAGLE) with linear attention.
            if self.num_speculative_blocks > 0:
                num_tokens += self.block_size * self.num_speculative_blocks
            return super().allocate_new_blocks(
                request_id, num_tokens, num_tokens_main_model
            )
        else:
            # We don't allocate blocks for lookahead tokens in align mode, because if
            # x * block_size tokens are scheduled, num_tokens is
            # x * block_size + num_lookahead_tokens and breaks the alignment.
            # We can ignore lookahead tokens because current draft models don't have
            # mamba layers.
            num_tokens = num_tokens_main_model
            req_blocks: list[KVCacheBlock] = self.req_to_blocks[request_id]
            # NOTE(tdouble): this is an over-estimate of how many blocks we need because
            # num_tokens can include draft tokens that will later be rejected.
            num_required_blocks = (
                cdiv(num_tokens, self.block_size) + self.num_speculative_blocks
            )
            if num_required_blocks == len(req_blocks):
                return []
            else:
                assert num_required_blocks > len(req_blocks), (
                    "num_required_blocks "
                    f"{num_required_blocks} < len(req_blocks) {len(req_blocks)}"
                )
                prev_block_len = len(req_blocks)
                blocks_allocated = request_id in self._allocated_block_reqs
                # Record the last state block
                if blocks_allocated:
                    # We always save the running state at the last
                    # (1 + num_speculative_blocks) block
                    self.last_state_block_idx[request_id] = (
                        prev_block_len - 1 - self.num_speculative_blocks
                    )
                elif prev_block_len > 0:
                    # When a new request hits the prefix cache, the last block
                    # saves the hit state.
                    self.last_state_block_idx[request_id] = prev_block_len - 1

                num_skipped_blocks = (
                    num_required_blocks - self.num_speculative_blocks - 1
                )
                # null blocks
                if prev_block_len < num_skipped_blocks:
                    req_blocks.extend(
                        [
                            self._null_block
                            for _ in range(prev_block_len, num_skipped_blocks)
                        ]
                    )

                if blocks_allocated:
                    # reuse previous speculative blocks in this step
                    for block_idx in range(
                        prev_block_len - self.num_speculative_blocks, prev_block_len
                    ):
                        if block_idx < num_skipped_blocks:
                            req_blocks.append(req_blocks[block_idx])
                            req_blocks[block_idx] = self._null_block
                        else:
                            break
                num_new_blocks = num_required_blocks - len(req_blocks)
                if blocks_allocated:
                    assert num_new_blocks <= 1
                else:
                    assert num_new_blocks <= self.num_speculative_blocks + 1
                new_blocks = self.block_pool.get_new_blocks(num_new_blocks)
                req_blocks.extend(new_blocks)
                self._allocated_block_reqs.add(request_id)
                return req_blocks[prev_block_len:]

    def free(self, request_id: str) -> None:
        if self.mamba_cache_mode == "align":
            self._allocated_block_reqs.discard(request_id)
            self.last_state_block_idx.pop(request_id, None)
        super().free(request_id)

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """
        Get the number of tokens whose mamba state are not needed anymore. Mamba only
        need to keep the state of the last computed token, so we return
        num_computed_tokens - 1.
        """
        return num_computed_tokens - 1

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        num_cached_blocks_before = self.num_cached_block.get(request.request_id, 0)
        super().cache_blocks(request, num_tokens)
        num_cached_blocks_after = self.num_cached_block.get(request.request_id, 0)
        if num_cached_blocks_after > num_cached_blocks_before:
            for block in self.req_to_blocks[request.request_id][
                num_cached_blocks_before:num_cached_blocks_after
            ]:
                if block.is_null:
                    continue
                assert block.block_hash is not None
                self.cached_blocks_this_step.add(block.block_hash)

    def new_step_starts(self) -> None:
        super().new_step_starts()
        self.cached_blocks_this_step.clear()


class CrossAttentionManager(SingleTypeKVCacheManager):
    """Manager for cross-attention KV cache in encoder-decoder models."""

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: Sequence[KVCacheBlock],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so  `new_computed_blocks` should always be empty.
        assert len(new_computed_blocks) == 0

    def cache_blocks(self, request: Request, num_tokens: int) -> None:
        # We do not cache blocks for cross-attention to be shared between
        # requests, so this method is not relevant.
        raise ValueError("Should not be called as prefix caching is disabled.")

    def get_num_common_prefix_blocks(self, running_request_id: str) -> int:
        # Cross-attention blocks contain request-specific encoder states
        # and are not shared between different requests
        return 0

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes: BlockHashList,
        max_length: int,
        kv_cache_group_ids: list[int],
        block_pool: BlockPool,
        kv_cache_spec: KVCacheSpec,
        use_eagle: bool,
        alignment_tokens: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ) -> tuple[list[KVCacheBlock], ...]:
        assert isinstance(kv_cache_spec, CrossAttentionSpec), (
            "CrossAttentionManager can only be used for cross-attention groups"
        )
        # Cross-attention does not benefit from prefix caching since:
        # 1. Encoder states are unique per request (different audio/image
        #    inputs)
        # 2. Encoder states are computed once per request, not incrementally
        # 3. No reusable prefix exists between different multimodal inputs
        # Return empty blocks to indicate no cache hits
        raise NotImplementedError("CrossAttentionManager does not support caching")


class SinkFullAttentionManager(FullAttentionManager):
    def __init__(
        self,
        kv_cache_spec: SinkFullAttentionSpec,
        block_pool: BlockPool,
        enable_caching: bool,
        kv_cache_group_id: int,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
    ):
        super().__init__(
            kv_cache_spec,
            block_pool,
            enable_caching,
            kv_cache_group_id,
            dcp_world_size,
            pcp_world_size,
        )
        sink_len = kv_cache_spec.sink_len
        assert sink_len is not None and sink_len > 0 and sink_len % self.block_size == 0
        num_sink_block = sink_len // self.block_size
        self.sink_blocks = self.block_pool.free_block_queue.popleft_n(num_sink_block)


spec_manager_map: dict[type[KVCacheSpec], type[SingleTypeKVCacheManager]] = {
    FullAttentionSpec: FullAttentionManager,
    MLAAttentionSpec: FullAttentionManager,
    SlidingWindowSpec: SlidingWindowManager,
    ChunkedLocalAttentionSpec: ChunkedLocalAttentionManager,
    MambaSpec: MambaManager,
    CrossAttentionSpec: CrossAttentionManager,
    SinkFullAttentionSpec: SinkFullAttentionManager,
}


def get_manager_for_kv_cache_spec(
    kv_cache_spec: KVCacheSpec, **kwargs
) -> SingleTypeKVCacheManager:
    manager_class = spec_manager_map[type(kv_cache_spec)]
    manager = manager_class(kv_cache_spec, **kwargs)
    return manager
