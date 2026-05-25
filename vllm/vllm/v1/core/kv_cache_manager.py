# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.precision_kv.demotion_planner import (
    ReflexCandidateBreakdown,
    RequestPrecisionBudget,
)
from vllm.v1.core.precision_kv.run_optimizer import DualPriceState
from vllm.v1.core.precision_kv.types import (
    KVPageRuntimeDescriptor,
    ReflexDemotion,
    ReflexRecovery,
)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class KVCacheBlocks:
    """
    The allocation result of KVCacheManager, work as the interface between
    Scheduler and KVCacheManager, to hide KVCacheManager's internal data
    structure from the Scheduler.
    """

    blocks: tuple[Sequence[KVCacheBlock], ...]
    """
    `blocks[i][j]` refers to the i-th kv_cache_group
    and the j-th block of tokens.We don't use block of
    tokens as the outer dimension because it assumes all
    kv_cache_groups have the same number of blocks, which is true for now but
    will be broken if we want to give different block_size to different
    kv_cache_groups in the future.

    Each single type KVCacheBlocks could be represented as:
    - list[KVCacheBlock] for more than one KVCacheBlock
    - an empty tuple for requests without KVCacheBlock
      (a precomputed KVCacheBlocks is in KVCacheManager to avoid GC overhead)
    """
    block_id_overrides: tuple[list[int], ...] | None = None
    """
    Optional worker-facing block IDs. ReFlexKV uses this to send encoded INT4
    block IDs for logical pages whose BF16 physical blocks have been released.
    """

    def __add__(self, other: "KVCacheBlocks") -> "KVCacheBlocks":
        """Adds two KVCacheBlocks instances."""
        blocks = tuple(
            list(itertools.chain(blk1, blk2))
            for blk1, blk2 in zip(self.blocks, other.blocks)
        )
        block_id_overrides = None
        if self.block_id_overrides is not None or other.block_id_overrides is not None:
            self_ids = self.get_block_ids()
            other_ids = other.get_block_ids()
            block_id_overrides = tuple(
                list(itertools.chain(ids1, ids2))
                for ids1, ids2 in zip(self_ids, other_ids)
            )
        return KVCacheBlocks(blocks, block_id_overrides)

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[False] = False,
    ) -> tuple[list[int], ...]: ...

    @overload
    def get_block_ids(
        self,
        allow_none: Literal[True] = True,
    ) -> tuple[list[int], ...] | None: ...

    def get_block_ids(
        self,
        allow_none: bool = False,
    ) -> tuple[list[int], ...] | None:
        """
        Converts the KVCacheBlocks instance to block_ids.

        Returns:
            tuple[list[int], ...]: A tuple of lists where:
                - the outer tuple corresponds to KV cache groups
                - each inner list contains the block_ids of the blocks in that
                  group
        """
        if allow_none and all(len(group) == 0 for group in self.blocks):
            return None
        if self.block_id_overrides is not None:
            if allow_none and all(len(group) == 0 for group in self.block_id_overrides):
                return None
            return self.block_id_overrides
        return tuple([blk.block_id for blk in group] for group in self.blocks)

    def get_unhashed_block_ids(self) -> list[int]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        assert len(self.blocks) == 1, "Only one group is supported"
        return [
            block.block_id
            for block in self.blocks[0]
            if block.block_hash is None and not block.is_null
        ]

    def get_unhashed_block_ids_all_groups(self) -> list[list[int]]:
        """Get block_ids of unhashed blocks from KVCacheBlocks instance."""
        # Skip padding blocks.
        return [
            [
                block.block_id
                for block in group
                if block.block_hash is None and not block.is_null
            ]
            for group in self.blocks
        ]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        hash_block_size: int,
        enable_caching: bool = True,
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        dcp_world_size: int = 1,
        pcp_world_size: int = 1,
        metrics_collector: KVCacheMetricsCollector | None = None,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        self.metrics_collector = metrics_collector
        # FIXME: make prefix cache stats conditional on log_stats. We still need
        # this comment because when the log stats is enabled there are still
        # potential configs we could expose in the future.
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=self.enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            dcp_world_size=dcp_world_size,
            pcp_world_size=pcp_world_size,
            hash_block_size=hash_block_size,
            metrics_collector=self.metrics_collector,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> PrefixCacheStats | None:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # We skip finding the prefix cache hit when prefix caching is
        # disabled or the request is marked as skipping kv cache read
        # (which happens when the request requires prompt logprobs
        # or calls a pooling model with all pooling).
        if not self.enable_caching or request.skip_reading_prefix_cache:
            return self.empty_kv_cache_blocks, 0

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1
        computed_blocks, num_new_computed_tokens = (
            self.coordinator.find_longest_cache_hit(
                request.block_hashes, max_cache_hit_length
            )
        )

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.record(
                num_tokens=request.num_tokens,
                num_hits=num_new_computed_tokens,
                preempted=request.num_preemptions > 0,
            )

        return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens

    def can_fit_full_sequence(
        self,
        request: Request,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_external_computed_tokens: int = 0,
        num_encoder_tokens: int = 0,
    ) -> bool:
        """Check if the KV cache has enough free blocks to hold the full
        sequence, accounting for prefix cache hits and sliding window.

        This is used as an admission gate to prevent over-admitting requests
        when chunked prefill would otherwise only check the first chunk.
        """
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        full_num_tokens = min(request.num_tokens, self.max_model_len)

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=full_num_tokens,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=total_computed_tokens,
            num_tokens_main_model=full_num_tokens,
        )

        return num_blocks_to_allocate <= self.block_pool.get_num_free_blocks()

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
    ) -> KVCacheBlocks | None:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of new tokens to be allocated and computed.
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed
                tokens, grouped as a tuple by kv cache groups.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such
                as eagle.
            num_external_computed_tokens: The number of tokens that their
                KV caches are not cached by vLLM but cached by the connector.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.
            num_encoder_tokens: The number of encoder tokens to allocate for
                cross-attention in encoder-decoder models(e.g., Whisper).
                For decoder-only models, this should be 0.

        Blocks layout:
        ```
        ----------------------------------------------------------------------
        | < comp > | < new_comp > | < ext_comp >  | < new >  | < lookahead > |
        ----------------------------------------------------------------------
                                                  |   < to be computed >     |
        ----------------------------------------------------------------------
                                  |            < to be allocated >           |
        ----------------------------------------------------------------------
                                  | < to be cached (roughly, |
                                  | details below)>          |
        ----------------------------------------------------------------------
        | Prefix-cached tokens from either vLLM   |
        | or connector. Can be safely removed if  |
        | they are outside sliding window.        |
        ----------------------------------------------------------------------
        |   < cached by vLLM >    | not cached by |
                                  | vLLM, but     |
        | ref_cnt  | ref_cnt not  | cached by     |
        | increased| increased yet| connector     |
        ----------------------------------------------------------------------
        ```

        Abbrivations:

        ```
        comp      = request.num_computed_tokens
        new_comp  = num_new_computed_tokens
                  = len(new_computed_blocks) * block_size
        ext_comp  = num_external_computed_tokens, cached by the connector
        new       = num_new_tokens, including unverified draft tokens
        lookahead = num_lookahead_tokens
        ```

        NOTE: for new tokens which include both verified and unverified draft
        tokens, we only cache the verified tokens (by capping the number at
        `request.num_tokens`).

        The allocation has three stages:
        - Free unnecessary blocks in `comp` and check
           if we have sufficient free blocks (return None if not).
        - Handle prefix tokens (`comp + new_comp + ext_comp`):
            - Free unnecessary blocks (e.g. outside sliding window)
            - Allocate new blocks for `ext_comp` tokens inside
              sliding window
        - Allocate new blocks for tokens to be computed (`new + lookahead`)

        Returns:
            A list of new allocated blocks.
        """
        # When loading KV data asynchronously, we may have zero new tokens to
        # compute while still allocating slots for externally computed tokens.
        if num_new_tokens == 0 and num_external_computed_tokens == 0:
            raise ValueError(
                "num_new_tokens must be greater than 0 when there are no "
                "external computed tokens"
            )

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = self.empty_kv_cache_blocks.blocks

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_local_computed_tokens = (
            request.num_computed_tokens + num_new_computed_tokens
        )
        total_computed_tokens = min(
            num_local_computed_tokens + num_external_computed_tokens,
            self.max_model_len,
        )
        num_tokens_main_model = total_computed_tokens + num_new_tokens
        num_tokens_need_slot = min(
            num_tokens_main_model + num_lookahead_tokens,
            self.max_model_len,
        )

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.coordinator.remove_skipped_blocks(
            request.request_id, total_computed_tokens
        )

        num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
            request_id=request.request_id,
            num_tokens=num_tokens_need_slot,
            new_computed_blocks=new_computed_block_list,
            num_encoder_tokens=num_encoder_tokens,
            total_computed_tokens=num_local_computed_tokens
            + num_external_computed_tokens,
            num_tokens_main_model=num_tokens_main_model,
        )

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        if (
            new_computed_block_list is not self.empty_kv_cache_blocks.blocks
            or num_external_computed_tokens > 0
        ):
            # Append the new computed blocks to the request blocks until now to
            # avoid the case where the new blocks cannot be allocated.
            self.coordinator.allocate_new_computed_blocks(
                request_id=request.request_id,
                new_computed_blocks=new_computed_block_list,
                num_local_computed_tokens=num_local_computed_tokens,
                num_external_computed_tokens=num_external_computed_tokens,
            )

        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id,
            num_tokens_need_slot,
            num_tokens_main_model,
            num_encoder_tokens,
        )

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return self.create_kv_cache_blocks(new_blocks)

        # NOTE(woosuk): We want to commit (cache) up to num_local_computed_tokens
        # + num_external_computed_tokens + num_new_tokens, but must exclude
        # "non-committable" tokens (e.g., draft tokens that could be rejected).
        # Therefore, we cap the number at `request.num_tokens`, ensuring only
        # "finalized" tokens are cached.
        num_tokens_to_cache = min(
            total_computed_tokens + num_new_tokens,
            request.num_tokens,
        )
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

        return self.create_kv_cache_blocks(new_blocks)

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that the tail blocks are evicted
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.coordinator.free(request.request_id)

    def remove_skipped_blocks(
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove the blocks that are no longer needed from `blocks` and replace
        the removed blocks with null_block.

        Args:
            request_id: The request ID.
            total_computed_tokens: The total number of computed tokens, including
                local computed tokens and external computed tokens.
        """
        self.coordinator.remove_skipped_blocks(request_id, total_computed_tokens)

    def evict_blocks(self, block_ids: set[int]) -> None:
        """evict blocks from the prefix cache by their block IDs.

        Args:
            block_ids: Set of block IDs to evict from cache.
        """
        self.block_pool.evict_blocks(block_ids)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(self, running_request_id: str) -> list[int]:
        """Calculate the number of common prefix blocks for each kv cache group.

        The function selects a running request and iterates through its blocks.
        A block is considered a common prefix block if ALL requests with
        allocated KV cache share it (i.e., ref_cnt equals the number of entries
        in req_to_blocks).

        NOTE(woosuk): The number of requests with allocated KV cache is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because having allocated KV cache only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must have allocated KV cache, the inverse
        is not necessarily true. There may be requests with allocated KV cache
        that are not scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled requests that do not share the
        common prefix. Currently, this case cannot be easily detected, so the
        function returns 0 in such cases.

        Args:
            running_request_id: The request ID of any running request, used to
                identify the common prefix blocks.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache
            group.
        """
        return self.coordinator.get_num_common_prefix_blocks(running_request_id)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        blocks = self.coordinator.get_blocks(request_id)
        block_id_overrides = self.coordinator.get_block_ids(request_id)
        return self.create_kv_cache_blocks(blocks, block_id_overrides)

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()

    def has_reflex_int4_blocks(self, request_id: str) -> bool:
        return self.coordinator.has_reflex_int4_blocks(request_id)

    def reserve_reflex_int4_landing_blocks(
        self,
        request_id: str,
        count: int,
    ) -> list[int]:
        return self.coordinator.reserve_reflex_int4_landing_blocks(
            request_id,
            count,
        )

    def record_reflex_int4_landing_pages(
        self,
        request_id: str,
        page_indices: list[int],
    ) -> None:
        self.coordinator.record_reflex_int4_landing_pages(
            request_id,
            page_indices,
        )

    def mark_reflex_int4_direct_landing_pages(
        self,
        request_id: str,
        page_indices: list[int],
        int4_block_ids: list[int],
    ) -> None:
        self.coordinator.mark_reflex_int4_direct_landing_pages(
            request_id,
            page_indices,
            int4_block_ids,
        )

    def release_reflex_int4_landing_blocks(self, request_id: str) -> None:
        self.coordinator.release_reflex_int4_landing_blocks(request_id)

    def commit_reflex_int4_landing_pages(
        self,
        request_id: str,
        page_indices: list[int],
        int4_block_ids: list[int],
    ) -> int:
        return self.coordinator.commit_reflex_int4_landing_pages(
            request_id,
            page_indices,
            int4_block_ids,
        )

    def recover_reflex_int4_pages(
        self,
        request_id: str,
        page_indices: list[int],
    ) -> int:
        return self.coordinator.recover_reflex_int4_pages(
            request_id,
            page_indices,
        )

    def promote_reflex_recoverable_pages(
        self,
        *,
        max_pages: int,
        prefill_page_risks_by_request: dict[str, list[float]] | None = None,
        remaining_decode_tokens_by_request: dict[str, int] | None = None,
        min_remaining_decode_tokens: int = 16,
    ) -> int:
        return self.coordinator.promote_reflex_recoverable_pages(
            max_pages=max_pages,
            prefill_page_risks_by_request=prefill_page_risks_by_request,
            remaining_decode_tokens_by_request=remaining_decode_tokens_by_request,
            min_remaining_decode_tokens=min_remaining_decode_tokens,
        )

    def get_reflex_page_runtime_descriptors(
        self,
        request_id: str,
        **kwargs,
    ) -> list[KVPageRuntimeDescriptor]:
        return self.coordinator.get_reflex_page_runtime_descriptors(
            request_id,
            **kwargs,
        )

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled.

        Args:
            request: The request to cache the blocks.
            num_computed_tokens: The number of computed tokens, including tokens
                that are already cached and tokens to be cached.
        """
        if self.enable_caching:
            self.coordinator.cache_blocks(request, num_computed_tokens)

    def create_kv_cache_blocks(
        self,
        blocks: tuple[list[KVCacheBlock], ...],
        block_id_overrides: tuple[list[int], ...] | None = None,
    ) -> KVCacheBlocks:
        # Only create new KVCacheBlocks for non-empty blocks
        return (
            KVCacheBlocks(blocks, block_id_overrides)
            if any(blocks)
            else self.empty_kv_cache_blocks
        )

    def take_new_block_ids(self) -> list[int]:
        """Drain and return new attention block IDs for zeroing."""
        ids: list[int] = []
        for mgr in self.coordinator.single_type_managers:
            ids.extend(mgr.take_new_block_ids())
        return ids

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
        prefill_page_risks_by_request: dict[str, list[float]] | None = None,
        compressible_pages_by_request: dict[str, set[int]] | None = None,
        copy_on_demote_pages_by_request: dict[str, set[int]] | None = None,
        recovery_shadow_pages_by_request: dict[str, set[int]] | None = None,
        recovery_shadow_pages_per_request: int = 0,
        dry_run: bool = False,
    ) -> int:
        return self.coordinator.plan_reflex_int4_demotions(
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
            cache_scope=cache_scope,
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
            recovery_shadow_pages_by_request=recovery_shadow_pages_by_request,
            recovery_shadow_pages_per_request=recovery_shadow_pages_per_request,
            dry_run=dry_run,
        )

    def get_last_reflex_int4_candidate_capacity(self) -> int:
        return self.coordinator.get_last_reflex_int4_candidate_capacity()

    def get_last_reflex_int4_candidate_breakdown(
        self,
    ) -> ReflexCandidateBreakdown:
        return self.coordinator.get_last_reflex_int4_candidate_breakdown()

    def get_reflex_precision_state_counts(
        self,
        request_id: str | None = None,
    ) -> dict[str, int]:
        return self.coordinator.get_reflex_precision_state_counts(request_id)

    def take_reflex_int4_demotions(self) -> list[ReflexDemotion]:
        return self.coordinator.take_reflex_int4_demotions()

    def take_reflex_int4_recoveries(self) -> list[ReflexRecovery]:
        return self.coordinator.take_reflex_int4_recoveries()

    def new_step_starts(self) -> None:
        """Called when a new step is started."""
        self.coordinator.new_step_starts()
