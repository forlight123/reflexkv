# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import logging
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import IntEnum
from typing import TYPE_CHECKING, Any

import httpx
import msgspec
import numpy as np
import torch
import zmq
import zmq.asyncio

from vllm import envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import (
    EngineId,
    TpKVTopology,
    get_current_attn_backend,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_utils import (
    MooncakeBootstrapServer,
    RegisterWorkerPayload,
)
from vllm.distributed.parallel_state import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    is_local_first_rank,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip, make_zmq_path, make_zmq_socket
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.v1.attention.ops.reflex_int4_codec import get_reflex_int4_codec
from vllm.v1.core.reflex_prefill_metadata import (
    get_reflex_prefill_metadata_recorder,
)
from vllm.v1.core.precision_kv.chunks import (
    RemoteKVChunk,
    normalize_remote_chunk_tokens,
    plan_remote_kv_chunk,
    remote_chunking_enabled,
    write_remote_kv_chunk_contract,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)

try:
    from mooncake.engine import TransferEngine
except ImportError:
    logger.warning(
        "Please install mooncake by following the instructions at "
        "https://github.com/kvcache-ai/Mooncake/blob/main/doc/en/build.md "
        "to run VLLM with MooncakeTransferEngine."
    )
    TransferEngine = None

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

ReqId = str  # Internal scheduler request ID
TransferId = str  # KV transfer coordination ID (shared by P/D)


@dataclass(frozen=True)
class TransferRegion:
    base_addr: int
    block_len: int
    kv_block_len: int


def _get_tp_ratio(local_tp_size: int, remote_tp_size: int) -> int:
    """Return the TP ratio used by heterogeneous TP transfer planning.

    Positive values mean one local rank maps into a larger remote KV region.
    Negative values mean one local rank must gather from multiple remote KV
    regions.
    """
    if local_tp_size >= remote_tp_size:
        assert local_tp_size % remote_tp_size == 0, (
            f"Local tensor parallel size {local_tp_size} is not divisible "
            f"by remote tensor parallel size {remote_tp_size}."
        )
        return local_tp_size // remote_tp_size

    assert remote_tp_size % local_tp_size == 0, (
        f"Remote tensor parallel size {remote_tp_size} is not divisible "
        f"by local tensor parallel size {local_tp_size}."
    )
    return -(remote_tp_size // local_tp_size)


def _expand_transfer_regions(
    base_addrs: list[int],
    block_lens: list[int],
    is_kv_layout_blocks_first: bool,
) -> list[TransferRegion]:
    """Expand registered KV tensors into the regions transferred by Mooncake."""
    assert len(base_addrs) == len(block_lens), (
        "Mooncake transfer regions require matching numbers of base addresses "
        f"and block lengths, got {len(base_addrs)} and {len(block_lens)}."
    )
    regions: list[TransferRegion] = []
    for base_addr, block_len in zip(base_addrs, block_lens):
        kv_block_len = block_len // 2 if is_kv_layout_blocks_first else block_len
        regions.append(
            TransferRegion(
                base_addr=base_addr,
                block_len=block_len,
                kv_block_len=kv_block_len,
            )
        )
        if is_kv_layout_blocks_first:
            regions.append(
                TransferRegion(
                    base_addr=base_addr + kv_block_len,
                    block_len=block_len,
                    kv_block_len=kv_block_len,
                )
            )
    return regions


def _compute_sender_transfer_plan(
    local_tp_rank: int,
    local_tp_size: int,
    remote_tp_rank: int,
    remote_tp_size: int,
    local_kv_block_len: int,
    remote_kv_block_len: int,
    producer_cache_replicated: bool,
) -> tuple[bool, int, int, int]:
    """Plan one producer-rank to one consumer-rank copy for heterogeneous TP."""
    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)

    if tp_ratio == 1:
        return True, 0, 0, local_kv_block_len

    if tp_ratio > 0:
        if producer_cache_replicated:
            return local_tp_rank % tp_ratio == 0, 0, 0, local_kv_block_len
        return (
            True,
            0,
            (local_tp_rank % tp_ratio) * local_kv_block_len,
            local_kv_block_len,
        )

    if producer_cache_replicated:
        return True, 0, 0, local_kv_block_len

    ratio_abs = -tp_ratio
    return (
        True,
        (remote_tp_rank % ratio_abs) * remote_kv_block_len,
        0,
        remote_kv_block_len,
    )


def _can_coalesce_block_transfers(
    local_region_block_len: int,
    remote_region_block_len: int,
    src_region_offset: int,
    dst_region_offset: int,
    transfer_len: int,
) -> bool:
    """Whether a contiguous block group can be emitted as one larger copy."""
    return (
        src_region_offset == 0
        and dst_region_offset == 0
        and transfer_len == local_region_block_len
        and transfer_len == remote_region_block_len
    )


def _validate_asymmetric_region_lengths(
    local_regions: list[TransferRegion],
    remote_regions: list[TransferRegion],
    local_tp_size: int,
    remote_tp_size: int,
    producer_cache_replicated: bool,
) -> str | None:
    """Validate transfer-region metadata for a fixed producer/consumer pair.

    This checks registered KV regions, not per-request block counts. A region
    corresponds to one registered KV tensor, or one K/V half after expansion
    for layouts that store K and V together.
    """
    if len(local_regions) != len(remote_regions):
        return (
            "Mooncake asymmetric TP requires matching KV region counts between "
            "producer and consumer."
        )

    if producer_cache_replicated:
        return None

    tp_ratio = _get_tp_ratio(local_tp_size, remote_tp_size)
    for idx, (local_region, remote_region) in enumerate(
        zip(local_regions, remote_regions)
    ):
        if tp_ratio == 1:
            if local_region.kv_block_len != remote_region.kv_block_len:
                return (
                    "Mooncake KV region length mismatch for homogeneous TP at "
                    f"region {idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}."
                )
        elif tp_ratio > 0:
            if remote_region.kv_block_len != local_region.kv_block_len * tp_ratio:
                return (
                    "Mooncake destination KV region length does not match the "
                    "producer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )
        else:
            ratio_abs = -tp_ratio
            if local_region.kv_block_len != remote_region.kv_block_len * ratio_abs:
                return (
                    "Mooncake source KV region length does not match the "
                    "consumer TP ratio at region "
                    f"{idx}: local={local_region.kv_block_len}, "
                    f"remote={remote_region.kv_block_len}, tp_ratio={tp_ratio}."
                )

    return None


def _get_tensor_dense_flag(tensor: torch.Tensor) -> bool | None:
    is_dense = getattr(tensor, "is_non_overlapping_and_dense", None)
    if callable(is_dense):
        return bool(is_dense())
    return None


class MooncakeXferMetadata(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    remote_hostname: str
    remote_port: int
    remote_tp_size: int
    remote_tp_rank: int
    req_blocks: dict[ReqId, tuple[TransferId, list[int]]]
    kv_caches_base_addr: list[int]
    block_lens: list[int]
    reflex_int4_landing_pages: dict[ReqId, list[int]] | None = None
    reflex_int4_landing_block_ids: dict[ReqId, list[int]] | None = None
    reflex_int4_landing_planned_blocks: dict[ReqId, int] | None = None
    reflex_int4_direct_landing: dict[ReqId, bool] | None = None
    reflex_int4_kv_caches_base_addr: list[int] | None = None
    reflex_int4_block_lens: list[int] | None = None
    reflex_remote_chunk_ids: dict[ReqId, int] | None = None
    reflex_remote_chunk_token_ranges: dict[ReqId, tuple[int, int]] | None = None
    reflex_remote_chunk_page_ranges: dict[ReqId, tuple[int, int]] | None = None
    reflex_remote_chunk_is_last: dict[ReqId, bool] | None = None


class MooncakeXferResponseStatus(IntEnum):
    # Transfer finished
    FINISH = 0
    # Continue to receive
    CONTINUE = 1
    # Something wrong, see err_msg
    ERROR = 2


class MooncakeXferResponse(
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
):
    status: MooncakeXferResponseStatus
    ok_reqs: list[ReqId] | None = None
    err_reqs: list[ReqId] | None = None
    err_msg: str | None = None


@dataclass
class MooncakeConnectorWorkerMetadata(KVConnectorWorkerMetadata):
    reflex_page_risks_by_request: dict[ReqId, list[float]]
    reflex_int4_materialized_landing_req_ids: set[ReqId] = field(
        default_factory=set
    )

    def aggregate(
        self, other: KVConnectorWorkerMetadata
    ) -> "MooncakeConnectorWorkerMetadata":
        if not isinstance(other, MooncakeConnectorWorkerMetadata):
            return self
        merged = {
            req_id: list(scores)
            for req_id, scores in self.reflex_page_risks_by_request.items()
        }
        for req_id, other_scores in other.reflex_page_risks_by_request.items():
            current_scores = merged.get(req_id)
            if current_scores is None:
                merged[req_id] = list(other_scores)
                continue
            score_count = min(len(current_scores), len(other_scores))
            for idx in range(score_count):
                current_scores[idx] = max(current_scores[idx], other_scores[idx])
            if len(other_scores) > len(current_scores):
                current_scores.extend(other_scores[len(current_scores) :])
        return MooncakeConnectorWorkerMetadata(
            reflex_page_risks_by_request=merged,
            reflex_int4_materialized_landing_req_ids=(
                set(self.reflex_int4_materialized_landing_req_ids)
                | set(other.reflex_int4_materialized_landing_req_ids)
            ),
        )


@dataclass
class PullReqMeta:
    d_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    remote_engine_id: EngineId
    remote_bootstrap_addr: str
    reflex_int4_landing_pages: list[int] | None = None
    reflex_int4_landing_block_ids: list[int] | None = None
    reflex_int4_landing_source_block_ids: list[int] | None = None
    reflex_int4_landing_planned_blocks: int = 0
    reflex_int4_direct_landing: bool = False
    reflex_remote_chunk_id: int | None = None
    reflex_remote_chunk_token_start: int = 0
    reflex_remote_chunk_token_end: int = 0
    reflex_remote_chunk_page_start: int = 0
    reflex_remote_chunk_page_end: int = 0
    reflex_remote_chunk_is_last: bool = False
    # Set expire time to avoid infinitely sending requests.
    expire_time: float = float("inf")
    # Designed for one D pairing to multiple P
    pull_tasks_count: int = 0


@dataclass
class SendBlockMeta:
    p_req_id: ReqId
    transfer_id: TransferId
    local_block_ids: list[int]
    ready: asyncio.Event
    reflex_remote_chunk_id: int | None = None
    release_after_send: bool = True
    expire_time: float = float("inf")
    need_send: int = 0
    sent: int = 0
    sending: int = 0


class MooncakeConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        # Use (engine_id, dp_rank) to group reqs with same dp.
        # See comments in MooncakeBootstrapServer.
        self.reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]] = defaultdict(dict)
        self.reqs_to_send: dict[ReqId, tuple[TransferId, list[int]]] = {}
        self.reqs_not_processed: set[TransferId] = set()
        self.reflex_remote_chunk_ids: dict[ReqId, int] | None = None
        self.reflex_remote_chunk_token_ranges: (
            dict[ReqId, tuple[int, int]] | None
        ) = None
        self.reflex_remote_chunk_page_ranges: (
            dict[ReqId, tuple[int, int]] | None
        ) = None
        self.reflex_remote_chunk_is_last: dict[ReqId, bool] | None = None
        self.reflex_remote_chunk_defer_ready: dict[ReqId, bool] | None = None
        self.reflex_remote_chunk_release_after_send: (
            dict[ReqId, bool] | None
        ) = None

    @staticmethod
    def _remote_chunk_fields(
        kv_transfer_params: dict[str, Any],
    ) -> tuple[int | None, int, int, int, int, bool]:
        raw_chunk_id = kv_transfer_params.get("reflex_remote_chunk_id")
        chunk_id = int(raw_chunk_id) if raw_chunk_id is not None else None
        return (
            chunk_id,
            int(kv_transfer_params.get("reflex_remote_chunk_token_start", 0)),
            int(kv_transfer_params.get("reflex_remote_chunk_token_end", 0)),
            int(kv_transfer_params.get("reflex_remote_chunk_page_start", 0)),
            int(kv_transfer_params.get("reflex_remote_chunk_page_end", 0)),
            bool(kv_transfer_params.get("reflex_remote_chunk_is_last", False)),
        )

    def _record_remote_chunk_contract(
        self,
        request_id: ReqId,
        kv_transfer_params: dict[str, Any],
    ) -> None:
        if not remote_chunking_enabled(kv_transfer_params):
            return
        (
            chunk_id,
            token_start,
            token_end,
            page_start,
            page_end,
            is_last,
        ) = self._remote_chunk_fields(kv_transfer_params)
        if chunk_id is None:
            return
        if self.reflex_remote_chunk_ids is None:
            self.reflex_remote_chunk_ids = {}
        if self.reflex_remote_chunk_token_ranges is None:
            self.reflex_remote_chunk_token_ranges = {}
        if self.reflex_remote_chunk_page_ranges is None:
            self.reflex_remote_chunk_page_ranges = {}
        if self.reflex_remote_chunk_is_last is None:
            self.reflex_remote_chunk_is_last = {}
        self.reflex_remote_chunk_ids[request_id] = chunk_id
        self.reflex_remote_chunk_token_ranges[request_id] = (
            token_start,
            token_end,
        )
        self.reflex_remote_chunk_page_ranges[request_id] = (
            page_start,
            page_end,
        )
        self.reflex_remote_chunk_is_last[request_id] = is_last

    def add_new_req(
        self,
        request_id: ReqId,
        local_block_ids: list[int],
        kv_transfer_params: dict[str, Any],
        load_remote_cache: bool = True,
    ):
        transfer_id = kv_transfer_params["transfer_id"]
        if load_remote_cache:
            remote_engine_id = kv_transfer_params["remote_engine_id"]
            (
                chunk_id,
                token_start,
                token_end,
                page_start,
                page_end,
                is_last,
            ) = self._remote_chunk_fields(kv_transfer_params)
            raw_landing_pages = kv_transfer_params.get("reflex_int4_landing_pages")
            landing_pages = (
                [int(page_idx) for page_idx in raw_landing_pages]
                if isinstance(raw_landing_pages, (list, tuple, set))
                else None
            )
            raw_landing_block_ids = kv_transfer_params.get(
                "reflex_int4_landing_block_ids"
            )
            landing_block_ids = (
                [int(block_id) for block_id in raw_landing_block_ids]
                if isinstance(raw_landing_block_ids, (list, tuple, set))
                else None
            )
            raw_landing_source_block_ids = kv_transfer_params.get(
                "reflex_int4_landing_source_block_ids"
            )
            landing_source_block_ids = (
                [int(block_id) for block_id in raw_landing_source_block_ids]
                if isinstance(raw_landing_source_block_ids, (list, tuple, set))
                else None
            )
            self.reqs_to_recv[remote_engine_id][request_id] = PullReqMeta(
                d_req_id=request_id,
                local_block_ids=local_block_ids,
                remote_engine_id=remote_engine_id,
                remote_bootstrap_addr=kv_transfer_params["remote_bootstrap_addr"],
                transfer_id=transfer_id,
                reflex_int4_landing_pages=landing_pages,
                reflex_int4_landing_block_ids=landing_block_ids,
                reflex_int4_landing_source_block_ids=landing_source_block_ids,
                reflex_int4_landing_planned_blocks=int(
                    kv_transfer_params.get(
                        "reflex_int4_landing_planned_blocks",
                        0,
                    )
                ),
                reflex_int4_direct_landing=bool(
                    kv_transfer_params.get("reflex_int4_direct_landing", False)
                ),
                reflex_remote_chunk_id=chunk_id,
                reflex_remote_chunk_token_start=token_start,
                reflex_remote_chunk_token_end=token_end,
                reflex_remote_chunk_page_start=page_start,
                reflex_remote_chunk_page_end=page_end,
                reflex_remote_chunk_is_last=is_last,
            )
            self._record_remote_chunk_contract(request_id, kv_transfer_params)
        else:
            self.reqs_to_send[request_id] = (transfer_id, local_block_ids)
            self._record_remote_chunk_contract(request_id, kv_transfer_params)
            if remote_chunking_enabled(kv_transfer_params):
                if self.reflex_remote_chunk_defer_ready is None:
                    self.reflex_remote_chunk_defer_ready = {}
                if self.reflex_remote_chunk_release_after_send is None:
                    self.reflex_remote_chunk_release_after_send = {}
                self.reflex_remote_chunk_defer_ready[request_id] = bool(
                    kv_transfer_params.get("reflex_remote_chunk_defer_ready", False)
                )
                self.reflex_remote_chunk_release_after_send[request_id] = bool(
                    kv_transfer_params.get(
                        "reflex_remote_chunk_release_after_send",
                        False,
                    )
                )


class MooncakeConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        assert vllm_config.kv_transfer_config is not None
        assert vllm_config.kv_transfer_config.engine_id is not None
        self.engine_id: EngineId = vllm_config.kv_transfer_config.engine_id

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler: MooncakeConnectorScheduler | None = (
                MooncakeConnectorScheduler(vllm_config, self.engine_id)
            )
            self.connector_worker: MooncakeConnectorWorker | None = None
        elif role == KVConnectorRole.WORKER:
            self.connector_scheduler = None
            self.connector_worker = MooncakeConnectorWorker(vllm_config, self.engine_id)

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig):
        if vllm_config.model_config is None:
            # This fallback mostly exists for unit tests that instantiate the
            # connector without a fully populated model config.
            logger.warning_once(
                "Unable to detect current VLLM config. "
                "Fallback to default kv cache layout."
            )
            return None
        if vllm_config.model_config.use_mla:
            return None
        logger.info_once(
            "MooncakeConnector setting KV cache layout to HND for "
            "heterogeneous TP-safe KV transfer."
        )
        return "HND"

    ############################################################
    # Scheduler Side Methods
    ############################################################

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def update_reflex_remote_decode_chunk_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_scheduled_tokens: int,
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_reflex_remote_decode_chunk_after_alloc(
            request,
            blocks,
            num_scheduled_tokens,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def pop_reflex_int4_materialized_landing_req(self, request_id: ReqId) -> bool:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.pop_reflex_int4_materialized_landing_req(
            request_id
        )

    ############################################################
    # Worker Side Methods
    ############################################################
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def register_reflex_int4_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        assert self.connector_worker is not None
        self.connector_worker.register_reflex_int4_kv_caches(kv_caches)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        """Get the finished recving and sending requests."""
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.build_connector_worker_meta()

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, MooncakeConnectorMetadata)
        self.connector_worker.start_load_kv(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """MooncakeConnector does not do layerwise saving."""
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs,
    ) -> None:
        """MooncakeConnector does not save explicitly."""
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        self.connector_worker.wait_for_save()


class MooncakeConnectorScheduler:
    """Implementation of Scheduler side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        self.vllm_config = vllm_config

        assert vllm_config.kv_transfer_config
        self.is_kv_producer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_producer"
        )
        self.is_kv_consumer: bool = (
            vllm_config.kv_transfer_config.kv_role == "kv_consumer"
        )
        logger.info("Initializing Mooncake Transfer Engine Scheduler %s", engine_id)

        # Requests that need to start recv/send.
        # New requests are added by update_state_after_alloc in
        # the scheduler. Used to make metadata passed to Worker.
        self._reqs_need_recv: dict[ReqId, tuple[Request, list[int]]] = {}
        self._reqs_need_send: dict[ReqId, tuple[Request, list[int]]] = {}
        # Reqs to remove from processed set because they're not to send after
        # remote prefill or aborted.
        self._reqs_not_processed: set[TransferId] = set()
        self._reflex_page_risks_by_request: dict[ReqId, list[float]] = {}
        self._reflex_int4_materialized_landing_req_ids: set[ReqId] = set()

    def _remote_chunk_tokens(self, params: dict[str, Any]) -> int:
        return int(params.get("reflex_remote_chunk_tokens", 512))

    def _prompt_token_count(self, request: "Request") -> int:
        prompt_tokens = getattr(request, "num_prompt_tokens", None)
        if prompt_tokens is not None:
            return int(prompt_tokens)
        token_ids = request.prompt_token_ids or []
        return len(token_ids)

    def _plan_next_remote_chunk(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> RemoteKVChunk | None:
        params = request.kv_transfer_params
        if not isinstance(params, dict) or not remote_chunking_enabled(params):
            return None
        return plan_remote_kv_chunk(
            request_id=request.request_id,
            prompt_token_count=self._prompt_token_count(request),
            num_computed_tokens=num_computed_tokens,
            block_size=self.vllm_config.cache_config.block_size,
            chunk_tokens=self._remote_chunk_tokens(params),
        )

    def _make_producer_remote_chunk(
        self,
        request: "Request",
        num_scheduled_tokens: int,
    ) -> RemoteKVChunk | None:
        params = request.kv_transfer_params
        if not isinstance(params, dict) or not remote_chunking_enabled(params):
            return None
        prompt_token_count = self._prompt_token_count(request)
        token_start = max(0, int(getattr(request, "num_computed_tokens", 0)))
        if token_start >= prompt_token_count:
            return None
        token_end = min(prompt_token_count, token_start + num_scheduled_tokens)
        if token_end <= token_start:
            return None
        block_size = max(1, int(self.vllm_config.cache_config.block_size))
        normalized_chunk_tokens = normalize_remote_chunk_tokens(
            self._remote_chunk_tokens(params),
            block_size,
        )
        page_start = token_start // block_size
        page_end = (token_end + block_size - 1) // block_size
        return RemoteKVChunk(
            request_id=request.request_id,
            chunk_id=token_start // normalized_chunk_tokens,
            token_start=token_start,
            token_end=token_end,
            page_start=page_start,
            page_end=page_end,
            block_size=block_size,
            is_last_chunk=token_end >= prompt_token_count,
        )

    def _make_final_remote_chunk(self, request: "Request") -> RemoteKVChunk | None:
        params = request.kv_transfer_params
        if not isinstance(params, dict) or not remote_chunking_enabled(params):
            return None
        prompt_token_count = self._prompt_token_count(request)
        if prompt_token_count <= 0:
            return None
        block_size = max(1, int(self.vllm_config.cache_config.block_size))
        normalized_chunk_tokens = normalize_remote_chunk_tokens(
            self._remote_chunk_tokens(params),
            block_size,
        )
        final_start = ((prompt_token_count - 1) // normalized_chunk_tokens) * (
            normalized_chunk_tokens
        )
        return plan_remote_kv_chunk(
            request_id=request.request_id,
            prompt_token_count=prompt_token_count,
            num_computed_tokens=final_start,
            block_size=block_size,
            chunk_tokens=normalized_chunk_tokens,
        )

    @staticmethod
    def _block_id_from_page_entry(block: Any) -> int | None:
        if getattr(block, "is_null", False):
            return None
        block_id = block if isinstance(block, int) else getattr(block, "block_id", None)
        if block_id is None:
            return None
        block_id = int(block_id)
        return block_id if block_id >= 0 else None

    def _local_block_ids_for_remote_prefill(
        self,
        blocks: "KVCacheBlocks",
        params: dict[str, Any],
        num_external_tokens: int,
    ) -> list[int]:
        if num_external_tokens <= 0:
            return []
        if not remote_chunking_enabled(params):
            return blocks.get_unhashed_block_ids()

        page_start = int(params.get("reflex_remote_chunk_page_start", 0))
        page_end = int(params.get("reflex_remote_chunk_page_end", 0))
        if page_end <= page_start:
            return blocks.get_unhashed_block_ids()

        block_groups = getattr(blocks, "blocks", ())
        if len(block_groups) != 1:
            return blocks.get_unhashed_block_ids()

        group_blocks = block_groups[0]
        local_block_ids: list[int] = []
        for page_idx in range(max(0, page_start), min(page_end, len(group_blocks))):
            block_id = self._block_id_from_page_entry(group_blocks[page_idx])
            if block_id is not None:
                local_block_ids.append(block_id)
        if params.get("reflex_int4_direct_landing"):
            return local_block_ids
        if len(local_block_ids) == page_end - page_start:
            return local_block_ids

        unhashed_block_ids = blocks.get_unhashed_block_ids()
        return unhashed_block_ids[page_start:page_end]

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int, bool]:
        """
        For remote prefill, pull all prompt blocks from remote
        asynchronously relative to engine execution.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
        Returns:
            * the number of tokens that can be loaded from the
              external KV cache beyond what is already computed.
            * true if the external KV cache tokens will be loaded
              asynchronously (between scheduler steps).
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector get_num_new_matched_tokens: "
            "num_computed_tokens=%s, kv_transfer_params=%s",
            num_computed_tokens,
            params,
        )

        if not params:
            return 0, False

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            chunk = self._plan_next_remote_chunk(request, num_computed_tokens)
            if chunk is not None:
                return chunk.num_tokens, True
            # Remote prefill without chunking: get all prompt blocks from remote.
            token_ids = request.prompt_token_ids or []
            count = len(token_ids) - num_computed_tokens
            if count > 0:
                return count, True

        # No remote prefill for this request.
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector update_state_after_alloc: "
            "req_id=%s num_external_tokens=%s, kv_transfer_params=%s",
            request.request_id,
            num_external_tokens,
            params,
        )

        if not params:
            return

        if params.get("do_remote_prefill"):
            assert not self.is_kv_producer
            if all(
                p in params
                for p in ("remote_engine_id", "remote_bootstrap_addr", "transfer_id")
            ):
                chunk = self._plan_next_remote_chunk(
                    request,
                    int(getattr(request, "num_computed_tokens", 0)),
                )
                if chunk is not None:
                    write_remote_kv_chunk_contract(params, chunk, role="decode")
                    params["reflex_remote_chunk_inflight"] = True
                # If remote_blocks and num_external_tokens = 0, we have
                # a full prefix cache hit on the D worker. We need to call
                # send_notif in _read_blocks to free the memory on the P.
                local_block_ids = self._local_block_ids_for_remote_prefill(
                    blocks,
                    params,
                    num_external_tokens,
                )
                raw_landing_pages = params.get("reflex_int4_landing_pages")
                if isinstance(raw_landing_pages, (list, tuple)):
                    source_block_ids: list[int] = []
                    block_groups = getattr(blocks, "blocks", ())
                    block_id_groups = blocks.get_block_ids()
                    if len(block_groups) == 1 and len(block_id_groups) == 1:
                        group_blocks = block_groups[0]
                        group_block_ids = block_id_groups[0]
                        for raw_page_idx in raw_landing_pages:
                            page_idx = int(raw_page_idx)
                            if page_idx < 0 or page_idx >= len(group_block_ids):
                                break
                            block = group_blocks[page_idx]
                            if getattr(block, "is_null", False):
                                break
                            if isinstance(block, int):
                                block_id = block
                            else:
                                block_id = int(group_block_ids[page_idx])
                            if block_id < 0:
                                break
                            source_block_ids.append(block_id)
                    if len(source_block_ids) == len(raw_landing_pages):
                        params["reflex_int4_landing_source_block_ids"] = (
                            source_block_ids
                        )
                    else:
                        params.pop("reflex_int4_landing_source_block_ids", None)
                # Get unhashed blocks to pull from remote.
                self._reqs_need_recv[request.request_id] = (request, local_block_ids)
            else:
                logger.warning(
                    "Got invalid KVTransferParams: %s. This "
                    "request will not utilize KVTransfer",
                    params,
                )
            # Only trigger 1 KV transfer per request.
            if remote_chunking_enabled(params):
                params["do_remote_prefill"] = not bool(
                    params.get("reflex_remote_chunk_is_last", False)
                )
            else:
                params["do_remote_prefill"] = False

        elif params.get("do_remote_decode"):
            assert not self.is_kv_consumer
            if remote_chunking_enabled(params):
                return
            if not params.get("transfer_id"):
                logger.warning("Missing transfer_id in kv_transfer_params from router!")
            else:
                # Add an empty list to worker to create event.
                self._reqs_need_send[request.request_id] = (request, [])

    def update_reflex_remote_decode_chunk_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_scheduled_tokens: int,
    ) -> None:
        params = request.kv_transfer_params
        if not isinstance(params, dict):
            return
        if not params.get("do_remote_decode") or not remote_chunking_enabled(params):
            return
        if not params.get("transfer_id"):
            logger.warning("Missing transfer_id in kv_transfer_params from router!")
            return
        chunk = self._make_producer_remote_chunk(request, num_scheduled_tokens)
        if chunk is None:
            return
        write_remote_kv_chunk_contract(params, chunk, role="prefill")
        params["reflex_remote_chunk_defer_ready"] = True
        params["reflex_remote_chunk_release_after_send"] = False
        self._reqs_need_send[request.request_id] = (
            request,
            blocks.get_unhashed_block_ids(),
        )
        logger.info(
            "ReFlexKV trace remote_chunk_send request=%s chunk_id=%d "
            "tokens=%d:%d pages=%d:%d release_after_send=False.",
            request.request_id,
            chunk.chunk_id,
            chunk.token_start,
            chunk.token_end,
            chunk.page_start,
            chunk.page_end,
        )

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        meta = MooncakeConnectorMetadata()

        # Loop through scheduled reqs and convert to PullReqMeta.
        if not self.is_kv_producer:
            for req_id, (req, block_ids) in self._reqs_need_recv.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                )
            self._reqs_need_recv.clear()

        if not self.is_kv_consumer:
            for req_id, (req, block_ids) in self._reqs_need_send.items():
                assert req.kv_transfer_params is not None
                meta.add_new_req(
                    request_id=req_id,
                    local_block_ids=block_ids,
                    kv_transfer_params=req.kv_transfer_params,
                    load_remote_cache=False,
                )
            self._reqs_need_send.clear()
            meta.reqs_not_processed = self._reqs_not_processed
            self._reqs_not_processed = set()

        return meta

    def update_connector_output(self, connector_output: KVConnectorOutput):
        worker_meta = connector_output.kv_connector_worker_meta
        if isinstance(worker_meta, MooncakeConnectorWorkerMetadata):
            if worker_meta.reflex_page_risks_by_request:
                logger.info(
                    "ReFlexKV trace page_metadata_receive requests=%d "
                    "pages=%d source=mooncake_worker_meta.",
                    len(worker_meta.reflex_page_risks_by_request),
                    sum(
                        len(scores)
                        for scores in worker_meta.reflex_page_risks_by_request.values()
                    ),
                )
            self._reflex_page_risks_by_request.update(
                worker_meta.reflex_page_risks_by_request
            )
            self._reflex_int4_materialized_landing_req_ids.update(
                worker_meta.reflex_int4_materialized_landing_req_ids
            )

    def pop_reflex_int4_materialized_landing_req(self, request_id: ReqId) -> bool:
        if request_id not in self._reflex_int4_materialized_landing_req_ids:
            return False
        self._reflex_int4_materialized_landing_req_ids.remove(request_id)
        return True

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Once a request is finished, determine whether request blocks
        should be freed now or will be sent asynchronously and freed later.
        """

        params = request.kv_transfer_params
        logger.debug(
            "MooncakeConnector request_finished, req_id=%s, request_status=%s, "
            "kv_transfer_params=%s",
            request.request_id,
            request.status,
            params,
        )
        if not params or not params.get("transfer_id"):
            self._reflex_page_risks_by_request.pop(request.request_id, None)
            return False, None

        if params.get("do_remote_prefill"):
            # If do_remote_prefill is still True when the request is finished,
            # update_state_after_alloc must not have been called (the request
            # must have been aborted before it was scheduled).
            # To avoid stranding the prefill blocks in the prefill instance,
            # we must add empty block_ids to _reqs_need_recv so that our
            # worker side will notify and free blocks in the prefill instance.
            assert not self.is_kv_producer
            self._reqs_need_recv[request.request_id] = (request, [])
            params["do_remote_prefill"] = False
            self._reflex_page_risks_by_request.pop(request.request_id, None)
            return False, None

        if not params.get("do_remote_decode"):
            self._reflex_page_risks_by_request.pop(request.request_id, None)
            return False, None

        assert not self.is_kv_consumer

        if request.status != RequestStatus.FINISHED_LENGTH_CAPPED:
            # Also include the case of a P/D Prefill request with immediate
            # block free (eg abort). Stop tracking this request.
            self._reqs_not_processed.add(params["transfer_id"])
            self._reflex_page_risks_by_request.pop(request.request_id, None)
            return False, None

        # TODO: check whether block_ids actually ever be 0. If not we could
        # remove the conditional below
        delay_free_blocks = len(block_ids) > 0

        if delay_free_blocks and remote_chunking_enabled(params):
            final_chunk = self._make_final_remote_chunk(request)
            if final_chunk is not None:
                write_remote_kv_chunk_contract(params, final_chunk, role="prefill")
                params["reflex_remote_chunk_defer_ready"] = False
                params["reflex_remote_chunk_release_after_send"] = True
                self._reqs_need_send[request.request_id] = (request, block_ids)
                logger.info(
                    "ReFlexKV trace remote_chunk_send request=%s chunk_id=%d "
                    "tokens=%d:%d pages=%d:%d release_after_send=True.",
                    request.request_id,
                    final_chunk.chunk_id,
                    final_chunk.token_start,
                    final_chunk.token_end,
                    final_chunk.page_start,
                    final_chunk.page_end,
                )
                reflex_page_risks = self._reflex_page_risks_by_request.pop(
                    request.request_id,
                    None,
                )
                kv_transfer_params = (
                    {"reflex_page_risks": reflex_page_risks}
                    if reflex_page_risks
                    else None
                )
                return True, kv_transfer_params

        if delay_free_blocks:
            self._reqs_need_send[request.request_id] = (request, block_ids)

        reflex_page_risks = self._reflex_page_risks_by_request.pop(
            request.request_id,
            None,
        )
        kv_transfer_params = (
            {"reflex_page_risks": reflex_page_risks}
            if reflex_page_risks
            else None
        )
        return delay_free_blocks, kv_transfer_params


class MooncakeConnectorWorker:
    """Implementation of Worker side methods"""

    def __init__(self, vllm_config: VllmConfig, engine_id: str):
        if TransferEngine is None:
            logger.error("Mooncake is not available")
            raise RuntimeError("Mooncake is not available")
        logger.info("Initializing Mooncake Transfer Engine worker %s", engine_id)

        self.vllm_config = vllm_config

        self.engine = TransferEngine()
        self.hostname = get_ip()

        assert (kv_transfer_config := vllm_config.kv_transfer_config)
        self.is_kv_producer: bool = kv_transfer_config.kv_role == "kv_producer"
        self.is_kv_consumer: bool = kv_transfer_config.kv_role == "kv_consumer"
        self.num_sender_workers = kv_transfer_config.kv_connector_extra_config.get(
            "num_workers", 10
        )
        # Create more tasks than workers to keep the thread pool saturated.
        # Tasks can await async events, so a surplus (2x is a robust heuristic)
        # prevents workers from idling.
        self.num_sender_tasks = self.num_sender_workers * 2
        protocol = kv_transfer_config.kv_connector_extra_config.get(  # type: ignore[union-attr]
            "mooncake_protocol", "rdma"
        )
        logger.info(
            "The Mooncake Transfer Engine is using %s as its protocol.", protocol
        )
        ret_value = self.engine.initialize(self.hostname, "P2PHANDSHAKE", protocol, "")
        if ret_value != 0:
            raise RuntimeError("Mooncake Transfer Engine initialization failed.")

        self.rpc_port = self.engine.get_rpc_port()

        logger.debug(
            "Mooncake Transfer Engine initialized at %s:%d",
            self.hostname,
            self.rpc_port,
        )

        self._remote_agents: dict[EngineId, dict[int, dict[int, str]]] = {}
        self._pending_bootstrap_queries: dict[str, asyncio.Event] = {}
        self.side_channel_port: int = 0  # we will bind it in register_kv_caches()
        self.engine_id: EngineId = engine_id
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_blocks = 0
        self.block_len_per_layer: list[int] = []
        self.seen_base_addresses: list[int] = []

        assert (parallel_config := vllm_config.parallel_config)
        dp_rank = parallel_config.data_parallel_index
        dp_local_rank = parallel_config.data_parallel_rank_local
        self.dp_rank = dp_local_rank if parallel_config.local_engines_only else dp_rank
        pp_size = vllm_config.parallel_config.pipeline_parallel_size
        if pp_size > 1:
            raise ValueError(
                "Mooncake Transfer Engine does not support pipeline parallelism yet."
            )
        self.pp_rank = get_pp_group().rank_in_group

        self.kv_caches_base_addr: list[int] = []
        self.device_kv_caches: dict[str, torch.Tensor] = {}
        self.reflex_int4_kv_caches: dict[str, torch.Tensor] = {}
        self.reflex_int4_kv_caches_base_addr: list[int] = []
        self.reflex_int4_block_len_per_layer: list[int] = []
        self.reflex_int4_num_blocks = 0
        self._reflex_int4_direct_scratch_cursor = 0
        self.reqs_need_send: dict[str, SendBlockMeta] = {}
        self._completed_send_keys: set[str] = set()
        self._deferred_ready_send_keys: set[str] = set()

        # For kv_both, we will act both prefiller and decoder.
        if not self.is_kv_consumer:
            # Background threads for sending kvcaches to D.
            self._sender_executor = ThreadPoolExecutor(
                max_workers=self.num_sender_workers,
                thread_name_prefix="vllm-mooncake-sender",
            )
            logger.debug(
                "Mooncake Prefiller: use %d workers to send kvcaches",
                self.num_sender_workers,
            )
            # An asyncio queue to buffer incoming requests for the sender
            self.sender_worker_queue = asyncio.Queue[tuple[bytes, bytes]]()
            self.sender_loop = asyncio.new_event_loop()
            # Background thread for processing new sending requests.
            self._sender_listener_t = threading.Thread(
                target=_async_loop, args=(self.sender_loop,), daemon=True
            )
            self._sender_listener_t.start()

            # Start bootstrap server on global rank 0.
            if should_launch_bootstrap_server(vllm_config):
                _, port = get_mooncake_bootstrap_addr(vllm_config)
                self.bootstrap_server = MooncakeBootstrapServer("0.0.0.0", port)
                self.bootstrap_server.start()

        if not self.is_kv_producer:
            self.receiver_loop = asyncio.new_event_loop()
            self._mooncake_receiver_t = threading.Thread(
                target=_async_loop, args=(self.receiver_loop,), daemon=True
            )
            self._mooncake_receiver_t.start()
            logger.debug("Mooncake Decoder: start receiver thread")

        self.finished_sending_reqs: set[ReqId] = set()
        self.finished_recving_reqs: set[ReqId] = set()
        self.reflex_int4_materialized_landing_reqs: set[ReqId] = set()
        self._pending_worker_meta: MooncakeConnectorWorkerMetadata | None = None

        self.block_size = vllm_config.cache_config.block_size
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.use_mla = self.model_config.use_mla

        # Get the attention backend from the first layer
        # NOTE (NickLucche) models with multiple backends are not supported yet
        backend = get_current_attn_backend(vllm_config)
        self.backend_name = backend.get_name()
        self.kv_cache_layout = get_kv_cache_layout()
        logger.debug("Detected attention backend %s", self.backend_name)
        logger.debug("Detected kv cache layout %s", self.kv_cache_layout)

        self._tp_size: dict[EngineId, int] = {self.engine_id: self.tp_size}
        self._block_size: dict[EngineId, int] = {self.engine_id: self.block_size}
        self.kv_topo = TpKVTopology(
            tp_rank=self.tp_rank,
            engine_id=self.engine_id,
            remote_tp_size=self._tp_size,  # shared state
            remote_block_size=self._block_size,  # shared state
            is_mla=self.use_mla,
            total_num_kv_heads=self.model_config.get_total_num_kv_heads(),
            attn_backends=[backend],
        )

        self.async_zmq_ctx = zmq.asyncio.Context()
        self._encoder = msgspec.msgpack.Encoder()
        self._xfer_meta_decoder = msgspec.msgpack.Decoder(MooncakeXferMetadata)
        self._xfer_resp_decoder = msgspec.msgpack.Decoder(MooncakeXferResponse)

    @staticmethod
    def _send_key(transfer_id: TransferId, chunk_id: int | None = None) -> str:
        if chunk_id is None:
            return transfer_id
        return f"{transfer_id}:reflex_chunk:{chunk_id}"

    @staticmethod
    def _chunk_id_for_req(
        meta: MooncakeXferMetadata,
        req_id: ReqId,
    ) -> int | None:
        chunk_ids = meta.reflex_remote_chunk_ids
        if not chunk_ids or req_id not in chunk_ids:
            return None
        return int(chunk_ids[req_id])

    def __del__(self):
        self.shutdown()

    def shutdown(self):
        """Cleanup background threads on destruction."""
        self.async_zmq_ctx.term()
        if not self.is_kv_consumer:
            self._sender_executor.shutdown(wait=False)
            if self.sender_loop.is_running():
                self.sender_loop.call_soon_threadsafe(self.sender_loop.stop)
                self._sender_listener_t.join()
            if should_launch_bootstrap_server(self.vllm_config) and hasattr(
                self, "bootstrap_server"
            ):
                self.bootstrap_server.shutdown()
        if not self.is_kv_producer and self.receiver_loop.is_running():
            self.receiver_loop.call_soon_threadsafe(self.receiver_loop.stop)
            self._mooncake_receiver_t.join()

    async def register_worker_with_bootstrap(self):
        host, port = get_mooncake_bootstrap_addr(self.vllm_config)
        url = make_zmq_path("http", host, port) + "/register"
        worker_addr = make_zmq_path("tcp", self.hostname, self.side_channel_port)
        payload = RegisterWorkerPayload(
            engine_id=self.engine_id,
            dp_rank=self.dp_rank,
            tp_rank=self.tp_rank,
            pp_rank=self.pp_rank,
            addr=worker_addr,
        )
        while True:
            try:
                async with httpx.AsyncClient() as client:
                    response = await client.post(url, json=payload.model_dump())
                    response.raise_for_status()
                logger.debug("Successfully registered with bootstrap server at %s", url)
                break
            except httpx.ConnectError:
                # Bootstrap server not ready, wait for a while and retry.
                await asyncio.sleep(1)
            except Exception as e:
                err_msg = (
                    e.response.text if isinstance(e, httpx.HTTPStatusError) else str(e)
                )
                logger.error(
                    "Error registering %s with bootstrap server: %s", payload, err_msg
                )
                raise e

    async def _mooncake_sender_listener(self, ready_event: threading.Event):
        """
        Background thread that listens for Mooncake requests, dispatches them
        to a thread pool, and sends acknowledgments upon completion.
        """

        sock = self.async_zmq_ctx.socket(zmq.ROUTER)
        self.side_channel_port = sock.bind_to_random_port(f"tcp://{self.hostname}")
        logger.debug(
            "Mooncake sender starting listening on path: tcp://%s:%d",
            self.hostname,
            self.side_channel_port,
        )

        await self.register_worker_with_bootstrap()

        # Create async worker tasks that process items from the queue
        sender_tasks = [
            asyncio.create_task(self._sender_worker(sock))
            for _ in range(self.num_sender_tasks)
        ]

        ready_event.set()

        try:
            while True:
                identity, metadata_bytes = await sock.recv_multipart()
                await self.sender_worker_queue.put((identity, metadata_bytes))
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake sender thread.")
        except Exception as e:
            logger.error("Error in Mooncake sender thread: %s. Exiting thread.", str(e))
        finally:
            # Clean up worker tasks
            for task in sender_tasks:
                task.cancel()
            await asyncio.gather(*sender_tasks, return_exceptions=True)
            sock.close()

    async def _sender_worker(self, sock: zmq.asyncio.Socket):
        while True:
            try:
                identity, metadata_bytes = await self.sender_worker_queue.get()
                try:
                    metadata = self._xfer_meta_decoder.decode(metadata_bytes)
                    await self.send_kv_to_decode(identity, sock, metadata)
                except Exception as e:
                    logger.error("Error processing Mooncake xfer request: %s", e)
                    error_response = MooncakeXferResponse(
                        status=MooncakeXferResponseStatus.ERROR, err_msg=str(e)
                    )
                    await sock.send_multipart(
                        (identity, self._encoder.encode(error_response))
                    )
                finally:
                    self.sender_worker_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in _sender_worker: %s", e)

    async def send_kv_to_decode(
        self, identity: bytes, sock: zmq.asyncio.Socket, meta: MooncakeXferMetadata
    ):
        pending_reqs: dict[ReqId, SendBlockMeta] = {}
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks(meta.remote_tp_size)
        if meta.remote_tp_rank not in remote_tp_ranks:
            # This D worker does not pair with the P worker.
            msg = (
                "This D tp_rank "
                f"{meta.remote_tp_rank} is not paired with P tp_rank "
                f"{self.tp_rank}; expected one of {remote_tp_ranks}."
            )
            logger.error(msg)
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        local_regions = self._get_transfer_regions(
            self.kv_caches_base_addr, self.block_len_per_layer
        )
        remote_regions = self._get_transfer_regions(
            meta.kv_caches_base_addr, meta.block_lens
        )
        validation_err = _validate_asymmetric_region_lengths(
            local_regions=local_regions,
            remote_regions=remote_regions,
            local_tp_size=self.tp_size,
            remote_tp_size=meta.remote_tp_size,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )
        if validation_err is not None:
            response = MooncakeXferResponse(
                status=MooncakeXferResponseStatus.ERROR,
                err_msg=validation_err,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))
            return
        for d_req_id, (transfer_id, _) in meta.req_blocks.items():
            chunk_id = self._chunk_id_for_req(meta, d_req_id)
            send_key = self._send_key(transfer_id, chunk_id)
            if send_key not in self.reqs_need_send:
                # This req is not enqueued in P side yet, create it here.
                self.reqs_need_send[send_key] = SendBlockMeta(
                    p_req_id="",
                    transfer_id=transfer_id,
                    local_block_ids=[],
                    ready=asyncio.Event(),
                    reflex_remote_chunk_id=chunk_id,
                    release_after_send=chunk_id is None,
                )
            send_meta = self.reqs_need_send[send_key]
            pending_reqs[d_req_id] = send_meta

        async def wait_and_ret(
            d_req_id: ReqId, send_meta: SendBlockMeta
        ) -> tuple[ReqId, SendBlockMeta]:
            await send_meta.ready.wait()
            return d_req_id, send_meta

        wait_tasks = [
            asyncio.create_task(wait_and_ret(d_req_id, send_meta))
            for d_req_id, send_meta in pending_reqs.items()
        ]

        while wait_tasks:
            done, pending = await asyncio.wait(
                wait_tasks,
                timeout=envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not done:
                # Timeout, abort all pending requests.
                for task in wait_tasks:
                    task.cancel()
                logger.warning(
                    "Timeout waiting for P side ready: %s", list(pending_reqs)
                )
                response = MooncakeXferResponse(
                    status=MooncakeXferResponseStatus.FINISH,
                    err_reqs=list(pending_reqs),
                    err_msg="Timeout waiting for P side ready.",
                )
                await sock.send_multipart((identity, self._encoder.encode(response)))
                break

            wait_tasks = list(pending)
            response_status = (
                MooncakeXferResponseStatus.CONTINUE
                if wait_tasks
                else MooncakeXferResponseStatus.FINISH
            )
            ready_reqs: list[tuple[ReqId, SendBlockMeta]] = []
            for task in done:
                d_req_id, send_meta = task.result()
                del pending_reqs[d_req_id]
                # Do we still in reqs_need_send (not expired)?
                send_key = self._send_key(
                    send_meta.transfer_id,
                    send_meta.reflex_remote_chunk_id,
                )
                if send_key in self.reqs_need_send:
                    # Mark it sending to avoid expiration.
                    send_meta.sending += 1
                    if not send_meta.need_send:
                        self.resolve_need_send(send_meta, remote_tp_ranks)
                    ready_reqs.append((d_req_id, send_meta))
                else:
                    # Otherwise (expired, very unlikely), just forget it.
                    logger.warning(
                        "Request %s expired before sending on P side.", d_req_id
                    )

            (
                src_ptrs,
                dst_ptrs,
                lengths,
                err_reqs,
                err_msg,
            ) = await self._build_transfer_params(
                ready_reqs,
                meta,
                local_regions,
                remote_regions,
            )
            err_req_set = set(err_reqs)
            ok_ready_reqs = [
                (d_req_id, send_meta)
                for d_req_id, send_meta in ready_reqs
                if d_req_id not in err_req_set
            ]

            if src_ptrs:
                remote_session = f"{meta.remote_hostname}:{meta.remote_port}"
                ret_value = await self.sender_loop.run_in_executor(
                    self._sender_executor,
                    self._send_blocks,
                    remote_session,
                    src_ptrs,
                    dst_ptrs,
                    lengths,
                )

                if ret_value != 0:
                    transfer_err_msg = f"Mooncake transfer engine returned {ret_value}"
                    err_msg = (
                        transfer_err_msg
                        if err_msg is None
                        else f"{err_msg}; {transfer_err_msg}"
                    )
                    err_reqs = list(err_reqs)
                    for d_req_id, _ in ok_ready_reqs:
                        err_reqs.append(d_req_id)
                        err_req_set.add(d_req_id)
                    ok_ready_reqs = []

            for d_req_id, send_meta in ready_reqs:
                send_meta.sending -= 1

                if d_req_id in err_req_set:
                    continue

                send_meta.sent += 1
                send_key = self._send_key(
                    send_meta.transfer_id,
                    send_meta.reflex_remote_chunk_id,
                )
                if (
                    send_meta.sent == send_meta.need_send
                    and self.reqs_need_send.pop(send_key, None) is not None
                ):
                    if send_meta.release_after_send:
                        self.finished_sending_reqs.add(send_meta.p_req_id)
                    else:
                        self._completed_send_keys.add(send_key)

            response = MooncakeXferResponse(
                status=response_status,
                ok_reqs=[d_req_id for d_req_id, _ in ok_ready_reqs] or None,
                err_reqs=err_reqs or None,
                err_msg=err_msg,
            )
            await sock.send_multipart((identity, self._encoder.encode(response)))

    def resolve_need_send(self, send_meta: SendBlockMeta, remote_tp_ranks: list[int]):
        # Prepare for heterogeneous TP (one P pairs to multiple D)
        send_meta.need_send = len(remote_tp_ranks)
        logger.debug(
            "Mooncake request %s will be served by %d consumer TP workers: %s",
            send_meta.transfer_id,
            send_meta.need_send,
            remote_tp_ranks,
        )

    def _allocate_reflex_int4_direct_scratch_ids(self, count: int) -> list[int]:
        num_blocks = int(getattr(self, "reflex_int4_num_blocks", 0) or 0)
        if count <= 0 or num_blocks <= 0:
            return []
        start = self._reflex_int4_direct_scratch_cursor
        scratch_ids = [(start + offset) % num_blocks for offset in range(count)]
        self._reflex_int4_direct_scratch_cursor = (start + count) % num_blocks
        return scratch_ids

    def _quantize_reflex_int4_direct_landing_blocks(
        self,
        *,
        src_bf16_block_ids: list[int],
        scratch_int4_block_ids: list[int],
    ) -> bool:
        if not src_bf16_block_ids:
            return True
        if len(src_bf16_block_ids) != len(scratch_int4_block_ids):
            return False
        if not self.reflex_int4_kv_caches:
            return False
        first_layer = next(
            (
                layer_name
                for layer_name in self.device_kv_caches
                if layer_name in self.reflex_int4_kv_caches
            ),
            None,
        )
        if first_layer is None:
            return False
        device = self.device_kv_caches[first_layer].device
        src_block_ids = torch.tensor(
            src_bf16_block_ids,
            dtype=torch.int64,
            device=device,
        )
        scratch_block_ids = torch.tensor(
            scratch_int4_block_ids,
            dtype=torch.int64,
            device=device,
        )
        for layer_name, bf16_cache in self.device_kv_caches.items():
            int4_cache = self.reflex_int4_kv_caches.get(layer_name)
            if int4_cache is None:
                continue
            get_reflex_int4_codec().quantize_blocks_to_cache(
                bf16_cache,
                int4_cache,
                src_block_ids,
                scratch_block_ids,
            )
        return True

    def _append_reflex_int4_direct_landing_transfers(
        self,
        *,
        d_req_id: ReqId,
        src_bf16_block_ids: list[int],
        dst_int4_block_ids: list[int],
        agent_meta: MooncakeXferMetadata,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> str | None:
        if not src_bf16_block_ids and not dst_int4_block_ids:
            return None
        if len(src_bf16_block_ids) != len(dst_int4_block_ids):
            return "direct landing block count mismatch"
        local_int4_regions = self._get_transfer_regions(
            getattr(self, "reflex_int4_kv_caches_base_addr", []),
            getattr(self, "reflex_int4_block_len_per_layer", []),
        )
        remote_int4_regions = self._get_transfer_regions(
            agent_meta.reflex_int4_kv_caches_base_addr or [],
            agent_meta.reflex_int4_block_lens or [],
        )
        if not local_int4_regions or len(local_int4_regions) != len(remote_int4_regions):
            return "direct landing INT4 sidecar regions unavailable"
        scratch_int4_block_ids = self._allocate_reflex_int4_direct_scratch_ids(
            len(src_bf16_block_ids)
        )
        if len(scratch_int4_block_ids) != len(src_bf16_block_ids):
            return "direct landing INT4 scratch unavailable"
        if not self._quantize_reflex_int4_direct_landing_blocks(
            src_bf16_block_ids=src_bf16_block_ids,
            scratch_int4_block_ids=scratch_int4_block_ids,
        ):
            return "direct landing producer quantization failed"
        group_src_ids, group_dst_ids = group_concurrent_contiguous(
            scratch_int4_block_ids,
            dst_int4_block_ids,
        )
        for local_region, remote_region in zip(
            local_int4_regions,
            remote_int4_regions,
        ):
            should_transfer, src_region_offset, dst_region_offset, transfer_len = (
                self._get_sender_transfer_plan(
                    local_kv_block_len=local_region.kv_block_len,
                    remote_kv_block_len=remote_region.kv_block_len,
                    remote_tp_rank=agent_meta.remote_tp_rank,
                    remote_tp_size=agent_meta.remote_tp_size,
                )
            )
            if not should_transfer:
                continue
            can_coalesce = _can_coalesce_block_transfers(
                local_region_block_len=local_region.block_len,
                remote_region_block_len=remote_region.block_len,
                src_region_offset=src_region_offset,
                dst_region_offset=dst_region_offset,
                transfer_len=transfer_len,
            )
            for group_src, group_dst in zip(group_src_ids, group_dst_ids):
                if can_coalesce:
                    src_ptrs.append(
                        local_region.base_addr
                        + group_src[0] * local_region.block_len
                        + src_region_offset
                    )
                    dst_ptrs.append(
                        remote_region.base_addr
                        + group_dst[0] * remote_region.block_len
                        + dst_region_offset
                    )
                    lengths.append(transfer_len * len(group_src))
                else:
                    for src_block_id, dst_block_id in zip(group_src, group_dst):
                        src_ptrs.append(
                            local_region.base_addr
                            + src_block_id * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + dst_block_id * remote_region.block_len
                            + dst_region_offset
                        )
                        lengths.append(transfer_len)
        logger.info(
            "ReFlexKV trace landing_direct_transfer request=%s pages=%d.",
            d_req_id,
            len(dst_int4_block_ids),
        )
        return None

    async def _build_transfer_params(
        self,
        ready_reqs: list[tuple[ReqId, SendBlockMeta]],
        agent_meta: MooncakeXferMetadata,
        local_regions: list[TransferRegion],
        remote_regions: list[TransferRegion],
    ) -> tuple[list[int], list[int], list[int], list[ReqId], str | None]:
        src_ptrs = []
        dst_ptrs = []
        lengths = []
        err_reqs: list[ReqId] = []
        err_msg: str | None = None
        remote_session = f"{agent_meta.remote_hostname}:{agent_meta.remote_port}"

        for d_req_id, send_meta in ready_reqs:
            _, remote_block_ids = agent_meta.req_blocks[d_req_id]
            num_remote_blocks = len(remote_block_ids)

            local_block_ids = send_meta.local_block_ids
            # Partial prefix cache hit: just read uncomputed blocks.
            num_local_blocks = len(local_block_ids)
            direct_landing_pages = (
                [
                    int(page_idx)
                    for page_idx in agent_meta.reflex_int4_landing_pages.get(
                        d_req_id,
                        (),
                    )
                ]
                if agent_meta.reflex_int4_landing_pages
                and agent_meta.reflex_int4_direct_landing
                and agent_meta.reflex_int4_direct_landing.get(d_req_id, False)
                else []
            )
            landing_block_ids = (
                agent_meta.reflex_int4_landing_block_ids.get(d_req_id, [])
                if agent_meta.reflex_int4_landing_block_ids
                else []
            )
            direct_landing_by_page = {
                page_idx: int(landing_block_ids[idx])
                for idx, page_idx in enumerate(direct_landing_pages)
                if idx < len(landing_block_ids)
            }
            chunk_page_ranges = agent_meta.reflex_remote_chunk_page_ranges or {}
            if d_req_id in chunk_page_ranges:
                transfer_page_start = int(chunk_page_ranges[d_req_id][0])
            else:
                transfer_page_start = 0
            if direct_landing_by_page:
                needed_source_blocks = num_remote_blocks + len(direct_landing_by_page)
                if num_local_blocks > needed_source_blocks:
                    transfer_page_start = max(
                        0,
                        num_local_blocks - needed_source_blocks,
                    )
                    local_block_ids = local_block_ids[-needed_source_blocks:]
                    num_local_blocks = len(local_block_ids)
                source_pairs = [
                    (transfer_page_start + idx, block_id)
                    for idx, block_id in enumerate(local_block_ids)
                ]
                bf16_source_block_ids = [
                    block_id
                    for page_idx, block_id in source_pairs
                    if page_idx not in direct_landing_by_page
                ]
                direct_source_block_ids = [
                    block_id
                    for page_idx, block_id in source_pairs
                    if page_idx in direct_landing_by_page
                ]
                direct_dest_block_ids = [
                    direct_landing_by_page[page_idx]
                    for page_idx, _block_id in source_pairs
                    if page_idx in direct_landing_by_page
                ]
                if len(bf16_source_block_ids) != num_remote_blocks:
                    logger.error(
                        "req %s: direct landing source mapping mismatch: "
                        "bf16_source=%d remote_bf16=%d direct=%d",
                        d_req_id,
                        len(bf16_source_block_ids),
                        num_remote_blocks,
                        len(direct_source_block_ids),
                    )
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = "direct landing source mapping mismatch"
                    continue
                local_block_ids = bf16_source_block_ids
                int4_err = self._append_reflex_int4_direct_landing_transfers(
                    d_req_id=d_req_id,
                    src_bf16_block_ids=direct_source_block_ids,
                    dst_int4_block_ids=direct_dest_block_ids,
                    agent_meta=agent_meta,
                    src_ptrs=src_ptrs,
                    dst_ptrs=dst_ptrs,
                    lengths=lengths,
                )
                if int4_err is not None:
                    err_reqs.append(d_req_id)
                    if err_msg is None:
                        err_msg = int4_err
                    continue
            elif num_remote_blocks == 0:
                continue
            elif num_local_blocks < num_remote_blocks:
                logger.error(
                    "req %s: local blocks(%d) less than remote blocks(%d)!",
                    d_req_id,
                    num_local_blocks,
                    num_remote_blocks,
                )
                err_reqs.append(d_req_id)
                if err_msg is None:
                    err_msg = "P num blocks less than D"
                continue
            if num_local_blocks > num_remote_blocks:
                local_block_ids = local_block_ids[-num_remote_blocks:]

            # Group by indices
            group_local_block_ids, group_remote_block_ids = group_concurrent_contiguous(
                local_block_ids, remote_block_ids
            )

            for local_region, remote_region in zip(local_regions, remote_regions):
                should_transfer, src_region_offset, dst_region_offset, transfer_len = (
                    self._get_sender_transfer_plan(
                        local_kv_block_len=local_region.kv_block_len,
                        remote_kv_block_len=remote_region.kv_block_len,
                        remote_tp_rank=agent_meta.remote_tp_rank,
                        remote_tp_size=agent_meta.remote_tp_size,
                    )
                )
                if not should_transfer:
                    # Replicated KV cache: only one producer rank in the TP group
                    # needs to send the actual bytes for this paired decoder rank.
                    # TODO: Account for replicated producer KV in
                    # get_target_remote_ranks() so we can avoid sending
                    # unnecessary ZMQ requests and remove this branch.
                    continue

                assert src_region_offset + transfer_len <= local_region.kv_block_len, (
                    "Computed source transfer region exceeds local KV block size."
                )
                assert dst_region_offset + transfer_len <= remote_region.kv_block_len, (
                    "Computed destination transfer region exceeds remote KV block size."
                )
                # Collapse one contiguous block group into a single larger
                # transfer descriptor when the per-block copy is identical.
                can_coalesce = _can_coalesce_block_transfers(
                    local_region_block_len=local_region.block_len,
                    remote_region_block_len=remote_region.block_len,
                    src_region_offset=src_region_offset,
                    dst_region_offset=dst_region_offset,
                    transfer_len=transfer_len,
                )

                for group_local_block_id, group_remote_block_id in zip(
                    group_local_block_ids, group_remote_block_ids
                ):
                    if can_coalesce:
                        src_ptrs.append(
                            local_region.base_addr
                            + group_local_block_id[0] * local_region.block_len
                            + src_region_offset
                        )
                        dst_ptrs.append(
                            remote_region.base_addr
                            + group_remote_block_id[0] * remote_region.block_len
                            + dst_region_offset
                        )
                        lengths.append(transfer_len * len(group_local_block_id))
                    else:
                        for local_block_id, remote_block_id in zip(
                            group_local_block_id, group_remote_block_id
                        ):
                            src_ptrs.append(
                                local_region.base_addr
                                + local_block_id * local_region.block_len
                                + src_region_offset
                            )
                            dst_ptrs.append(
                                remote_region.base_addr
                                + remote_block_id * remote_region.block_len
                                + dst_region_offset
                            )
                            lengths.append(transfer_len)

                if local_region is local_regions[0]:
                    logger.debug(
                        "Mooncake transfer plan for request %s: local_tp=%d "
                        "remote_tp=%d remote_tp_rank=%d local_block_len=%d "
                        "remote_block_len=%d src_offset=%d dst_offset=%d "
                        "transfer_len=%d coalesce=%s",
                        d_req_id,
                        self.tp_size,
                        agent_meta.remote_tp_size,
                        agent_meta.remote_tp_rank,
                        local_region.block_len,
                        remote_region.block_len,
                        src_region_offset,
                        dst_region_offset,
                        transfer_len,
                        can_coalesce,
                    )

            logger.debug(
                "Sending kv_caches for request %s (%d blocks) to %s",
                d_req_id,
                num_remote_blocks,
                remote_session,
            )

        return src_ptrs, dst_ptrs, lengths, err_reqs, err_msg

    def _send_blocks(
        self,
        remote_session: str,
        src_ptrs: list[int],
        dst_ptrs: list[int],
        lengths: list[int],
    ) -> int:
        start_time = time.perf_counter()
        ret_value = self.engine.batch_transfer_sync_write(
            remote_session, src_ptrs, dst_ptrs, lengths
        )
        if ret_value == 0:
            logger.debug(
                "Sending to %s done, took %s",
                remote_session,
                time.perf_counter() - start_time,
            )
        return ret_value

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register the KV Cache data in mooncake."""

        logger.info("Registering KV_Caches. use_mla: %s", self.use_mla)

        kv_data_ptrs = []
        kv_data_lens = []
        seen_base_addresses = []
        self.block_len_per_layer = []

        split_k_and_v = self.kv_topo.split_k_and_v
        tensor_size_bytes = None
        for layer_name, cache_or_caches in kv_caches.items():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            logger.debug(
                "registering layer %s with %d cache tensor(s)",
                layer_name,
                len(cache_list),
            )

            for cache in cache_list:
                self._log_debug_cache_registration(layer_name, cache)
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue

                seen_base_addresses.append(base_addr)
                curr_tensor_size_bytes = cache.nbytes

                if tensor_size_bytes is None:
                    tensor_size_bytes = curr_tensor_size_bytes
                    self.num_blocks = cache.shape[0]
                assert cache.shape[0] == self.num_blocks, (
                    "All kv cache tensors must have the same number of blocks"
                )
                assert curr_tensor_size_bytes % self.num_blocks == 0, (
                    "Mooncake expects each kv cache tensor size to be "
                    "divisible by the number of blocks."
                )
                self.block_len_per_layer.append(
                    curr_tensor_size_bytes // self.num_blocks
                )

                kernel_block_size = cache.shape[-2 if self.use_mla else -3]
                assert self.block_size == kernel_block_size
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(curr_tensor_size_bytes)

        self.kv_caches_base_addr = seen_base_addresses
        self.seen_base_addresses = seen_base_addresses

        ret_value = self.engine.batch_register_memory(kv_data_ptrs, kv_data_lens)
        if ret_value != 0:
            raise RuntimeError("Mooncake batch memory registration failed.")

        assert tensor_size_bytes is not None
        assert self.num_blocks != 0
        self.device_kv_caches = kv_caches
        logger.debug(
            "registered num_blocks=%d block_lens=%s",
            self.num_blocks,
            self.block_len_per_layer,
        )

        # No need to launch server for D node.
        if self.is_kv_consumer:
            return

        ready_event = threading.Event()
        asyncio.run_coroutine_threadsafe(
            self._mooncake_sender_listener(ready_event), self.sender_loop
        )
        ready_event.wait()  # Wait for listener ZMQ socket to be ready.

    def register_reflex_int4_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor],
    ):
        """Register ReFlexKV INT4 sidecar caches.

        Decoder workers use these buffers as landing destinations; producer
        workers use them as temporary compressed sources for direct landing.
        """

        kv_data_ptrs: list[int] = []
        kv_data_lens: list[int] = []
        seen_base_addresses: list[int] = []
        block_lens: list[int] = []
        num_blocks: int | None = None
        split_k_and_v = self.kv_topo.split_k_and_v
        for cache_or_caches in kv_caches.values():
            cache_list = cache_or_caches if split_k_and_v else [cache_or_caches]
            for cache in cache_list:
                base_addr = cache.data_ptr()
                if base_addr in seen_base_addresses:
                    continue
                seen_base_addresses.append(base_addr)
                if num_blocks is None:
                    num_blocks = int(cache.shape[0])
                assert int(cache.shape[0]) == num_blocks, (
                    "All ReFlexKV INT4 sidecar tensors must have the same "
                    "number of blocks"
                )
                assert cache.nbytes % num_blocks == 0, (
                    "Mooncake expects each ReFlexKV INT4 sidecar tensor size "
                    "to be divisible by the number of blocks."
                )
                block_lens.append(cache.nbytes // num_blocks)
                kv_data_ptrs.append(base_addr)
                kv_data_lens.append(cache.nbytes)
        if kv_data_ptrs:
            ret_value = self.engine.batch_register_memory(
                kv_data_ptrs,
                kv_data_lens,
            )
            if ret_value != 0:
                raise RuntimeError(
                    "Mooncake ReFlexKV INT4 sidecar memory registration failed."
                )
        self.reflex_int4_kv_caches = kv_caches
        self.reflex_int4_kv_caches_base_addr = seen_base_addresses
        self.reflex_int4_block_len_per_layer = block_lens
        self.reflex_int4_num_blocks = int(num_blocks or 0)

    async def fetch_finished_recving_reqs(self) -> set[ReqId]:
        finished_recving_reqs = self.finished_recving_reqs
        self.finished_recving_reqs = set()
        return finished_recving_reqs

    async def fetch_reflex_int4_materialized_landing_reqs(self) -> set[ReqId]:
        materialized_reqs = self.reflex_int4_materialized_landing_reqs
        self.reflex_int4_materialized_landing_reqs = set()
        return materialized_reqs

    async def fetch_finished_recving_and_reflex_int4_materialized_landing_reqs(
        self,
    ) -> tuple[set[ReqId], set[ReqId]]:
        finished_recving_reqs = self.finished_recving_reqs
        materialized_reqs = self.reflex_int4_materialized_landing_reqs
        self.finished_recving_reqs = set()
        self.reflex_int4_materialized_landing_reqs = set()
        return finished_recving_reqs, materialized_reqs

    async def fetch_finished_sending_reqs(self) -> set[ReqId]:
        finished_sending_reqs = self.finished_sending_reqs
        self.finished_sending_reqs = set()

        # Handle timeout to avoid stranding blocks on remote.
        now = time.perf_counter()

        expired_transfer_id = []
        for transfer_id, send_meta in self.reqs_need_send.items():
            if (
                send_meta.p_req_id
                and send_meta.expire_time < now
                and send_meta.sending == 0
            ):
                logger.warning(
                    "Request %s timed out after %d seconds without "
                    "being sent. Freeing its blocks on the producer side.",
                    send_meta.p_req_id,
                    envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT,
                )
                finished_sending_reqs.add(send_meta.p_req_id)
                expired_transfer_id.append(transfer_id)

        for transfer_id in expired_transfer_id:
            del self.reqs_need_send[transfer_id]

        return finished_sending_reqs

    def get_finished(
        self,
        finished_req_ids: set[str] | None = None,
    ) -> tuple[set[str] | None, set[str] | None]:
        """
        Get requests that are done sending or recving on this specific worker.
        The scheduler process (via the MultiprocExecutor) will use this output
        to track which workers are done.
        """
        if not self.is_kv_consumer:
            recorder = get_reflex_prefill_metadata_recorder()
            page_risks = recorder.drain_completed_requests(
                block_size=self.block_size
            )
            if finished_req_ids:
                page_risks.update(
                    recorder.finalize_requests(
                        finished_req_ids,
                        prompt_tokens_by_request={},
                        block_size=self.block_size,
                    )
            )
            if page_risks:
                logger.info(
                    "ReFlexKV trace page_metadata_produce requests=%d "
                    "pages=%d source=prefill_recorder.",
                    len(page_risks),
                    sum(len(scores) for scores in page_risks.values()),
                )
                self._merge_pending_worker_meta(
                    MooncakeConnectorWorkerMetadata(page_risks)
                )

        recv_and_materialized_fut = None
        send_fut = None
        if not self.is_kv_producer:
            recv_and_materialized_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_recving_and_reflex_int4_materialized_landing_reqs(),
                self.receiver_loop,
            )

        if not self.is_kv_consumer:
            send_fut = asyncio.run_coroutine_threadsafe(
                self.fetch_finished_sending_reqs(), self.sender_loop
            )

        if recv_and_materialized_fut:
            finished_recving_reqs, materialized_landing_reqs = (
                recv_and_materialized_fut.result()
            )
        else:
            finished_recving_reqs = set()
            materialized_landing_reqs = set()
        finished_sending_reqs = send_fut.result() if send_fut else set()
        if materialized_landing_reqs:
            self._merge_pending_worker_meta(
                MooncakeConnectorWorkerMetadata(
                    reflex_page_risks_by_request={},
                    reflex_int4_materialized_landing_req_ids=materialized_landing_reqs,
                )
            )

        if finished_sending_reqs or finished_recving_reqs:
            logger.debug(
                "Rank %s, get_finished: %s requests done sending "
                "and %s requests done recving",
                self.tp_rank,
                len(finished_sending_reqs),
                len(finished_recving_reqs),
            )

        return finished_sending_reqs or None, finished_recving_reqs or None

    def _merge_pending_worker_meta(
        self,
        meta: MooncakeConnectorWorkerMetadata,
    ) -> None:
        if self._pending_worker_meta is None:
            self._pending_worker_meta = meta
        else:
            self._pending_worker_meta = self._pending_worker_meta.aggregate(meta)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        meta = self._pending_worker_meta
        self._pending_worker_meta = None
        return meta

    async def receive_kv_from_single_worker(
        self,
        worker_addr: str,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        req_ids = set(pull_metas)
        metadata = MooncakeXferMetadata(
            remote_hostname=self.hostname,
            remote_port=self.rpc_port,
            remote_tp_size=self.tp_size,
            remote_tp_rank=self.tp_rank,
            req_blocks={
                req_id: (pull_meta.transfer_id, pull_meta.local_block_ids)
                for req_id, pull_meta in pull_metas.items()
            },
            kv_caches_base_addr=self.kv_caches_base_addr,
            block_lens=self.block_len_per_layer,
            reflex_int4_landing_pages={
                req_id: list(pull_meta.reflex_int4_landing_pages)
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_int4_landing_pages
            }
            or None,
            reflex_int4_landing_block_ids={
                req_id: list(pull_meta.reflex_int4_landing_block_ids)
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_int4_landing_block_ids
            }
            or None,
            reflex_int4_landing_planned_blocks={
                req_id: pull_meta.reflex_int4_landing_planned_blocks
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_int4_landing_planned_blocks > 0
            }
            or None,
            reflex_int4_direct_landing={
                req_id: True
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_int4_direct_landing
            }
            or None,
            reflex_int4_kv_caches_base_addr=(
                list(getattr(self, "reflex_int4_kv_caches_base_addr", []))
                or None
            ),
            reflex_int4_block_lens=(
                list(getattr(self, "reflex_int4_block_len_per_layer", []))
                or None
            ),
            reflex_remote_chunk_ids={
                req_id: int(pull_meta.reflex_remote_chunk_id)
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_remote_chunk_id is not None
            }
            or None,
            reflex_remote_chunk_token_ranges={
                req_id: (
                    pull_meta.reflex_remote_chunk_token_start,
                    pull_meta.reflex_remote_chunk_token_end,
                )
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_remote_chunk_id is not None
            }
            or None,
            reflex_remote_chunk_page_ranges={
                req_id: (
                    pull_meta.reflex_remote_chunk_page_start,
                    pull_meta.reflex_remote_chunk_page_end,
                )
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_remote_chunk_id is not None
            }
            or None,
            reflex_remote_chunk_is_last={
                req_id: bool(pull_meta.reflex_remote_chunk_is_last)
                for req_id, pull_meta in pull_metas.items()
                if pull_meta.reflex_remote_chunk_id is not None
            }
            or None,
        )

        encoded_data = self._encoder.encode(metadata)
        logger.debug(
            "Size of encoded MooncakeXferMetadata: %d bytes", len(encoded_data)
        )
        logger.debug(
            "Sending kv transfer request for %s on path: %s", req_ids, worker_addr
        )

        # Send query for the request.
        try:
            with make_zmq_socket(
                self.async_zmq_ctx, worker_addr, zmq.DEALER, bind=False, linger=0
            ) as sock:
                # If something goes wrong, let P wait timeout first (in asyncio.wait()).
                sock.setsockopt(
                    zmq.RCVTIMEO, (envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT + 60) * 1000
                )
                await sock.send(encoded_data)
                while True:
                    ret_msg = await sock.recv()
                    response = self._xfer_resp_decoder.decode(ret_msg)
                    if response.status == MooncakeXferResponseStatus.ERROR:
                        logger.error(
                            "Error happens during transferring kvcache for %s: %s",
                            req_ids,
                            response.err_msg,
                        )
                        return
                    self.process_pulling_result(response, pull_metas)
                    if response.status == MooncakeXferResponseStatus.FINISH:
                        break
        except zmq.ContextTerminated:
            logger.debug("ZMQ context terminated, exiting Mooncake receiver thread.")
        except Exception as e:
            logger.error("MooncakeXferMetadata transfer failed for %s: %s", req_ids, e)
            return

    def _materialize_reflex_int4_landing_pages(
        self,
        pull_meta: PullReqMeta,
    ) -> bool:
        landing_pages = pull_meta.reflex_int4_landing_pages or []
        landing_block_ids = pull_meta.reflex_int4_landing_block_ids or []
        landing_source_block_ids = (
            pull_meta.reflex_int4_landing_source_block_ids or []
        )
        if not landing_pages or not landing_block_ids:
            return False
        if len(landing_pages) != len(landing_block_ids):
            logger.warning(
                "Skipping ReFlexKV INT4 landing materialization for %s: "
                "%d pages but %d destination blocks.",
                pull_meta.d_req_id,
                len(landing_pages),
                len(landing_block_ids),
            )
            return False
        if pull_meta.reflex_int4_direct_landing:
            logger.info(
                "ReFlexKV trace landing_materialize request=%s pages=%d "
                "outcome=direct_int4_transfer.",
                pull_meta.d_req_id,
                len(landing_pages),
            )
            return True
        if len(landing_pages) != len(landing_source_block_ids):
            logger.warning(
                "Skipping ReFlexKV INT4 landing materialization for %s: "
                "%d pages but %d BF16 source blocks.",
                pull_meta.d_req_id,
                len(landing_pages),
                len(landing_source_block_ids),
            )
            return False

        int4_kv_caches = getattr(self, "reflex_int4_kv_caches", {})
        if not int4_kv_caches:
            return False

        src_block_ids_list: list[int] = []
        dst_block_ids_list: list[int] = []
        for source_block_id, int4_block_id in zip(
            landing_source_block_ids,
            landing_block_ids,
        ):
            if source_block_id < 0 or int4_block_id < 0:
                return False
            src_block_ids_list.append(source_block_id)
            dst_block_ids_list.append(int4_block_id)

        if len(src_block_ids_list) != len(landing_pages):
            return False

        layer_copies = 0
        first_layer = next(
            (
                layer_name
                for layer_name in self.device_kv_caches
                if layer_name in int4_kv_caches
            ),
            None,
        )
        if first_layer is None:
            return False

        device = self.device_kv_caches[first_layer].device
        cpu_start = time.perf_counter()
        start_event = end_event = None
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        src_block_ids = torch.tensor(
            src_block_ids_list,
            dtype=torch.int64,
            device=device,
        )
        dst_block_ids = torch.tensor(
            dst_block_ids_list,
            dtype=torch.int64,
            device=device,
        )
        for layer_name, bf16_cache in self.device_kv_caches.items():
            int4_cache = int4_kv_caches.get(layer_name)
            if int4_cache is None:
                continue
            get_reflex_int4_codec().quantize_blocks_to_cache(
                bf16_cache,
                int4_cache,
                src_block_ids,
                dst_block_ids,
            )
            layer_copies += len(src_block_ids_list)
        kernel_launches = sum(
            1 for layer_name in self.device_kv_caches if layer_name in int4_kv_caches
        )

        gpu_ms = -1.0
        if start_event is not None and end_event is not None:
            end_event.record()
            end_event.synchronize()
            gpu_ms = start_event.elapsed_time(end_event)
        logger.info(
            "ReFlexKV trace landing_materialize request=%s pages=%d "
            "layer_copies=%d kernel_launches=%d cpu_ms=%.3f gpu_ms=%.3f.",
            pull_meta.d_req_id,
            len(src_block_ids_list),
            layer_copies,
            kernel_launches,
            (time.perf_counter() - cpu_start) * 1000.0,
            gpu_ms,
        )
        return True

    def process_pulling_result(
        self,
        response: MooncakeXferResponse,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        ok_reqs: list[ReqId] = response.ok_reqs or []

        for req_id in ok_reqs:
            pull_meta = pull_metas[req_id]
            # No race because we are in async loop.
            pull_meta.pull_tasks_count -= 1
            if pull_meta.pull_tasks_count == 0:
                if self._materialize_reflex_int4_landing_pages(pull_meta):
                    self.reflex_int4_materialized_landing_reqs.add(
                        pull_meta.d_req_id
                    )
                self.finished_recving_reqs.add(pull_meta.d_req_id)

        if ok_reqs:
            logger.debug("pulling kv_caches for %s finished", ok_reqs)

        if response.err_reqs:
            logger.error(
                "pulling kv_caches for %s failed: %s",
                response.err_reqs,
                response.err_msg,
            )

    async def _connect_to_prefiller_bootstrap(self, remote_bootstrap_addr: str):
        url = remote_bootstrap_addr + "/query"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url)
                response.raise_for_status()
                data: dict = response.json()
                for _, dp_entry in data.items():
                    remote_engine_id = dp_entry["engine_id"]
                    self._remote_agents[remote_engine_id] = {
                        int(tp_rank): {
                            int(pp_rank): worker_addr
                            for pp_rank, worker_addr in tp_entry.items()
                        }
                        for tp_rank, tp_entry in dp_entry["worker_addr"].items()
                    }
                    self._tp_size[remote_engine_id] = len(dp_entry["worker_addr"])
        except Exception as e:
            logger.error(
                "Failed to connect to bootstrap server %s: %s",
                remote_bootstrap_addr,
                e,
            )

        # Always notify others regardless of connection success or failure.
        self._pending_bootstrap_queries[remote_bootstrap_addr].set()
        del self._pending_bootstrap_queries[remote_bootstrap_addr]

    def receive_kv(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_tp_ranks = self.kv_topo.get_target_remote_ranks_from_engine_id(
            remote_engine_id
        )
        count = len(remote_tp_ranks)
        logger.debug(
            "Receiving Mooncake KV for engine %s from producer TP ranks %s",
            remote_engine_id,
            remote_tp_ranks,
        )
        for pull_meta in pull_metas.values():
            pull_meta.pull_tasks_count = count
        for remote_tp_rank in remote_tp_ranks:
            worker_addr = self._remote_agents[remote_engine_id][remote_tp_rank][0]
            asyncio.create_task(
                self.receive_kv_from_single_worker(worker_addr, pull_metas)
            )

    async def handle_new_engine_id(
        self,
        remote_engine_id: EngineId,
        pull_metas: dict[ReqId, PullReqMeta],
    ):
        remote_bootstrap_addr = next(iter(pull_metas.values())).remote_bootstrap_addr
        if remote_bootstrap_addr not in self._pending_bootstrap_queries:
            self._pending_bootstrap_queries[remote_bootstrap_addr] = asyncio.Event()
            await self._connect_to_prefiller_bootstrap(remote_bootstrap_addr)
        else:
            await self._pending_bootstrap_queries[remote_bootstrap_addr].wait()

        if remote_engine_id not in self._remote_agents:
            logger.error(
                "Failed to find remote engine_id %s from bootstrap server %s",
                remote_engine_id,
                remote_bootstrap_addr,
            )
            return

        self.receive_kv(remote_engine_id, pull_metas)

    async def _start_load_kv(
        self, reqs_to_recv: dict[EngineId, dict[ReqId, PullReqMeta]]
    ):
        for remote_engine_id, pull_metas in reqs_to_recv.items():
            if remote_engine_id not in self._remote_agents:
                asyncio.create_task(
                    self.handle_new_engine_id(remote_engine_id, pull_metas)
                )
            else:
                self.receive_kv(remote_engine_id, pull_metas)

    async def record_send_reqs(self, metadata: MooncakeConnectorMetadata):
        for p_req_id, (transfer_id, block_ids) in metadata.reqs_to_send.items():
            chunk_id = (
                metadata.reflex_remote_chunk_ids.get(p_req_id)
                if metadata.reflex_remote_chunk_ids
                and p_req_id in metadata.reflex_remote_chunk_ids
                else None
            )
            send_key = self._send_key(transfer_id, chunk_id)
            defer_ready = bool(
                metadata.reflex_remote_chunk_defer_ready
                and metadata.reflex_remote_chunk_defer_ready.get(p_req_id, False)
            )
            release_after_send = (
                bool(
                    metadata.reflex_remote_chunk_release_after_send.get(
                        p_req_id,
                        False,
                    )
                )
                if metadata.reflex_remote_chunk_release_after_send
                else chunk_id is None
            )
            if block_ids:
                if release_after_send and send_key in self._completed_send_keys:
                    self._completed_send_keys.remove(send_key)
                    self.finished_sending_reqs.add(p_req_id)
                    continue
                # Already gone through request_finished()
                send_meta = self.reqs_need_send.get(send_key)
                if send_meta is None:
                    send_meta = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                        reflex_remote_chunk_id=chunk_id,
                        release_after_send=release_after_send,
                    )
                    self.reqs_need_send[send_key] = send_meta
                send_meta.p_req_id = p_req_id
                send_meta.local_block_ids = block_ids
                send_meta.reflex_remote_chunk_id = chunk_id
                send_meta.release_after_send = (
                    send_meta.release_after_send or release_after_send
                )
                send_meta.expire_time = (
                    time.perf_counter() + envs.VLLM_MOONCAKE_ABORT_REQUEST_TIMEOUT
                )
                if defer_ready:
                    self._deferred_ready_send_keys.add(send_key)
                else:
                    send_meta.ready.set()
            else:
                # From update_state_after_alloc(),
                # but not reach request_finished() yet
                # This may be already created by send_kv_to_decode()
                # when D is sending MooncakeXferMetadata.
                if send_key not in self.reqs_need_send:
                    self.reqs_need_send[send_key] = SendBlockMeta(
                        p_req_id=p_req_id,
                        transfer_id=transfer_id,
                        local_block_ids=[],
                        ready=asyncio.Event(),
                        reflex_remote_chunk_id=chunk_id,
                        release_after_send=release_after_send,
                    )
        for transfer_id in metadata.reqs_not_processed:
            matching_keys = [
                key
                for key, send_meta in self.reqs_need_send.items()
                if send_meta.transfer_id == transfer_id
            ]
            for key in matching_keys:
                send_meta = self.reqs_need_send.pop(key)
                if send_meta:
                    assert not send_meta.ready.is_set()

    def start_load_kv(self, metadata: MooncakeConnectorMetadata):
        if not self.is_kv_producer and metadata.reqs_to_recv:
            asyncio.run_coroutine_threadsafe(
                self._start_load_kv(metadata.reqs_to_recv), self.receiver_loop
            )

        if not self.is_kv_consumer and (
            metadata.reqs_to_send or metadata.reqs_not_processed
        ):
            asyncio.run_coroutine_threadsafe(
                self.record_send_reqs(metadata), self.sender_loop
            )

    async def _mark_deferred_ready_send_reqs(self):
        if not self._deferred_ready_send_keys:
            return
        for send_key in list(self._deferred_ready_send_keys):
            send_meta = self.reqs_need_send.get(send_key)
            if send_meta is not None:
                send_meta.ready.set()
        self._deferred_ready_send_keys.clear()

    def wait_for_save(self):
        if not self._deferred_ready_send_keys:
            return
        sender_loop = getattr(self, "sender_loop", None)
        if sender_loop is not None and sender_loop.is_running():
            future = asyncio.run_coroutine_threadsafe(
                self._mark_deferred_ready_send_reqs(),
                sender_loop,
            )
            future.result()
            return
        for send_key in list(self._deferred_ready_send_keys):
            send_meta = self.reqs_need_send.get(send_key)
            if send_meta is not None:
                send_meta.ready.set()
        self._deferred_ready_send_keys.clear()

    def _producer_cache_is_replicated(self) -> bool:
        return self.kv_topo.replicates_kv_cache(self.engine_id)

    def _get_transfer_regions(
        self, base_addrs: list[int], block_lens: list[int]
    ) -> list[TransferRegion]:
        return _expand_transfer_regions(
            base_addrs=base_addrs,
            block_lens=block_lens,
            is_kv_layout_blocks_first=self.kv_topo.is_kv_layout_blocks_first,
        )

    def _get_sender_transfer_plan(
        self,
        local_kv_block_len: int,
        remote_kv_block_len: int,
        remote_tp_rank: int,
        remote_tp_size: int,
    ) -> tuple[bool, int, int, int]:
        return _compute_sender_transfer_plan(
            local_tp_rank=self.tp_rank,
            local_tp_size=self.tp_size,
            remote_tp_rank=remote_tp_rank,
            remote_tp_size=remote_tp_size,
            local_kv_block_len=local_kv_block_len,
            remote_kv_block_len=remote_kv_block_len,
            producer_cache_replicated=self._producer_cache_is_replicated(),
        )

    def _log_debug_cache_registration(
        self, layer_name: str, cache: torch.Tensor
    ) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        logger.debug(
            "Mooncake register view layer=%s shape=%s stride=%s "
            "storage_offset=%d contiguous=%s dense=%s data_ptr=%d",
            layer_name,
            tuple(cache.shape),
            tuple(cache.stride()),
            cache.storage_offset(),
            cache.is_contiguous(),
            _get_tensor_dense_flag(cache),
            cache.data_ptr(),
        )


def group_concurrent_contiguous(
    src_indices: list[int], dst_indices: list[int]
) -> tuple[list[list[int]], list[list[int]]]:
    """Vectorised NumPy implementation."""
    if len(src_indices) == 0:
        return [], []

    brk = np.where((np.diff(src_indices) != 1) | (np.diff(dst_indices) != 1))[0] + 1
    src_groups = np.split(src_indices, brk)
    dst_groups = np.split(dst_indices, brk)

    src_groups = [g.tolist() for g in src_groups]
    dst_groups = [g.tolist() for g in dst_groups]

    return src_groups, dst_groups


def get_mooncake_side_channel_port(vllm_config: VllmConfig) -> int:
    # This logic is now centralized
    return (
        envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
        + vllm_config.parallel_config.data_parallel_index
        * vllm_config.parallel_config.tensor_parallel_size
    )


def _async_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


def should_launch_bootstrap_server(vllm_config: VllmConfig) -> bool:
    assert (parallel_config := vllm_config.parallel_config)
    # In hybrid or external LB mode,
    # each instance should have its own bootstrap server.
    #
    # In internal LB mode,
    # only the real global first rank need to launch the bootstrap server.
    return is_local_first_rank() and (
        parallel_config.local_engines_only or parallel_config.data_parallel_index == 0
    )


def get_mooncake_bootstrap_addr(vllm_config: VllmConfig) -> tuple[str, int]:
    """
    Returns the address of the Mooncake bootstrap server.
    This is only used by prefillers to register workers.
    Decoders should get addr from kv_transfer_params.
    """
    assert (parallel_config := vllm_config.parallel_config)
    if parallel_config.local_engines_only:
        # In hybrid or external LB mode, connect to local server.
        host = "127.0.0.1"
    else:
        host = parallel_config.data_parallel_master_ip
    port = envs.VLLM_MOONCAKE_BOOTSTRAP_PORT
    return (host, port)
