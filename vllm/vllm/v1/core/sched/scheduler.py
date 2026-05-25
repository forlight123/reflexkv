# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
import math
import os
import time
from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any

import numpy as np

from vllm import envs
from vllm.compilation.cuda_graph import CUDAGraphStat
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorMetadata,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.factory import ECConnectorFactory
from vllm.distributed.kv_events import EventPublisherFactory, KVEventBatch
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsReader,
)
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.precision_kv.chunks import (
    normalize_remote_chunk_tokens,
    remote_chunking_enabled,
)
from vllm.v1.core.precision_kv.contracts import (
    clear_reflex_int4_landing_contract,
    has_reflex_int4_landing_contract,
)
from vllm.v1.core.precision_kv.controller import (
    PrecisionAdmissionController,
    PrecisionAdmissionState,
)
from vllm.v1.core.precision_kv.demotion_planner import (
    RequestBudgetCandidate,
    RequestPrecisionBudget,
    allocate_request_release_budgets,
)
from vllm.v1.core.precision_kv.frontier import (
    AdmissionTicket,
    FeasibleFrontierCache,
    FeasibleFrontierSummary,
)
from vllm.v1.core.precision_kv.landing import (
    PrecisionLandingDecision,
    PrecisionLandingPlanner,
    PrecisionLandingState,
)
from vllm.v1.core.precision_kv.policy import (
    CandidateFunnelSnapshot,
    PrecisionKVPolicy,
    PrecisionPressureDecision,
    PrecisionPressureState,
)
from vllm.v1.core.precision_kv.risk import (
    derive_compressible_pages_from_risks,
    select_bf16_shadow_pages,
    synthesize_remote_chunk_landing_pages,
)
from vllm.v1.core.precision_kv.run_optimizer import DualPriceState
from vllm.v1.core.precision_kv.types import (
    PrecisionState,
)
from vllm.v1.core.sched.interface import PauseState, SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.request_queue import (
    RequestQueue,
    SchedulingPolicy,
    create_request_queue,
)
from vllm.v1.core.sched.utils import check_stop, remove_all
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs
from vllm.v1.kv_cache_interface import AttentionSpec, KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics, PerfStats
from vllm.v1.metrics.stats import PrefixCacheStats, SchedulerStats
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus, StreamingUpdate
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

logger = init_logger(__name__)

DEFAULT_REFLEX_BF16_SHADOW_PAGES_PER_REQUEST = 1
DEFAULT_REFLEX_BACKGROUND_PROMOTION_PAGES_PER_STEP = 1


class Scheduler(SchedulerInterface):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )

        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        if self.vllm_config.kv_transfer_config is not None:
            assert not self.is_encoder_decoder, (
                "Encoder-decoder models are not currently supported with KV connectors"
            )
            self.connector = KVConnectorFactory.create_connector(
                config=self.vllm_config,
                role=KVConnectorRole.SCHEDULER,
                kv_cache_config=self.kv_cache_config,
            )
            if self.log_stats:
                self.connector_prefix_cache_stats = PrefixCacheStats()
            kv_load_failure_policy = (
                self.vllm_config.kv_transfer_config.kv_load_failure_policy
            )
            self.recompute_kv_load_failures = kv_load_failure_policy == "recompute"

        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_index,
        )
        self.ec_connector = None
        if self.vllm_config.ec_transfer_config is not None:
            self.ec_connector = ECConnectorFactory.create_connector(
                config=self.vllm_config, role=ECConnectorRole.SCHEDULER
            )

        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = block_size
        self.dcp_world_size = vllm_config.parallel_config.decode_context_parallel_size
        self.pcp_world_size = vllm_config.parallel_config.prefill_context_parallel_size

        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.reflex_int4_materialized_landing_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Encoder-related.
        # Calculate encoder cache size if applicable
        supports_mm_inputs = mm_registry.supports_multimodal_inputs(
            vllm_config.model_config
        )
        mm_budget = (
            MultiModalBudget(vllm_config, mm_registry) if supports_mm_inputs else None
        )

        # NOTE: Text-only encoder-decoder models are implemented as
        # multi-modal models for convenience
        # Example: https://github.com/vllm-project/bart-plugin
        if self.is_encoder_decoder:
            assert mm_budget and len(mm_budget.mm_max_toks_per_item) <= 1, (
                "Encoder-decoder models are expected to implement the "
                "multimodal interface with at most one modality."
            )

        self.max_num_encoder_input_tokens = (
            mm_budget.encoder_compute_budget if mm_budget else 0
        )
        encoder_cache_size = mm_budget.encoder_cache_size if mm_budget else 0
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        speculative_config = vllm_config.speculative_config
        self.use_eagle = False
        self.num_spec_tokens = self.num_lookahead_tokens = 0
        if speculative_config:
            self.num_spec_tokens = speculative_config.num_speculative_tokens
            if speculative_config.use_eagle():
                self.use_eagle = True
                self.num_lookahead_tokens = self.num_spec_tokens
            if speculative_config.uses_draft_model():
                self.num_lookahead_tokens = self.num_spec_tokens

        # Create the KV cache manager.
        self.kv_cache_manager = KVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=self.use_eagle,
            log_stats=self.log_stats,
            enable_kv_cache_events=self.enable_kv_cache_events,
            dcp_world_size=self.dcp_world_size,
            pcp_world_size=self.pcp_world_size,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
        )
        # Bind GPU block pool to the KV connector. This must happen after
        # kv_cache_manager is constructed so block_pool is available.
        if self.connector is not None and hasattr(
            self.connector, "bind_gpu_block_pool"
        ):
            self.connector.bind_gpu_block_pool(self.kv_cache_manager.block_pool)
        self._reflex_int4_scheduler_step = 0
        self._reflex_int4_last_demote_step = -1024
        self._reflex_int4_low_watermark = 0.05
        self._reflex_int4_high_watermark = 0.10
        self._reflex_int4_demote_cooldown_steps = 4
        self._reflex_int4_prev_step_had_prefill = False
        self._reflex_int4_last_demote_candidate_capacity = 0
        self._reflex_int4_last_candidate_breakdown = None
        self._reflex_int4_frontier_cache = FeasibleFrontierCache(
            max_age_steps=self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_FRONTIER_CACHE_MAX_AGE_STEPS",
                2,
            )
        )
        self._reflex_int4_admission_tickets: dict[str, AdmissionTicket] = {}
        self._reflex_int4_frontier_event_steps: dict[str, int] = {}
        self._reflex_int4_admission_ticket_retry_delay_steps = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_ADMISSION_TICKET_RETRY_DELAY_STEPS",
                8,
            )
        )
        self._reflex_int4_admission_ticket_max_retry_delay_steps = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_ADMISSION_TICKET_MAX_RETRY_DELAY_STEPS",
                64,
            )
        )
        self._reflex_int4_keep_recent_pages = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_KEEP_RECENT_PAGES",
            16,
        )
        self._reflex_int4_keep_initial_pages = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_KEEP_INITIAL_PAGES",
            4,
        )
        self._reflex_int4_max_int4_fraction_per_request = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_MAX_INT4_FRACTION_PER_REQUEST",
            1.0,
            minimum=0.0,
            maximum=1.0,
        )
        self._reflex_int4_quality_debt_max_fraction = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_QUALITY_DEBT_MAX_INT4_FRACTION",
            1.0,
            minimum=0.0,
            maximum=1.0,
        )
        self._reflex_int4_short_decode_tokens = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_SHORT_DECODE_TOKENS",
            128,
        )
        self._reflex_int4_short_decode_max_int4_fraction = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_SHORT_DECODE_MAX_INT4_FRACTION",
            0.0,
            minimum=0.0,
            maximum=1.0,
        )
        self._reflex_int4_short_admission_max_int4_fraction = (
            self._read_reflex_float_env(
                "SEMANTIQ_REFLEX_SHORT_ADMISSION_MAX_INT4_FRACTION",
                0.03,
                minimum=0.0,
                maximum=1.0,
            )
        )
        self._reflex_int4_cold_admission_max_int4_fraction = (
            self._read_reflex_float_env(
                "SEMANTIQ_REFLEX_COLD_ADMISSION_MAX_INT4_FRACTION",
                0.10,
                minimum=0.0,
                maximum=1.0,
            )
        )
        self._reflex_int4_cold_admission_emergency_free_ratio = (
            self._read_reflex_float_env(
                "SEMANTIQ_REFLEX_COLD_ADMISSION_EMERGENCY_FREE_RATIO",
                0.05,
                minimum=0.0,
                maximum=1.0,
            )
        )
        self._reflex_int4_risk_warmup_tokens = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_RISK_WARMUP_TOKENS",
            self.block_size,
        )
        self._reflex_int4_survival_warmup_tokens = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_SURVIVAL_WARMUP_TOKENS",
                128,
            )
        )
        self._reflex_int4_sparse_window_pages = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_SPARSE_WINDOW_PAGES",
            32,
        )
        self._reflex_int4_short_max_demote_per_window = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_SHORT_MAX_DEMOTE_PER_WINDOW",
                1,
            )
        )
        self._reflex_int4_max_demote_per_window = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_MAX_DEMOTE_PER_WINDOW",
            2,
        )
        self._reflex_int4_admission_sparse_window_pages = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_ADMISSION_SPARSE_WINDOW_PAGES",
                self._reflex_int4_sparse_window_pages,
            )
        )
        self._reflex_int4_admission_max_demote_per_window = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_ADMISSION_MAX_DEMOTE_PER_WINDOW",
                max(self._reflex_int4_max_demote_per_window, 8),
            )
        )
        self._reflex_int4_admission_pressure_min_int4_fraction = (
            self._read_reflex_float_env(
                "SEMANTIQ_REFLEX_ADMISSION_PRESSURE_MIN_INT4_FRACTION",
                0.10,
                minimum=0.0,
                maximum=1.0,
            )
        )
        self._reflex_int4_admission_landing_max_int4_fraction = (
            self._read_reflex_float_env(
                "SEMANTIQ_REFLEX_ADMISSION_LANDING_MAX_INT4_FRACTION",
                0.75,
                minimum=0.0,
                maximum=1.0,
            )
        )
        self._reflex_int4_mixed_landing_admission_enabled = os.environ.get(
            "SEMANTIQ_REFLEX_ENABLE_MIXED_LANDING_ADMISSION",
            "0",
        ).strip().lower() in {"1", "true", "yes", "on"}
        self._reflex_int4_direct_landing_enabled = self._read_reflex_bool_env(
            "SEMANTIQ_REFLEX_ENABLE_DIRECT_INT4_LANDING",
            True,
        )
        self._reflex_int4_low_risk_score_fraction = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_LOW_RISK_SCORE_FRACTION",
            0.25,
            minimum=0.0,
            maximum=1.0,
        )
        self._reflex_int4_page_selection_policy = (
            os.environ.get(
                "SEMANTIQ_REFLEX_PAGE_SELECTION_POLICY",
                "relevance_sparse",
            )
            .strip()
            .lower()
            .replace("-", "_")
        )
        if self._reflex_int4_page_selection_policy not in {
            "oldest",
            "distance",
            "random",
            "relevance",
            "relevance_sparse",
            "frontier_dual",
        }:
            logger.warning(
                "Invalid SEMANTIQ_REFLEX_PAGE_SELECTION_POLICY=%r; using "
                "relevance_sparse.",
                self._reflex_int4_page_selection_policy,
            )
            self._reflex_int4_page_selection_policy = "relevance_sparse"
        self._reflex_int4_dual_price_state = DualPriceState()
        self._reflex_int4_dual_eta = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_DUAL_ETA",
            0.05,
            minimum=0.0,
            maximum=10.0,
        )
        self._reflex_int4_dual_kv_target = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_DUAL_KV_TARGET",
            0.85,
            minimum=0.0,
            maximum=1.0,
        )
        self._reflex_int4_dual_waiting_target = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_DUAL_WAITING_TARGET",
            0,
        )
        self._reflex_int4_dual_migration_target = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_DUAL_MIGRATION_TARGET",
            1,
        )
        self._reflex_int4_slo_pressure_step = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_SLO_PRESSURE_STEP",
            0.25,
            minimum=0.0,
            maximum=10.0,
        )
        self._reflex_int4_min_slo_pressure = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_MIN_SLO_PRESSURE",
            0.5,
            minimum=0.0,
            maximum=10.0,
        )
        self._reflex_int4_max_slo_pressure = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_MAX_SLO_PRESSURE",
            1.5,
            minimum=0.0,
            maximum=10.0,
        )
        self._reflex_int4_decode_pressure_warmup_tokens = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_DECODE_PRESSURE_WARMUP_TOKENS",
                32,
            )
        )
        self._reflex_int4_decode_pressure_ramp_tokens = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_DECODE_PRESSURE_RAMP_TOKENS",
                512,
            )
        )
        self._reflex_int4_short_prefill_pages = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_SHORT_PREFILL_PAGES",
            64,
        )
        self._reflex_int4_long_prefill_pages = self._read_reflex_nonnegative_int_env(
            "SEMANTIQ_REFLEX_LONG_PREFILL_PAGES",
            512,
        )
        self._reflex_int4_global_evidence_min_prompt_pages = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_MIN_PROMPT_PAGES",
                self._reflex_int4_long_prefill_pages,
            )
        )
        self._reflex_int4_global_evidence_min_decode_tokens = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_MIN_DECODE_TOKENS",
                self._reflex_int4_short_decode_tokens + 1,
            )
        )
        self._reflex_int4_global_evidence_landing_max_int4_fraction = (
            self._read_reflex_float_env(
                "SEMANTIQ_REFLEX_GLOBAL_EVIDENCE_LANDING_MAX_INT4_FRACTION",
                0.08,
                minimum=0.0,
                maximum=1.0,
            )
        )
        self._reflex_int4_reasoning_prompt_protection_max_pages = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_REASONING_PROMPT_PROTECTION_MAX_PAGES",
                self._reflex_int4_short_prefill_pages,
            )
        )
        self._reflex_int4_reasoning_prompt_protection_min_decode_tokens = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_REASONING_PROMPT_PROTECTION_MIN_DECODE_TOKENS",
                1024,
            )
        )
        self._reflex_int4_page_level_protection_enabled = self._read_reflex_bool_env(
            "SEMANTIQ_REFLEX_ENABLE_PAGE_LEVEL_PROTECTION",
            True,
        )
        self._reflex_int4_long_prompt_protected_head_pages = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_LONG_PROMPT_PROTECTED_HEAD_PAGES",
                self._reflex_int4_keep_initial_pages,
            )
        )
        self._reflex_int4_long_prompt_protected_tail_pages = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_LONG_PROMPT_PROTECTED_TAIL_PAGES",
                self._reflex_int4_keep_initial_pages,
            )
        )
        self._reflex_int4_prompt_high_risk_protection_threshold = (
            self._read_reflex_float_env(
                "SEMANTIQ_REFLEX_PROMPT_HIGH_RISK_PROTECTION_THRESHOLD",
                0.85,
                minimum=0.0,
                maximum=1.0,
            )
        )
        self._reflex_int4_background_demotions_per_step = 16
        self._reflex_int4_background_min_demotions_per_step = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_BACKGROUND_MIN_DEMOTIONS_PER_STEP",
                min(8, self._reflex_int4_background_demotions_per_step),
            )
        )
        self._reflex_int4_background_free_floor_blocks = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_BACKGROUND_FREE_FLOOR_BLOCKS",
                max(
                    self._reflex_int4_background_demotions_per_step,
                    2 * self.max_num_running_reqs,
                ),
            )
        )
        self._reflex_int4_fast_demotions_per_step = max(
            self._reflex_int4_background_demotions_per_step,
            self._estimate_required_blocks(
                max(self.max_num_scheduled_tokens, self.max_model_len)
            ),
        )
        self._reflex_int4_admission_reserve_blocks = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_ADMISSION_RESERVE_BLOCKS",
                32,
            )
        )
        self._reflex_int4_recovery_shadow_pages_per_request = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_BF16_SHADOW_PAGES_PER_REQUEST",
                DEFAULT_REFLEX_BF16_SHADOW_PAGES_PER_REQUEST,
            )
        )
        self._reflex_int4_background_promotion_free_ratio = self._read_reflex_float_env(
            "SEMANTIQ_REFLEX_BACKGROUND_PROMOTION_FREE_RATIO",
            0.60,
            minimum=0.0,
            maximum=1.0,
        )
        self._reflex_int4_background_promotion_pages_per_step = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_BACKGROUND_PROMOTION_PAGES_PER_STEP",
                DEFAULT_REFLEX_BACKGROUND_PROMOTION_PAGES_PER_STEP,
            )
        )
        self._reflex_int4_promotion_min_remaining_decode_tokens = (
            self._read_reflex_nonnegative_int_env(
                "SEMANTIQ_REFLEX_PROMOTION_MIN_REMAINING_DECODE_TOKENS",
                16,
            )
        )

        self.use_pp = self.parallel_config.pipeline_parallel_size > 1
        self.use_v2_model_runner = envs.VLLM_USE_V2_MODEL_RUNNER
        self.scheduler_reserve_full_isl = (
            self.scheduler_config.scheduler_reserve_full_isl
        )

        self.has_mamba_layers = kv_cache_config.has_mamba_layers
        self.needs_kv_cache_zeroing = kv_cache_config.needs_kv_cache_zeroing
        self.need_mamba_block_aligned_split = (
            self.has_mamba_layers and self.cache_config.mamba_cache_mode == "align"
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)

        if self.vllm_config.model_config.enable_return_routed_experts:
            assert self.dcp_world_size == 1 and self.pcp_world_size == 1, (
                "enable_return_routed_experts does not support context parallelism "
                "(dcp_world_size > 1 or pcp_world_size > 1)"
            )

            self.routed_experts_reader = RoutedExpertsReader.create()

            assert len(kv_cache_config.kv_cache_groups) > 0, (
                "enable_return_routed_experts requires at least one kv cache group"
            )
            # Find the attention group for routed experts indexing.
            self.routed_experts_attn_gid = 0
            for gid, group in enumerate(kv_cache_config.kv_cache_groups):
                if isinstance(group.kv_cache_spec, AttentionSpec):
                    self.routed_experts_attn_gid = gid
                    break
            min_block_size = min(
                [
                    group.kv_cache_spec.block_size
                    for group in kv_cache_config.kv_cache_groups
                ]
            )
            num_groups = len(kv_cache_config.kv_cache_groups)
            self.max_num_kv_tokens = (
                kv_cache_config.num_blocks // num_groups
            ) * min_block_size
            dcp_size = self.vllm_config.parallel_config.decode_context_parallel_size
            pcp_size = self.vllm_config.parallel_config.prefill_context_parallel_size
            if pcp_size * dcp_size > 1:
                self.max_num_kv_tokens *= pcp_size * dcp_size

            self.routed_experts_reader.attach_buffer(
                max_num_kv_tokens=self.max_num_kv_tokens,
                vllm_config=self.vllm_config,
            )

        self._pause_state: PauseState = PauseState.UNPAUSED

    def _mamba_block_aligned_split(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_local_computed_tokens: int = 0,
        num_external_computed_tokens: int = 0,
    ) -> int:
        assert num_external_computed_tokens == 0, (
            "External KV connector is not verified yet"
        )
        num_computed_tokens = (
            request.num_computed_tokens
            + num_new_local_computed_tokens
            + num_external_computed_tokens
        )
        # Perform block-aligned splitting at prefill phase, including:
        # * non-resumed requests: num_computed_tokens < num_prompt_tokens + 0
        # * resumed requests: num_computed_tokens < (
        #                       num_prompt_tokens + num_output_tokens
        #                     )
        # NOTE: Use `request.num_tokens - 1` to bypass normal decoding.
        if num_computed_tokens < max(request.num_prompt_tokens, request.num_tokens - 1):
            # To enable block-aligned caching of the Mamba state, `num_new_tokens`
            # must be a multiple of `block_size`.
            # As an exception, if `num_new_tokens` is less than `block_size`, the
            # state is simply not cached, requiring no special handling.
            # Additionally, when Eagle mode is enabled, FullAttn prunes the last
            # matching block. To prevent this from causing a Mamba cache miss, the
            # last chunk must be not smaller than `block_size`.
            block_size = self.cache_config.block_size
            last_cache_position = request.num_tokens - request.num_tokens % block_size
            # eagle prune
            if self.use_eagle:
                last_cache_position = max(last_cache_position - block_size, 0)
            num_computed_tokens_after_sched = num_computed_tokens + num_new_tokens
            if num_computed_tokens_after_sched < last_cache_position:
                # align to block_size
                num_new_tokens = num_new_tokens // block_size * block_size
            elif (
                num_computed_tokens
                < last_cache_position
                < num_computed_tokens_after_sched
            ):
                # force to cache the last chunk
                num_new_tokens = last_cache_position - num_computed_tokens
            else:
                # prefill the last few tokens
                pass
        return num_new_tokens

    def schedule(self) -> SchedulerOutput:
        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self._reflex_int4_scheduler_step += 1
        if self.finished_req_ids:
            self._record_reflex_int4_frontier_event("request_finished")
        self.kv_cache_manager.new_step_starts()
        reflex_demotion_only_output = self._try_reflex_int4_demotion_only_step()
        if reflex_demotion_only_output is not None:
            return reflex_demotion_only_output
        reflex_defer_decode_for_prefill = (
            self._reflex_int4_should_defer_decode_for_prefill()
        )

        # First, schedule the RUNNING requests.
        req_index = 0
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if self._reflex_int4_should_defer_running_request(
                request, reflex_defer_decode_for_prefill
            ):
                req_index += 1
                continue

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)
            num_new_tokens = self._limit_reflex_remote_decode_chunk_tokens(
                request,
                num_new_tokens,
            )

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    if (
                        self._try_reflex_int4_demote(
                            target_bf16_blocks=self._estimate_reflex_demote_target(
                                num_new_tokens + self.num_lookahead_tokens,
                                force_allocate_failure=True,
                            ),
                            force=True,
                            reason="allocation_failure",
                        )
                        > 0
                    ):
                        # Demoted BF16 blocks are released at the next scheduler
                        # step, after the worker has copied them into INT4.
                        # Retrying allocation in the same step can reuse the
                        # source blocks before the copy runs.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    preempted_req = self._select_preemption_victim(
                        self.running,
                        allow_reflex_int4_protected=(
                            self._should_preempt_reflex_int4_protected_request()
                        ),
                    )
                    if preempted_req is None:
                        break
                    self.running.remove(preempted_req)
                    if preempted_req in scheduled_running_reqs:
                        preempted_req_id = preempted_req.request_id
                        scheduled_running_reqs.remove(preempted_req)
                        token_budget += num_scheduled_tokens.pop(preempted_req_id)
                        req_to_new_blocks.pop(preempted_req_id)
                        scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                        preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                            preempted_req_id, None
                        )
                        if preempted_encoder_inputs:
                            # Restore encoder compute budget if the preempted
                            # request had encoder inputs scheduled in this step.
                            num_embeds_to_restore = sum(
                                preempted_req.get_num_encoder_embeds(i)
                                for i in preempted_encoder_inputs
                            )
                            encoder_compute_budget += num_embeds_to_restore
                        req_index -= 1

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            self._update_reflex_remote_decode_chunk_send_after_alloc(
                request,
                new_blocks,
                num_new_tokens,
            )
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                if self._should_skip_reflex_int4_waiting_request_by_ticket(request):
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens.
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                elif self._should_load_reflex_remote_prefill_chunk(request):
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    ext_tokens, load_kv_async = (
                        self.connector.get_num_new_matched_tokens(
                            request,
                            request.num_computed_tokens,
                        )
                    )
                    if ext_tokens is None:
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    request.num_external_computed_tokens = ext_tokens
                    num_external_computed_tokens = ext_tokens
                    connector_prefix_cache_queries = (
                        request.num_tokens - request.num_computed_tokens
                    )
                    connector_prefix_cache_hits = num_external_computed_tokens
                    num_computed_tokens = (
                        request.num_computed_tokens + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    num_new_tokens = min(num_new_tokens, token_budget)
                    num_new_tokens = self._limit_reflex_remote_decode_chunk_tokens(
                        request,
                        num_new_tokens,
                    )
                    assert num_new_tokens > 0

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                if (
                    self.scheduler_reserve_full_isl
                    and load_kv_async
                    and self._is_reflex_remote_chunk_load(request)
                ):
                    full_sequence_fits = True
                elif self.scheduler_reserve_full_isl:
                    full_sequence_fits = self.kv_cache_manager.can_fit_full_sequence(
                        request,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_external_computed_tokens=num_external_computed_tokens,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                else:
                    full_sequence_fits = True

                if self.scheduler_reserve_full_isl and not full_sequence_fits:
                    landing_decision = None
                    frontier_summary_for_ticket = None
                    if not num_scheduled_tokens:
                        landing_decision = (
                            self._plan_and_persist_reflex_int4_landing_contract(
                                request=request,
                                reason="full_sequence_reserve",
                                num_lookahead_tokens=effective_lookahead_tokens,
                            )
                        )
                        frontier_summary_for_ticket = (
                            self._get_reflex_int4_frontier_cache().latest()
                        )
                        self._try_reflex_int4_demote(
                            target_bf16_blocks=(
                                self._estimate_reflex_admission_demote_target(
                                    request,
                                    num_lookahead_tokens=effective_lookahead_tokens,
                                )
                            ),
                            force=True,
                            reason="full_sequence_reserve",
                        )
                        if (
                            landing_decision is not None
                            and landing_decision.admission_feasible_with_landing
                            and landing_decision.planned_int4_landing_blocks
                            >= landing_decision.residual_deficit_after_running
                            and self._has_reflex_int4_landing_contract(request)
                            and self._can_reflex_int4_mixed_landing_close_admission_gap()
                        ):
                            full_sequence_fits = True
                    if full_sequence_fits:
                        logger.info(
                            "ReFlexKV trace landing_policy request=%s "
                            "outcome=admit_with_mixed_landing planned_pages=%d "
                            "bf16_deficit=%d.",
                            request_id,
                            landing_decision.planned_int4_landing_blocks
                            if landing_decision is not None
                            else 0,
                            landing_decision.bf16_deficit_blocks
                            if landing_decision is not None
                            else 0,
                        )
                    else:
                        if request.has_encoder_inputs:
                            self.encoder_cache_manager.free(request)
                        if self.cache_config.cache_dtype == "reflex_int4":
                            self._clear_reflex_int4_landing_contract(request)
                            required_blocks = (
                                landing_decision.required_blocks
                                if landing_decision is not None
                                else (
                                    self._estimate_reflex_admission_needed_blocks(
                                        request
                                    )
                                    + self._reflex_int4_admission_reserve_blocks
                                )
                            )
                            ticket = self._record_reflex_int4_admission_ticket(
                                request=request,
                                required_blocks=required_blocks,
                                blocked_reason="full_sequence_reserve",
                                cached_frontier_summary=frontier_summary_for_ticket,
                            )
                            logger.info(
                                "ReFlexKV trace admission_control request=%s "
                                "outcome=defer_full_sequence_reserve "
                                "reason=no_progress_after_full_sequence_reserve "
                                "blocked_reason=%s next_retry_step=%d "
                                "frontier_age=%d frontier_levels=%s "
                                "frontier_rejection_reasons=%s.",
                                request_id,
                                ticket.blocked_reason,
                                ticket.next_retry_step,
                                self._reflex_frontier_age_or_minus_one(
                                    ticket.cached_frontier_summary
                                ),
                                self._format_reflex_frontier_levels(
                                    ticket.cached_frontier_summary
                                ),
                                self._format_reflex_frontier_rejection_reasons(
                                    ticket.cached_frontier_summary
                                ),
                            )
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue
                        break

                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_new_computed_tokens=num_new_local_computed_tokens,
                        new_computed_blocks=new_computed_blocks,
                        num_lookahead_tokens=effective_lookahead_tokens,
                        num_external_computed_tokens=num_external_computed_tokens,
                        delay_cache_blocks=load_kv_async,
                        num_encoder_tokens=num_encoder_tokens,
                    )
                    if new_blocks is not None:
                        break
                    if not num_scheduled_tokens:
                        self._try_reflex_int4_demote(
                            target_bf16_blocks=(
                                self._reflex_int4_allocation_failure_demote_target(
                                    request,
                                    num_new_tokens=num_new_tokens,
                                    num_lookahead_tokens=effective_lookahead_tokens,
                                )
                            ),
                            force=True,
                            reason="allocation_failure",
                        )
                    break

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    if self.cache_config.cache_dtype == "reflex_int4":
                        ticket = self._record_reflex_int4_admission_ticket(
                            request=request,
                            required_blocks=(
                                self._estimate_reflex_admission_needed_blocks(request)
                                + self._reflex_int4_admission_reserve_blocks
                            ),
                            blocked_reason="allocation_failure",
                        )
                        logger.info(
                            "ReFlexKV trace admission_control request=%s "
                            "outcome=defer_allocation_failure "
                            "reason=no_progress_after_allocation_failure "
                            "next_retry_step=%d.",
                            request_id,
                            ticket.next_retry_step,
                        )
                        request_queue.pop_request()
                        step_skipped_waiting.prepend_request(request)
                        continue
                    break

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self._update_reflex_remote_decode_chunk_send_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_new_tokens,
                    )
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                self._clear_reflex_int4_admission_ticket(request_id)
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    request.reflex_remote_transfer_start_time = time.perf_counter()
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )
        reflex_int4_demotions = (
            self.kv_cache_manager.take_reflex_int4_demotions() or None
        )
        self._try_reflex_int4_background_promote()
        reflex_int4_recoveries = (
            self.kv_cache_manager.take_reflex_int4_recoveries() or None
            if hasattr(self.kv_cache_manager, "take_reflex_int4_recoveries")
            else None
        )

        scheduler_output = SchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            reflex_int4_demotions=reflex_int4_demotions,
            reflex_int4_recoveries=reflex_int4_recoveries,
        )
        self._reflex_int4_prev_step_had_prefill = self._reflex_int4_step_has_prefill(
            num_scheduled_tokens
        )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _estimate_required_blocks(self, num_tokens: int) -> int:
        return max(1, (num_tokens + self.block_size - 1) // self.block_size)

    @staticmethod
    def _request_kv_transfer_params(request: Request) -> dict[str, Any] | None:
        params = getattr(request, "kv_transfer_params", None)
        return params if isinstance(params, dict) else None

    def _should_load_reflex_remote_prefill_chunk(self, request: Request) -> bool:
        if self.connector is None:
            return False
        params = self._request_kv_transfer_params(request)
        return bool(
            params
            and params.get("do_remote_prefill")
            and remote_chunking_enabled(params)
        )

    def _is_reflex_remote_chunk_load(self, request: Request) -> bool:
        params = self._request_kv_transfer_params(request)
        return bool(params and remote_chunking_enabled(params))

    def _reflex_remote_chunk_sealed_pages(self, request: Request) -> int:
        params = self._request_kv_transfer_params(request)
        if not (params and remote_chunking_enabled(params)):
            return 0
        committed_page_end = params.get("reflex_remote_chunk_committed_page_end")
        chunk_inflight = bool(params.get("reflex_remote_chunk_inflight"))
        if committed_page_end is not None:
            try:
                committed_pages = max(0, int(committed_page_end))
            except (TypeError, ValueError):
                committed_pages = 0
            if chunk_inflight:
                return committed_pages
            if committed_pages > 0:
                return committed_pages
        elif chunk_inflight:
            return 0
        raw_page_end = params.get("reflex_remote_chunk_page_end")
        if raw_page_end is not None:
            try:
                return max(0, int(raw_page_end))
            except (TypeError, ValueError):
                return 0
        raw_token_end = params.get("reflex_remote_chunk_token_end")
        if raw_token_end is not None:
            try:
                return max(0, int(raw_token_end) // self.block_size)
            except (TypeError, ValueError):
                return 0
        return 0

    def _commit_reflex_remote_chunk(self, request: Request) -> None:
        params = self._request_kv_transfer_params(request)
        if not (params and remote_chunking_enabled(params)):
            return
        raw_chunk_id = params.get("reflex_remote_chunk_id")
        raw_token_end = params.get("reflex_remote_chunk_token_end")
        raw_page_end = params.get("reflex_remote_chunk_page_end")
        if raw_chunk_id is None or raw_page_end is None:
            return
        try:
            chunk_id = int(raw_chunk_id)
            token_end = max(0, int(raw_token_end or 0))
            page_end = max(0, int(raw_page_end))
        except (TypeError, ValueError):
            return
        try:
            previous_token_end = int(
                params.get("reflex_remote_chunk_committed_token_end", 0) or 0
            )
            previous_page_end = int(
                params.get("reflex_remote_chunk_committed_page_end", 0) or 0
            )
        except (TypeError, ValueError):
            previous_token_end = 0
            previous_page_end = 0
        committed_token_end = max(
            previous_token_end,
            token_end,
        )
        committed_page_end = max(
            previous_page_end,
            page_end,
        )
        params["reflex_remote_chunk_committed_token_end"] = committed_token_end
        params["reflex_remote_chunk_committed_page_end"] = committed_page_end
        params["reflex_remote_chunk_inflight"] = False
        logger.info(
            "ReFlexKV trace remote_chunk_commit request=%s chunk_id=%d "
            "token_end=%d page_end=%d committed_token_end=%d "
            "committed_page_end=%d is_last=%s.",
            request.request_id,
            chunk_id,
            token_end,
            page_end,
            committed_token_end,
            committed_page_end,
            bool(params.get("reflex_remote_chunk_is_last", False)),
        )

    def _reflex_remote_chunk_inflight_pages(self, request: Request) -> set[int]:
        params = self._request_kv_transfer_params(request)
        if not (
            params
            and remote_chunking_enabled(params)
            and params.get("reflex_remote_chunk_inflight")
        ):
            return set()
        try:
            committed_page_end = max(
                0,
                int(params.get("reflex_remote_chunk_committed_page_end", 0) or 0),
            )
            page_start = max(
                committed_page_end,
                int(params.get("reflex_remote_chunk_page_start", 0) or 0),
            )
            page_end = max(0, int(params.get("reflex_remote_chunk_page_end", 0) or 0))
        except (TypeError, ValueError):
            return set()
        if page_end <= page_start:
            return set()
        return set(range(page_start, page_end))

    @staticmethod
    def _reflex_copy_on_demote_pages(request: Request) -> set[int]:
        params = Scheduler._request_kv_transfer_params(request)
        if not params:
            return set()
        raw_pages = params.get("reflex_copy_on_demote_pages")
        if raw_pages is None:
            raw_pages = params.get("reflex_prefix_copy_on_demote_pages")
        if raw_pages is None:
            return set()
        if isinstance(raw_pages, int):
            raw_iterable = (raw_pages,)
        elif isinstance(raw_pages, (list, tuple, set, frozenset)):
            raw_iterable = raw_pages
        else:
            return set()
        pages: set[int] = set()
        for raw_page in raw_iterable:
            try:
                page_idx = int(raw_page)
            except (TypeError, ValueError):
                continue
            if page_idx >= 0:
                pages.add(page_idx)
        return pages

    def _limit_reflex_remote_decode_chunk_tokens(
        self,
        request: Request,
        num_new_tokens: int,
    ) -> int:
        params = self._request_kv_transfer_params(request)
        if not (
            params
            and params.get("do_remote_decode")
            and remote_chunking_enabled(params)
        ):
            return num_new_tokens
        prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
        if computed_tokens >= prompt_tokens:
            return num_new_tokens
        chunk_tokens = normalize_remote_chunk_tokens(
            int(params.get("reflex_remote_chunk_tokens", 512)),
            self.block_size,
        )
        remaining_prompt_tokens = prompt_tokens - computed_tokens
        return max(1, min(num_new_tokens, chunk_tokens, remaining_prompt_tokens))

    def _update_reflex_remote_decode_chunk_send_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_scheduled_tokens: int,
    ) -> None:
        if self.connector is None or num_scheduled_tokens <= 0:
            return
        params = self._request_kv_transfer_params(request)
        if not (
            params
            and params.get("do_remote_decode")
            and remote_chunking_enabled(params)
        ):
            return
        update_chunk = getattr(
            self.connector,
            "update_reflex_remote_decode_chunk_after_alloc",
            None,
        )
        if callable(update_chunk):
            update_chunk(request, blocks, num_scheduled_tokens)

    @staticmethod
    def _read_reflex_nonnegative_int_env(name: str, default: int) -> int:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        try:
            return max(0, int(raw_value))
        except ValueError:
            logger.warning(
                "Invalid %s=%r for ReFlexKV; using default %d.",
                name,
                raw_value,
                default,
            )
            return default

    @staticmethod
    def _read_reflex_bool_env(name: str, default: bool) -> bool:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        logger.warning(
            "Invalid %s=%r for ReFlexKV; using default %s.",
            name,
            raw_value,
            default,
        )
        return default

    @staticmethod
    def _read_reflex_float_env(
        name: str,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        try:
            value = float(raw_value)
        except ValueError:
            logger.warning(
                "Invalid %s=%r for ReFlexKV; using default %.4f.",
                name,
                raw_value,
                default,
            )
            return default
        if value < minimum or value > maximum:
            logger.warning(
                "Invalid %s=%r for ReFlexKV; expected %.4f <= value <= %.4f; "
                "using default %.4f.",
                name,
                raw_value,
                minimum,
                maximum,
                default,
            )
            return default
        return value

    def _estimate_reflex_demote_target(
        self,
        num_tokens: int,
        *,
        force_allocate_failure: bool = False,
    ) -> int:
        if self.cache_config.cache_dtype != "reflex_int4":
            return 0
        required_blocks = self._estimate_required_blocks(num_tokens)
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        total_blocks = max(1, self.kv_cache_manager.block_pool.num_gpu_blocks)
        free_ratio = free_blocks / total_blocks
        deficit_blocks = max(0, required_blocks - free_blocks)

        if not force_allocate_failure and free_ratio > self._reflex_int4_high_watermark:
            return 0
        if deficit_blocks <= 0 and free_ratio >= self._reflex_int4_low_watermark:
            return 0

        if force_allocate_failure:
            target_blocks = deficit_blocks if deficit_blocks > 0 else required_blocks
            limit = self._reflex_int4_fast_demotions_per_step
        else:
            if deficit_blocks > 0:
                target_blocks = deficit_blocks
            else:
                background_floor = max(
                    0,
                    getattr(
                        self,
                        "_reflex_int4_background_free_floor_blocks",
                        self._reflex_int4_background_demotions_per_step,
                    ),
                )
                if free_blocks >= background_floor:
                    return 0
                target_blocks = background_floor - free_blocks
                min_background_batch = max(
                    0,
                    getattr(
                        self,
                        "_reflex_int4_background_min_demotions_per_step",
                        0,
                    ),
                )
                if min_background_batch > 0:
                    target_blocks = max(target_blocks, min_background_batch)
            limit = self._reflex_int4_background_demotions_per_step
        return min(target_blocks, limit)

    def _estimate_reflex_admission_needed_blocks(
        self,
        request: Request,
        *,
        num_new_tokens: int | None = None,
        num_lookahead_tokens: int = 0,
    ) -> int:
        chunk_target_tokens = self._reflex_remote_chunk_admission_target_tokens(
            request,
            num_new_tokens=num_new_tokens,
        )
        if self.scheduler_reserve_full_isl and chunk_target_tokens is None:
            target_tokens = min(request.num_tokens, self.max_model_len)
        else:
            if chunk_target_tokens is not None:
                target_tokens = chunk_target_tokens
            else:
                if num_new_tokens is None:
                    remaining_tokens = max(
                        request.num_tokens - request.num_computed_tokens,
                        1,
                    )
                    num_new_tokens = min(
                        remaining_tokens,
                        self.max_num_scheduled_tokens,
                    )
                target_tokens = request.num_computed_tokens + num_new_tokens
            target_tokens = min(target_tokens, self.max_model_len)

        if num_lookahead_tokens > 0:
            target_tokens = min(
                target_tokens + num_lookahead_tokens,
                self.max_model_len,
            )
        required_blocks = self._estimate_required_blocks(target_tokens)
        existing_blocks = self._reflex_int4_existing_allocated_blocks(request)
        return max(0, required_blocks - existing_blocks)

    def _reflex_remote_chunk_admission_target_tokens(
        self,
        request: Request,
        *,
        num_new_tokens: int | None = None,
    ) -> int | None:
        if not self._is_reflex_remote_chunk_load(request):
            return None
        params = self._request_kv_transfer_params(request)
        if params is None:
            return None
        prompt_tokens = int(
            getattr(request, "num_prompt_tokens", 0)
            or getattr(request, "num_tokens", 0)
            or 0
        )
        if prompt_tokens <= 0:
            return None
        computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
        raw_chunk_end = params.get("reflex_remote_chunk_token_end")
        try:
            chunk_end = int(raw_chunk_end) if raw_chunk_end is not None else 0
        except (TypeError, ValueError):
            chunk_end = 0
        if chunk_end <= computed_tokens:
            if num_new_tokens is not None and num_new_tokens > 0:
                chunk_tokens = int(num_new_tokens)
            else:
                chunk_tokens = normalize_remote_chunk_tokens(
                    int(params.get("reflex_remote_chunk_tokens", 512)),
                    self.block_size,
                )
            chunk_end = computed_tokens + max(1, chunk_tokens)
        return min(max(chunk_end, computed_tokens), prompt_tokens, self.max_model_len)

    def _reflex_int4_existing_allocated_blocks(self, request: Request) -> int:
        get_blocks = getattr(self.kv_cache_manager, "get_blocks", None)
        if not callable(get_blocks):
            return 0
        request_id = getattr(request, "request_id", None)
        if request_id is None:
            return 0
        try:
            blocks = get_blocks(request_id)
        except (KeyError, RuntimeError, ValueError):
            return 0
        block_groups = getattr(blocks, "blocks", blocks)
        if not isinstance(block_groups, (list, tuple)):
            return 0
        if not block_groups:
            return 0
        first_group = block_groups[0]
        if not isinstance(first_group, (list, tuple)):
            return len(block_groups)
        return sum(len(group) for group in block_groups)

    def _estimate_reflex_admission_demote_target(
        self,
        request: Request,
        *,
        num_new_tokens: int | None = None,
        num_lookahead_tokens: int = 0,
        reserve_blocks: int | None = None,
    ) -> int:
        if self.cache_config.cache_dtype != "reflex_int4":
            return 0
        needed_blocks = self._estimate_reflex_admission_needed_blocks(
            request,
            num_new_tokens=num_new_tokens,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        if reserve_blocks is None:
            reserve_blocks = self._reflex_int4_admission_reserve_blocks
        return max(0, needed_blocks + reserve_blocks - free_blocks)

    def _reflex_int4_allocation_failure_demote_target(
        self,
        request: Request,
        *,
        num_new_tokens: int | None = None,
        num_lookahead_tokens: int = 0,
    ) -> int:
        target_blocks = self._estimate_reflex_admission_demote_target(
            request,
            num_new_tokens=num_new_tokens,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        if target_blocks <= 0:
            return 0
        # An exact one-block release can be consumed by the active decode
        # request before the waiting prefill retries. Keep a small admission
        # slack floor on allocation failures so the retry can make progress.
        target_blocks = max(
            target_blocks,
            self._reflex_int4_admission_reserve_blocks,
        )
        return min(target_blocks, self._reflex_int4_fast_demotions_per_step)

    def _get_precision_admission_controller(
        self,
    ) -> PrecisionAdmissionController:
        controller = getattr(self, "_precision_kv_admission_controller", None)
        if controller is None:
            controller = PrecisionAdmissionController()
            self._precision_kv_admission_controller = controller
        return controller

    def _get_precision_landing_planner(
        self,
    ) -> PrecisionLandingPlanner:
        planner = getattr(self, "_precision_kv_landing_planner", None)
        if planner is None:
            planner = PrecisionLandingPlanner()
            self._precision_kv_landing_planner = planner
        return planner

    def _get_precision_kv_policy(self) -> PrecisionKVPolicy:
        policy = getattr(self, "_precision_kv_policy", None)
        if policy is None:
            policy = PrecisionKVPolicy()
            self._precision_kv_policy = policy
        return policy

    @staticmethod
    def _reflex_len(value: Any) -> int:
        if value is None:
            return 0
        try:
            return len(value)
        except TypeError:
            return 1 if value else 0

    def _plan_reflex_int4_pressure_policy(
        self,
        *,
        reason: str,
        target_bf16_blocks: int,
    ) -> PrecisionPressureDecision:
        block_pool = self.kv_cache_manager.block_pool
        return self._get_precision_kv_policy().plan_pressure(
            PrecisionPressureState(
                reason=reason,
                target_bf16_blocks=target_bf16_blocks,
                free_bf16_blocks=block_pool.get_num_free_blocks(),
                total_bf16_blocks=block_pool.num_gpu_blocks,
                waiting_requests=self._reflex_len(getattr(self, "waiting", None)),
                skipped_waiting_requests=self._reflex_len(
                    getattr(self, "skipped_waiting", None)
                ),
                candidate_funnel=CandidateFunnelSnapshot.from_object(
                    getattr(self, "_reflex_int4_last_candidate_breakdown", None)
                ),
                base_low_risk_score_fraction=(
                    self._reflex_int4_low_risk_score_fraction
                ),
            )
        )

    def _reflex_int4_landing_eligible_blocks(self, request: Request) -> int:
        return len(self._reflex_int4_landing_eligible_pages(request))

    def _reflex_int4_landing_eligible_pages(
        self,
        request: Request,
        *,
        max_int4_fraction: float | None = None,
        respect_global_evidence_cap: bool = True,
    ) -> tuple[int, ...]:
        if self.cache_config.cache_dtype != "reflex_int4":
            return ()

        params = getattr(request, "kv_transfer_params", None) or {}
        if not isinstance(params, dict):
            return ()

        raw_risks = params.get("reflex_page_risks")
        page_risks: list[float] = []
        if isinstance(raw_risks, (list, tuple)):
            page_risks = [float(score) for score in raw_risks]

        explicit_pages: set[int] | None = None
        raw_pages = params.get("reflex_compressible_pages")
        if isinstance(raw_pages, (list, tuple, set)):
            explicit_pages = {int(page_idx) for page_idx in raw_pages}
        else:
            raw_mask = params.get("reflex_compressible_mask")
            if isinstance(raw_mask, (list, tuple)):
                explicit_pages = {
                    idx for idx, enabled in enumerate(raw_mask) if bool(enabled)
                }

        request_tokens = min(
            max(0, int(getattr(request, "num_tokens", 0))),
            self.max_model_len,
        )
        request_pages = (
            (request_tokens + self.block_size - 1) // self.block_size
            if request_tokens > 0
            else 0
        )
        page_count = max(request_pages, len(page_risks))
        if explicit_pages:
            page_count = max(page_count, max(explicit_pages) + 1)
        if page_count <= 0:
            return ()
        existing_int4_pages = self._reflex_int4_existing_int4_page_indices(
            getattr(request, "request_id", "")
        )
        chunk_range = self._reflex_remote_chunk_page_range_for_landing(
            request,
            page_count=page_count,
        )

        if explicit_pages is None and not page_risks:
            if max_int4_fraction is None:
                max_int4_fraction = self._reflex_int4_max_int4_fraction_per_request
            if respect_global_evidence_cap:
                max_int4_fraction = self._reflex_int4_landing_fraction_for_request(
                    request,
                    max_int4_fraction,
                )
            if chunk_range is None or not bool(
                getattr(self, "_reflex_int4_direct_landing_enabled", False)
            ):
                return ()
            synthetic_pages = synthesize_remote_chunk_landing_pages(
                page_start=chunk_range[0],
                page_end=chunk_range[1],
                page_count=page_count,
                keep_initial_pages=self._reflex_int4_keep_initial_pages,
                keep_recent_pages=self._reflex_int4_keep_recent_pages,
                protected_prompt_pages=(
                    self._reflex_int4_protected_prompt_pages(request)
                ),
                max_int4_fraction=max_int4_fraction,
                short_prefill_pages=self._reflex_int4_short_prefill_pages,
            )
            protected_prompt_page_indices = (
                self._reflex_int4_protected_prompt_page_indices(request)
            )
            return tuple(
                page_idx
                for page_idx in synthetic_pages
                if page_idx not in protected_prompt_page_indices
                and page_idx not in existing_int4_pages
            )

        candidate_pages: list[int] = []
        seen_pages: set[int] = set()
        if explicit_pages is not None:
            for page_idx in sorted(explicit_pages):
                candidate_pages.append(page_idx)
                seen_pages.add(page_idx)
        if page_risks:
            for page_idx in sorted(
                range(len(page_risks)),
                key=lambda idx: (page_risks[idx], idx),
            ):
                if page_idx in seen_pages:
                    continue
                candidate_pages.append(page_idx)
                seen_pages.add(page_idx)

        keep_initial_pages = max(0, self._reflex_int4_keep_initial_pages)
        keep_recent_pages = max(0, self._reflex_int4_keep_recent_pages)
        protected_prompt_pages = self._reflex_int4_protected_prompt_pages(request)
        protected_prompt_page_indices = self._reflex_int4_protected_prompt_page_indices(
            request
        )
        recent_start = max(0, page_count - keep_recent_pages)
        chunk_start, chunk_end = (
            (0, page_count) if chunk_range is None else (chunk_range[0], chunk_range[1])
        )
        filtered_pages = [
            page_idx
            for page_idx in candidate_pages
            if 0 <= page_idx < page_count
            and chunk_start <= page_idx < chunk_end
            and page_idx >= keep_initial_pages
            and page_idx >= protected_prompt_pages
            and page_idx not in protected_prompt_page_indices
            and page_idx not in existing_int4_pages
            and page_idx < recent_start
        ]

        if max_int4_fraction is None:
            max_int4_fraction = self._reflex_int4_max_int4_fraction_per_request
        if respect_global_evidence_cap:
            max_int4_fraction = self._reflex_int4_landing_fraction_for_request(
                request,
                max_int4_fraction,
            )
        max_int4_fraction = min(
            1.0,
            max(0.0, max_int4_fraction),
        )
        max_int4_pages = int(page_count * max_int4_fraction)
        return tuple(filtered_pages[: max(0, max_int4_pages)])

    def _reflex_int4_landing_metadata_trace_fields(
        self,
        request: Request,
    ) -> tuple[str, int, int, int]:
        params = getattr(request, "kv_transfer_params", None) or {}
        if not isinstance(params, dict):
            return "none", 0, 0, 0

        raw_risks = params.get("reflex_page_risks")
        real_risk_pages = len(raw_risks) if isinstance(raw_risks, (list, tuple)) else 0

        explicit_pages = 0
        raw_pages = params.get("reflex_compressible_pages")
        if isinstance(raw_pages, (list, tuple, set)):
            explicit_pages = len(raw_pages)
        else:
            raw_mask = params.get("reflex_compressible_mask")
            if isinstance(raw_mask, (list, tuple)):
                explicit_pages = sum(1 for enabled in raw_mask if bool(enabled))

        synthetic_pages = 0
        source_parts: list[str] = []
        if explicit_pages > 0:
            source_parts.append("explicit_compressible")
        if real_risk_pages > 0:
            source_parts.append("real_risk")
        if not source_parts:
            synthetic_pages = len(self._reflex_int4_landing_eligible_pages(request))
            if synthetic_pages > 0:
                source_parts.append("synthetic_chunk")
        if not source_parts:
            source_parts.append("none")
        return "_".join(source_parts), real_risk_pages, explicit_pages, synthetic_pages

    def _reflex_remote_chunk_page_range_for_landing(
        self,
        request: Request,
        *,
        page_count: int,
    ) -> tuple[int, int] | None:
        params = self._request_kv_transfer_params(request)
        if not (
            params
            and params.get("do_remote_prefill")
            and remote_chunking_enabled(params)
        ):
            return None
        raw_page_start = params.get("reflex_remote_chunk_page_start")
        raw_page_end = params.get("reflex_remote_chunk_page_end")
        try:
            page_start = int(raw_page_start) if raw_page_start is not None else None
            page_end = int(raw_page_end) if raw_page_end is not None else None
        except (TypeError, ValueError):
            page_start = None
            page_end = None
        if page_start is None or page_end is None or page_end <= page_start:
            computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
            chunk_end_tokens = self._reflex_remote_chunk_admission_target_tokens(
                request
            )
            if chunk_end_tokens is None or chunk_end_tokens <= computed_tokens:
                return None
            page_start = computed_tokens // self.block_size
            page_end = (chunk_end_tokens + self.block_size - 1) // self.block_size
        page_start = max(0, min(int(page_start), page_count))
        page_end = max(page_start, min(int(page_end), page_count))
        if page_end <= page_start:
            return None
        return page_start, page_end

    def _reflex_int4_is_global_evidence_request(self, request: Request) -> bool:
        try:
            prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0
        if prompt_tokens <= 0:
            try:
                prompt_tokens = int(getattr(request, "num_tokens", 0) or 0)
            except (TypeError, ValueError):
                prompt_tokens = 0
        prompt_pages = (
            (prompt_tokens + self.block_size - 1) // self.block_size
            if prompt_tokens > 0
            else 0
        )
        min_prompt_pages = getattr(
            self,
            "_reflex_int4_global_evidence_min_prompt_pages",
            self._reflex_int4_long_prefill_pages,
        )
        min_decode_tokens = getattr(
            self,
            "_reflex_int4_global_evidence_min_decode_tokens",
            self._reflex_int4_short_decode_tokens + 1,
        )
        return prompt_pages >= max(
            0, min_prompt_pages
        ) and self._estimate_reflex_remaining_decode_tokens(request) >= max(
            0, min_decode_tokens
        )

    def _reflex_int4_landing_fraction_for_request(
        self,
        request: Request,
        requested_fraction: float,
    ) -> float:
        if not self._reflex_int4_is_global_evidence_request(request):
            return requested_fraction
        global_evidence_fraction = getattr(
            self,
            "_reflex_int4_global_evidence_landing_max_int4_fraction",
            0.08,
        )
        return min(requested_fraction, global_evidence_fraction)

    def _reflex_int4_admission_landing_fraction(
        self,
        request: Request,
    ) -> float:
        base_fraction = max(
            self._reflex_int4_max_int4_fraction_per_request,
            self._reflex_int4_admission_landing_max_int4_fraction
            * self._reflex_int4_slo_demotion_pressure(request),
        )
        base_fraction = self._reflex_int4_landing_fraction_for_request(
            request,
            base_fraction,
        )
        return min(1.0, max(0.0, base_fraction))

    def _plan_reflex_int4_landing_frontier(
        self,
        *,
        request: Request,
        needed_blocks: int,
        reserve_blocks: int,
        free_blocks: int,
        running_feasible_release: int,
    ) -> PrecisionLandingDecision:
        planner = self._get_precision_landing_planner()
        eligible_int4_pages = self._reflex_int4_landing_eligible_pages(request)
        allow_reserve_relaxation = (
            bool(getattr(self, "_reflex_int4_direct_landing_enabled", False))
            and self._reflex_remote_chunk_page_range_for_landing(
                request,
                page_count=max(
                    0,
                    (
                        min(
                            max(0, int(getattr(request, "num_tokens", 0))),
                            self.max_model_len,
                        )
                        + self.block_size
                        - 1
                    )
                    // self.block_size,
                ),
            )
            is not None
        )
        decision = planner.plan_landing(
            PrecisionLandingState(
                request_id=getattr(request, "request_id", "<unknown>"),
                needed_blocks=needed_blocks,
                reserve_blocks=reserve_blocks,
                free_bf16_blocks=free_blocks,
                running_feasible_release=running_feasible_release,
                eligible_int4_landing_blocks=len(eligible_int4_pages),
                eligible_int4_landing_pages=eligible_int4_pages,
                allow_reserve_relaxation=allow_reserve_relaxation,
            )
        )
        if decision.reason != "int4_landing_frontier_insufficient":
            return decision

        total_bf16_blocks = max(
            0,
            int(getattr(self.kv_cache_manager.block_pool, "num_gpu_blocks", 0)),
        )
        hard_bf16_capacity_gap = (
            total_bf16_blocks > 0
            and decision.required_blocks > total_bf16_blocks
            and decision.residual_deficit_after_running > 0
        )

        pressure_fraction = self._reflex_int4_admission_landing_fraction(request)
        if pressure_fraction > self._reflex_int4_max_int4_fraction_per_request:
            pressure_eligible_pages = self._reflex_int4_landing_eligible_pages(
                request,
                max_int4_fraction=pressure_fraction,
            )
            if len(pressure_eligible_pages) > len(eligible_int4_pages):
                pressure_decision = planner.plan_landing(
                    PrecisionLandingState(
                        request_id=getattr(request, "request_id", "<unknown>"),
                        needed_blocks=needed_blocks,
                        reserve_blocks=reserve_blocks,
                        free_bf16_blocks=free_blocks,
                        running_feasible_release=running_feasible_release,
                        eligible_int4_landing_blocks=len(pressure_eligible_pages),
                        eligible_int4_landing_pages=pressure_eligible_pages,
                        allow_reserve_relaxation=allow_reserve_relaxation,
                    )
                )
                if pressure_decision.admission_feasible_with_landing:
                    return pressure_decision
        elif not hard_bf16_capacity_gap:
            return decision

        if hard_bf16_capacity_gap:
            emergency_fraction = max(
                self._reflex_int4_max_int4_fraction_per_request,
                self._reflex_int4_admission_landing_max_int4_fraction,
            )
            emergency_pages = self._reflex_int4_landing_eligible_pages(
                request,
                max_int4_fraction=emergency_fraction,
                respect_global_evidence_cap=False,
            )
            emergency_decision = planner.plan_landing(
                PrecisionLandingState(
                    request_id=getattr(request, "request_id", "<unknown>"),
                    needed_blocks=needed_blocks,
                    reserve_blocks=reserve_blocks,
                    free_bf16_blocks=free_blocks,
                    running_feasible_release=running_feasible_release,
                    eligible_int4_landing_blocks=len(emergency_pages),
                    eligible_int4_landing_pages=emergency_pages,
                    allow_reserve_relaxation=allow_reserve_relaxation,
                )
            )
            if emergency_decision.admission_feasible_with_landing:
                return replace(
                    emergency_decision,
                    reason="emergency_mixed_landing_feasible",
                )
        return decision

    @staticmethod
    def _has_reflex_int4_landing_contract(request: Request) -> bool:
        return has_reflex_int4_landing_contract(request)

    @staticmethod
    def _reflex_int4_landing_contract_page_indices(
        request: Request,
    ) -> set[int]:
        if not has_reflex_int4_landing_contract(request):
            return set()
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return set()
        raw_pages = params.get("reflex_int4_landing_pages")
        if not isinstance(raw_pages, (list, tuple)):
            return set()
        page_indices: set[int] = set()
        for raw_page in raw_pages:
            try:
                page_idx = int(raw_page)
            except (TypeError, ValueError):
                continue
            if page_idx >= 0:
                page_indices.add(page_idx)
        return page_indices

    def _can_reflex_int4_mixed_landing_close_admission_gap(self) -> bool:
        # Direct landing is executable capacity: the decoder reserves INT4
        # sidecar pages and avoids BF16 slots for those pages before transfer.
        if bool(getattr(self, "_reflex_int4_direct_landing_enabled", False)):
            return True
        # Staged landing materializes from decoder-side BF16 blocks. It can
        # reduce steady-state BF16 after transfer, but it cannot reduce the
        # BF16 capacity needed to admit the transfer itself.
        return bool(
            getattr(
                self,
                "_reflex_int4_mixed_landing_admission_enabled",
                False,
            )
        )

    def _is_reflex_int4_demotion_protected_request(
        self,
        request: Request,
    ) -> bool:
        request_level_protected = (
            getattr(request, "is_prefill_chunk", False)
            or getattr(request, "status", None) == RequestStatus.WAITING_FOR_REMOTE_KVS
            or self._has_reflex_int4_landing_contract(request)
        )
        if (
            request_level_protected
            and self._reflex_int4_has_page_level_demotion_frontier(request)
        ):
            return False
        return request_level_protected

    def _reflex_int4_has_page_level_demotion_frontier(
        self,
        request: Request,
    ) -> bool:
        if not bool(getattr(self, "_reflex_int4_page_level_protection_enabled", True)):
            return False
        if self._has_reflex_int4_landing_contract(request):
            if not self._reflex_int4_landing_contract_page_indices(request):
                return False
            if self._reflex_remote_chunk_sealed_pages(request) > 0:
                return True
            return getattr(
                request, "status", None
            ) != RequestStatus.WAITING_FOR_REMOTE_KVS and not getattr(
                request, "is_prefill_chunk", False
            )
        return self._reflex_remote_chunk_sealed_pages(request) > 0

    def _is_reflex_int4_preemption_protected_request(
        self,
        request: Request,
    ) -> bool:
        if self._has_reflex_int4_landing_contract(request):
            return True
        has_reflex_int4_blocks = getattr(
            self.kv_cache_manager,
            "has_reflex_int4_blocks",
            None,
        )
        return bool(
            has_reflex_int4_blocks is not None
            and has_reflex_int4_blocks(request.request_id)
        )

    def _select_preemption_victim(
        self,
        candidates: Iterable[Request],
        *,
        allow_reflex_int4_protected: bool = False,
    ) -> Request | None:
        ordered_candidates = list(candidates)
        if not ordered_candidates:
            return None
        if self.policy == SchedulingPolicy.PRIORITY:
            ordered_candidates.sort(
                key=lambda request: (request.priority, request.arrival_time),
                reverse=True,
            )
        else:
            ordered_candidates.reverse()

        if getattr(self.cache_config, "cache_dtype", None) != "reflex_int4":
            return ordered_candidates[0]

        for request in ordered_candidates:
            if not self._is_reflex_int4_preemption_protected_request(request):
                return request
        if allow_reflex_int4_protected:
            for request in ordered_candidates:
                if self._has_reflex_int4_landing_contract(request):
                    continue
                logger.info(
                    "ReFlexKV trace preemption_policy request=%s "
                    "outcome=select_reflex_int4_protected "
                    "reason=hard_precision_pressure.",
                    getattr(request, "request_id", "<unknown>"),
                )
                return request
        return None

    def _should_preempt_reflex_int4_protected_request(self) -> bool:
        if getattr(self.cache_config, "cache_dtype", None) != "reflex_int4":
            return False
        block_pool = getattr(self.kv_cache_manager, "block_pool", None)
        if block_pool is None or block_pool.get_num_free_blocks() > 0:
            return False
        breakdown = getattr(self, "_reflex_int4_last_candidate_breakdown", None)
        if breakdown is None:
            return False
        after_int4_pool_limit = int(getattr(breakdown, "after_int4_pool_limit", 0) or 0)
        selected_actual = int(getattr(breakdown, "selected_actual", 0) or 0)
        int4_free_blocks = int(getattr(breakdown, "int4_free_blocks", -1))
        return (
            after_int4_pool_limit == 0
            and selected_actual == 0
            and int4_free_blocks == 0
        )

    @staticmethod
    def _clear_reflex_int4_landing_contract(request: Request) -> None:
        clear_reflex_int4_landing_contract(request)

    def _commit_reflex_int4_landing_contract(self, request: Request) -> int:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return 0
        landing_pages = params.get("reflex_int4_landing_pages")
        landing_block_ids = params.get("reflex_int4_landing_block_ids")
        if not isinstance(landing_pages, (list, tuple)) or not isinstance(
            landing_block_ids, (list, tuple)
        ):
            return 0
        if len(landing_pages) == 0:
            self._clear_reflex_int4_landing_contract(request)
            return 0

        materialized = False
        pop_materialized = getattr(
            self.connector,
            "pop_reflex_int4_materialized_landing_req",
            None,
        )
        if callable(pop_materialized):
            materialized = bool(pop_materialized(request.request_id))
        else:
            materialized_req_ids = getattr(
                self,
                "reflex_int4_materialized_landing_req_ids",
                set(),
            )
            materialized = request.request_id in materialized_req_ids
            if materialized:
                materialized_req_ids.discard(request.request_id)

        if not materialized:
            release_landing_blocks = getattr(
                self.kv_cache_manager,
                "release_reflex_int4_landing_blocks",
                None,
            )
            if callable(release_landing_blocks):
                release_landing_blocks(request.request_id)
            logger.info(
                "ReFlexKV trace landing_policy request=%s "
                "outcome=fallback_unmaterialized planned_pages=%d "
                "materialized=False reason=no_materialized_signal.",
                request.request_id,
                len(landing_pages),
            )
            self._clear_reflex_int4_landing_contract(request)
            logger.warning(
                "Skipping ReFlexKV landing commit for %s because worker did "
                "not report INT4 sidecar materialization.",
                request.request_id,
            )
            return 0

        commit_landing_pages = getattr(
            self.kv_cache_manager,
            "commit_reflex_int4_landing_pages",
            None,
        )
        if not callable(commit_landing_pages):
            self._clear_reflex_int4_landing_contract(request)
            return 0

        page_indices = [int(page_idx) for page_idx in landing_pages]
        int4_block_ids = [int(block_id) for block_id in landing_block_ids]
        committed = commit_landing_pages(
            request.request_id,
            page_indices,
            int4_block_ids,
        )
        if committed:
            logger.info(
                "ReFlexKV trace landing_commit request=%s pages=%d committed=%d.",
                request.request_id,
                len(page_indices),
                committed,
            )
        check_invariants = getattr(
            self.kv_cache_manager,
            "check_reflex_int4_invariants",
            None,
        )
        if callable(check_invariants):
            violations = check_invariants(request.request_id)
            if violations:
                logger.error(
                    "ReFlexKV landing commit invariant violation for %s: %s",
                    request.request_id,
                    violations[0],
                )
                raise RuntimeError(violations[0])
        self._clear_reflex_int4_landing_contract(request)
        return committed

    def _should_persist_reflex_int4_landing_contract(
        self,
        request: Request,
        landing_decision: PrecisionLandingDecision | None,
    ) -> bool:
        if landing_decision is None:
            return False
        if (
            not landing_decision.mixed_landing_required
            or not landing_decision.admission_feasible_with_landing
            or landing_decision.planned_int4_landing_blocks <= 0
        ):
            return False
        if getattr(request, "status", None) == RequestStatus.WAITING_FOR_REMOTE_KVS:
            return True
        if not self._can_reflex_int4_mixed_landing_close_admission_gap():
            return False
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return False
        return bool(
            params.get("do_remote_prefill")
            or self._has_reflex_int4_landing_contract(request)
        )

    def _persist_reflex_int4_landing_contract(
        self,
        request: Request,
        landing_decision: PrecisionLandingDecision,
    ) -> None:
        params = getattr(request, "kv_transfer_params", None)
        if not isinstance(params, dict):
            return
        existing_pages = params.get("reflex_int4_landing_pages")
        existing_block_ids = params.get("reflex_int4_landing_block_ids")
        has_existing_landing_contract = (
            isinstance(existing_pages, (list, tuple))
            and isinstance(existing_block_ids, (list, tuple))
            and len(existing_pages) > 0
            and len(existing_pages) == len(existing_block_ids)
        )
        if (
            has_existing_landing_contract
            and getattr(request, "status", None) == RequestStatus.WAITING_FOR_REMOTE_KVS
        ):
            return
        if (
            not landing_decision.mixed_landing_required
            or not landing_decision.admission_feasible_with_landing
            or landing_decision.planned_int4_landing_blocks <= 0
        ):
            self._clear_reflex_int4_landing_contract(request)
            return
        landing_pages = list(landing_decision.planned_int4_landing_pages)
        existing_block_ids = params.get("reflex_int4_landing_block_ids")
        if isinstance(existing_block_ids, (list, tuple)) and len(
            existing_block_ids
        ) == len(landing_pages):
            landing_block_ids = [int(block_id) for block_id in existing_block_ids]
        else:
            release_landing_blocks = getattr(
                self.kv_cache_manager,
                "release_reflex_int4_landing_blocks",
                None,
            )
            if callable(release_landing_blocks):
                release_landing_blocks(getattr(request, "request_id", ""))
            reserve_landing_blocks = getattr(
                self.kv_cache_manager,
                "reserve_reflex_int4_landing_blocks",
                None,
            )
            if not callable(reserve_landing_blocks):
                self._clear_reflex_int4_landing_contract(request)
                return
            landing_block_ids = reserve_landing_blocks(
                getattr(request, "request_id", ""),
                len(landing_pages),
            )
            if len(landing_block_ids) != len(landing_pages):
                self._clear_reflex_int4_landing_contract(request)
                return
        record_landing_pages = getattr(
            self.kv_cache_manager,
            "record_reflex_int4_landing_pages",
            None,
        )
        if callable(record_landing_pages):
            record_landing_pages(
                getattr(request, "request_id", ""),
                landing_pages,
            )
        direct_landing = bool(
            getattr(self, "_reflex_int4_direct_landing_enabled", False)
        )
        if direct_landing:
            mark_direct_landing = getattr(
                self.kv_cache_manager,
                "mark_reflex_int4_direct_landing_pages",
                None,
            )
            if callable(mark_direct_landing):
                mark_direct_landing(
                    getattr(request, "request_id", ""),
                    landing_pages,
                    list(landing_block_ids),
                )
        request_id = getattr(request, "request_id", "")
        existing_pages_list = (
            [int(page) for page in existing_pages]
            if isinstance(existing_pages, (list, tuple))
            else []
        )
        existing_block_ids_list = (
            [int(block_id) for block_id in existing_block_ids]
            if isinstance(existing_block_ids, (list, tuple))
            else []
        )
        contract_changed = (
            existing_pages_list != landing_pages
            or existing_block_ids_list != list(landing_block_ids)
            or bool(params.get("reflex_int4_direct_landing")) != direct_landing
        )
        params["reflex_int4_landing_pages"] = landing_pages
        params["reflex_int4_landing_block_ids"] = list(landing_block_ids)
        params["reflex_int4_direct_landing"] = direct_landing
        params["reflex_int4_landing_required_blocks"] = (
            landing_decision.residual_deficit_after_running
        )
        params["reflex_int4_landing_planned_blocks"] = (
            landing_decision.planned_int4_landing_blocks
        )
        params["reflex_int4_landing_reason"] = landing_decision.reason
        if contract_changed:
            logger.info(
                "ReFlexKV trace landing_contract request=%s pages=%d "
                "direct=%s required_blocks=%d planned_blocks=%d reason=%s.",
                request_id,
                len(landing_pages),
                direct_landing,
                landing_decision.residual_deficit_after_running,
                landing_decision.planned_int4_landing_blocks,
                landing_decision.reason,
            )

    def _plan_and_persist_reflex_int4_landing_contract(
        self,
        *,
        request: Request,
        reason: str,
        num_lookahead_tokens: int = 0,
        reserve_blocks: int | None = None,
    ) -> PrecisionLandingDecision | None:
        if self.cache_config.cache_dtype != "reflex_int4":
            return None
        if reserve_blocks is None:
            reserve_blocks = self._reflex_int4_admission_reserve_blocks
        needed_blocks = self._estimate_reflex_admission_needed_blocks(
            request,
            num_lookahead_tokens=num_lookahead_tokens,
        )
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        requested_release = max(0, needed_blocks + reserve_blocks - free_blocks)
        feasible_release = 0
        if requested_release > 0:
            feasible_release = self._estimate_reflex_int4_feasible_release(
                target_bf16_blocks=requested_release,
                reason=reason,
            )
        landing_decision = self._plan_reflex_int4_landing_frontier(
            request=request,
            needed_blocks=needed_blocks,
            reserve_blocks=reserve_blocks,
            free_blocks=free_blocks,
            running_feasible_release=feasible_release,
        )
        params = getattr(request, "kv_transfer_params", None)
        if landing_decision.mixed_landing_required and (
            not isinstance(params, dict)
            or (
                not bool(params.get("do_remote_prefill"))
                and not self._has_reflex_int4_landing_contract(request)
            )
        ):
            self._clear_reflex_int4_landing_contract(request)
            return landing_decision
        self._persist_reflex_int4_landing_contract(request, landing_decision)
        return landing_decision

    def _get_reflex_int4_frontier_cache(self) -> FeasibleFrontierCache:
        cache = getattr(self, "_reflex_int4_frontier_cache", None)
        if cache is None:
            cache = FeasibleFrontierCache(
                max_age_steps=getattr(
                    self,
                    "_reflex_int4_frontier_cache_max_age_steps",
                    2,
                )
            )
            self._reflex_int4_frontier_cache = cache
        return cache

    def _record_reflex_int4_frontier_event(self, event: str) -> None:
        event_steps = getattr(self, "_reflex_int4_frontier_event_steps", None)
        if event_steps is None:
            event_steps = {}
            self._reflex_int4_frontier_event_steps = event_steps
        event_steps[event] = getattr(self, "_reflex_int4_scheduler_step", 0)

    def _get_reflex_int4_frontier_events(self) -> frozenset[str]:
        current_step = getattr(self, "_reflex_int4_scheduler_step", 0)
        event_steps = getattr(self, "_reflex_int4_frontier_event_steps", None)
        if not event_steps:
            return frozenset()
        fresh_events = set()
        for event, step in event_steps.items():
            age = current_step - step
            if age > 1:
                continue
            if event == "bf16_freed" and age == 0:
                continue
            fresh_events.add(event)
        stale_events = [
            event for event, step in event_steps.items() if current_step - step > 1
        ]
        for event in stale_events:
            event_steps.pop(event, None)
        return frozenset(fresh_events)

    def _get_reflex_int4_admission_tickets(self) -> dict[str, AdmissionTicket]:
        tickets = getattr(self, "_reflex_int4_admission_tickets", None)
        if tickets is None:
            tickets = {}
            self._reflex_int4_admission_tickets = tickets
        return tickets

    def _record_reflex_int4_admission_ticket(
        self,
        *,
        request: Request,
        required_blocks: int,
        blocked_reason: str,
        cached_frontier_summary: FeasibleFrontierSummary | None = None,
    ) -> AdmissionTicket:
        current_step = getattr(self, "_reflex_int4_scheduler_step", 0)
        retry_delay_steps = max(
            1,
            getattr(
                self,
                "_reflex_int4_admission_ticket_retry_delay_steps",
                8,
            ),
        )
        max_retry_delay_steps = max(
            retry_delay_steps,
            getattr(
                self,
                "_reflex_int4_admission_ticket_max_retry_delay_steps",
                64,
            ),
        )
        request_id = getattr(request, "request_id", "<unknown>")
        existing_ticket = self._get_reflex_int4_admission_tickets().get(request_id)
        retry_count = (
            existing_ticket.retry_count + 1 if existing_ticket is not None else 0
        )
        backoff_multiplier = 1 << min(retry_count, 10)
        retry_delay_steps = min(
            max_retry_delay_steps,
            retry_delay_steps * backoff_multiplier,
        )
        retry_on_events = {"request_finished"}
        if blocked_reason != "full_sequence_reserve":
            retry_on_events.add("bf16_freed")
        ticket = AdmissionTicket(
            request_id=request_id,
            required_blocks=max(0, int(required_blocks)),
            blocked_reason=blocked_reason,
            created_step=(
                existing_ticket.created_step
                if existing_ticket is not None
                else current_step
            ),
            last_retry_step=current_step,
            next_retry_step=current_step + retry_delay_steps,
            retry_count=retry_count,
            retry_on_events=frozenset(retry_on_events),
            cached_frontier_summary=(
                cached_frontier_summary
                if cached_frontier_summary is not None
                else self._get_reflex_int4_frontier_cache().latest()
            ),
        )
        self._get_reflex_int4_admission_tickets()[request_id] = ticket
        return ticket

    def _should_skip_reflex_int4_waiting_request_by_ticket(
        self,
        request: Request,
    ) -> bool:
        if self.cache_config.cache_dtype != "reflex_int4":
            return False
        request_id = getattr(request, "request_id", None)
        if request_id is None:
            return False
        ticket = self._get_reflex_int4_admission_tickets().get(request_id)
        if ticket is None:
            return False
        current_step = getattr(self, "_reflex_int4_scheduler_step", 0)
        frontier_events = self._get_reflex_int4_frontier_events()
        if ticket.should_retry(
            current_step=current_step,
            events=frontier_events,
        ):
            return False
        if current_step == ticket.last_retry_step + 1 or current_step % 64 == 0:
            logger.info(
                "ReFlexKV trace admission_control request=%s "
                "outcome=skip_admission_ticket blocked_reason=%s "
                "required_blocks=%d next_retry_step=%d current_step=%d "
                "frontier_age=%d frontier_levels=%s "
                "frontier_rejection_reasons=%s.",
                request_id,
                ticket.blocked_reason,
                ticket.required_blocks,
                ticket.next_retry_step,
                current_step,
                self._reflex_frontier_age_or_minus_one(ticket.cached_frontier_summary),
                self._format_reflex_frontier_levels(ticket.cached_frontier_summary),
                self._format_reflex_frontier_rejection_reasons(
                    ticket.cached_frontier_summary
                ),
            )
        return True

    def _clear_reflex_int4_admission_ticket(self, request_id: str) -> None:
        self._get_reflex_int4_admission_tickets().pop(request_id, None)

    def _reflex_frontier_age_or_minus_one(
        self,
        summary: FeasibleFrontierSummary | None,
    ) -> int:
        if summary is None:
            return -1
        return summary.cached_frontier_age(
            current_step=getattr(self, "_reflex_int4_scheduler_step", 0)
        )

    @staticmethod
    def _format_reflex_frontier_levels(
        summary: FeasibleFrontierSummary | None,
    ) -> str:
        if summary is None:
            return "none"
        return summary.format_levels()

    @staticmethod
    def _format_reflex_frontier_rejection_reasons(
        summary: FeasibleFrontierSummary | None,
    ) -> str:
        if summary is None:
            return "none"
        return summary.format_rejection_reasons()

    @staticmethod
    def _dominant_reflex_frontier_rejection_reason(
        summary: FeasibleFrontierSummary | None,
    ) -> str:
        if summary is None:
            return "frontier_unknown"
        best_reason = "frontier_infeasible"
        best_count = 0
        for reason, count in summary.blocked_by_reason.items():
            count = int(count)
            if count > best_count:
                best_reason = getattr(reason, "value", str(reason))
                best_count = count
        return best_reason if best_count > 0 else "frontier_infeasible"

    def _classify_reflex_admission_blocked_reason(
        self,
        *,
        admission_success_after_demote: bool,
        admission_infeasible: bool,
        planned_release: int,
        actual_release: int,
        candidate_release_capacity: int,
        requested_release: int,
        landing_decision: PrecisionLandingDecision | None,
        frontier_summary: FeasibleFrontierSummary | None,
    ) -> str:
        if admission_success_after_demote:
            return "none"
        if (
            landing_decision is not None
            and landing_decision.reason == "mixed_landing_requires_bf16_staging"
        ):
            return "mixed_landing_requires_bf16_staging"
        if admission_infeasible:
            return self._dominant_reflex_frontier_rejection_reason(frontier_summary)
        if planned_release > actual_release:
            return "partial_release"
        if candidate_release_capacity < requested_release:
            return self._dominant_reflex_frontier_rejection_reason(frontier_summary)
        return "admission_waiting"

    def _update_reflex_int4_dual_price_state(self) -> DualPriceState:
        state = getattr(
            self,
            "_reflex_int4_dual_price_state",
            DualPriceState(),
        )
        block_pool = self.kv_cache_manager.block_pool
        total_blocks = max(1, int(getattr(block_pool, "num_gpu_blocks", 1)))
        free_blocks = int(block_pool.get_num_free_blocks())
        kv_usage = 1.0 - min(1.0, max(0.0, free_blocks / total_blocks))
        waiting_requests = self._reflex_int4_queue_len(
            getattr(self, "waiting", None)
        ) + self._reflex_int4_queue_len(getattr(self, "skipped_waiting", None))
        updated = state.updated(
            kv_usage=kv_usage,
            kv_target=getattr(self, "_reflex_int4_dual_kv_target", 0.85),
            waiting_requests=waiting_requests,
            waiting_target=getattr(self, "_reflex_int4_dual_waiting_target", 0),
            migration_backlog=0,
            migration_target=getattr(self, "_reflex_int4_dual_migration_target", 1),
            eta=getattr(self, "_reflex_int4_dual_eta", 0.05),
        )
        self._reflex_int4_dual_price_state = updated
        return updated

    @staticmethod
    def _reflex_int4_queue_len(queue) -> int:
        if queue is None:
            return 0
        try:
            return len(queue)
        except TypeError:
            return int(bool(queue))

    def _build_reflex_int4_demotion_planning_kwargs(
        self,
        *,
        target_bf16_blocks: int,
        reason: str,
    ) -> dict[str, Any]:
        pressure_decision = self._plan_reflex_int4_pressure_policy(
            reason=reason,
            target_bf16_blocks=target_bf16_blocks,
        )
        (
            prefill_page_risks_by_request,
            compressible_pages_by_request,
        ) = self._build_reflex_prefill_page_metadata_inputs(
            low_risk_score_fraction=pressure_decision.low_risk_score_fraction,
        )
        shadow_pages_per_request = getattr(
            self,
            "_reflex_int4_recovery_shadow_pages_per_request",
            0,
        )
        recovery_shadow_pages_by_request = {
            request_id: select_bf16_shadow_pages(
                page_risks,
                max_pages=shadow_pages_per_request,
            )
            for request_id, page_risks in prefill_page_risks_by_request.items()
            if shadow_pages_per_request > 0
        }
        requests = getattr(self, "requests", {})
        metadata_sources = [
            self._reflex_int4_landing_metadata_trace_fields(request)
            for request in requests.values()
        ]
        synthetic_pages = sum(fields[3] for fields in metadata_sources)
        synthetic_requests = sum(1 for fields in metadata_sources if fields[3] > 0)
        real_risk_requests = len(prefill_page_risks_by_request)
        real_risk_pages = sum(
            len(page_risks) for page_risks in prefill_page_risks_by_request.values()
        )
        compressible_requests = len(compressible_pages_by_request)
        compressible_pages = sum(
            len(pages) for pages in compressible_pages_by_request.values()
        )
        shadow_requests = len(recovery_shadow_pages_by_request)
        shadow_pages = sum(
            len(pages) for pages in recovery_shadow_pages_by_request.values()
        )
        if real_risk_pages or compressible_pages or shadow_pages or synthetic_pages:
            logger.info(
                "ReFlexKV trace page_metadata_plan reason=%s "
                "real_risk_requests=%d real_risk_pages=%d "
                "compressible_requests=%d compressible_pages=%d "
                "shadow_requests=%d shadow_pages=%d "
                "synthetic_requests=%d synthetic_pages=%d.",
                reason,
                real_risk_requests,
                real_risk_pages,
                compressible_requests,
                compressible_pages,
                shadow_requests,
                shadow_pages,
                synthetic_requests,
                synthetic_pages,
            )
        selection_policy = getattr(
            self,
            "_reflex_int4_page_selection_policy",
            "relevance_sparse",
        )
        low_risk_only = selection_policy in {"relevance", "relevance_sparse"}
        sparse_window_pages = (
            self._reflex_int4_sparse_window_pages
            if selection_policy in {"relevance_sparse", "frontier_dual"}
            else 0
        )
        max_demote_per_window = (
            self._reflex_int4_max_demote_per_window
            if selection_policy in {"relevance_sparse", "frontier_dual"}
            else 0
        )
        if reason in {
            "admission_waiting",
            "allocation_failure",
            "full_sequence_reserve",
        } and selection_policy in {"relevance_sparse", "frontier_dual"}:
            sparse_window_pages = self._reflex_int4_admission_sparse_window_pages
            max_demote_per_window = max(
                max_demote_per_window,
                self._reflex_int4_admission_max_demote_per_window,
            )
        if selection_policy in {"relevance_sparse", "frontier_dual"}:
            max_demote_per_window = max(
                max_demote_per_window,
                int(
                    max_demote_per_window
                    * pressure_decision.max_demote_per_window_multiplier
                ),
            )
        protected_prompt_pages_by_request: dict[str, int] = {}
        protected_pages_by_request: dict[str, set[int]] = {}
        sealed_pages_by_request: dict[str, int] = {}
        remote_inflight_pages_by_request: dict[str, set[int]] = {}
        copy_on_demote_pages_by_request: dict[str, set[int]] = {}
        allow_partial_prefill_demotion_request_ids: set[str] = set()
        for request_id, request in requests.items():
            protected_page_indices = self._reflex_int4_protected_prompt_page_indices(
                request
            )
            landing_contract_pages = self._reflex_int4_landing_contract_page_indices(
                request
            )
            if landing_contract_pages:
                protected_page_indices = protected_page_indices | landing_contract_pages
            if protected_page_indices:
                protected_pages_by_request[request_id] = protected_page_indices
            protected_prefix_pages = self._reflex_contiguous_prefix_len(
                protected_page_indices
            )
            if protected_prefix_pages > 0:
                protected_prompt_pages_by_request[request_id] = protected_prefix_pages
            sealed_pages = self._reflex_remote_chunk_sealed_pages(request)
            if sealed_pages > 0:
                sealed_pages_by_request[request_id] = sealed_pages
            remote_inflight_pages = self._reflex_remote_chunk_inflight_pages(request)
            if remote_inflight_pages:
                remote_inflight_pages_by_request[request_id] = remote_inflight_pages
            copy_on_demote_pages = self._reflex_copy_on_demote_pages(request)
            if copy_on_demote_pages:
                copy_on_demote_pages_by_request[request_id] = copy_on_demote_pages
            prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
            computed_tokens = int(getattr(request, "num_computed_tokens", 0) or 0)
            if (
                self._is_reflex_remote_chunk_load(request)
                and 0 < computed_tokens < prompt_tokens
            ) or sealed_pages > 0:
                allow_partial_prefill_demotion_request_ids.add(request_id)
        dual_price_state = None
        if selection_policy == "frontier_dual":
            dual_price_state = self._update_reflex_int4_dual_price_state()
        admission_waiting_emergency = (
            reason == "admission_waiting"
            and target_bf16_blocks
            > getattr(self, "_reflex_int4_admission_reserve_blocks", 0)
        )
        emergency_release = selection_policy == "frontier_dual" and (
            reason in {
                "allocation_failure",
                "decode_cache_full",
                "full_sequence_reserve",
            }
            or admission_waiting_emergency
        )
        return {
            "target_bf16_blocks": target_bf16_blocks,
            "keep_recent_pages": self._reflex_int4_keep_recent_pages,
            "keep_initial_pages": self._reflex_int4_keep_initial_pages,
            "max_int4_fraction_per_request": (
                self._reflex_int4_max_int4_fraction_per_request
            ),
            "low_risk_only": low_risk_only,
            "sparse_window_pages": sparse_window_pages,
            "max_demote_per_window": max_demote_per_window,
            "selection_policy": selection_policy,
            "dual_price_state": dual_price_state,
            "emergency_release": emergency_release,
            "cache_scope": reason,
            "request_precision_budgets": (
                self._build_reflex_int4_request_precision_budgets(
                    reason=reason,
                    target_bf16_blocks=target_bf16_blocks,
                    pressure_decision=pressure_decision,
                )
            ),
            "computed_tokens_by_request": {
                request_id: request.num_computed_tokens
                for request_id, request in requests.items()
            },
            "prompt_tokens_by_request": {
                request_id: request.num_prompt_tokens
                for request_id, request in requests.items()
            },
            "protected_request_ids": {
                request_id
                for request_id, request in requests.items()
                if self._is_reflex_int4_demotion_protected_request(request)
            },
            "allow_partial_prefill_demotion_request_ids": (
                allow_partial_prefill_demotion_request_ids
            ),
            "protected_prompt_pages_by_request": (protected_prompt_pages_by_request),
            "protected_pages_by_request": protected_pages_by_request,
            "sealed_pages_by_request": sealed_pages_by_request,
            "remote_inflight_pages_by_request": remote_inflight_pages_by_request,
            "prefill_page_risks_by_request": prefill_page_risks_by_request,
            "compressible_pages_by_request": compressible_pages_by_request,
            "copy_on_demote_pages_by_request": copy_on_demote_pages_by_request,
            "recovery_shadow_pages_by_request": recovery_shadow_pages_by_request,
            "recovery_shadow_pages_per_request": (shadow_pages_per_request),
        }

    def _estimate_reflex_int4_feasible_release(
        self,
        *,
        target_bf16_blocks: int,
        reason: str,
    ) -> int:
        if self.cache_config.cache_dtype != "reflex_int4":
            return 0
        if target_bf16_blocks <= 0:
            self._reflex_int4_last_demote_candidate_capacity = 0
            return 0
        if not hasattr(self.kv_cache_manager, "plan_reflex_int4_demotions"):
            self._reflex_int4_last_demote_candidate_capacity = target_bf16_blocks
            return target_bf16_blocks

        cached_summary = self._get_reflex_int4_frontier_cache().get(
            reason=reason,
            target_release=target_bf16_blocks,
            current_step=getattr(self, "_reflex_int4_scheduler_step", 0),
        )
        if cached_summary is not None:
            cached_feasible_release = min(
                target_bf16_blocks,
                cached_summary.candidate_release_capacity,
            )
            self._reflex_int4_last_demote_candidate_capacity = (
                cached_summary.candidate_release_capacity
            )
            self._reflex_int4_last_candidate_breakdown = (
                cached_summary.candidate_breakdown
            )
            logger.info(
                "ReFlexKV trace feasible_frontier_cache outcome=hit "
                "reason=%s target_release=%d feasible_release=%d "
                "candidate_release_capacity=%d cached_frontier_age=%d "
                "levels=%s blocked=%s.",
                reason,
                target_bf16_blocks,
                cached_feasible_release,
                cached_summary.candidate_release_capacity,
                cached_summary.cached_frontier_age(
                    current_step=getattr(
                        self,
                        "_reflex_int4_scheduler_step",
                        0,
                    )
                ),
                cached_summary.format_levels(),
                cached_summary.format_rejection_reasons(),
            )
            return cached_feasible_release

        plan_kwargs = self._build_reflex_int4_demotion_planning_kwargs(
            target_bf16_blocks=target_bf16_blocks,
            reason=reason,
        )
        feasible_release = self.kv_cache_manager.plan_reflex_int4_demotions(
            **plan_kwargs,
            dry_run=True,
        )
        if hasattr(
            self.kv_cache_manager,
            "get_last_reflex_int4_candidate_capacity",
        ):
            self._reflex_int4_last_demote_candidate_capacity = (
                self.kv_cache_manager.get_last_reflex_int4_candidate_capacity()
            )
        else:
            self._reflex_int4_last_demote_candidate_capacity = feasible_release
        if hasattr(
            self.kv_cache_manager,
            "get_last_reflex_int4_candidate_breakdown",
        ):
            breakdown = self.kv_cache_manager.get_last_reflex_int4_candidate_breakdown()
            self._reflex_int4_last_candidate_breakdown = breakdown
            summary = FeasibleFrontierSummary.from_candidate_breakdown(
                scheduler_step=getattr(self, "_reflex_int4_scheduler_step", 0),
                reason=reason,
                target_release=target_bf16_blocks,
                feasible_release=feasible_release,
                candidate_breakdown=breakdown,
            )
            self._get_reflex_int4_frontier_cache().update(summary)
            logger.info(
                "ReFlexKV trace feasible_frontier_cache outcome=miss "
                "reason=%s target_release=%d feasible_release=%d "
                "candidate_release_capacity=%d cached_frontier_age=0 "
                "levels=%s blocked=%s.",
                reason,
                target_bf16_blocks,
                feasible_release,
                summary.candidate_release_capacity,
                summary.format_levels(),
                summary.format_rejection_reasons(),
            )
        return feasible_release

    def _plan_reflex_int4_admission_release(
        self,
        *,
        request: Request,
        needed_blocks: int,
        reserve_blocks: int,
        free_blocks: int,
        requested_release: int,
        reason: str,
    ):
        feasible_release = self._estimate_reflex_int4_feasible_release(
            target_bf16_blocks=requested_release,
            reason=reason,
        )
        return self._get_precision_admission_controller().plan_admission(
            PrecisionAdmissionState(
                request_id=getattr(request, "request_id", "<unknown>"),
                needed_blocks=needed_blocks,
                reserve_blocks=reserve_blocks,
                free_bf16_blocks=free_blocks,
                requested_release=requested_release,
                feasible_release=feasible_release,
            )
        )

    def _emit_reflex_int4_control_only_output(self) -> SchedulerOutput:
        scheduler_output = SchedulerOutput.make_empty()
        scheduler_output.finished_req_ids = self.finished_req_ids
        scheduler_output.free_encoder_mm_hashes = (
            self.encoder_cache_manager.get_freed_mm_hashes()
        )
        take_demotions = getattr(
            self.kv_cache_manager,
            "take_reflex_int4_demotions",
            None,
        )
        scheduler_output.reflex_int4_demotions = (
            take_demotions() or None if callable(take_demotions) else None
        )
        take_recoveries = getattr(
            self.kv_cache_manager,
            "take_reflex_int4_recoveries",
            None,
        )
        scheduler_output.reflex_int4_recoveries = (
            take_recoveries() or None if callable(take_recoveries) else None
        )
        if self.connector is not None:
            meta = self._build_kv_connector_meta(self.connector, scheduler_output)
            scheduler_output.kv_connector_metadata = meta
        self._reflex_int4_prev_step_had_prefill = False
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _try_reflex_int4_demotion_only_step(self) -> SchedulerOutput | None:
        if self.cache_config.cache_dtype != "reflex_int4":
            return None
        if not self.running:
            return None
        if (
            self._reflex_int4_scheduler_step - self._reflex_int4_last_demote_step
            < self._reflex_int4_demote_cooldown_steps
        ):
            return None
        if self._reflex_int4_prev_step_had_prefill:
            return None
        if any(getattr(request, "is_prefill_chunk", False) for request in self.running):
            return None

        if not (self.waiting or self.skipped_waiting):
            target_blocks = self._estimate_reflex_demote_target(
                1,
                force_allocate_failure=False,
            )
            force_decode_release = (
                self.kv_cache_manager.block_pool.get_num_free_blocks() <= 0
            )
            if force_decode_release and target_blocks > 0:
                min_background_batch = max(
                    1,
                    getattr(
                        self,
                        "_reflex_int4_background_min_demotions_per_step",
                        1,
                    ),
                )
                target_blocks = min(
                    max(target_blocks, min_background_batch),
                    self._reflex_int4_background_demotions_per_step,
                )
            actual_release_blocks = self._try_reflex_int4_demote(
                target_bf16_blocks=target_blocks,
                force=force_decode_release,
                reason=(
                    "decode_cache_full"
                    if force_decode_release
                    else "background_pressure"
                ),
            )
            if actual_release_blocks <= 0:
                return None
            return self._emit_reflex_int4_control_only_output()

        request_queue = self._select_waiting_queue_for_scheduling()
        if request_queue is None:
            return None
        try:
            request = request_queue.peek_request()
        except IndexError:
            return None
        if self._should_skip_reflex_int4_waiting_request_by_ticket(request):
            return None

        needed_blocks = self._estimate_reflex_admission_needed_blocks(request)
        free_blocks = self.kv_cache_manager.block_pool.get_num_free_blocks()
        reserve_blocks = self._reflex_int4_admission_reserve_blocks
        admission_deficit_blocks = max(
            0,
            needed_blocks + reserve_blocks - free_blocks,
        )
        target_blocks = admission_deficit_blocks
        if target_blocks > 0:
            target_blocks = max(target_blocks, reserve_blocks)
            target_blocks = min(
                target_blocks,
                self._reflex_int4_fast_demotions_per_step,
            )
            pressure_decision = self._plan_reflex_int4_pressure_policy(
                reason="admission_waiting",
                target_bf16_blocks=target_blocks,
            )
            target_blocks = min(
                pressure_decision.target_release_blocks,
                self._reflex_int4_fast_demotions_per_step,
            )
        if target_blocks > 0:
            logger.info(
                "ReFlexKV admission controller request=%s needed_blocks=%d "
                "reserve_blocks=%d bf16_free=%d/%d admission_deficit=%d "
                "target_release=%d.",
                getattr(request, "request_id", "<unknown>"),
                needed_blocks,
                reserve_blocks,
                free_blocks,
                self.kv_cache_manager.block_pool.num_gpu_blocks,
                admission_deficit_blocks,
                target_blocks,
            )
        else:
            self._reflex_int4_last_demote_candidate_capacity = 0
            return None
        self._reflex_int4_last_demote_candidate_capacity = 0
        decision = self._plan_reflex_int4_admission_release(
            request=request,
            needed_blocks=needed_blocks,
            reserve_blocks=reserve_blocks,
            free_blocks=free_blocks,
            requested_release=target_blocks,
            reason="admission_waiting",
        )
        landing_decision = self._plan_reflex_int4_landing_frontier(
            request=request,
            needed_blocks=needed_blocks,
            reserve_blocks=reserve_blocks,
            free_blocks=free_blocks,
            running_feasible_release=decision.feasible_release,
        )
        if getattr(request, "status", None) != RequestStatus.WAITING_FOR_REMOTE_KVS:
            if self._should_persist_reflex_int4_landing_contract(
                request,
                landing_decision,
            ):
                self._persist_reflex_int4_landing_contract(
                    request,
                    landing_decision,
                )
            else:
                self._clear_reflex_int4_landing_contract(request)
        landing_decision_for_admission = (
            landing_decision
            if self._can_reflex_int4_mixed_landing_close_admission_gap()
            else None
        )
        precision_plan = (
            self._get_precision_admission_controller().plan_precision_admission(
                admission_decision=decision,
                landing_decision=landing_decision_for_admission,
            )
        )
        landing_decision_for_log = landing_decision
        if (
            landing_decision is not None
            and landing_decision.mixed_landing_required
            and not self._can_reflex_int4_mixed_landing_close_admission_gap()
        ):
            landing_decision_for_log = replace(
                landing_decision,
                planned_int4_landing_blocks=0,
                admission_feasible_with_landing=False,
                reason="mixed_landing_requires_bf16_staging",
                planned_int4_landing_pages=(),
            )
        planned_release = precision_plan.planned_release
        admission_infeasible = precision_plan.admission_infeasible
        frontier_summary_for_log = self._get_reflex_int4_frontier_cache().latest()
        actual_release_blocks = 0
        if planned_release > 0:
            actual_release_blocks = self._try_reflex_int4_demote(
                target_bf16_blocks=planned_release,
                force=True,
                reason="admission_waiting",
            )
        candidate_release_capacity = max(
            self._reflex_int4_last_demote_candidate_capacity,
            decision.feasible_release,
            actual_release_blocks,
        )
        self._log_reflex_admission_control(
            request=request,
            requested_release=target_blocks,
            candidate_release_capacity=candidate_release_capacity,
            feasible_release=decision.feasible_release,
            planned_release=planned_release,
            actual_release=actual_release_blocks,
            needed_blocks=needed_blocks,
            reserve_blocks=reserve_blocks,
            free_blocks_before=free_blocks,
            admission_deficit_blocks=admission_deficit_blocks,
            admission_infeasible=admission_infeasible,
            landing_decision=landing_decision_for_log,
            frontier_summary=frontier_summary_for_log,
        )
        if actual_release_blocks == 0:
            return None

        return self._emit_reflex_int4_control_only_output()

    def _log_reflex_admission_control(
        self,
        *,
        request: Request,
        requested_release: int,
        candidate_release_capacity: int,
        feasible_release: int,
        planned_release: int,
        actual_release: int,
        needed_blocks: int,
        reserve_blocks: int,
        free_blocks_before: int,
        admission_deficit_blocks: int,
        admission_infeasible: bool,
        landing_decision: PrecisionLandingDecision | None = None,
        frontier_summary: FeasibleFrontierSummary | None = None,
    ) -> None:
        free_after_estimated = free_blocks_before + max(0, actual_release)
        required_blocks = needed_blocks + reserve_blocks
        admission_success_after_demote = free_after_estimated >= required_blocks
        admission_wait_reduction = min(
            max(0, admission_deficit_blocks),
            max(0, actual_release),
        )
        landing_mixed_feasible = False
        landing_required_int4_blocks = 0
        landing_eligible_int4_blocks = 0
        landing_planned_int4_blocks = 0
        landing_residual_bf16_deficit = 0
        landing_reason = "none"
        (
            landing_metadata_source,
            landing_real_risk_pages,
            landing_explicit_compressible_pages,
            landing_synthetic_pages,
        ) = self._reflex_int4_landing_metadata_trace_fields(request)
        if landing_decision is not None:
            landing_mixed_feasible = (
                landing_decision.mixed_landing_required
                and landing_decision.admission_feasible_with_landing
            )
            landing_required_int4_blocks = (
                landing_decision.residual_deficit_after_running
            )
            landing_eligible_int4_blocks = landing_decision.eligible_int4_landing_blocks
            landing_planned_int4_blocks = landing_decision.planned_int4_landing_blocks
            landing_residual_bf16_deficit = (
                landing_decision.residual_deficit_after_running
            )
            landing_reason = landing_decision.reason
        if frontier_summary is None:
            frontier_summary = self._get_reflex_int4_frontier_cache().latest()
        blocked_reason = self._classify_reflex_admission_blocked_reason(
            admission_success_after_demote=admission_success_after_demote,
            admission_infeasible=admission_infeasible,
            planned_release=planned_release,
            actual_release=actual_release,
            candidate_release_capacity=candidate_release_capacity,
            requested_release=requested_release,
            landing_decision=landing_decision,
            frontier_summary=frontier_summary,
        )
        logger.info(
            "ReFlexKV trace admission_control request=%s requested_release=%d "
            "candidate_release_capacity=%d feasible_release=%d "
            "planned_release=%d actual_release=%d "
            "admission_success_after_demote=%s admission_blocked=%s "
            "admission_infeasible=%s admission_wait_reduction=%d free_before=%d "
            "free_after_estimated=%d needed_blocks=%d reserve_blocks=%d "
            "landing_mixed_feasible=%s landing_required_int4_blocks=%d "
            "landing_eligible_int4_blocks=%d landing_planned_int4_blocks=%d "
            "landing_residual_bf16_deficit=%d landing_reason=%s "
            "landing_metadata_source=%s landing_real_risk_pages=%d "
            "landing_explicit_compressible_pages=%d "
            "landing_synthetic_pages=%d "
            "blocked_reason=%s frontier_age=%d frontier_levels=%s "
            "frontier_rejection_reasons=%s.",
            getattr(request, "request_id", "<unknown>"),
            requested_release,
            candidate_release_capacity,
            feasible_release,
            planned_release,
            actual_release,
            admission_success_after_demote,
            not admission_success_after_demote,
            admission_infeasible,
            admission_wait_reduction,
            free_blocks_before,
            free_after_estimated,
            needed_blocks,
            reserve_blocks,
            landing_mixed_feasible,
            landing_required_int4_blocks,
            landing_eligible_int4_blocks,
            landing_planned_int4_blocks,
            landing_residual_bf16_deficit,
            landing_reason,
            landing_metadata_source,
            landing_real_risk_pages,
            landing_explicit_compressible_pages,
            landing_synthetic_pages,
            blocked_reason,
            self._reflex_frontier_age_or_minus_one(frontier_summary),
            self._format_reflex_frontier_levels(frontier_summary),
            self._format_reflex_frontier_rejection_reasons(frontier_summary),
        )

    def _reflex_int4_step_has_prefill(
        self, num_scheduled_tokens: dict[str, int]
    ) -> bool:
        if self.cache_config.cache_dtype != "reflex_int4":
            return False
        for request_id, num_tokens in num_scheduled_tokens.items():
            if num_tokens <= 0:
                continue
            request = self.requests.get(request_id)
            if request is None:
                continue
            if request.num_computed_tokens < request.num_prompt_tokens:
                return True
        return False

    def _reflex_int4_should_defer_decode_for_prefill(self) -> bool:
        if self.cache_config.cache_dtype != "reflex_int4":
            return False
        has_prefill = any(
            self._reflex_int4_request_needs_local_prefill(request)
            for request in self.running
        )
        request_queue = self._select_waiting_queue_for_scheduling()
        if request_queue is not None:
            try:
                waiting_request = request_queue.peek_request()
            except IndexError:
                waiting_request = None
            if (
                waiting_request is not None
                and self._reflex_int4_request_needs_local_prefill(waiting_request)
            ):
                has_prefill = True
        if not has_prefill:
            return False
        return any(
            self.kv_cache_manager.has_reflex_int4_blocks(request.request_id)
            for request in self.running
            if request.num_computed_tokens >= request.num_prompt_tokens
        )

    def _reflex_int4_request_needs_local_prefill(self, request: Request) -> bool:
        if request.num_computed_tokens >= request.num_prompt_tokens:
            return False
        # On a disaggregated decode worker, a fresh waiting request may still
        # be unable to reserve its full remote prefix yet. Treating waiting
        # remote-prefill work as local prefill can starve already-running
        # mixed-precision decode requests, preventing them from releasing KV.
        if self.connector is not None:
            return False
        return True

    def _reflex_int4_should_defer_running_request(
        self,
        request: Request,
        defer_decode_for_prefill: bool | None = None,
    ) -> bool:
        if defer_decode_for_prefill is None:
            defer_decode_for_prefill = (
                self._reflex_int4_should_defer_decode_for_prefill()
            )
        if not defer_decode_for_prefill:
            return False
        if request.num_computed_tokens < request.num_prompt_tokens:
            return False
        return self.kv_cache_manager.has_reflex_int4_blocks(request.request_id)

    def _estimate_reflex_remaining_decode_tokens(self, request: Request) -> int:
        max_tokens = int(getattr(request, "max_tokens", 0) or 0)
        output_token_ids = getattr(request, "output_token_ids", ())
        try:
            generated_tokens = len(output_token_ids)
        except TypeError:
            generated_tokens = 0
        return max(0, max_tokens - generated_tokens)

    def _reflex_int4_slo_demotion_pressure(self, request: Request) -> float:
        try:
            request_priority = int(getattr(request, "priority", 0) or 0)
        except (TypeError, ValueError):
            request_priority = 0
        pressure = 1.0 + (request_priority * self._reflex_int4_slo_pressure_step)
        lower = min(
            self._reflex_int4_min_slo_pressure,
            self._reflex_int4_max_slo_pressure,
        )
        upper = max(
            self._reflex_int4_min_slo_pressure,
            self._reflex_int4_max_slo_pressure,
        )
        return min(upper, max(lower, pressure))

    def _reflex_int4_generated_decode_tokens(self, request: Request) -> int:
        output_token_ids = getattr(request, "output_token_ids", ())
        try:
            return max(0, len(output_token_ids))
        except TypeError:
            return 0

    def _reflex_int4_prompt_pages(self, request: Request) -> int:
        try:
            prompt_tokens = int(getattr(request, "num_prompt_tokens", 0) or 0)
        except (TypeError, ValueError):
            prompt_tokens = 0
        if prompt_tokens <= 0:
            return 0
        return (prompt_tokens + self.block_size - 1) // self.block_size

    def _reflex_int4_protected_prompt_pages(self, request: Request) -> int:
        prompt_pages = self._reflex_int4_prompt_pages(request)
        if prompt_pages <= 0:
            return 0
        max_prompt_pages = getattr(
            self,
            "_reflex_int4_reasoning_prompt_protection_max_pages",
            0,
        )
        if max_prompt_pages <= 0 or prompt_pages > max_prompt_pages:
            return 0

        generated_decode_tokens = self._reflex_int4_generated_decode_tokens(request)
        remaining_decode_tokens = self._estimate_reflex_remaining_decode_tokens(request)
        decode_budget = generated_decode_tokens + remaining_decode_tokens
        min_decode_tokens = getattr(
            self,
            "_reflex_int4_reasoning_prompt_protection_min_decode_tokens",
            1024,
        )
        if decode_budget < max(0, min_decode_tokens):
            return 0
        return prompt_pages

    def _reflex_int4_protected_prompt_page_indices(
        self,
        request: Request,
    ) -> set[int]:
        prompt_pages = self._reflex_int4_prompt_pages(request)
        if prompt_pages <= 0:
            return set()

        protected_prefix_pages = min(
            prompt_pages,
            self._reflex_int4_protected_prompt_pages(request),
        )
        protected_pages = set(range(protected_prefix_pages))
        if protected_prefix_pages >= prompt_pages:
            return protected_pages

        long_prompt_threshold = max(
            0,
            int(getattr(self, "_reflex_int4_short_prefill_pages", 64) or 0),
        )
        if prompt_pages <= long_prompt_threshold:
            return protected_pages

        head_pages = min(
            prompt_pages,
            max(
                0,
                int(
                    getattr(
                        self,
                        "_reflex_int4_long_prompt_protected_head_pages",
                        self._reflex_int4_keep_initial_pages,
                    )
                    or 0
                ),
            ),
        )
        if head_pages > 0:
            protected_pages.update(range(head_pages))

        tail_pages = min(
            prompt_pages,
            max(
                0,
                int(
                    getattr(
                        self,
                        "_reflex_int4_long_prompt_protected_tail_pages",
                        self._reflex_int4_keep_initial_pages,
                    )
                    or 0
                ),
            ),
        )
        if tail_pages > 0:
            protected_pages.update(range(prompt_pages - tail_pages, prompt_pages))

        threshold = float(
            getattr(
                self,
                "_reflex_int4_prompt_high_risk_protection_threshold",
                0.85,
            )
        )
        params = getattr(request, "kv_transfer_params", None) or {}
        raw_risks = (
            params.get("reflex_page_risks") if isinstance(params, dict) else None
        )
        if isinstance(raw_risks, (list, tuple)):
            for page_idx, risk in enumerate(raw_risks[:prompt_pages]):
                try:
                    risk_score = float(risk)
                except (TypeError, ValueError):
                    continue
                if risk_score >= threshold:
                    protected_pages.add(page_idx)
        return protected_pages

    @staticmethod
    def _reflex_contiguous_prefix_len(page_indices: set[int]) -> int:
        prefix_len = 0
        while prefix_len in page_indices:
            prefix_len += 1
        return prefix_len

    @staticmethod
    def _reflex_int4_ramp_pressure(
        *,
        value: int,
        low_at: int,
        high_at: int,
        low_pressure: float,
        high_pressure: float,
    ) -> float:
        if high_at <= low_at:
            return high_pressure if value >= high_at else low_pressure
        if value <= low_at:
            return low_pressure
        if value >= high_at:
            return high_pressure
        ratio = (value - low_at) / max(1, high_at - low_at)
        return low_pressure + ratio * (high_pressure - low_pressure)

    def _reflex_int4_decode_demotion_pressure(self, request: Request) -> float:
        generated_tokens = self._reflex_int4_generated_decode_tokens(request)
        return self._reflex_int4_ramp_pressure(
            value=generated_tokens,
            low_at=self._reflex_int4_decode_pressure_warmup_tokens,
            high_at=self._reflex_int4_decode_pressure_ramp_tokens,
            low_pressure=0.5,
            high_pressure=1.25,
        )

    def _reflex_int4_prefill_demotion_pressure(self, request: Request) -> float:
        prompt_pages = self._reflex_int4_prompt_pages(request)
        return self._reflex_int4_ramp_pressure(
            value=prompt_pages,
            low_at=self._reflex_int4_short_prefill_pages,
            high_at=self._reflex_int4_long_prefill_pages,
            low_pressure=0.5,
            high_pressure=1.5,
        )

    def _reflex_int4_request_demotion_pressure(self, request: Request) -> float:
        return (
            self._reflex_int4_slo_demotion_pressure(request)
            * self._reflex_int4_decode_demotion_pressure(request)
            * self._reflex_int4_prefill_demotion_pressure(request)
        )

    def _reflex_int4_existing_int4_pages(self, request_id: str) -> int:
        get_counts = getattr(
            self.kv_cache_manager,
            "get_reflex_precision_state_counts",
            None,
        )
        if not callable(get_counts):
            return 0
        try:
            counts = get_counts(request_id)
        except (KeyError, RuntimeError):
            return 0
        return max(0, int(counts.get("INT4_ACTIVE", 0) or 0))

    def _reflex_int4_existing_int4_page_indices(
        self,
        request_id: str,
    ) -> set[int]:
        if not request_id:
            return set()
        get_descriptors = getattr(
            self.kv_cache_manager,
            "get_reflex_page_runtime_descriptors",
            None,
        )
        if not callable(get_descriptors):
            coordinator = getattr(self.kv_cache_manager, "coordinator", None)
            single_type_managers = getattr(
                coordinator,
                "single_type_managers",
                (),
            )

            def get_descriptors(request_id: str):
                descriptors = []
                for manager in single_type_managers:
                    manager_get = getattr(
                        manager,
                        "get_reflex_page_runtime_descriptors",
                        None,
                    )
                    if callable(manager_get):
                        descriptors.extend(manager_get(request_id))
                return descriptors

        try:
            descriptors = get_descriptors(request_id)
        except (KeyError, RuntimeError, ValueError, TypeError):
            return set()
        return {
            int(descriptor.page_idx)
            for descriptor in descriptors
            if getattr(descriptor, "precision", None) == PrecisionState.INT4
        }

    def _build_reflex_int4_request_precision_budgets(
        self,
        *,
        reason: str,
        target_bf16_blocks: int,
        pressure_decision: PrecisionPressureDecision | None = None,
    ) -> dict[str, RequestPrecisionBudget]:
        if self.cache_config.cache_dtype != "reflex_int4":
            return {}
        if pressure_decision is None:
            pressure_decision = self._plan_reflex_int4_pressure_policy(
                reason=reason,
                target_bf16_blocks=target_bf16_blocks,
            )

        requests = getattr(self, "requests", {})
        candidate_requests = [
            request
            for request in requests.values()
            if (
                (
                    request.num_computed_tokens >= request.num_prompt_tokens
                    or self._reflex_int4_has_page_level_demotion_frontier(request)
                )
                and not self._is_reflex_int4_demotion_protected_request(request)
            )
        ]
        if not candidate_requests:
            return {}

        max_page_count = max(
            max(
                0,
                request.num_computed_tokens // self.block_size,
                self._reflex_remote_chunk_sealed_pages(request),
            )
            for request in candidate_requests
        )
        max_page_count = max(1, max_page_count)
        short_decode_tokens = self._reflex_int4_short_decode_tokens
        short_decode_fraction = min(
            self._reflex_int4_max_int4_fraction_per_request,
            self._reflex_int4_short_decode_max_int4_fraction,
        )
        short_admission_fraction = min(
            self._reflex_int4_max_int4_fraction_per_request,
            self._reflex_int4_short_admission_max_int4_fraction,
        )
        admission_pressure = reason in {
            "admission_waiting",
            "allocation_failure",
            "full_sequence_reserve",
        }
        block_pool = self.kv_cache_manager.block_pool
        bf16_free_ratio = block_pool.get_num_free_blocks() / max(
            1, block_pool.num_gpu_blocks
        )
        cold_admission_emergency = admission_pressure and (
            target_bf16_blocks > 0
            or bf16_free_ratio <= self._reflex_int4_cold_admission_emergency_free_ratio
        )
        cold_admission_fraction = min(
            self._reflex_int4_max_int4_fraction_per_request,
            self._reflex_int4_cold_admission_max_int4_fraction,
        )

        budgets: dict[str, RequestPrecisionBudget] = {}
        candidates: list[RequestBudgetCandidate] = []
        budget_inputs: dict[
            str, tuple[int, int, float, float, int, bool, int, int, int, int]
        ] = {}
        budget_trace_inputs: dict[str, tuple[int, int, int, int, int]] = {}
        for request in candidate_requests:
            page_count = max(
                0,
                request.num_computed_tokens // self.block_size,
                self._reflex_remote_chunk_sealed_pages(request),
            )
            if page_count <= 0:
                continue
            remaining_decode_tokens = self._estimate_reflex_remaining_decode_tokens(
                request
            )
            generated_decode_tokens = self._reflex_int4_generated_decode_tokens(request)
            demotion_pressure = self._reflex_int4_request_demotion_pressure(request)
            max_fraction = self._reflex_int4_max_int4_fraction_per_request
            is_short_decode = remaining_decode_tokens <= short_decode_tokens
            cold_fraction_cap: float | None = None
            if generated_decode_tokens < self._reflex_int4_risk_warmup_tokens:
                if cold_admission_emergency and cold_admission_fraction > 0.0:
                    max_fraction = cold_admission_fraction
                    if is_short_decode:
                        max_fraction = min(max_fraction, short_admission_fraction)
                    cold_fraction_cap = max_fraction
                else:
                    max_fraction = 0.0
            elif is_short_decode:
                max_fraction = min(
                    max_fraction,
                    short_admission_fraction
                    if admission_pressure
                    else short_decode_fraction,
                )
            elif generated_decode_tokens < self._reflex_int4_survival_warmup_tokens:
                max_fraction = max_fraction if admission_pressure else 0.0
            elif not admission_pressure:
                max_fraction = min(
                    max_fraction,
                    max(short_decode_fraction, max_fraction * 0.5),
                )
            if max_fraction > 0.0:
                cap_pressure = demotion_pressure
                if is_short_decode and admission_pressure:
                    cap_pressure = self._reflex_int4_slo_demotion_pressure(request)
                max_fraction = min(
                    self._reflex_int4_max_int4_fraction_per_request,
                    max_fraction * cap_pressure,
                )
                if cold_fraction_cap is not None:
                    max_fraction = min(max_fraction, cold_fraction_cap)
                if (
                    admission_pressure
                    and generated_decode_tokens
                    >= self._reflex_int4_survival_warmup_tokens
                    and target_bf16_blocks > self._reflex_int4_admission_reserve_blocks
                ):
                    min_pressure_fraction = (
                        self._reflex_int4_admission_pressure_min_int4_fraction
                        * self._reflex_int4_slo_demotion_pressure(request)
                    )
                    max_fraction = min(
                        self._reflex_int4_max_int4_fraction_per_request,
                        max(max_fraction, min_pressure_fraction),
                    )
            max_int4_pages = int(page_count * max_fraction)
            prompt_pages = min(self._reflex_int4_prompt_pages(request), page_count)
            decode_pages = max(0, page_count - prompt_pages)
            max_prompt_int4_pages = min(
                max_int4_pages,
                int(prompt_pages * max_fraction),
            )
            max_decode_int4_pages = min(
                max(0, max_int4_pages - max_prompt_int4_pages),
                int(decode_pages * max_fraction),
            )
            quality_debt_budget_pages = int(
                page_count
                * min(
                    max_fraction,
                    getattr(
                        self,
                        "_reflex_int4_quality_debt_max_fraction",
                        1.0,
                    ),
                )
            )
            existing_int4_pages = self._reflex_int4_existing_int4_pages(
                request.request_id
            )
            remaining_int4_capacity = max(
                0,
                max_int4_pages - existing_int4_pages,
            )
            max_demote_per_window = (
                self._reflex_int4_short_max_demote_per_window
                if is_short_decode
                else self._reflex_int4_max_demote_per_window
            )
            if admission_pressure:
                max_demote_per_window = max(
                    max_demote_per_window,
                    self._reflex_int4_admission_max_demote_per_window,
                )
            if (
                is_short_decode
                and admission_pressure
                and self._reflex_int4_slo_demotion_pressure(request) > 1.0
            ):
                max_demote_per_window = max(
                    max_demote_per_window,
                    min(self._reflex_int4_max_demote_per_window, 2),
                )
            if (
                admission_pressure
                and pressure_decision.max_demote_per_window_multiplier > 1.0
            ):
                max_demote_per_window = max(
                    max_demote_per_window,
                    int(
                        math.ceil(
                            max_demote_per_window
                            * pressure_decision.max_demote_per_window_multiplier
                        )
                    ),
                )
            footprint_score = page_count / max_page_count
            remaining_score = min(
                1.0,
                remaining_decode_tokens / max(1, short_decode_tokens),
            )
            priority = (
                (page_count * (0.5 + remaining_score)) + footprint_score
            ) * demotion_pressure
            budget_inputs[request.request_id] = (
                max_int4_pages,
                remaining_int4_capacity,
                priority,
                max_fraction,
                page_count,
                is_short_decode,
                max_demote_per_window,
                max_prompt_int4_pages,
                max_decode_int4_pages,
                quality_debt_budget_pages,
            )
            try:
                request_priority = int(getattr(request, "priority", 0) or 0)
            except (TypeError, ValueError):
                request_priority = 0
            budget_trace_inputs[request.request_id] = (
                request_priority,
                generated_decode_tokens,
                remaining_decode_tokens,
                self._reflex_int4_prompt_pages(request),
                max_demote_per_window,
            )
            candidates.append(
                RequestBudgetCandidate(
                    request_id=request.request_id,
                    capacity_blocks=remaining_int4_capacity,
                    utility=priority,
                )
            )

        release_budget_target = int(
            target_bf16_blocks
            * max(1.0, pressure_decision.request_release_budget_multiplier)
        )
        release_budgets = allocate_request_release_budgets(
            candidates,
            target_bf16_blocks=release_budget_target,
        )
        for request_id, (
            max_int4_pages,
            remaining_int4_capacity,
            priority,
            max_fraction,
            _page_count,
            _is_short_decode,
            max_demote_per_window,
            max_prompt_int4_pages,
            max_decode_int4_pages,
            quality_debt_budget_pages,
        ) in budget_inputs.items():
            budgets[request_id] = RequestPrecisionBudget(
                max_int4_pages=max_int4_pages,
                priority=priority,
                max_int4_fraction=max_fraction,
                release_budget_blocks=release_budgets.get(request_id, 0),
                max_demote_per_window=max_demote_per_window,
                max_prompt_int4_pages=max_prompt_int4_pages,
                max_decode_int4_pages=max_decode_int4_pages,
                quality_debt_budget_pages=quality_debt_budget_pages,
            )
            (
                request_priority,
                generated_decode_tokens,
                remaining_decode_tokens,
                prompt_pages,
                trace_max_demote_per_window,
            ) = budget_trace_inputs[request_id]
            logger.info(
                "ReFlexKV trace precision_budget request=%s "
                "max_int4_pages=%d priority=%.6f max_int4_fraction=%.6f "
                "remaining_int4_capacity=%d release_budget_blocks=%d "
                "max_demote_per_window=%d "
                "request_priority=%d generated_decode_tokens=%d "
                "remaining_decode_tokens=%d prompt_pages=%d.",
                request_id,
                max_int4_pages,
                priority,
                max_fraction,
                remaining_int4_capacity,
                release_budgets.get(request_id, 0),
                trace_max_demote_per_window,
                request_priority,
                generated_decode_tokens,
                remaining_decode_tokens,
                prompt_pages,
            )
        return budgets

    def _build_reflex_prefill_page_metadata_inputs(
        self,
        *,
        low_risk_score_fraction: float | None = None,
    ) -> tuple[dict[str, list[float]], dict[str, set[int]]]:
        page_risks_by_request: dict[str, list[float]] = {}
        compressible_pages_by_request: dict[str, set[int]] = {}
        if low_risk_score_fraction is None:
            low_risk_score_fraction = self._reflex_int4_low_risk_score_fraction
        low_risk_score_fraction = min(1.0, max(0.0, low_risk_score_fraction))
        for request_id, request in getattr(self, "requests", {}).items():
            params = getattr(request, "kv_transfer_params", None) or {}
            raw_risks = params.get("reflex_page_risks")
            if isinstance(raw_risks, (list, tuple)):
                page_risks_by_request[request_id] = [
                    float(score) for score in raw_risks
                ]

            raw_pages = params.get("reflex_compressible_pages")
            if isinstance(raw_pages, (list, tuple, set)):
                compressible_pages_by_request[request_id] = {
                    int(page_idx) for page_idx in raw_pages
                }
                continue

            raw_mask = params.get("reflex_compressible_mask")
            if isinstance(raw_mask, (list, tuple)):
                compressible_pages_by_request[request_id] = {
                    idx for idx, enabled in enumerate(raw_mask) if bool(enabled)
                }
                continue

            page_risks = page_risks_by_request.get(request_id)
            if page_risks:
                low_risk_pages = derive_compressible_pages_from_risks(
                    page_risks,
                    fraction=low_risk_score_fraction,
                )
                if low_risk_pages:
                    compressible_pages_by_request[request_id] = set(low_risk_pages)
        return page_risks_by_request, compressible_pages_by_request

    def _reflex_int4_free_ratio(self) -> float:
        block_pool = self.kv_cache_manager.block_pool
        return block_pool.get_num_free_blocks() / max(1, block_pool.num_gpu_blocks)

    def _try_reflex_int4_background_promote(self) -> int:
        if self.cache_config.cache_dtype != "reflex_int4":
            return 0
        if not hasattr(self.kv_cache_manager, "promote_reflex_recoverable_pages"):
            return 0
        max_promotion_pages = getattr(
            self,
            "_reflex_int4_background_promotion_pages_per_step",
            0,
        )
        if max_promotion_pages <= 0:
            return 0
        if getattr(self, "waiting", None) or getattr(self, "skipped_waiting", None):
            return 0
        free_ratio = self._reflex_int4_free_ratio()
        promotion_free_ratio = getattr(
            self,
            "_reflex_int4_background_promotion_free_ratio",
            0.60,
        )
        if free_ratio < promotion_free_ratio:
            return 0

        requests = getattr(self, "requests", {})
        if not requests:
            return 0
        prefill_page_risks_by_request, _ = (
            self._build_reflex_prefill_page_metadata_inputs()
        )
        remaining_decode_tokens_by_request = {
            request_id: self._estimate_reflex_remaining_decode_tokens(request)
            for request_id, request in requests.items()
        }
        promoted = self.kv_cache_manager.promote_reflex_recoverable_pages(
            max_pages=max_promotion_pages,
            prefill_page_risks_by_request=prefill_page_risks_by_request,
            remaining_decode_tokens_by_request=remaining_decode_tokens_by_request,
            min_remaining_decode_tokens=getattr(
                self,
                "_reflex_int4_promotion_min_remaining_decode_tokens",
                16,
            ),
        )
        if promoted > 0:
            logger.info(
                "ReFlexKV trace recovery_plan reason=background_promotion "
                "promoted_pages=%d free_ratio=%.4f.",
                promoted,
                free_ratio,
            )
        return promoted

    def _try_reflex_int4_demote(
        self,
        *,
        target_bf16_blocks: int,
        force: bool = False,
        reason: str = "pressure",
    ) -> int:
        if self.cache_config.cache_dtype != "reflex_int4":
            return 0
        if target_bf16_blocks <= 0:
            return 0
        if (
            not force
            and self._reflex_int4_scheduler_step
            - self._reflex_int4_last_demote_step
            < self._reflex_int4_demote_cooldown_steps
        ):
            return 0
        block_pool = self.kv_cache_manager.block_pool
        free_blocks_before = block_pool.get_num_free_blocks()
        plan_start = time.perf_counter()
        plan_kwargs = self._build_reflex_int4_demotion_planning_kwargs(
            target_bf16_blocks=target_bf16_blocks,
            reason=reason,
        )
        planned_blocks = self.kv_cache_manager.plan_reflex_int4_demotions(
            **plan_kwargs,
            dry_run=False,
        )
        if hasattr(
            self.kv_cache_manager,
            "get_last_reflex_int4_candidate_capacity",
        ):
            self._reflex_int4_last_demote_candidate_capacity = (
                self.kv_cache_manager.get_last_reflex_int4_candidate_capacity()
            )
        else:
            self._reflex_int4_last_demote_candidate_capacity = planned_blocks
        plan_ms = (time.perf_counter() - plan_start) * 1000.0
        if hasattr(
            self.kv_cache_manager,
            "get_last_reflex_int4_candidate_breakdown",
        ):
            breakdown = self.kv_cache_manager.get_last_reflex_int4_candidate_breakdown()
            self._reflex_int4_last_candidate_breakdown = breakdown
            summary = FeasibleFrontierSummary.from_candidate_breakdown(
                scheduler_step=getattr(self, "_reflex_int4_scheduler_step", 0),
                reason=reason,
                target_release=target_bf16_blocks,
                feasible_release=planned_blocks,
                candidate_breakdown=breakdown,
            )
            self._get_reflex_int4_frontier_cache().update(summary)
            after_frontier_optimizer = getattr(
                breakdown,
                "after_frontier_optimizer",
                breakdown.after_sparse_window_quota,
            )
            logger.info(
                "ReFlexKV trace candidate_breakdown reason=%s "
                "selection_policy=%s target_release=%d actual_release=%d "
                "emergency_release=%s "
                "raw_bf16_pages=%d open_bf16_pages=%d "
                "remote_inflight_bf16_pages=%d open_tail_bf16_pages=%d "
                "request_protected_bf16_pages=%d shared_bf16_pages=%d "
                "prompt_protected_bf16_pages=%d copy_on_demote_pages=%d "
                "eligible_full_unshared_pages=%d "
                "after_initial_recent_protection=%d "
                "after_low_risk_filter=%d after_request_budget_cap=%d "
                "after_sparse_window_quota=%d after_frontier_optimizer=%d "
                "after_int4_pool_limit=%d int4_free_blocks=%d "
                "frontier_optimizer_budget=%d "
                "selected_actual=%d frontier_levels=%s "
                "rejection_reasons=%s.",
                reason,
                plan_kwargs["selection_policy"],
                target_bf16_blocks,
                planned_blocks,
                plan_kwargs["emergency_release"],
                breakdown.raw_bf16_pages,
                getattr(breakdown, "open_bf16_pages", 0),
                getattr(breakdown, "remote_inflight_bf16_pages", 0),
                getattr(breakdown, "open_tail_bf16_pages", 0),
                getattr(breakdown, "request_protected_bf16_pages", 0),
                getattr(breakdown, "shared_bf16_pages", 0),
                getattr(breakdown, "prompt_protected_bf16_pages", 0),
                getattr(breakdown, "copy_on_demote_pages", 0),
                breakdown.eligible_full_unshared_pages,
                breakdown.after_initial_recent_protection,
                breakdown.after_low_risk_filter,
                breakdown.after_request_budget_cap,
                breakdown.after_sparse_window_quota,
                after_frontier_optimizer,
                breakdown.after_int4_pool_limit,
                getattr(breakdown, "int4_free_blocks", -1),
                getattr(breakdown, "frontier_optimizer_budget", -1),
                breakdown.selected_actual,
                summary.format_levels(),
                summary.format_rejection_reasons(),
            )
        if planned_blocks > 0:
            self._get_reflex_int4_frontier_cache().invalidate()
            self._record_reflex_int4_frontier_event("bf16_freed")
            self._reflex_int4_last_demote_step = self._reflex_int4_scheduler_step
            skipped_pages = max(0, target_bf16_blocks - planned_blocks)
            logger.info(
                "ReFlexKV planned BF16->INT4 KV block demotions for %s; "
                "target_release=%d actual_release=%d skipped_pages=%d "
                "candidate_release_capacity=%d bf16_free=%d/%d "
                "bf16_free_before=%d plan_ms=%.3f.",
                reason,
                target_bf16_blocks,
                planned_blocks,
                skipped_pages,
                self._reflex_int4_last_demote_candidate_capacity,
                block_pool.get_num_free_blocks(),
                block_pool.num_gpu_blocks,
                free_blocks_before,
                plan_ms,
            )
        elif reason == "background_pressure" and not force:
            self._reflex_int4_last_demote_step = self._reflex_int4_scheduler_step
        logger.info(
            "ReFlexKV trace stage_profile request=all phase=planner "
            "ms=%.3f reason=%s target_release=%d actual_release=%d.",
            plan_ms,
            reason,
            target_bf16_blocks,
            planned_blocks,
        )
        return planned_blocks

    def _build_kv_connector_meta(
        self, connector: KVConnectorBase_V1, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return connector.build_connector_meta(scheduler_output)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        """Preempt a request and put it back to the waiting queue.

        NOTE: The request should be popped from the running queue outside of this
        method.
        """
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        self.kv_cache_manager.free(request)
        self.encoder_cache_manager.free(request)
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        if request.spec_token_ids:
            request.spec_token_ids = []
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        # Advance the number of computed tokens for the request AFTER
        # the request is scheduled.
        # 1. The scheduler_output of the current step has to include the
        #    original number of scheduled tokens to determine input IDs.
        # 2. Advance the number of computed tokens here allowing us to
        #    schedule the prefill request again immediately in the next
        #    scheduling step.
        # 3. If some tokens (e.g. spec tokens) are rejected later, the number of
        #    computed tokens will be adjusted in update_from_output.
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        for req_id, num_scheduled_token in num_scheduled_tokens.items():
            request = self.requests[req_id]
            request.num_computed_tokens += num_scheduled_token
            request.is_prefill_chunk = request.num_computed_tokens < (
                request.num_tokens + request.num_output_placeholders
            )
            scheduler_output.has_structured_output_requests |= (
                request.use_structured_output and not request.is_prefill_chunk
            )

            # NOTE: _free_encoder_inputs relies on num_computed_tokens, which
            # may be updated again in _update_from_output for speculative
            # decoding. However, it is safe to call the method here because
            # encoder inputs are always part of the prompt, not the output,
            # and thus are unaffected by speculative decoding.
            if request.has_encoder_inputs:
                self._free_encoder_inputs(request)

        # Clear the finished request IDs.
        # NOTE: We shouldn't do self.finished_req_ids.clear() here because
        # it will also affect the scheduler output.
        self.finished_req_ids = set()

    def _update_request_as_session(
        self, session: Request, update: StreamingUpdate
    ) -> None:
        """
        Updates the waiting session with the next streaming update.

        Discards the last sampled output token from the prior input chunk.
        """

        # Current streaming input behaviour: Keep only computed output tokens
        # (discard final sampled output token).
        num_computed_tokens = session.num_computed_tokens
        kept_output_tokens = session._all_token_ids[
            session.num_prompt_tokens : num_computed_tokens
        ]
        del session._all_token_ids[num_computed_tokens:]
        session._output_token_ids.clear()
        assert session.prompt_token_ids is not None
        # Extend prompt with kept output tokens.
        session.prompt_token_ids.extend(kept_output_tokens)

        if update.mm_features:
            base = session.num_tokens
            for mm_feature in update.mm_features:
                mm_feature.mm_position = replace(
                    mm_feature.mm_position, offset=mm_feature.mm_position.offset + base
                )
            session.mm_features.extend(update.mm_features)

        session._all_token_ids.extend(update.prompt_token_ids or ())
        session.prompt_token_ids.extend(update.prompt_token_ids or ())
        # Update block hashes for the new tokens.
        session.update_block_hashes()
        session.num_prompt_tokens = len(session.prompt_token_ids)
        session.arrival_time = update.arrival_time
        session.sampling_params = update.sampling_params
        if session.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.num_waiting_for_streaming_input -= 1
        session.status = RequestStatus.WAITING

        if self.log_stats:
            session.record_event(EngineCoreEventType.QUEUED)

    def _make_cached_request_data(
        self,
        running_reqs: list[Request],
        resumed_reqs: list[Request],
        num_scheduled_tokens: dict[str, int],
        spec_decode_tokens: dict[str, list[int]],
        req_to_new_blocks: dict[str, KVCacheBlocks],
    ) -> CachedRequestData:
        req_ids: list[str] = []
        new_token_ids: list[list[int]] = []
        new_block_ids: list[tuple[list[int], ...] | None] = []
        all_token_ids: dict[str, list[int]] = {}
        num_computed_tokens: list[int] = []
        num_output_tokens: list[int] = []
        resumed_req_ids = set()

        num_running_reqs = len(running_reqs)
        for idx, req in enumerate(itertools.chain(running_reqs, resumed_reqs)):
            req_id = req.request_id
            req_ids.append(req_id)
            # NOTE: In PP+async scheduling, we consume token ids via a direct GPU
            # broadcast path (`input_batch.prev_sampled_token_ids`), so we can
            # omit this payload.
            if self.use_pp and not self.scheduler_config.async_scheduling:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker. Otherwise, we don't
                # need to send the sampled tokens back because the model runner
                # will cache them.
                num_tokens = num_scheduled_tokens[req_id] - len(
                    spec_decode_tokens.get(req_id, ())
                )
                token_ids = req.all_token_ids[
                    req.num_computed_tokens : req.num_computed_tokens + num_tokens
                ]
                new_token_ids.append(token_ids)
            scheduled_in_prev_step = req_id in self.prev_step_scheduled_req_ids
            if idx >= num_running_reqs:
                assert not scheduled_in_prev_step
                resumed_req_ids.add(req_id)
            if not scheduled_in_prev_step:
                all_token_ids[req_id] = req.all_token_ids.copy()
            new_block_ids.append(
                req_to_new_blocks[req_id].get_block_ids(allow_none=True)
            )
            num_computed_tokens.append(req.num_computed_tokens)
            num_output_tokens.append(
                req.num_output_tokens + req.num_output_placeholders
            )

        return CachedRequestData(
            req_ids=req_ids,
            resumed_req_ids=resumed_req_ids,
            new_token_ids=new_token_ids,
            all_token_ids=all_token_ids,
            new_block_ids=new_block_ids,
            num_computed_tokens=num_computed_tokens,
            num_output_tokens=num_output_tokens,
        )

    def _try_schedule_encoder_inputs(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        encoder_compute_budget: int,
        shift_computed_tokens: int = 0,
    ) -> tuple[list[int], int, int, list[int]]:
        """
        Determine which encoder inputs need to be scheduled in the current step,
        and update `num_new_tokens` and encoder token budget accordingly.

        An encoder input will be scheduled if:
        - Its output tokens overlap with the range of tokens being computed
        in this step, i.e.,
        [num_computed_tokens, num_computed_tokens + num_new_tokens).
        - It is not already computed and stored in the encoder cache.
        - It is not exist on remote encoder cache (via ECConnector)
        - There is sufficient encoder token budget to process it.
        - The encoder cache has space to store it.

        If an encoder input cannot be scheduled due to cache or budget
        limitations, the method adjusts `num_new_tokens` to schedule only the
        decoder tokens up to just before the unschedulable encoder input.

        Note that num_computed_tokens includes both locally cached
        blocks and externally cached blocks (via KVConnector).
        """
        if num_new_tokens == 0 or not request.has_encoder_inputs:
            return [], num_new_tokens, encoder_compute_budget, []
        encoder_inputs_to_schedule: list[int] = []
        mm_features = request.mm_features
        assert mm_features is not None
        assert len(mm_features) > 0
        external_load_encoder_input = []

        # NOTE: since scheduler operates on the request level (possibly with
        # multiple encoder inputs per request), we need to create temporary
        # trackers for accounting at the encoder input level.
        mm_hashes_to_schedule = set()
        num_embeds_to_schedule = 0
        for i, mm_feature in enumerate(mm_features):
            start_pos = mm_feature.mm_position.offset
            num_encoder_tokens = mm_feature.mm_position.length
            num_encoder_embeds = mm_feature.mm_position.get_num_embeds()
            item_identifier = mm_feature.identifier

            # The encoder output is needed if the two ranges overlap:
            # [num_computed_tokens, num_computed_tokens + num_new_tokens) and
            # [start_pos, start_pos + num_encoder_tokens)
            if (
                start_pos
                >= num_computed_tokens + num_new_tokens + shift_computed_tokens
            ):
                # The encoder input is not needed in this step.
                break

            if self.is_encoder_decoder and num_computed_tokens > 0:
                assert start_pos == 0, (
                    "Encoder input should be processed at the beginning of "
                    "the sequence when encoder-decoder models are used."
                )
                # Encoder input has already been computed
                # The calculation here is a bit different. We don't turn encoder
                # output into tokens that get processed by the decoder and
                # reflected in num_computed_tokens. Instead, start_pos reflects
                # the position where we need to ensure we calculate encoder
                # inputs. This should always be 0 to ensure we calculate encoder
                # inputs before running the decoder.  Once we've calculated some
                # decoder tokens (num_computed_tokens > 0), then we know we
                # already calculated encoder inputs and can skip here.
                continue
            elif start_pos + num_encoder_tokens <= num_computed_tokens:
                # The encoder input is already computed and stored
                # in the decoder's KV cache.
                continue

            if not self.is_encoder_decoder:
                # We are not using the encoder cache for encoder-decoder models,
                # yet.
                if item_identifier in mm_hashes_to_schedule:
                    # The same encoder input has already been scheduled in the
                    # current step.
                    continue

                if self.encoder_cache_manager.check_and_update_cache(request, i):
                    # The encoder input is already computed and cached from a
                    # previous step.
                    continue

            # If no encoder input chunking is allowed, we do not want to
            # partially schedule a multimodal item. If the scheduled range would
            # only cover part of the mm input, roll back to before the mm item.
            if (
                self.scheduler_config.disable_chunked_mm_input
                and num_computed_tokens < start_pos
                and (num_computed_tokens + num_new_tokens)
                < (start_pos + num_encoder_tokens)
            ):
                # Account for EAGLE shift when rolling back to avoid
                # encoder cache miss. This ensures the scheduled range
                # stops before start_pos even with the shift.
                num_new_tokens = max(
                    0, start_pos - (num_computed_tokens + shift_computed_tokens)
                )
                break
            if not self.encoder_cache_manager.can_allocate(
                request, i, encoder_compute_budget, num_embeds_to_schedule
            ):
                # The encoder cache is full or the encoder budget is exhausted.
                # NOTE(woosuk): We assume that the encoder input tokens should
                # be processed altogether, as the encoder usually uses
                # bidirectional attention.
                if num_computed_tokens + shift_computed_tokens < start_pos:
                    # We only schedule the decoder tokens just before the
                    # encoder input.
                    num_new_tokens = start_pos - (
                        num_computed_tokens + shift_computed_tokens
                    )
                else:
                    # Because of prefix caching, num_computed_tokens is greater
                    # than start_pos even though its encoder input is not
                    # available. In this case, we can't schedule any token for
                    # the request in this step.
                    num_new_tokens = 0
                break

            # Calculate the number of embeddings to schedule in the current range
            # of scheduled encoder placeholder tokens.
            start_idx_rel = max(0, num_computed_tokens - start_pos)
            end_idx_rel = min(
                num_encoder_tokens, num_computed_tokens + num_new_tokens - start_pos
            )
            curr_embeds_start, curr_embeds_end = (
                mm_feature.mm_position.get_embeds_indices_in_range(
                    start_idx_rel, end_idx_rel
                )
            )
            # There's no embeddings in the current range of encoder placeholder tokens
            # so we can skip the encoder input.
            if curr_embeds_end - curr_embeds_start == 0:
                continue

            if self.ec_connector is not None and self.ec_connector.has_cache_item(
                item_identifier
            ):
                mm_hashes_to_schedule.add(item_identifier)
                external_load_encoder_input.append(i)
                num_embeds_to_schedule += num_encoder_embeds
                continue

            num_embeds_to_schedule += num_encoder_embeds
            encoder_compute_budget -= num_encoder_embeds
            mm_hashes_to_schedule.add(item_identifier)
            encoder_inputs_to_schedule.append(i)

        return (
            encoder_inputs_to_schedule,
            num_new_tokens,
            encoder_compute_budget,
            external_load_encoder_input,
        )

    def get_grammar_bitmask(
        self, scheduler_output: SchedulerOutput
    ) -> GrammarOutput | None:
        # Collect list of scheduled request ids that use structured output.
        # The corresponding rows of the bitmask will be in this order.
        if not scheduler_output.has_structured_output_requests:
            return None

        structured_output_request_ids = [
            req_id
            for req_id in scheduler_output.num_scheduled_tokens
            if (req := self.requests.get(req_id))
            and (req.use_structured_output and not req.is_prefill_chunk)
        ]
        if not structured_output_request_ids:
            return None

        bitmask = self.structured_output_manager.grammar_bitmask(
            self.requests,
            structured_output_request_ids,
            scheduler_output.scheduled_spec_decode_tokens,
        )
        return GrammarOutput(structured_output_request_ids, bitmask)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        pooler_outputs = model_runner_output.pooler_output
        num_nans_in_logits = model_runner_output.num_nans_in_logits
        kv_connector_output = model_runner_output.kv_connector_output
        cudagraph_stats = model_runner_output.cudagraph_stats

        perf_stats: PerfStats | None = None
        if self.perf_metrics and self.perf_metrics.is_enabled():
            perf_stats = self.perf_metrics.get_step_perf_stats_per_gpu(scheduler_output)

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        spec_decoding_stats: SpecDecodingStats | None = None
        kv_connector_stats: KVConnectorStats | None = (
            kv_connector_output.kv_connector_stats if kv_connector_output else None
        )
        if kv_connector_stats and self.connector:
            kv_stats = self.connector.get_kv_connector_stats()
            if kv_stats:
                kv_connector_stats = kv_connector_stats.aggregate(kv_stats)

        connector_output_applied = False
        if (
            kv_connector_output
            and kv_connector_output.kv_connector_worker_meta is not None
            and self.connector is not None
        ):
            self.connector.update_connector_output(kv_connector_output)
            connector_output_applied = True

        failed_kv_load_req_ids = None
        if kv_connector_output and kv_connector_output.invalid_block_ids:
            # These blocks contain externally computed tokens that failed to
            # load. Identify affected requests and adjust their computed token
            # count to trigger recomputation of the invalid blocks.
            failed_kv_load_req_ids = self._handle_invalid_blocks(
                kv_connector_output.invalid_block_ids,
                num_scheduled_tokens,
            )

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        stopped_running_reqs: set[Request] = set()
        stopped_preempted_reqs: set[Request] = set()
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            assert num_tokens_scheduled > 0
            if failed_kv_load_req_ids and req_id in failed_kv_load_req_ids:
                # skip failed or rescheduled requests from KV load failure
                continue
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request is already finished. This can happen if the
                # request is aborted while the model is executing it (e.g.,
                # in pipeline parallelism or in async scheduling).
                # NOTE(Kuntai): When delay_free_blocks=True (for async KV
                # cache transfer in KV connector), the aborted request will not
                # be set to None (in order to finish async KV transfer).
                # In this case, we use is_finished() to check.
                continue

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            scheduled_spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id)
            )
            if scheduled_spec_token_ids and generated_token_ids:
                num_draft_tokens = len(scheduled_spec_token_ids)
                num_accepted = len(generated_token_ids) - 1
                num_rejected = num_draft_tokens - num_accepted
                # num_computed_tokens represents the number of tokens
                # processed in the current step, considering scheduled
                # tokens and rejections. If some tokens are rejected,
                # num_computed_tokens is decreased by the number of rejected
                # tokens.
                if request.num_computed_tokens > 0:
                    request.num_computed_tokens -= num_rejected
                # If async scheduling, num_output_placeholders also includes
                # the scheduled spec tokens count and so is similarly adjusted.
                if request.num_output_placeholders > 0:
                    request.num_output_placeholders -= num_rejected
                spec_decoding_stats = self.make_spec_decoding_stats(
                    spec_decoding_stats,
                    num_draft_tokens=num_draft_tokens,
                    num_accepted_tokens=num_accepted,
                    num_invalid_spec_tokens=scheduler_output.num_invalid_spec_tokens,
                    request_id=req_id,
                )

            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            pooler_output = pooler_outputs[req_index] if pooler_outputs else None
            kv_transfer_params = None
            status_before_stop = request.status

            # Check for stop and update request status.
            if new_token_ids:
                new_token_ids, stopped = self._update_request_with_output(
                    request, new_token_ids
                )
            elif request.pooling_params and pooler_output is not None:
                # Pooling stops as soon as there is output.
                request.status = RequestStatus.FINISHED_STOPPED
                stopped = True

            routed_experts = None
            finish_reason = None
            if stopped:
                routed_experts = self._get_routed_experts(request)

                # Capture finish_reason BEFORE _handle_stopped_request, which may
                # reset the status to WAITING for streaming requests that continue.
                finish_reason = request.get_finished_reason()
                finished = self._handle_stopped_request(request)
                if finished:
                    kv_transfer_params = self._free_request(request)

                if status_before_stop == RequestStatus.RUNNING:
                    stopped_running_reqs.add(request)
                else:
                    stopped_preempted_reqs.add(request)

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            if new_token_ids and self.structured_output_manager.should_advance(request):
                struct_output_request = request.structured_output_request
                assert struct_output_request is not None
                assert struct_output_request.grammar is not None
                ok = struct_output_request.grammar.accept_tokens(req_id, new_token_ids)
                if not ok:
                    logger.warning(
                        "Unexpected: grammar rejected tokens %s for request %s.",
                        new_token_ids,
                        req_id,
                    )

            if num_nans_in_logits is not None and req_id in num_nans_in_logits:
                request.num_nans_in_logits = num_nans_in_logits[req_id]

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            if (
                new_token_ids
                or pooler_output is not None
                or kv_transfer_params
                or stopped
            ):
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        pooling_output=pooler_output,
                        stop_reason=request.stop_reason,
                        events=request.take_events(),
                        kv_transfer_params=kv_transfer_params,
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                        num_external_computed_tokens=request.num_external_computed_tokens,
                        routed_experts=routed_experts,
                        num_nans_in_logits=request.num_nans_in_logits,
                    )
                )
            else:
                # Invariant: EngineCore returns no partial prefill outputs.
                assert not prompt_logprobs_tensors

        # Remove the stopped requests from the running and waiting queues.
        if stopped_running_reqs:
            self.running = remove_all(self.running, stopped_running_reqs)
        if stopped_preempted_reqs:
            # This is a rare case and unlikely to impact performance.
            self.waiting.remove_requests(stopped_preempted_reqs)

        if failed_kv_load_req_ids and not self.recompute_kv_load_failures:
            requests = [self.requests[req_id] for req_id in failed_kv_load_req_ids]
            self.finish_requests(failed_kv_load_req_ids, RequestStatus.FINISHED_ERROR)
            for request in requests:
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=request.request_id,
                        new_token_ids=[],
                        finish_reason=request.get_finished_reason(),
                        events=request.take_events(),
                        trace_headers=request.trace_headers,
                        num_cached_tokens=request.num_cached_tokens,
                    )
                )

        # KV Connector: update state for finished KV Transfers.
        if kv_connector_output:
            self._update_from_kv_xfer_finished(
                kv_connector_output,
                update_connector=not connector_output_applied,
            )

        # collect KV cache events from KV cache manager
        events = self.kv_cache_manager.take_events()

        # collect KV cache events from connector
        if self.connector is not None:
            connector_events = self.connector.take_events()
            if connector_events:
                if events is None:
                    events = list(connector_events)
                else:
                    events.extend(connector_events)

        # publish collected KV cache events
        if events:
            batch = KVEventBatch(ts=time.time(), events=events)
            self.kv_event_publisher.publish(batch)

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }

        finished_req_ids = self.finished_req_ids_dict
        if finished_req_ids:
            # Include ids of requests that finished since last outputs
            # were sent.
            for client_index, finished_set in finished_req_ids.items():
                # Set finished request set in EngineCoreOutputs for this client.
                if (eco := engine_core_outputs.get(client_index)) is not None:
                    eco.finished_requests = finished_set
                else:
                    engine_core_outputs[client_index] = EngineCoreOutputs(
                        finished_requests=finished_set
                    )
            finished_req_ids.clear()

        if (
            stats := self.make_stats(
                spec_decoding_stats, kv_connector_stats, cudagraph_stats, perf_stats
            )
        ) is not None:
            # Return stats to only one of the front-ends.
            if (eco := next(iter(engine_core_outputs.values()), None)) is None:
                # We must return the stats even if there are no request
                # outputs this step.
                engine_core_outputs[0] = eco = EngineCoreOutputs()
            eco.scheduler_stats = stats

        return engine_core_outputs

    @staticmethod
    def _is_blocked_waiting_status(status: RequestStatus) -> bool:
        return status in (
            RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR,
            RequestStatus.WAITING_FOR_REMOTE_KVS,
            RequestStatus.WAITING_FOR_STREAMING_REQ,
        )

    def _enqueue_waiting_request(self, request: Request) -> None:
        if self._is_blocked_waiting_status(request.status):
            self.skipped_waiting.add_request(request)
        else:
            self.waiting.add_request(request)

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        if self.policy == SchedulingPolicy.FCFS:
            return self.skipped_waiting or self.waiting or None

        # PRIORITY mode: compare queue heads when both queues are non-empty.
        if self.waiting and self.skipped_waiting:
            waiting_req = self.waiting.peek_request()
            skipped_req = self.skipped_waiting.peek_request()
            return self.waiting if waiting_req < skipped_req else self.skipped_waiting

        return self.waiting or self.skipped_waiting or None

    def _handle_stopped_request(self, request: Request) -> bool:
        """Return True if finished (can be False for resumable requests)."""
        if not request.resumable:
            return True

        if request.streaming_queue:
            update = request.streaming_queue.popleft()
            if update is None:
                # Streaming request finished.
                return True
            self._update_request_as_session(request, update)
        else:
            request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
            self.num_waiting_for_streaming_input += 1

        self._enqueue_waiting_request(request)
        return False

    def _get_routed_experts(self, request: Request) -> np.ndarray | None:
        if not self.vllm_config.model_config.enable_return_routed_experts:
            return None

        kv_blocks = self.kv_cache_manager.get_blocks(request.request_id)
        block_ids = kv_blocks.get_block_ids()[self.routed_experts_attn_gid]
        num_tokens = request.num_tokens - 1

        # compute slot mapping using attention group's block_size
        block_ids_array = np.array(block_ids, dtype=np.int32)
        num_blocks = len(block_ids)
        attn_group = self.kv_cache_config.kv_cache_groups[self.routed_experts_attn_gid]
        block_size = attn_group.kv_cache_spec.block_size

        # generate block offsets
        block_offsets = np.arange(0, block_size)

        # compute slot mapping: slot = block_id * block_size + offset
        slot_mapping = (
            block_offsets.reshape((1, block_size))
            + block_ids_array.reshape((num_blocks, 1)) * block_size
        ).flatten()[:num_tokens]

        return self.routed_experts_reader.get_routed_experts(indices=slot_mapping)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        # Append generated tokens and check for stop. Note that if
        # a request is still being prefilled, we expect the model runner
        # to return empty token ids for the request.
        stopped = False
        for num_new, output_token_id in enumerate(new_token_ids, 1):
            request.append_output_token_ids(output_token_id)

            # Check for stop and update request state.
            # This must be called before we make the EngineCoreOutput.
            stopped = check_stop(request, self.max_model_len)
            if stopped:
                del new_token_ids[num_new:]  # Trim new tokens if needed.
                break
        return new_token_ids, stopped

    def _free_encoder_inputs(self, request: Request) -> None:
        cached_encoder_input_ids = self.encoder_cache_manager.get_cached_input_ids(
            request
        )
        # OPTIMIZATION: Avoid list(set) if the set is empty.
        if not cached_encoder_input_ids:
            return

        # Here, we use list(set) to avoid modifying the set while iterating
        # over it.
        for input_id in list(cached_encoder_input_ids):
            mm_feature = request.mm_features[input_id]
            start_pos = mm_feature.mm_position.offset
            num_tokens = mm_feature.mm_position.length
            if self.is_encoder_decoder and request.num_computed_tokens > 0:
                # With Whisper, as soon as we've generated a single token,
                # we know we're done with the encoder input. Cross Attention
                # KVs have been calculated and cached already.
                self.encoder_cache_manager.free_encoder_input(request, input_id)
            elif start_pos + num_tokens <= request.num_computed_tokens:
                # The encoder output is already processed and stored
                # in the decoder's KV cache.
                self.encoder_cache_manager.free_encoder_input(request, input_id)

    def update_draft_token_ids(self, draft_token_ids: DraftTokenIds) -> None:
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            if request.is_prefill_chunk:
                # Ignore draft tokens for prefill chunks.
                if request.spec_token_ids:
                    request.spec_token_ids = []
                continue

            # Add newly generated spec token ids to the request.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)  # type: ignore[union-attr]
            request.spec_token_ids = spec_token_ids

    def update_draft_token_ids_in_output(
        self, draft_token_ids: DraftTokenIds, scheduler_output: SchedulerOutput
    ) -> None:
        num_invalid_spec_tokens: dict[str, int] = {}

        sched_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id, spec_token_ids in zip(
            draft_token_ids.req_ids,
            draft_token_ids.draft_token_ids,
        ):
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # The request may have been finished. Skip.
                continue

            placeholder_spec_tokens = sched_spec_tokens.get(req_id)
            if not placeholder_spec_tokens:
                continue

            orig_num_spec_tokens = len(placeholder_spec_tokens)
            # Trim drafts to scheduled number of spec tokens
            # (needed for chunked prefill case for example).
            del spec_token_ids[orig_num_spec_tokens:]
            # Filter out spec tokens which do not adhere to the grammar.
            if self.structured_output_manager.should_advance(request):
                metadata = request.structured_output_request
                assert metadata is not None and metadata.grammar is not None
                spec_token_ids = metadata.grammar.validate_tokens(spec_token_ids)
            # Pad to original number of spec tokens.
            num_invalid_tokens = orig_num_spec_tokens - len(spec_token_ids)
            if num_invalid_tokens:
                spec_token_ids.extend([-1] * num_invalid_tokens)
                num_invalid_spec_tokens[req_id] = num_invalid_tokens

            sched_spec_tokens[req_id] = spec_token_ids

        scheduler_output.num_invalid_spec_tokens = num_invalid_spec_tokens

    def get_request_counts(self) -> tuple[int, int]:
        """Returns (num_running_reqs, num_waiting_reqs)."""
        return len(self.running), len(self.waiting) + len(self.skipped_waiting)

    def add_request(self, request: Request) -> None:
        existing = self.requests.get(request.request_id)
        if existing is not None:
            update = StreamingUpdate.from_request(request)
            if existing.status != RequestStatus.WAITING_FOR_STREAMING_REQ:
                assert existing.streaming_queue is not None, "duplicate request id"
                # Queue next input chunk (or finished sentinel).
                existing.streaming_queue.append(update)
            elif update is not None:
                # Commence next input chunk.
                self._update_request_as_session(existing, update)
            else:
                # Streaming-input session finished.
                self.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        else:
            if request.resumable:
                request.streaming_queue = deque()
            self._enqueue_waiting_request(request)
            self.requests[request.request_id] = request
            if self.log_stats:
                request.record_event(EngineCoreEventType.QUEUED)

    def finish_requests(
        self, request_ids: str | Iterable[str] | None, finished_status: RequestStatus
    ) -> list[tuple[str, int]]:
        """Handles the finish signal from outside the scheduler.

        For example, the API server can abort a request when the client
        disconnects.

        If request_ids is None, all requests will be finished.

        Returns:
            Tuple of (req_id, client_index) for requests that were aborted. Will not
            include any that were already finished.
        """
        assert RequestStatus.is_finished(finished_status)
        if isinstance(request_ids, str):
            request_ids = (request_ids,)
        elif request_ids is not None:
            request_ids = set(request_ids)
        else:
            request_ids = self.requests.keys()

        running_requests_to_remove = set()
        waiting_requests_to_remove = []
        valid_requests = []

        # First pass: collect requests to remove from queues
        for req_id in request_ids:
            request = self.requests.get(req_id)
            if request is None or request.is_finished():
                # Invalid request ID.
                continue

            valid_requests.append(request)
            if request.status == RequestStatus.RUNNING:
                running_requests_to_remove.add(request)
            else:
                if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                    self.num_waiting_for_streaming_input -= 1
                waiting_requests_to_remove.append(request)

        # Remove all requests from queues at once for better efficiency
        if running_requests_to_remove:
            self.running = remove_all(self.running, running_requests_to_remove)
        if waiting_requests_to_remove:
            self.waiting.remove_requests(waiting_requests_to_remove)
            self.skipped_waiting.remove_requests(waiting_requests_to_remove)

        # Second pass: set status and free requests
        for request in valid_requests:
            delay_free_blocks = False
            if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                delay_free_blocks = (
                    request.request_id not in self.finished_recving_kv_req_ids
                )
                self.finished_recving_kv_req_ids.discard(request.request_id)
                self.failed_recving_kv_req_ids.discard(request.request_id)

            request.status = finished_status
            self._free_request(request, delay_free_blocks=delay_free_blocks)

        return [(r.request_id, r.client_index) for r in valid_requests]

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        assert request.is_finished()

        connector_delay_free_blocks, kv_xfer_params = self._connector_finished(request)
        self.encoder_cache_manager.free(request)
        request_id = request.request_id
        self.finished_req_ids.add(request_id)
        if self.finished_req_ids_dict is not None:
            self.finished_req_ids_dict[request.client_index].add(request_id)

        delay_free_blocks |= connector_delay_free_blocks
        if not delay_free_blocks:
            self._free_blocks(request)

        return kv_xfer_params

    def _free_blocks(self, request: Request):
        assert request.is_finished()
        self.kv_cache_manager.free(request)
        del self.requests[request.request_id]

    @property
    def pause_state(self) -> PauseState:
        return self._pause_state

    def set_pause_state(self, pause_state: PauseState) -> None:
        self._pause_state = pause_state

    def get_num_unfinished_requests(self) -> int:
        if self._pause_state == PauseState.PAUSED_ALL:
            return 0
        if self._pause_state == PauseState.PAUSED_NEW:
            return len(self.running)
        num_waiting = (
            len(self.waiting)
            + len(self.skipped_waiting)
            - self.num_waiting_for_streaming_input
        )
        return num_waiting + len(self.running)

    def has_finished_requests(self) -> bool:
        return len(self.finished_req_ids) > 0

    def reset_prefix_cache(
        self, reset_running_requests: bool = False, reset_connector: bool = False
    ) -> bool:
        """Reset the KV prefix cache.

        If reset_running_requests is True, all the running requests will be
        preempted and moved to the waiting queue.
        Otherwise, this method will only reset the KV prefix cache when there
        is no running requests taking KV cache.
        """
        if reset_running_requests:
            # For logging.
            timestamp = time.monotonic()
            # Invalidate all the current running requests KV's by pushing them to
            # the waiting queue. In this case, we can reduce the ref count of all
            # the kv blocks to 0 and thus we can make sure the reset is successful.
            # Preempt in reverse order so the requests will be added back to the
            # running queue in FIFO order.
            while self.running:
                request = self.running.pop()
                self._preempt_request(request, timestamp)
                # NOTE(zhuohan): For async scheduling, we need to discard the latest
                # output token on the fly to avoid a redundant repetitive output token.
                request.num_output_placeholders = 0
                request.discard_latest_async_tokens = True

            # Clear scheduled request ids cache. Since we are forcing preemption
            # + resumption in the same step, we must act as if these requests were
            # not scheduled in the prior step. They will be flushed from the
            # persistent batch in the model runner.
            self.prev_step_scheduled_req_ids.clear()

        reset_successful = self.kv_cache_manager.reset_prefix_cache()
        if reset_running_requests and not reset_successful:
            raise RuntimeError(
                "Failed to reset KV cache even when all the running requests are "
                "preempted and moved to the waiting queue. This is likely due to "
                "the presence of running requests waiting for remote KV transfer, "
                "which is not supported yet."
            )

        if reset_connector:
            reset_successful = self.reset_connector_cache() and reset_successful

        return reset_successful

    def reset_connector_cache(self) -> bool:
        if self.connector is None:
            logger.warning("reset_connector called but no KV connector is configured.")
            return False

        if self.connector.reset_cache() is False:
            return False

        if self.log_stats:
            assert self.connector_prefix_cache_stats is not None
            self.connector_prefix_cache_stats.reset = True

        return True

    def reset_encoder_cache(self) -> None:
        """Reset the encoder cache to invalidate all cached encoder outputs.

        This should be called when model weights are updated to ensure
        stale vision embeddings are not reused.
        """
        self.encoder_cache_manager.reset()

    def make_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None = None,
        kv_connector_stats: KVConnectorStats | None = None,
        cudagraph_stats: CUDAGraphStat | None = None,
        perf_stats: PerfStats | None = None,
    ) -> SchedulerStats | None:
        if not self.log_stats:
            return None
        prefix_cache_stats = self.kv_cache_manager.make_prefix_cache_stats()
        assert prefix_cache_stats is not None
        connector_prefix_cache_stats: PrefixCacheStats | None = None
        if self.connector_prefix_cache_stats is not None:
            connector_prefix_cache_stats = self.connector_prefix_cache_stats
            self.connector_prefix_cache_stats = PrefixCacheStats()
        eviction_events = (
            self.kv_metrics_collector.drain_events()
            if self.kv_metrics_collector is not None
            else []
        )
        spec_stats = spec_decoding_stats
        connector_stats_payload = (
            kv_connector_stats.data if kv_connector_stats else None
        )
        return SchedulerStats(
            num_running_reqs=len(self.running),
            num_waiting_reqs=len(self.waiting) + len(self.skipped_waiting),
            kv_cache_usage=self.kv_cache_manager.usage,
            encoder_cache_usage=self._get_encoder_cache_usage(),
            prefix_cache_stats=prefix_cache_stats,
            connector_prefix_cache_stats=connector_prefix_cache_stats,
            kv_cache_eviction_events=eviction_events,
            spec_decoding_stats=spec_stats,
            kv_connector_stats=connector_stats_payload,
            cudagraph_stats=cudagraph_stats,
            perf_stats=perf_stats,
        )

    def _get_encoder_cache_usage(self) -> float:
        """Get encoder cache usage as a fraction (0.0 to 1.0)."""
        ecm = self.encoder_cache_manager
        if ecm.cache_size == 0:
            return 0.0
        used_slots = ecm.cache_size - ecm.num_free_slots
        return used_slots / ecm.cache_size

    def make_spec_decoding_stats(
        self,
        spec_decoding_stats: SpecDecodingStats | None,
        num_draft_tokens: int,
        num_accepted_tokens: int,
        num_invalid_spec_tokens: dict[str, int] | None,
        request_id: str,
    ) -> SpecDecodingStats | None:
        if not self.log_stats or not num_draft_tokens:
            return None
        if spec_decoding_stats is None:
            spec_decoding_stats = SpecDecodingStats.new(self.num_spec_tokens)
        if num_invalid_spec_tokens:
            num_draft_tokens -= num_invalid_spec_tokens.get(request_id, 0)
        spec_decoding_stats.observe_draft(
            num_draft_tokens=num_draft_tokens, num_accepted_tokens=num_accepted_tokens
        )
        return spec_decoding_stats

    def shutdown(self) -> None:
        if self.kv_event_publisher:
            self.kv_event_publisher.shutdown()
        if self.connector is not None:
            self.connector.shutdown()

    ########################################################################
    # KV Connector Related Methods
    ########################################################################

    def get_kv_connector(self) -> KVConnectorBase_V1 | None:
        return self.connector

    def _connector_finished(
        self, request: Request
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Invoke the KV connector request_finished() method if applicable.

        Returns optional kv transfer parameters to be included with the
        request outputs.
        """
        if self.connector is None:
            return False, None

        # Free any out-of-window prefix blocks before we hand the block table to
        # the connector.
        self.kv_cache_manager.remove_skipped_blocks(
            request_id=request.request_id,
            total_computed_tokens=request.num_tokens,
        )

        block_ids = self.kv_cache_manager.get_block_ids(request.request_id)

        if not isinstance(self.connector, SupportsHMA):
            # NOTE(Kuntai): We should deprecate this code path after we enforce
            # all connectors to support HMA.
            # Hybrid memory allocator should be already turned off for this
            # code path, but let's double-check here.
            assert len(self.kv_cache_config.kv_cache_groups) == 1
            return self.connector.request_finished(request, block_ids[0])

        return self.connector.request_finished_all_groups(request, block_ids)

    def _update_waiting_for_remote_kv(self, request: Request) -> None:
        """
        KV Connector: update request state after async recv is finished.

        When the kv transfer is ready, we cache the blocks
        and the request state will be moved back to WAITING from
        WAITING_FOR_REMOTE_KV.
        """
        assert self.connector is not None

        if request.request_id in self.failed_recving_kv_req_ids:
            # Request had KV load failures; num_computed_tokens was already
            # updated in _update_requests_with_invalid_blocks
            if request.num_computed_tokens:
                # Cache any valid computed tokens.
                self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            else:
                # No valid computed tokens, release allocated blocks.
                # There may be a local cache hit on retry.
                self.kv_cache_manager.free(request)

            self.failed_recving_kv_req_ids.remove(request.request_id)
        else:
            # Now that the blocks are ready, actually cache them.
            # This will cache the blocks iff caching is enabled.
            self.kv_cache_manager.cache_blocks(request, request.num_computed_tokens)
            self._commit_reflex_int4_landing_contract(request)
            self._commit_reflex_remote_chunk(request)

            # on a full prompt hit, we need to re-compute the last token
            # in order to be able to sample the next token
            if request.num_computed_tokens == request.num_tokens:
                request.num_computed_tokens = request.num_tokens - 1

            # Count the number of prefix cached tokens.
            if request.num_cached_tokens < 0:
                request.num_cached_tokens = request.num_computed_tokens

        remote_transfer_start = getattr(
            request,
            "reflex_remote_transfer_start_time",
            None,
        )
        if remote_transfer_start is not None:
            logger.info(
                "ReFlexKV trace stage_profile request=%s phase=remote_transfer "
                "ms=%.3f source=scheduler.",
                request.request_id,
                (time.perf_counter() - float(remote_transfer_start)) * 1000.0,
            )
            try:
                delattr(request, "reflex_remote_transfer_start_time")
            except AttributeError:
                pass

        self.finished_recving_kv_req_ids.remove(request.request_id)

    def _try_promote_blocked_waiting_request(self, request: Request) -> bool:
        """
        Try to promote a blocked waiting request back to schedulable states.
        """
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            # finished_recving_kv_req_ids is populated during
            # update_from_output(), based on worker-side connector signals
            # in KVConnectorOutput.finished_recving
            if request.request_id not in self.finished_recving_kv_req_ids:
                return False
            self._update_waiting_for_remote_kv(request)
            if request.num_preemptions:
                request.status = RequestStatus.PREEMPTED
            else:
                request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR:
            structured_output_req = request.structured_output_request
            if not (structured_output_req and structured_output_req.grammar):
                return False
            request.status = RequestStatus.WAITING
            return True

        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            assert not request.streaming_queue
            return False

        raise AssertionError(
            "Unexpected blocked waiting status in promotion: "
            f"{request.status.name} for request {request.request_id}"
        )

    def _update_from_kv_xfer_finished(
        self,
        kv_connector_output: KVConnectorOutput,
        *,
        update_connector: bool = True,
    ):
        """
        KV Connector: update the scheduler state based on the output.

        The Worker side connectors add finished_recving and
        finished_sending reqs to the output.
        * if finished_sending: free the blocks
        # if finished_recving: add to state so we can
            schedule the request during the next step.
        """

        if update_connector and self.connector is not None:
            self.connector.update_connector_output(kv_connector_output)

        worker_meta = kv_connector_output.kv_connector_worker_meta
        materialized_landing_req_ids = getattr(
            worker_meta,
            "reflex_int4_materialized_landing_req_ids",
            None,
        )
        if materialized_landing_req_ids:
            self.reflex_int4_materialized_landing_req_ids.update(
                str(req_id) for req_id in materialized_landing_req_ids
            )

        # KV Connector:: update recv and send status from last step.
        for req_id in kv_connector_output.finished_recving or ():
            logger.debug("Finished recving KV transfer for request %s", req_id)
            assert req_id in self.requests
            req = self.requests[req_id]
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                self.finished_recving_kv_req_ids.add(req_id)
            else:
                assert RequestStatus.is_finished(req.status)
                self._free_blocks(self.requests[req_id])
        for req_id in kv_connector_output.finished_sending or ():
            logger.debug("Finished sending KV transfer for request %s", req_id)
            assert req_id in self.requests
            self._free_blocks(self.requests[req_id])

    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        """
        Identify and update requests affected by invalid KV cache blocks.

        This method scans the given requests, detects those with invalid blocks
        and adjusts their `num_computed_tokens` to the longest valid prefix.
        For observability, it also accumulates the total number of tokens that
        will need to be recomputed across all affected requests.

        Args:
            requests: The set of requests to scan for invalid blocks.
            invalid_block_ids: IDs of invalid blocks.
            num_scheduled_tokens: req_id -> number of scheduled tokens.
            evict_blocks: Whether to collect blocks for eviction (False for
                async requests which aren't cached yet).

        Returns:
            tuple:
                - affected_req_ids (set[str]): IDs of requests impacted by
                invalid blocks.
                - total_affected_tokens (int): Total number of tokens that must
                be recomputed across all affected requests.
                - blocks_to_evict (set[int]): Block IDs to evict from cache,
                including invalid blocks and downstream dependent blocks.
        """
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        # If a block is invalid and shared by multiple requests in the batch,
        # these requests must be rescheduled, but only the first will recompute
        # it. This set tracks blocks already marked for recomputation.
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    # This invalid block is shared with a previous request
                    # and was already marked for recomputation.
                    # This means this request can still consider this block
                    # as computed when rescheduled.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    # This request has already marked an invalid block for
                    # recomputation and updated its num_computed_tokens.
                    continue

                marked_invalid_block = True
                # Truncate the computed tokens at the first failed block
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens
                request.num_external_computed_tokens -= num_affected_tokens
                # collect invalid block and all downstream dependent blocks
                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    # All invalid blocks of this request are shared with
                    # previous requests and will be recomputed by them.
                    # Revert to considering only cached tokens as computed.
                    # Currently this only applies to sync loading; Async
                    # loading does not yet support block sharing
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict

    def _handle_invalid_blocks(
        self, invalid_block_ids: set[int], num_scheduled_tokens: dict[str, int]
    ) -> set[str]:
        """
        Handle requests affected by invalid KV cache blocks.

        Returns:
            Set of affected request IDs to skip in update_from_output main loop.
        """
        should_fail = not self.recompute_kv_load_failures

        # handle async KV loads (not cached yet, evict_blocks=False)
        async_load_reqs = (
            req
            for req in self.skipped_waiting
            if req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        )
        async_failed_req_ids, num_failed_tokens, _ = (
            self._update_requests_with_invalid_blocks(
                async_load_reqs,
                invalid_block_ids,
                num_scheduled_tokens,
                evict_blocks=False,
            )
        )

        total_failed_requests = len(async_failed_req_ids)
        total_failed_tokens = num_failed_tokens

        # handle sync loads (may be cached, collect blocks for eviction)
        sync_failed_req_ids, num_failed_tokens, sync_blocks_to_evict = (
            self._update_requests_with_invalid_blocks(
                self.running, invalid_block_ids, num_scheduled_tokens, evict_blocks=True
            )
        )

        total_failed_requests += len(sync_failed_req_ids)
        total_failed_tokens += num_failed_tokens

        if not total_failed_requests:
            return set()

        # evict invalid blocks and downstream dependent blocks from cache
        # only when not using recompute policy (where blocks will be recomputed
        # and reused by other requests sharing them)
        if sync_blocks_to_evict and not self.recompute_kv_load_failures:
            self.kv_cache_manager.evict_blocks(sync_blocks_to_evict)

        if should_fail:
            all_failed_req_ids = async_failed_req_ids | sync_failed_req_ids
            logger.error(
                "Failing %d request(s) due to KV load failure "
                "(failure_policy=fail, %d tokens affected). Request IDs: %s",
                total_failed_requests,
                total_failed_tokens,
                all_failed_req_ids,
            )
            return all_failed_req_ids

        logger.warning(
            "Recovered from KV load failure: "
            "%d request(s) rescheduled (%d tokens affected).",
            total_failed_requests,
            total_failed_tokens,
        )

        # Mark async requests with KV load failures for retry once loading completes
        self.failed_recving_kv_req_ids |= async_failed_req_ids
        # Return sync affected IDs to skip in update_from_output
        return sync_failed_req_ids
