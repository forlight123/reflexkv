# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision-elastic KV runtime helpers."""

from vllm.v1.core.precision_kv.accounting import (
    ReflexBlockTableStats,
    summarize_reflex_block_table,
)
from vllm.v1.core.precision_kv.controller import (
    PrecisionAdmissionController,
    PrecisionAdmissionDecision,
    PrecisionAdmissionPlan,
    PrecisionAdmissionState,
)
from vllm.v1.core.precision_kv.contracts import (
    PrefixPrecisionContract,
    PrefixPrecisionContractManager,
    PrefixPrecisionVersion,
    REFLEX_INT4_LANDING_CONTRACT_KEYS,
    clear_reflex_int4_landing_contract,
    has_reflex_int4_landing_contract,
)
from vllm.v1.core.precision_kv.demotion_planner import (
    DistanceDemotionPlanner,
    ReflexCandidateBreakdown,
    ReflexDemotionPlan,
    RequestBudgetCandidate,
    RequestPrecisionBudget,
    allocate_request_release_budgets,
)
from vllm.v1.core.precision_kv.frontier import (
    AdmissionTicket,
    FeasibleFrontierCache,
    FeasibleFrontierSummary,
    PageCompressibilityLevel,
    RejectionReason,
)
from vllm.v1.core.precision_kv.landing import (
    PrecisionLandingDecision,
    PrecisionLandingPlanner,
    PrecisionLandingState,
)
from vllm.v1.core.precision_kv.risk import (
    PageRiskSummary,
    PrefillRiskEstimator,
    derive_compressible_pages_from_risks,
    select_bf16_shadow_pages,
    synthesize_remote_chunk_landing_pages,
)
from vllm.v1.core.precision_kv.run_optimizer import (
    DualPriceState,
    DualRunOptimizer,
    RunCandidate,
)
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
)

__all__ = [
    "AdmissionTicket",
    "DistanceDemotionPlanner",
    "DualPriceState",
    "DualRunOptimizer",
    "FeasibleFrontierCache",
    "FeasibleFrontierSummary",
    "Int4BlockPool",
    "KVPageLifecycle",
    "KVPageRuntimeDescriptor",
    "MemoryTier",
    "PageCompressibilityLevel",
    "PageRiskSummary",
    "PrecisionState",
    "PrecisionAdmissionController",
    "PrecisionAdmissionDecision",
    "PrecisionAdmissionPlan",
    "PrecisionAdmissionState",
    "PrecisionLandingDecision",
    "PrecisionLandingPlanner",
    "PrecisionLandingState",
    "PrefixPrecisionContract",
    "PrefixPrecisionContractManager",
    "PrefixPrecisionVersion",
    "RecoveryClass",
    "ReflexBlockTableStats",
    "ReflexCandidateBreakdown",
    "ReflexDemotion",
    "ReflexDemotionPlan",
    "ReflexPageMeta",
    "ReflexRecovery",
    "ReflexRecoveryArtifact",
    "RequestBudgetCandidate",
    "RequestPrecisionBudget",
    "RunCandidate",
    "RejectionReason",
    "REFLEX_INT4_LANDING_CONTRACT_KEYS",
    "allocate_request_release_budgets",
    "clear_reflex_int4_landing_contract",
    "derive_compressible_pages_from_risks",
    "has_reflex_int4_landing_contract",
    "select_bf16_shadow_pages",
    "summarize_reflex_block_table",
    "synthesize_remote_chunk_landing_pages",
    "PrefillRiskEstimator",
]
