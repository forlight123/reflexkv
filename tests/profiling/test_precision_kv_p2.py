from types import SimpleNamespace

import torch

from vllm.v1.attention.ops.reflex_quantizer import RiskAwareInt4Quantizer
from vllm.v1.core.precision_kv.contracts import (
    PrefixPrecisionContractManager,
)
from vllm.v1.core.precision_kv.risk import (
    PrefillRiskEstimator,
    derive_compressible_pages_from_risks,
    select_bf16_shadow_pages,
    synthesize_remote_chunk_landing_pages,
)
from vllm.v1.core.precision_kv.types import PrecisionState


def test_prefill_risk_estimator_builds_semantic_page_summaries():
    estimator = PrefillRiskEstimator()
    q_tail_anchor = torch.tensor([1.0, 0.0])
    page_key_anchors = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0]),
    ]

    summaries = estimator.estimate_from_anchors(
        request_id="req-risk",
        q_tail_anchor=q_tail_anchor,
        page_key_anchors=page_key_anchors,
        block_size=16,
    )

    assert [summary.page_idx for summary in summaries] == [0, 1, 2]
    assert summaries[0].risk_score > summaries[1].risk_score
    assert summaries[1].risk_score > summaries[2].risk_score
    assert summaries[2].semantic_hash != summaries[0].semantic_hash
    assert summaries[2].compressible is True


def test_risk_helpers_select_low_risk_landing_and_high_risk_shadow_pages():
    risks = [0.9, 0.1, 0.2, 0.8]

    assert derive_compressible_pages_from_risks(risks, fraction=0.5) == {1, 2}
    assert select_bf16_shadow_pages(risks, max_pages=2) == {0, 3}


def test_synthetic_direct_landing_pages_are_chunk_local_and_short_prompt_safe():
    pages = synthesize_remote_chunk_landing_pages(
        page_start=64,
        page_end=96,
        page_count=512,
        keep_initial_pages=4,
        keep_recent_pages=4,
        protected_prompt_pages=0,
        max_int4_fraction=1.0,
        short_prefill_pages=64,
    )

    assert pages == tuple(range(64, 96))

    short_pages = synthesize_remote_chunk_landing_pages(
        page_start=0,
        page_end=16,
        page_count=16,
        keep_initial_pages=0,
        keep_recent_pages=0,
        protected_prompt_pages=0,
        max_int4_fraction=1.0,
        short_prefill_pages=64,
    )

    assert short_pages == ()


def test_risk_aware_int4_quantizer_uses_residual_compensation_for_high_risk_pages():
    values = torch.tensor(
        [
            [0.0, 0.2, 0.4, 0.6, 3.7, -3.1, 0.9, -0.8],
            [1.2, -1.4, 1.6, -1.8, 0.05, -0.05, 0.1, -0.1],
        ],
        dtype=torch.float32,
    )
    base = RiskAwareInt4Quantizer(group_size=4, residual_topk=0)
    compensated = RiskAwareInt4Quantizer(group_size=4, residual_topk=2)

    base_capsule = base.quantize_tensor(values, risk_score=0.1)
    compensated_capsule = compensated.quantize_tensor(values, risk_score=0.9)

    base_error = torch.mean((base.dequantize_tensor(base_capsule) - values) ** 2)
    compensated_error = torch.mean(
        (compensated.dequantize_tensor(compensated_capsule) - values) ** 2
    )

    assert compensated_capsule.residual_indices.numel() > 0
    assert compensated_error < base_error


def test_prefix_precision_contract_manager_tracks_copy_on_demote_versions():
    manager = PrefixPrecisionContractManager()
    manager.register_shared_prefix(
        prefix_id="prefix-a",
        owner_request_id="req-a",
        page_indices=[0, 1, 2],
        precision=PrecisionState.BF16,
    )

    assert manager.requires_copy_on_demote(
        prefix_id="prefix-a",
        request_id="req-b",
        page_idx=1,
    )

    new_version = manager.copy_on_demote(
        prefix_id="prefix-a",
        request_id="req-b",
        page_indices=[1],
        target_precision=PrecisionState.INT4,
    )

    assert new_version.version_id == 2
    assert new_version.owner_request_id == "req-b"
    assert new_version.precision == PrecisionState.INT4
    assert new_version.page_indices == (1,)
    assert manager.active_version("prefix-a", "req-a").precision == (
        PrecisionState.BF16
    )
    assert manager.active_version("prefix-a", "req-b").precision == (
        PrecisionState.INT4
    )
