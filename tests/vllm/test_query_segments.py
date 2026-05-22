import json
import math
from pathlib import Path

import pytest
import torch

import vllm.semantiq.query_segments as query_segments_module
from vllm.semantiq.query_segments import (
    ENV_ENABLE,
    ENV_FAKE_QUANT_ENABLE,
    ENV_FAKE_QUANT_METHOD,
    ENV_FAKE_QUANT_SEED,
    ENV_FORCE_BIT_WIDTH,
    ENV_FORCE_HEAD_ID,
    ENV_FORCE_LAYER_NAME,
    ENV_FORCE_RANK,
    ENV_FORCE_SIDE,
    ENV_OUTPUT,
    ENV_PAGE_SIZE,
    ENV_PRIOR_PATH,
    ENV_THRESHOLD,
    _build_dump_path,
    capture_query_segments_from_batch,
    configure_query_segment_runtime,
    QuerySegmentConfig,
    _HeadState,
    QuerySegmentRuntime,
    _LayerState,
    _PageState,
    _Segment,
    SemantiqWorkerExtension,
    _finalize_segment_stability,
    get_query_segment_runtime,
    reset_query_segment_runtime,
    _select_random_fake_quant_bit_width,
    _select_fake_quant_bit_width,
    _select_hybrid_key_bit_width,
    _update_recent_segment_attention_scores,
)
from vllm.semantiq.prior import SemantiqPrior, resolve_k_base_bits, resolve_semantiq_rank


def test_select_fake_quant_bit_width_uses_fixed_thresholds():
    assert _select_fake_quant_bit_width(attention_score=0.70, stability=0.90) == 8
    assert _select_fake_quant_bit_width(attention_score=0.40, stability=0.70) == 4
    assert _select_fake_quant_bit_width(attention_score=0.20, stability=0.90) == 2
    assert _select_fake_quant_bit_width(attention_score=0.70, stability=0.20) == 2


def test_select_random_fake_quant_bit_width_uses_seeded_selection_key():
    selection_key = "req-a:model.layers.0.attn:0:0"

    assert _select_random_fake_quant_bit_width(fake_quant_seed=0, selection_key=selection_key) == 2
    assert _select_random_fake_quant_bit_width(fake_quant_seed=9, selection_key=selection_key) == 4
    assert _select_random_fake_quant_bit_width(fake_quant_seed=3, selection_key=selection_key) == 8


def test_query_segment_config_from_env_reads_fake_quant_method(monkeypatch):
    monkeypatch.setenv(ENV_ENABLE, "1")
    monkeypatch.setenv(ENV_PAGE_SIZE, "16")
    monkeypatch.setenv(ENV_THRESHOLD, "0.8")
    monkeypatch.setenv(ENV_FAKE_QUANT_ENABLE, "1")
    monkeypatch.setenv(ENV_FAKE_QUANT_SEED, "7")
    monkeypatch.setenv(ENV_FAKE_QUANT_METHOD, "0")

    config = QuerySegmentConfig.from_env()

    assert config.fake_quant_enabled is True
    assert config.fake_quant_seed == 7
    assert config.fake_quant_method == 0


def test_configure_query_segment_runtime_updates_worker_local_state(monkeypatch):
    monkeypatch.delenv(ENV_ENABLE, raising=False)
    monkeypatch.delenv(ENV_OUTPUT, raising=False)
    monkeypatch.delenv(ENV_FORCE_LAYER_NAME, raising=False)
    monkeypatch.delenv(ENV_FORCE_RANK, raising=False)
    monkeypatch.delenv(ENV_FORCE_HEAD_ID, raising=False)
    monkeypatch.delenv(ENV_FORCE_SIDE, raising=False)
    monkeypatch.delenv(ENV_FORCE_BIT_WIDTH, raising=False)

    config = configure_query_segment_runtime(
        {
            "enabled": True,
            "page_size": 32,
            "similarity_threshold": 0.75,
            "fake_quant_enabled": True,
            "force_layer_name": "model.layers.3.attn",
            "force_head_id": 2,
            "force_side": "key",
            "force_bit_width": 4,
        }
    )

    assert config.enabled is True
    assert config.page_size == 32
    assert config.similarity_threshold == pytest.approx(0.75)
    assert config.fake_quant_enabled is True
    assert config.force_layer_name == "model.layers.3.attn"
    assert config.force_head_id == 2
    assert config.force_side == "key"
    assert config.force_bit_width == 4
    assert get_query_segment_runtime().snapshot()["config"]["force_head_id"] == 2

    extension = SemantiqWorkerExtension()
    snapshot = extension.semantiq_snapshot_query_segment_runtime()

    assert snapshot["config"]["force_layer_name"] == "model.layers.3.attn"

    configure_query_segment_runtime(
        {
            "enabled": False,
            "output_path": None,
            "fake_quant_enabled": False,
            "force_layer_name": None,
            "force_rank": None,
            "force_head_id": None,
            "force_side": None,
            "force_bit_width": None,
        }
    )


def test_finalize_segment_stability_treats_single_page_segment_as_stable():
    assert _finalize_segment_stability(total_similarity=0.0, count=0) == 1.0
    assert _finalize_segment_stability(total_similarity=1.7, count=2) == 0.85


def _build_kv_cache(
    keys: torch.Tensor,
    values: torch.Tensor,
    *,
    block_size: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, _, head_dim = keys.shape
    num_blocks = (num_tokens + block_size - 1) // block_size
    kv_cache = torch.zeros(
        (2, num_blocks, block_size, num_kv_heads, head_dim),
        dtype=torch.float32,
    )
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    block_ids = slot_mapping // block_size
    block_offsets = slot_mapping % block_size
    kv_cache[0, block_ids, block_offsets] = keys
    kv_cache[1, block_ids, block_offsets] = values
    return kv_cache, slot_mapping


def _write_prior(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_segment(
    *,
    segment_id: int = 0,
    start_page: int = 0,
    end_page: int = 0,
    attention_score: float = 0.9,
    attention_shift: float = 0.0,
    stability: float = 1.0,
) -> _Segment:
    return _Segment(
        segment_id=segment_id,
        start_page=start_page,
        end_page=end_page,
        split_from_previous_similarity=None,
        attention_score=attention_score,
        attention_shift=attention_shift,
        stability=stability,
    )


def _build_runtime(
    *,
    prior_path: Path | None = None,
    fake_quant_method: int = 1,
    fake_quant_seed: int = 0,
    force_layer_name: str | None = None,
    force_head_id: int | None = None,
    force_side: str | None = None,
    force_bit_width: int | None = None,
    page_size: int = 16,
    similarity_threshold: float = 0.8,
) -> QuerySegmentRuntime:
    return QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=page_size,
            similarity_threshold=similarity_threshold,
            prior_path=None if prior_path is None else str(prior_path),
            fake_quant_enabled=True,
            fake_quant_seed=fake_quant_seed,
            fake_quant_method=fake_quant_method,
            force_layer_name=force_layer_name,
            force_head_id=force_head_id,
            force_side=force_side,
            force_bit_width=force_bit_width,
        )
    )


def _quantize_once(
    runtime: QuerySegmentRuntime,
    segment: _Segment,
    *,
    latest_query: torch.Tensor | None = None,
    page_slots: torch.Tensor | None = None,
) -> tuple[int | None, int | None, tuple[int, ...] | None]:
    latest_query = (
        latest_query
        if latest_query is not None
        else torch.tensor([0.9, 0.8, 0.1, 0.0, 0.5, 0.4], dtype=torch.float32)
    )
    page_slots = (
        page_slots
        if page_slots is not None
        else torch.tensor([0], dtype=torch.int64)
    )
    kv_cache = torch.tensor(
        [
            [[[[-1.0, 0.5, 1.5, -0.5, 0.25, 0.75]]]],
            [[[[-1.0, 0.5, 1.5, -0.5, 0.25, 0.75]]]],
        ],
        dtype=torch.float32,
    )
    layer_state = _LayerState(
        num_kv_heads=1,
        head_dim=latest_query.shape[-1],
        partial_page_sum=torch.zeros((1, latest_query.shape[-1]), dtype=torch.float32),
        head_states=[_HeadState(segments=[segment], pending_quantized_segment_ids=[0])],
        pages=[
            _PageState(
                slots=page_slots,
                page_idx=0,
                start_token=0,
                end_token=int(page_slots.numel()) - 1,
                prefill_tokens=int(page_slots.numel()),
                decode_tokens=0,
            )
        ],
        page_index_to_state={
            0: _PageState(
                slots=page_slots,
                page_idx=0,
                start_token=0,
                end_token=int(page_slots.numel()) - 1,
                prefill_tokens=int(page_slots.numel()),
                decode_tokens=0,
            )
        },
    )
    runtime._maybe_quantize_pending_segments(
        request_id="req-a",
        layer_name="model.layers.0.attn",
        layer_state=layer_state,
        head_id=0,
        kv_cache=kv_cache,
        latest_query=latest_query,
    )
    return (
        segment.key_bit_width,
        segment.value_bit_width,
        segment.query_topk_channels,
    )


def test_query_segment_config_rejects_force_side_value_immediately(monkeypatch):
    monkeypatch.setenv(ENV_ENABLE, "1")
    monkeypatch.setenv(ENV_PAGE_SIZE, "16")
    monkeypatch.setenv(ENV_THRESHOLD, "0.8")
    monkeypatch.setenv(ENV_FAKE_QUANT_ENABLE, "1")
    monkeypatch.setenv(ENV_FORCE_LAYER_NAME, "model.layers.0.attn")
    monkeypatch.setenv(ENV_FORCE_HEAD_ID, "0")
    monkeypatch.setenv(ENV_FORCE_SIDE, "value")
    monkeypatch.setenv(ENV_FORCE_BIT_WIDTH, "4")

    with pytest.raises(ValueError, match="unsupported"):
        QuerySegmentConfig.from_env()


def test_query_segment_config_rejects_force_side_value_on_direct_construction():
    with pytest.raises(ValueError, match="unsupported"):
        QuerySegmentConfig(
            enabled=True,
            page_size=16,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            force_layer_name="model.layers.0.attn",
            force_head_id=0,
            force_side="value",
            force_bit_width=4,
        )


def test_runtime_uses_default_k_base_bits_when_prior_path_is_absent():
    runtime = _build_runtime(fake_quant_method=1)
    assert runtime._resolve_prior_k_base_bits(
        layer_name="model.layers.0.attn",
        head_id=0,
        num_kv_heads=1,
    ) == 4


def test_runtime_rejects_invalid_prior_path(tmp_path):
    missing = tmp_path / "missing-prior.json"

    with pytest.raises(FileNotFoundError):
        _build_runtime(prior_path=missing)


def test_runtime_rejects_missing_k_base_bits(tmp_path):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "kv_head",
                "page_size": 16,
                "default_k_base_bits": 4,
            }
        },
    )

    with pytest.raises(ValueError, match="k_base_bits"):
        _build_runtime(prior_path=prior_path)


def test_runtime_rejects_invalid_bit_values(tmp_path):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "kv_head",
                "page_size": 16,
                "default_k_base_bits": 4,
            },
            "k_base_bits": {"model.layers.0.attn": {"0": [3]}},
        },
    )

    with pytest.raises(ValueError, match="must be one of"):
        _build_runtime(prior_path=prior_path)


def test_runtime_rejects_malformed_rank_maps(tmp_path):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "kv_head",
                "page_size": 16,
                "default_k_base_bits": 4,
            },
            "k_base_bits": {"model.layers.0.attn": [4]},
        },
    )

    with pytest.raises(ValueError, match="rank map"):
        _build_runtime(prior_path=prior_path)


def test_runtime_rejects_granularity_mismatch(tmp_path):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "token",
                "page_size": 16,
                "default_k_base_bits": 4,
            },
            "k_base_bits": {"model.layers.0.attn": {"0": [4]}},
        },
    )

    with pytest.raises(ValueError, match="granularity"):
        _build_runtime(prior_path=prior_path)


def test_resolve_semantiq_rank_obeys_precedence(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "3")

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 7)

    assert resolve_semantiq_rank() == "7"


def test_runtime_uses_resolve_semantiq_rank_for_snapshot_rank(monkeypatch):
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(torch.distributed, "is_available", lambda: False)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    import vllm.semantiq.query_segments as query_segments

    monkeypatch.setattr(query_segments, "resolve_semantiq_rank", lambda: "7")

    runtime = _build_runtime()

    assert runtime.snapshot()["rank"] == "7"


def test_runtime_rejects_page_size_mismatch(tmp_path):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "kv_head",
                "page_size": 32,
                "default_k_base_bits": 4,
            },
            "k_base_bits": {"model.layers.0.attn": {"0": [4]}},
        },
    )

    with pytest.raises(ValueError, match="page_size"):
        _build_runtime(prior_path=prior_path, page_size=16)


def test_runtime_rejects_rank_length_mismatch(tmp_path):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "kv_head",
                "page_size": 16,
                "default_k_base_bits": 4,
            },
            "k_base_bits": {
                "model.layers.0.attn": {"0": [4, 4]},
            },
        },
    )

    runtime = _build_runtime(prior_path=prior_path)
    segment = _make_segment()

    with pytest.raises(ValueError, match="head count"):
        _quantize_once(runtime, segment)


def test_runtime_quant_method_zero_keeps_k_randomization_but_sets_v_to_two(tmp_path):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "kv_head",
                "page_size": 16,
                "default_k_base_bits": 4,
            },
            "k_base_bits": {"model.layers.0.attn": {"0": [4]}},
        },
    )

    runtime = _build_runtime(prior_path=prior_path, fake_quant_method=0, fake_quant_seed=3)
    segment = _make_segment(attention_score=0.9, stability=1.0)

    key_bits, value_bits, _ = _quantize_once(runtime, segment)

    assert key_bits == 8
    assert value_bits == 2


def test_select_hybrid_key_bit_width_promotes_high_shift_segment_from_base_four():
    assert (
        _select_hybrid_key_bit_width(
            4,
            attention_score=0.45,
            attention_shift=0.95,
            stability=0.10,
        )
        == 8
    )
    assert (
        _select_hybrid_key_bit_width(
            4,
            attention_score=0.45,
            attention_shift=0.05,
            stability=0.10,
        )
        == 4
    )


def test_runtime_keeps_newly_closed_decode_segment_at_four_bits():
    runtime = _build_runtime(fake_quant_method=1)
    segment = _make_segment(
        attention_score=0.05,
        attention_shift=0.0,
        stability=-1.0,
    )
    segment.decode_page_count = 1

    key_bits, value_bits, _ = _quantize_once(runtime, segment)

    assert key_bits == 4
    assert value_bits == 2


def test_runtime_allows_prefill_only_segment_to_demote_to_two_bits():
    runtime = _build_runtime(fake_quant_method=1)
    segment = _make_segment(
        attention_score=0.05,
        attention_shift=0.0,
        stability=-1.0,
    )
    segment.prefill_page_count = 1

    key_bits, value_bits, _ = _quantize_once(runtime, segment)

    assert key_bits == 2
    assert value_bits == 2


def test_runtime_force_side_key_overrides_computed_k(monkeypatch):
    monkeypatch.setenv(ENV_ENABLE, "1")
    monkeypatch.setenv(ENV_PAGE_SIZE, "16")
    monkeypatch.setenv(ENV_THRESHOLD, "0.8")
    monkeypatch.setenv(ENV_FAKE_QUANT_ENABLE, "1")
    monkeypatch.setenv(ENV_FORCE_LAYER_NAME, "model.layers.0.attn")
    monkeypatch.setenv(ENV_FORCE_HEAD_ID, "0")
    monkeypatch.setenv(ENV_FORCE_SIDE, "key")
    monkeypatch.setenv(ENV_FORCE_BIT_WIDTH, "8")

    config = QuerySegmentConfig.from_env()

    assert config.force_side == "key"


def test_gqa_projection_averages_grouped_query_heads_before_topk_selection():
    runtime = _build_runtime()
    segment = _make_segment()
    latest_query = torch.tensor([1.0, 1.0, 0.0, 0.0, 9.0, 8.0], dtype=torch.float32)

    key_bits, value_bits, query_topk_channels = _quantize_once(
        runtime,
        segment,
        latest_query=latest_query,
    )

    assert key_bits is not None
    assert value_bits is not None
    assert query_topk_channels == (0, 1, 4, 5)


def test_resolve_k_base_bits_falls_back_to_default_base_bits():
    prior = SemantiqPrior(
        k_base_bits={"model.layers.0.attn": {"0": (2,)}},
        default_k_base_bits=4,
        page_size=16,
        granularity="kv_head",
    )

    assert resolve_k_base_bits(prior, "missing", 0) == 4
    assert resolve_k_base_bits(prior, "model.layers.0.attn", 1) == 4
    assert resolve_k_base_bits(prior, "model.layers.0.attn", 0, rank="1") == 4


def test_runtime_falls_back_to_four_for_missing_prior_entries(tmp_path, monkeypatch):
    prior_path = tmp_path / "prior.json"
    _write_prior(
        prior_path,
        {
            "meta": {
                "granularity": "kv_head",
                "page_size": 16,
                "default_k_base_bits": 8,
            },
            "k_base_bits": {
                "model.layers.0.attn": {"0": [8]},
            },
        },
    )
    monkeypatch.setenv("RANK", "0")
    runtime = _build_runtime(prior_path=prior_path)

    assert (
        runtime._resolve_prior_k_base_bits(
            layer_name="missing.layer",
            head_id=0,
            num_kv_heads=1,
        )
        == 4
    )
    assert (
        runtime._resolve_prior_k_base_bits(
            layer_name="model.layers.0.attn",
            head_id=1,
            num_kv_heads=1,
        )
        == 4
    )

    monkeypatch.setenv("RANK", "1")
    assert (
        runtime._resolve_prior_k_base_bits(
            layer_name="model.layers.0.attn",
            head_id=0,
            num_kv_heads=1,
        )
        == 4
    )


def _read_kv_cache(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    kv_head: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_size = kv_cache.shape[2]
    block_ids = slot_mapping // block_size
    block_offsets = slot_mapping % block_size
    return (
        kv_cache[0, block_ids, block_offsets, kv_head].clone(),
        kv_cache[1, block_ids, block_offsets, kv_head].clone(),
    )


def test_query_segment_runtime_splits_and_merges_by_head():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=4,
    )

    snapshot = runtime.snapshot()
    layer_snapshot = snapshot["requests"]["req-a"]["layers"]["model.layers.0.attn"]

    assert layer_snapshot["num_completed_pages"] == 2
    assert layer_snapshot["segments_by_head"]["0"] == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 0,
            "num_pages": 1,
            "split_from_previous_similarity": None,
        },
        {
            "segment_id": 1,
            "start_page": 1,
            "end_page": 1,
            "num_pages": 1,
            "split_from_previous_similarity": 0.0,
        },
    ]
    assert layer_snapshot["segments_by_head"]["1"] == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 1,
            "num_pages": 2,
            "split_from_previous_similarity": None,
        }
    ]


def test_query_segment_runtime_keeps_request_boundaries_and_partial_pages():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.5,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ]
    )

    runtime.capture_batch(
        request_ids=["req-a", "req-b"],
        layer_name="model.layers.1.attn",
        query=query,
        query_start_loc=torch.tensor([0, 3, 5], dtype=torch.int32),
        seq_lens=torch.tensor([3, 2], dtype=torch.int32),
        num_actual_tokens=5,
    )

    snapshot = runtime.snapshot()
    req_a = snapshot["requests"]["req-a"]["layers"]["model.layers.1.attn"]
    req_b = snapshot["requests"]["req-b"]["layers"]["model.layers.1.attn"]

    assert req_a["num_completed_pages"] == 1
    assert req_a["partial_page_token_count"] == 1
    assert req_b["num_completed_pages"] == 1
    assert req_b["partial_page_token_count"] == 0


def test_query_segment_runtime_realigns_after_prefix_cached_partial_page():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
        )
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor([[[1.0, 0.0]]]),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=1,
        is_prefilling=torch.tensor([True]),
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([True]),
    )

    layer = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"]

    assert layer["skipped_batches"] == 0
    assert layer["num_tokens_seen"] == 6
    assert layer["num_completed_pages"] == 1
    assert layer["unobserved_prefix_tokens"] == 3
    assert layer["unobserved_prefix_pages"] == 2
    assert layer["pending_alignment_skip_tokens"] == 0
    assert layer["pages"] == [
        {
            "page_idx": 2,
            "start_token": 4,
            "end_token": 5,
            "num_tokens": 2,
            "phase": "prefill",
            "phase_token_counts": {"prefill": 2, "decode": 0},
        }
    ]
    assert layer["segments_by_head"]["0"] == [
        {
            "segment_id": 0,
            "start_page": 2,
            "end_page": 2,
            "num_pages": 1,
            "split_from_previous_similarity": None,
        }
    ]


def test_query_segment_runtime_quantizes_prefix_cached_observed_page():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    keys = torch.tensor(
        [
            [[0.10, 0.20]],
            [[0.30, 0.40]],
            [[0.31, -0.73]],
            [[1.27, -0.22]],
            [[-0.37, 0.63]],
            [[0.55, -0.49]],
        ]
    )
    values = torch.tensor(
        [
            [[0.50, 0.60]],
            [[0.70, 0.80]],
            [[-0.43, 0.79]],
            [[0.68, -1.02]],
            [[0.76, 0.28]],
            [[-0.34, 1.05]],
        ]
    )
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)
    original_keys, original_values = _read_kv_cache(kv_cache, slot_mapping)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=4,
        slot_mapping=slot_mapping[2:],
        kv_cache=kv_cache,
        num_kv_heads=1,
        is_prefilling=torch.tensor([True]),
    )

    captured_keys, captured_values = _read_kv_cache(kv_cache, slot_mapping)
    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert torch.allclose(captured_keys[2:4], original_keys[2:4])
    assert not torch.allclose(captured_values[2:4], original_values[2:4])
    assert torch.allclose(captured_keys[:2], original_keys[:2])
    assert torch.allclose(captured_values[:2], original_values[:2])
    assert torch.allclose(captured_keys[4:], original_keys[4:])
    assert torch.allclose(captured_values[4:], original_values[4:])
    assert segments[0]["start_page"] == 1
    assert segments[0]["end_page"] == 1
    assert segments[0]["quantized"] is True
    assert segments[1]["start_page"] == 2
    assert segments[1]["quantized"] is False


def test_query_segment_runtime_reports_stability_for_single_page_segment():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    values = keys.clone()
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    segment = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"][0]

    assert segment["stability"] == 1.0
    assert segment["quantized"] is False


def test_query_segment_runtime_tracks_key_prototype_from_key_cache():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[1.0, 3.0]],
            [[5.0, 7.0]],
        ]
    )
    values = torch.tensor(
        [
            [[11.0, 13.0]],
            [[17.0, 19.0]],
        ]
    )
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    segment = runtime._requests["req-a"]["model.layers.0.attn"].head_states[0].segments[0]

    assert segment.key_prototype_count == 1
    assert segment.key_prototype_sum is not None
    assert segment.key_prototype_sum.device.type == "cpu"
    assert segment.key_prototype_sum.dtype == torch.float32
    assert torch.allclose(segment.key_prototype_sum, torch.tensor([3.0, 5.0]))
    assert not torch.allclose(segment.key_prototype_sum, torch.tensor([14.0, 16.0]))


def test_query_segment_runtime_does_not_split_on_query_key_proxy_after_single_page_segment():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.85, 0.5267827]],
            [[0.85, 0.5267827]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.2, 0.0]],
            [[0.2, 0.0]],
            [[1.7, 1.0535654]],
            [[1.7, 1.0535654]],
        ]
    )
    values = keys.clone()
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=4,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert math.isclose(
        torch.nn.functional.cosine_similarity(query[0, 0], query[2, 0], dim=0).item(),
        0.85,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert segments == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 1,
            "num_pages": 2,
            "split_from_previous_similarity": None,
        },
    ]


def test_query_segment_runtime_splits_on_query_key_proxy_when_segment_has_multiple_pages():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.85, 0.5267827]],
            [[0.85, 0.5267827]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.2, 0.0]],
            [[0.2, 0.0]],
            [[0.2, 0.0]],
            [[0.2, 0.0]],
            [[1.7, 1.0535654]],
            [[1.7, 1.0535654]],
        ]
    )
    values = keys.clone()
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert math.isclose(
        torch.nn.functional.cosine_similarity(query[2, 0], query[4, 0], dim=0).item(),
        0.85,
        rel_tol=0.0,
        abs_tol=1e-6,
    )
    assert segments == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 1,
            "num_pages": 2,
            "split_from_previous_similarity": None,
        },
        {
            "segment_id": 1,
            "start_page": 2,
            "end_page": 2,
            "num_pages": 1,
            "split_from_previous_similarity": 0.85,
        },
    ]


def test_query_segment_runtime_does_not_split_pure_prefill_segment_on_max_pages():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            max_segment_pages=2,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        is_prefilling=torch.tensor([True]),
    )

    snapshot = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"]
    segments = snapshot["segments_by_head"]["0"]
    trace = snapshot["trace_by_head"]["0"]

    assert segments == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 2,
            "num_pages": 3,
            "split_from_previous_similarity": None,
        },
    ]
    assert trace[-1]["event_type"] == "segment_extended"
    assert trace[-1]["segmentation_method"] in {
        "query_cosine",
        "query_key_proxy_observe",
    }


def test_query_segment_runtime_keeps_long_prefill_segment_open_on_first_decode_page():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            max_segment_pages=2,
        )
    )
    prefill_query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    decode_query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=prefill_query,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=4,
        is_prefilling=torch.tensor([True]),
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=decode_query,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([False]),
    )

    snapshot = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"]
    segments = snapshot["segments_by_head"]["0"]
    trace = snapshot["trace_by_head"]["0"]

    assert segments == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 2,
            "num_pages": 3,
            "split_from_previous_similarity": None,
        },
    ]
    assert trace[-1]["phase"] == "decode"
    assert trace[-1]["event_type"] == "segment_extended"


def test_query_segment_runtime_splits_after_decode_tail_reaches_max_pages():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            max_segment_pages=2,
        )
    )
    prefill_query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    decode_query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=prefill_query,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=4,
        is_prefilling=torch.tensor([True]),
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=decode_query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([10], dtype=torch.int32),
        num_actual_tokens=6,
        is_prefilling=torch.tensor([False]),
    )

    snapshot = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"]
    segments = snapshot["segments_by_head"]["0"]
    trace = snapshot["trace_by_head"]["0"]

    assert segments == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 3,
            "num_pages": 4,
            "split_from_previous_similarity": None,
        },
        {
            "segment_id": 1,
            "start_page": 4,
            "end_page": 4,
            "num_pages": 1,
            "split_from_previous_similarity": 1.0,
        },
    ]
    assert trace[-1]["phase"] == "decode"
    assert trace[-1]["segmentation_method"] == "max_pages"


def test_query_segment_runtime_falls_back_to_query_cosine_without_key_prototypes():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ]
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=4,
    )

    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert segments == [
        {
            "segment_id": 0,
            "start_page": 0,
            "end_page": 0,
            "num_pages": 1,
            "split_from_previous_similarity": None,
        },
        {
            "segment_id": 1,
            "start_page": 1,
            "end_page": 1,
            "num_pages": 1,
            "split_from_previous_similarity": 0.0,
        },
    ]


def test_query_segment_runtime_skips_key_prototype_update_without_slots():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
        )
    )
    kv_cache = torch.zeros((2, 1, 2, 1, 2), dtype=torch.float32)
    layer_state = _LayerState(
        num_kv_heads=1,
        head_dim=2,
        partial_page_sum=torch.zeros((1, 2), dtype=torch.float32),
        head_states=[_HeadState()],
    )

    runtime._finalize_page(layer_state, kv_cache)

    segment = layer_state.head_states[0].segments[0]

    assert segment.key_prototype_sum is None
    assert segment.key_prototype_count == 0
    assert segment.stability == 1.0


def test_query_segment_runtime_keeps_recent_segment_fp16_until_split():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.31, -0.73]],
            [[1.27, -0.22]],
            [[0.44, -1.11]],
            [[0.92, 0.18]],
        ]
    )
    values = torch.tensor(
        [
            [[-0.43, 0.79]],
            [[0.68, -1.02]],
            [[1.11, 0.37]],
            [[-0.58, -0.94]],
        ]
    )
    kv_cache, slot_mapping = _build_kv_cache(
        keys,
        values,
        block_size=2,
        num_kv_heads=1,
    )
    original_keys, original_values = _read_kv_cache(kv_cache, slot_mapping)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=4,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    captured_keys, captured_values = _read_kv_cache(kv_cache, slot_mapping)
    snapshot = runtime.snapshot()
    segment = snapshot["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"][0]

    assert torch.allclose(captured_keys, original_keys)
    assert torch.allclose(captured_values, original_values)
    assert segment["quantized"] is False
    assert segment["is_recent"] is True
    assert "bit_width" not in segment


def test_query_segment_runtime_quantizes_previous_segment_when_new_segment_appears():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.31, -0.73]],
            [[1.27, -0.22]],
            [[0.44, -1.11]],
            [[0.92, 0.18]],
            [[-0.37, 0.63]],
            [[0.55, -0.49]],
        ]
    )
    values = torch.tensor(
        [
            [[-0.43, 0.79]],
            [[0.68, -1.02]],
            [[1.11, 0.37]],
            [[-0.58, -0.94]],
            [[0.76, 0.28]],
            [[-0.34, 1.05]],
        ]
    )
    kv_cache, slot_mapping = _build_kv_cache(
        keys,
        values,
        block_size=2,
        num_kv_heads=1,
    )
    original_keys, original_values = _read_kv_cache(kv_cache, slot_mapping)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    captured_keys, captured_values = _read_kv_cache(kv_cache, slot_mapping)
    snapshot = runtime.snapshot()
    segments = snapshot["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert torch.allclose(captured_keys[:4], original_keys[:4])
    assert not torch.allclose(captured_values[:4], original_values[:4])
    assert torch.allclose(captured_keys[4:], original_keys[4:])
    assert torch.allclose(captured_values[4:], original_values[4:])
    assert segments[0]["quantized"] is True
    assert segments[0]["bit_width"] == _select_fake_quant_bit_width(
        attention_score=segments[0]["attention_score"],
        stability=segments[0]["stability"],
    )
    assert segments[0]["key_bit_width"] == segments[0]["bit_width"]
    assert segments[0]["value_bit_width"] <= segments[0]["key_bit_width"]
    assert segments[0]["query_topk_channels"] == [0, 1]
    assert segments[0]["is_recent"] is False
    assert segments[1]["quantized"] is False
    assert segments[1]["is_recent"] is True


def test_query_segment_runtime_requantizes_expired_decode_grace_segment():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
            max_segment_pages=2,
        )
    )
    latest_query = torch.tensor([0.9, 0.8, 0.1, 0.0, 0.5, 0.4], dtype=torch.float32)
    page_slots = torch.tensor([0], dtype=torch.int64)
    kv_cache = torch.tensor(
        [
            [[[[-1.0, 0.5, 1.5, -0.5, 0.25, 0.75]]]],
            [[[[-1.0, 0.5, 1.5, -0.5, 0.25, 0.75]]]],
        ],
        dtype=torch.float32,
    )
    segment = _make_segment(
        attention_score=0.05,
        attention_shift=0.0,
        stability=-1.0,
    )
    segment.decode_page_count = 1
    recent_segment = _make_segment(
        segment_id=1,
        start_page=1,
        end_page=1,
        attention_score=0.8,
        attention_shift=0.0,
        stability=1.0,
    )
    recent_segment.decode_page_count = 1
    head_state = _HeadState(
        segments=[segment, recent_segment],
        pending_quantized_segment_ids=[0],
        completed_nonprefill_pages=3,
    )
    layer_state = _LayerState(
        num_kv_heads=1,
        head_dim=latest_query.shape[-1],
        partial_page_sum=torch.zeros((1, latest_query.shape[-1]), dtype=torch.float32),
        head_states=[head_state],
        pages=[
            _PageState(
                slots=page_slots,
                page_idx=0,
                start_token=0,
                end_token=int(page_slots.numel()) - 1,
                prefill_tokens=0,
                decode_tokens=int(page_slots.numel()),
            )
        ],
        page_index_to_state={
            0: _PageState(
                slots=page_slots,
                page_idx=0,
                start_token=0,
                end_token=int(page_slots.numel()) - 1,
                prefill_tokens=0,
                decode_tokens=int(page_slots.numel()),
            )
        },
    )

    runtime._maybe_quantize_pending_segments(
        request_id="req-a",
        layer_name="model.layers.0.attn",
        layer_state=layer_state,
        head_id=0,
        kv_cache=kv_cache,
        latest_query=latest_query,
    )

    assert segment.key_bit_width == 4
    assert segment.decode_grace_floor_applied is True
    assert segment.decode_grace_expires_after_nonprefill_pages == 5

    head_state.completed_nonprefill_pages = 5
    runtime._maybe_schedule_decode_grace_requantization(head_state)

    assert head_state.pending_requantized_segment_ids == [0]

    runtime._maybe_quantize_pending_segments(
        request_id="req-a",
        layer_name="model.layers.0.attn",
        layer_state=layer_state,
        head_id=0,
        kv_cache=kv_cache,
        latest_query=latest_query,
    )

    assert segment.key_bit_width == 2
    assert segment.value_bit_width == 2
    assert segment.decode_grace_floor_applied is False
    assert segment.decode_grace_expires_after_nonprefill_pages is None


def test_update_recent_segment_attention_scores_only_touches_current_and_prev():
    head_state = _HeadState(
        segments=[
            _Segment(0, 0, 0, None, attention_score=0.11),
            _Segment(1, 1, 1, 0.0, attention_score=0.22),
            _Segment(2, 2, 2, 0.0, attention_score=0.33),
        ]
    )
    head_state.segments[0].key_prototype_sum = torch.tensor([5.0, 0.0])
    head_state.segments[0].key_prototype_count = 1
    head_state.segments[1].key_prototype_sum = torch.tensor([0.0, 2.0])
    head_state.segments[1].key_prototype_count = 1
    head_state.segments[2].key_prototype_sum = torch.tensor([3.0, 0.0])
    head_state.segments[2].key_prototype_count = 1

    _update_recent_segment_attention_scores(
        head_state,
        latest_query=torch.tensor([2.0, 0.0]),
    )

    expected_raw_scores = torch.tensor([0.0, 6.0 / math.sqrt(2.0)], dtype=torch.float32)
    expected_normalized = torch.softmax(expected_raw_scores, dim=0).tolist()

    # Only the latest two segments should be rescored.
    assert head_state.segments[0].attention_score == 0.11
    assert head_state.segments[1].attention_score == pytest.approx(
        0.5 * expected_normalized[0] + 0.5 * 0.22
    )
    assert head_state.segments[2].attention_score == pytest.approx(
        0.5 * expected_normalized[1] + 0.5 * 0.33
    )


def test_query_segment_runtime_assigns_8_bits_to_high_attention_stable_segment():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.98, 0.19899748]],
            [[0.98, 0.19899748]],
            [[0.6, 0.8]],
            [[0.6, 0.8]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.8, 0.6]],
            [[0.8, 0.6]],
            [[0.8, 0.6]],
            [[0.8, 0.6]],
            [[1.0, -1.0]],
            [[1.0, -1.0]],
        ]
    )
    values = keys.clone()
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert segments[0]["bit_width"] == 8
    assert segments[0]["key_bit_width"] == 8
    assert segments[0]["value_bit_width"] == 2
    assert segments[0]["query_topk_channels"] == [0, 1]
    assert segments[0]["stability"] >= 0.85
    assert segments[0]["attention_score"] >= 0.65


def test_query_segment_runtime_assigns_4_bits_to_medium_segment():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.6, 0.8]],
            [[0.6, 0.8]],
            [[0.8, 0.6]],
            [[0.8, 0.6]],
        ]
    )
    keys = query.clone()
    values = query.clone()
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert segments[0]["bit_width"] == 4
    assert segments[0]["key_bit_width"] == 4
    assert segments[0]["value_bit_width"] == 2
    assert segments[0]["query_topk_channels"] == [0, 1]


def test_query_segment_runtime_snapshot_records_key_only_shift_signal():
    runtime = _build_runtime(fake_quant_method=1)
    segment = _make_segment(
        attention_score=0.45,
        attention_shift=0.95,
        stability=0.10,
    )

    key_bits, value_bits, _ = _quantize_once(runtime, segment)

    assert key_bits == 8
    assert value_bits == 2
    assert segment.attention_shift == pytest.approx(0.95)
    assert segment.normalized_attention_shift == pytest.approx(0.95)
    assert segment.importance is not None
    assert segment.importance >= 0.70


def test_query_segment_runtime_assigns_2_bits_to_low_attention_or_unstable_segment():
    runtime = _build_runtime(fake_quant_method=1)
    segment = _make_segment(attention_score=0.0, stability=-1.0)

    key_bits, value_bits, _ = _quantize_once(runtime, segment)

    assert key_bits == 2
    assert value_bits == 2


def test_query_segment_runtime_preserves_top4_query_channels_only_for_keys():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[9.0, 1.0, 8.0, 0.5, 7.0, 6.0]],
            [[9.0, 1.0, 8.0, 0.5, 7.0, 6.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.17, 0.23, -0.41, 0.36, -0.58, 0.69]],
            [[0.31, -0.47, 0.52, -0.63, 0.74, -0.85]],
            [[0.26, 0.37, -0.48, 0.59, -0.71, 0.82]],
            [[0.14, -0.28, 0.39, -0.51, 0.62, -0.73]],
            [[-0.37, 0.63, 0.12, -0.54, 0.22, -0.31]],
            [[0.55, -0.49, -0.27, 0.46, -0.18, 0.29]],
        ]
    )
    values = torch.tensor(
        [
            [[-0.19, 0.27, -0.35, 0.44, -0.53, 0.61]],
            [[0.22, -0.33, 0.47, -0.56, 0.68, -0.79]],
            [[-0.24, 0.38, -0.49, 0.57, -0.66, 0.77]],
            [[0.18, -0.29, 0.43, -0.52, 0.64, -0.75]],
            [[0.41, -0.37, 0.28, -0.19, 0.15, -0.11]],
            [[-0.32, 0.26, -0.21, 0.14, -0.09, 0.05]],
        ]
    )
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)
    original_keys, original_values = _read_kv_cache(kv_cache, slot_mapping)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    captured_keys, captured_values = _read_kv_cache(kv_cache, slot_mapping)
    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]
    protected_channels = [0, 2, 4, 5]
    non_protected_channels = [1, 3]

    assert segments[0]["quantized"] is True
    assert segments[0]["query_topk_channels"] == protected_channels
    assert torch.allclose(
        captured_keys[:4][:, protected_channels],
        original_keys[:4][:, protected_channels],
    )
    assert not torch.allclose(
        captured_keys[:4][:, non_protected_channels],
        original_keys[:4][:, non_protected_channels],
    )
    assert not torch.allclose(
        captured_values[:4][:, protected_channels],
        original_values[:4][:, protected_channels],
    )


def test_query_segment_runtime_bit_assignment_does_not_depend_on_seed():
    runtime_a = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    runtime_b = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=999,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.6, 0.8]],
            [[0.6, 0.8]],
            [[0.8, 0.6]],
            [[0.8, 0.6]],
        ]
    )

    def capture(runtime: QuerySegmentRuntime) -> int:
        keys = query.clone()
        values = query.clone()
        kv_cache, slot_mapping = _build_kv_cache(
            keys, values, block_size=2, num_kv_heads=1
        )
        runtime.capture_batch(
            request_ids=["req-a"],
            layer_name="model.layers.0.attn",
            query=query,
            query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
            seq_lens=torch.tensor([6], dtype=torch.int32),
            num_actual_tokens=6,
            slot_mapping=slot_mapping,
            kv_cache=kv_cache,
            num_kv_heads=1,
        )
        return runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
            "segments_by_head"
        ]["0"][0]["bit_width"]

    bit_width_a = capture(runtime_a)
    bit_width_b = capture(runtime_b)

    assert bit_width_a == bit_width_b


def test_query_segment_runtime_random_quant_method_uses_seeded_selection():
    selection_key = "req-a:model.layers.0.attn:0:1"
    expected_bit_width = _select_random_fake_quant_bit_width(
        fake_quant_seed=3,
        selection_key=selection_key,
    )
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=3,
            fake_quant_method=0,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[-1.0, 0.0]],
            [[-1.0, 0.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.31, -0.73]],
            [[1.27, -0.22]],
            [[0.44, -1.11]],
            [[0.92, 0.18]],
            [[-0.37, 0.63]],
            [[0.55, -0.49]],
        ]
    )
    values = keys.clone()
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    segment = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"][1]

    assert segment["quantized"] is True
    assert segment["bit_width"] == expected_bit_width


def test_query_segment_runtime_seed_only_env_change_preserves_state(monkeypatch):
    monkeypatch.setenv(ENV_ENABLE, "1")
    monkeypatch.setenv(ENV_PAGE_SIZE, "2")
    monkeypatch.setenv(ENV_THRESHOLD, "0.8")
    monkeypatch.setenv(ENV_FAKE_QUANT_ENABLE, "0")
    monkeypatch.setenv(ENV_FAKE_QUANT_SEED, "7")
    monkeypatch.delenv(ENV_PRIOR_PATH, raising=False)
    reset_query_segment_runtime()

    runtime_before = get_query_segment_runtime()
    capture_query_segments_from_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]]),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
    )

    snapshot_before = runtime_before.snapshot()
    assert snapshot_before["config"]["fake_quant_seed"] == 7
    assert "req-a" in snapshot_before["requests"]

    monkeypatch.setenv(ENV_FAKE_QUANT_SEED, "11")
    runtime_after = get_query_segment_runtime()
    snapshot_after = runtime_after.snapshot()

    assert runtime_after is runtime_before
    assert snapshot_after["config"]["fake_quant_seed"] == 11
    assert snapshot_after["requests"] == snapshot_before["requests"]
    reset_query_segment_runtime()


@pytest.mark.parametrize(
    ("forced_side", "force_bit_width"),
    [("key", 2)],
)
def test_query_segment_runtime_forced_quant_target_only_updates_targeted_side(
    forced_side,
    force_bit_width,
):
    baseline_runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
            force_layer_name="model.layers.0.attn",
            force_head_id=0,
            force_side=forced_side,
            force_bit_width=force_bit_width,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            [[9.0, 1.0, 8.0, 0.5, 7.0, 6.0]],
            [[9.0, 1.0, 8.0, 0.5, 7.0, 6.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.17, 0.23, -0.41, 0.36, -0.58, 0.69]],
            [[0.31, -0.47, 0.52, -0.63, 0.74, -0.85]],
            [[0.26, 0.37, -0.48, 0.59, -0.71, 0.82]],
            [[0.14, -0.28, 0.39, -0.51, 0.62, -0.73]],
            [[-0.37, 0.63, 0.12, -0.54, 0.22, -0.31]],
            [[0.55, -0.49, -0.27, 0.46, -0.18, 0.29]],
        ]
    )
    values = torch.tensor(
        [
            [[-0.19, 0.27, -0.35, 0.44, -0.53, 0.61]],
            [[0.22, -0.33, 0.47, -0.56, 0.68, -0.79]],
            [[-0.24, 0.38, -0.49, 0.57, -0.66, 0.77]],
            [[0.18, -0.29, 0.43, -0.52, 0.64, -0.75]],
            [[0.41, -0.37, 0.28, -0.19, 0.15, -0.11]],
            [[-0.32, 0.26, -0.21, 0.14, -0.09, 0.05]],
        ]
    )
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)
    baseline_kv_cache, baseline_slot_mapping = _build_kv_cache(
        keys,
        values,
        block_size=2,
        num_kv_heads=1,
    )
    original_keys, original_values = _read_kv_cache(kv_cache, slot_mapping)

    baseline_runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=baseline_slot_mapping,
        kv_cache=baseline_kv_cache,
        num_kv_heads=1,
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    baseline_segment = baseline_runtime.snapshot()["requests"]["req-a"]["layers"][
        "model.layers.0.attn"
    ]["segments_by_head"]["0"][0]
    baseline_keys, baseline_values = _read_kv_cache(
        baseline_kv_cache,
        baseline_slot_mapping,
    )
    captured_keys, captured_values = _read_kv_cache(kv_cache, slot_mapping)
    segment = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"][0]

    if forced_side == "key":
        assert segment["key_bit_width"] == force_bit_width
        assert segment["value_bit_width"] == baseline_segment["value_bit_width"]
        assert not torch.allclose(captured_keys[:4], baseline_keys[:4])
        assert torch.allclose(captured_values[:4], baseline_values[:4])
    else:
        assert segment["key_bit_width"] == baseline_segment["key_bit_width"]
        assert segment["value_bit_width"] == force_bit_width
        assert torch.allclose(captured_keys[:4], baseline_keys[:4])
        assert not torch.allclose(captured_values[:4], baseline_values[:4])
    assert torch.allclose(captured_keys[4:], original_keys[4:])
    assert torch.allclose(captured_values[4:], original_values[4:])


def test_query_segment_runtime_refreshes_forced_quant_target_when_env_changes(
    monkeypatch,
):
    monkeypatch.setenv(ENV_ENABLE, "1")
    monkeypatch.setenv(ENV_PAGE_SIZE, "2")
    monkeypatch.setenv(ENV_THRESHOLD, "0.8")
    monkeypatch.setenv(ENV_FAKE_QUANT_ENABLE, "1")
    monkeypatch.setenv(ENV_FAKE_QUANT_SEED, "0")
    monkeypatch.setenv(ENV_FORCE_LAYER_NAME, "model.layers.0.attn")
    monkeypatch.setenv(ENV_FORCE_HEAD_ID, "0")
    monkeypatch.setenv(ENV_FORCE_SIDE, "key")
    monkeypatch.setenv(ENV_FORCE_BIT_WIDTH, "8")
    monkeypatch.delenv(ENV_PRIOR_PATH, raising=False)
    reset_query_segment_runtime()

    runtime_before = get_query_segment_runtime()
    assert runtime_before.config.force_side == "key"

    monkeypatch.setenv(ENV_FORCE_BIT_WIDTH, "2")
    runtime_after = get_query_segment_runtime()

    assert runtime_after is not runtime_before
    assert runtime_after.config.force_side == "key"
    assert runtime_after.config.force_bit_width == 2
    assert runtime_after.snapshot()["requests"] == {}


def test_query_segment_config_from_env_rejects_empty_forced_layer_name(monkeypatch):
    monkeypatch.setenv(ENV_FORCE_LAYER_NAME, "   ")
    monkeypatch.setenv(ENV_FORCE_HEAD_ID, "0")
    monkeypatch.setenv(ENV_FORCE_SIDE, "key")
    monkeypatch.setenv(ENV_FORCE_BIT_WIDTH, "2")

    with pytest.raises(ValueError, match=ENV_FORCE_LAYER_NAME):
        QuerySegmentConfig.from_env()


def test_query_segment_runtime_force_target_from_env_only_changes_exact_target(
    monkeypatch,
):
    query = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [9.0, 1.0, 8.0, 0.5, 7.0, 6.0],
                [9.0, 1.0, 8.0, 0.5, 7.0, 6.0],
            ],
            [
                [9.0, 1.0, 8.0, 0.5, 7.0, 6.0],
                [9.0, 1.0, 8.0, 0.5, 7.0, 6.0],
            ],
        ]
    )
    keys = torch.tensor(
        [
            [
                [0.17, 0.23, -0.41, 0.36, -0.58, 0.69],
                [-0.12, 0.19, -0.27, 0.33, -0.41, 0.52],
            ],
            [
                [0.31, -0.47, 0.52, -0.63, 0.74, -0.85],
                [0.28, -0.34, 0.46, -0.57, 0.61, -0.72],
            ],
            [
                [0.26, 0.37, -0.48, 0.59, -0.71, 0.82],
                [-0.22, 0.31, -0.43, 0.54, -0.65, 0.76],
            ],
            [
                [0.14, -0.28, 0.39, -0.51, 0.62, -0.73],
                [0.11, -0.24, 0.37, -0.49, 0.58, -0.69],
            ],
            [
                [-0.37, 0.63, 0.12, -0.54, 0.22, -0.31],
                [0.08, 0.14, -0.19, 0.25, -0.32, 0.38],
            ],
            [
                [0.55, -0.49, -0.27, 0.46, -0.18, 0.29],
                [-0.05, 0.09, -0.14, 0.2, -0.26, 0.31],
            ],
        ]
    )
    values = torch.tensor(
        [
            [
                [-0.19, 0.27, -0.35, 0.44, -0.53, 0.61],
                [0.16, -0.22, 0.29, -0.37, 0.45, -0.54],
            ],
            [
                [0.22, -0.33, 0.47, -0.56, 0.68, -0.79],
                [-0.18, 0.25, -0.31, 0.39, -0.48, 0.57],
            ],
            [
                [-0.24, 0.38, -0.49, 0.57, -0.66, 0.77],
                [0.21, -0.29, 0.36, -0.44, 0.53, -0.62],
            ],
            [
                [0.18, -0.29, 0.43, -0.52, 0.64, -0.75],
                [-0.15, 0.23, -0.34, 0.41, -0.5, 0.6],
            ],
            [
                [0.41, -0.37, 0.28, -0.19, 0.15, -0.11],
                [0.13, -0.17, 0.22, -0.28, 0.35, -0.4],
            ],
            [
                [-0.32, 0.26, -0.21, 0.14, -0.09, 0.05],
                [-0.11, 0.15, -0.2, 0.24, -0.3, 0.36],
            ],
        ]
    )
    baseline_runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    baseline_layer0_kv_cache, baseline_layer0_slot_mapping = _build_kv_cache(
        keys,
        values,
        block_size=2,
        num_kv_heads=2,
    )
    baseline_layer1_kv_cache, baseline_layer1_slot_mapping = _build_kv_cache(
        keys,
        values,
        block_size=2,
        num_kv_heads=2,
    )
    baseline_runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=baseline_layer0_slot_mapping,
        kv_cache=baseline_layer0_kv_cache,
        num_kv_heads=2,
    )
    baseline_runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.1.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=baseline_layer1_slot_mapping,
        kv_cache=baseline_layer1_kv_cache,
        num_kv_heads=2,
    )

    monkeypatch.setenv(ENV_ENABLE, "1")
    monkeypatch.setenv(ENV_PAGE_SIZE, "2")
    monkeypatch.setenv(ENV_THRESHOLD, "0.8")
    monkeypatch.setenv(ENV_FAKE_QUANT_ENABLE, "1")
    monkeypatch.setenv(ENV_FAKE_QUANT_SEED, "0")
    monkeypatch.setenv(ENV_FORCE_LAYER_NAME, "model.layers.0.attn")
    monkeypatch.setenv(ENV_FORCE_HEAD_ID, "0")
    monkeypatch.setenv(ENV_FORCE_SIDE, "key")
    monkeypatch.setenv(ENV_FORCE_BIT_WIDTH, "2")
    monkeypatch.delenv(ENV_PRIOR_PATH, raising=False)
    reset_query_segment_runtime()

    layer0_kv_cache, layer0_slot_mapping = _build_kv_cache(
        keys,
        values,
        block_size=2,
        num_kv_heads=2,
    )
    layer1_kv_cache, layer1_slot_mapping = _build_kv_cache(
        keys,
        values,
        block_size=2,
        num_kv_heads=2,
    )
    capture_query_segments_from_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=layer0_slot_mapping,
        kv_cache=layer0_kv_cache,
        num_kv_heads=2,
    )
    capture_query_segments_from_batch(
        request_ids=["req-a"],
        layer_name="model.layers.1.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=layer1_slot_mapping,
        kv_cache=layer1_kv_cache,
        num_kv_heads=2,
    )

    runtime = get_query_segment_runtime()
    snapshot = runtime.snapshot()
    forced_layer = snapshot["requests"]["req-a"]["layers"]["model.layers.0.attn"]
    other_layer = snapshot["requests"]["req-a"]["layers"]["model.layers.1.attn"]
    baseline_snapshot = baseline_runtime.snapshot()
    baseline_forced_layer = baseline_snapshot["requests"]["req-a"]["layers"][
        "model.layers.0.attn"
    ]
    baseline_other_layer = baseline_snapshot["requests"]["req-a"]["layers"][
        "model.layers.1.attn"
    ]

    assert runtime.config.force_layer_name == "model.layers.0.attn"
    assert runtime.config.force_head_id == 0
    assert runtime.config.force_side == "key"
    assert runtime.config.force_bit_width == 2
    assert forced_layer["segments_by_head"]["0"][0]["key_bit_width"] == 2
    assert (
        forced_layer["segments_by_head"]["0"][0]["value_bit_width"]
        == baseline_forced_layer["segments_by_head"]["0"][0]["value_bit_width"]
    )
    assert (
        forced_layer["segments_by_head"]["1"]
        == baseline_forced_layer["segments_by_head"]["1"]
    )
    assert other_layer["segments_by_head"] == baseline_other_layer["segments_by_head"]

def test_query_segment_runtime_records_phase_aware_trace_and_summary():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
        )
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 4], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=4,
        is_prefilling=torch.tensor([True]),
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([False]),
    )

    layer = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"]

    assert layer["pages"] == [
        {
            "page_idx": 0,
            "start_token": 0,
            "end_token": 1,
            "num_tokens": 2,
            "phase": "prefill",
            "phase_token_counts": {"prefill": 2, "decode": 0},
        },
        {
            "page_idx": 1,
            "start_token": 2,
            "end_token": 3,
            "num_tokens": 2,
            "phase": "prefill",
            "phase_token_counts": {"prefill": 2, "decode": 0},
        },
        {
            "page_idx": 2,
            "start_token": 4,
            "end_token": 5,
            "num_tokens": 2,
            "phase": "decode",
            "phase_token_counts": {"prefill": 0, "decode": 2},
        },
    ]

    trace = layer["trace_by_head"]["0"]
    assert [event["event_type"] for event in trace] == [
        "segment_started",
        "segment_extended",
        "segment_split",
    ]
    assert [event["phase"] for event in trace] == ["prefill", "prefill", "decode"]
    assert trace[2]["segment_id"] == 1
    assert trace[2]["closed_segment_id"] == 0
    assert trace[2]["split_similarity"] == 0.0
    assert trace[2]["similarity_threshold"] == 0.8

    head_summary = layer["summary"]["heads"]["0"]
    assert head_summary["total_segments"] == 2
    assert head_summary["page_finalizations_by_phase"] == {
        "prefill": 2,
        "decode": 1,
        "mixed": 0,
        "unknown": 0,
    }
    assert head_summary["segment_creation_by_phase"] == {
        "prefill": 1,
        "decode": 1,
        "mixed": 0,
        "unknown": 0,
    }
    assert head_summary["split_events_by_phase"] == {
        "prefill": 0,
        "decode": 1,
        "mixed": 0,
        "unknown": 0,
    }


def test_query_segment_runtime_marks_mixed_phase_pages_in_trace_and_summary():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=4,
            similarity_threshold=0.8,
        )
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 3], dtype=torch.int32),
        seq_lens=torch.tensor([3], dtype=torch.int32),
        num_actual_tokens=3,
        is_prefilling=torch.tensor([True]),
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor([[[1.0, 0.0]]]),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=1,
        is_prefilling=torch.tensor([False]),
    )

    layer = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"]

    assert layer["pages"] == [
        {
            "page_idx": 0,
            "start_token": 0,
            "end_token": 3,
            "num_tokens": 4,
            "phase": "mixed",
            "phase_token_counts": {"prefill": 3, "decode": 1},
        }
    ]
    assert layer["trace_by_head"]["0"][0]["phase"] == "mixed"
    assert layer["summary"]["heads"]["0"]["page_finalizations_by_phase"] == {
        "prefill": 0,
        "decode": 0,
        "mixed": 1,
        "unknown": 0,
    }
    assert layer["summary"]["heads"]["0"]["segment_creation_by_phase"] == {
        "prefill": 0,
        "decode": 0,
        "mixed": 1,
        "unknown": 0,
    }


def test_query_segment_runtime_persists_progress_during_capture(tmp_path):
    output_path = tmp_path / "segments.json"
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            output_path=str(output_path),
        )
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([True]),
    )

    persisted_path = _build_dump_path(str(output_path))
    assert persisted_path.exists()

    payload = json.loads(persisted_path.read_text(encoding="utf-8"))
    layer = payload["requests"]["req-a"]["layers"]["model.layers.0.attn"]

    assert layer["num_completed_pages"] == 1
    assert layer["pages"][0]["phase"] == "prefill"
    assert layer["trace_by_head"]["0"][0]["event_type"] == "segment_started"


def test_query_segment_runtime_throttles_progress_persistence(tmp_path, monkeypatch):
    output_path = tmp_path / "segments.json"
    monotonic_values = iter((100.0, 100.0))
    monkeypatch.setattr(query_segments_module.time, "monotonic", lambda: next(monotonic_values))
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            output_path=str(output_path),
            progress_persist_interval_seconds=60.0,
        )
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([True]),
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([False]),
    )

    persisted_path = _build_dump_path(str(output_path))
    progress_payload = json.loads(persisted_path.read_text(encoding="utf-8"))
    progress_layer = progress_payload["requests"]["req-a"]["layers"]["model.layers.0.attn"]

    assert progress_layer["num_completed_pages"] == 1
    assert runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "num_completed_pages"
    ] == 2


def test_query_segment_runtime_final_dump_bypasses_progress_throttle(tmp_path, monkeypatch):
    output_path = tmp_path / "segments.json"
    monotonic_values = iter((100.0, 100.0))
    monkeypatch.setattr(query_segments_module.time, "monotonic", lambda: next(monotonic_values))
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            output_path=str(output_path),
            progress_persist_interval_seconds=60.0,
        )
    )

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([True]),
    )
    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
            ]
        ),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        num_actual_tokens=2,
        is_prefilling=torch.tensor([False]),
    )

    dumped_path = runtime.dump(_build_dump_path(str(output_path)))
    final_payload = json.loads(dumped_path.read_text(encoding="utf-8"))
    final_layer = final_payload["requests"]["req-a"]["layers"]["model.layers.0.attn"]

    assert final_layer["num_completed_pages"] == 2


def test_query_segment_runtime_quantizes_all_closed_segments_when_multiple_segments_close_in_batch():
    runtime = QuerySegmentRuntime(
        QuerySegmentConfig(
            enabled=True,
            page_size=2,
            similarity_threshold=0.8,
            fake_quant_enabled=True,
            fake_quant_seed=0,
        )
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[-1.0, 0.0]],
            [[-1.0, 0.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[0.31, -0.73]],
            [[1.27, -0.22]],
            [[0.44, -1.11]],
            [[0.92, 0.18]],
            [[-0.37, 0.63]],
            [[0.55, -0.49]],
        ]
    )
    values = torch.tensor(
        [
            [[-0.43, 0.79]],
            [[0.68, -1.02]],
            [[1.11, 0.37]],
            [[-0.58, -0.94]],
            [[0.52, -0.81]],
            [[-1.24, 0.43]],
        ]
    )
    kv_cache, slot_mapping = _build_kv_cache(keys, values, block_size=2, num_kv_heads=1)
    original_keys, original_values = _read_kv_cache(kv_cache, slot_mapping)

    runtime.capture_batch(
        request_ids=["req-a"],
        layer_name="model.layers.0.attn",
        query=query,
        query_start_loc=torch.tensor([0, 6], dtype=torch.int32),
        seq_lens=torch.tensor([6], dtype=torch.int32),
        num_actual_tokens=6,
        slot_mapping=slot_mapping,
        kv_cache=kv_cache,
        num_kv_heads=1,
    )

    captured_keys, captured_values = _read_kv_cache(kv_cache, slot_mapping)
    segments = runtime.snapshot()["requests"]["req-a"]["layers"]["model.layers.0.attn"][
        "segments_by_head"
    ]["0"]

    assert torch.allclose(captured_keys[:2], original_keys[:2])
    assert not torch.allclose(captured_values[:2], original_values[:2])
    assert torch.allclose(captured_keys[2:4], original_keys[2:4])
    assert not torch.allclose(captured_values[2:4], original_values[2:4])
    assert torch.allclose(captured_keys[4:], original_keys[4:])
    assert torch.allclose(captured_values[4:], original_values[4:])
    assert [segment["quantized"] for segment in segments] == [True, True, False]
