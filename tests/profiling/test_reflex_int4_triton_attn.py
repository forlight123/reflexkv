from types import SimpleNamespace

import torch

import vllm.v1.attention.backends.triton_attn as triton_attn_module
from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl


def test_triton_attention_forwards_layer_reflex_int4_cache(monkeypatch):
    captured = {}

    def fake_unified_attention(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(triton_attn_module, "unified_attention", fake_unified_attention)

    impl = TritonAttentionImpl(
        num_heads=1,
        head_size=32,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    layer = SimpleNamespace(
        _q_scale_float=1.0,
        _k_scale=torch.ones((1, 1), dtype=torch.float32),
        _v_scale=torch.ones((1, 1), dtype=torch.float32),
        reflex_int4_kv_cache=torch.empty(
            (1, 2, 16, 1, 17), dtype=torch.uint8
        ),
    )
    attn_metadata = SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=1,
        block_table=torch.tensor([[-1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=1,
        seq_threshold_3D=1,
        num_par_softmax_segments=1,
        softmax_segm_output=torch.empty((1, 1, 1, 32), dtype=torch.float32),
        softmax_segm_max=torch.empty((1, 1, 1), dtype=torch.float32),
        softmax_segm_expsum=torch.empty((1, 1, 1), dtype=torch.float32),
        mm_prefix_range_tensor=None,
        has_reflex_int4_blocks=True,
    )
    query = torch.zeros((1, 1, 32), dtype=torch.float32)
    kv_cache = torch.zeros((1, 2, 16, 1, 32), dtype=torch.float32)
    output = torch.empty_like(query)

    impl.forward(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output=output,
    )

    assert captured["reflex_int4_kv_cache"] is layer.reflex_int4_kv_cache


def test_triton_attention_forwards_impl_reflex_int4_cache(monkeypatch):
    captured = {}

    def fake_unified_attention(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(triton_attn_module, "unified_attention", fake_unified_attention)

    impl = TritonAttentionImpl(
        num_heads=1,
        head_size=32,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )
    impl.reflex_int4_kv_cache = torch.empty(
        (1, 2, 16, 1, 17), dtype=torch.uint8
    )
    layer = SimpleNamespace(
        _q_scale_float=1.0,
        _k_scale=torch.ones((1, 1), dtype=torch.float32),
        _v_scale=torch.ones((1, 1), dtype=torch.float32),
    )
    attn_metadata = SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=1,
        block_table=torch.tensor([[-1]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=1,
        seq_threshold_3D=1,
        num_par_softmax_segments=1,
        softmax_segm_output=torch.empty((1, 1, 1, 32), dtype=torch.float32),
        softmax_segm_max=torch.empty((1, 1, 1), dtype=torch.float32),
        softmax_segm_expsum=torch.empty((1, 1, 1), dtype=torch.float32),
        mm_prefix_range_tensor=None,
        has_reflex_int4_blocks=True,
    )
    query = torch.zeros((1, 1, 32), dtype=torch.float32)
    kv_cache = torch.zeros((1, 2, 16, 1, 32), dtype=torch.float32)
    output = torch.empty_like(query)

    impl.forward(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output=output,
    )

    assert captured["reflex_int4_kv_cache"] is impl.reflex_int4_kv_cache


def test_triton_attention_skips_reflex_cache_without_int4_blocks(monkeypatch):
    captured = {}

    def fake_unified_attention(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(triton_attn_module, "unified_attention", fake_unified_attention)

    impl = TritonAttentionImpl(
        num_heads=1,
        head_size=32,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="reflex_int4",
    )
    layer = SimpleNamespace(
        _q_scale_float=1.0,
        _k_scale=torch.ones((1, 1), dtype=torch.float32),
        _v_scale=torch.ones((1, 1), dtype=torch.float32),
        reflex_int4_kv_cache=torch.empty(
            (1, 2, 16, 1, 17), dtype=torch.uint8
        ),
    )
    attn_metadata = SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=1,
        seq_threshold_3D=1,
        num_par_softmax_segments=1,
        softmax_segm_output=torch.empty((1, 1, 1, 32), dtype=torch.float32),
        softmax_segm_max=torch.empty((1, 1, 1), dtype=torch.float32),
        softmax_segm_expsum=torch.empty((1, 1, 1), dtype=torch.float32),
        mm_prefix_range_tensor=None,
        has_reflex_int4_blocks=False,
    )
    query = torch.zeros((1, 1, 32), dtype=torch.float32)
    kv_cache = torch.zeros((1, 2, 16, 1, 32), dtype=torch.float32)
    output = torch.empty_like(query)

    impl.forward(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output=output,
    )

    assert captured["reflex_int4_kv_cache"] is None


def test_triton_attention_keeps_3d_decode_available_for_reflex_int4(monkeypatch):
    captured = {}

    def fake_unified_attention(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(triton_attn_module, "unified_attention", fake_unified_attention)

    impl = TritonAttentionImpl(
        num_heads=1,
        head_size=32,
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="reflex_int4",
    )
    layer = SimpleNamespace(
        _q_scale_float=1.0,
        _k_scale=torch.ones((1, 1), dtype=torch.float32),
        _v_scale=torch.ones((1, 1), dtype=torch.float32),
    )
    attn_metadata = SimpleNamespace(
        use_cascade=False,
        num_actual_tokens=1,
        block_table=torch.tensor([[0]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([1], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=1,
        seq_threshold_3D=1,
        num_par_softmax_segments=1,
        softmax_segm_output=torch.empty((1, 1, 1, 32), dtype=torch.float32),
        softmax_segm_max=torch.empty((1, 1, 1), dtype=torch.float32),
        softmax_segm_expsum=torch.empty((1, 1, 1), dtype=torch.float32),
        mm_prefix_range_tensor=None,
        has_reflex_int4_blocks=False,
    )
    query = torch.zeros((1, 1, 32), dtype=torch.float32)
    kv_cache = torch.zeros((1, 2, 16, 1, 32), dtype=torch.float32)
    output = torch.empty_like(query)

    impl.forward(
        layer,
        query,
        query,
        query,
        kv_cache,
        attn_metadata,
        output=output,
    )

    assert captured["seq_threshold_3D"] == 1
