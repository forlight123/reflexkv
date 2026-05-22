from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.attention import attention


def _make_attention_forward_fixture(*, use_direct_call: bool) -> SimpleNamespace:
    return SimpleNamespace(
        calculate_kv_scales=False,
        query_quant=None,
        num_heads=1,
        num_kv_heads=1,
        head_size=4,
        head_size_v=4,
        use_direct_call=use_direct_call,
        attn_backend=SimpleNamespace(forward_includes_kv_cache_update=False),
        kv_sharing_target_layer_name=None,
        layer_name="layer.0",
        impl=SimpleNamespace(supports_quant_query_input=False),
    )


def test_semantiq_unified_kv_cache_update_captures_after_kv_update(monkeypatch):
    events: list[str] = []
    query = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])
    key = torch.tensor([[[11.0, 12.0, 13.0, 14.0]], [[15.0, 16.0, 17.0, 18.0]]])
    value = torch.tensor([[[21.0, 22.0, 23.0, 24.0]], [[25.0, 26.0, 27.0, 28.0]]])
    kv_cache = torch.zeros(2, 1, 2, 1, 4, dtype=torch.float32)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
    expected_key = key.view(2, 1, 4)[:, 0]
    expected_value = value.view(2, 1, 4)[:, 0]
    attn_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
    )
    def do_kv_cache_update(*args, **kwargs):
        del args, kwargs
        events.append("kv")
        kv_cache[0, 0, :, 0].copy_(key[:, 0])
        kv_cache[1, 0, :, 0].copy_(value[:, 0])

    fake_impl = SimpleNamespace(do_kv_cache_update=Mock(side_effect=do_kv_cache_update))
    fake_layer = SimpleNamespace(impl=fake_impl, num_kv_heads=1)

    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs={"semantiq_request_ids": ("req-a",)}),
    )
    monkeypatch.setattr(
        attention,
        "get_attention_context",
        lambda layer_name: (attn_metadata, fake_layer, kv_cache, slot_mapping),
    )
    def capture_query_segments_from_batch(**kwargs):
        events.append("capture")
        assert torch.allclose(kwargs["kv_cache"][0, 0, :, 0], key[:, 0])
        assert torch.allclose(kwargs["kv_cache"][1, 0, :, 0], value[:, 0])

    monkeypatch.setattr(
        attention,
        "capture_query_segments_from_batch",
        capture_query_segments_from_batch,
    )

    assert hasattr(torch.ops.vllm, "semantiq_unified_kv_cache_update")
    dummy = attention.semantiq_unified_kv_cache_update(query, key, value, "layer.0")

    assert events == ["kv", "capture"]
    fake_impl.do_kv_cache_update.assert_called_once()
    assert dummy.numel() == 0
    assert dummy.device == kv_cache.device
    assert dummy.dtype == kv_cache.dtype


def test_semantiq_unified_kv_cache_update_skips_capture_when_kv_update_is_skipped(
    monkeypatch,
):
    capture_calls: list[dict[str, object]] = []
    query = torch.randn(2, 1, 4)
    key = torch.randn(2, 1, 4)
    value = torch.randn(2, 1, 4)
    kv_cache = torch.zeros(2, 1, 1, 1, 4, dtype=torch.float32)
    attn_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
    )
    fake_impl = SimpleNamespace(do_kv_cache_update=Mock())
    fake_layer = SimpleNamespace(impl=fake_impl, num_kv_heads=1)

    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: SimpleNamespace(additional_kwargs={"semantiq_request_ids": ("req-a",)}),
    )
    monkeypatch.setattr(
        attention,
        "get_attention_context",
        lambda layer_name: (attn_metadata, fake_layer, kv_cache, None),
    )
    monkeypatch.setattr(
        attention,
        "capture_query_segments_from_batch",
        lambda **kwargs: capture_calls.append(kwargs),
    )

    attention.semantiq_unified_kv_cache_update(query, key, value, "layer.0")

    fake_impl.do_kv_cache_update.assert_not_called()
    assert capture_calls == []


@pytest.mark.parametrize("use_direct_call", [True, False])
def test_attention_forward_uses_semantiq_kv_update_when_capture_enabled(
    monkeypatch, use_direct_call
):
    attn = _make_attention_forward_fixture(use_direct_call=use_direct_call)
    query = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    key = torch.tensor([[11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0]])
    value = torch.tensor([[21.0, 22.0, 23.0, 24.0], [25.0, 26.0, 27.0, 28.0]])
    expected_key = key.view(2, 1, 4)[:, 0]
    expected_value = value.view(2, 1, 4)[:, 0]
    kv_cache = torch.zeros(2, 1, 2, 1, 4, dtype=torch.float32)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
    attn_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
    )
    capture_calls: list[dict[str, object]] = []
    kv_update_calls: list[str] = []

    monkeypatch.setattr(attention, "is_query_segment_capture_enabled", lambda: True)
    monkeypatch.setattr(
        attention,
        "maybe_capture_query_segments",
        Mock(side_effect=AssertionError("legacy python capture path should not run")),
    )
    monkeypatch.setattr(
        attention.torch.ops.vllm,
        "maybe_capture_query_segments",
        Mock(side_effect=AssertionError("legacy torch.ops capture path should not run")),
        raising=False,
    )
    monkeypatch.setattr(
        attention,
        "unified_kv_cache_update",
        Mock(side_effect=AssertionError("baseline KV update should not run")),
    )
    monkeypatch.setattr(
        attention.torch.ops.vllm,
        "unified_kv_cache_update",
        Mock(side_effect=AssertionError("legacy torch.ops KV update should not run")),
        raising=False,
    )

    def do_kv_cache_update(
        attn_layer, key_arg, value_arg, kv_cache_arg, layer_slot_mapping_arg
    ):
        del attn_layer
        kv_update_calls.append("kv")
        assert layer_slot_mapping_arg is slot_mapping
        kv_cache_arg[0, 0, :, 0].copy_(key_arg[:, 0])
        kv_cache_arg[1, 0, :, 0].copy_(value_arg[:, 0])

    fake_impl = SimpleNamespace(
        do_kv_cache_update=Mock(side_effect=do_kv_cache_update)
    )
    fake_layer = SimpleNamespace(impl=fake_impl, num_kv_heads=1)
    forward_context = SimpleNamespace(
        additional_kwargs={"semantiq_request_ids": ("req-a",)}
    )

    def capture_query_segments_from_batch(**kwargs):
        capture_calls.append(kwargs)
        assert kwargs["request_ids"] == ("req-a",)
        assert kwargs["layer_name"] == "layer.0"
        assert kwargs["num_kv_heads"] == 1
        assert "force_side" not in kwargs
        assert torch.allclose(kwargs["kv_cache"][0, 0, :, 0], expected_key)
        assert torch.allclose(kwargs["kv_cache"][1, 0, :, 0], expected_value)

    monkeypatch.setattr(
        attention,
        "get_forward_context",
        lambda: forward_context,
    )
    monkeypatch.setattr(
        attention,
        "get_attention_context",
        lambda layer_name: (attn_metadata, fake_layer, kv_cache, slot_mapping),
    )
    monkeypatch.setattr(
        attention,
        "capture_query_segments_from_batch",
        capture_query_segments_from_batch,
    )

    if use_direct_call:
        unified_attention = Mock(side_effect=lambda *args, **kwargs: None)
        monkeypatch.setattr(
            attention, "unified_attention_with_output", unified_attention
        )
        result = attention.Attention.forward(attn, query, key, value)
        unified_attention.assert_called_once()
        assert unified_attention.call_args.kwargs["kv_cache_dummy_dep"] is not None
        assert result.shape == (2, 4)
    else:
        semantiq_update = Mock(side_effect=attention.semantiq_unified_kv_cache_update)
        unified_attention = Mock(side_effect=lambda *args, **kwargs: None)
        monkeypatch.setattr(
            attention.torch.ops.vllm,
            "semantiq_unified_kv_cache_update",
            semantiq_update,
            raising=False,
        )
        monkeypatch.setattr(
            attention.torch.ops.vllm,
            "unified_attention_with_output",
            unified_attention,
            raising=False,
        )
        result = attention.Attention.forward(attn, query, key, value)
        semantiq_update.assert_called_once()
        unified_attention.assert_called_once()
        assert unified_attention.call_args.kwargs["kv_cache_dummy_dep"] is not None
        assert result.shape == (2, 4)

    assert kv_update_calls == ["kv"]
    assert len(capture_calls) == 1
    assert torch.allclose(kv_cache[0, 0, :, 0], expected_key)
    assert torch.allclose(kv_cache[1, 0, :, 0], expected_value)


def test_semantiq_unified_kv_cache_update_fake_impl_returns_empty_tensor():
    query = torch.randn(2, 1, 4)
    key = torch.randn(2, 1, 4)
    value = torch.randn(2, 1, 4)

    dummy = attention.semantiq_unified_kv_cache_update_fake(
        query, key, value, "layer.0"
    )

    assert dummy.numel() == 0
    assert dummy.device == query.device
    assert dummy.dtype == query.dtype
