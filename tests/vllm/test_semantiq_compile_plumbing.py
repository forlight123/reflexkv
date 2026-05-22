from types import SimpleNamespace

import torch
from torch._higher_order_ops import auto_functionalized
from torch.fx.experimental.proxy_tensor import make_fx

from vllm.compilation.passes.fusion import rope_kvcache_fusion
from vllm.compilation.passes.fx_utils import find_auto_fn, find_auto_fn_maybe, is_func
from vllm.compilation.passes.utility.fix_functionalization import (
    FixFunctionalizationPass,
)
from vllm.config import CompilationConfig, CompilationMode


def test_splitting_ops_recognize_semantiq_kv_cache_update_boundary():
    config = CompilationConfig(
        splitting_ops=[
            "vllm::semantiq_unified_kv_cache_update",
            "vllm::unified_mla_kv_cache_update",
        ]
    )

    assert config.splitting_ops_contain_kv_cache_update() is True


def test_set_splitting_ops_for_v1_adds_semantiq_kv_cache_update_boundary():
    config = CompilationConfig(mode=CompilationMode.VLLM_COMPILE, splitting_ops=None)
    config.set_splitting_ops_for_v1(all2all_backend="allgather_reducescatter")

    assert config.splitting_ops is not None
    assert "vllm::unified_kv_cache_update" in config.splitting_ops
    assert "vllm::semantiq_unified_kv_cache_update" in config.splitting_ops
    assert "vllm::unified_mla_kv_cache_update" in config.splitting_ops
    assert all("v_floor_bits" not in op for op in config.splitting_ops)


def test_fused_rope_and_semantiq_kv_cache_update_captures_after_kv_update(
    monkeypatch,
):
    events: list[str] = []
    query = torch.tensor([[[1.0, 2.0, 3.0, 4.0]], [[5.0, 6.0, 7.0, 8.0]]])
    key = torch.tensor([[[11.0, 12.0, 13.0, 14.0]], [[15.0, 16.0, 17.0, 18.0]]])
    value = torch.tensor([[[21.0, 22.0, 23.0, 24.0]], [[25.0, 26.0, 27.0, 28.0]]])
    positions = torch.tensor([0, 1], dtype=torch.long)
    cos_sin_cache = torch.randn(16, 4, dtype=torch.float32)
    kv_cache = torch.zeros(2, 1, 2, 1, 4, dtype=torch.float32)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
    attn_metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        num_actual_tokens=2,
    )

    def do_rope_and_kv_cache_update(
        attn_layer,
        query_arg,
        key_arg,
        value_arg,
        positions_arg,
        cos_sin_cache_arg,
        is_neox_arg,
        kv_cache_arg,
        slot_mapping_arg,
    ):
        del attn_layer, query_arg, positions_arg, cos_sin_cache_arg, is_neox_arg
        assert slot_mapping_arg is slot_mapping
        events.append("kv")
        kv_cache_arg[0, 0, :, 0].copy_(key_arg[:, 0])
        kv_cache_arg[1, 0, :, 0].copy_(value_arg[:, 0])

    fake_layer = SimpleNamespace(
        impl=SimpleNamespace(do_rope_and_kv_cache_update=do_rope_and_kv_cache_update),
        num_kv_heads=1,
    )

    monkeypatch.setattr(
        rope_kvcache_fusion,
        "get_attention_context",
        lambda layer_name: (attn_metadata, fake_layer, kv_cache, slot_mapping),
    )

    def capture_after_kv_update(**kwargs):
        events.append("capture")
        assert kwargs["attn_metadata"] is attn_metadata
        assert kwargs["attn_layer"] is fake_layer
        assert kwargs["layer_slot_mapping"] is slot_mapping
        assert torch.allclose(kwargs["kv_cache"][0, 0, :, 0], key[:, 0])
        assert torch.allclose(kwargs["kv_cache"][1, 0, :, 0], value[:, 0])

    monkeypatch.setattr(
        rope_kvcache_fusion,
        "_capture_query_segments_after_kv_update",
        capture_after_kv_update,
    )

    dummy = rope_kvcache_fusion.fused_rope_and_semantiq_kv_cache_update_impl(
        query,
        key,
        value,
        positions,
        cos_sin_cache,
        True,
        "layer.0",
    )

    assert events == ["kv", "capture"]
    assert dummy.numel() == 0
    assert dummy.device == kv_cache.device
    assert dummy.dtype == kv_cache.dtype


def test_fix_functionalization_defunctionalizes_semantiq_fused_rope_kv_op():
    class SemantiqFusedRopeKVModule(torch.nn.Module):
        def forward(self, query, key, value, positions, cos_sin_cache):
            results = auto_functionalized(
                torch.ops.vllm.fused_rope_and_semantiq_kv_cache_update.default,
                query=query,
                key=key,
                value=value,
                positions=positions,
                cos_sin_cache=cos_sin_cache,
                is_neox=True,
                layer_name="layer.0",
            )
            return results[0], results[1], results[2]

    def summarize(outputs):
        return [
            (tensor.shape, tensor.dtype, tensor.device.type)
            for tensor in outputs
        ]

    config = SimpleNamespace(
        compilation_config=SimpleNamespace(
            splitting_ops=None,
            use_inductor_graph_partition=False,
            pass_config=SimpleNamespace(),
        ),
        model_config=SimpleNamespace(dtype=torch.float32),
        device_config=None,
    )
    inputs = (
        torch.empty(2, 1, 4, device="meta"),
        torch.empty(2, 1, 4, device="meta"),
        torch.empty(2, 1, 4, device="meta"),
        torch.empty(2, dtype=torch.long, device="meta"),
        torch.empty(16, 4, device="meta"),
    )

    graph_module = make_fx(SemantiqFusedRopeKVModule())(*inputs)

    find_auto_fn(
        graph_module.graph.nodes,
        torch.ops.vllm.fused_rope_and_semantiq_kv_cache_update.default,
    )
    before_summary = summarize(graph_module(*inputs))

    FixFunctionalizationPass(config)(graph_module.graph)
    graph_module.recompile()

    assert (
        find_auto_fn_maybe(
            graph_module.graph.nodes,
            torch.ops.vllm.fused_rope_and_semantiq_kv_cache_update.default,
        )
        is None
    )
    assert any(
        is_func(node, torch.ops.vllm.fused_rope_and_semantiq_kv_cache_update.default)
        for node in graph_module.graph.nodes
    )

    after_summary = summarize(graph_module(*inputs))
    assert after_summary == before_summary
