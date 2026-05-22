from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import (
    get_kv_cache_config_from_groups,
    get_reflex_int4_sidecar_page_size,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)


def test_reflex_int4_budget_defaults_to_bf16_first_auto_split():
    layer_name = "model.layers.0.self_attn.attn"
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        cache_dtype_str="reflex_int4",
    )
    group_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={layer_name: spec},
    )
    sidecar_page_bytes = get_reflex_int4_sidecar_page_size(spec)
    available_memory = 40 * spec.page_size_bytes
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None),
        model_config=SimpleNamespace(max_model_len=512),
    )

    kv_cache_config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=group_spec)],
        available_memory=available_memory,
    )

    expected_bf16_blocks = int(available_memory * 0.9 // spec.page_size_bytes)
    expected_int4_blocks = (
        available_memory - int(available_memory * 0.9)
    ) // sidecar_page_bytes

    assert kv_cache_config.num_blocks == expected_bf16_blocks
    assert kv_cache_config.reflex_int4_num_blocks == expected_int4_blocks
    assert kv_cache_config.kv_cache_tensors[0].size == (
        expected_bf16_blocks * spec.page_size_bytes
    )
    assert kv_cache_config.reflex_int4_budget is not None
    assert kv_cache_config.reflex_int4_budget["bf16_budget_fraction"] == 0.9
    assert kv_cache_config.reflex_int4_budget["int4_budget_fraction"] == 0.1


def test_reflex_int4_budget_fraction_can_be_overridden(monkeypatch):
    layer_name = "model.layers.0.self_attn.attn"
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        cache_dtype_str="reflex_int4",
    )
    group_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={layer_name: spec},
    )
    sidecar_page_bytes = (
        2
        * spec.block_size
        * spec.num_kv_heads
        * FullAttentionSpec.int4_packed_head_size_bytes(spec.head_size)
    )
    available_memory = 40 * spec.page_size_bytes
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(num_gpu_blocks_override=None)
    )
    monkeypatch.setenv("SEMANTIQ_REFLEX_INT4_BUDGET_FRACTION", "0.25")

    kv_cache_config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=group_spec)],
        available_memory=available_memory,
    )

    expected_bf16_blocks = int(available_memory * 0.75 // spec.page_size_bytes)
    expected_int4_blocks = int(available_memory * 0.25 // sidecar_page_bytes)

    assert kv_cache_config.num_blocks == expected_bf16_blocks
    assert kv_cache_config.reflex_int4_num_blocks == expected_int4_blocks
    assert kv_cache_config.reflex_int4_budget is not None
    assert kv_cache_config.reflex_int4_budget["bf16_budget_fraction"] == 0.75
    assert kv_cache_config.reflex_int4_budget["int4_budget_fraction"] == 0.25


def test_reflex_int4_block_override_rebudgets_remaining_memory_to_int4():
    layer_name = "model.layers.0.self_attn.attn"
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=32,
        dtype=torch.bfloat16,
        cache_dtype_str="reflex_int4",
    )
    group_spec = UniformTypeKVCacheSpecs(
        block_size=16,
        kv_cache_specs={layer_name: spec},
    )
    sidecar_page_bytes = (
        2
        * spec.block_size
        * spec.num_kv_heads
        * FullAttentionSpec.int4_packed_head_size_bytes(spec.head_size)
    )
    available_memory = 40 * spec.page_size_bytes
    bf16_blocks_override = 8
    vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(
            num_gpu_blocks_override=bf16_blocks_override,
        ),
        model_config=SimpleNamespace(max_model_len=512),
    )

    kv_cache_config = get_kv_cache_config_from_groups(
        vllm_config,
        [KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=group_spec)],
        available_memory=available_memory,
    )

    expected_bf16_budget_bytes = bf16_blocks_override * spec.page_size_bytes
    expected_int4_blocks = (
        available_memory - expected_bf16_budget_bytes
    ) // sidecar_page_bytes

    assert kv_cache_config.num_blocks == bf16_blocks_override
    assert kv_cache_config.reflex_int4_num_blocks == expected_int4_blocks
    assert kv_cache_config.reflex_int4_budget is not None
    assert (
        kv_cache_config.reflex_int4_budget["bf16_budget_bytes"]
        == expected_bf16_budget_bytes
    )
    assert (
        kv_cache_config.reflex_int4_budget["int4_num_blocks"]
        == expected_int4_blocks
    )
