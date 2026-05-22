from contextlib import nullcontext
from types import SimpleNamespace

import torch

import vllm.v1.worker.gpu_model_runner as gpu_model_runner_module
from vllm.v1.attention.ops.reflex_int4_codec import ReflexInt4Codec
from vllm.v1.core.precision_kv.types import (
    RecoveryClass,
    ReflexDemotion,
    ReflexRecovery,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _FakeLateInteractionRunner:

    def on_requests_finished(self, req_ids):
        self.finished = set(req_ids)


class _FakeBlockTable:

    def __init__(self):
        self.demotions = None

    def apply_reflex_int4_demotions(self, *, row_by_req_id, demotions):
        self.demotions = (dict(row_by_req_id), list(demotions))


class _FakeInputBatch:

    def __init__(self):
        self.req_id_to_index = {"req-0": 0}
        self.block_table = _FakeBlockTable()
        self.removed_req_ids = []
        self.condensed = False
        self.metadata_refreshed = False

    def remove_request(self, req_id):
        self.removed_req_ids.append(req_id)
        self.req_id_to_index.pop(req_id, None)

    def condense(self):
        self.condensed = True

    def refresh_metadata(self):
        self.metadata_refreshed = True


def test_reflex_int4_demotion_only_step_preserves_persistent_batch(monkeypatch):
    runner = object.__new__(GPUModelRunner)
    runner.requests = {"req-0": object()}
    runner.num_prompt_logprobs = {}
    runner.late_interaction_runner = _FakeLateInteractionRunner()
    runner.input_batch = _FakeInputBatch()
    runner.encoder_cache = {}
    runner.speculative_config = None
    runner.use_async_spec_decode = False
    runner._execute_reflex_int4_demotions = lambda demotions: None
    runner._may_reorder_batch = lambda scheduler_output: None
    monkeypatch.setattr(
        gpu_model_runner_module,
        "get_pp_group",
        lambda: SimpleNamespace(is_last_rank=True),
    )

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.reflex_int4_demotions = [
        ReflexDemotion(
            request_id="req-0",
            page_idx=0,
            bf16_block_id=3,
            int4_block_id=1,
            encoded_block_table_id=-2,
            kv_cache_group_id=0,
        )
    ]

    runner._update_states(scheduler_output)

    assert runner.input_batch.removed_req_ids == []
    assert runner.input_batch.req_id_to_index == {"req-0": 0}
    assert runner.input_batch.block_table.demotions is not None


def test_reflex_int4_demotion_only_step_polls_kv_connector(monkeypatch):
    runner = object.__new__(GPUModelRunner)
    runner.execute_model_state = None
    runner.routed_experts_initialized = False
    runner.speculative_config = None
    runner.reflex_int4_trace_enabled = False
    runner.cache_config = SimpleNamespace(
        cache_dtype="reflex_int4",
        kv_sharing_fast_prefill=False,
    )
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend=None,
        data_parallel_size=1,
    )
    runner.vllm_config = object()
    runner.synchronize_input_prep = lambda: nullcontext()
    runner._update_states = lambda scheduler_output: None
    connector_group = SimpleNamespace(handle_preemptions=lambda metadata: None)
    monkeypatch.setattr(
        gpu_model_runner_module,
        "has_ec_transfer",
        lambda: False,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "has_kv_transfer_group",
        lambda: True,
    )
    monkeypatch.setattr(
        gpu_model_runner_module,
        "get_kv_transfer_group",
        lambda: connector_group,
    )

    sentinel = object()
    calls = []

    def fake_no_forward(scheduler_output, vllm_config):
        calls.append((scheduler_output, vllm_config))
        return sentinel

    runner.kv_connector_no_forward = fake_no_forward

    scheduler_output = SchedulerOutput.make_empty()
    scheduler_output.kv_connector_metadata = object()
    scheduler_output.reflex_int4_demotions = [
        ReflexDemotion(
            request_id="req-0",
            page_idx=0,
            bf16_block_id=3,
            int4_block_id=1,
            encoded_block_table_id=-2,
            kv_cache_group_id=0,
        )
    ]

    assert runner.execute_model(scheduler_output) is sentinel
    assert calls == [(scheduler_output, runner.vllm_config)]


def test_reflex_int4_sidecar_is_bound_for_uniform_attention_specs():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="reflex_int4")
    runner.runner_only_attn_layers = set()
    runner.reflex_int4_kv_caches = {}
    runner.reflex_int4_group_layer_names = {}

    layer_names = [
        "model.layers.0.self_attn.attn",
        "model.layers.1.self_attn.attn",
    ]
    kv_caches = {
        layer_name: torch.empty((4, 2, 16, 1, 32), dtype=torch.bfloat16)
        for layer_name in layer_names
    }
    forward_context = {
        layer_name: SimpleNamespace(impl=SimpleNamespace(), kv_cache=kv_cache)
        for layer_name, kv_cache in kv_caches.items()
    }
    runner.compilation_config = SimpleNamespace(
        static_forward_context=forward_context
    )

    specs = {
        layer_name: FullAttentionSpec(
            block_size=16,
            num_kv_heads=1,
            head_size=32,
            dtype=torch.bfloat16,
            cache_dtype_str="reflex_int4",
        )
        for layer_name in layer_names
    }
    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=layer_names,
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs=specs,
                ),
            )
        ],
    )

    runner._initialize_reflex_int4_kv_caches(kv_cache_config, kv_caches)

    assert set(runner.reflex_int4_kv_caches) == set(layer_names)
    assert runner.reflex_int4_group_layer_names == {0: layer_names}
    for layer_name in layer_names:
        int4_cache = runner.reflex_int4_kv_caches[layer_name]
        layer = forward_context[layer_name]
        assert layer.reflex_int4_kv_cache is int4_cache
        assert layer.impl.reflex_int4_kv_cache is int4_cache
        assert int4_cache.shape == (4, 2, 16, 1, 17)


def test_reflex_int4_sidecar_uses_configured_int4_capacity():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="reflex_int4")
    runner.runner_only_attn_layers = set()
    runner.reflex_int4_kv_caches = {}
    runner.reflex_int4_group_layer_names = {}

    layer_name = "model.layers.0.self_attn.attn"
    kv_caches = {
        layer_name: torch.empty((4, 2, 16, 1, 32), dtype=torch.bfloat16)
    }
    forward_context = {
        layer_name: SimpleNamespace(
            impl=SimpleNamespace(), kv_cache=kv_caches[layer_name]
        )
    }
    runner.compilation_config = SimpleNamespace(
        static_forward_context=forward_context
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        reflex_int4_num_blocks=11,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        layer_name: FullAttentionSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=32,
                            dtype=torch.bfloat16,
                            cache_dtype_str="reflex_int4",
                        )
                    },
                ),
            )
        ],
    )

    runner._initialize_reflex_int4_kv_caches(kv_cache_config, kv_caches)

    assert runner.reflex_int4_kv_caches[layer_name].shape == (11, 2, 16, 1, 17)


def test_reflex_mooncake_prefill_auto_dtype_without_compression_skips_int4_staging():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="ReFlexMooncakeConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={},
        )
    )
    runner.runner_only_attn_layers = set()
    runner.reflex_int4_kv_caches = {}
    runner.reflex_int4_group_layer_names = {}

    layer_name = "model.layers.0.self_attn.attn"
    kv_caches = {
        layer_name: torch.empty((4, 2, 16, 1, 32), dtype=torch.bfloat16)
    }
    forward_context = {
        layer_name: SimpleNamespace(
            impl=SimpleNamespace(), kv_cache=kv_caches[layer_name]
        )
    }
    runner.compilation_config = SimpleNamespace(
        static_forward_context=forward_context
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        layer_name: FullAttentionSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=32,
                            dtype=torch.bfloat16,
                            cache_dtype_str="auto",
                        )
                    },
                ),
            )
        ],
    )

    runner._initialize_reflex_int4_kv_caches(kv_cache_config, kv_caches)

    assert runner.reflex_int4_kv_caches == {}
    assert not hasattr(forward_context[layer_name], "reflex_int4_kv_cache")


def test_reflex_mooncake_auto_dtype_ignores_legacy_compression_flag():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="ReFlexMooncakeConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "reflex_enable_handoff_compression": True,
            },
        )
    )
    runner.runner_only_attn_layers = set()
    runner.reflex_int4_kv_caches = {}
    runner.reflex_int4_group_layer_names = {}

    layer_name = "model.layers.0.self_attn.attn"
    kv_caches = {
        layer_name: torch.empty((4, 2, 16, 1, 32), dtype=torch.bfloat16)
    }
    forward_context = {
        layer_name: SimpleNamespace(
            impl=SimpleNamespace(), kv_cache=kv_caches[layer_name]
        )
    }
    runner.compilation_config = SimpleNamespace(
        static_forward_context=forward_context
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        layer_name: FullAttentionSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=32,
                            dtype=torch.bfloat16,
                            cache_dtype_str="auto",
                        )
                    },
                ),
            )
        ],
    )

    runner._initialize_reflex_int4_kv_caches(kv_cache_config, kv_caches)

    assert runner.reflex_int4_kv_caches == {}
    assert not hasattr(forward_context[layer_name], "reflex_int4_kv_cache")


def test_reflex_int4_flash_layout_allocates_block_major_int4_staging():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="reflex_int4")
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="ReFlexMooncakeConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={},
        )
    )
    runner.runner_only_attn_layers = set()
    runner.reflex_int4_kv_caches = {}
    runner.reflex_int4_group_layer_names = {}

    layer_name = "model.layers.0.self_attn.attn"
    kv_caches = {
        layer_name: torch.empty((2, 4, 16, 1, 32), dtype=torch.bfloat16)
    }
    forward_context = {
        layer_name: SimpleNamespace(
            impl=SimpleNamespace(), kv_cache=kv_caches[layer_name]
        )
    }
    runner.compilation_config = SimpleNamespace(
        static_forward_context=forward_context
    )

    kv_cache_config = KVCacheConfig(
        num_blocks=4,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                layer_names=[layer_name],
                kv_cache_spec=UniformTypeKVCacheSpecs(
                    block_size=16,
                    kv_cache_specs={
                        layer_name: FullAttentionSpec(
                            block_size=16,
                            num_kv_heads=1,
                            head_size=32,
                            dtype=torch.bfloat16,
                            cache_dtype_str="reflex_int4",
                        )
                    },
                ),
            )
        ],
    )

    runner._initialize_reflex_int4_kv_caches(kv_cache_config, kv_caches)

    assert runner.reflex_int4_kv_caches[layer_name].shape == (4, 2, 16, 1, 17)


def test_reflex_int4_sidecar_is_registered_with_kv_transfer_group():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="reflex_int4")
    runner.cross_layers_kv_cache = None
    runner.cross_layers_attn_backend = None
    runner.reflex_int4_kv_caches = {
        "layer.0": torch.empty(1, dtype=torch.uint8),
    }
    kv_caches = {"layer.0": torch.empty(1)}
    calls = []

    class FakeKVTransferGroup:

        def register_kv_caches(self, caches):
            calls.append(("bf16", caches))

        def register_reflex_int4_kv_caches(self, caches):
            calls.append(("int4", caches))

        def set_host_xfer_buffer_ops(self, ops):
            calls.append(("copy_ops", ops))

    runner._register_kv_caches_with_transfer_group(
        FakeKVTransferGroup(), kv_caches
    )

    assert calls[0] == ("bf16", kv_caches)
    assert calls[1] == ("int4", runner.reflex_int4_kv_caches)
    assert calls[2][0] == "copy_ops"
    assert len(calls) == 3


def test_reflex_mooncake_auto_dtype_does_not_register_int4_staging_with_kv_transfer_group():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="auto")
    runner.vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(
            kv_connector="ReFlexMooncakeConnector",
            kv_connector_extra_config={
                "reflex_enable_handoff_compression": True,
            },
        )
    )
    runner.cross_layers_kv_cache = None
    runner.cross_layers_attn_backend = None
    runner.reflex_int4_kv_caches = {
        "layer.0": torch.empty(1, dtype=torch.uint8),
    }
    kv_caches = {"layer.0": torch.empty(1)}
    calls = []

    class FakeKVTransferGroup:

        def register_kv_caches(self, caches):
            calls.append(("bf16", caches))

        def register_reflex_int4_kv_caches(self, caches):
            calls.append(("int4", caches))

        def set_host_xfer_buffer_ops(self, ops):
            calls.append(("copy_ops", ops))

    runner._register_kv_caches_with_transfer_group(
        FakeKVTransferGroup(), kv_caches
    )

    assert calls[0] == ("bf16", kv_caches)
    assert calls[1][0] == "copy_ops"
    assert len(calls) == 2


def test_reflex_int4_demote_exec_batches_pages_per_layer(monkeypatch):
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="reflex_int4")
    runner.reflex_int4_trace_enabled = False
    runner.device = torch.device("cpu")
    runner.reflex_int4_group_layer_names = {
        0: ["layer.0", "layer.1"],
    }
    runner.kv_cache_by_layer_name = {
        "layer.0": torch.empty((8, 2, 4, 1, 8), device=runner.device),
        "layer.1": torch.empty((8, 2, 4, 1, 8), device=runner.device),
    }
    runner.reflex_int4_kv_caches = {
        "layer.0": torch.empty((8, 2, 4, 1, 5), device=runner.device, dtype=torch.uint8),
        "layer.1": torch.empty((8, 2, 4, 1, 5), device=runner.device, dtype=torch.uint8),
    }
    calls = []

    original_quantize_blocks = ReflexInt4Codec.quantize_blocks_to_cache

    def fake_quantize_blocks(
        self,
        src_cache,
        int4_cache,
        src_block_ids,
        dst_block_ids,
    ):
        calls.append(
            (
                src_cache,
                int4_cache,
                id(src_block_ids),
                id(dst_block_ids),
                tuple(src_block_ids.cpu().tolist()),
                tuple(dst_block_ids.cpu().tolist()),
            )
        )

    monkeypatch.setattr(
        ReflexInt4Codec,
        "quantize_blocks_to_cache",
        fake_quantize_blocks,
    )

    runner._execute_reflex_int4_demotions(
        [
            ReflexDemotion(
                request_id="req-a",
                page_idx=0,
                bf16_block_id=3,
                int4_block_id=5,
                encoded_block_table_id=-6,
                kv_cache_group_id=0,
            ),
            ReflexDemotion(
                request_id="req-a",
                page_idx=1,
                bf16_block_id=4,
                int4_block_id=6,
                encoded_block_table_id=-7,
                kv_cache_group_id=0,
            ),
        ]
    )

    assert len(calls) == 2
    assert calls[0][2] == calls[1][2]
    assert calls[0][3] == calls[1][3]
    assert calls[0][4:] == ((3, 4), (5, 6))
    assert calls[1][4:] == ((3, 4), (5, 6))
    monkeypatch.setattr(
        ReflexInt4Codec,
        "quantize_blocks_to_cache",
        original_quantize_blocks,
    )


def test_reflex_int4_recoverable_demotion_stores_cpu_bf16_shadow():
    runner = object.__new__(GPUModelRunner)
    runner.reflex_bf16_shadow_store = {}
    bf16_cache = torch.arange(
        4 * 2 * 2 * 1 * 4,
        dtype=torch.float32,
    ).reshape(4, 2, 2, 1, 4).to(torch.bfloat16)
    demotion = ReflexDemotion(
        request_id="req-a",
        page_idx=1,
        bf16_block_id=2,
        int4_block_id=5,
        encoded_block_table_id=-6,
        kv_cache_group_id=0,
        recovery_class=RecoveryClass.BF16_SHADOW,
    )

    runner._store_reflex_bf16_shadow_pages("layer.0", bf16_cache, [demotion])

    key = (0, "req-a", 1, "layer.0")
    assert key in runner.reflex_bf16_shadow_store
    torch.testing.assert_close(
        runner.reflex_bf16_shadow_store[key],
        bf16_cache[2].cpu(),
    )


def test_reflex_int4_recovery_restores_bf16_shadow_to_new_block():
    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(cache_dtype="reflex_int4")
    runner.reflex_int4_trace_enabled = False
    runner.device = torch.device("cpu")
    runner.reflex_int4_group_layer_names = {0: ["layer.0"]}
    runner.kv_cache_by_layer_name = {
        "layer.0": torch.zeros((4, 2, 2, 1, 4), dtype=torch.bfloat16),
    }
    shadow = torch.arange(
        2 * 2 * 1 * 4,
        dtype=torch.float32,
    ).reshape(2, 2, 1, 4).to(torch.bfloat16)
    runner.reflex_bf16_shadow_store = {
        (0, "req-a", 1, "layer.0"): shadow.clone(),
    }

    runner._execute_reflex_int4_recoveries(
        [
            ReflexRecovery(
                request_id="req-a",
                page_idx=1,
                int4_block_id=5,
                bf16_block_id=3,
                encoded_block_table_id=3,
                recovery_class=RecoveryClass.BF16_SHADOW,
                kv_cache_group_id=0,
            )
        ]
    )

    torch.testing.assert_close(
        runner.kv_cache_by_layer_name["layer.0"][3],
        shadow,
    )
    assert runner.reflex_bf16_shadow_store == {}
