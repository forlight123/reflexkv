import importlib.util

import pytest
from types import SimpleNamespace

from vllm.v1.core.precision_kv import (
    accounting,
    contracts,
    demotion_planner,
    types,
)


def test_old_reflex_int4_compat_module_is_removed():
    assert importlib.util.find_spec("vllm.v1.core.reflex_int4") is None


def test_precision_kv_exports_canonical_state_planner_and_accounting_modules():
    assert types.PrecisionState.BF16.value == "bf16"
    assert demotion_planner.ReflexCandidateBreakdown().selected_actual == 0
    assert accounting.ReflexBlockTableStats(
        total_blocks=1,
        bf16_blocks=1,
        int4_blocks=0,
    ).bf16_blocks == 1


def test_reflex_int4_codec_exposes_quantization_backend_boundary():
    from vllm.v1.attention.ops import int4_kv_cache
    from vllm.v1.attention.ops.reflex_int4_codec import (
        ReflexInt4Codec,
        get_reflex_int4_codec,
    )

    codec = get_reflex_int4_codec()

    assert isinstance(codec, ReflexInt4Codec)
    assert codec.packed_head_size_bytes(16) == (
        int4_kv_cache.int4_packed_head_size_bytes(16)
    )
    with pytest.raises(ValueError):
        codec.packed_head_size_bytes(15)


def test_prefix_precision_contract_helper_owns_landing_fields():
    request = SimpleNamespace(
        kv_transfer_params={
            "reflex_int4_landing_pages": [1, 2],
            "reflex_int4_landing_block_ids": [11, 12],
            "reflex_int4_landing_required_blocks": 2,
            "reflex_int4_landing_planned_blocks": 2,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
            "unrelated": True,
        }
    )

    assert contracts.has_reflex_int4_landing_contract(request)

    contracts.clear_reflex_int4_landing_contract(request)

    assert contracts.has_reflex_int4_landing_contract(request) is False
    assert request.kv_transfer_params == {"unrelated": True}
