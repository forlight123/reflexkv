from vllm.v1.core.precision_kv.types import (
    RecoveryClass,
    ReflexDemotion,
    ReflexRecovery,
)
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.gpu_model_runner import (
    _apply_reflex_int4_demotions_to_request_states,
    _apply_reflex_int4_recoveries_to_request_states,
)


def test_reflex_int4_demotions_update_cached_request_state_block_ids():
    request_state = CachedRequestState(
        req_id="req-a",
        prompt_token_ids=[],
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([10, 11, 12],),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    demotion = ReflexDemotion(
        request_id="req-a",
        page_idx=1,
        bf16_block_id=11,
        int4_block_id=2,
        encoded_block_table_id=-3,
        kv_cache_group_id=0,
    )

    _apply_reflex_int4_demotions_to_request_states(
        {"req-a": request_state},
        [demotion],
    )

    assert request_state.block_ids == ([10, -3, 12],)


def test_reflex_int4_recoveries_update_cached_request_state_block_ids():
    request_state = CachedRequestState(
        req_id="req-a",
        prompt_token_ids=[],
        mm_features=[],
        sampling_params=None,
        generator=None,
        block_ids=([10, -3, 12],),
        num_computed_tokens=0,
        output_token_ids=[],
    )
    recovery = ReflexRecovery(
        request_id="req-a",
        page_idx=1,
        int4_block_id=2,
        bf16_block_id=8,
        encoded_block_table_id=8,
        recovery_class=RecoveryClass.BF16_SHADOW,
        kv_cache_group_id=0,
    )

    _apply_reflex_int4_recoveries_to_request_states(
        {"req-a": request_state},
        [recovery],
    )

    assert request_state.block_ids == ([10, 8, 12],)
