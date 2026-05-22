from vllm.v1.core.precision_kv.chunks import (
    plan_remote_kv_chunk,
    write_remote_kv_chunk_contract,
)


def test_remote_kv_chunk_planner_uses_block_aligned_non_final_chunks():
    chunk = plan_remote_kv_chunk(
        request_id="req",
        prompt_token_count=1537,
        num_computed_tokens=0,
        block_size=16,
        chunk_tokens=513,
    )

    assert chunk is not None
    assert chunk.chunk_id == 0
    assert chunk.token_start == 0
    assert chunk.token_end == 512
    assert chunk.page_start == 0
    assert chunk.page_end == 32
    assert chunk.num_pages == 32
    assert chunk.is_last_chunk is False


def test_remote_kv_chunk_planner_allows_partial_final_chunk():
    chunk = plan_remote_kv_chunk(
        request_id="req",
        prompt_token_count=1537,
        num_computed_tokens=1536,
        block_size=16,
        chunk_tokens=512,
    )

    assert chunk is not None
    assert chunk.chunk_id == 3
    assert chunk.token_start == 1536
    assert chunk.token_end == 1537
    assert chunk.page_start == 96
    assert chunk.page_end == 97
    assert chunk.num_pages == 1
    assert chunk.is_last_chunk is True


def test_remote_kv_chunk_contract_is_plain_kv_transfer_params():
    params = {"transfer_id": "xfer"}
    chunk = plan_remote_kv_chunk(
        request_id="req",
        prompt_token_count=1024,
        num_computed_tokens=512,
        block_size=16,
        chunk_tokens=512,
    )
    assert chunk is not None

    write_remote_kv_chunk_contract(params, chunk, role="decode")

    assert params["reflex_remote_chunk_enabled"] is True
    assert params["reflex_remote_chunk_role"] == "decode"
    assert params["reflex_remote_chunk_id"] == 1
    assert params["reflex_remote_chunk_token_start"] == 512
    assert params["reflex_remote_chunk_token_end"] == 1024
    assert params["reflex_remote_chunk_page_start"] == 32
    assert params["reflex_remote_chunk_page_end"] == 64
    assert params["reflex_remote_chunk_is_last"] is True
