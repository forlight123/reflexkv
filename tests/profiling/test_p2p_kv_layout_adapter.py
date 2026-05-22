import torch

from vllm.distributed.kv_transfer.kv_connector.v1.p2p.p2p_nccl_connector import (
    _adapt_kv_cache_layout_for_injection,
    _iter_kv_block_chunks,
    _p2p_kv_tensor_id,
    _slice_kv_cache_blocks,
)


def test_adapts_flashattention_kv_first_cache_to_block_first_layer():
    layer = torch.empty(8, 2, 16, 4, 32)
    kv_cache = torch.empty(2, 3, 16, 4, 32)

    adapted = _adapt_kv_cache_layout_for_injection(layer, kv_cache)

    assert adapted.shape == (3, 2, 16, 4, 32)
    assert not adapted.is_contiguous()
    assert adapted.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr()


def test_adapts_block_first_cache_to_flashattention_kv_first_layer():
    layer = torch.empty(2, 8, 16, 4, 32)
    kv_cache = torch.empty(3, 2, 16, 4, 32)

    adapted = _adapt_kv_cache_layout_for_injection(layer, kv_cache)

    assert adapted.shape == (2, 3, 16, 4, 32)
    assert not adapted.is_contiguous()
    assert adapted.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr()


def test_keeps_matching_layout_unchanged():
    layer = torch.empty(8, 2, 16, 4, 32)
    kv_cache = torch.empty(3, 2, 16, 4, 32)

    adapted = _adapt_kv_cache_layout_for_injection(layer, kv_cache)

    assert adapted is kv_cache


def test_iter_kv_block_chunks_splits_block_ids_without_copying_all_blocks():
    block_ids = torch.arange(10)

    chunks = list(_iter_kv_block_chunks(block_ids, chunk_blocks=4))

    assert [(start, end, ids.tolist()) for start, end, ids in chunks] == [
        (0, 4, [0, 1, 2, 3]),
        (4, 8, [4, 5, 6, 7]),
        (8, 10, [8, 9]),
    ]


def test_iter_kv_block_chunks_keeps_single_chunk_when_disabled():
    block_ids = torch.arange(3)

    chunks = list(_iter_kv_block_chunks(block_ids, chunk_blocks=0))

    assert len(chunks) == 1
    assert chunks[0][0] == 0
    assert chunks[0][1] == 3
    assert chunks[0][2] is block_ids


def test_p2p_kv_tensor_id_uses_legacy_id_for_unchunked_transfer():
    assert _p2p_kv_tensor_id("req", "layer.0", 0, 10, 10) == "req#layer.0"


def test_p2p_kv_tensor_id_includes_chunk_offsets():
    assert (
        _p2p_kv_tensor_id("req", "layer.0", 32, 64, 128)
        == "req#layer.0#blocks32:64"
    )


def test_slice_kv_cache_blocks_uses_view_for_block_first_layout():
    kv_cache = torch.arange(6 * 2 * 4).reshape(6, 2, 4)

    chunk = _slice_kv_cache_blocks(kv_cache, block_dim=0, start=2, end=5)

    assert chunk.shape == (3, 2, 4)
    assert chunk.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr()
    assert torch.equal(chunk, kv_cache[2:5])


def test_slice_kv_cache_blocks_uses_view_for_kv_first_layout():
    kv_cache = torch.arange(2 * 6 * 4).reshape(2, 6, 4)

    chunk = _slice_kv_cache_blocks(kv_cache, block_dim=1, start=1, end=4)

    assert chunk.shape == (2, 3, 4)
    assert chunk.untyped_storage().data_ptr() == kv_cache.untyped_storage().data_ptr()
    assert torch.equal(chunk, kv_cache[:, 1:4])
