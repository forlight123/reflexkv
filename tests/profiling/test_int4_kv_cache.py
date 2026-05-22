import unittest

import torch

from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.attention.ops.int4_kv_cache import (
    int4_dequantize_kv_cache,
    int4_quantize_blocks_to_cache,
    int4_quantize_and_cache,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec


class Int4KVCacheTest(unittest.TestCase):
    def test_int4_dtype_uses_uint8_storage(self):
        self.assertIs(STR_DTYPE_TO_TORCH_DTYPE["int4"], torch.uint8)

    def test_full_attention_int4_page_size_uses_packed_payload_and_scale(self):
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.uint8,
            cache_dtype_str="int4",
        )

        packed_head_bytes = 128 // 2 + 1
        expected = 16 * 8 * (packed_head_bytes + packed_head_bytes)
        self.assertEqual(spec.page_size_bytes, expected)

    def test_triton_int4_cache_shape_uses_packed_head_width(self):
        shape = TritonAttentionBackend.get_kv_cache_shape(
            num_blocks=32,
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            cache_dtype_str="int4",
        )

        self.assertEqual(shape, (32, 2, 16, 8, 65))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_int4_quantize_and_dequantize_round_trips_cached_slots(self):
        key = torch.randn(5, 2, 8, device="cuda", dtype=torch.float16) * 0.5
        value = torch.randn(5, 2, 8, device="cuda", dtype=torch.float16) * 0.5
        cache = torch.zeros((3, 2, 4, 2, 5), device="cuda", dtype=torch.uint8)
        slot_mapping = torch.tensor([0, 1, 4, 7, -1], device="cuda")

        int4_quantize_and_cache(key, value, cache, slot_mapping)
        dequant = int4_dequantize_kv_cache(
            cache,
            head_size=8,
            dtype=torch.float16,
            num_blocks=2,
        )
        self.assertEqual(dequant.shape[0], 2)

        for token_idx, slot in enumerate(slot_mapping.tolist()):
            if slot < 0:
                continue
            block = slot // 4
            offset = slot % 4
            key_error = torch.max(
                torch.abs(dequant[block, 0, offset] - key[token_idx])
            ).item()
            value_error = torch.max(
                torch.abs(dequant[block, 1, offset] - value[token_idx])
            ).item()
            self.assertLess(
                key_error,
                0.16,
            )
            self.assertLess(value_error, 0.16)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_int4_quantize_blocks_matches_per_block_quantize(self):
        src_cache = torch.randn(
            (6, 2, 4, 2, 8), device="cuda", dtype=torch.float16
        ) * 0.5
        batched_cache = torch.zeros((6, 2, 4, 2, 5), device="cuda", dtype=torch.uint8)
        reference_cache = torch.zeros_like(batched_cache)
        src_block_ids = torch.tensor([4, 1, 5], device="cuda", dtype=torch.int64)
        dst_block_ids = torch.tensor([0, 3, 2], device="cuda", dtype=torch.int64)

        int4_quantize_blocks_to_cache(
            src_cache,
            batched_cache,
            src_block_ids,
            dst_block_ids,
        )
        for src_block_id, dst_block_id in zip(
            src_block_ids.tolist(), dst_block_ids.tolist()
        ):
            slot_mapping = torch.arange(
                dst_block_id * 4,
                (dst_block_id + 1) * 4,
                device="cuda",
                dtype=torch.long,
            )
            int4_quantize_and_cache(
                src_cache[src_block_id, 0],
                src_cache[src_block_id, 1],
                reference_cache,
                slot_mapping,
            )

        self.assertTrue(torch.equal(batched_cache, reference_cache))


if __name__ == "__main__":
    unittest.main()
