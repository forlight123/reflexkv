import unittest

import torch

from reflex_cuda_test_utils import select_reflex_cuda_test_device
from vllm.v1.attention.ops.int4_kv_cache import (
    int4_dequantize_kv_cache,
    int4_packed_head_size_bytes,
    int4_quantize_and_cache,
)
from vllm.v1.attention.ops.reflex_int4_kv_cache import (
    materialize_reflex_int4_kv_cache,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ReflexInt4MaterializeTest(unittest.TestCase):

    def test_materialize_mixed_bf16_and_int4_blocks_into_compact_cache(self):
        block_size = 2
        head_size = 4
        num_kv_heads = 1
        dtype = torch.bfloat16
        device = select_reflex_cuda_test_device()

        bf16_cache = torch.zeros(
            (3, 2, block_size, num_kv_heads, head_size),
            dtype=dtype,
            device=device,
        )
        bf16_cache[0] = torch.arange(
            2 * block_size * num_kv_heads * head_size,
            dtype=torch.float32,
            device=device,
        ).view(2, block_size, num_kv_heads, head_size).to(dtype)
        bf16_cache[2] = (bf16_cache[0].float() + 100).to(dtype)

        int4_cache = torch.empty(
            (1, 2, block_size, num_kv_heads,
             int4_packed_head_size_bytes(head_size)),
            dtype=torch.uint8,
            device=device,
        )
        key = torch.tensor(
            [[[1.0, -2.0, 3.0, -4.0]], [[2.0, -3.0, 4.0, -5.0]]],
            dtype=dtype,
            device=device,
        )
        value = key + 10
        int4_quantize_and_cache(
            key, value, int4_cache, torch.tensor([0, 1], device=device)
        )

        block_table = torch.tensor([[0, -1, 2]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([6], dtype=torch.int32, device=device)

        materialized = materialize_reflex_int4_kv_cache(
            bf16_cache=bf16_cache,
            int4_cache=int4_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            head_size=head_size,
            dtype=dtype,
        )

        expected_int4 = int4_dequantize_kv_cache(
            int4_cache, head_size=head_size, dtype=dtype
        )
        self.assertEqual(materialized.kv_cache.shape,
                         (3, 2, block_size, num_kv_heads, head_size))
        self.assertTrue(torch.equal(materialized.block_table,
                                    torch.tensor([[0, 1, 2]],
                                                 dtype=torch.int32,
                                                 device=device)))
        self.assertTrue(torch.equal(materialized.kv_cache[0], bf16_cache[0]))
        self.assertTrue(torch.equal(materialized.kv_cache[2], bf16_cache[2]))
        torch.testing.assert_close(
            materialized.kv_cache[1].float(),
            expected_int4[0].float(),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
