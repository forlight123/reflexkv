import unittest

import torch

from reflex_cuda_test_utils import select_reflex_cuda_test_device
from vllm.v1.attention.ops.int4_kv_cache import (
    int4_packed_head_size_bytes,
    int4_quantize_and_cache,
)
from vllm.v1.attention.ops.reflex_int4_kv_cache import (
    materialize_reflex_int4_kv_cache,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class ReflexInt4DirectAttentionTest(unittest.TestCase):

    def test_direct_attention_reads_mixed_bf16_and_int4_pages(self):
        device = select_reflex_cuda_test_device()
        dtype = torch.bfloat16
        block_size = 2
        head_size = 16
        num_kv_heads = 1
        num_query_heads = 1

        bf16_cache = torch.empty(
            (1, 2, block_size, num_kv_heads, head_size),
            dtype=dtype,
            device=device,
        )
        bf16_cache[0] = torch.arange(
            2 * block_size * num_kv_heads * head_size,
            dtype=torch.float32,
            device=device,
        ).view(2, block_size, num_kv_heads, head_size).to(dtype)

        int4_cache = torch.empty(
            (
                1,
                2,
                block_size,
                num_kv_heads,
                int4_packed_head_size_bytes(head_size),
            ),
            dtype=torch.uint8,
            device=device,
        )
        int4_key = (
            torch.arange(block_size * num_kv_heads * head_size,
                         dtype=torch.float32,
                         device=device)
            .view(block_size, num_kv_heads, head_size)
            .sub(9.0)
            .to(dtype)
        )
        int4_value = (int4_key.float() * 0.5 + 3.0).to(dtype)
        int4_quantize_and_cache(
            int4_key,
            int4_value,
            int4_cache,
            torch.tensor([0, 1], dtype=torch.int64, device=device),
        )

        block_table = torch.tensor([[0, -1]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([4], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        q = torch.linspace(
            -0.5,
            0.5,
            steps=num_query_heads * head_size,
            dtype=torch.float32,
            device=device,
        ).view(1, num_query_heads, head_size).to(dtype)
        scale = head_size**-0.5
        descale = torch.ones((1, num_kv_heads), dtype=torch.float32, device=device)

        materialized = materialize_reflex_int4_kv_cache(
            bf16_cache=bf16_cache,
            int4_cache=int4_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            head_size=head_size,
            dtype=dtype,
        )
        materialized_k, materialized_v = materialized.kv_cache.unbind(1)
        expected = torch.empty_like(q)
        unified_attention(
            q=q,
            k=materialized_k,
            v=materialized_v,
            out=expected,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seq_lens,
            max_seqlen_k=4,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=materialized.block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=descale,
            v_descale=descale,
            seq_threshold_3D=None,
        )

        direct = torch.empty_like(q)
        bf16_k, bf16_v = bf16_cache.unbind(1)
        unified_attention(
            q=q,
            k=bf16_k,
            v=bf16_v,
            out=direct,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seq_lens,
            max_seqlen_k=4,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=descale,
            v_descale=descale,
            seq_threshold_3D=None,
            reflex_int4_kv_cache=int4_cache,
        )

        torch.testing.assert_close(
            direct.float(),
            expected.float(),
            atol=0,
            rtol=0,
        )

    def test_3d_decode_reads_mixed_bf16_and_int4_pages(self):
        device = select_reflex_cuda_test_device()
        dtype = torch.bfloat16
        block_size = 2
        head_size = 16
        num_kv_heads = 1
        num_query_heads = 1
        num_segments = 1

        bf16_cache = torch.empty(
            (1, 2, block_size, num_kv_heads, head_size),
            dtype=dtype,
            device=device,
        )
        bf16_cache[0] = torch.arange(
            2 * block_size * num_kv_heads * head_size,
            dtype=torch.float32,
            device=device,
        ).view(2, block_size, num_kv_heads, head_size).to(dtype)

        int4_cache = torch.empty(
            (
                1,
                2,
                block_size,
                num_kv_heads,
                int4_packed_head_size_bytes(head_size),
            ),
            dtype=torch.uint8,
            device=device,
        )
        int4_key = (
            torch.arange(
                block_size * num_kv_heads * head_size,
                dtype=torch.float32,
                device=device,
            )
            .view(block_size, num_kv_heads, head_size)
            .sub(9.0)
            .to(dtype)
        )
        int4_value = (int4_key.float() * 0.5 + 3.0).to(dtype)
        int4_quantize_and_cache(
            int4_key,
            int4_value,
            int4_cache,
            torch.tensor([0, 1], dtype=torch.int64, device=device),
        )

        block_table = torch.tensor([[0, -1]], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([4], dtype=torch.int32, device=device)
        cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
        q = torch.linspace(
            -0.5,
            0.5,
            steps=num_query_heads * head_size,
            dtype=torch.float32,
            device=device,
        ).view(1, num_query_heads, head_size).to(dtype)
        scale = head_size**-0.5
        descale = torch.ones((1, num_kv_heads), dtype=torch.float32, device=device)

        materialized = materialize_reflex_int4_kv_cache(
            bf16_cache=bf16_cache,
            int4_cache=int4_cache,
            block_table=block_table,
            seq_lens=seq_lens,
            block_size=block_size,
            head_size=head_size,
            dtype=dtype,
        )
        materialized_k, materialized_v = materialized.kv_cache.unbind(1)
        expected = torch.empty_like(q)
        unified_attention(
            q=q,
            k=materialized_k,
            v=materialized_v,
            out=expected,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seq_lens,
            max_seqlen_k=4,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=materialized.block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=descale,
            v_descale=descale,
            seq_threshold_3D=128,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=torch.empty(
                (1, num_query_heads, num_segments, head_size),
                dtype=torch.float32,
                device=device,
            ),
            softmax_segm_max=torch.empty(
                (1, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
            softmax_segm_expsum=torch.empty(
                (1, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
        )

        direct = torch.empty_like(q)
        bf16_k, bf16_v = bf16_cache.unbind(1)
        unified_attention(
            q=q,
            k=bf16_k,
            v=bf16_v,
            out=direct,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=1,
            seqused_k=seq_lens,
            max_seqlen_k=4,
            softmax_scale=scale,
            causal=True,
            window_size=(-1, -1),
            block_table=block_table,
            softcap=0.0,
            q_descale=None,
            k_descale=descale,
            v_descale=descale,
            seq_threshold_3D=128,
            num_par_softmax_segments=num_segments,
            softmax_segm_output=torch.empty(
                (1, num_query_heads, num_segments, head_size),
                dtype=torch.float32,
                device=device,
            ),
            softmax_segm_max=torch.empty(
                (1, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
            softmax_segm_expsum=torch.empty(
                (1, num_query_heads, num_segments),
                dtype=torch.float32,
                device=device,
            ),
            reflex_int4_kv_cache=int4_cache,
        )

        torch.testing.assert_close(
            direct.float(),
            expected.float(),
            atol=0,
            rtol=0,
        )


if __name__ == "__main__":
    unittest.main()
