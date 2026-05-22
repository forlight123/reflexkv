import unittest

import torch

from vllm.config.cache import CacheConfig
from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.utils.torch_utils import (
    get_kv_cache_torch_dtype,
    kv_cache_dtype_str_to_dtype,
)


class _ModelConfig:
    dtype = torch.bfloat16


class ReflexInt4ConfigTest(unittest.TestCase):

    def test_reflex_int4_is_accepted_as_cache_dtype(self):
        config = CacheConfig(cache_dtype="reflex_int4")

        self.assertEqual(config.cache_dtype, "reflex_int4")

    def test_reflex_int4_disables_prefix_caching_for_v0(self):
        config = CacheConfig(
            cache_dtype="reflex_int4",
            enable_prefix_caching=True,
        )

        self.assertFalse(config.enable_prefix_caching)

    def test_reflex_int4_primary_cache_uses_model_dtype(self):
        self.assertIs(
            get_kv_cache_torch_dtype("reflex_int4", torch.bfloat16),
            torch.bfloat16,
        )
        self.assertIs(
            kv_cache_dtype_str_to_dtype("reflex_int4", _ModelConfig()),
            torch.bfloat16,
        )

    def test_reflex_int4_primary_page_size_is_bf16_sized(self):
        spec = FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=8,
            dtype=torch.bfloat16,
            cache_dtype_str="reflex_int4",
        )

        self.assertEqual(spec.page_size_bytes, 16 * 2 * (8 + 8) * 2)

    def test_triton_backend_accepts_reflex_int4_dtype(self):
        self.assertTrue(
            TritonAttentionBackend.supports_kv_cache_dtype("reflex_int4")
        )


if __name__ == "__main__":
    unittest.main()
