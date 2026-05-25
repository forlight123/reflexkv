import numpy as np
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.ubatch_utils import UBatchSlice, split_attn_metadata


def _metadata_with_int4_counts() -> CommonAttentionMetadata:
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor([0, 1, 2], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        seq_lens=torch.tensor([16, 32], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
        max_query_len=1,
        max_seq_len=32,
        block_table_tensor=torch.tensor([[0, 1], [2, -1]], dtype=torch.int32),
        slot_mapping=torch.tensor([15, 31], dtype=torch.int64),
        has_reflex_int4_blocks=True,
        reflex_int4_block_counts_cpu=np.array([0, 1], dtype=np.int32),
    )


def test_microbatch_metadata_recomputes_reflex_int4_dispatch_flag():
    first, second = split_attn_metadata(
        [
            UBatchSlice(slice(0, 1), slice(0, 1)),
            UBatchSlice(slice(1, 2), slice(1, 2)),
        ],
        _metadata_with_int4_counts(),
    )

    assert first.has_reflex_int4_blocks is False
    assert second.has_reflex_int4_blocks is True
    assert first.reflex_int4_block_counts_cpu.tolist() == [0]
    assert second.reflex_int4_block_counts_cpu.tolist() == [1]
