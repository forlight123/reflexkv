import numpy as np

from vllm.v1.core.precision_kv.accounting import summarize_reflex_block_table


def test_summarize_reflex_block_table_counts_only_active_pages():
    block_table = np.array(
        [
            [3, -1, -2, 0, 0],
            [4, 5, 0, 0, 0],
            [-3, 9, 10, -4, 0],
        ],
        dtype=np.int32,
    )
    num_blocks_per_row = np.array([3, 2, 4], dtype=np.int32)

    stats = summarize_reflex_block_table(
        block_table,
        num_blocks_per_row,
        num_rows=3,
    )

    assert stats.total_blocks == 9
    assert stats.bf16_blocks == 5
    assert stats.int4_blocks == 4


def test_summarize_reflex_block_table_ignores_unused_rows():
    block_table = np.array(
        [
            [-1, 2],
            [-2, -3],
        ],
        dtype=np.int32,
    )
    num_blocks_per_row = np.array([1, 2], dtype=np.int32)

    stats = summarize_reflex_block_table(
        block_table,
        num_blocks_per_row,
        num_rows=1,
    )

    assert stats.total_blocks == 1
    assert stats.bf16_blocks == 0
    assert stats.int4_blocks == 1
