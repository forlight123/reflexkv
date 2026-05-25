import unittest

import torch

from vllm.v1.core.precision_kv.types import (
    RecoveryClass,
    ReflexDemotion,
    ReflexRecovery,
)
from vllm.v1.worker.block_table import MultiGroupBlockTable


class ReflexInt4WorkerBlockTableTest(unittest.TestCase):

    def test_demoted_page_updates_worker_block_table_entry(self):
        block_table = MultiGroupBlockTable(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=8,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[2],
            kernel_block_sizes=[2],
        )
        block_table.add_row(([5, 6, 7],), row_idx=0)

        block_table.update_block_id(
            kv_cache_group_id=0,
            row_idx=0,
            page_idx=1,
            block_id=-2,
        )

        self.assertEqual(block_table[0].block_table.np[0, 1], -2)

    def test_demoted_page_update_rejects_group_out_of_range(self):
        block_table = MultiGroupBlockTable(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=8,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[2],
            kernel_block_sizes=[2],
        )

        with self.assertRaises(IndexError):
            block_table.apply_reflex_int4_demotions(
                row_by_req_id={"req-a": 0},
                demotions=[
                    ReflexDemotion(
                        request_id="req-a",
                        page_idx=0,
                        bf16_block_id=5,
                        int4_block_id=0,
                        encoded_block_table_id=-1,
                        kv_cache_group_id=1,
                    )
                ],
            )

    def test_recovered_page_updates_worker_block_table_entry(self):
        block_table = MultiGroupBlockTable(
            max_num_reqs=2,
            max_model_len=16,
            max_num_batched_tokens=8,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[2],
            kernel_block_sizes=[2],
        )
        block_table.add_row(([5, -2, 7],), row_idx=0)

        block_table.apply_reflex_int4_recoveries(
            row_by_req_id={"req-a": 0},
            recoveries=[
                ReflexRecovery(
                    request_id="req-a",
                    page_idx=1,
                    int4_block_id=1,
                    bf16_block_id=9,
                    encoded_block_table_id=9,
                    recovery_class=RecoveryClass.BF16_SHADOW,
                    kv_cache_group_id=0,
                )
            ],
        )

        self.assertEqual(block_table[0].block_table.np[0, 1], 9)

    def test_worker_block_table_tracks_int4_counts_incrementally(self):
        block_table = MultiGroupBlockTable(
            max_num_reqs=3,
            max_model_len=16,
            max_num_batched_tokens=8,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[2],
            kernel_block_sizes=[2],
        )
        block_table.add_row(([5, -2, 7],), row_idx=0)
        block_table.add_row(([-1, -3],), row_idx=1)

        self.assertEqual(block_table[0].num_reflex_int4_blocks_per_row[0], 1)
        self.assertEqual(block_table[0].num_reflex_int4_blocks_per_row[1], 2)

        block_table.update_block_id(
            kv_cache_group_id=0,
            row_idx=0,
            page_idx=2,
            block_id=-4,
        )
        block_table.update_block_id(
            kv_cache_group_id=0,
            row_idx=1,
            page_idx=0,
            block_id=11,
        )

        self.assertEqual(block_table[0].num_reflex_int4_blocks_per_row[0], 2)
        self.assertEqual(block_table[0].num_reflex_int4_blocks_per_row[1], 1)

        block_table.move_row(0, 2)
        self.assertEqual(block_table[0].num_reflex_int4_blocks_per_row[2], 2)

        block_table.clear_row(2)
        self.assertEqual(block_table[0].num_reflex_int4_blocks_per_row[2], 0)


if __name__ == "__main__":
    unittest.main()
