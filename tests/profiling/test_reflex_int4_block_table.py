import unittest

import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.precision_kv.types import (
    KVPageLifecycle,
    MemoryTier,
    PrecisionState,
    RecoveryClass,
    decode_block_table_entry,
)
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec


class ReflexInt4BlockTableTest(unittest.TestCase):

    def _make_manager(
        self,
        num_gpu_blocks: int = 8,
        reflex_int4_num_blocks: int | None = None,
        enable_caching: bool = False,
    ) -> FullAttentionManager:
        spec = FullAttentionSpec(
            block_size=2,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.bfloat16,
            cache_dtype_str="reflex_int4",
        )
        pool = BlockPool(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
            hash_block_size=2,
        )
        return FullAttentionManager(
            spec,
            block_pool=pool,
            enable_caching=enable_caching,
            kv_cache_group_id=0,
            reflex_int4_num_blocks=reflex_int4_num_blocks,
        )

    def test_manager_demotes_oldest_pages_and_delays_bf16_release(self):
        manager = self._make_manager()
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=10, num_tokens_main_model=10
        )
        old_block_ids = [block.block_id for block in new_blocks]
        free_before = manager.block_pool.get_num_free_blocks()

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=2,
            computed_tokens_by_request={"req-a": 10},
        )

        self.assertEqual(released, 2)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual(
            manager.get_reflex_block_ids("req-a"),
            (
                demotions[0].encoded_block_table_id,
                demotions[1].encoded_block_table_id,
                old_block_ids[2],
                old_block_ids[3],
                old_block_ids[4],
            ),
        )
        self.assertEqual(manager.block_pool.get_num_free_blocks(), free_before)

        self.assertEqual([d.page_idx for d in demotions], [0, 1])
        self.assertEqual([d.bf16_block_id for d in demotions],
                         old_block_ids[:2])
        self.assertTrue(
            all(
                decode_block_table_entry(d.encoded_block_table_id)[0]
                == PrecisionState.INT4
                for d in demotions
            )
        )
        manager.new_step_starts()
        self.assertEqual(
            manager.block_pool.get_num_free_blocks(), free_before + 2
        )

    def test_pending_bf16_release_is_not_reused_in_same_step(self):
        manager = self._make_manager()
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=10, num_tokens_main_model=10
        )
        old_block_ids = [block.block_id for block in new_blocks]

        manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=2,
            computed_tokens_by_request={"req-a": 10},
        )
        req_b_blocks = manager.allocate_new_blocks(
            "req-b", num_tokens=4, num_tokens_main_model=4
        )

        self.assertEqual(
            [block.block_id for block in req_b_blocks],
            [6, 7],
        )
        self.assertNotIn(old_block_ids[0], [block.block_id for block in req_b_blocks])
        self.assertNotIn(old_block_ids[1], [block.block_id for block in req_b_blocks])

    def test_kv_cache_blocks_can_return_reflex_encoded_ids(self):
        manager = self._make_manager()
        manager.allocate_new_blocks("req-a", num_tokens=8, num_tokens_main_model=8)
        manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=1,
            computed_tokens_by_request={"req-a": 8},
        )

        kv_blocks = KVCacheBlocks(
            blocks=(manager.req_to_blocks["req-a"],),
            block_id_overrides=(list(manager.get_reflex_block_ids("req-a")),),
        )

        self.assertTrue(all(block_id < 0 for block_id in kv_blocks.get_block_ids()[0][:2]))

    def test_manager_releases_int4_blocks_when_request_is_freed(self):
        manager = self._make_manager()
        manager.allocate_new_blocks("req-a", num_tokens=8, num_tokens_main_model=8)
        manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=1,
            computed_tokens_by_request={"req-a": 8},
        )
        assert manager.reflex_int4_pool is not None
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 2)

        manager.free("req-a")

        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 0)

    def test_manager_does_not_reserve_int4_blocks_for_live_bf16_pages(self):
        manager = self._make_manager(num_gpu_blocks=10)
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )

        assert manager.reflex_int4_pool is not None
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 0)

    def test_manager_does_not_demote_allocated_but_uncomputed_pages(self):
        manager = self._make_manager()
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        old_block_ids = [block.block_id for block in new_blocks]

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 4},
        )

        self.assertEqual(released, 2)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual(
            manager.get_reflex_block_ids("req-a"),
            (
                demotions[0].encoded_block_table_id,
                demotions[1].encoded_block_table_id,
                old_block_ids[2],
                old_block_ids[3],
            ),
        )

    def test_manager_does_not_demote_active_prefill_pages(self):
        manager = self._make_manager()
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        old_block_ids = [block.block_id for block in new_blocks]

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 4},
            prompt_tokens_by_request={"req-a": 8},
        )

        self.assertEqual(released, 0)
        self.assertEqual(
            manager.get_reflex_block_ids("req-a"),
            tuple(old_block_ids),
        )

    def test_manager_demotes_after_prefill_is_complete(self):
        manager = self._make_manager()
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        old_block_ids = [block.block_id for block in new_blocks]

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            prompt_tokens_by_request={"req-a": 8},
        )

        self.assertEqual(released, 4)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual(
            manager.get_reflex_block_ids("req-a"),
            tuple(d.encoded_block_table_id for d in demotions),
        )
        self.assertEqual([d.bf16_block_id for d in demotions], old_block_ids)

    def test_manager_marks_demoted_prompt_pages_in_trace_metadata(self):
        manager = self._make_manager()
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            prompt_tokens_by_request={"req-a": 4},
        )

        self.assertEqual(released, 4)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual([demotion.page_idx for demotion in demotions], [2, 3, 0, 1])
        self.assertEqual(
            [demotion.is_prompt_page for demotion in demotions],
            [False, False, True, True],
        )
        self.assertEqual(
            [demotion.prompt_pages for demotion in demotions],
            [2, 2, 2, 2],
        )

    def test_manager_reports_prompt_pages_per_demoted_request(self):
        manager = self._make_manager(num_gpu_blocks=16)
        manager.allocate_new_blocks(
            "short-req", num_tokens=4, num_tokens_main_model=4
        )
        manager.allocate_new_blocks(
            "long-req", num_tokens=8, num_tokens_main_model=8
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=6,
            keep_recent_pages=0,
            computed_tokens_by_request={
                "short-req": 4,
                "long-req": 8,
            },
            prompt_tokens_by_request={
                "short-req": 4,
                "long-req": 8,
            },
        )

        self.assertEqual(released, 6)
        demotions = manager.take_reflex_int4_demotions()
        prompt_pages_by_request = {
            demotion.request_id: demotion.prompt_pages
            for demotion in demotions
        }
        self.assertEqual(prompt_pages_by_request["short-req"], 2)
        self.assertEqual(prompt_pages_by_request["long-req"], 4)

    def test_manager_demotes_cacheable_unshared_blocks_with_hash(self):
        manager = self._make_manager(enable_caching=True)
        blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        for page_idx, block in enumerate(blocks):
            block.block_hash = ("test-hash", page_idx)

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
        )

        self.assertEqual(released, 2)
        self.assertEqual(
            manager.get_last_reflex_int4_candidate_breakdown()
            .eligible_full_unshared_pages,
            4,
        )

    def test_dynamic_demote_does_not_skip_live_bf16_numeric_ids(self):
        manager = self._make_manager(num_gpu_blocks=8)
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
        )

        self.assertEqual(released, 2)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual([d.int4_block_id for d in demotions], [0, 1])
        self.assertIn(1, [d.bf16_block_id for d in demotions])

    def test_manager_does_not_demote_protected_prefill_request(self):
        manager = self._make_manager()
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        old_block_ids = [block.block_id for block in new_blocks]

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            prompt_tokens_by_request={"req-a": 8},
            protected_request_ids={"req-a"},
        )

        self.assertEqual(released, 0)
        self.assertEqual(
            manager.get_reflex_block_ids("req-a"),
            tuple(old_block_ids),
        )

    def test_manager_demotes_sealed_chunk_pages_before_full_prompt_loaded(self):
        manager = self._make_manager(num_gpu_blocks=12)
        manager.allocate_new_blocks(
            "req-a", num_tokens=12, num_tokens_main_model=12
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            prompt_tokens_by_request={"req-a": 20},
            allow_partial_prefill_demotion_request_ids={"req-a"},
        )

        self.assertEqual(released, 4)
        self.assertEqual(
            manager.get_last_reflex_int4_candidate_breakdown()
            .eligible_full_unshared_pages,
            4,
        )
        self.assertEqual(
            [demotion.page_idx for demotion in manager.take_reflex_int4_demotions()],
            [0, 1, 2, 3],
        )

    def test_manager_uses_remote_chunk_sealed_page_frontier(self):
        manager = self._make_manager(num_gpu_blocks=12)
        manager.allocate_new_blocks(
            "req-a", num_tokens=12, num_tokens_main_model=12
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 0},
            prompt_tokens_by_request={"req-a": 20},
            sealed_pages_by_request={"req-a": 4},
        )

        self.assertEqual(released, 4)
        self.assertEqual(
            manager.get_last_reflex_int4_candidate_breakdown()
            .eligible_full_unshared_pages,
            4,
        )
        self.assertEqual(
            [demotion.page_idx for demotion in manager.take_reflex_int4_demotions()],
            [0, 1, 2, 3],
        )

    def test_manager_copy_on_demote_shared_page_releases_only_request_ref(self):
        manager = self._make_manager(num_gpu_blocks=8)
        blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=4, num_tokens_main_model=4
        )
        blocks[0].ref_cnt = 2

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=1,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 4},
            copy_on_demote_pages_by_request={"req-a": {0}},
        )

        self.assertEqual(released, 1)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual([demotion.page_idx for demotion in demotions], [0])
        self.assertTrue(demotions[0].copy_on_demote)
        manager.new_step_starts()
        self.assertEqual(blocks[0].ref_cnt, 1)

    def test_manager_allows_decode_generated_pages_with_low_risk_prompt_mask(self):
        manager = self._make_manager(num_gpu_blocks=8)
        manager.allocate_new_blocks(
            "req-a", num_tokens=12, num_tokens_main_model=12
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=5,
            keep_recent_pages=0,
            low_risk_only=True,
            computed_tokens_by_request={"req-a": 12},
            prompt_tokens_by_request={"req-a": 4},
            prefill_page_risks_by_request={"req-a": [0.9, 0.1]},
            compressible_pages_by_request={"req-a": {1}},
        )

        self.assertEqual(released, 5)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual([d.page_idx for d in demotions], [2, 3, 4, 5, 1])

    def test_manager_protects_reasoning_prompt_pages_but_allows_decode_pages(self):
        manager = self._make_manager(num_gpu_blocks=10)
        manager.allocate_new_blocks(
            "req-a", num_tokens=12, num_tokens_main_model=12
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=6,
            keep_recent_pages=0,
            low_risk_only=True,
            computed_tokens_by_request={"req-a": 12},
            prompt_tokens_by_request={"req-a": 4},
            protected_prompt_pages_by_request={"req-a": 2},
            prefill_page_risks_by_request={"req-a": [0.1, 0.1]},
            compressible_pages_by_request={"req-a": {0, 1}},
        )

        self.assertEqual(released, 4)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual([demotion.page_idx for demotion in demotions], [2, 3, 4, 5])
        self.assertTrue(all(not demotion.is_prompt_page for demotion in demotions))

    def test_manager_demotion_is_limited_by_int4_pool_capacity(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=2,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
        )

        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual(released, 2)
        self.assertEqual(len(demotions), 2)
        self.assertEqual([d.int4_block_id for d in demotions], [0, 1])
        assert manager.reflex_int4_pool is not None
        self.assertEqual(manager.reflex_int4_pool.num_free_blocks, 0)

    def test_manager_dry_run_does_not_patch_page_table_or_reserve_int4(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=2,
        )
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        old_block_ids = tuple(block.block_id for block in new_blocks)
        assert manager.reflex_int4_pool is not None

        feasible_release = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            dry_run=True,
        )

        self.assertEqual(feasible_release, 2)
        self.assertEqual(manager.take_reflex_int4_demotions(), [])
        self.assertEqual(manager.get_reflex_block_ids("req-a"), old_block_ids)
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 0)
        self.assertEqual(manager.reflex_int4_pool.num_free_blocks, 2)

    def test_manager_reserves_and_frees_landing_int4_blocks(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=3,
        )
        assert manager.reflex_int4_pool is not None

        reserved = manager.reserve_reflex_int4_landing_blocks("req-a", 2)

        self.assertEqual(reserved, [0, 1])
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 2)
        manager.free("req-a")
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 0)
        self.assertEqual(manager.reflex_int4_pool.num_free_blocks, 3)

    def test_manager_landing_reservation_is_atomic_when_pool_is_short(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=1,
        )
        assert manager.reflex_int4_pool is not None

        reserved = manager.reserve_reflex_int4_landing_blocks("req-a", 2)

        self.assertEqual(reserved, [])
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 0)
        self.assertEqual(manager.reflex_int4_pool.num_free_blocks, 1)

    def test_manager_commits_landing_pages_to_int4_block_table(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=4,
        )
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        old_block_ids = [block.block_id for block in new_blocks]
        reserved = manager.reserve_reflex_int4_landing_blocks("req-a", 2)
        free_before_commit = manager.block_pool.get_num_free_blocks()

        committed = manager.commit_reflex_int4_landing_pages(
            "req-a",
            page_indices=[1, 3],
            int4_block_ids=reserved,
        )

        self.assertEqual(committed, 2)
        block_ids = manager.get_reflex_block_ids("req-a")
        self.assertEqual(block_ids[0], old_block_ids[0])
        self.assertEqual(decode_block_table_entry(block_ids[1]),
                         (PrecisionState.INT4, reserved[0]))
        self.assertEqual(block_ids[2], old_block_ids[2])
        self.assertEqual(decode_block_table_entry(block_ids[3]),
                         (PrecisionState.INT4, reserved[1]))
        self.assertEqual(manager.req_to_blocks["req-a"][1], manager._null_block)
        self.assertEqual(manager.req_to_blocks["req-a"][3], manager._null_block)
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 2)
        self.assertEqual(manager.req_to_reflex_landing_int4_ids["req-a"], [])
        self.assertEqual(
            manager.block_pool.get_num_free_blocks(),
            free_before_commit,
        )

        manager.new_step_starts()

        self.assertEqual(
            manager.block_pool.get_num_free_blocks(),
            free_before_commit + 2,
        )
        manager.free("req-a")
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 0)

    def test_manager_landing_pages_keep_bf16_staging_blocks_until_commit(self):
        manager = self._make_manager(
            num_gpu_blocks=6,
            reflex_int4_num_blocks=4,
        )
        reserved = manager.reserve_reflex_int4_landing_blocks(
            "req-a",
            2,
            page_indices=[1, 3],
        )

        needed_blocks = manager.get_num_blocks_to_allocate(
            request_id="req-a",
            num_tokens=8,
            new_computed_blocks=[],
            total_computed_tokens=8,
            num_tokens_main_model=8,
        )

        self.assertEqual(needed_blocks, 4)

        free_before_alloc = manager.block_pool.get_num_free_blocks()
        manager.allocate_new_computed_blocks(
            request_id="req-a",
            new_computed_blocks=[],
            num_local_computed_tokens=0,
            num_external_computed_tokens=8,
        )

        self.assertEqual(manager.block_pool.get_num_free_blocks(),
                         free_before_alloc - 4)
        self.assertNotEqual(manager.req_to_blocks["req-a"][0], manager._null_block)
        self.assertNotEqual(manager.req_to_blocks["req-a"][1], manager._null_block)
        self.assertNotEqual(manager.req_to_blocks["req-a"][2], manager._null_block)
        self.assertNotEqual(manager.req_to_blocks["req-a"][3], manager._null_block)

        committed = manager.commit_reflex_int4_landing_pages(
            "req-a",
            page_indices=[1, 3],
            int4_block_ids=reserved,
        )

        self.assertEqual(committed, 2)
        block_ids = manager.get_reflex_block_ids("req-a")
        self.assertEqual(decode_block_table_entry(block_ids[1]),
                         (PrecisionState.INT4, reserved[0]))
        self.assertEqual(decode_block_table_entry(block_ids[3]),
                         (PrecisionState.INT4, reserved[1]))
        self.assertEqual(manager.block_pool.get_num_free_blocks(),
                         free_before_alloc - 4)
        manager.new_step_starts()
        self.assertEqual(manager.block_pool.get_num_free_blocks(),
                         free_before_alloc - 2)
        self.assertEqual(manager.check_reflex_int4_invariants("req-a"), [])

    def test_manager_direct_landing_pages_do_not_allocate_bf16_staging(self):
        manager = self._make_manager(
            num_gpu_blocks=6,
            reflex_int4_num_blocks=4,
        )
        reserved = manager.reserve_reflex_int4_landing_blocks(
            "req-a",
            2,
            page_indices=[1, 3],
        )
        manager.mark_reflex_int4_direct_landing_pages(
            "req-a",
            page_indices=[1, 3],
            int4_block_ids=reserved,
        )

        needed_blocks = manager.get_num_blocks_to_allocate(
            request_id="req-a",
            num_tokens=8,
            new_computed_blocks=[],
            total_computed_tokens=8,
            num_tokens_main_model=8,
        )

        self.assertEqual(needed_blocks, 2)

        free_before_alloc = manager.block_pool.get_num_free_blocks()
        manager.allocate_new_computed_blocks(
            request_id="req-a",
            new_computed_blocks=[],
            num_local_computed_tokens=0,
            num_external_computed_tokens=8,
        )

        self.assertEqual(manager.block_pool.get_num_free_blocks(),
                         free_before_alloc - 2)
        self.assertNotEqual(manager.req_to_blocks["req-a"][0],
                            manager._null_block)
        self.assertEqual(manager.req_to_blocks["req-a"][1],
                         manager._null_block)
        self.assertNotEqual(manager.req_to_blocks["req-a"][2],
                            manager._null_block)
        self.assertEqual(manager.req_to_blocks["req-a"][3],
                         manager._null_block)
        block_ids = manager.get_reflex_block_ids("req-a")
        self.assertEqual(decode_block_table_entry(block_ids[1]),
                         (PrecisionState.INT4, reserved[0]))
        self.assertEqual(decode_block_table_entry(block_ids[3]),
                         (PrecisionState.INT4, reserved[1]))

        committed = manager.commit_reflex_int4_landing_pages(
            "req-a",
            page_indices=[1, 3],
            int4_block_ids=reserved,
        )

        self.assertEqual(committed, 2)
        self.assertEqual(manager.block_pool.get_num_free_blocks(),
                         free_before_alloc - 2)
        self.assertEqual(manager.check_reflex_int4_invariants("req-a"), [])

    def test_manager_reports_landing_commit_lifecycle_counts_and_invariants(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=4,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        reserved = manager.reserve_reflex_int4_landing_blocks("req-a", 2)

        manager.commit_reflex_int4_landing_pages(
            "req-a",
            page_indices=[1, 3],
            int4_block_ids=reserved,
        )

        self.assertEqual(
            manager.get_reflex_precision_state_counts("req-a"),
            {
                "BF16_ACTIVE": 2,
                "INT4_ACTIVE": 2,
                "INT4_RECOVERABLE": 0,
                "BF16_RECOVERED": 0,
                "RELEASE_PENDING": 2,
                "LANDING_RESERVED": 0,
            },
        )
        self.assertEqual(manager.check_reflex_int4_invariants("req-a"), [])

    def test_manager_invariant_detects_corrupt_bf16_page_table_entry(self):
        manager = self._make_manager()
        manager.allocate_new_blocks(
            "req-a", num_tokens=4, num_tokens_main_model=4
        )
        manager._ensure_reflex_block_ids("req-a")
        manager.req_to_blocks["req-a"][0] = manager._null_block

        violations = manager.check_reflex_int4_invariants("req-a")

        self.assertTrue(violations)
        self.assertIn("BF16", violations[0])

    def test_manager_exposes_precision_elastic_page_descriptors(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=4,
        )
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        reserved = manager.reserve_reflex_int4_landing_blocks(
            "req-a",
            2,
            page_indices=[1, 3],
        )

        descriptors = manager.get_reflex_page_runtime_descriptors(
            "req-a",
            computed_tokens_by_request={"req-a": 8},
            prompt_tokens_by_request={"req-a": 8},
            prefill_page_risks_by_request={"req-a": [0.9, 0.1, 0.4, 0.2]},
            compressible_pages_by_request={"req-a": {1, 3}},
            keep_recent_pages=0,
            keep_initial_pages=0,
        )

        self.assertEqual(len(descriptors), 4)
        self.assertEqual(descriptors[0].precision, PrecisionState.BF16)
        self.assertEqual(descriptors[0].lifecycle, KVPageLifecycle.BF16_ACTIVE)
        self.assertEqual(descriptors[0].tier, MemoryTier.GPU)
        self.assertEqual(descriptors[0].physical_block_id, new_blocks[0].block_id)
        self.assertEqual(descriptors[1].precision, PrecisionState.BF16)
        self.assertEqual(descriptors[1].planned_precision, PrecisionState.INT4)
        self.assertEqual(descriptors[1].lifecycle, KVPageLifecycle.INT4_LANDING)
        self.assertEqual(descriptors[1].landing_int4_block_id, reserved[0])
        self.assertEqual(descriptors[1].risk_score, 0.1)
        self.assertTrue(descriptors[1].is_low_risk)
        self.assertFalse(descriptors[1].is_recent_protected)

        manager.commit_reflex_int4_landing_pages(
            "req-a",
            page_indices=[1, 3],
            int4_block_ids=reserved,
        )
        pending = manager.get_reflex_page_runtime_descriptors("req-a")

        self.assertEqual(pending[1].precision, PrecisionState.INT4)
        self.assertEqual(pending[1].lifecycle, KVPageLifecycle.RELEASE_PENDING)
        self.assertTrue(pending[1].bf16_release_pending)

        manager.new_step_starts()
        active = manager.get_reflex_page_runtime_descriptors("req-a")

        self.assertEqual(active[1].precision, PrecisionState.INT4)
        self.assertEqual(active[1].lifecycle, KVPageLifecycle.INT4_ACTIVE)
        self.assertFalse(active[1].bf16_release_pending)

    def test_manager_rejects_unreserved_landing_int4_commit(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=4,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )

        with self.assertRaisesRegex(RuntimeError, "is not reserved"):
            manager.commit_reflex_int4_landing_pages(
                "req-a",
                page_indices=[1],
                int4_block_ids=[2],
            )

    def test_manager_protects_initial_and_recent_pages(self):
        manager = self._make_manager(num_gpu_blocks=8)
        new_blocks = manager.allocate_new_blocks(
            "req-a", num_tokens=12, num_tokens_main_model=12
        )
        old_block_ids = [block.block_id for block in new_blocks]

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=6,
            keep_initial_pages=2,
            keep_recent_pages=1,
            computed_tokens_by_request={"req-a": 12},
        )

        self.assertEqual(released, 3)
        demotions = manager.take_reflex_int4_demotions()
        self.assertEqual([d.page_idx for d in demotions], [2, 3, 4])
        self.assertEqual(
            manager.get_reflex_block_ids("req-a"),
            (
                old_block_ids[0],
                old_block_ids[1],
                demotions[0].encoded_block_table_id,
                demotions[1].encoded_block_table_id,
                demotions[2].encoded_block_table_id,
                old_block_ids[5],
            ),
        )

    def test_manager_limits_per_request_int4_fraction(self):
        manager = self._make_manager(num_gpu_blocks=10)
        manager.allocate_new_blocks(
            "req-a", num_tokens=16, num_tokens_main_model=16
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=8,
            keep_recent_pages=0,
            max_int4_fraction_per_request=0.5,
            computed_tokens_by_request={"req-a": 16},
        )

        self.assertEqual(released, 4)
        self.assertEqual(
            sum(
                1
                for entry in manager.get_reflex_block_ids("req-a")
                if decode_block_table_entry(entry)[0] == PrecisionState.INT4
            ),
            4,
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=4,
            keep_recent_pages=0,
            max_int4_fraction_per_request=0.5,
            computed_tokens_by_request={"req-a": 16},
        )

        self.assertEqual(released, 0)

    def test_manager_marks_selected_demotions_as_recoverable_bf16_shadow(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=4,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )

        released = manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            recovery_shadow_pages_by_request={"req-a": {0}},
        )

        self.assertEqual(released, 2)
        manager.take_reflex_int4_demotions()
        pending = manager.get_reflex_page_runtime_descriptors("req-a")
        self.assertEqual(pending[0].precision, PrecisionState.INT4)
        self.assertEqual(pending[0].recovery_class, RecoveryClass.BF16_SHADOW)
        self.assertTrue(pending[0].has_recovery_artifact)
        self.assertEqual(pending[1].recovery_class, RecoveryClass.NONE)
        self.assertFalse(pending[1].has_recovery_artifact)

        manager.new_step_starts()
        active = manager.get_reflex_page_runtime_descriptors("req-a")
        self.assertEqual(
            active[0].lifecycle,
            KVPageLifecycle.INT4_ACTIVE_RECOVERABLE,
        )
        self.assertEqual(active[1].lifecycle, KVPageLifecycle.INT4_ACTIVE)

    def test_manager_recovers_int4_page_from_bf16_shadow(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=4,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )

        manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            recovery_shadow_pages_by_request={"req-a": {0}},
        )
        demotions = manager.take_reflex_int4_demotions()
        manager.new_step_starts()
        assert manager.reflex_int4_pool is not None
        self.assertEqual(manager.reflex_int4_pool.num_allocated_blocks, 2)

        recovered = manager.recover_reflex_int4_pages("req-a", [0])

        self.assertEqual(recovered, 1)
        block_ids = manager.get_reflex_block_ids("req-a")
        self.assertEqual(
            decode_block_table_entry(block_ids[0])[0],
            PrecisionState.BF16,
        )
        self.assertEqual(
            manager.reflex_int4_pool.num_allocated_blocks,
            1,
        )
        descriptors = manager.get_reflex_page_runtime_descriptors("req-a")
        self.assertEqual(descriptors[0].precision, PrecisionState.BF16)
        self.assertEqual(
            descriptors[0].lifecycle,
            KVPageLifecycle.BF16_RECOVERED,
        )
        self.assertFalse(descriptors[0].has_recovery_artifact)
        recoveries = manager.take_reflex_int4_recoveries()
        self.assertEqual(len(recoveries), 1)
        self.assertEqual(recoveries[0].request_id, "req-a")
        self.assertEqual(recoveries[0].page_idx, 0)
        self.assertEqual(recoveries[0].int4_block_id, demotions[0].int4_block_id)
        self.assertEqual(recoveries[0].bf16_block_id, descriptors[0].bf16_block_id)
        self.assertEqual(recoveries[0].recovery_class, RecoveryClass.BF16_SHADOW)
        self.assertEqual(manager.check_reflex_int4_invariants("req-a"), [])

    def test_manager_does_not_recover_int4_page_without_shadow(self):
        manager = self._make_manager(
            num_gpu_blocks=8,
            reflex_int4_num_blocks=4,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            recovery_shadow_pages_by_request={"req-a": {0}},
        )
        manager.take_reflex_int4_demotions()
        manager.new_step_starts()
        assert manager.reflex_int4_pool is not None
        allocated_before = manager.reflex_int4_pool.num_allocated_blocks

        recovered = manager.recover_reflex_int4_pages("req-a", [1])

        self.assertEqual(recovered, 0)
        self.assertEqual(
            manager.reflex_int4_pool.num_allocated_blocks,
            allocated_before,
        )
        self.assertEqual(manager.take_reflex_int4_recoveries(), [])

    def test_manager_has_no_relevance_hit_recovery_planner(self):
        manager = self._make_manager(
            num_gpu_blocks=10,
            reflex_int4_num_blocks=4,
        )

        self.assertFalse(
            hasattr(manager, "plan_reflex_precision_fault_recoveries")
        )

    def test_manager_background_promotes_high_risk_recoverable_pages(self):
        manager = self._make_manager(
            num_gpu_blocks=12,
            reflex_int4_num_blocks=6,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=12, num_tokens_main_model=12
        )
        manager.plan_reflex_int4_demotions(
            target_bf16_blocks=3,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 12},
            recovery_shadow_pages_by_request={"req-a": {0, 1, 2}},
        )
        manager.take_reflex_int4_demotions()
        manager.new_step_starts()

        promoted = manager.promote_reflex_recoverable_pages(
            max_pages=2,
            prefill_page_risks_by_request={"req-a": [0.1, 0.9, 0.6]},
            remaining_decode_tokens_by_request={"req-a": 64},
            min_remaining_decode_tokens=16,
        )

        self.assertEqual(promoted, 2)
        recoveries = manager.take_reflex_int4_recoveries()
        self.assertEqual([recovery.page_idx for recovery in recoveries], [1, 2])
        descriptors = manager.get_reflex_page_runtime_descriptors("req-a")
        self.assertEqual(
            descriptors[1].lifecycle,
            KVPageLifecycle.BF16_RECOVERED,
        )
        self.assertEqual(
            descriptors[2].lifecycle,
            KVPageLifecycle.BF16_RECOVERED,
        )
        self.assertEqual(
            descriptors[0].lifecycle,
            KVPageLifecycle.INT4_ACTIVE_RECOVERABLE,
        )

    def test_manager_background_promotion_skips_almost_finished_requests(self):
        manager = self._make_manager(
            num_gpu_blocks=10,
            reflex_int4_num_blocks=4,
        )
        manager.allocate_new_blocks(
            "req-a", num_tokens=8, num_tokens_main_model=8
        )
        manager.plan_reflex_int4_demotions(
            target_bf16_blocks=2,
            keep_recent_pages=0,
            computed_tokens_by_request={"req-a": 8},
            recovery_shadow_pages_by_request={"req-a": {0, 1}},
        )
        manager.take_reflex_int4_demotions()
        manager.new_step_starts()

        promoted = manager.promote_reflex_recoverable_pages(
            max_pages=2,
            prefill_page_risks_by_request={"req-a": [0.9, 0.8]},
            remaining_decode_tokens_by_request={"req-a": 1},
            min_remaining_decode_tokens=16,
        )

        self.assertEqual(promoted, 0)
        self.assertEqual(manager.take_reflex_int4_recoveries(), [])


if __name__ == "__main__":
    unittest.main()
