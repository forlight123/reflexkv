from types import SimpleNamespace

import torch

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorMetadata,
    MooncakeConnectorScheduler,
    MooncakeConnectorWorker,
    MooncakeConnectorWorkerMetadata,
    MooncakeXferResponse,
    MooncakeXferResponseStatus,
    MooncakeXferMetadata,
    PullReqMeta,
)
import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector as mooncake_module
from vllm.distributed.kv_transfer.kv_connector.v1.reflex_mooncake_connector import (
    ReFlexMooncakeConnector,
)
from vllm.v1.attention.ops.reflex_int4_codec import ReflexInt4Codec
from vllm.v1.outputs import KVConnectorOutput


class _FakeBlocks:

    def __init__(self, block_ids):
        self._block_ids = list(block_ids)
        self.blocks = [self._block_ids]

    def get_unhashed_block_ids(self):
        return list(self._block_ids)

    def get_block_ids(self):
        return (list(self._block_ids),)


class _FakeRequest:

    def __init__(
        self,
        request_id="decode-req",
        *,
        prompt_tokens=2048,
        num_computed_tokens=0,
        kv_transfer_params=None,
    ):
        self.request_id = request_id
        self.prompt_token_ids = list(range(prompt_tokens))
        self.num_prompt_tokens = prompt_tokens
        self.num_tokens = prompt_tokens
        self.num_computed_tokens = num_computed_tokens
        self.kv_transfer_params = kv_transfer_params or {}


def test_reflex_mooncake_connector_is_registered():
    connector_cls = KVConnectorFactory.get_connector_class_by_name(
        "ReFlexMooncakeConnector"
    )

    assert connector_cls is ReFlexMooncakeConnector


def test_mooncake_decode_remote_prefill_returns_one_reflex_chunk_at_a_time():
    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler.is_kv_producer = False
    scheduler.is_kv_consumer = True
    scheduler.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16)
    )
    request = _FakeRequest(
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "xfer",
            "remote_engine_id": "prefill",
            "remote_bootstrap_addr": "http://127.0.0.1:8998",
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_tokens": 512,
        }
    )

    count, async_load = scheduler.get_num_new_matched_tokens(request, 0)

    assert (count, async_load) == (512, True)
    assert "reflex_remote_chunk_id" not in request.kv_transfer_params

    scheduler._reqs_need_recv = {}
    scheduler.update_state_after_alloc(request, _FakeBlocks(range(32)), count)

    params = request.kv_transfer_params
    assert params["do_remote_prefill"] is True
    assert params["reflex_remote_chunk_id"] == 0
    assert params["reflex_remote_chunk_token_start"] == 0
    assert params["reflex_remote_chunk_token_end"] == 512
    assert params["reflex_remote_chunk_page_start"] == 0
    assert params["reflex_remote_chunk_page_end"] == 32
    assert params["reflex_remote_chunk_is_last"] is False
    assert scheduler._reqs_need_recv["decode-req"][1] == list(range(32))


def test_mooncake_decode_remote_prefill_slices_target_blocks_to_chunk_pages():
    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler.is_kv_producer = False
    scheduler.is_kv_consumer = True
    scheduler.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16)
    )
    scheduler._reqs_need_recv = {}
    request = _FakeRequest(
        num_computed_tokens=256,
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "xfer",
            "remote_engine_id": "prefill",
            "remote_bootstrap_addr": "http://127.0.0.1:8998",
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_tokens": 256,
        },
    )

    count, async_load = scheduler.get_num_new_matched_tokens(request, 256)

    assert (count, async_load) == (256, True)
    scheduler.update_state_after_alloc(request, _FakeBlocks(range(64)), count)

    params = request.kv_transfer_params
    assert params["reflex_remote_chunk_id"] == 1
    assert params["reflex_remote_chunk_page_start"] == 16
    assert params["reflex_remote_chunk_page_end"] == 32
    assert scheduler._reqs_need_recv["decode-req"][1] == list(range(16, 32))


def test_mooncake_decode_remote_prefill_clears_flag_after_final_chunk():
    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler.is_kv_producer = False
    scheduler.is_kv_consumer = True
    scheduler.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16)
    )
    scheduler._reqs_need_recv = {}
    request = _FakeRequest(
        num_computed_tokens=1536,
        kv_transfer_params={
            "do_remote_prefill": True,
            "transfer_id": "xfer",
            "remote_engine_id": "prefill",
            "remote_bootstrap_addr": "http://127.0.0.1:8998",
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_tokens": 512,
        },
    )

    count, async_load = scheduler.get_num_new_matched_tokens(request, 1536)
    scheduler.update_state_after_alloc(request, _FakeBlocks(range(96, 128)), count)

    assert (count, async_load) == (512, True)
    assert request.kv_transfer_params["reflex_remote_chunk_id"] == 3
    assert request.kv_transfer_params["reflex_remote_chunk_is_last"] is True
    assert request.kv_transfer_params["do_remote_prefill"] is False


def test_mooncake_metadata_carries_reflex_remote_chunk_contract():
    metadata = MooncakeConnectorMetadata()

    metadata.add_new_req(
        request_id="decode-req",
        local_block_ids=[3, 4],
        kv_transfer_params={
            "transfer_id": "xfer-1",
            "remote_engine_id": "prefill-engine",
            "remote_bootstrap_addr": "http://127.0.0.1:8998",
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_id": 2,
            "reflex_remote_chunk_token_start": 1024,
            "reflex_remote_chunk_token_end": 1536,
            "reflex_remote_chunk_page_start": 64,
            "reflex_remote_chunk_page_end": 96,
            "reflex_remote_chunk_is_last": False,
        },
    )

    pull_meta = metadata.reqs_to_recv["prefill-engine"]["decode-req"]
    assert pull_meta.reflex_remote_chunk_id == 2
    assert pull_meta.reflex_remote_chunk_token_start == 1024
    assert pull_meta.reflex_remote_chunk_token_end == 1536
    assert pull_meta.reflex_remote_chunk_page_start == 64
    assert pull_meta.reflex_remote_chunk_page_end == 96
    assert pull_meta.reflex_remote_chunk_is_last is False


def test_mooncake_producer_records_deferred_reflex_chunk_send():
    scheduler = object.__new__(MooncakeConnectorScheduler)
    scheduler.is_kv_producer = True
    scheduler.is_kv_consumer = False
    scheduler.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16)
    )
    scheduler._reqs_need_recv = {}
    scheduler._reqs_need_send = {}
    scheduler._reqs_not_processed = set()
    request = _FakeRequest(
        request_id="prefill-req",
        prompt_tokens=2048,
        kv_transfer_params={
            "do_remote_decode": True,
            "transfer_id": "xfer-prefill",
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_tokens": 512,
        },
    )

    scheduler.update_reflex_remote_decode_chunk_after_alloc(
        request,
        _FakeBlocks(range(32)),
        num_scheduled_tokens=512,
    )
    meta = scheduler.build_connector_meta(None)

    assert "prefill-req" in meta.reqs_to_send
    assert meta.reqs_to_send["prefill-req"] == ("xfer-prefill", list(range(32)))
    assert meta.reflex_remote_chunk_ids == {"prefill-req": 0}
    assert meta.reflex_remote_chunk_defer_ready == {"prefill-req": True}
    assert meta.reflex_remote_chunk_release_after_send == {"prefill-req": False}


def test_mooncake_worker_defers_chunk_ready_until_wait_for_save():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.reqs_need_send = {}
    worker.finished_sending_reqs = set()
    worker._completed_send_keys = set()
    worker._deferred_ready_send_keys = set()
    metadata = MooncakeConnectorMetadata()
    metadata.add_new_req(
        request_id="prefill-req",
        local_block_ids=[10, 11],
        kv_transfer_params={
            "transfer_id": "xfer-prefill",
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_id": 0,
            "reflex_remote_chunk_token_start": 0,
            "reflex_remote_chunk_token_end": 32,
            "reflex_remote_chunk_page_start": 0,
            "reflex_remote_chunk_page_end": 2,
            "reflex_remote_chunk_is_last": False,
            "reflex_remote_chunk_defer_ready": True,
            "reflex_remote_chunk_release_after_send": False,
        },
        load_remote_cache=False,
    )

    mooncake_module.asyncio.run(worker.record_send_reqs(metadata))

    send_key = "xfer-prefill:reflex_chunk:0"
    assert send_key in worker.reqs_need_send
    assert worker.reqs_need_send[send_key].ready.is_set() is False
    assert worker.reqs_need_send[send_key].release_after_send is False

    worker.wait_for_save()

    assert worker.reqs_need_send[send_key].ready.is_set() is True


def test_mooncake_worker_release_marker_finishes_completed_chunk_send():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.reqs_need_send = {}
    worker.finished_sending_reqs = set()
    worker._completed_send_keys = {"xfer-prefill:reflex_chunk:3"}
    worker._deferred_ready_send_keys = set()
    metadata = MooncakeConnectorMetadata()
    metadata.add_new_req(
        request_id="prefill-req",
        local_block_ids=[10, 11],
        kv_transfer_params={
            "transfer_id": "xfer-prefill",
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_id": 3,
            "reflex_remote_chunk_token_start": 1536,
            "reflex_remote_chunk_token_end": 2048,
            "reflex_remote_chunk_page_start": 96,
            "reflex_remote_chunk_page_end": 128,
            "reflex_remote_chunk_is_last": True,
            "reflex_remote_chunk_release_after_send": True,
        },
        load_remote_cache=False,
    )

    mooncake_module.asyncio.run(worker.record_send_reqs(metadata))

    assert worker.finished_sending_reqs == {"prefill-req"}
    assert worker._completed_send_keys == set()
    assert worker.reqs_need_send == {}


def test_reflex_mooncake_connector_has_no_handoff_compression_path():
    connector = ReFlexMooncakeConnector.__new__(ReFlexMooncakeConnector)

    assert not hasattr(connector, "pop_reflex_remote_handoff_int4_blocks")


def test_mooncake_metadata_carries_reflex_landing_contract():
    metadata = MooncakeConnectorMetadata()

    metadata.add_new_req(
        request_id="decode-req",
        local_block_ids=[3, 4, 5, 6],
        kv_transfer_params={
            "transfer_id": "xfer-1",
            "remote_engine_id": "prefill-engine",
            "remote_bootstrap_addr": "http://127.0.0.1:8998",
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_landing_source_block_ids": [4, 6],
            "reflex_int4_landing_planned_blocks": 2,
        },
    )

    pull_meta = metadata.reqs_to_recv["prefill-engine"]["decode-req"]
    assert pull_meta.reflex_int4_landing_pages == [1, 3]
    assert pull_meta.reflex_int4_landing_block_ids == [9, 10]
    assert pull_meta.reflex_int4_landing_source_block_ids == [4, 6]
    assert pull_meta.reflex_int4_landing_planned_blocks == 2


def test_mooncake_metadata_carries_direct_landing_contract():
    metadata = MooncakeConnectorMetadata()

    metadata.add_new_req(
        request_id="decode-req",
        local_block_ids=[3, 5],
        kv_transfer_params={
            "transfer_id": "xfer-1",
            "remote_engine_id": "prefill-engine",
            "remote_bootstrap_addr": "http://127.0.0.1:8998",
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_direct_landing": True,
            "reflex_int4_landing_planned_blocks": 2,
        },
    )

    pull_meta = metadata.reqs_to_recv["prefill-engine"]["decode-req"]
    assert pull_meta.local_block_ids == [3, 5]
    assert pull_meta.reflex_int4_landing_pages == [1, 3]
    assert pull_meta.reflex_int4_landing_block_ids == [9, 10]
    assert pull_meta.reflex_int4_direct_landing is True


def test_mooncake_xfer_metadata_serializes_reflex_landing_contract():
    metadata = MooncakeXferMetadata(
        remote_hostname="127.0.0.1",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"decode-req": ("xfer-1", [3, 4])},
        kv_caches_base_addr=[1000],
        block_lens=[256],
        reflex_int4_landing_pages={"decode-req": [1, 3]},
        reflex_int4_landing_block_ids={"decode-req": [9, 10]},
        reflex_int4_direct_landing={"decode-req": True},
        reflex_int4_kv_caches_base_addr=[2000],
        reflex_int4_block_lens=[80],
    )

    assert metadata.reflex_int4_landing_pages == {"decode-req": [1, 3]}
    assert metadata.reflex_int4_landing_block_ids == {"decode-req": [9, 10]}
    assert metadata.reflex_int4_direct_landing == {"decode-req": True}
    assert metadata.reflex_int4_kv_caches_base_addr == [2000]
    assert metadata.reflex_int4_block_lens == [80]


def test_mooncake_worker_materializes_landing_pages_to_int4_sidecar(monkeypatch):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.device_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 8), dtype=torch.bfloat16),
        "layer.1": torch.empty((16, 2, 4, 1, 8), dtype=torch.bfloat16),
    }
    worker.reflex_int4_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 5), dtype=torch.uint8),
        "layer.1": torch.empty((16, 2, 4, 1, 5), dtype=torch.uint8),
    }
    calls = []

    def fake_quantize_blocks(
        self,
        src_cache,
        int4_cache,
        src_block_ids,
        dst_block_ids,
    ):
        calls.append(
            (
                src_cache,
                int4_cache,
                tuple(src_block_ids.cpu().tolist()),
                tuple(dst_block_ids.cpu().tolist()),
            )
        )

    monkeypatch.setattr(
        ReflexInt4Codec,
        "quantize_blocks_to_cache",
        fake_quantize_blocks,
    )

    pull_meta = PullReqMeta(
        d_req_id="decode-req",
        transfer_id="xfer-1",
        local_block_ids=[10, 11, 12, 13],
        remote_engine_id="prefill-engine",
        remote_bootstrap_addr="http://127.0.0.1:8998",
        reflex_int4_landing_pages=[1, 3],
        reflex_int4_landing_block_ids=[5, 6],
        reflex_int4_landing_source_block_ids=[21, 23],
    )

    materialized = worker._materialize_reflex_int4_landing_pages(pull_meta)

    assert materialized is True
    assert len(calls) == 2
    assert calls[0][2:] == ((21, 23), (5, 6))
    assert calls[1][2:] == ((21, 23), (5, 6))


def test_mooncake_worker_refuses_landing_without_explicit_source_blocks(
    monkeypatch,
):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.device_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 8), dtype=torch.bfloat16),
    }
    worker.reflex_int4_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 5), dtype=torch.uint8),
    }
    calls = []

    def fake_quantize_blocks(self, *args):
        calls.append(args)

    monkeypatch.setattr(
        ReflexInt4Codec,
        "quantize_blocks_to_cache",
        fake_quantize_blocks,
    )

    pull_meta = PullReqMeta(
        d_req_id="decode-req",
        transfer_id="xfer-1",
        local_block_ids=[10, 11],
        remote_engine_id="prefill-engine",
        remote_bootstrap_addr="http://127.0.0.1:8998",
        reflex_int4_landing_pages=[1, 3],
        reflex_int4_landing_block_ids=[5, 6],
    )

    materialized = worker._materialize_reflex_int4_landing_pages(pull_meta)

    assert materialized is False
    assert calls == []


def test_mooncake_worker_accepts_direct_landing_without_bf16_source_blocks(
    monkeypatch,
):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.device_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 8), dtype=torch.bfloat16),
    }
    worker.reflex_int4_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 5), dtype=torch.uint8),
    }
    calls = []

    def fake_quantize_blocks(self, *args):
        calls.append(args)

    monkeypatch.setattr(
        ReflexInt4Codec,
        "quantize_blocks_to_cache",
        fake_quantize_blocks,
    )

    pull_meta = PullReqMeta(
        d_req_id="decode-req",
        transfer_id="xfer-1",
        local_block_ids=[10, 12],
        remote_engine_id="prefill-engine",
        remote_bootstrap_addr="http://127.0.0.1:8998",
        reflex_int4_landing_pages=[1, 3],
        reflex_int4_landing_block_ids=[5, 6],
        reflex_int4_direct_landing=True,
    )

    materialized = worker._materialize_reflex_int4_landing_pages(pull_meta)

    assert materialized is True
    assert calls == []


def test_mooncake_worker_builds_direct_landing_int4_transfer(monkeypatch):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.device_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 8), dtype=torch.bfloat16),
    }
    worker.reflex_int4_kv_caches = {
        "layer.0": torch.empty((16, 2, 4, 1, 5), dtype=torch.uint8),
    }
    worker.reflex_int4_kv_caches_base_addr = [1000]
    worker.reflex_int4_block_len_per_layer = [80]
    worker.reflex_int4_num_blocks = 16
    worker._reflex_int4_direct_scratch_cursor = 0
    worker._get_transfer_regions = lambda base_addrs, block_lens: [
        mooncake_module.TransferRegion(base_addrs[0], block_lens[0], block_lens[0])
    ]
    worker._get_sender_transfer_plan = lambda **kwargs: (
        True,
        0,
        0,
        kwargs["local_kv_block_len"],
    )
    calls = []

    def fake_quantize_blocks(
        self,
        src_cache,
        int4_cache,
        src_block_ids,
        dst_block_ids,
    ):
        calls.append(
            (
                tuple(src_block_ids.cpu().tolist()),
                tuple(dst_block_ids.cpu().tolist()),
            )
        )

    monkeypatch.setattr(
        ReflexInt4Codec,
        "quantize_blocks_to_cache",
        fake_quantize_blocks,
    )
    metadata = MooncakeXferMetadata(
        remote_hostname="127.0.0.1",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"decode-req": ("xfer-1", [30, 32])},
        kv_caches_base_addr=[3000],
        block_lens=[256],
        reflex_int4_kv_caches_base_addr=[2000],
        reflex_int4_block_lens=[80],
    )
    src_ptrs = []
    dst_ptrs = []
    lengths = []

    err = worker._append_reflex_int4_direct_landing_transfers(
        d_req_id="decode-req",
        src_bf16_block_ids=[4, 6],
        dst_int4_block_ids=[9, 10],
        agent_meta=metadata,
        src_ptrs=src_ptrs,
        dst_ptrs=dst_ptrs,
        lengths=lengths,
    )

    assert err is None
    assert calls == [((4, 6), (0, 1))]
    assert src_ptrs == [1000]
    assert dst_ptrs == [2000 + 9 * 80]
    assert lengths == [160]


def test_mooncake_worker_builds_direct_landing_when_no_bf16_blocks():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    calls = []
    worker._append_reflex_int4_direct_landing_transfers = (
        lambda **kwargs: calls.append(
            (
                kwargs["src_bf16_block_ids"],
                kwargs["dst_int4_block_ids"],
            )
        )
        or None
    )
    send_meta = mooncake_module.SendBlockMeta(
        p_req_id="prefill-req",
        transfer_id="xfer-1",
        local_block_ids=[4, 5],
        ready=mooncake_module.asyncio.Event(),
    )
    metadata = MooncakeXferMetadata(
        remote_hostname="127.0.0.1",
        remote_port=1234,
        remote_tp_size=1,
        remote_tp_rank=0,
        req_blocks={"decode-req": ("xfer-1", [])},
        kv_caches_base_addr=[3000],
        block_lens=[256],
        reflex_int4_landing_pages={"decode-req": [0, 1]},
        reflex_int4_landing_block_ids={"decode-req": [9, 10]},
        reflex_int4_direct_landing={"decode-req": True},
        reflex_int4_kv_caches_base_addr=[2000],
        reflex_int4_block_lens=[80],
    )

    result = mooncake_module.asyncio.run(
        worker._build_transfer_params(
            [("decode-req", send_meta)],
            metadata,
            local_regions=[],
            remote_regions=[],
        )
    )

    assert result == ([], [], [], [], None)
    assert calls == [([4, 5], [9, 10])]


def test_mooncake_worker_process_pulling_result_materializes_after_last_pull(
    monkeypatch,
):
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.finished_recving_reqs = set()
    worker.reflex_int4_materialized_landing_reqs = set()
    materialized = []

    def fake_materialize(pull_meta):
        materialized.append(pull_meta.d_req_id)
        return True

    monkeypatch.setattr(
        worker,
        "_materialize_reflex_int4_landing_pages",
        fake_materialize,
    )
    pull_meta = PullReqMeta(
        d_req_id="decode-req",
        transfer_id="xfer-1",
        local_block_ids=[10, 11],
        remote_engine_id="prefill-engine",
        remote_bootstrap_addr="http://127.0.0.1:8998",
        reflex_int4_landing_pages=[1],
        reflex_int4_landing_block_ids=[5],
        pull_tasks_count=1,
    )

    worker.process_pulling_result(
        MooncakeXferResponse(
            status=MooncakeXferResponseStatus.FINISH,
            ok_reqs=["decode-req"],
        ),
        {"decode-req": pull_meta},
    )

    assert materialized == ["decode-req"]
    assert worker.finished_recving_reqs == {"decode-req"}
    assert worker.reflex_int4_materialized_landing_reqs == {"decode-req"}


def test_mooncake_worker_fetches_finished_and_materialized_landing_atomically():
    worker = object.__new__(MooncakeConnectorWorker)
    worker.shutdown = lambda: None
    worker.finished_recving_reqs = {"decode-req"}
    worker.reflex_int4_materialized_landing_reqs = {"decode-req"}

    finished, materialized = mooncake_module.asyncio.run(
        worker.fetch_finished_recving_and_reflex_int4_materialized_landing_reqs()
    )

    assert finished == {"decode-req"}
    assert materialized == {"decode-req"}
    assert worker.finished_recving_reqs == set()
    assert worker.reflex_int4_materialized_landing_reqs == set()


def test_mooncake_worker_metadata_merges_materialized_landing_reqs():
    left = MooncakeConnectorWorkerMetadata(
        reflex_page_risks_by_request={"req-a": [0.1]},
        reflex_int4_materialized_landing_req_ids={"req-a"},
    )
    right = MooncakeConnectorWorkerMetadata(
        reflex_page_risks_by_request={"req-b": [0.2]},
        reflex_int4_materialized_landing_req_ids={"req-b"},
    )

    merged = left.aggregate(right)

    assert merged.reflex_page_risks_by_request == {
        "req-a": [0.1],
        "req-b": [0.2],
    }
    assert merged.reflex_int4_materialized_landing_req_ids == {
        "req-a",
        "req-b",
    }


def test_kv_connector_output_merge_preserves_worker_metadata():
    left = KVConnectorOutput(
        kv_connector_worker_meta=MooncakeConnectorWorkerMetadata(
            reflex_page_risks_by_request={},
            reflex_int4_materialized_landing_req_ids={"req-a"},
        )
    )
    right = KVConnectorOutput(
        kv_connector_worker_meta=MooncakeConnectorWorkerMetadata(
            reflex_page_risks_by_request={},
            reflex_int4_materialized_landing_req_ids={"req-b"},
        )
    )

    merged = KVConnectorOutput.merge(left, right)

    assert isinstance(merged.kv_connector_worker_meta, MooncakeConnectorWorkerMetadata)
    assert merged.kv_connector_worker_meta.reflex_int4_materialized_landing_req_ids == {
        "req-a",
        "req-b",
    }
