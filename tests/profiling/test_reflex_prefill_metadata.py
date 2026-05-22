from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnectorWorkerMetadata,
)
from vllm.v1.core.reflex_prefill_metadata import (
    ReflexPrefillMetadataRecorder,
    ReflexPrefillRequestInfo,
    maybe_record_reflex_prefill_layer,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


class _FakeRecorder:
    def __init__(self):
        self.calls = []

    @contextmanager
    def batch(self, infos, *, block_size):
        self.calls.append((list(infos), block_size))
        yield


def test_reflex_prefill_recorder_scores_pages_from_query_key_anchors(monkeypatch):
    monkeypatch.setenv("SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA", "1")
    monkeypatch.setenv("SEMANTIQ_REFLEX_PREFILL_Q_TAIL_TOKENS", "4")

    recorder = ReflexPrefillMetadataRecorder()
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    key = torch.tensor(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    info = ReflexPrefillRequestInfo(
        request_id="req-a",
        query_start=0,
        query_end=4,
        prompt_start=0,
        prompt_end=4,
        prompt_tokens=4,
    )

    with recorder.batch([info], block_size=2):
        recorder.record_layer("model.layers.0.self_attn", query, key)

    scores = recorder.drain_completed_requests(block_size=2)["req-a"]

    assert len(scores) == 2
    assert scores[0] < scores[1]


def test_reflex_prefill_recorder_emits_page_risk_summaries(monkeypatch):
    monkeypatch.setenv("SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA", "1")
    monkeypatch.setenv("SEMANTIQ_REFLEX_PREFILL_Q_TAIL_TOKENS", "6")

    recorder = ReflexPrefillMetadataRecorder()
    query = torch.tensor([[[1.0, 0.0]]] * 6)
    key = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[-1.0, 0.0]],
            [[-1.0, 0.0]],
        ]
    )
    info = ReflexPrefillRequestInfo(
        request_id="req-summary",
        query_start=0,
        query_end=6,
        prompt_start=0,
        prompt_end=6,
        prompt_tokens=6,
    )

    with recorder.batch([info], block_size=2):
        recorder.record_layer("model.layers.0.self_attn", query, key)

    summaries = recorder.finalize_request_summaries(
        "req-summary",
        prompt_tokens=6,
        block_size=2,
    )

    assert summaries is not None
    assert [summary.page_idx for summary in summaries] == [0, 1, 2]
    assert summaries[0].risk_score > summaries[1].risk_score
    assert summaries[1].risk_score > summaries[2].risk_score
    assert summaries[2].compressible is True
    assert summaries[0].semantic_hash != summaries[2].semantic_hash


def test_mooncake_worker_metadata_aggregates_page_risks_by_max():
    left = MooncakeConnectorWorkerMetadata(
        {"req-a": [0.1, 0.4], "req-b": [0.2]}
    )
    right = MooncakeConnectorWorkerMetadata(
        {"req-a": [0.3, 0.2, 0.9], "req-c": [0.7]}
    )

    merged = left.aggregate(right)

    assert merged.reflex_page_risks_by_request == {
        "req-a": [0.3, 0.4, 0.9],
        "req-b": [0.2],
        "req-c": [0.7],
    }


def test_prefill_metadata_context_uses_scheduler_tokens_by_request(
    monkeypatch,
):
    monkeypatch.setenv("SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA", "1")

    import vllm.v1.worker.gpu_model_runner as gpu_model_runner

    recorder = _FakeRecorder()
    monkeypatch.setattr(
        gpu_model_runner,
        "get_reflex_prefill_metadata_recorder",
        lambda: recorder,
    )

    runner = object.__new__(GPUModelRunner)
    runner.cache_config = SimpleNamespace(block_size=16)
    runner.input_batch = SimpleNamespace(
        req_ids=["req-a"],
        num_prompt_tokens=np.array([128], dtype=np.int64),
        num_computed_tokens_cpu=np.array([32], dtype=np.int64),
    )
    runner.query_start_loc = SimpleNamespace(
        np=np.array([11], dtype=np.int32)
    )
    runner.requests = {
        "req-a": SimpleNamespace(
            sampling_params=SimpleNamespace(
                extra_args={
                    "kv_transfer_params": {"do_remote_decode": True}
                }
            )
        )
    }

    with runner._reflex_prefill_metadata_context(
        num_reqs=1,
        num_scheduled_tokens={"req-a": 8},
    ):
        pass

    infos, block_size = recorder.calls[0]
    assert block_size == 16
    assert infos == [
        ReflexPrefillRequestInfo(
            request_id="req-a",
            query_start=11,
            query_end=19,
            prompt_start=32,
            prompt_end=40,
            prompt_tokens=128,
        )
    ]


def test_reflex_prefill_recorder_records_from_torch_compile(monkeypatch):
    monkeypatch.setenv("SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA", "1")

    import vllm.v1.core.reflex_prefill_metadata as metadata

    recorder = ReflexPrefillMetadataRecorder()
    monkeypatch.setattr(metadata, "_RECORDER", recorder)
    info = ReflexPrefillRequestInfo(
        request_id="req-compiled",
        query_start=0,
        query_end=4,
        prompt_start=0,
        prompt_end=4,
        prompt_tokens=4,
    )
    query = torch.tensor(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    key = torch.tensor(
        [
            [[0.0, 1.0]],
            [[0.0, 1.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )

    @torch.compile(backend="eager")
    def compiled_record(q, k):
        maybe_record_reflex_prefill_layer("model.layers.0.self_attn", q, k)
        return q + k

    with recorder.batch([info], block_size=2):
        compiled_record(query, key)

    scores = recorder.drain_completed_requests(block_size=2)["req-compiled"]

    assert len(scores) == 2
    assert scores[0] < scores[1]
