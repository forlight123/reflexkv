import importlib.util
import asyncio
from pathlib import Path
from types import SimpleNamespace


def _load_proxy_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "vllm"
        / "examples"
        / "online_serving"
        / "disaggregated_serving"
        / "mooncake_connector"
        / "mooncake_connector_proxy.py"
    )
    spec = importlib.util.spec_from_file_location("mooncake_connector_proxy", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_proxy_builds_independent_prefill_and_decode_payloads():
    proxy = _load_proxy_module()
    base_request = {
        "model": "llama",
        "prompt": "prompt",
        "max_tokens": 64,
        "max_completion_tokens": 64,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    prefill_client_info = {
        "bootstrap_addr": "http://127.0.0.1:8998",
        "dp_engine_id": {0: "engine-0"},
    }

    prefill_payload = proxy._build_prefill_request_data(base_request, "req-1")
    decode_payload = proxy._build_decode_request_data(
        base_request,
        prefill_client_info,
        0,
        "req-1",
        {
            "reflex_page_risks": [0.4, 0.1],
            "reflex_compressible_pages": [1],
            "ignored": True,
        },
    )

    assert "kv_transfer_params" not in base_request
    assert base_request["stream"] is True
    assert base_request["max_tokens"] == 64
    assert base_request["max_completion_tokens"] == 64
    assert "stream_options" in base_request

    assert prefill_payload["stream"] is False
    assert prefill_payload["max_tokens"] == 1
    assert prefill_payload["max_completion_tokens"] == 1
    assert "stream_options" not in prefill_payload
    assert prefill_payload["kv_transfer_params"] == {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": "xfer-req-1",
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
    }

    assert decode_payload["stream"] is True
    assert decode_payload["max_tokens"] == 64
    assert decode_payload["max_completion_tokens"] == 64
    assert decode_payload["kv_transfer_params"] == {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": "http://127.0.0.1:8998",
        "remote_engine_id": "engine-0",
        "transfer_id": "xfer-req-1",
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
        "reflex_page_risks": [0.4, 0.1],
        "reflex_compressible_pages": [1],
    }

    assert proxy._extract_reflex_kv_transfer_params(
        {"kv_transfer_params": {"reflex_page_risks": [0.3], "x": 1}}
    ) == {"reflex_page_risks": [0.3]}


def test_proxy_prefers_client_request_id_for_transfer_and_headers():
    proxy = _load_proxy_module()

    class _Request:
        headers = {"X-Request-Id": "header-rid"}

    assert proxy._resolve_request_id(
        {"request_id": "body-rid"},
        _Request(),
    ) == "body-rid"
    assert proxy._resolve_request_id(
        {},
        _Request(),
    ) == "header-rid"


def test_proxy_prefill_slot_is_held_until_released():
    proxy = _load_proxy_module()
    app = SimpleNamespace(
        state=SimpleNamespace(prefill_semaphore=asyncio.Semaphore(1))
    )

    async def scenario():
        acquired = await proxy._acquire_prefill_slot(app)
        assert acquired is True

        second = asyncio.create_task(proxy._acquire_prefill_slot(app))
        await asyncio.sleep(0)
        assert not second.done()

        proxy._release_prefill_slot(app, acquired)
        assert await asyncio.wait_for(second, timeout=0.1) is True
        proxy._release_prefill_slot(app, True)

    asyncio.run(scenario())


def test_proxy_prefill_slot_releases_when_prefill_task_finishes():
    proxy = _load_proxy_module()
    app = SimpleNamespace(
        state=SimpleNamespace(prefill_semaphore=asyncio.Semaphore(1))
    )

    async def scenario():
        async def fast_prefill():
            await asyncio.sleep(0)
            return {"reflex_page_risks": [0.2]}

        acquired = await proxy._acquire_prefill_slot(app)
        prefill_task = asyncio.create_task(fast_prefill())
        proxy._release_prefill_slot_when_prefill_done(
            app,
            acquired,
            prefill_task,
            "req-prefill-done",
        )

        await prefill_task
        await asyncio.sleep(0)

        second = await asyncio.wait_for(
            proxy._acquire_prefill_slot(app),
            timeout=0.1,
        )
        proxy._release_prefill_slot(app, second)

    asyncio.run(scenario())


def test_proxy_decode_backpressure_waits_for_capacity_metrics():
    proxy = _load_proxy_module()
    proxy.global_args = SimpleNamespace(
        decode_backpressure_policy="metrics",
        decode_backpressure_max_kv_usage=0.80,
        decode_backpressure_max_waiting=0,
        decode_backpressure_poll_interval_sec=0.001,
        decode_backpressure_timeout_sec=0.5,
    )

    class _Response:
        def __init__(self, text: str):
            self.text = text

        def raise_for_status(self):
            return None

    class _Client:
        def __init__(self):
            self.calls = 0

        async def get(self, path: str):
            assert path == "/metrics"
            self.calls += 1
            if self.calls == 1:
                return _Response(
                    "\n".join(
                        [
                            'vllm:kv_cache_usage_perc{engine="0"} 0.92',
                            'vllm:num_requests_waiting{engine="0"} 1',
                            'vllm:num_requests_running{engine="0"} 4',
                        ]
                    )
                )
            return _Response(
                "\n".join(
                    [
                        'vllm:kv_cache_usage_perc{engine="0"} 0.50',
                        'vllm:num_requests_waiting{engine="0"} 0',
                        'vllm:num_requests_running{engine="0"} 2',
                    ]
                )
            )

    async def scenario():
        client = _Client()
        await proxy._await_decode_backpressure(
            {"client": client, "url": "http://decode"},
            "req-pressure",
        )
        assert client.calls == 2

    asyncio.run(scenario())


def test_proxy_decode_backpressure_disabled_skips_metrics_query():
    proxy = _load_proxy_module()
    proxy.global_args = SimpleNamespace(decode_backpressure_policy="off")

    class _Client:
        async def get(self, path: str):
            raise AssertionError("disabled backpressure should not query metrics")

    async def scenario():
        await proxy._await_decode_backpressure(
            {"client": _Client(), "url": "http://decode"},
            "req-off",
        )

    asyncio.run(scenario())


def test_proxy_decode_admission_serializes_local_pending_waiting():
    proxy = _load_proxy_module()
    proxy.global_args = SimpleNamespace(
        decode_backpressure_policy="metrics",
        decode_backpressure_max_kv_usage=0.90,
        decode_backpressure_max_waiting=0,
        decode_backpressure_poll_interval_sec=0.001,
        decode_backpressure_timeout_sec=0.5,
    )

    class _Response:
        text = "\n".join(
            [
                'vllm:kv_cache_usage_perc{engine="0"} 0.10',
                'vllm:num_requests_waiting{engine="0"} 0',
                'vllm:num_requests_running{engine="0"} 0',
            ]
        )

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, path: str):
            assert path == "/metrics"
            return _Response()

    async def scenario():
        decode_client = {"client": _Client(), "url": "http://decode"}

        first = await proxy._acquire_decode_admission(
            decode_client,
            "req-first",
        )
        second = asyncio.create_task(
            proxy._acquire_decode_admission(
                decode_client,
                "req-second",
            )
        )

        await asyncio.sleep(0.01)
        assert not second.done()

        proxy._release_decode_admission(first)
        second_token = await asyncio.wait_for(second, timeout=0.2)
        proxy._release_decode_admission(second_token)

    asyncio.run(scenario())


def test_proxy_decode_admission_adaptive_waiting_allows_reflex_headroom():
    proxy = _load_proxy_module()
    proxy._decode_admission_locks.clear()
    proxy._decode_admission_pending.clear()
    proxy.global_args = SimpleNamespace(
        decode_backpressure_policy="metrics",
        decode_backpressure_max_kv_usage=0.90,
        decode_backpressure_max_waiting=0,
        decode_backpressure_waiting_policy="adaptive",
        decode_backpressure_adaptive_max_waiting=4,
        decode_backpressure_adaptive_kv_headroom_per_waiting=0.04,
        decode_backpressure_poll_interval_sec=0.001,
        decode_backpressure_timeout_sec=0.5,
    )

    class _Response:
        text = "\n".join(
            [
                'vllm:cache_config_info{engine="0",cache_dtype="reflex_int4"} 1',
                'vllm:kv_cache_usage_perc{engine="0"} 0.10',
                'vllm:num_requests_waiting{engine="0"} 0',
                'vllm:num_requests_running{engine="0"} 1',
            ]
        )

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, path: str):
            assert path == "/metrics"
            return _Response()

    async def scenario():
        decode_client = {"client": _Client(), "url": "http://decode"}

        first = await proxy._acquire_decode_admission(
            decode_client,
            "req-first",
        )
        second = await asyncio.wait_for(
            proxy._acquire_decode_admission(
                decode_client,
                "req-second",
            ),
            timeout=0.2,
        )

        proxy._release_decode_admission(first)
        proxy._release_decode_admission(second)

    asyncio.run(scenario())


def test_proxy_decode_admission_release_can_be_settled_before_response_headers():
    proxy = _load_proxy_module()
    proxy.global_args = SimpleNamespace(
        decode_backpressure_policy="metrics",
        decode_backpressure_max_kv_usage=0.90,
        decode_backpressure_max_waiting=0,
        decode_backpressure_poll_interval_sec=0.001,
        decode_backpressure_timeout_sec=0.5,
        decode_backpressure_admission_settle_sec=0.001,
    )

    class _Response:
        text = "\n".join(
            [
                'vllm:kv_cache_usage_perc{engine="0"} 0.10',
                'vllm:num_requests_waiting{engine="0"} 0',
                'vllm:num_requests_running{engine="0"} 1',
            ]
        )

        def raise_for_status(self):
            return None

    class _Client:
        async def get(self, path: str):
            assert path == "/metrics"
            return _Response()

    async def scenario():
        decode_client = {"client": _Client(), "url": "http://decode"}

        first = await proxy._acquire_decode_admission(
            decode_client,
            "req-first",
        )
        proxy._schedule_decode_admission_release(first)
        await asyncio.sleep(0.02)

        second = await asyncio.wait_for(
            proxy._acquire_decode_admission(
                decode_client,
                "req-second",
            ),
            timeout=0.2,
        )
        proxy._release_decode_admission(second)

    asyncio.run(scenario())


def test_proxy_cleanup_cancels_prefill_when_cleanup_is_cancelled():
    proxy = _load_proxy_module()

    async def scenario():
        started = asyncio.Event()

        async def slow_prefill():
            started.set()
            await asyncio.sleep(60)

        prefill_task = asyncio.create_task(slow_prefill())
        await started.wait()

        cleanup_task = asyncio.create_task(
            proxy._cleanup_prefill_task(prefill_task, "req-cancel")
        )
        await asyncio.sleep(0)
        cleanup_task.cancel()

        await asyncio.wait_for(cleanup_task, timeout=0.2)
        assert prefill_task.cancelled()

    asyncio.run(scenario())


def test_proxy_waits_for_prefill_metadata_when_timeout_is_configured():
    proxy = _load_proxy_module()
    proxy.global_args = SimpleNamespace(prefill_metadata_wait_timeout_sec=0.1)

    async def scenario():
        async def fast_prefill():
            await asyncio.sleep(0)
            return {"reflex_page_risks": [0.3, 0.1]}

        prefill_task = asyncio.create_task(fast_prefill())

        params = await proxy._await_prefill_metadata_if_configured(
            prefill_task,
            "req-meta",
        )

        assert params == {"reflex_page_risks": [0.3, 0.1]}
        assert prefill_task.done()

    asyncio.run(scenario())


def test_proxy_metadata_wait_timeout_leaves_prefill_task_running():
    proxy = _load_proxy_module()
    proxy.global_args = SimpleNamespace(prefill_metadata_wait_timeout_sec=0.001)

    async def scenario():
        async def slow_prefill():
            await asyncio.sleep(60)
            return {"reflex_page_risks": [0.3]}

        prefill_task = asyncio.create_task(slow_prefill())

        params = await proxy._await_prefill_metadata_if_configured(
            prefill_task,
            "req-timeout",
        )

        assert params is None
        assert not prefill_task.done()
        prefill_task.cancel()
        await proxy._await_prefill_task_safely(prefill_task, "req-timeout")

    asyncio.run(scenario())
