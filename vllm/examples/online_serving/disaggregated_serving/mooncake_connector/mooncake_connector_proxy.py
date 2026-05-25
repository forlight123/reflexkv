# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import asyncio
import ipaddress
import itertools
import os
import time
import urllib
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

_decode_admission_locks: dict[str, asyncio.Lock] = {}
_decode_admission_pending: dict[str, int] = {}


def _emit_reflex_stage_profile(
    request_id: str,
    *,
    phase: str,
    ms: float,
    **fields: Any,
) -> None:
    extras = " ".join(
        f"{key}={str(value).replace(' ', '_')}"
        for key, value in fields.items()
        if value is not None
    )
    suffix = f" {extras}" if extras else ""
    print(
        "ReFlexKV trace stage_profile "
        f"request={request_id} phase={phase} ms={ms:.3f}{suffix}."
    )


def maybe_wrap_ipv6_address(address: str) -> str:
    try:
        ipaddress.IPv6Address(address)
        return f"[{address}]"
    except ValueError:
        return address


def make_http_path(host: str, port: int) -> str:
    return f"http://{host}:{port}"


def prefiller_cycle(prefill_clients: list[Any]):
    while True:
        for prefill_client in prefill_clients:
            for i in range(prefill_client["dp_size"]):
                yield prefill_client, i


async def get_prefiller_info(prefill_clients: list, ready: asyncio.Event):
    for prefill_client in prefill_clients:
        while True:
            try:
                # Wait for prefill service to be ready
                response = await prefill_client["client"].get("/health")
                response.raise_for_status()
            except Exception:
                await asyncio.sleep(1)
                continue

            response = await prefill_client["client"].get(
                prefill_client["bootstrap_addr"] + "/query"
            )
            response.raise_for_status()
            data = response.json()
            break

        for dp_rank, dp_entry in data.items():
            prefill_client["dp_engine_id"][int(dp_rank)] = dp_entry["engine_id"]
        dp_size = len(data)
        prefill_client["dp_size"] = dp_size
        print(f"Inited prefiller {prefill_client['url']} with dp_size={dp_size}")

    ready.set()
    print("All prefiller instances are ready.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    """
    # Startup: Initialize client pools for prefiller and decoder services
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.ready = asyncio.Event()
    prefill_max_inflight = max(0, int(global_args.prefill_max_inflight or 0))
    app.state.prefill_semaphore = (
        asyncio.Semaphore(prefill_max_inflight) if prefill_max_inflight > 0 else None
    )

    # Create prefill clients
    for i, (url, bootstrap_port) in enumerate(global_args.prefill):
        parsed_url = urllib.parse.urlparse(url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
                "url": url,
                "bootstrap_addr": make_http_path(hostname, bootstrap_port or 8998),
                "dp_engine_id": {},
            }
        )

    # Create decode clients
    for i, url in enumerate(global_args.decode):
        parsed_url = urllib.parse.urlparse(url)
        hostname = maybe_wrap_ipv6_address(parsed_url.hostname)
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None,
                        max_keepalive_connections=None,
                    ),
                ),
            }
        )

    asyncio.create_task(get_prefiller_info(app.state.prefill_clients, app.state.ready))

    # Initialize round-robin iterators
    app.state.prefill_iterator = prefiller_cycle(app.state.prefill_clients)
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))

    print(
        f"Got {len(app.state.prefill_clients)} prefill clients "
        f"and {len(app.state.decode_clients)} decode clients."
    )

    yield

    # Shutdown: Close all clients
    for client_info in app.state.prefill_clients:
        await client_info["client"].aclose()

    for client_info in app.state.decode_clients:
        await client_info["client"].aclose()


# Update FastAPI app initialization to use lifespan
app = FastAPI(lifespan=lifespan)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--port", type=int, default=8000)
    # Always use 127.0.0.1 as localhost binds to IPv6 which is blocked on CI
    parser.add_argument("--host", type=str, default="127.0.0.1")

    # For prefiller instances
    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_raw",
        metavar=("URL", "bootstrap_port"),
        help=(
            "Prefill server URL and optional bootstrap port. "
            "Can be specified multiple times. "
            "Format: --prefill URL [BOOTSTRAP_PORT]. "
            "BOOTSTRAP_PORT can be a port number, "
            "'none', or omitted (defaults to none)."
        ),
    )

    # For decoder instances
    parser.add_argument(
        "--decode",
        nargs=1,
        action="append",
        dest="decode_raw",
        metavar=("URL",),
        help="Decode server URL. Can be specified multiple times.",
    )
    parser.add_argument(
        "--prefill-max-inflight",
        type=int,
        default=0,
        help=(
            "Maximum producer-side prefill tasks the proxy lets run "
            "concurrently. Use 0 to let decode admission/backpressure decide."
        ),
    )
    parser.add_argument(
        "--reflex-remote-chunk-tokens",
        type=int,
        default=int(os.environ.get("SEMANTIQ_REFLEX_REMOTE_CHUNK_TOKENS", "512")),
        help="Remote KV chunk size, in tokens, for ReFlexKV P/D transfer.",
    )
    parser.add_argument(
        "--prefill-metadata-wait-timeout-sec",
        type=float,
        default=float(
            os.environ.get("SEMANTIQ_REFLEX_PREFILL_METADATA_WAIT_TIMEOUT_SEC", "0")
        ),
        help=(
            "Maximum time to wait for the prefill response before starting "
            "decode, so returned ReFlexKV metadata can be attached to the "
            "decode request. Use 0 to preserve fully overlapped prefill/decode."
        ),
    )
    parser.add_argument(
        "--decode-backpressure-policy",
        choices=["off", "metrics"],
        default=os.environ.get("SEMANTIQ_PROXY_DECODE_BACKPRESSURE_POLICY", "off"),
        help=(
            "Proxy-side decode capacity gate. 'metrics' polls each decode "
            "server's /metrics before starting a new remote prefill."
        ),
    )
    parser.add_argument(
        "--decode-backpressure-max-kv-usage",
        type=float,
        default=float(
            os.environ.get("SEMANTIQ_PROXY_DECODE_BACKPRESSURE_MAX_KV_USAGE", "0.90")
        ),
        help="Maximum decode KV cache usage ratio allowed before prefill admission.",
    )
    parser.add_argument(
        "--decode-backpressure-max-waiting",
        type=int,
        default=int(os.environ.get("SEMANTIQ_PROXY_DECODE_BACKPRESSURE_MAX_WAITING", "0")),
        help=(
            "Maximum decode waiting requests allowed before proxy prefill "
            "admission waits. Use a negative value to ignore waiting requests."
        ),
    )
    parser.add_argument(
        "--decode-backpressure-waiting-policy",
        choices=["fixed", "adaptive"],
        default=os.environ.get(
            "SEMANTIQ_PROXY_DECODE_BACKPRESSURE_WAITING_POLICY",
            "fixed",
        ),
        help=(
            "Waiting gate policy. 'adaptive' lets reflex_int4 decode admit "
            "extra waiting requests when KV headroom is available."
        ),
    )
    parser.add_argument(
        "--decode-backpressure-adaptive-max-waiting",
        type=int,
        default=int(
            os.environ.get(
                "SEMANTIQ_PROXY_DECODE_BACKPRESSURE_ADAPTIVE_MAX_WAITING",
                "4",
            )
        ),
        help="Maximum extra waiting requests allowed by adaptive waiting.",
    )
    parser.add_argument(
        "--decode-backpressure-adaptive-kv-headroom-per-waiting",
        type=float,
        default=float(
            os.environ.get(
                "SEMANTIQ_PROXY_DECODE_BACKPRESSURE_ADAPTIVE_KV_HEADROOM_PER_WAITING",
                "0.04",
            )
        ),
        help="KV usage headroom consumed per adaptive waiting slot.",
    )
    parser.add_argument(
        "--decode-backpressure-poll-interval-sec",
        type=float,
        default=float(
            os.environ.get(
                "SEMANTIQ_PROXY_DECODE_BACKPRESSURE_POLL_INTERVAL_SEC", "0.05"
            )
        ),
        help="Polling interval while decode backpressure is active.",
    )
    parser.add_argument(
        "--decode-backpressure-timeout-sec",
        type=float,
        default=float(
            os.environ.get("SEMANTIQ_PROXY_DECODE_BACKPRESSURE_TIMEOUT_SEC", "300")
        ),
        help=(
            "Maximum time to wait for decode capacity before failing open. "
            "Use 0 to wait indefinitely."
        ),
    )
    parser.add_argument(
        "--decode-backpressure-admission-settle-sec",
        type=float,
        default=float(
            os.environ.get(
                "SEMANTIQ_PROXY_DECODE_BACKPRESSURE_ADMISSION_SETTLE_SEC", "1.0"
            )
        ),
        help=(
            "How long a locally admitted decode request should count against "
            "proxy-side admission before decode /metrics is expected to show "
            "it. This avoids holding the local admission slot until a "
            "non-streaming completion finishes."
        ),
    )

    args = parser.parse_args()
    args.prefill = _parse_prefill_urls(args.prefill_raw)
    args.decode = _parse_decode_urls(args.decode_raw)

    return args


# From sglang router_args.py
def _parse_prefill_urls(prefill_list):
    """Parse prefill URLs from --prefill arguments.

    Format: --prefill URL [BOOTSTRAP_PORT]
    Example:
        --prefill http://prefill1:8080 9000  # With bootstrap port
        --prefill http://prefill2:8080 none  # Explicitly no bootstrap port
        --prefill http://prefill3:8080       # Defaults to no bootstrap port
    """
    if not prefill_list:
        return []

    prefill_urls = []
    for prefill_args in prefill_list:
        url = prefill_args[0]

        # Handle optional bootstrap port
        if len(prefill_args) >= 2:
            bootstrap_port_str = prefill_args[1]
            # Handle 'none' as None
            if bootstrap_port_str.lower() == "none":
                bootstrap_port = None
            else:
                try:
                    bootstrap_port = int(bootstrap_port_str)
                except ValueError as e:
                    raise ValueError(
                        f"Invalid bootstrap port: {bootstrap_port_str}. Must be a number or 'none'"  # noqa: E501
                    ) from e
        else:
            # No bootstrap port specified, default to None
            bootstrap_port = None

        prefill_urls.append((url, bootstrap_port))

    return prefill_urls


def _parse_decode_urls(decode_list):
    """Parse decode URLs from --decode arguments.

    Format: --decode URL
    Example: --decode http://decode1:8081 --decode http://decode2:8081
    """
    if not decode_list:
        return []

    # decode_list is a list of single-element lists due to nargs=1
    return [url[0] for url in decode_list]


def get_next_client(app, service_type: str):
    """
    Get the next client in round-robin fashion.

    Args:
        app: The FastAPI app instance
        service_type: Either 'prefill' or 'decode'

    Returns:
        The next client to use
    """
    if service_type == "prefill":
        return next(app.state.prefill_iterator)
    elif service_type == "decode":
        client_idx = next(app.state.decode_iterator)
        return app.state.decode_clients[client_idx]
    else:
        raise ValueError(f"Unknown service type: {service_type}")


def _reflex_remote_chunk_tokens() -> int:
    args = globals().get("global_args")
    if args is None:
        return 512
    return int(getattr(args, "reflex_remote_chunk_tokens", 512))


def _build_prefill_request_data(req_data: dict, request_id: str) -> dict:
    data = req_data.copy()
    data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": f"xfer-{request_id}",
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": _reflex_remote_chunk_tokens(),
    }
    data["stream"] = False
    data["max_tokens"] = 1
    if "max_completion_tokens" in data:
        data["max_completion_tokens"] = 1
    if "stream_options" in data:
        del data["stream_options"]
    return data


def _build_decode_request_data(
    req_data: dict,
    prefill_client_info: dict,
    prefill_dp_rank: int,
    request_id: str,
    prefill_kv_transfer_params: dict | None = None,
) -> dict:
    data = req_data.copy()
    kv_transfer_params = {
        "do_remote_decode": False,
        "do_remote_prefill": True,
        "remote_bootstrap_addr": prefill_client_info["bootstrap_addr"],
        "remote_engine_id": prefill_client_info["dp_engine_id"][prefill_dp_rank],
        "transfer_id": f"xfer-{request_id}",
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": _reflex_remote_chunk_tokens(),
    }
    for key, value in (prefill_kv_transfer_params or {}).items():
        if key.startswith("reflex_"):
            kv_transfer_params[key] = value
    data["kv_transfer_params"] = kv_transfer_params
    return data


def _extract_reflex_kv_transfer_params(response_body: dict) -> dict:
    kv_transfer_params = response_body.get("kv_transfer_params")
    if not isinstance(kv_transfer_params, dict):
        return {}
    return {
        key: value
        for key, value in kv_transfer_params.items()
        if key.startswith("reflex_")
    }


def _resolve_request_id(req_data: dict, request: Request) -> str:
    request_id = req_data.get("request_id")
    if request_id:
        return str(request_id)
    header_request_id = request.headers.get("X-Request-Id")
    if header_request_id:
        return str(header_request_id)
    return str(uuid.uuid4())


async def _acquire_prefill_slot(app: FastAPI) -> bool:
    semaphore = getattr(app.state, "prefill_semaphore", None)
    if semaphore is None:
        return False
    await semaphore.acquire()
    return True


def _release_prefill_slot(app: FastAPI, acquired: bool) -> None:
    if not acquired:
        return
    semaphore = getattr(app.state, "prefill_semaphore", None)
    if semaphore is not None:
        semaphore.release()


def _release_prefill_slot_when_prefill_done(
    app: FastAPI,
    acquired: bool,
    prefill_task: asyncio.Task,
    request_id: str,
) -> bool:
    if not acquired:
        return False

    def _release_on_done(_task: asyncio.Task) -> None:
        _release_prefill_slot(app, True)

    prefill_task.add_done_callback(_release_on_done)
    return True


def _parse_prometheus_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) != 2:
            continue
        key, value = parts
        try:
            metrics[key] = float(value)
        except ValueError:
            continue
    return metrics


def _metric_values(metrics: dict[str, float], prefix: str) -> list[float]:
    return [
        value
        for key, value in metrics.items()
        if key == prefix or key.startswith(prefix + "{")
    ]


def _max_metric(metrics: dict[str, float], prefix: str) -> float | None:
    values = _metric_values(metrics, prefix)
    return max(values) if values else None


def _decode_backpressure_policy() -> str:
    args = globals().get("global_args")
    if args is None:
        return "off"
    return str(getattr(args, "decode_backpressure_policy", "off") or "off")


def _decode_backpressure_waiting_policy() -> str:
    args = globals().get("global_args")
    if args is None:
        return "fixed"
    return str(getattr(args, "decode_backpressure_waiting_policy", "fixed") or "fixed")


def _decode_backpressure_float(name: str, default: float) -> float:
    args = globals().get("global_args")
    if args is None:
        return default
    try:
        return float(getattr(args, name, default))
    except (TypeError, ValueError):
        return default


def _decode_backpressure_admission_settle_sec() -> float:
    return max(
        0.0,
        _decode_backpressure_float("decode_backpressure_admission_settle_sec", 1.0),
    )


def _decode_backpressure_int(name: str, default: int) -> int:
    args = globals().get("global_args")
    if args is None:
        return default
    try:
        return int(getattr(args, name, default))
    except (TypeError, ValueError):
        return default


def _decode_cache_dtype(metrics: dict[str, float]) -> str | None:
    for key in metrics:
        if not key.startswith("vllm:cache_config_info"):
            continue
        marker = 'cache_dtype="'
        start = key.find(marker)
        if start < 0:
            continue
        start += len(marker)
        end = key.find('"', start)
        if end > start:
            return key[start:end]
    return None


def _effective_decode_backpressure_max_waiting(
    metrics: dict[str, float],
    *,
    configured_max_waiting: int,
    kv_usage: float | None,
    max_kv_usage: float,
) -> int:
    if configured_max_waiting < 0:
        return configured_max_waiting
    if _decode_backpressure_waiting_policy() != "adaptive":
        return configured_max_waiting
    if _decode_cache_dtype(metrics) != "reflex_int4":
        return configured_max_waiting
    if kv_usage is None:
        return configured_max_waiting
    headroom = max(0.0, max_kv_usage - float(kv_usage))
    if headroom <= 0.0:
        return configured_max_waiting
    headroom_per_waiting = max(
        0.001,
        _decode_backpressure_float(
            "decode_backpressure_adaptive_kv_headroom_per_waiting",
            0.04,
        ),
    )
    adaptive_max_waiting = max(
        0,
        _decode_backpressure_int(
            "decode_backpressure_adaptive_max_waiting",
            4,
        ),
    )
    adaptive_waiting = min(
        adaptive_max_waiting,
        int(headroom / headroom_per_waiting),
    )
    return max(configured_max_waiting, adaptive_waiting)


class _DecodeAdmissionToken:
    __slots__ = ("active", "key")

    def __init__(self, key: str | None = None, active: bool = False):
        self.key = key
        self.active = active


def _decode_admission_key(decode_client_info: dict) -> str:
    return str(decode_client_info.get("url") or id(decode_client_info.get("client")))


def _decode_admission_lock(key: str) -> asyncio.Lock:
    lock = _decode_admission_locks.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _decode_admission_locks[key] = lock
    return lock


def _local_pending_waiting(local_pending_waiting: Any) -> int:
    try:
        if callable(local_pending_waiting):
            return max(0, int(local_pending_waiting()))
        return max(0, int(local_pending_waiting))
    except (TypeError, ValueError):
        return 0


async def _fetch_decode_metrics(decode_client_info: dict) -> dict[str, float]:
    response = await decode_client_info["client"].get("/metrics")
    response.raise_for_status()
    return _parse_prometheus_metrics(response.text)


async def _await_decode_backpressure(
    decode_client_info: dict,
    request_id: str,
    local_pending_waiting: Any = 0,
) -> None:
    if _decode_backpressure_policy() != "metrics":
        return

    max_kv_usage = max(
        0.0,
        _decode_backpressure_float("decode_backpressure_max_kv_usage", 0.90),
    )
    max_waiting = _decode_backpressure_int("decode_backpressure_max_waiting", 0)
    poll_interval = max(
        0.001,
        _decode_backpressure_float("decode_backpressure_poll_interval_sec", 0.05),
    )
    timeout = max(
        0.0,
        _decode_backpressure_float("decode_backpressure_timeout_sec", 300.0),
    )
    loop = asyncio.get_running_loop()
    started = loop.time()
    last_log = 0.0

    while True:
        try:
            metrics = await _fetch_decode_metrics(decode_client_info)
        except Exception as exc:  # noqa: BLE001 - fail open if telemetry breaks.
            print(
                "Decode backpressure metrics unavailable for request "
                f"{request_id}: {exc!r}; admitting prefill."
            )
            return

        kv_usage = _max_metric(metrics, "vllm:kv_cache_usage_perc")
        waiting = _max_metric(metrics, "vllm:num_requests_waiting")
        effective_max_waiting = _effective_decode_backpressure_max_waiting(
            metrics,
            configured_max_waiting=max_waiting,
            kv_usage=kv_usage,
            max_kv_usage=max_kv_usage,
        )
        pending_waiting = _local_pending_waiting(local_pending_waiting)
        effective_waiting = (
            None if waiting is None else waiting + float(pending_waiting)
        )
        kv_ready = kv_usage is None or kv_usage <= max_kv_usage
        waiting_ready = (
            effective_max_waiting < 0
            or effective_waiting is None
            or effective_waiting <= effective_max_waiting
        )
        if kv_ready and waiting_ready:
            return

        now = loop.time()
        if timeout > 0.0 and now - started >= timeout:
            print(
                "Decode backpressure timed out for request "
                f"{request_id} after {timeout:.3f}s "
                f"(kv_usage={kv_usage}, waiting={waiting}, "
                f"local_pending_waiting={pending_waiting}); admitting prefill."
            )
            return
        if now - last_log >= 5.0:
            decode_url = decode_client_info.get("url", "decode")
            print(
                "Decode backpressure holding request "
                f"{request_id} for {decode_url}: "
                f"kv_usage={kv_usage}, max_kv_usage={max_kv_usage}, "
                f"waiting={waiting}, local_pending_waiting={pending_waiting}, "
                f"max_waiting={effective_max_waiting}."
            )
            last_log = now
        await asyncio.sleep(poll_interval)


async def _acquire_decode_admission(
    decode_client_info: dict,
    request_id: str,
) -> _DecodeAdmissionToken:
    if _decode_backpressure_policy() != "metrics":
        await _await_decode_backpressure(decode_client_info, request_id)
        return _DecodeAdmissionToken()

    max_waiting = _decode_backpressure_int("decode_backpressure_max_waiting", 0)
    if max_waiting < 0:
        await _await_decode_backpressure(decode_client_info, request_id)
        return _DecodeAdmissionToken()

    key = _decode_admission_key(decode_client_info)
    lock = _decode_admission_lock(key)

    async with lock:
        # Metrics are sampled before decode sees the new request. Count local
        # admissions that have crossed this gate but are not reflected yet.
        await _await_decode_backpressure(
            decode_client_info,
            request_id,
            local_pending_waiting=lambda: _decode_admission_pending.get(key, 0),
        )
        _decode_admission_pending[key] = _decode_admission_pending.get(key, 0) + 1
        return _DecodeAdmissionToken(key=key, active=True)


def _release_decode_admission(token: _DecodeAdmissionToken | None) -> None:
    if token is None or not token.active or token.key is None:
        return
    pending = max(0, _decode_admission_pending.get(token.key, 0) - 1)
    if pending:
        _decode_admission_pending[token.key] = pending
    else:
        _decode_admission_pending.pop(token.key, None)
    token.active = False


def _schedule_decode_admission_release(
    token: _DecodeAdmissionToken | None,
) -> None:
    if token is None or not token.active:
        return
    settle_sec = _decode_backpressure_admission_settle_sec()
    if settle_sec <= 0.0:
        _release_decode_admission(token)
        return
    loop = asyncio.get_running_loop()
    loop.call_later(settle_sec, _release_decode_admission, token)


async def _await_prefill_task_safely(
    prefill_task: asyncio.Task, request_id: str
) -> None:
    try:
        await prefill_task
    except asyncio.CancelledError:
        pass
    except BaseExceptionGroup as exc:
        print(f"Prefill task failed for request {request_id}: {exc!r}")
    except Exception as exc:  # noqa: BLE001 - proxy should log and continue.
        print(f"Prefill task failed for request {request_id}: {exc!r}")


async def _cleanup_prefill_task(prefill_task: asyncio.Task, request_id: str) -> None:
    if prefill_task.done():
        await _await_prefill_task_safely(prefill_task, request_id)
        return

    should_cancel_prefill = False
    try:
        await asyncio.wait_for(asyncio.shield(prefill_task), timeout=5.0)
        return
    except asyncio.TimeoutError:
        print(f"Prefill task cleanup timed out for request {request_id}; cancelling.")
        should_cancel_prefill = True
    except asyncio.CancelledError:
        print(f"Prefill task cleanup cancelled for request {request_id}; cancelling.")
        should_cancel_prefill = True
    except BaseExceptionGroup as exc:
        print(f"Prefill task failed for request {request_id}: {exc!r}")
        return
    except Exception as exc:  # noqa: BLE001 - proxy should log and continue.
        print(f"Prefill task failed for request {request_id}: {exc!r}")
        return

    if should_cancel_prefill:
        prefill_task.cancel()
        await _await_prefill_task_safely(prefill_task, request_id)


def _prefill_metadata_wait_timeout_sec() -> float:
    args = globals().get("global_args")
    if args is None:
        return 0.0
    try:
        return max(0.0, float(getattr(args, "prefill_metadata_wait_timeout_sec", 0.0)))
    except (TypeError, ValueError):
        return 0.0


async def _await_prefill_metadata_if_configured(
    prefill_task: asyncio.Task,
    request_id: str,
) -> dict | None:
    started = time.perf_counter()
    emitted = False

    def _emit_metadata_wait(outcome: str) -> None:
        nonlocal emitted
        if emitted:
            return
        emitted = True
        _emit_reflex_stage_profile(
            request_id,
            phase="prefill_metadata_wait",
            ms=(time.perf_counter() - started) * 1000.0,
            outcome=outcome,
            source="proxy",
        )

    if prefill_task.done():
        try:
            result = await prefill_task
            _emit_metadata_wait("done")
            return result
        except asyncio.CancelledError:
            _emit_metadata_wait("cancelled")
            return None
        except BaseExceptionGroup as exc:
            print(f"Prefill task failed for request {request_id}: {exc!r}")
            _emit_metadata_wait("failed")
            return None
        except Exception as exc:  # noqa: BLE001 - proxy should log and continue.
            print(f"Prefill task failed for request {request_id}: {exc!r}")
            _emit_metadata_wait("failed")
            return None

    timeout = _prefill_metadata_wait_timeout_sec()
    if timeout <= 0.0:
        _emit_metadata_wait("disabled")
        return None

    try:
        result = await asyncio.wait_for(asyncio.shield(prefill_task), timeout=timeout)
        _emit_metadata_wait("ready")
        return result
    except asyncio.TimeoutError:
        print(
            "Prefill metadata wait timed out for request "
            f"{request_id} after {timeout:.3f}s; decode will use fallback metadata."
        )
        _emit_metadata_wait("timeout")
        return None
    except asyncio.CancelledError:
        print(f"Prefill metadata wait cancelled for request {request_id}.")
        _emit_metadata_wait("cancelled")
        return None
    except BaseExceptionGroup as exc:
        print(f"Prefill task failed for request {request_id}: {exc!r}")
        _emit_metadata_wait("failed")
        return None
    except Exception as exc:  # noqa: BLE001 - proxy should log and continue.
        print(f"Prefill task failed for request {request_id}: {exc!r}")
        _emit_metadata_wait("failed")
        return None


async def send_request_to_service(
    client_info: dict, dp_rank: int, endpoint: str, req_data: dict, request_id: str
) -> dict:
    """
    Send a request to a service using a client from the pool.
    """
    req_data = _build_prefill_request_data(req_data, request_id)
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
        "X-data-parallel-rank": str(dp_rank),
    }

    started = time.perf_counter()
    outcome = "ok"
    response = None
    try:
        response = await client_info["client"].post(
            endpoint, json=req_data, headers=headers
        )
        response.raise_for_status()
        try:
            body = response.json()
        except ValueError:
            body = {}
        return _extract_reflex_kv_transfer_params(body)
    except Exception:
        outcome = "error"
        raise
    finally:
        if response is not None:
            await response.aclose()
        _emit_reflex_stage_profile(
            request_id,
            phase="prefill",
            ms=(time.perf_counter() - started) * 1000.0,
            outcome=outcome,
            source="proxy",
            dp_rank=dp_rank,
        )


async def stream_service_response(
    prefill_client_info: dict,
    prefill_dp_rank: int,
    decode_client_info: dict,
    endpoint: str,
    req_data: dict,
    request_id: str,
    prefill_kv_transfer_params: dict | None = None,
    decode_admission_token: _DecodeAdmissionToken | None = None,
):
    """
    Asynchronously stream response from a service using a client from the pool.
    """
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "X-Request-Id": request_id,
    }

    req_data = _build_decode_request_data(
        req_data,
        prefill_client_info,
        prefill_dp_rank,
        request_id,
        prefill_kv_transfer_params,
    )
    _schedule_decode_admission_release(decode_admission_token)

    decode_started = time.perf_counter()
    first_chunk_at: float | None = None
    chunk_count = 0
    async with decode_client_info["client"].stream(
        "POST", endpoint, json=req_data, headers=headers
    ) as response:
        response.raise_for_status()
        _release_decode_admission(decode_admission_token)
        async for chunk in response.aiter_bytes():
            if first_chunk_at is None:
                first_chunk_at = time.perf_counter()
                _emit_reflex_stage_profile(
                    request_id,
                    phase="decode_waiting",
                    ms=(first_chunk_at - decode_started) * 1000.0,
                    source="proxy",
                )
            chunk_count += 1
            yield chunk
    finished = time.perf_counter()
    if first_chunk_at is None:
        _emit_reflex_stage_profile(
            request_id,
            phase="decode_waiting",
            ms=(finished - decode_started) * 1000.0,
            source="proxy",
            chunks=0,
        )
    else:
        _emit_reflex_stage_profile(
            request_id,
            phase="decode_running",
            ms=(finished - first_chunk_at) * 1000.0,
            source="proxy",
            chunks=chunk_count,
        )


async def _handle_completions(api: str, request: Request):
    if not app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")

    prefill_slot_acquired = False
    decode_admission_token: _DecodeAdmissionToken | None = None
    try:
        req_data = await request.json()
        request_id = _resolve_request_id(req_data, request)
        decode_client_info = get_next_client(request.app, "decode")
        stage_started = time.perf_counter()
        decode_admission_token = await _acquire_decode_admission(
            decode_client_info,
            request_id,
        )
        _emit_reflex_stage_profile(
            request_id,
            phase="proxy_wait",
            ms=(time.perf_counter() - stage_started) * 1000.0,
            stage="decode_admission",
            source="proxy",
        )
        stage_started = time.perf_counter()
        prefill_slot_acquired = await _acquire_prefill_slot(request.app)
        _emit_reflex_stage_profile(
            request_id,
            phase="proxy_wait",
            ms=(time.perf_counter() - stage_started) * 1000.0,
            stage="prefill_slot",
            source="proxy",
        )

        # Get the next prefill client in round-robin fashion
        prefill_client_info, prefill_dp_rank = get_next_client(request.app, "prefill")

        # Send request to prefill service
        prefill_task = asyncio.create_task(
            send_request_to_service(
                prefill_client_info, prefill_dp_rank, api, req_data, request_id
            )
        )
        if _release_prefill_slot_when_prefill_done(
            request.app,
            prefill_slot_acquired,
            prefill_task,
            request_id,
        ):
            prefill_slot_acquired = False

        # Stream response from decode service
        async def generate_stream():
            try:
                prefill_kv_transfer_params = (
                    await _await_prefill_metadata_if_configured(
                        prefill_task,
                        request_id,
                    )
                )
                async for chunk in stream_service_response(
                    prefill_client_info,
                    prefill_dp_rank,
                    decode_client_info,
                    api,
                    req_data,
                    request_id=request_id,
                    prefill_kv_transfer_params=prefill_kv_transfer_params,
                    decode_admission_token=decode_admission_token,
                ):
                    yield chunk
            finally:
                _release_decode_admission(decode_admission_token)
                await _cleanup_prefill_task(prefill_task, request_id)
                _release_prefill_slot(request.app, prefill_slot_acquired)

        return StreamingResponse(generate_stream(), media_type="application/json")

    except Exception as e:
        _release_decode_admission(decode_admission_token)
        _release_prefill_slot(request.app, prefill_slot_acquired)
        import sys
        import traceback

        exc_info = sys.exc_info()
        print(f"Error occurred in disagg prefill proxy server - {api} endpoint")
        print(e)
        print("".join(traceback.format_exception(*exc_info)))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


if __name__ == "__main__":
    global global_args
    global_args = parse_args()

    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
