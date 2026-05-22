import json
import os
from types import SimpleNamespace

import pytest


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _build_request_output(prompt, token_ids, token_logprobs):
    prompt_logprobs = [None]
    for token_id, logprob in zip(token_ids[1:], token_logprobs, strict=True):
        prompt_logprobs.append(
            {token_id: SimpleNamespace(logprob=float(logprob), rank=1, decoded_token=None)}
        )
    return SimpleNamespace(
        prompt=prompt,
        prompt_token_ids=list(token_ids),
        prompt_logprobs=prompt_logprobs,
        outputs=[SimpleNamespace(text="", token_ids=[], cumulative_logprob=None, logprobs=None)],
        finished=True,
    )


def _build_env_snapshot():
    return {
        "fake_quant": os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE"),
        "prior_path": os.environ.get("SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH"),
        "rank": os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_RANK"),
        "layer": os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_LAYER_NAME"),
        "head": os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_HEAD_ID"),
        "side": os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_SIDE"),
        "bit_width": os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_BIT_WIDTH"),
    }


@pytest.fixture
def calibration_fixture(tmp_path):
    data_file = tmp_path / "tiny.jsonl"
    output_path = tmp_path / "prior.json"
    _write_jsonl(
        data_file,
        [
            {"prompt": "alpha"},
            {"prompt": "beta"},
            {"prompt": "gamma"},
        ],
    )
    return data_file, output_path


def test_parse_args_reads_thresholds_and_output_path():
    from eval.bench.semantiq_prior import parse_args

    args = parse_args(
        [
            "--model",
            "/models/local",
            "--data-file",
            "/tmp/data.jsonl",
            "--output-path",
            "/tmp/prior.json",
            "--k-tau-2",
            "0.11",
            "--k-tau-4",
            "0.02",
            "--tensor-parallel-size",
            "2",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.95",
            "--no-enforce-eager",
            "--semantiq-segment-page-size",
            "32",
        ]
    )

    assert args.model == "/models/local"
    assert args.data_file == "/tmp/data.jsonl"
    assert args.output_path == "/tmp/prior.json"
    assert args.k_tau_2 == pytest.approx(0.11)
    assert args.k_tau_4 == pytest.approx(0.02)
    assert args.tensor_parallel_size == 2
    assert args.max_model_len == 8192
    assert args.gpu_memory_utilization == pytest.approx(0.95)
    assert args.enforce_eager is False
    assert args.semantiq_segment_page_size == 32


def test_parse_args_defaults_to_enforce_eager():
    from eval.bench.semantiq_prior import parse_args

    args = parse_args(
        [
            "--model",
            "/models/local",
            "--data-file",
            "/tmp/data.jsonl",
            "--output-path",
            "/tmp/prior.json",
            "--k-tau-2",
            "0.11",
            "--k-tau-4",
            "0.02",
        ]
    )

    assert args.enforce_eager is True


def test_runner_writes_normalized_prior_artifact_with_exact_orchestration(
    monkeypatch,
    calibration_fixture,
):
    from eval.bench import semantiq_prior as runner

    data_file, output_path = calibration_fixture
    worker_snapshot = {
        "rank": "0",
        "requests": {
            "req-0": {
                "layers": {
                    "model.layers.0.attn": {
                        "segments_by_head": {
                            "2": [{"segment_id": 0}],
                            "0": [{"segment_id": 0}],
                        }
                    },
                    "model.layers.1.attn": {
                        "segments_by_head": {"1": [{"segment_id": 0}]}
                    },
                }
            }
        }
    }
    token_scores = {
        ("baseline", None, None, None): [[-1.0, -2.0], [-3.0, -4.0]],
        ("key", "2", "0", "0"): [[-1.05, -2.05], [-3.05, -4.05]],
        ("key", "4", "0", "0"): [[-1.01, -2.01], [-3.01, -4.01]],
        ("key", "2", "2", "0"): [[-1.12, -2.12], [-3.12, -4.12]],
        ("key", "4", "2", "0"): [[-1.02, -2.02], [-3.02, -4.02]],
        ("key", "2", "1", "0"): [[-1.12, -2.12], [-3.12, -4.12]],
        ("key", "4", "1", "0"): [[-1.00, -2.00], [-3.00, -4.00]],
    }
    generate_snapshots = []
    rpc_calls = []
    init_kwargs = []
    shutdown_calls = []

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.max_tokens = kwargs["max_tokens"]
            self.temperature = kwargs["temperature"]
            self.prompt_logprobs = kwargs["prompt_logprobs"]

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            self.model = model
            self.kwargs = kwargs
            init_kwargs.append(dict(kwargs))
            assert kwargs["tensor_parallel_size"] == 2
            assert kwargs["max_model_len"] == 8192
            assert kwargs["gpu_memory_utilization"] == pytest.approx(0.95)
            assert kwargs["enforce_eager"] is True
            assert (
                kwargs["worker_extension_cls"]
                == "vllm.semantiq.query_segments.SemantiqWorkerExtension"
            )
            self.runtime_config = {
                "enabled": False,
                "page_size": 16,
                "similarity_threshold": 0.8,
                "output_path": None,
                "prior_path": None,
                "fake_quant_enabled": False,
                "force_rank": None,
                "force_layer_name": None,
                "force_head_id": None,
                "force_side": None,
                "force_bit_width": None,
            }
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(
                    shutdown=lambda timeout=None: shutdown_calls.append(
                        ("engine_core", timeout)
                    )
                ),
                renderer=SimpleNamespace(
                    shutdown=lambda: shutdown_calls.append(("renderer", None))
                ),
            )

        def generate(self, prompts, sampling_params, use_tqdm):
            assert prompts == ["alpha", "beta"]
            assert use_tqdm is False
            assert sampling_params.max_tokens == 1
            assert sampling_params.temperature == 0.0
            assert sampling_params.prompt_logprobs == 1
            generate_snapshots.append(dict(self.runtime_config))
            force_side = self.runtime_config["force_side"]
            force_bit_width = self.runtime_config["force_bit_width"]
            force_head = self.runtime_config["force_head_id"]
            force_rank = self.runtime_config["force_rank"]
            key = (
                "baseline" if not self.runtime_config["fake_quant_enabled"] else force_side,
                None if force_bit_width is None else str(force_bit_width),
                None if force_head is None else str(force_head),
                None if force_rank is None else str(force_rank),
            )
            return [
                _build_request_output(prompt, [10, 11, 12], token_logprobs)
                for prompt, token_logprobs in zip(prompts, token_scores[key], strict=True)
            ]

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            del timeout, kwargs
            rpc_calls.append((method, args))
            if method == "semantiq_configure_query_segment_runtime":
                (overrides,) = args
                self.runtime_config = {**self.runtime_config, **dict(overrides)}
                return [None]
            if method == "semantiq_snapshot_query_segment_runtime":
                return [worker_snapshot]
            raise AssertionError(f"Unexpected collective RPC {method!r}")

    monkeypatch.setattr(runner, "_import_vllm", lambda: (FakeLLM, FakeSamplingParams))

    runner.main(
        [
            "--model",
            "offline-local-model",
            "--data-file",
            str(data_file),
            "--output-path",
            str(output_path),
            "--max-samples",
            "2",
            "--k-tau-2",
            "0.10",
            "--k-tau-4",
            "0.02",
            "--tensor-parallel-size",
            "2",
            "--max-model-len",
            "8192",
            "--gpu-memory-utilization",
            "0.95",
            "--semantiq-segment-page-size",
            "64",
        ]
    )

    artifact = _read_json(output_path)

    assert artifact["k_base_bits"] == {
        "model.layers.0.attn": {"0": [2, 8, 4]},
        "model.layers.1.attn": {"0": [8, 4]},
    }
    assert artifact["meta"] == {
        "model": "offline-local-model",
        "metric": "delta_nll",
        "granularity": "kv_head",
        "page_size": 64,
        "default_k_base_bits": 4,
        "aggregation": "mean_token_delta_nll",
        "k_thresholds": {"2": 0.10, "4": 0.02},
    }
    assert generate_snapshots == [
        {"enabled": True, "page_size": 64, "similarity_threshold": 0.8, "output_path": None, "prior_path": None, "fake_quant_enabled": False, "force_rank": None, "force_layer_name": None, "force_head_id": None, "force_side": None, "force_bit_width": None},
        {"enabled": True, "page_size": 64, "similarity_threshold": 0.8, "output_path": None, "prior_path": None, "fake_quant_enabled": True, "force_rank": "0", "force_layer_name": "model.layers.0.attn", "force_head_id": 0, "force_side": "key", "force_bit_width": 2},
        {"enabled": True, "page_size": 64, "similarity_threshold": 0.8, "output_path": None, "prior_path": None, "fake_quant_enabled": True, "force_rank": "0", "force_layer_name": "model.layers.0.attn", "force_head_id": 0, "force_side": "key", "force_bit_width": 4},
        {"enabled": True, "page_size": 64, "similarity_threshold": 0.8, "output_path": None, "prior_path": None, "fake_quant_enabled": True, "force_rank": "0", "force_layer_name": "model.layers.0.attn", "force_head_id": 2, "force_side": "key", "force_bit_width": 2},
        {"enabled": True, "page_size": 64, "similarity_threshold": 0.8, "output_path": None, "prior_path": None, "fake_quant_enabled": True, "force_rank": "0", "force_layer_name": "model.layers.0.attn", "force_head_id": 2, "force_side": "key", "force_bit_width": 4},
        {"enabled": True, "page_size": 64, "similarity_threshold": 0.8, "output_path": None, "prior_path": None, "fake_quant_enabled": True, "force_rank": "0", "force_layer_name": "model.layers.1.attn", "force_head_id": 1, "force_side": "key", "force_bit_width": 2},
        {"enabled": True, "page_size": 64, "similarity_threshold": 0.8, "output_path": None, "prior_path": None, "fake_quant_enabled": True, "force_rank": "0", "force_layer_name": "model.layers.1.attn", "force_head_id": 1, "force_side": "key", "force_bit_width": 4},
    ]
    assert [method for method, _ in rpc_calls] == [
        "semantiq_configure_query_segment_runtime",
        "semantiq_snapshot_query_segment_runtime",
        "semantiq_configure_query_segment_runtime",
        "semantiq_configure_query_segment_runtime",
        "semantiq_configure_query_segment_runtime",
        "semantiq_configure_query_segment_runtime",
        "semantiq_configure_query_segment_runtime",
        "semantiq_configure_query_segment_runtime",
    ]
    assert len(init_kwargs) == 1
    assert len(shutdown_calls) == 2


def test_runner_preserves_tp_rank_scoped_heads_in_prior_artifact(
    monkeypatch, calibration_fixture
):
    from eval.bench import semantiq_prior as runner

    data_file, output_path = calibration_fixture
    worker_snapshots = [
        {
            "rank": "0",
            "requests": {
                "req-0": {
                    "layers": {
                        "model.layers.0.attn": {
                            "segments_by_head": {"0": [{"segment_id": 0}]}
                        }
                    }
                }
            },
        },
        {
            "rank": "1",
            "requests": {
                "req-0": {
                    "layers": {
                        "model.layers.0.attn": {
                            "segments_by_head": {"0": [{"segment_id": 0}]}
                        }
                    }
                }
            },
        },
    ]
    token_scores = {
        ("baseline", None, None, None): [[-1.0, -2.0], [-3.0, -4.0]],
        ("key", "2", "0", "0"): [[-1.01, -2.01], [-3.01, -4.01]],
        ("key", "4", "0", "0"): [[-1.001, -2.001], [-3.001, -4.001]],
        ("key", "2", "0", "1"): [[-1.30, -2.30], [-3.30, -4.30]],
        ("key", "4", "0", "1"): [[-1.01, -2.01], [-3.01, -4.01]],
    }

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.max_tokens = kwargs["max_tokens"]
            self.temperature = kwargs["temperature"]
            self.prompt_logprobs = kwargs["prompt_logprobs"]

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            del model
            self.runtime_config = {
                "enabled": False,
                "page_size": 16,
                "similarity_threshold": 0.8,
                "output_path": None,
                "prior_path": None,
                "fake_quant_enabled": False,
                "force_rank": None,
                "force_layer_name": None,
                "force_head_id": None,
                "force_side": None,
                "force_bit_width": None,
            }
            self.llm_engine = SimpleNamespace(
                engine_core=SimpleNamespace(shutdown=lambda timeout=None: None),
                renderer=SimpleNamespace(shutdown=lambda: None),
            )

        def generate(self, prompts, sampling_params, use_tqdm):
            del sampling_params, use_tqdm
            assert prompts == ["alpha", "beta"]
            key = (
                "baseline"
                if not self.runtime_config["fake_quant_enabled"]
                else self.runtime_config["force_side"],
                None
                if self.runtime_config["force_bit_width"] is None
                else str(self.runtime_config["force_bit_width"]),
                None
                if self.runtime_config["force_head_id"] is None
                else str(self.runtime_config["force_head_id"]),
                None
                if self.runtime_config["force_rank"] is None
                else str(self.runtime_config["force_rank"]),
            )
            return [
                _build_request_output(prompt, [10, 11, 12], token_logprobs)
                for prompt, token_logprobs in zip(prompts, token_scores[key], strict=True)
            ]

        def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
            del timeout, kwargs
            if method == "semantiq_configure_query_segment_runtime":
                (overrides,) = args
                self.runtime_config = {**self.runtime_config, **dict(overrides)}
                return [None, None]
            if method == "semantiq_snapshot_query_segment_runtime":
                return worker_snapshots
            raise AssertionError(f"Unexpected collective RPC {method!r}")

    monkeypatch.setattr(runner, "_import_vllm", lambda: (FakeLLM, FakeSamplingParams))

    runner.main(
        [
            "--model",
            "offline-local-model",
            "--data-file",
            str(data_file),
            "--output-path",
            str(output_path),
            "--max-samples",
            "2",
            "--k-tau-2",
            "0.10",
            "--k-tau-4",
            "0.02",
        ]
    )

    artifact = _read_json(output_path)

    assert artifact["k_base_bits"] == {"model.layers.0.attn": {"0": [2], "1": [4]}}
    assert "v_floor_bits" not in artifact


def test_runner_respects_max_samples(monkeypatch, calibration_fixture):
    from eval.bench import semantiq_prior as runner

    data_file, output_path = calibration_fixture
    captured_prompt_batches = []
    runtime = SimpleNamespace(
        snapshot=lambda: {
            "requests": {
                "req-0": {
                    "layers": {
                        "model.layers.0.attn": {
                            "segments_by_head": {"0": [{"segment_id": 0}]}
                        }
                    }
                }
            }
        }
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.max_tokens = kwargs["max_tokens"]
            self.temperature = kwargs["temperature"]
            self.prompt_logprobs = kwargs["prompt_logprobs"]

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            del model, kwargs

        def generate(self, prompts, sampling_params, use_tqdm):
            del sampling_params, use_tqdm
            captured_prompt_batches.append(list(prompts))
            return [
                _build_request_output(prompt, [10, 11], [-1.0]) for prompt in prompts
            ]

    monkeypatch.setattr(runner, "_import_vllm", lambda: (FakeLLM, FakeSamplingParams))
    monkeypatch.setattr(runner, "get_query_segment_runtime", lambda: runtime)
    monkeypatch.setattr(runner, "reset_query_segment_runtime", lambda: None)

    runner.main(
        [
            "--model",
            "offline-local-model",
            "--data-file",
            str(data_file),
            "--output-path",
            str(output_path),
            "--max-samples",
            "2",
            "--k-tau-2",
            "0.1",
            "--k-tau-4",
            "0.1",
        ]
    )

    assert captured_prompt_batches
    assert all(batch == ["alpha", "beta"] for batch in captured_prompt_batches)


def test_runner_uses_reasoning_prompt_field_precedence(monkeypatch, calibration_fixture):
    from eval.bench import semantiq_prior as runner

    data_file, output_path = calibration_fixture
    data_file.write_text(
        json.dumps({"prompt": "prompt-value", "problem": "problem-value"}) + "\n",
        encoding="utf-8",
    )
    captured_prompt_batches = []
    runtime = SimpleNamespace(
        snapshot=lambda: {
            "requests": {
                "req-0": {
                    "layers": {
                        "model.layers.0.attn": {
                            "segments_by_head": {"0": [{"segment_id": 0}]}
                        }
                    }
                }
            }
        }
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            del kwargs

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            del model, kwargs

        def generate(self, prompts, sampling_params, use_tqdm):
            del sampling_params, use_tqdm
            captured_prompt_batches.append(list(prompts))
            return [_build_request_output(prompts[0], [10, 11], [-1.0])]

    monkeypatch.setattr(runner, "_import_vllm", lambda: (FakeLLM, FakeSamplingParams))
    monkeypatch.setattr(runner, "get_query_segment_runtime", lambda: runtime)
    monkeypatch.setattr(runner, "reset_query_segment_runtime", lambda: None)

    runner.main(
        [
            "--model",
            "offline-local-model",
            "--data-file",
            str(data_file),
            "--output-path",
            str(output_path),
            "--k-tau-2",
            "0.1",
            "--k-tau-4",
            "0.1",
        ]
    )

    assert captured_prompt_batches
    assert all(batch == ["problem-value"] for batch in captured_prompt_batches)


def test_collect_target_heads_rejects_non_numeric_head_ids():
    from eval.bench.semantiq_prior import _collect_target_heads

    snapshot = {
        "requests": {
            "req-0": {
                "layers": {
                    "model.layers.0.attn": {
                        "segments_by_head": {"zero": [{"segment_id": 0}]}
                    }
                }
            }
        }
    }

    with pytest.raises(
        ValueError,
        match=r"Invalid SemantiQ calibration head id 'zero' for layer 'model\.layers\.0\.attn'",
    ):
        _collect_target_heads(snapshot)


def test_collect_target_heads_uses_shared_rank_resolver_for_local_fallback(
    monkeypatch,
):
    from eval.bench import semantiq_prior as runner

    monkeypatch.setattr(runner, "resolve_semantiq_rank", lambda: "7")

    snapshot = {
        "requests": {
            "req-0": {
                "layers": {
                    "model.layers.0.attn": {
                        "segments_by_head": {"0": [{"segment_id": 0}]}
                    }
                }
            }
        }
    }

    assert runner._collect_target_heads(snapshot) == [("7", "model.layers.0.attn", 0)]


def test_runner_sets_forced_quant_target_env(monkeypatch, calibration_fixture):
    from eval.bench import semantiq_prior as runner

    data_file, output_path = calibration_fixture
    reset_env_snapshots = []
    runtime = SimpleNamespace(
        snapshot=lambda: {
            "requests": {
                "req-0": {
                    "layers": {
                        "model.layers.3.attn": {
                            "segments_by_head": {"0": [{"segment_id": 0}]}
                        }
                    }
                }
            }
        }
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            del kwargs

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            del model, kwargs

        def generate(self, prompts, sampling_params, use_tqdm):
            del sampling_params, use_tqdm
            return [
                _build_request_output(prompt, [10, 11], [-1.0]) for prompt in prompts[:2]
            ]

    def fake_reset():
        reset_env_snapshots.append(_build_env_snapshot())

    monkeypatch.setattr(runner, "_import_vllm", lambda: (FakeLLM, FakeSamplingParams))
    monkeypatch.setattr(runner, "get_query_segment_runtime", lambda: runtime)
    monkeypatch.setattr(runner, "reset_query_segment_runtime", fake_reset)

    runner.main(
        [
            "--model",
            "offline-local-model",
            "--data-file",
            str(data_file),
            "--output-path",
            str(output_path),
            "--max-samples",
            "2",
            "--k-tau-2",
            "0.1",
            "--k-tau-4",
            "0.1",
        ]
    )

    assert reset_env_snapshots[0] == {
        "fake_quant": "0",
        "prior_path": None,
        "rank": None,
        "layer": None,
        "head": None,
        "side": None,
        "bit_width": None,
    }
    assert reset_env_snapshots[1] == {
        "fake_quant": "1",
        "prior_path": None,
        "rank": "0",
        "layer": "model.layers.3.attn",
        "head": "0",
        "side": "key",
        "bit_width": "2",
    }
    os.environ.pop("SEMANTIQ_QUERY_SEGMENTS_FORCE_RANK", None)
    assert os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_LAYER_NAME") is None
    assert os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_HEAD_ID") is None
    assert os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_SIDE") is None
    assert os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FORCE_BIT_WIDTH") is None


def test_runner_enables_fake_quant_only_for_perturbed_passes(
    monkeypatch,
    calibration_fixture,
):
    from eval.bench import semantiq_prior as runner

    data_file, output_path = calibration_fixture
    fake_quant_flags = []
    runtime = SimpleNamespace(
        snapshot=lambda: {
            "requests": {
                "req-0": {
                    "layers": {
                        "model.layers.0.attn": {
                            "segments_by_head": {"0": [{"segment_id": 0}]}
                        }
                    }
                }
            }
        }
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            del kwargs

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            del model, kwargs

        def generate(self, prompts, sampling_params, use_tqdm):
            del prompts, sampling_params, use_tqdm
            fake_quant_flags.append(
                os.environ.get("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE")
            )
            return [
                _build_request_output("alpha", [10, 11], [-1.0]),
                _build_request_output("beta", [10, 11], [-1.0]),
            ]

    monkeypatch.setattr(runner, "_import_vllm", lambda: (FakeLLM, FakeSamplingParams))
    monkeypatch.setattr(runner, "get_query_segment_runtime", lambda: runtime)
    monkeypatch.setattr(runner, "reset_query_segment_runtime", lambda: None)

    runner.main(
        [
            "--model",
            "offline-local-model",
            "--data-file",
            str(data_file),
            "--output-path",
            str(output_path),
            "--max-samples",
            "2",
            "--k-tau-2",
            "0.1",
            "--k-tau-4",
            "0.1",
        ]
    )

    assert fake_quant_flags[0] == "0"
    assert all(flag == "1" for flag in fake_quant_flags[1:])


def test_runner_clears_stale_prior_path_during_calibration_and_restores_it(
    monkeypatch,
    calibration_fixture,
):
    from eval.bench import semantiq_prior as runner

    data_file, output_path = calibration_fixture
    seen_prior_paths = []
    runtime = SimpleNamespace(
        snapshot=lambda: {
            "requests": {
                "req-0": {
                    "layers": {
                        "model.layers.0.attn": {
                            "segments_by_head": {"0": [{"segment_id": 0}]}
                        }
                    }
                }
            }
        }
    )

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            del kwargs

    class FakeLLM:
        def __init__(self, *, model, **kwargs):
            del model, kwargs

        def generate(self, prompts, sampling_params, use_tqdm):
            del prompts, sampling_params, use_tqdm
            seen_prior_paths.append(
                os.environ.get("SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH")
            )
            return [
                _build_request_output("alpha", [10, 11], [-1.0]),
                _build_request_output("beta", [10, 11], [-1.0]),
            ]

    monkeypatch.setattr(runner, "_import_vllm", lambda: (FakeLLM, FakeSamplingParams))
    monkeypatch.setattr(runner, "get_query_segment_runtime", lambda: runtime)
    monkeypatch.setattr(runner, "reset_query_segment_runtime", lambda: None)
    monkeypatch.setenv(
        "SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH",
        "/tmp/stale-prior.json",
    )

    runner.main(
        [
            "--model",
            "offline-local-model",
            "--data-file",
            str(data_file),
            "--output-path",
            str(output_path),
            "--max-samples",
            "2",
            "--k-tau-2",
            "0.1",
            "--k-tau-4",
            "0.1",
        ]
    )

    assert seen_prior_paths
    assert all(path is None for path in seen_prior_paths)
    assert os.environ.get("SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH") == "/tmp/stale-prior.json"
