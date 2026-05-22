import os
from types import SimpleNamespace

import pytest

from eval.backends import get_backend_class, list_backends, register_backend
from eval.backends.base import BaseBackend, GenerationConfig, PromptRecord
from eval.backends.semantiq_backend import SemantiqBackend
from eval.backends.vllm_backend import VllmBackend


class FakeBackend(BaseBackend):
    name = "fake"

    def build(self, args):
        return None

    def generate(self, records, gen_config):
        return []


def test_register_backend_round_trips_custom_backend():
    register_backend(FakeBackend)

    assert get_backend_class("fake").name == "fake"


def test_list_backends_contains_registered_backend():
    register_backend(FakeBackend)

    assert "fake" in list_backends()


def test_get_backend_class_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_backend_class("unknown")


class _FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False
        assert add_generation_prompt is True
        return f"CHAT::{messages[0]['content']}"


class _FakeOutputText:
    def __init__(self, text):
        self.text = text


class _FakeRequestOutput:
    def __init__(self, text):
        self.outputs = [_FakeOutputText(text)]


class _FakeLLM:
    def __init__(self, texts):
        self._texts = texts

    def get_tokenizer(self):
        return _FakeTokenizer()

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert prompts == ["prompt"]
        assert sampling_params.max_tokens == 32
        assert use_tqdm is False
        return [_FakeRequestOutput(text) for text in self._texts]


def test_vllm_backend_get_prompt_formatter_uses_tokenizer_chat_template():
    backend = VllmBackend()
    backend._llm = _FakeLLM(["answer"])

    formatter = backend.get_prompt_formatter()

    assert formatter("hello") == "CHAT::hello"


def test_vllm_backend_normalizes_generate_output():
    backend = VllmBackend()
    backend._llm = _FakeLLM(["answer"])
    backend.build_sampling_params = lambda gen_config: type(
        "FakeSamplingParams",
        (),
        {"max_tokens": gen_config.max_new_tokens},
    )()

    outputs = backend.generate(
        [PromptRecord(dataset="2wikimqa", prompt="prompt", answers=["answer"])],
        GenerationConfig(max_new_tokens=32),
    )

    assert outputs[0]["pred"] == "answer"


def test_vllm_backend_build_uses_create_llm_hook(monkeypatch):
    backend = VllmBackend()
    created = {}

    monkeypatch.setattr(backend, "build_engine_args", lambda args: {"model": args.model})
    monkeypatch.setattr(
        backend,
        "_create_llm",
        lambda engine_args: (created.__setitem__("engine_args", engine_args), _FakeLLM([]))[1],
    )

    class Args:
        model = "/tmp/model"

    backend.build(Args())

    assert created["engine_args"] == {"model": "/tmp/model"}


def test_semantiq_backend_is_registered():
    assert get_backend_class("semantiq").name == "semantiq"


def test_semantiq_backend_delegates_to_vllm_hooks():
    backend = SemantiqBackend()

    assert hasattr(backend, "build_engine_args")
    assert hasattr(backend, "build_sampling_params")


def test_semantiq_backend_build_engine_args_preserves_eager_setting(monkeypatch):
    backend = SemantiqBackend()
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_ENABLE", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_PAGE_SIZE", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_SEED", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_METHOD", raising=False)

    engine_args = SimpleNamespace(model="/tmp/model", enforce_eager=False)

    monkeypatch.setattr(
        VllmBackend,
        "build_engine_args",
        lambda self, args: engine_args,
    )

    class Args:
        model = "/tmp/model"
        semantiq_segment_enable = True
        semantiq_segment_page_size = 32
        semantiq_segment_similarity_threshold = 0.7
        semantiq_segment_output = "/tmp/segments.json"
        semantiq_fake_quant_enable = True
        semantiq_fake_quant_seed = 7
        semantiq_quant_method = 0

    args = Args()
    args.enforce_eager = False

    returned = backend.build_engine_args(args)

    assert returned is engine_args
    assert returned.enforce_eager is False
    assert args.enforce_eager is False
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_ENABLE"] == "1"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_PAGE_SIZE"] == "32"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_SIMILARITY_THRESHOLD"] == "0.7"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_OUTPUT_PATH"] == "/tmp/segments.json"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE"] == "1"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_SEED"] == "7"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_METHOD"] == "0"


def test_semantiq_backend_applies_query_segment_env_overrides(monkeypatch):
    backend = SemantiqBackend()
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_ENABLE", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_PAGE_SIZE", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_SIMILARITY_THRESHOLD", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_OUTPUT_PATH", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_SEED", raising=False)
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_METHOD", raising=False)

    class Args:
        semantiq_segment_enable = True
        semantiq_segment_page_size = 32
        semantiq_segment_similarity_threshold = 0.7
        semantiq_segment_output = "/tmp/segments.json"
        semantiq_fake_quant_enable = True
        semantiq_fake_quant_seed = 7
        semantiq_quant_method = 0

    engine_args = {"model": "/tmp/model"}

    args = Args()
    args.enforce_eager = False

    returned = backend.apply_semantiq_overrides(engine_args, args)

    assert returned is engine_args
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_ENABLE"] == "1"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_PAGE_SIZE"] == "32"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_SIMILARITY_THRESHOLD"] == "0.7"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_OUTPUT_PATH"] == "/tmp/segments.json"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE"] == "1"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_SEED"] == "7"
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_METHOD"] == "0"


def test_semantiq_backend_keeps_eager_setting_when_segment_capture_enabled(monkeypatch):
    backend = SemantiqBackend()
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_ENABLE", raising=False)

    class Args:
        semantiq_segment_enable = True
        semantiq_segment_page_size = 16
        semantiq_segment_similarity_threshold = 0.8
        semantiq_segment_output = None
        semantiq_fake_quant_enable = False
        semantiq_fake_quant_seed = 0
        semantiq_quant_method = 1

    engine_args = {"model": "/tmp/model", "enforce_eager": False}

    args = Args()
    args.enforce_eager = False

    returned = backend.apply_semantiq_overrides(engine_args, args)

    assert returned is engine_args
    assert engine_args["enforce_eager"] is False
    assert args.enforce_eager is False
