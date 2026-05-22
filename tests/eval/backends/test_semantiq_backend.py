import argparse
import os
from types import SimpleNamespace

from eval.backends.semantiq_backend import SemantiqBackend
from eval.backends.vllm_backend import VllmBackend


def _mode_name(value):
    return value if isinstance(value, str) else value.name


def test_semantiq_backend_wires_prior_path_cli_arg_to_env(monkeypatch):
    backend = SemantiqBackend()
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH", raising=False)
    monkeypatch.setattr(VllmBackend, "add_cli_args", lambda parser: parser)

    parser = argparse.ArgumentParser()
    SemantiqBackend.add_cli_args(parser)
    args = parser.parse_args(["--semantiq-prior-path", "/tmp/prior.json"])

    engine_args = {"model": "/tmp/model"}
    returned = backend.apply_semantiq_overrides(engine_args, args)

    assert returned is engine_args
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH"] == "/tmp/prior.json"
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH", raising=False)


def test_semantiq_backend_preserves_prior_path_env_for_k_base_artifact(
    monkeypatch,
):
    backend = SemantiqBackend()
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH", raising=False)
    monkeypatch.setattr(VllmBackend, "add_cli_args", lambda parser: parser)
    engine_args = {"model": "/tmp/model"}
    monkeypatch.setattr(
        VllmBackend,
        "build_engine_args",
        lambda self, args: engine_args,
    )

    parser = argparse.ArgumentParser()
    SemantiqBackend.add_cli_args(parser)
    args = parser.parse_args(["--semantiq-prior-path", "/tmp/prior.json"])

    returned = backend.build_engine_args(args)

    assert returned is engine_args
    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH"] == "/tmp/prior.json"
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_PRIOR_PATH", raising=False)


def test_semantiq_backend_enables_fake_quant_env(monkeypatch):
    backend = SemantiqBackend()
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE", raising=False)
    monkeypatch.setattr(VllmBackend, "add_cli_args", lambda parser: parser)
    monkeypatch.setattr(
        VllmBackend,
        "build_engine_args",
        lambda self, args: {"model": "/tmp/model"},
    )

    parser = argparse.ArgumentParser()
    SemantiqBackend.add_cli_args(parser)
    args = parser.parse_args(["--semantiq-fake-quant-enable"])

    backend.build_engine_args(args)

    assert os.environ["SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE"] == "1"
    monkeypatch.delenv("SEMANTIQ_QUERY_SEGMENTS_FAKE_QUANT_ENABLE", raising=False)


def test_semantiq_backend_disables_cudagraphs_for_segment_capture(monkeypatch):
    backend = SemantiqBackend()
    monkeypatch.setattr(VllmBackend, "add_cli_args", lambda parser: parser)

    parser = argparse.ArgumentParser()
    SemantiqBackend.add_cli_args(parser)
    args = parser.parse_args(["--semantiq-segment-enable"])

    engine_args = SimpleNamespace(
        compilation_config=SimpleNamespace(
            cudagraph_mode="FULL_AND_PIECEWISE",
            max_cudagraph_capture_size=512,
            cudagraph_capture_sizes=[1, 2, 4],
        )
    )

    backend.apply_semantiq_overrides(engine_args, args)

    assert _mode_name(engine_args.compilation_config.cudagraph_mode) == "NONE"
    assert engine_args.compilation_config.max_cudagraph_capture_size == 0
    assert engine_args.compilation_config.cudagraph_capture_sizes == []


def test_semantiq_backend_disables_cudagraphs_for_fake_quant(monkeypatch):
    backend = SemantiqBackend()
    monkeypatch.setattr(VllmBackend, "add_cli_args", lambda parser: parser)

    parser = argparse.ArgumentParser()
    SemantiqBackend.add_cli_args(parser)
    args = parser.parse_args(["--semantiq-fake-quant-enable"])

    engine_args = {
        "compilation_config": {
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "max_cudagraph_capture_size": 512,
            "cudagraph_capture_sizes": [1, 2, 4],
        }
    }

    backend.apply_semantiq_overrides(engine_args, args)

    assert _mode_name(engine_args["compilation_config"]["cudagraph_mode"]) == "NONE"
    assert engine_args["compilation_config"]["max_cudagraph_capture_size"] == 0
    assert engine_args["compilation_config"]["cudagraph_capture_sizes"] == []
