import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path


def _make_fake_python(fake_bin, log_file):
    fake_python = fake_bin / "python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys

            with open(os.environ["FAKE_PYTHON_LOG"], "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "argv": sys.argv,
                        "pythonpath": os.environ.get("PYTHONPATH"),
                        "vllm_disable_compile_cache": os.environ.get(
                            "VLLM_DISABLE_COMPILE_CACHE"
                        ),
                    },
                    handle,
                )
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)
    return fake_python


def _run_reasoning_script(tmp_path, extra_env=None, cwd=None):
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "run_reasoning.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_file = tmp_path / "python-log.json"
    fake_python = _make_fake_python(fake_bin, log_file)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["FAKE_PYTHON_LOG"] = str(log_file)
    if extra_env:
        env.update(extra_env)

    subprocess.run(
        ["/bin/bash", str(script_path)],
        cwd=repo_root if cwd is None else cwd,
        env=env,
        check=True,
    )

    return fake_python, repo_root, json.loads(log_file.read_text(encoding="utf-8"))


def test_run_reasoning_script_uses_math500_task_by_default(tmp_path):
    fake_python, repo_root, payload = _run_reasoning_script(tmp_path)

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "semantiq_1",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-prior-path",
        str(repo_root / "outputs" / "priors" / "_debug_hybrid_k_base_tp4.json"),
        "--semantiq-quant-method",
        "1",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "1"


def test_run_reasoning_script_allows_task_override(tmp_path):
    fake_python, repo_root, payload = _run_reasoning_script(tmp_path, {"TASK": "aime2025"})

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "aime2025",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "semantiq_1",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-prior-path",
        str(repo_root / "outputs" / "priors" / "_debug_hybrid_k_base_tp4.json"),
        "--semantiq-quant-method",
        "1",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "1"


def test_run_reasoning_script_switches_backend_without_prefix_caching(tmp_path):
    fake_python, repo_root, payload = _run_reasoning_script(tmp_path, {"BACKEND": "vllm"})

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "vllm",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "vllm_1",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
    ]
    assert "--semantiq-fake-quant-enable" not in payload["argv"]
    assert "--semantiq-segment-page-size" not in payload["argv"]
    assert "--semantiq-quant-method" not in payload["argv"]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] is None


def test_run_reasoning_script_includes_run_name_when_set(tmp_path):
    fake_python, repo_root, payload = _run_reasoning_script(tmp_path, {"RUN_NAME": "reasoning-smoke"})

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "reasoning-smokesemantiq_1",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-prior-path",
        str(repo_root / "outputs" / "priors" / "_debug_hybrid_k_base_tp4.json"),
        "--semantiq-quant-method",
        "1",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "1"


def test_run_reasoning_script_allows_max_samples_override(tmp_path):
    fake_python, repo_root, payload = _run_reasoning_script(
        tmp_path, {"MAX_SAMPLES": "17"}
    )

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "semantiq_1",
        "--max-samples",
        "17",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-prior-path",
        str(repo_root / "outputs" / "priors" / "_debug_hybrid_k_base_tp4.json"),
        "--semantiq-quant-method",
        "1",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "1"


def test_run_reasoning_script_allows_quant_method_override(tmp_path):
    fake_python, repo_root, payload = _run_reasoning_script(
        tmp_path, {"QUANT_METHOD": "0"}
    )

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "semantiq_0",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-quant-method",
        "0",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "1"


def test_run_reasoning_script_passes_semantiq_prior_path_when_set(tmp_path):
    prior_path = tmp_path / "prior.json"
    fake_python, repo_root, payload = _run_reasoning_script(
        tmp_path, {"SEMANTIQ_PRIOR_PATH": str(prior_path)}
    )

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "semantiq_1",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-prior-path",
        str(prior_path),
        "--semantiq-quant-method",
        "1",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "1"


def test_run_reasoning_script_defaults_prior_path_from_repo_root(tmp_path):
    external_cwd = tmp_path / "external"
    external_cwd.mkdir()
    fake_python, repo_root, payload = _run_reasoning_script(tmp_path, cwd=external_cwd)

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "semantiq_1",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-prior-path",
        str(repo_root / "outputs" / "priors" / "_debug_hybrid_k_base_tp4.json"),
        "--semantiq-quant-method",
        "1",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "1"


def test_run_reasoning_script_allows_compile_cache_override(tmp_path):
    fake_python, repo_root, payload = _run_reasoning_script(
        tmp_path, {"VLLM_DISABLE_COMPILE_CACHE": "0"}
    )

    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.reasoning",
        "--backend",
        "semantiq",
        "--dataset",
        "math500",
        "--data-dir",
        str(repo_root / "data"),
        "--output-dir",
        str(repo_root / "outputs" / "reasoning"),
        "--run-name",
        "semantiq_1",
        "--max-samples",
        "500",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
        "--no-enable-prefix-caching",
        "--semantiq-fake-quant-enable",
        "--semantiq-segment-page-size",
        "16",
        "--semantiq-prior-path",
        str(repo_root / "outputs" / "priors" / "_debug_hybrid_k_base_tp4.json"),
        "--semantiq-quant-method",
        "1",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["vllm_disable_compile_cache"] == "0"
