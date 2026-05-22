import json
import os
import stat
import subprocess
import textwrap
from pathlib import Path


def test_run_longbench_script_uses_baked_in_defaults_without_uv(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "run_longbench.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_file = tmp_path / "python-log.json"
    fake_nvidia_smi = fake_bin / "nvidia-smi"
    fake_nvidia_smi.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            print("0, 100")
            print("1, 512")
            print("2, 24576")
            """
        ),
        encoding="utf-8",
    )
    fake_nvidia_smi.chmod(fake_nvidia_smi.stat().st_mode | stat.S_IEXEC)
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
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    },
                    handle,
                )
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["FAKE_PYTHON_LOG"] = str(log_file)
    env.pop("QUANT_METHOD", None)

    subprocess.run(
        ["/bin/bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )

    payload = json.loads(log_file.read_text(encoding="utf-8"))
    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.longbench",
        "--backend",
        "semantiq",
        "--semantiq-fake-quant-enable",
        "--semantiq-quant-method",
        "0",
        "--semantiq-segment-page-size",
        "16",
        "--dataset",
        "qasper",
        "--data-dir",
        "/home/ytm/datasets/LongBench/data",
        "--output-dir",
        str(repo_root / "outputs" / "longbench"),
        "--run-name",
        "semantiq_random",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
    ]
    assert payload["pythonpath"] == f"{repo_root}:{repo_root / 'vllm'}"
    assert payload["cuda_visible_devices"] == "4,5,6,7"


def test_run_longbench_script_switches_to_semantiq_backend_via_env(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "run_longbench.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_file = tmp_path / "python-log.json"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys

            with open(os.environ["FAKE_PYTHON_LOG"], "w", encoding="utf-8") as handle:
                json.dump({"argv": sys.argv}, handle)
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["FAKE_PYTHON_LOG"] = str(log_file)
    env["BACKEND"] = "semantiq"
    env["BLOCK_SIZE"] = "32"
    env["QUANT_METHOD"] = "0"
    env["RUN_NAME"] = "semantiq-p32"

    subprocess.run(
        ["/bin/bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )

    payload = json.loads(log_file.read_text(encoding="utf-8"))
    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.longbench",
        "--backend",
        "semantiq",
        "--semantiq-fake-quant-enable",
        "--semantiq-quant-method",
        "0",
        "--semantiq-segment-page-size",
        "32",
        "--dataset",
        "qasper",
        "--data-dir",
        "/home/ytm/datasets/LongBench/data",
        "--output-dir",
        str(repo_root / "outputs" / "longbench"),
        "--run-name",
        "semantiq-p32",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "32",
    ]


def test_run_longbench_script_disables_prefix_caching_for_vllm_backend(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "run_longbench.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    log_file = tmp_path / "python-log.json"
    fake_python = fake_bin / "python"
    fake_python.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import os
            import sys

            with open(os.environ["FAKE_PYTHON_LOG"], "w", encoding="utf-8") as handle:
                json.dump({"argv": sys.argv}, handle)
            """
        ),
        encoding="utf-8",
    )
    fake_python.chmod(fake_python.stat().st_mode | stat.S_IEXEC)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:/usr/bin:/bin"
    env["FAKE_PYTHON_LOG"] = str(log_file)
    env["BACKEND"] = "vllm"
    env["RUN_NAME"] = "vllm-no-prefix-cache"

    subprocess.run(
        ["/bin/bash", str(script_path)],
        cwd=repo_root,
        env=env,
        check=True,
    )

    payload = json.loads(log_file.read_text(encoding="utf-8"))
    assert payload["argv"] == [
        str(fake_python),
        "-m",
        "eval.bench.longbench",
        "--no-enable-prefix-caching",
        "--dataset",
        "qasper",
        "--data-dir",
        "/home/ytm/datasets/LongBench/data",
        "--output-dir",
        str(repo_root / "outputs" / "longbench"),
        "--run-name",
        "vllm-no-prefix-cache",
        "--model",
        "/home/ytm/models/Llama-3.1-8B-Instruct",
        "--tensor-parallel-size",
        "4",
        "--block-size",
        "16",
    ]
