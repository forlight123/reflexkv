import json
from pathlib import Path

from gen_data.prepare_ruler_datasets import (
    build_ruler_prepare_command,
    normalize_ruler_row,
    ruler_length_label,
)


def test_ruler_length_label_uses_k_suffix_for_common_context_lengths():
    assert ruler_length_label(4096) == "4k"
    assert ruler_length_label(8192) == "8k"
    assert ruler_length_label(32768) == "32k"


def test_build_ruler_prepare_command_targets_official_prepare_script(tmp_path):
    command = build_ruler_prepare_command(
        ruler_root=Path("data/RULER"),
        save_dir=tmp_path,
        task="niah_single_1",
        max_seq_length=4096,
        num_samples=10,
        tokenizer_path=Path("/models/llama"),
        tokenizer_type="hf",
        random_seed=7,
    )

    assert command[:2] == ["python", "data/RULER/scripts/data/prepare.py"]
    assert "--task" in command
    assert "niah_single_1" in command
    assert "--max_seq_length" in command
    assert "4096" in command
    assert "--num_samples" in command
    assert "10" in command
    assert "--tokenizer_path" in command
    assert "/models/llama" in command


def test_normalize_ruler_row_writes_reasoning_loader_schema():
    source = {
        "index": 3,
        "input": "Find the hidden value.",
        "outputs": ["1234567"],
        "length": 4088,
    }

    row = normalize_ruler_row(
        source,
        dataset="ruler_niah_single_1_4k",
        task="niah_single_1",
        max_seq_length=4096,
    )

    assert row == {
        "problem": "Find the hidden value.",
        "answers": ["1234567"],
        "source": "ruler",
        "unique_id": "ruler_niah_single_1_4k/3",
        "task": "niah_single_1",
        "max_seq_length": 4096,
        "length": 4088,
    }
    json.dumps(row)
