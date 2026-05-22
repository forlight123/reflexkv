import json

import pytest

from vllm.semantiq.prior import (
    SemantiqPrior,
    load_semantiq_prior,
    lookup_k_base_bits,
    resolve_k_base_bits,
    resolve_semantiq_rank,
)


def test_load_semantiq_prior_reads_k_base_bits_and_default_base(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "granularity": "kv_head",
                    "page_size": 16,
                    "default_k_base_bits": 4,
                },
                "k_base_bits": {"layer.0": {"0": [2, 4], "1": [8, 4]}},
            }
        ),
        encoding="utf-8",
    )

    prior = load_semantiq_prior(str(path))

    assert prior == SemantiqPrior(
        k_base_bits={"layer.0": {"0": (2, 4), "1": (8, 4)}},
        default_k_base_bits=4,
        page_size=16,
        granularity="kv_head",
    )
    assert lookup_k_base_bits(prior, "layer.0", 1, rank="0") == 4
    assert prior.default_k_base_bits == 4
    assert resolve_k_base_bits(prior, "layer.0", 9, rank="0") == 4


def test_load_semantiq_prior_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_semantiq_prior(str(tmp_path / "missing.json"))


def test_load_semantiq_prior_rejects_missing_k_base_bits(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps({"meta": {"granularity": "kv_head", "page_size": 16}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="k_base_bits is required"):
        load_semantiq_prior(str(path))


@pytest.mark.parametrize("bad_bit_value", [1, 3, 5, 9, True, 2.5])
def test_load_semantiq_prior_rejects_invalid_bit_values(tmp_path, bad_bit_value):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"granularity": "kv_head", "page_size": 16},
                "k_base_bits": {"layer.0": {"0": [bad_bit_value]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be one of"):
        load_semantiq_prior(str(path))


def test_load_semantiq_prior_rejects_malformed_rank_maps(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"granularity": "kv_head", "page_size": 16},
                "k_base_bits": {"layer.0": [2, 4]},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="rank map"):
        load_semantiq_prior(str(path))


def test_load_semantiq_prior_rejects_non_list_rank_entries(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"granularity": "kv_head", "page_size": 16},
                "k_base_bits": {"layer.0": {"0": {"head": 2}}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must be a list of bits"):
        load_semantiq_prior(str(path))


def test_load_semantiq_prior_rejects_mismatched_rank_head_counts(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"granularity": "kv_head", "page_size": 16},
                "k_base_bits": {
                    "layer.0": {
                        "0": [2, 4],
                        "1": [8],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same head count"):
        load_semantiq_prior(str(path))


@pytest.mark.parametrize(
    "meta, message",
    [
        ({"page_size": 16}, "meta.granularity is required"),
        ({"granularity": "token", "page_size": 16}, "must be 'kv_head'"),
        ({"granularity": "kv_head"}, "meta.page_size is required"),
    ],
)
def test_load_semantiq_prior_rejects_invalid_meta(tmp_path, meta, message):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": meta,
                "k_base_bits": {"layer.0": {"0": [2]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        load_semantiq_prior(str(path))


def test_load_semantiq_prior_requires_positive_page_size(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": {"granularity": "kv_head", "page_size": 0},
                "k_base_bits": {"layer.0": {"0": [2]}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="positive integer"):
        load_semantiq_prior(str(path))


def test_load_semantiq_prior_preserves_default_k_base_bits_in_artifact_roundtrip(tmp_path):
    path = tmp_path / "prior.json"
    path.write_text(
        json.dumps(
            {
                "meta": {
                    "granularity": "kv_head",
                    "page_size": 16,
                    "default_k_base_bits": 8,
                },
                "k_base_bits": {"layer.0": {"0": [8]}},
            }
        ),
        encoding="utf-8",
    )

    prior = load_semantiq_prior(str(path))
    assert prior.default_k_base_bits == 8
    assert json.loads(path.read_text(encoding="utf-8"))["meta"][
        "default_k_base_bits"
    ] == 8


def test_resolve_semantiq_rank_uses_distributed_rank_first(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 7)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "5")

    assert resolve_semantiq_rank() == "7"


def test_resolve_semantiq_rank_falls_back_when_get_rank_raises(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: True)

    def raise_runtime_error():
        raise RuntimeError("rank unavailable during shutdown")

    monkeypatch.setattr("torch.distributed.get_rank", raise_runtime_error)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "5")

    assert resolve_semantiq_rank() == "3"


def test_resolve_semantiq_rank_falls_back_to_env_and_default(monkeypatch):
    monkeypatch.setattr("torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("torch.distributed.is_initialized", lambda: False)
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "5")

    assert resolve_semantiq_rank() == "3"

    monkeypatch.delenv("RANK", raising=False)
    assert resolve_semantiq_rank() == "5"

    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert resolve_semantiq_rank() == "0"
