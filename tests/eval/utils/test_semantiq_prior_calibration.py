import json

from eval.utils.semantiq_prior_calibration import (
    build_prior_artifact,
    mean_token_delta_nll,
    normalize_floor_pair,
    select_k_base_bits,
)
from vllm.semantiq.prior import load_semantiq_prior


def test_mean_token_delta_nll_returns_average_difference():
    assert mean_token_delta_nll([1.0, 2.5, 4.0], [1.5, 3.0, 4.5]) == 0.5


def test_select_k_base_bits_prefers_lower_precision_when_under_thresholds():
    assert select_k_base_bits(
        delta_at_2=0.05, delta_at_4=0.01, tau_2=0.10, tau_4=0.02
    ) == 2
    assert select_k_base_bits(
        delta_at_2=0.12, delta_at_4=0.01, tau_2=0.10, tau_4=0.02
    ) == 4
    assert select_k_base_bits(
        delta_at_2=0.12, delta_at_4=0.03, tau_2=0.10, tau_4=0.02
    ) == 8
    assert select_k_base_bits(
        delta_at_2=0.12, delta_at_4=None, tau_2=0.10, tau_4=0.02
    ) == 8


def test_normalize_floor_pair_raises_value_side_to_key_floor():
    assert normalize_floor_pair(2, 4) == (4, 4)
    assert normalize_floor_pair(8, 2) == (8, 2)


def test_build_prior_artifact_preserves_default_k_base_bits():
    meta = {
        "model_name": "semantiq-test",
        "dataset": {
            "name": "tiny",
            "split": "validation",
            "prompt_template": {"id": "p1", "version": 3},
        },
        "thresholds": {"tau_2": 0.02, "tau_4": 0.01},
        "capture": {"seed": 7, "notes": ["forced", "offline-runner-pending"]},
        "granularity": "kv_head",
        "page_size": 16,
        "default_k_base_bits": 4,
    }

    artifact = build_prior_artifact(
        k_base_bits={"model.layers.0.attn": {"0": [4, 8]}},
        meta=meta,
    )

    assert artifact == {
        "k_base_bits": {"model.layers.0.attn": {"0": [4, 8]}},
        "meta": meta,
    }


def test_build_prior_artifact_roundtrips_full_meta_payload(tmp_path):
    artifact = build_prior_artifact(
        k_base_bits={"model.layers.0.attn": {"0": [8]}},
        meta={
            "granularity": "kv_head",
            "page_size": 16,
            "default_k_base_bits": 8,
            "capture": {
                "model_revision": "abc123",
                "sweeps": [{"side": "key", "bit_width": 2, "delta": 0.013}],
            },
            "extra": {"nested": {"list": [1, 2, 3]}},
        },
    )
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    loaded = load_semantiq_prior(str(path))
    roundtripped = json.loads(path.read_text(encoding="utf-8"))

    assert loaded.k_base_bits == {"model.layers.0.attn": {"0": (8,)}}
    assert loaded.default_k_base_bits == 8
    assert loaded.page_size == 16
    assert loaded.granularity == "kv_head"
    assert roundtripped["meta"] == artifact["meta"]

