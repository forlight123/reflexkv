from __future__ import annotations

from copy import deepcopy

_ALLOWED_BITS = (2, 4, 8)


def mean_token_delta_nll(baseline: list[float], perturbed: list[float]) -> float:
    if len(baseline) != len(perturbed):
        raise ValueError("baseline and perturbed must have the same length")
    if not baseline:
        raise ValueError("baseline and perturbed must be non-empty")
    return sum(
        float(perturbed_value) - float(baseline_value)
        for baseline_value, perturbed_value in zip(baseline, perturbed)
    ) / float(len(baseline))


def select_k_base_bits(
    delta_at_2: float,
    delta_at_4: float | None,
    tau_2: float,
    tau_4: float,
) -> int:
    if float(delta_at_2) <= float(tau_2):
        return 2
    if delta_at_4 is not None and float(delta_at_4) <= float(tau_4):
        return 4
    return 8


def normalize_floor_pair(raw_k: int, raw_v: int) -> tuple[int, int]:
    if raw_k not in _ALLOWED_BITS:
        raise ValueError(f"raw_k must be one of {_ALLOWED_BITS}, got {raw_k}")
    if raw_v not in _ALLOWED_BITS:
        raise ValueError(f"raw_v must be one of {_ALLOWED_BITS}, got {raw_v}")
    return max(raw_k, raw_v), raw_v


def build_prior_artifact(
    *,
    k_base_bits: dict[str, object],
    meta: dict[str, object],
) -> dict[str, object]:
    return {
        "k_base_bits": deepcopy(k_base_bits),
        "meta": deepcopy(meta),
    }


# Backward-compatible aliases for nearby code paths during the migration.
select_floor_bits = select_k_base_bits
