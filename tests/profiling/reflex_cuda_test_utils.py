import os
import unittest

import torch


def _preferred_gpu_ids() -> list[int]:
    raw_ids = os.environ.get("SEMANTIQ_REFLEX_TEST_GPU_IDS", "6,7")
    preferred: list[int] = []
    for item in raw_ids.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            preferred.append(int(item))
        except ValueError:
            continue
    return preferred


def select_reflex_cuda_test_device(
    *,
    min_free_bytes: int = 512 * 1024 * 1024,
) -> torch.device:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")

    visible_indices = list(range(torch.cuda.device_count()))
    preferred = [
        index for index in _preferred_gpu_ids() if index in set(visible_indices)
    ]
    candidate_indices = preferred or visible_indices

    best_index: int | None = None
    best_free_bytes = -1
    for index in candidate_indices:
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(index)
        except RuntimeError:
            continue
        if free_bytes > best_free_bytes:
            best_index = index
            best_free_bytes = int(free_bytes)

    if best_index is None:
        raise unittest.SkipTest("No queryable CUDA device is available")
    if best_free_bytes < min_free_bytes:
        raise unittest.SkipTest(
            "No CUDA device has enough free memory for ReFlexKV CUDA tests: "
            f"best_free_bytes={best_free_bytes}, required={min_free_bytes}."
        )

    torch.cuda.set_device(best_index)
    return torch.device("cuda", best_index)
