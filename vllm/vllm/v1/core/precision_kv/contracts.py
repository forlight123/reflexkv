# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix and mixed-precision landing contract helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vllm.v1.core.precision_kv.types import PrecisionState


REFLEX_INT4_LANDING_CONTRACT_KEYS = (
    "reflex_int4_landing_pages",
    "reflex_int4_landing_block_ids",
    "reflex_int4_landing_required_blocks",
    "reflex_int4_landing_planned_blocks",
    "reflex_int4_landing_reason",
    "reflex_int4_direct_landing",
)


def _request_kv_params(request: Any) -> dict[str, Any] | None:
    params = getattr(request, "kv_transfer_params", None)
    return params if isinstance(params, dict) else None


def has_reflex_int4_landing_contract(request: Any) -> bool:
    params = _request_kv_params(request)
    if params is None:
        return False
    landing_pages = params.get("reflex_int4_landing_pages")
    landing_block_ids = params.get("reflex_int4_landing_block_ids")
    return (
        isinstance(landing_pages, (list, tuple))
        and isinstance(landing_block_ids, (list, tuple))
        and len(landing_pages) > 0
        and len(landing_pages) == len(landing_block_ids)
    )


def clear_reflex_int4_landing_contract(request: Any) -> None:
    params = _request_kv_params(request)
    if params is None:
        return
    for key in REFLEX_INT4_LANDING_CONTRACT_KEYS:
        params.pop(key, None)


@dataclass(frozen=True)
class PrefixPrecisionVersion:
    prefix_id: str
    version_id: int
    owner_request_id: str
    page_indices: tuple[int, ...]
    precision: PrecisionState


@dataclass(frozen=True)
class PrefixPrecisionContract:
    prefix_id: str
    versions: tuple[PrefixPrecisionVersion, ...]


class PrefixPrecisionContractManager:
    """Control-plane ownership tracker for shared prefix precision state."""

    def __init__(self) -> None:
        self._versions_by_prefix: dict[str, list[PrefixPrecisionVersion]] = {}
        self._active_by_request: dict[tuple[str, str], int] = {}

    def register_shared_prefix(
        self,
        *,
        prefix_id: str,
        owner_request_id: str,
        page_indices: list[int] | tuple[int, ...],
        precision: PrecisionState,
    ) -> PrefixPrecisionVersion:
        version = PrefixPrecisionVersion(
            prefix_id=prefix_id,
            version_id=1,
            owner_request_id=owner_request_id,
            page_indices=tuple(int(page_idx) for page_idx in page_indices),
            precision=precision,
        )
        self._versions_by_prefix[prefix_id] = [version]
        self._active_by_request[(prefix_id, owner_request_id)] = version.version_id
        return version

    def contract(self, prefix_id: str) -> PrefixPrecisionContract | None:
        versions = self._versions_by_prefix.get(prefix_id)
        if not versions:
            return None
        return PrefixPrecisionContract(
            prefix_id=prefix_id,
            versions=tuple(versions),
        )

    def active_version(
        self,
        prefix_id: str,
        request_id: str,
    ) -> PrefixPrecisionVersion:
        versions = self._versions_by_prefix.get(prefix_id)
        if not versions:
            raise KeyError(f"Unknown prefix precision contract {prefix_id!r}.")
        active_id = self._active_by_request.get((prefix_id, request_id))
        if active_id is None:
            return versions[0]
        for version in versions:
            if version.version_id == active_id:
                return version
        raise RuntimeError(
            "Prefix precision contract points at a missing version: "
            f"prefix={prefix_id}, request={request_id}, version={active_id}."
        )

    def requires_copy_on_demote(
        self,
        *,
        prefix_id: str,
        request_id: str,
        page_idx: int,
    ) -> bool:
        version = self.active_version(prefix_id, request_id)
        if page_idx not in version.page_indices:
            return False
        return version.owner_request_id != request_id

    def copy_on_demote(
        self,
        *,
        prefix_id: str,
        request_id: str,
        page_indices: list[int] | tuple[int, ...],
        target_precision: PrecisionState,
    ) -> PrefixPrecisionVersion:
        versions = self._versions_by_prefix.get(prefix_id)
        if not versions:
            raise KeyError(f"Unknown prefix precision contract {prefix_id!r}.")
        new_version_id = max(version.version_id for version in versions) + 1
        version = PrefixPrecisionVersion(
            prefix_id=prefix_id,
            version_id=new_version_id,
            owner_request_id=request_id,
            page_indices=tuple(int(page_idx) for page_idx in page_indices),
            precision=target_precision,
        )
        versions.append(version)
        self._active_by_request[(prefix_id, request_id)] = new_version_id
        return version
