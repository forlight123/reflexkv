# SPDX-License-Identifier: Apache-2.0
"""Mooncake connector name used by ReFlexKV PD experiments.

ReFlexKV precision changes are now confined to the decode worker. The P/D
handoff stays on the standard Mooncake path and transfers the primary KV cache
without handoff-time INT4 compression.
"""

from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.mooncake_connector import (
    MooncakeConnector,
)


class ReFlexMooncakeConnector(MooncakeConnector):
    """Compatibility alias for the standard Mooncake connector."""

    pass
