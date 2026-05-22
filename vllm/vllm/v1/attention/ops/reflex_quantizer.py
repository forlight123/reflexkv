# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Risk-aware INT4 quantizer primitives for ReFlexKV research paths."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QuantizedTensorCapsule:
    """INT4 payload plus optional residual compensation."""

    qvalues: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, ...]
    group_size: int
    residual_indices: torch.Tensor
    residual_values: torch.Tensor
    risk_score: float = 0.0


class RiskAwareInt4Quantizer:
    """Groupwise symmetric INT4 quantizer with residual capsules.

    The runtime CUDA codec still owns the packed production data path. This
    class is the swappable quantizer boundary used for algorithmic ablations and
    recovery/compensation experiments.
    """

    def __init__(
        self,
        *,
        group_size: int = 32,
        residual_topk: int = 0,
        residual_risk_floor: float = 0.5,
        eps: float = 1.0e-6,
    ) -> None:
        if group_size <= 0:
            raise ValueError("group_size must be positive.")
        if residual_topk < 0:
            raise ValueError("residual_topk must be non-negative.")
        self.group_size = int(group_size)
        self.residual_topk = int(residual_topk)
        self.residual_risk_floor = float(residual_risk_floor)
        self.eps = float(eps)

    def quantize_tensor(
        self,
        tensor: torch.Tensor,
        *,
        risk_score: float = 0.0,
    ) -> QuantizedTensorCapsule:
        values = tensor.detach().float()
        shape = tuple(values.shape)
        flat = values.reshape(-1)
        padded, original_numel = self._pad_to_groups(flat)
        grouped = padded.reshape(-1, self.group_size)
        amax = grouped.abs().amax(dim=1)
        scales = torch.clamp(amax / 7.0, min=self.eps)
        qvalues = torch.round(grouped / scales[:, None]).clamp(-8, 7).to(
            torch.int8
        )
        restored = (qvalues.float() * scales[:, None]).reshape(-1)[
            :original_numel
        ]
        residual_indices, residual_values = self._build_residual(
            original=flat,
            restored=restored,
            risk_score=risk_score,
        )
        return QuantizedTensorCapsule(
            qvalues=qvalues,
            scales=scales,
            shape=shape,
            group_size=self.group_size,
            residual_indices=residual_indices,
            residual_values=residual_values,
            risk_score=float(risk_score),
        )

    def dequantize_tensor(self, capsule: QuantizedTensorCapsule) -> torch.Tensor:
        restored = (capsule.qvalues.float() * capsule.scales[:, None]).reshape(-1)
        numel = 1
        for dim in capsule.shape:
            numel *= int(dim)
        restored = restored[:numel].clone()
        if capsule.residual_indices.numel() > 0:
            restored[capsule.residual_indices.long()] += capsule.residual_values
        return restored.reshape(capsule.shape)

    def _pad_to_groups(self, flat: torch.Tensor) -> tuple[torch.Tensor, int]:
        original_numel = flat.numel()
        if original_numel == 0:
            return flat.new_zeros((self.group_size,)), 0
        remainder = original_numel % self.group_size
        if remainder == 0:
            return flat, original_numel
        pad = self.group_size - remainder
        return torch.cat([flat, flat.new_zeros((pad,))], dim=0), original_numel

    def _build_residual(
        self,
        *,
        original: torch.Tensor,
        restored: torch.Tensor,
        risk_score: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.residual_topk <= 0 or original.numel() == 0:
            return (
                torch.empty((0,), dtype=torch.long, device=original.device),
                torch.empty((0,), dtype=original.dtype, device=original.device),
            )
        if risk_score < self.residual_risk_floor:
            return (
                torch.empty((0,), dtype=torch.long, device=original.device),
                torch.empty((0,), dtype=original.dtype, device=original.device),
            )
        budget = min(self.residual_topk, original.numel())
        residual = original - restored
        indices = torch.topk(residual.abs(), k=budget).indices.long()
        return indices, residual[indices].to(original.dtype)
