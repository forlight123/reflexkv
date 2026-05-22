#!/usr/bin/env python3
"""Diagnose ReFlexKV bottlenecks from a pressure or mixed summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _count(mapping: dict[str, Any] | None, key: str) -> int:
    if not isinstance(mapping, dict):
        return 0
    value = mapping.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _int(summary: dict[str, Any], key: str) -> int:
    value = summary.get(key, 0)
    return int(value) if isinstance(value, (int, float)) else 0


def _add(
    diagnostics: list[dict[str, Any]],
    *,
    area: str,
    severity: str,
    signal: int | float,
    finding: str,
    action: str,
) -> None:
    if signal <= 0:
        return
    diagnostics.append(
        {
            "area": area,
            "severity": severity,
            "signal": signal,
            "finding": finding,
            "action": action,
        }
    )


def _page_lifecycle_action(summary: dict[str, Any]) -> str:
    subsignals = {
        "remote_inflight": _int(
            summary,
            "candidate_remote_inflight_bf16_pages_total",
        ),
        "open_tail": _int(summary, "candidate_open_tail_bf16_pages_total"),
        "request_protected": _int(
            summary,
            "candidate_request_protected_bf16_pages_total",
        ),
        "shared": _int(summary, "candidate_shared_bf16_pages_total"),
        "prompt_protected": _int(
            summary,
            "candidate_prompt_protected_bf16_pages_total",
        ),
    }
    nonzero = [
        f"{name}={value}" for name, value in subsignals.items() if value > 0
    ]
    if not nonzero:
        return (
            "Audit chunk sealing, shared-prefix protection, and "
            "copy-on-demote contracts."
        )
    return (
        "Audit page lifecycle subsignals before changing policy: "
        + ", ".join(nonzero)
        + "."
    )


def diagnose_summary(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ordered engineering actions from structured ReFlexKV metrics."""

    blocked = summary.get("admission_blocked_reason_counts")
    rejection = summary.get("admission_blocked_frontier_rejection_reason_totals")
    if not isinstance(rejection, dict):
        rejection = summary.get("admission_frontier_rejection_reason_totals")
    persisted_landing = _int(summary, "landing_contract_persisted_pages_total")
    materialized_landing = _int(summary, "landing_materialized_pages_total")
    synthetic_pages = _int(summary, "page_metadata_plan_synthetic_pages_total")
    real_risk_pages = _int(summary, "page_metadata_plan_real_risk_pages_total")
    fallback_unmaterialized = _int(
        summary,
        "landing_fallback_unmaterialized_total",
    )

    diagnostics: list[dict[str, Any]] = []
    _add(
        diagnostics,
        area="direct_landing_materialization",
        severity="P0",
        signal=(
            max(0, persisted_landing - materialized_landing)
            + fallback_unmaterialized
        ),
        finding=(
            "mixed/direct landing contract was persisted, but fewer pages "
            "were materialized or the worker materialization signal was missing."
        ),
        action=(
            "Inspect Mooncake landing transfer, materialized req-id propagation, "
            "and landing commit accounting before tuning scheduler policy."
        ),
    )
    _add(
        diagnostics,
        area="p_side_risk_metadata",
        severity="P0",
        signal=synthetic_pages if synthetic_pages > real_risk_pages else 0,
        finding="D-side planner is relying more on synthetic chunk fallback than real P-side risk.",
        action=(
            "Check prefill page metadata production, connector aggregation, and "
            "risk metadata arrival timing."
        ),
    )
    _add(
        diagnostics,
        area="chunk_admission",
        severity="P0",
        signal=(
            _count(blocked, "full_sequence_reserve")
            + _count(blocked, "mixed_landing_requires_bf16_staging")
        ),
        finding="admission is still dominated by full-sequence or BF16 staging pressure.",
        action="Move more of the waiting path to explicit chunk/partial admission.",
    )
    _add(
        diagnostics,
        area="request_precision_budget",
        severity="P1",
        signal=(
            _count(rejection, "request_fraction_cap")
            + _count(rejection, "request_release_budget")
            + _count(rejection, "quality_debt_cap")
            + _count(rejection, "request_budget")
        ),
        finding="request-level budget is the dominant feasible-frontier limiter.",
        action="Split prompt/decode budgets and relax low-risk tiers under admission pressure.",
    )
    _add(
        diagnostics,
        area="sparse_window_quota",
        severity="P1",
        signal=_count(rejection, "sparse_quota"),
        finding="sparse window quota is limiting releasable pages.",
        action="Tune window size and per-window demotion count using mixed workload traces.",
    )
    _add(
        diagnostics,
        area="frontier_dual_optimizer",
        severity="P1",
        signal=_count(rejection, "frontier_optimizer"),
        finding="frontier optimizer is pruning otherwise feasible candidates.",
        action="Inspect reduced-cost weights for memory, admission, quality debt, and migration backlog.",
    )
    _add(
        diagnostics,
        area="page_lifecycle",
        severity="P1",
        signal=_count(rejection, "shared_or_open"),
        finding="many pages are blocked because they are shared/open.",
        action=_page_lifecycle_action(summary),
    )

    severity_rank = {"P0": 0, "P1": 1, "P2": 2}
    area_rank = {
        "direct_landing_materialization": 0,
        "p_side_risk_metadata": 1,
        "chunk_admission": 2,
        "request_precision_budget": 3,
        "sparse_window_quota": 4,
        "frontier_dual_optimizer": 5,
        "page_lifecycle": 6,
    }
    diagnostics.sort(
        key=lambda item: (
            severity_rank.get(str(item["severity"]), 9),
            area_rank.get(str(item["area"]), 99),
            -float(item["signal"]),
        )
    )
    return diagnostics


def load_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"summary must be a JSON object: {path}")
    if "reflex_trace" in payload and isinstance(payload["reflex_trace"], dict):
        return payload["reflex_trace"]
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose ReFlexKV blocked reasons from summary JSON."
    )
    parser.add_argument("summary", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    diagnostics = diagnose_summary(load_summary(args.summary))
    text = json.dumps(diagnostics, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
