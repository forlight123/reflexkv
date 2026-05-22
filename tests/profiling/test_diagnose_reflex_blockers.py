from scripts.profiling.diagnose_reflex_blockers import diagnose_summary


def test_diagnose_summary_prioritizes_landing_and_metadata_race():
    diagnostics = diagnose_summary(
        {
            "admission_planned_int4_landing_total": 128,
            "landing_contract_persisted_pages_total": 128,
            "landing_materialized_pages_total": 0,
            "landing_fallback_unmaterialized_total": 4,
            "page_metadata_plan_synthetic_pages_total": 256,
            "page_metadata_plan_real_risk_pages_total": 0,
            "admission_blocked_reason_counts": {
                "mixed_landing_requires_bf16_staging": 12,
                "full_sequence_reserve": 8,
            },
            "admission_frontier_rejection_reason_totals": {
                "request_budget": 5,
                "sparse_quota": 3,
            },
        }
    )

    assert diagnostics[0]["area"] == "direct_landing_materialization"
    assert diagnostics[0]["severity"] == "P0"
    assert diagnostics[1]["area"] == "p_side_risk_metadata"
    assert any(item["area"] == "chunk_admission" for item in diagnostics)


def test_diagnose_summary_does_not_treat_admission_trial_landing_as_contract():
    diagnostics = diagnose_summary(
        {
            "admission_planned_int4_landing_total": 512,
            "landing_contract_persisted_pages_total": 0,
            "landing_materialized_pages_total": 0,
            "landing_fallback_unmaterialized_total": 0,
            "page_metadata_plan_synthetic_pages_total": 0,
            "page_metadata_plan_real_risk_pages_total": 512,
            "admission_blocked_reason_counts": {},
            "admission_frontier_rejection_reason_totals": {},
        }
    )

    assert all(
        item["area"] != "direct_landing_materialization"
        for item in diagnostics
    )


def test_diagnose_summary_uses_blocked_only_frontier_rejection_when_available():
    diagnostics = diagnose_summary(
        {
            "admission_frontier_rejection_reason_totals": {
                "request_budget": 1000,
                "sparse_quota": 500,
            },
            "admission_blocked_frontier_rejection_reason_totals": {
                "request_budget": 0,
                "sparse_quota": 0,
            },
            "admission_blocked_reason_counts": {},
        }
    )

    assert all(
        item["area"] not in {"request_precision_budget", "sparse_window_quota"}
        for item in diagnostics
    )


def test_diagnose_summary_maps_budget_and_sparse_rejections():
    diagnostics = diagnose_summary(
        {
            "admission_planned_int4_landing_total": 0,
            "landing_materialized_pages_total": 0,
            "page_metadata_plan_synthetic_pages_total": 0,
            "page_metadata_plan_real_risk_pages_total": 512,
            "admission_blocked_reason_counts": {},
            "admission_frontier_rejection_reason_totals": {
                "request_fraction_cap": 100,
                "request_release_budget": 80,
                "sparse_quota": 60,
                "frontier_optimizer": 40,
                "shared_or_open": 20,
            },
        }
    )

    assert [item["area"] for item in diagnostics[:4]] == [
        "request_precision_budget",
        "sparse_window_quota",
        "frontier_dual_optimizer",
        "page_lifecycle",
    ]


def test_diagnose_summary_reports_page_lifecycle_subsignals():
    diagnostics = diagnose_summary(
        {
            "admission_blocked_frontier_rejection_reason_totals": {
                "shared_or_open": 100,
            },
            "candidate_remote_inflight_bf16_pages_total": 12,
            "candidate_open_tail_bf16_pages_total": 3,
            "candidate_request_protected_bf16_pages_total": 4,
            "candidate_shared_bf16_pages_total": 5,
            "candidate_prompt_protected_bf16_pages_total": 6,
        }
    )

    page_lifecycle = next(
        item for item in diagnostics if item["area"] == "page_lifecycle"
    )
    assert "remote_inflight=12" in page_lifecycle["action"]
    assert "open_tail=3" in page_lifecycle["action"]
    assert "shared=5" in page_lifecycle["action"]
