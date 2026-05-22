from vllm.v1.core.precision_kv.policy import (
    CandidateFunnelSnapshot,
    PrecisionKVPolicy,
    PrecisionPressureState,
    RequestPolicyState,
)


def test_policy_expands_target_and_window_when_budget_and_sparse_cap_dominate():
    policy = PrecisionKVPolicy()

    decision = policy.plan_pressure(
        PrecisionPressureState(
            reason="admission_waiting",
            target_bf16_blocks=128,
            free_bf16_blocks=64,
            total_bf16_blocks=4096,
            waiting_requests=1,
            candidate_funnel=CandidateFunnelSnapshot(
                after_initial_recent_protection=185607,
                after_low_risk_filter=40273,
                after_request_budget_cap=680,
                after_sparse_window_quota=432,
                after_int4_pool_limit=432,
            ),
        )
    )

    assert decision.pressure_active is True
    assert decision.target_release_blocks > 128
    assert decision.request_release_budget_multiplier > 1.0
    assert decision.max_demote_per_window_multiplier > 1.0
    assert "request_budget_cap" in decision.policy_reasons
    assert "sparse_window_quota" in decision.policy_reasons


def test_policy_scales_relaxation_with_candidate_funnel_severity():
    policy = PrecisionKVPolicy()

    decision = policy.plan_pressure(
        PrecisionPressureState(
            reason="admission_waiting",
            target_bf16_blocks=64,
            free_bf16_blocks=16,
            total_bf16_blocks=4096,
            waiting_requests=4,
            candidate_funnel=CandidateFunnelSnapshot(
                after_initial_recent_protection=8192,
                after_low_risk_filter=4096,
                after_request_budget_cap=128,
                after_sparse_window_quota=16,
                after_frontier_optimizer=16,
                after_int4_pool_limit=16,
            ),
        )
    )

    assert decision.request_release_budget_multiplier >= 4.0
    assert decision.max_demote_per_window_multiplier >= 4.0
    assert decision.target_release_blocks >= 256


def test_policy_reports_frontier_optimizer_without_blaming_sparse_quota():
    policy = PrecisionKVPolicy()

    decision = policy.plan_pressure(
        PrecisionPressureState(
            reason="admission_waiting",
            target_bf16_blocks=64,
            free_bf16_blocks=32,
            total_bf16_blocks=4096,
            waiting_requests=1,
            candidate_funnel=CandidateFunnelSnapshot(
                after_initial_recent_protection=512,
                after_low_risk_filter=256,
                after_request_budget_cap=128,
                after_sparse_window_quota=128,
                after_frontier_optimizer=16,
                after_int4_pool_limit=16,
            ),
        )
    )

    assert decision.target_release_blocks > 64
    assert "frontier_optimizer" in decision.policy_reasons
    assert "sparse_window_quota" not in decision.policy_reasons


def test_policy_relaxes_low_risk_fraction_only_under_pressure():
    policy = PrecisionKVPolicy()

    low_pressure = policy.plan_pressure(
        PrecisionPressureState(
            reason="background_pressure",
            target_bf16_blocks=0,
            free_bf16_blocks=2048,
            total_bf16_blocks=4096,
            waiting_requests=0,
            candidate_funnel=CandidateFunnelSnapshot(
                after_initial_recent_protection=1000,
                after_low_risk_filter=100,
                after_request_budget_cap=100,
                after_sparse_window_quota=100,
                after_int4_pool_limit=100,
            ),
            base_low_risk_score_fraction=0.25,
        )
    )
    high_pressure = policy.plan_pressure(
        PrecisionPressureState(
            reason="admission_waiting",
            target_bf16_blocks=128,
            free_bf16_blocks=64,
            total_bf16_blocks=4096,
            waiting_requests=1,
            candidate_funnel=CandidateFunnelSnapshot(
                after_initial_recent_protection=1000,
                after_low_risk_filter=100,
                after_request_budget_cap=100,
                after_sparse_window_quota=100,
                after_int4_pool_limit=100,
            ),
            base_low_risk_score_fraction=0.25,
        )
    )

    assert low_pressure.low_risk_score_fraction == 0.25
    assert high_pressure.low_risk_score_fraction > 0.25
    assert "low_risk_filter" in high_pressure.policy_reasons


def test_policy_keeps_reasoning_prompt_budget_lower_than_long_context_budget():
    policy = PrecisionKVPolicy()
    pressure = PrecisionPressureState(
        reason="admission_waiting",
        target_bf16_blocks=256,
        free_bf16_blocks=64,
        total_bf16_blocks=4096,
        waiting_requests=1,
    )

    math_like = policy.plan_request_budget(
        pressure,
        RequestPolicyState(
            request_id="math-like",
            page_count=40,
            prompt_pages=8,
            protected_prompt_pages=8,
            generated_decode_tokens=64,
            remaining_decode_tokens=2048,
            priority=1,
            base_max_int4_fraction=0.5,
            is_short_decode=False,
        ),
    )
    long_context = policy.plan_request_budget(
        pressure,
        RequestPolicyState(
            request_id="long-context",
            page_count=1024,
            prompt_pages=1024,
            protected_prompt_pages=0,
            generated_decode_tokens=256,
            remaining_decode_tokens=512,
            priority=0,
            base_max_int4_fraction=0.5,
            is_short_decode=False,
        ),
    )

    assert math_like.protected_prompt_pages == 8
    assert long_context.max_int4_pages > math_like.max_int4_pages
    assert long_context.priority > math_like.priority
