from types import SimpleNamespace

from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.precision_kv.landing import PrecisionLandingDecision
from vllm.v1.core.precision_kv.controller import PrecisionAdmissionDecision
from vllm.v1.core.precision_kv.frontier import (
    FeasibleFrontierCache,
    FeasibleFrontierSummary,
)
from vllm.v1.core.precision_kv.run_optimizer import DualPriceState
from vllm.v1.core.precision_kv.demotion_planner import ReflexCandidateBreakdown
from vllm.v1.core.precision_kv.types import (
    KVPageLifecycle,
    KVPageRuntimeDescriptor,
    MemoryTier,
    PrecisionState,
    RecoveryClass,
    ReflexDemotion,
)
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler


class _FakeBlockPool:

    def __init__(self, *, free_blocks: int, total_blocks: int):
        self._free_blocks = free_blocks
        self.num_gpu_blocks = total_blocks

    def get_num_free_blocks(self):
        return self._free_blocks


class _FakeBlocks:

    def __init__(self, block_ids: list[int] | None = None):
        self.block_ids = block_ids or []
        self.blocks = [self.block_ids] if self.block_ids else [[]]

    def get_block_ids(self, allow_none: bool = False):
        return (self.block_ids,)


class _FakeReflexManager:

    def __init__(self, *, release_blocks: int, stale_raw_pages: int = 0):
        self.release_blocks = release_blocks
        self.targets = []
        self._last_breakdown = ReflexCandidateBreakdown(
            raw_bf16_pages=stale_raw_pages,
            after_int4_pool_limit=stale_raw_pages,
        )

    def plan_reflex_int4_demotions(self, *, target_bf16_blocks: int, **kwargs):
        self.targets.append(target_bf16_blocks)
        if target_bf16_blocks <= 0:
            self._last_breakdown = ReflexCandidateBreakdown()
            return 0
        released = min(self.release_blocks, target_bf16_blocks)
        self._last_breakdown = ReflexCandidateBreakdown(
            raw_bf16_pages=released,
            after_int4_pool_limit=released,
            selected_actual=released,
        )
        return released

    def get_last_reflex_int4_candidate_capacity(self):
        return self._last_breakdown.after_int4_pool_limit

    def get_last_reflex_int4_candidate_breakdown(self):
        return self._last_breakdown


class _FakeKVCacheCoordinator(KVCacheCoordinator):

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        return (), 0


def _make_scheduler_for_reflex_target(
    *,
    free_blocks: int,
    total_blocks: int = 4096,
    block_size: int = 16,
    max_num_scheduled_tokens: int = 8192,
    max_model_len: int = 32768,
):
    scheduler = object.__new__(Scheduler)
    scheduler.cache_config = SimpleNamespace(cache_dtype="reflex_int4")
    scheduler.block_size = block_size
    scheduler.max_num_scheduled_tokens = max_num_scheduled_tokens
    scheduler.max_model_len = max_model_len
    scheduler.scheduler_reserve_full_isl = True
    scheduler.connector = None
    scheduler.kv_cache_manager = SimpleNamespace(
        block_pool=_FakeBlockPool(
            free_blocks=free_blocks,
            total_blocks=total_blocks,
        )
    )
    scheduler._reflex_int4_background_demotions_per_step = 16
    scheduler._reflex_int4_background_min_demotions_per_step = 8
    scheduler._reflex_int4_background_free_floor_blocks = 32
    scheduler._reflex_int4_fast_demotions_per_step = (
        max(max_num_scheduled_tokens, max_model_len) + block_size - 1
    ) // block_size
    scheduler._reflex_int4_max_demotions_per_step = 16
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_prev_step_had_prefill = False
    scheduler._reflex_int4_low_watermark = 0.05
    scheduler._reflex_int4_high_watermark = 0.10
    scheduler._reflex_int4_demote_cooldown_steps = 4
    scheduler._reflex_int4_admission_reserve_blocks = 32
    scheduler._reflex_int4_keep_recent_pages = 4
    scheduler._reflex_int4_keep_initial_pages = 4
    scheduler._reflex_int4_max_int4_fraction_per_request = 1.0
    scheduler._reflex_int4_short_decode_tokens = 128
    scheduler._reflex_int4_short_decode_max_int4_fraction = 0.0
    scheduler._reflex_int4_short_admission_max_int4_fraction = 0.03
    scheduler._reflex_int4_risk_warmup_tokens = block_size
    scheduler._reflex_int4_survival_warmup_tokens = 128
    scheduler._reflex_int4_sparse_window_pages = 32
    scheduler._reflex_int4_short_max_demote_per_window = 1
    scheduler._reflex_int4_max_demote_per_window = 2
    scheduler._reflex_int4_admission_sparse_window_pages = 32
    scheduler._reflex_int4_admission_max_demote_per_window = 8
    scheduler._reflex_int4_admission_pressure_min_int4_fraction = 0.10
    scheduler._reflex_int4_admission_landing_max_int4_fraction = 0.75
    scheduler._reflex_int4_low_risk_score_fraction = 0.25
    scheduler._reflex_int4_slo_pressure_step = 0.25
    scheduler._reflex_int4_min_slo_pressure = 0.5
    scheduler._reflex_int4_max_slo_pressure = 1.5
    scheduler._reflex_int4_cold_admission_max_int4_fraction = 0.10
    scheduler._reflex_int4_cold_admission_emergency_free_ratio = 0.05
    scheduler._reflex_int4_decode_pressure_warmup_tokens = 32
    scheduler._reflex_int4_decode_pressure_ramp_tokens = 512
    scheduler._reflex_int4_short_prefill_pages = 64
    scheduler._reflex_int4_long_prefill_pages = 512
    scheduler._reflex_int4_global_evidence_min_prompt_pages = 512
    scheduler._reflex_int4_global_evidence_min_decode_tokens = 129
    scheduler._reflex_int4_global_evidence_landing_max_int4_fraction = 0.08
    scheduler._reflex_int4_reasoning_prompt_protection_max_pages = 64
    scheduler._reflex_int4_reasoning_prompt_protection_min_decode_tokens = 1024
    scheduler._reflex_int4_page_level_protection_enabled = True
    scheduler._reflex_int4_long_prompt_protected_head_pages = 4
    scheduler._reflex_int4_long_prompt_protected_tail_pages = 4
    scheduler._reflex_int4_prompt_high_risk_protection_threshold = 0.85
    scheduler._reflex_int4_page_selection_policy = "relevance_sparse"
    return scheduler


class _FakeWaitingKVManager:

    def __init__(
        self,
        *,
        blocked_request_id: str,
        free_blocks: int = 0,
        allow_blocked_allocation: bool = False,
    ):
        self.block_pool = _FakeBlockPool(free_blocks=free_blocks,
                                         total_blocks=4096)
        self.empty_kv_cache_blocks = _FakeBlocks()
        self.blocked_request_id = blocked_request_id
        self.allow_blocked_allocation = allow_blocked_allocation
        self.allocated_request_ids = []
        self.blocks_by_request_id: dict[str, _FakeBlocks] = {}

    def new_step_starts(self):
        pass

    def get_computed_blocks(self, request):
        return self.empty_kv_cache_blocks, 0

    def can_fit_full_sequence(self, request, **kwargs):
        return request.request_id != self.blocked_request_id

    def allocate_slots(self, request, num_new_tokens, **kwargs):
        self.allocated_request_ids.append(request.request_id)
        if (
            request.request_id == self.blocked_request_id
            and not self.allow_blocked_allocation
        ):
            raise AssertionError("blocked request must not allocate slots")
        block_ids = [len(self.allocated_request_ids)]
        blocks = _FakeBlocks(block_ids)
        self.blocks_by_request_id[request.request_id] = blocks
        return blocks

    def get_blocks(self, request_id):
        return self.blocks_by_request_id.get(request_id, _FakeBlocks())

    def get_num_common_prefix_blocks(self, request_id):
        return [0]

    def take_reflex_int4_demotions(self):
        return []

    def take_reflex_int4_recoveries(self):
        return []

    def has_reflex_int4_blocks(self, request_id):
        return False


def _make_waiting_request(
    request_id: str,
    *,
    num_prompt_tokens: int,
) -> SimpleNamespace:
    prompt_token_ids = list(range(num_prompt_tokens))
    return SimpleNamespace(
        request_id=request_id,
        status=scheduler_module.RequestStatus.WAITING,
        is_prefill_chunk=False,
        num_tokens=num_prompt_tokens,
        num_prompt_tokens=num_prompt_tokens,
        num_computed_tokens=0,
        num_external_computed_tokens=0,
        num_cached_tokens=-1,
        num_preemptions=0,
        prompt_token_ids=prompt_token_ids,
        all_token_ids=prompt_token_ids.copy(),
        _all_token_ids=prompt_token_ids.copy(),
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        lora_request=None,
        prompt_embeds=None,
        has_encoder_inputs=False,
        kv_transfer_params={},
        spec_token_ids=[],
        num_output_placeholders=0,
        num_output_tokens=0,
        output_token_ids=[],
        max_tokens=32,
        priority=0,
        arrival_time=0,
    )


def _make_scheduler_for_waiting_admission() -> Scheduler:
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=0,
        max_num_scheduled_tokens=2048,
    )
    scheduler.cache_config.block_size = scheduler.block_size
    scheduler.scheduler_config = SimpleNamespace(
        long_prefill_token_threshold=0,
        enable_chunked_prefill=True,
        async_scheduling=False,
    )
    scheduler.policy = scheduler_module.SchedulingPolicy.FCFS
    scheduler.waiting = scheduler_module.create_request_queue(scheduler.policy)
    scheduler.skipped_waiting = scheduler_module.create_request_queue(
        scheduler.policy)
    scheduler.running = []
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_encoder_input_tokens = 0
    scheduler._pause_state = scheduler_module.PauseState.UNPAUSED
    scheduler.finished_req_ids = set()
    scheduler.prev_step_scheduled_req_ids = set()
    scheduler.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: [],
        allocate=lambda request, input_id: None,
        free=lambda request: None,
    )
    scheduler.lora_config = None
    scheduler.use_eagle = False
    scheduler.need_mamba_block_aligned_split = False
    scheduler.num_lookahead_tokens = 0
    scheduler.is_encoder_decoder = False
    scheduler.use_v2_model_runner = False
    scheduler.use_pp = False
    scheduler.log_stats = False
    scheduler.ec_connector = None
    scheduler.connector_prefix_cache_stats = None
    scheduler.needs_kv_cache_zeroing = False
    scheduler._update_after_schedule = lambda scheduler_output: None
    scheduler._plan_and_persist_reflex_int4_landing_contract = (
        lambda **kwargs: None)
    scheduler._try_reflex_int4_demote = (
        lambda *, target_bf16_blocks, force=False, reason="pressure": 0)
    scheduler._try_reflex_int4_background_promote = lambda: 0
    return scheduler


def test_reflex_int4_defers_blocked_full_sequence_reserve_head_request():
    scheduler = _make_scheduler_for_waiting_admission()
    scheduler.kv_cache_manager = _FakeWaitingKVManager(
        blocked_request_id="long-blocked")
    long_request = _make_waiting_request(
        "long-blocked",
        num_prompt_tokens=24000,
    )
    short_request = _make_waiting_request(
        "math-short",
        num_prompt_tokens=256,
    )
    scheduler.requests = {
        long_request.request_id: long_request,
        short_request.request_id: short_request,
    }
    scheduler.waiting.add_request(long_request)
    scheduler.waiting.add_request(short_request)

    scheduler_output = scheduler.schedule()

    assert [req.req_id for req in scheduler_output.scheduled_new_reqs] == [
        "math-short"
    ]
    assert scheduler.kv_cache_manager.allocated_request_ids == ["math-short"]
    assert [request.request_id for request in scheduler.running] == [
        "math-short"
    ]
    assert [request.request_id for request in scheduler.skipped_waiting] == [
        "long-blocked"
    ]


def test_reflex_int4_full_sequence_reserve_defers_residual_mixed_landing():
    scheduler = _make_scheduler_for_waiting_admission()
    scheduler.kv_cache_manager = _FakeWaitingKVManager(
        blocked_request_id="long-mixed-landing",
        free_blocks=200,
        allow_blocked_allocation=True,
    )
    request = _make_waiting_request(
        "long-mixed-landing",
        num_prompt_tokens=12000,
    )
    scheduler.requests = {request.request_id: request}
    scheduler.waiting.add_request(request)
    landing_decision = PrecisionLandingDecision(
        request_id=request.request_id,
        required_blocks=712,
        bf16_deficit_blocks=512,
        residual_deficit_after_running=280,
        eligible_int4_landing_blocks=280,
        planned_int4_landing_blocks=280,
        planned_int4_landing_pages=tuple(range(280)),
        admission_feasible_with_landing=True,
        mixed_landing_required=True,
        reason="emergency_mixed_landing_feasible",
    )

    def fake_plan_landing(**kwargs):
        request.kv_transfer_params["reflex_int4_landing_pages"] = list(
            landing_decision.planned_int4_landing_pages
        )
        request.kv_transfer_params["reflex_int4_landing_block_ids"] = list(
            range(8000, 8000 + landing_decision.planned_int4_landing_blocks)
        )
        return landing_decision

    scheduler._plan_and_persist_reflex_int4_landing_contract = fake_plan_landing

    scheduler_output = scheduler.schedule()

    assert scheduler_output.scheduled_new_reqs == []
    assert scheduler.kv_cache_manager.allocated_request_ids == []
    assert scheduler.running == []
    assert [request.request_id for request in scheduler.skipped_waiting] == [
        "long-mixed-landing"
    ]


def test_reflex_remote_chunk_admission_bypasses_full_sequence_reserve():
    scheduler = _make_scheduler_for_waiting_admission()
    scheduler.kv_cache_manager = _FakeWaitingKVManager(
        blocked_request_id="remote-chunk",
        free_blocks=64,
        allow_blocked_allocation=True,
    )

    class _Connector:

        def __init__(self):
            self.queries = []
            self.allocs = []

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            self.queries.append((request.request_id, num_computed_tokens))
            return 512, True

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            self.allocs.append((request.request_id, num_external_tokens))

        def build_connector_meta(self, scheduler_output):
            return SimpleNamespace()

    connector = _Connector()
    scheduler.connector = connector
    request = _make_waiting_request(
        "remote-chunk",
        num_prompt_tokens=2048,
    )
    request.num_computed_tokens = 512
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
    }
    scheduler.requests = {request.request_id: request}
    scheduler.waiting.add_request(request)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.scheduled_new_reqs == []
    assert connector.queries == [("remote-chunk", 512)]
    assert connector.allocs == [("remote-chunk", 512)]
    assert request.status == scheduler_module.RequestStatus.WAITING_FOR_REMOTE_KVS
    assert request.num_computed_tokens == 1024
    assert scheduler.kv_cache_manager.allocated_request_ids == ["remote-chunk"]


def test_reflex_remote_chunk_admission_target_uses_chunk_frontier():
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=0,
        block_size=16,
        max_model_len=8192,
    )
    request = _make_waiting_request(
        "remote-chunk-target",
        num_prompt_tokens=4096,
    )
    request.num_computed_tokens = 512
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
        "reflex_remote_chunk_token_end": 1024,
    }

    needed_blocks = scheduler._estimate_reflex_admission_needed_blocks(request)

    assert needed_blocks == 64


def test_reflex_remote_producer_prefill_is_capped_to_chunk_size():
    scheduler = _make_scheduler_for_waiting_admission()
    scheduler.kv_cache_manager = _FakeWaitingKVManager(
        blocked_request_id="not-blocked",
        free_blocks=256,
        allow_blocked_allocation=True,
    )

    class _Connector:

        def __init__(self):
            self.chunk_sends = []

        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 0, False

        def update_reflex_remote_decode_chunk_after_alloc(
            self,
            request,
            blocks,
            num_scheduled_tokens,
        ):
            self.chunk_sends.append((request.request_id, num_scheduled_tokens))

        def update_state_after_alloc(self, request, blocks, num_external_tokens):
            pass

        def build_connector_meta(self, scheduler_output):
            return SimpleNamespace()

    connector = _Connector()
    scheduler.connector = connector
    request = _make_waiting_request(
        "producer-prefill",
        num_prompt_tokens=2048,
    )
    request.kv_transfer_params = {
        "do_remote_decode": True,
        "transfer_id": "xfer-producer",
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
    }
    scheduler.requests = {request.request_id: request}
    scheduler.waiting.add_request(request)

    scheduler_output = scheduler.schedule()

    assert scheduler_output.num_scheduled_tokens == {"producer-prefill": 512}
    assert connector.chunk_sends == [("producer-prefill", 512)]


def test_reflex_int4_admission_demotion_step_does_not_persist_staged_landing_before_remote_wait():
    scheduler = _make_scheduler_for_waiting_admission()
    scheduler.kv_cache_manager = _FakeWaitingKVManager(
        blocked_request_id="remote-long",
        free_blocks=900,
    )
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = (
        lambda **kwargs: 0
    )
    scheduler.kv_cache_manager.reserve_reflex_int4_landing_blocks = (
        lambda request_id, count: list(range(8000, 8000 + count))
    )
    scheduler.running = [SimpleNamespace(is_prefill_chunk=False)]
    request = _make_waiting_request(
        "remote-long",
        num_prompt_tokens=1024 * scheduler.block_size,
    )
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "reflex_page_risks": [0.1] * 1024,
    }
    scheduler.requests = {request.request_id: request}
    scheduler.waiting.add_request(request)

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is None
    assert "reflex_int4_landing_required_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_planned_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params


def test_reflex_int4_preemption_prefers_bf16_only_running_request():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler.policy = scheduler_module.SchedulingPolicy.PRIORITY
    mixed_req = SimpleNamespace(
        request_id="mixed-running",
        priority=10,
        arrival_time=10,
        kv_transfer_params={},
    )
    bf16_req = SimpleNamespace(
        request_id="bf16-running",
        priority=0,
        arrival_time=0,
        kv_transfer_params={},
    )
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "mixed-running"
    )

    assert scheduler._select_preemption_victim([bf16_req, mixed_req]) is bf16_req


def test_reflex_int4_preemption_defers_when_all_running_requests_are_protected():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler.policy = scheduler_module.SchedulingPolicy.FCFS
    int4_req = SimpleNamespace(
        request_id="int4-running",
        priority=0,
        arrival_time=0,
        kv_transfer_params={},
    )
    landing_req = SimpleNamespace(
        request_id="landing-running",
        priority=0,
        arrival_time=1,
        kv_transfer_params={
            "reflex_int4_landing_pages": [0],
            "reflex_int4_landing_block_ids": [8000],
        },
    )
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "int4-running"
    )

    assert scheduler._select_preemption_victim([int4_req, landing_req]) is None


def test_reflex_int4_preemption_can_fallback_to_int4_request_under_hard_pressure():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler.policy = scheduler_module.SchedulingPolicy.FCFS
    int4_req = SimpleNamespace(
        request_id="int4-running",
        priority=0,
        arrival_time=0,
        kv_transfer_params={},
    )
    landing_req = SimpleNamespace(
        request_id="landing-running",
        priority=0,
        arrival_time=1,
        kv_transfer_params={
            "reflex_int4_landing_pages": [0],
            "reflex_int4_landing_block_ids": [8000],
        },
    )
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "int4-running"
    )

    victim = scheduler._select_preemption_victim(
        [int4_req, landing_req],
        allow_reflex_int4_protected=True,
    )

    assert victim is int4_req


def test_reflex_int4_coordinator_resets_unvisited_candidate_breakdown():
    coordinator = object.__new__(_FakeKVCacheCoordinator)
    first = _FakeReflexManager(release_blocks=1)
    second = _FakeReflexManager(release_blocks=4, stale_raw_pages=99)
    coordinator.single_type_managers = (first, second)

    released = coordinator.plan_reflex_int4_demotions(target_bf16_blocks=1)

    assert released == 1
    assert first.targets == [1]
    assert second.targets == [0]
    breakdown = coordinator.get_last_reflex_int4_candidate_breakdown()
    assert breakdown.raw_bf16_pages == 1
    assert breakdown.after_int4_pool_limit == 1


def test_reflex_int4_kv_cache_manager_forwards_dry_run_to_coordinator():
    manager = object.__new__(KVCacheManager)
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 7

    manager.coordinator = SimpleNamespace(
        plan_reflex_int4_demotions=fake_plan,
    )

    released = manager.plan_reflex_int4_demotions(
        target_bf16_blocks=11,
        dry_run=True,
    )

    assert released == 7
    assert captured["target_bf16_blocks"] == 11
    assert captured["dry_run"] is True


def test_reflex_int4_fast_target_uses_waiting_chunk_deficit():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)

    target = scheduler._estimate_reflex_demote_target(
        8192,
        force_allocate_failure=True,
    )

    assert target == 512


def test_reflex_int4_background_target_remains_small_when_below_free_floor():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=8)

    target = scheduler._estimate_reflex_demote_target(
        1,
        force_allocate_failure=False,
    )

    assert target == 16


def test_reflex_int4_background_target_skips_when_decode_buffer_is_sufficient():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=100)

    target = scheduler._estimate_reflex_demote_target(
        1,
        force_allocate_failure=False,
    )

    assert target == 0


def test_reflex_int4_admission_target_uses_waiting_need_plus_reserve():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=100)
    request = SimpleNamespace(num_tokens=4096, num_computed_tokens=0)

    target = scheduler._estimate_reflex_admission_demote_target(request)

    assert target == 188


def test_reflex_int4_admission_target_is_zero_when_next_request_fits():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=300)
    request = SimpleNamespace(num_tokens=4096, num_computed_tokens=0)

    target = scheduler._estimate_reflex_admission_demote_target(request)

    assert target == 0


def test_reflex_int4_waiting_demotion_only_step_uses_fast_deficit():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.running = [object(), object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(num_tokens=32768, num_computed_tokens=0)
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    demotion = ReflexDemotion(
        request_id="req-0",
        page_idx=0,
        bf16_block_id=0,
        int4_block_id=0,
        encoded_block_table_id=-1,
        kv_cache_group_id=0,
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [demotion]
    scheduler._update_after_schedule = lambda scheduler_output: None
    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return target_bf16_blocks

    scheduler._try_reflex_int4_demote = fake_try_demote

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert requested_targets == [(1888, True, "admission_waiting")]


def test_reflex_int4_waiting_demotion_only_step_batches_small_deficit():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=543)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(num_tokens=8192, num_computed_tokens=0)
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    demotion = ReflexDemotion(
        request_id="req-0",
        page_idx=0,
        bf16_block_id=0,
        int4_block_id=0,
        encoded_block_table_id=-1,
        kv_cache_group_id=0,
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [demotion]
    scheduler._update_after_schedule = lambda scheduler_output: None
    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return target_bf16_blocks

    scheduler._try_reflex_int4_demote = fake_try_demote

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert requested_targets == [(32, True, "admission_waiting")]


def test_reflex_int4_waiting_demotion_only_step_skips_planner_when_request_fits(
    monkeypatch,
):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    request = SimpleNamespace(
        request_id="waiting-fits",
        num_tokens=8192,
        num_computed_tokens=0,
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )

    def fail_plan(**kwargs):
        raise AssertionError("zero-deficit admission should not plan demotion")

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fail_plan
    scheduler._try_reflex_int4_demote = (
        lambda *, target_bf16_blocks, force=False, reason="pressure": (
            (_ for _ in ()).throw(
                AssertionError("zero-deficit admission should not demote")
            )
        )
    )
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    assert scheduler._try_reflex_int4_demotion_only_step() is None
    assert not [
        message
        for message in log_messages
        if "ReFlexKV trace admission_control" in message
    ]


def test_reflex_int4_waiting_demotion_only_step_logs_closed_loop_metrics(
    monkeypatch,
):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=500)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(
        request_id="waiting-0",
        num_tokens=8192,
        num_computed_tokens=0,
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: []
    scheduler._update_after_schedule = lambda scheduler_output: None
    scheduler._reflex_int4_last_demote_candidate_capacity = 20
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = (
        lambda **kwargs: kwargs["target_bf16_blocks"]
    )

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        assert target_bf16_blocks == 44
        assert force is True
        assert reason == "admission_waiting"
        return 20

    scheduler._try_reflex_int4_demote = fake_try_demote
    log_messages = []

    def fake_log(message, *args):
        log_messages.append(message % args)

    monkeypatch.setattr(scheduler_module.logger, "info", fake_log)

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    closed_loop_logs = [
        message
        for message in log_messages
        if "ReFlexKV trace admission_control" in message
    ]
    assert closed_loop_logs
    assert "requested_release=44" in closed_loop_logs[0]
    assert "candidate_release_capacity=44" in closed_loop_logs[0]
    assert "feasible_release=44" in closed_loop_logs[0]
    assert "planned_release=44" in closed_loop_logs[0]
    assert "actual_release=20" in closed_loop_logs[0]
    assert "admission_success_after_demote=False" in closed_loop_logs[0]
    assert "admission_infeasible=False" in closed_loop_logs[0]
    assert "admission_wait_reduction=20" in closed_loop_logs[0]


def test_reflex_int4_waiting_demotion_only_step_logs_blocked_admission(
    monkeypatch,
):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=500)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(
        request_id="waiting-0",
        num_tokens=8192,
        num_computed_tokens=0,
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    scheduler._reflex_int4_last_demote_candidate_capacity = 0
    scheduler._try_reflex_int4_demote = (
        lambda *, target_bf16_blocks, force=False, reason="pressure": 0
    )
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    assert scheduler._try_reflex_int4_demotion_only_step() is None
    closed_loop_logs = [
        message
        for message in log_messages
        if "ReFlexKV trace admission_control" in message
    ]
    assert closed_loop_logs
    assert "actual_release=0" in closed_loop_logs[0]
    assert "admission_blocked=True" in closed_loop_logs[0]


def test_reflex_int4_waiting_demotion_only_step_executes_partial_frontier(
    monkeypatch,
):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=500)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(
        request_id="waiting-0",
        num_tokens=8192,
        num_computed_tokens=0,
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    dry_run_calls = []

    def fake_plan(**kwargs):
        dry_run_calls.append(kwargs)
        assert kwargs["dry_run"] is True
        assert kwargs["target_bf16_blocks"] == 44
        return 20

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return 20

    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [
        ReflexDemotion(
            request_id="req-0",
            page_idx=0,
            bf16_block_id=0,
            int4_block_id=0,
            encoded_block_table_id=-1,
            kv_cache_group_id=0,
        )
    ]
    scheduler._update_after_schedule = lambda scheduler_output: None
    scheduler._try_reflex_int4_demote = fake_try_demote
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert len(dry_run_calls) == 1
    assert requested_targets == [(20, True, "admission_waiting")]
    closed_loop_logs = [
        message
        for message in log_messages
        if "ReFlexKV trace admission_control" in message
    ]
    assert closed_loop_logs
    assert "requested_release=44" in closed_loop_logs[0]
    assert "feasible_release=20" in closed_loop_logs[0]
    assert "planned_release=20" in closed_loop_logs[0]
    assert "actual_release=20" in closed_loop_logs[0]
    assert "admission_infeasible=True" in closed_loop_logs[0]
    assert "admission_wait_reduction=20" in closed_loop_logs[0]


def test_reflex_int4_waiting_step_uses_controller_precision_plan():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=500)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(
        request_id="waiting-controller-plan",
        num_tokens=8192,
        num_computed_tokens=0,
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    admission = PrecisionAdmissionDecision(
        request_id=request.request_id,
        required_blocks=544,
        requested_release=44,
        feasible_release=44,
        planned_release=44,
        free_after_planned=544,
        admission_success_after_planned=True,
        admission_infeasible=False,
    )
    landing = PrecisionLandingDecision(
        request_id=request.request_id,
        required_blocks=544,
        bf16_deficit_blocks=44,
        residual_deficit_after_running=7,
        eligible_int4_landing_blocks=37,
        planned_int4_landing_blocks=37,
        planned_int4_landing_pages=tuple(range(37)),
        admission_feasible_with_landing=True,
        mixed_landing_required=True,
        reason="mixed_landing_feasible",
    )
    controller_calls = []

    class FakeController:

        def plan_admission(self, state):
            return admission

        def plan_precision_admission(
            self,
            *,
            admission_decision,
            landing_decision,
        ):
            controller_calls.append((admission_decision, landing_decision))
            return SimpleNamespace(
                planned_release=7,
                admission_infeasible=False,
            )

    scheduler._precision_kv_admission_controller = FakeController()
    scheduler._estimate_reflex_int4_feasible_release = (
        lambda *, target_bf16_blocks, reason: 44
    )
    scheduler._plan_reflex_int4_landing_frontier = lambda **kwargs: landing
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [
        ReflexDemotion(
            request_id="req-0",
            page_idx=0,
            bf16_block_id=0,
            int4_block_id=0,
            encoded_block_table_id=-1,
            kv_cache_group_id=0,
        )
    ]
    scheduler._update_after_schedule = lambda scheduler_output: None
    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return target_bf16_blocks

    scheduler._try_reflex_int4_demote = fake_try_demote

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert controller_calls == [(admission, None)]
    assert requested_targets == [(7, True, "admission_waiting")]


def test_reflex_int4_landing_frontier_uses_waiting_request_risk_metadata():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_low_risk_score_fraction = 0.5
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="waiting-risk",
        num_tokens=64 * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 16,
        },
    )

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=1024,
        reserve_blocks=32,
        free_blocks=900,
        running_feasible_release=20,
    )

    assert decision.required_blocks == 1056
    assert decision.residual_deficit_after_running == 136
    assert decision.eligible_int4_landing_blocks == 32
    assert decision.admission_feasible_with_landing is False
    assert decision.reason == "int4_landing_frontier_insufficient"


def test_reflex_int4_landing_frontier_marks_mixed_landing_feasible():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.75
    scheduler._reflex_int4_low_risk_score_fraction = 0.75
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="waiting-risk",
        num_tokens=256 * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 64,
        },
    )

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=1024,
        reserve_blocks=32,
        free_blocks=900,
        running_feasible_release=20,
    )

    assert decision.residual_deficit_after_running == 136
    assert decision.eligible_int4_landing_blocks == 192
    assert decision.planned_int4_landing_blocks == 136
    assert decision.planned_int4_landing_pages == tuple(
        sorted(
            range(256),
            key=lambda idx: ([0.9, 0.8, 0.1, 0.2] * 64)[idx],
        )[:136]
    )
    assert decision.admission_feasible_with_landing is True
    assert decision.mixed_landing_required is True
    assert decision.reason == "mixed_landing_feasible"


def test_reflex_int4_direct_remote_chunk_landing_synthesizes_current_chunk_pages():
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=0,
        block_size=16,
        max_model_len=8192,
    )
    scheduler._reflex_int4_direct_landing_enabled = True
    scheduler._reflex_int4_max_int4_fraction_per_request = 1.0
    scheduler._reflex_int4_admission_landing_max_int4_fraction = 1.0
    scheduler._reflex_int4_keep_initial_pages = 4
    scheduler._reflex_int4_keep_recent_pages = 0
    request = _make_waiting_request(
        "remote-direct-fallback",
        num_prompt_tokens=8192,
    )
    request.num_computed_tokens = 1024
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
    }

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=32,
        reserve_blocks=32,
        free_blocks=0,
        running_feasible_release=0,
    )

    assert decision.required_blocks == 64
    assert decision.needed_deficit_after_running == 32
    assert decision.eligible_int4_landing_blocks == 32
    assert decision.planned_int4_landing_blocks == 32
    assert decision.planned_int4_landing_pages == tuple(range(64, 96))
    assert decision.reserve_relaxed is True
    assert decision.admission_feasible_with_landing is True
    assert decision.reason == "mixed_landing_relaxed_reserve_feasible"


def test_reflex_int4_allocation_failure_target_keeps_admission_slack():
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=63,
        block_size=16,
        max_model_len=8192,
    )
    scheduler._reflex_int4_admission_reserve_blocks = 32
    request = _make_waiting_request(
        "remote-chunk",
        num_prompt_tokens=8192,
    )
    request.num_computed_tokens = 1024
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
    }
    scheduler.kv_cache_manager.get_blocks = (
        lambda request_id: _FakeBlocks(list(range(64)))
    )

    target = scheduler._reflex_int4_allocation_failure_demote_target(
        request,
        num_new_tokens=512,
        num_lookahead_tokens=0,
    )

    assert target == 32


def test_reflex_int4_direct_remote_chunk_fallback_protects_short_prompts():
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=0,
        block_size=16,
        max_model_len=8192,
    )
    scheduler._reflex_int4_direct_landing_enabled = True
    scheduler._reflex_int4_max_int4_fraction_per_request = 1.0
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = _make_waiting_request(
        "math-like-short",
        num_prompt_tokens=16 * scheduler.block_size,
    )
    request.kv_transfer_params = {
        "do_remote_prefill": True,
        "reflex_remote_chunk_enabled": True,
        "reflex_remote_chunk_tokens": 512,
    }

    pages = scheduler._reflex_int4_landing_eligible_pages(request)

    assert pages == ()


def test_reflex_int4_landing_frontier_uses_admission_fraction_when_global_cap_insufficient():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=325)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_admission_landing_max_int4_fraction = 0.75
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="waiting-risk",
        num_tokens=1024 * scheduler.block_size,
        num_computed_tokens=0,
        priority=0,
        kv_transfer_params={
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 256,
        },
    )

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=909,
        reserve_blocks=32,
        free_blocks=325,
        running_feasible_release=0,
    )

    assert decision.residual_deficit_after_running == 616
    assert decision.eligible_int4_landing_blocks == 768
    assert decision.planned_int4_landing_blocks == 616
    assert decision.admission_feasible_with_landing is True
    assert decision.reason == "mixed_landing_feasible"


def test_reflex_int4_partial_remote_allocation_lands_only_missing_blocks():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_admission_landing_max_int4_fraction = 0.75
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    allocated_blocks = [object()] * 797
    scheduler.kv_cache_manager.get_blocks = lambda request_id: allocated_blocks
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = (
        lambda **kwargs: 0
    )
    scheduler.kv_cache_manager.reserve_reflex_int4_landing_blocks = (
        lambda request_id, count: list(range(9000, 9000 + count))
    )
    request = SimpleNamespace(
        request_id="partial-remote",
        num_tokens=1025 * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.1] * 1025,
        },
    )

    assert scheduler._estimate_reflex_admission_needed_blocks(request) == 228

    decision = scheduler._plan_and_persist_reflex_int4_landing_contract(
        request=request,
        reason="full_sequence_reserve",
        reserve_blocks=0,
    )

    assert decision is not None
    assert decision.residual_deficit_after_running == 228
    assert decision.planned_int4_landing_blocks == 228
    assert decision.admission_feasible_with_landing is True
    assert request.kv_transfer_params["reflex_int4_landing_planned_blocks"] == 228


def test_reflex_int4_landing_frontier_extends_beyond_prefill_hints():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=4)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_low_risk_score_fraction = 0.25
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    page_risks = [
        0.91,
        0.81,
        0.71,
        0.61,
        0.11,
        0.21,
        0.31,
        0.41,
        0.51,
        0.12,
        0.22,
        0.32,
        0.42,
        0.52,
        0.62,
        0.72,
    ]
    request = SimpleNamespace(
        request_id="waiting-risk",
        num_tokens=len(page_risks) * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "reflex_page_risks": page_risks,
            "reflex_compressible_pages": [4, 9],
        },
    )

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=10,
        reserve_blocks=0,
        free_blocks=4,
        running_feasible_release=0,
    )

    assert decision.residual_deficit_after_running == 6
    assert decision.eligible_int4_landing_blocks == 8
    assert decision.planned_int4_landing_blocks == 6
    assert decision.planned_int4_landing_pages[:2] == (4, 9)
    assert decision.reason == "mixed_landing_feasible"


def test_reflex_int4_waiting_step_does_not_persist_staged_landing_contract():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.75
    scheduler._reflex_int4_low_risk_score_fraction = 0.75
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(
        request_id="waiting-contract",
        num_tokens=1024 * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 256,
        },
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = (
        lambda **kwargs: 20
    )
    scheduler.kv_cache_manager.reserve_reflex_int4_landing_blocks = (
        lambda request_id, count: list(range(9000, 9000 + count))
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: []
    scheduler._try_reflex_int4_demote = (
        lambda *, target_bf16_blocks, force=False, reason="pressure": 0
    )
    scheduler._update_after_schedule = lambda scheduler_output: None

    assert scheduler._try_reflex_int4_demotion_only_step() is None

    assert "reflex_int4_landing_required_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_planned_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_reason" not in request.kv_transfer_params
    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params


def test_reflex_int4_waiting_step_combines_small_demotion_with_mixed_landing():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=700)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(
        request_id="waiting-combined",
        num_tokens=1105 * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 300,
        },
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )

    def fake_plan(**kwargs):
        assert kwargs["dry_run"] is True
        assert kwargs["target_bf16_blocks"] == 437
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan
    scheduler.kv_cache_manager.reserve_reflex_int4_landing_blocks = (
        lambda request_id, count: list(range(9000, 9000 + count))
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [
        ReflexDemotion(
            request_id="req-0",
            page_idx=0,
            bf16_block_id=0,
            int4_block_id=0,
            encoded_block_table_id=-1,
            kv_cache_group_id=0,
        )
    ]
    scheduler._update_after_schedule = lambda scheduler_output: None
    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return target_bf16_blocks

    scheduler._try_reflex_int4_demote = fake_try_demote

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert requested_targets == [(8, True, "admission_waiting")]
    assert "reflex_int4_landing_required_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_planned_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params


def test_reflex_int4_background_demotion_only_step_runs_without_waiting():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=8)
    scheduler.running = [SimpleNamespace(is_prefill_chunk=False)]
    scheduler.waiting = False
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [
        ReflexDemotion(
            request_id="req-0",
            page_idx=0,
            bf16_block_id=0,
            int4_block_id=0,
            encoded_block_table_id=-1,
            kv_cache_group_id=0,
        )
    ]
    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return target_bf16_blocks

    scheduler._try_reflex_int4_demote = fake_try_demote
    scheduler._update_after_schedule = lambda scheduler_output: None

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert requested_targets == [(16, False, "background_pressure")]
    assert scheduler_output.reflex_int4_demotions is not None


def test_reflex_int4_background_target_uses_min_batch_under_pressure():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=31)
    scheduler._reflex_int4_background_free_floor_blocks = 32
    scheduler._reflex_int4_background_min_demotions_per_step = 8
    scheduler._reflex_int4_background_demotions_per_step = 16

    target = scheduler._estimate_reflex_demote_target(
        1,
        force_allocate_failure=False,
    )

    assert target == 8


def test_reflex_int4_background_noop_demote_enters_cooldown():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=31)
    scheduler._reflex_int4_scheduler_step = 42
    scheduler._reflex_int4_last_demote_step = 0
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = lambda **kwargs: 0

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=8,
        force=False,
        reason="background_pressure",
    )

    assert released == 0
    assert scheduler._reflex_int4_last_demote_step == 42


def test_reflex_int4_full_sequence_reserve_persists_landing_contract():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.75
    scheduler._reflex_int4_low_risk_score_fraction = 0.75
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="full-sequence-contract",
        num_tokens=1024 * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 256,
        },
    )
    calls = []

    def fake_plan(**kwargs):
        calls.append(kwargs)
        assert kwargs["dry_run"] is True
        return 20

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan
    scheduler.kv_cache_manager.reserve_reflex_int4_landing_blocks = (
        lambda request_id, count: list(range(8000, 8000 + count))
    )

    decision = scheduler._plan_and_persist_reflex_int4_landing_contract(
        request=request,
        reason="full_sequence_reserve",
    )

    assert decision is not None
    assert calls
    assert request.kv_transfer_params["reflex_int4_landing_required_blocks"] == 136
    assert request.kv_transfer_params["reflex_int4_landing_planned_blocks"] == 136
    assert request.kv_transfer_params["reflex_int4_landing_reason"] == (
        "mixed_landing_feasible"
    )
    assert request.kv_transfer_params["reflex_int4_landing_block_ids"] == list(
        range(8000, 8136)
    )


def test_reflex_int4_full_sequence_reserve_skips_landing_without_remote_prefill():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.75
    scheduler._reflex_int4_low_risk_score_fraction = 0.75
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="local-prefill-contract",
        num_tokens=1024 * scheduler.block_size,
        num_computed_tokens=0,
        kv_transfer_params={
            "do_remote_prefill": False,
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 256,
        },
    )
    scheduler.kv_cache_manager.reserve_reflex_int4_landing_blocks = (
        lambda request_id, count: list(range(8000, 8000 + count))
    )
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = (
        lambda **kwargs: 0
    )

    decision = scheduler._plan_and_persist_reflex_int4_landing_contract(
        request=request,
        reason="full_sequence_reserve",
    )

    assert decision is not None
    assert decision.mixed_landing_required is True
    assert "reflex_int4_landing_required_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_planned_blocks" not in request.kv_transfer_params
    assert "reflex_int4_landing_reason" not in request.kv_transfer_params
    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params


def test_reflex_int4_landing_contract_clears_later_bf16_fit_before_remote_wait():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=2000)
    request = SimpleNamespace(
        request_id="sticky-landing-contract",
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_landing_required_blocks": 2,
            "reflex_int4_landing_planned_blocks": 2,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )
    decision = PrecisionLandingDecision(
        request_id="sticky-landing-contract",
        required_blocks=64,
        bf16_deficit_blocks=0,
        residual_deficit_after_running=0,
        eligible_int4_landing_blocks=2,
        planned_int4_landing_blocks=0,
        admission_feasible_with_landing=True,
        mixed_landing_required=False,
        reason="bf16_fit",
    )

    scheduler._persist_reflex_int4_landing_contract(request, decision)

    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params
    assert "reflex_int4_landing_reason" not in request.kv_transfer_params


def test_reflex_int4_landing_contract_cleared_when_bf16_fits_before_remote_wait():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=2000)
    request = SimpleNamespace(
        request_id="pre-transfer-stale-landing-contract",
        status=scheduler_module.RequestStatus.WAITING,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_landing_required_blocks": 2,
            "reflex_int4_landing_planned_blocks": 2,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )
    decision = PrecisionLandingDecision(
        request_id="pre-transfer-stale-landing-contract",
        required_blocks=64,
        bf16_deficit_blocks=0,
        residual_deficit_after_running=0,
        eligible_int4_landing_blocks=2,
        planned_int4_landing_blocks=0,
        admission_feasible_with_landing=True,
        mixed_landing_required=False,
        reason="bf16_fit",
    )

    scheduler._persist_reflex_int4_landing_contract(request, decision)

    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params
    assert "reflex_int4_landing_reason" not in request.kv_transfer_params


def test_reflex_int4_global_evidence_request_caps_landing_fraction():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=720)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_admission_landing_max_int4_fraction = 0.75
    scheduler._reflex_int4_global_evidence_landing_max_int4_fraction = 0.05
    scheduler._reflex_int4_global_evidence_min_prompt_pages = 512
    scheduler._reflex_int4_global_evidence_min_decode_tokens = 129
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="long-summary",
        num_tokens=1024 * scheduler.block_size,
        num_prompt_tokens=1024 * scheduler.block_size,
        max_tokens=512,
        output_token_ids=[],
        priority=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.1] * 1024,
        },
    )

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=800,
        reserve_blocks=0,
        free_blocks=720,
        running_feasible_release=0,
    )

    assert decision.reason == "int4_landing_frontier_insufficient"
    assert decision.eligible_int4_landing_blocks == 51


def test_reflex_int4_global_evidence_hard_capacity_gap_uses_emergency_landing():
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=720,
        total_blocks=736,
    )
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_admission_landing_max_int4_fraction = 0.75
    scheduler._reflex_int4_global_evidence_landing_max_int4_fraction = 0.05
    scheduler._reflex_int4_global_evidence_min_prompt_pages = 512
    scheduler._reflex_int4_global_evidence_min_decode_tokens = 129
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="long-summary-hard-cap",
        num_tokens=1024 * scheduler.block_size,
        num_prompt_tokens=1024 * scheduler.block_size,
        max_tokens=512,
        output_token_ids=[],
        priority=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.1] * 1024,
        },
    )

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=1024,
        reserve_blocks=32,
        free_blocks=720,
        running_feasible_release=0,
    )

    assert decision.admission_feasible_with_landing is True
    assert decision.planned_int4_landing_blocks == 336
    assert decision.eligible_int4_landing_blocks == 768
    assert decision.reason == "emergency_mixed_landing_feasible"


def test_reflex_int4_reasoning_like_short_prompt_disables_int4_landing():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=4)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_low_risk_score_fraction = 1.0
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="math-like",
        num_tokens=8 * scheduler.block_size,
        num_prompt_tokens=8 * scheduler.block_size,
        max_tokens=4096,
        output_token_ids=[],
        priority=1,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.1] * 8,
        },
    )

    decision = scheduler._plan_reflex_int4_landing_frontier(
        request=request,
        needed_blocks=10,
        reserve_blocks=0,
        free_blocks=4,
        running_feasible_release=0,
    )

    assert decision.reason == "int4_landing_frontier_insufficient"
    assert decision.eligible_int4_landing_blocks == 0
    assert decision.planned_int4_landing_pages == ()


def test_reflex_int4_landing_contract_clears_later_insufficient_frontier_before_remote_wait():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=2000)
    request = SimpleNamespace(
        request_id="sticky-landing-contract",
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_landing_required_blocks": 2,
            "reflex_int4_landing_planned_blocks": 2,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )
    decision = PrecisionLandingDecision(
        request_id="sticky-landing-contract",
        required_blocks=64,
        bf16_deficit_blocks=40,
        residual_deficit_after_running=40,
        eligible_int4_landing_blocks=2,
        planned_int4_landing_blocks=0,
        admission_feasible_with_landing=False,
        mixed_landing_required=True,
        reason="int4_landing_frontier_insufficient",
    )

    scheduler._persist_reflex_int4_landing_contract(request, decision)

    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params
    assert "reflex_int4_landing_reason" not in request.kv_transfer_params


def test_reflex_int4_landing_contract_survives_remote_wait_bf16_fit():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=2000)
    request = SimpleNamespace(
        request_id="sticky-remote-landing-contract",
        status=scheduler_module.RequestStatus.WAITING_FOR_REMOTE_KVS,
        kv_transfer_params={
            "do_remote_prefill": False,
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_landing_required_blocks": 2,
            "reflex_int4_landing_planned_blocks": 2,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )
    decision = PrecisionLandingDecision(
        request_id="sticky-remote-landing-contract",
        required_blocks=64,
        bf16_deficit_blocks=0,
        residual_deficit_after_running=0,
        eligible_int4_landing_blocks=2,
        planned_int4_landing_blocks=0,
        admission_feasible_with_landing=True,
        mixed_landing_required=False,
        reason="bf16_fit",
    )

    scheduler._persist_reflex_int4_landing_contract(request, decision)

    assert request.kv_transfer_params["reflex_int4_landing_pages"] == [1, 3]
    assert request.kv_transfer_params["reflex_int4_landing_block_ids"] == [9, 10]
    assert request.kv_transfer_params["reflex_int4_landing_reason"] == (
        "mixed_landing_feasible"
    )


def test_reflex_int4_landing_contract_survives_remote_wait_replan():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=2000)
    request = SimpleNamespace(
        request_id="sticky-remote-landing-contract",
        status=scheduler_module.RequestStatus.WAITING_FOR_REMOTE_KVS,
        kv_transfer_params={
            "do_remote_prefill": False,
            "reflex_int4_landing_pages": [1, 3, 5],
            "reflex_int4_landing_block_ids": [9, 10, 11],
            "reflex_int4_landing_required_blocks": 3,
            "reflex_int4_landing_planned_blocks": 3,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )
    decision = PrecisionLandingDecision(
        request_id="sticky-remote-landing-contract",
        required_blocks=64,
        bf16_deficit_blocks=0,
        residual_deficit_after_running=2,
        eligible_int4_landing_blocks=2,
        planned_int4_landing_blocks=2,
        planned_int4_landing_pages=(7, 9),
        admission_feasible_with_landing=True,
        mixed_landing_required=True,
        reason="mixed_landing_feasible",
    )

    scheduler._persist_reflex_int4_landing_contract(request, decision)

    assert request.kv_transfer_params["reflex_int4_landing_pages"] == [1, 3, 5]
    assert request.kv_transfer_params["reflex_int4_landing_block_ids"] == [
        9,
        10,
        11,
    ]
    assert request.kv_transfer_params["reflex_int4_landing_planned_blocks"] == 3


def test_reflex_int4_plan_preserves_remote_wait_landing_contract_after_connector_flip():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=4)
    scheduler._reflex_int4_max_int4_fraction_per_request = 1.0
    scheduler._reflex_int4_low_risk_score_fraction = 1.0
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = lambda **kwargs: 0
    scheduler.kv_cache_manager.get_last_reflex_int4_candidate_capacity = (
        lambda: 0
    )
    scheduler.kv_cache_manager.get_last_reflex_int4_candidate_breakdown = (
        lambda: ReflexCandidateBreakdown()
    )
    request = SimpleNamespace(
        request_id="remote-wait-existing-contract",
        status=scheduler_module.RequestStatus.WAITING_FOR_REMOTE_KVS,
        num_tokens=16 * scheduler.block_size,
        num_prompt_tokens=16 * scheduler.block_size,
        max_tokens=64,
        output_token_ids=[],
        priority=0,
        kv_transfer_params={
            "do_remote_prefill": False,
            "reflex_page_risks": [0.1] * 16,
            "reflex_int4_landing_pages": [2, 3, 4],
            "reflex_int4_landing_block_ids": [90, 91, 92],
            "reflex_int4_landing_required_blocks": 3,
            "reflex_int4_landing_planned_blocks": 3,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )

    decision = scheduler._plan_and_persist_reflex_int4_landing_contract(
        request=request,
        reason="full_sequence_reserve",
        reserve_blocks=0,
    )

    assert decision is not None
    assert decision.mixed_landing_required is True
    assert request.kv_transfer_params["reflex_int4_landing_pages"] == [2, 3, 4]
    assert request.kv_transfer_params["reflex_int4_landing_block_ids"] == [
        90,
        91,
        92,
    ]
    assert (
        request.kv_transfer_params["reflex_int4_landing_reason"]
        == "mixed_landing_feasible"
    )


def test_reflex_int4_remote_kv_completion_commits_landing_contract():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler.connector = object()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"remote-req"}
    scheduler.reflex_int4_materialized_landing_req_ids = {"remote-req"}
    cached_tokens = []
    commits = []

    def fake_cache_blocks(request, num_tokens):
        cached_tokens.append((request.request_id, num_tokens))

    def fake_commit(request_id, page_indices, int4_block_ids):
        commits.append((request_id, list(page_indices), list(int4_block_ids)))
        return len(page_indices)

    scheduler.kv_cache_manager.cache_blocks = fake_cache_blocks
    scheduler.kv_cache_manager.commit_reflex_int4_landing_pages = fake_commit
    request = SimpleNamespace(
        request_id="remote-req",
        num_computed_tokens=64,
        num_tokens=128,
        num_cached_tokens=-1,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_page_risks": [0.9, 0.1],
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_landing_required_blocks": 2,
            "reflex_int4_landing_planned_blocks": 2,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )

    scheduler._update_waiting_for_remote_kv(request)

    assert cached_tokens == [("remote-req", 64)]
    assert commits == [("remote-req", [1, 3], [9, 10])]
    assert request.num_cached_tokens == 64
    assert "reflex_page_risks" in request.kv_transfer_params
    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert "reflex_int4_landing_block_ids" not in request.kv_transfer_params
    assert "remote-req" not in scheduler.finished_recving_kv_req_ids


def test_reflex_int4_remote_kv_completion_keeps_bf16_when_landing_not_materialized(
    monkeypatch,
):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler.connector = object()
    scheduler.failed_recving_kv_req_ids = set()
    scheduler.finished_recving_kv_req_ids = {"remote-req"}
    scheduler.reflex_int4_materialized_landing_req_ids = set()
    commits = []
    releases = []
    log_messages = []

    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    scheduler.kv_cache_manager.cache_blocks = lambda request, num_tokens: None
    scheduler.kv_cache_manager.commit_reflex_int4_landing_pages = (
        lambda request_id, page_indices, int4_block_ids: commits.append(request_id)
    )
    scheduler.kv_cache_manager.release_reflex_int4_landing_blocks = (
        lambda request_id: releases.append(request_id)
    )
    request = SimpleNamespace(
        request_id="remote-req",
        num_computed_tokens=64,
        num_tokens=128,
        num_cached_tokens=-1,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
            "reflex_int4_landing_required_blocks": 2,
            "reflex_int4_landing_planned_blocks": 2,
            "reflex_int4_landing_reason": "mixed_landing_feasible",
        },
    )

    scheduler._update_waiting_for_remote_kv(request)

    assert commits == []
    assert releases == ["remote-req"]
    assert "reflex_int4_landing_pages" not in request.kv_transfer_params
    assert request.num_cached_tokens == 64
    assert any(
        "ReFlexKV trace landing_policy" in message
        and "outcome=fallback_unmaterialized" in message
        and "planned_pages=2" in message
        for message in log_messages
    )


def test_reflex_int4_demotion_planning_protects_remote_landing_requests():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_page_level_protection_enabled = False
    landing_req = SimpleNamespace(
        request_id="landing-req",
        status=scheduler_module.RequestStatus.WAITING_FOR_REMOTE_KVS,
        is_prefill_chunk=False,
        num_computed_tokens=1024,
        num_prompt_tokens=1024,
        num_tokens=1024,
        kv_transfer_params={
            "reflex_int4_landing_pages": [1, 3],
            "reflex_int4_landing_block_ids": [9, 10],
        },
    )
    normal_req = SimpleNamespace(
        request_id="normal-req",
        status=scheduler_module.RequestStatus.RUNNING,
        is_prefill_chunk=False,
        num_computed_tokens=1024,
        num_prompt_tokens=1024,
        num_tokens=1024,
        kv_transfer_params={},
    )
    scheduler.requests = {
        "landing-req": landing_req,
        "normal-req": normal_req,
    }

    kwargs = scheduler._build_reflex_int4_demotion_planning_kwargs(
        target_bf16_blocks=10,
        reason="admission_waiting",
    )

    assert kwargs["protected_request_ids"] == {"landing-req"}
    assert "landing-req" not in kwargs["request_precision_budgets"]
    assert "normal-req" in kwargs["request_precision_budgets"]


def test_reflex_int4_landing_contract_uses_page_level_protection_after_commit():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    request = SimpleNamespace(
        request_id="landing-partial",
        status=scheduler_module.RequestStatus.WAITING_FOR_REMOTE_KVS,
        is_prefill_chunk=False,
        num_computed_tokens=96 * scheduler.block_size,
        num_prompt_tokens=256 * scheduler.block_size,
        num_tokens=256 * scheduler.block_size,
        max_tokens=128,
        output_token_ids=[],
        priority=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_page_start": 96,
            "reflex_remote_chunk_page_end": 128,
            "reflex_remote_chunk_committed_page_end": 96,
            "reflex_remote_chunk_inflight": True,
            "reflex_int4_landing_pages": [100, 112],
            "reflex_int4_landing_block_ids": [900, 901],
        },
    )
    scheduler.requests = {"landing-partial": request}

    kwargs = scheduler._build_reflex_int4_demotion_planning_kwargs(
        target_bf16_blocks=16,
        reason="full_sequence_reserve",
    )

    assert "landing-partial" not in kwargs["protected_request_ids"]
    assert kwargs["protected_pages_by_request"]["landing-partial"] >= {
        100,
        112,
    }
    assert kwargs["sealed_pages_by_request"] == {"landing-partial": 96}
    assert "landing-partial" in kwargs["request_precision_budgets"]


def test_reflex_int4_remote_waiting_uses_page_level_protection_after_commit():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    request = SimpleNamespace(
        request_id="remote-partial",
        status=scheduler_module.RequestStatus.WAITING_FOR_REMOTE_KVS,
        is_prefill_chunk=False,
        num_computed_tokens=64 * scheduler.block_size,
        num_prompt_tokens=256 * scheduler.block_size,
        num_tokens=256 * scheduler.block_size,
        max_tokens=128,
        output_token_ids=[],
        priority=0,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_page_start": 64,
            "reflex_remote_chunk_page_end": 96,
            "reflex_remote_chunk_committed_page_end": 64,
            "reflex_remote_chunk_inflight": True,
            "reflex_remote_chunk_tokens": 512,
        },
    )
    scheduler.requests = {"remote-partial": request}

    kwargs = scheduler._build_reflex_int4_demotion_planning_kwargs(
        target_bf16_blocks=16,
        reason="admission_waiting",
    )

    assert "remote-partial" not in kwargs["protected_request_ids"]
    assert (
        "remote-partial"
        in kwargs["allow_partial_prefill_demotion_request_ids"]
    )
    assert kwargs["sealed_pages_by_request"] == {"remote-partial": 64}
    assert kwargs["remote_inflight_pages_by_request"] == {
        "remote-partial": set(range(64, 96)),
    }
    assert "remote-partial" in kwargs["request_precision_budgets"]


def test_reflex_int4_long_prompt_contract_protects_head_tail_and_high_risk():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_long_prompt_protected_head_pages = 4
    scheduler._reflex_int4_long_prompt_protected_tail_pages = 4
    scheduler._reflex_int4_prompt_high_risk_protection_threshold = 0.80
    page_risks = [0.1] * 128
    page_risks[70] = 0.95
    request = SimpleNamespace(
        request_id="long-prompt",
        status=scheduler_module.RequestStatus.RUNNING,
        is_prefill_chunk=False,
        num_computed_tokens=128 * scheduler.block_size,
        num_prompt_tokens=128 * scheduler.block_size,
        num_tokens=128 * scheduler.block_size,
        max_tokens=128,
        output_token_ids=[0] * 32,
        priority=0,
        kv_transfer_params={
            "reflex_page_risks": page_risks,
            "reflex_compressible_pages": list(range(128)),
        },
    )
    scheduler.requests = {"long-prompt": request}

    kwargs = scheduler._build_reflex_int4_demotion_planning_kwargs(
        target_bf16_blocks=16,
        reason="admission_waiting",
    )

    assert kwargs["protected_prompt_pages_by_request"] == {"long-prompt": 4}
    assert kwargs["protected_pages_by_request"]["long-prompt"] == (
        set(range(4)) | set(range(124, 128)) | {70}
    )


def test_reflex_int4_landing_eligible_pages_excludes_existing_int4_pages():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="landing-after-demote",
        num_tokens=8 * scheduler.block_size,
        num_prompt_tokens=8 * scheduler.block_size,
        kv_transfer_params={
            "reflex_compressible_pages": [1, 2, 3, 4],
        },
    )
    scheduler.kv_cache_manager.get_reflex_page_runtime_descriptors = (
        lambda request_id, **kwargs: [
            KVPageRuntimeDescriptor(
                request_id=request_id,
                page_idx=2,
                precision=PrecisionState.INT4,
                tier=MemoryTier.GPU,
                lifecycle=KVPageLifecycle.INT4_ACTIVE,
                physical_block_id=87,
                int4_block_id=87,
            )
        ]
    )

    pages = scheduler._reflex_int4_landing_eligible_pages(
        request,
        max_int4_fraction=1.0,
        respect_global_evidence_cap=False,
    )

    assert pages == (1, 3, 4)


def test_reflex_int4_landing_eligible_pages_stays_inside_remote_chunk_range():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    scheduler._reflex_int4_keep_initial_pages = 0
    scheduler._reflex_int4_keep_recent_pages = 0
    request = SimpleNamespace(
        request_id="risk-chunk",
        num_tokens=1024 * scheduler.block_size,
        num_prompt_tokens=1024 * scheduler.block_size,
        kv_transfer_params={
            "do_remote_prefill": True,
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_page_start": 768,
            "reflex_remote_chunk_page_end": 800,
            "reflex_remote_chunk_token_end": 800 * scheduler.block_size,
            "reflex_page_risks": [0.1] * 1024,
            "reflex_compressible_pages": [56, 790, 800],
        },
    )

    pages = scheduler._reflex_int4_landing_eligible_pages(
        request,
        max_int4_fraction=1.0,
        respect_global_evidence_cap=False,
    )

    assert 790 in pages
    assert 56 not in pages
    assert 800 not in pages
    assert all(768 <= page_idx < 800 for page_idx in pages)


def test_reflex_int4_existing_int4_pages_reads_coordinator_managers():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    descriptor = KVPageRuntimeDescriptor(
        request_id="coordinator-backed",
        page_idx=56,
        precision=PrecisionState.INT4,
        tier=MemoryTier.GPU,
        lifecycle=KVPageLifecycle.INT4_ACTIVE,
        physical_block_id=86,
        int4_block_id=86,
    )
    scheduler.kv_cache_manager = SimpleNamespace(
        coordinator=SimpleNamespace(
            single_type_managers=[
                SimpleNamespace(
                    get_reflex_page_runtime_descriptors=(
                        lambda request_id, **kwargs: [descriptor]
                    )
                )
            ]
        )
    )

    assert scheduler._reflex_int4_existing_int4_page_indices(
        "coordinator-backed"
    ) == {56}


def test_reflex_int4_request_budget_trace_includes_slo_and_decode_state(
    monkeypatch,
):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )
    request = SimpleNamespace(
        request_id="budget-req",
        status=scheduler_module.RequestStatus.RUNNING,
        is_prefill_chunk=False,
        num_computed_tokens=1024,
        num_prompt_tokens=512,
        num_tokens=1024,
        max_tokens=256,
        output_token_ids=list(range(64)),
        priority=2,
        kv_transfer_params={},
    )
    scheduler.requests = {"budget-req": request}

    budgets = scheduler._build_reflex_int4_request_precision_budgets(
        reason="admission_waiting",
        target_bf16_blocks=24,
    )

    assert "budget-req" in budgets
    assert any(
        "ReFlexKV trace precision_budget" in message
        and "request=budget-req" in message
        and "request_priority=2" in message
        and "generated_decode_tokens=64" in message
        and "remaining_decode_tokens=192" in message
        and "release_budget_blocks=" in message
        for message in log_messages
    )


def test_reflex_int4_admission_control_logs_landing_frontier(monkeypatch):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=900)
    request = SimpleNamespace(request_id="waiting-landing")
    landing_decision = PrecisionLandingDecision(
        request_id="waiting-landing",
        required_blocks=1056,
        bf16_deficit_blocks=156,
        residual_deficit_after_running=136,
        eligible_int4_landing_blocks=192,
        planned_int4_landing_blocks=136,
        admission_feasible_with_landing=True,
        mixed_landing_required=True,
        reason="mixed_landing_feasible",
    )
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    scheduler._log_reflex_admission_control(
        request=request,
        requested_release=156,
        candidate_release_capacity=20,
        feasible_release=20,
        planned_release=0,
        actual_release=0,
        needed_blocks=1024,
        reserve_blocks=32,
        free_blocks_before=900,
        admission_deficit_blocks=156,
        admission_infeasible=True,
        landing_decision=landing_decision,
    )

    assert log_messages
    assert "landing_mixed_feasible=True" in log_messages[0]
    assert "landing_required_int4_blocks=136" in log_messages[0]
    assert "landing_eligible_int4_blocks=192" in log_messages[0]
    assert "landing_planned_int4_blocks=136" in log_messages[0]
    assert "landing_reason=mixed_landing_feasible" in log_messages[0]


def test_reflex_int4_admission_control_logs_frontier_rejection_reason(monkeypatch):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=1)
    scheduler._reflex_int4_scheduler_step = 21
    request = SimpleNamespace(request_id="waiting-frontier")
    breakdown = ReflexCandidateBreakdown(
        raw_bf16_pages=100,
        eligible_full_unshared_pages=100,
        after_initial_recent_protection=100,
        after_low_risk_filter=90,
        after_request_budget_cap=10,
        after_sparse_window_quota=8,
        after_frontier_optimizer=8,
        after_int4_pool_limit=8,
        selected_actual=8,
    )
    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=21,
        reason="admission_waiting",
        target_release=20,
        feasible_release=8,
        candidate_breakdown=breakdown,
    )
    scheduler._get_reflex_int4_frontier_cache().update(summary)
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    scheduler._log_reflex_admission_control(
        request=request,
        requested_release=20,
        candidate_release_capacity=8,
        feasible_release=8,
        planned_release=8,
        actual_release=0,
        needed_blocks=18,
        reserve_blocks=2,
        free_blocks_before=1,
        admission_deficit_blocks=19,
        admission_infeasible=True,
    )

    assert log_messages
    assert "blocked_reason=request_budget" in log_messages[0]
    assert "frontier_age=0" in log_messages[0]
    assert "frontier_levels=pinned:0,protected:10,candidate:90" in log_messages[0]
    assert "frontier_rejection_reasons=shared_or_open:0" in log_messages[0]
    assert "request_budget:80" in log_messages[0]
    assert "sparse_quota:2" in log_messages[0]


def test_reflex_int4_admission_control_logs_preserved_frontier_after_cache_invalidation(
    monkeypatch,
):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=1)
    scheduler._reflex_int4_scheduler_step = 22
    request = SimpleNamespace(request_id="waiting-preserved-frontier")
    breakdown = ReflexCandidateBreakdown(
        raw_bf16_pages=32,
        eligible_full_unshared_pages=32,
        after_initial_recent_protection=32,
        after_low_risk_filter=32,
        after_request_budget_cap=4,
        after_sparse_window_quota=4,
        after_frontier_optimizer=4,
        after_int4_pool_limit=4,
        selected_actual=4,
    )
    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=22,
        reason="admission_waiting",
        target_release=10,
        feasible_release=4,
        candidate_breakdown=breakdown,
    )
    scheduler._get_reflex_int4_frontier_cache().update(summary)
    preserved_summary = scheduler._get_reflex_int4_frontier_cache().latest()
    scheduler._get_reflex_int4_frontier_cache().invalidate()
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    scheduler._log_reflex_admission_control(
        request=request,
        requested_release=10,
        candidate_release_capacity=4,
        feasible_release=4,
        planned_release=4,
        actual_release=4,
        needed_blocks=18,
        reserve_blocks=2,
        free_blocks_before=1,
        admission_deficit_blocks=19,
        admission_infeasible=True,
        frontier_summary=preserved_summary,
    )

    assert log_messages
    assert "frontier_age=0" in log_messages[0]
    assert "frontier_rejection_reasons=shared_or_open:0" in log_messages[0]
    assert "request_budget:28" in log_messages[0]


def test_reflex_int4_waiting_demotion_only_step_executes_feasible_frontier():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=500)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(
        request_id="waiting-0",
        num_tokens=8192,
        num_computed_tokens=0,
    )
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    demotion = ReflexDemotion(
        request_id="req-0",
        page_idx=0,
        bf16_block_id=0,
        int4_block_id=0,
        encoded_block_table_id=-1,
        kv_cache_group_id=0,
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [demotion]
    scheduler._update_after_schedule = lambda scheduler_output: None
    scheduler.kv_cache_manager.plan_reflex_int4_demotions = (
        lambda **kwargs: kwargs["target_bf16_blocks"]
    )
    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return target_bf16_blocks

    scheduler._try_reflex_int4_demote = fake_try_demote

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert requested_targets == [(44, True, "admission_waiting")]


def test_reflex_int4_feasible_release_uses_cached_frontier_summary():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler._reflex_int4_scheduler_step = 20
    scheduler._reflex_int4_frontier_cache = FeasibleFrontierCache(
        max_age_steps=4
    )
    breakdown = ReflexCandidateBreakdown(
        raw_bf16_pages=8,
        eligible_full_unshared_pages=8,
        after_initial_recent_protection=8,
        after_low_risk_filter=8,
        after_request_budget_cap=8,
        after_sparse_window_quota=8,
        after_int4_pool_limit=7,
        selected_actual=7,
    )
    summary = FeasibleFrontierSummary.from_candidate_breakdown(
        scheduler_step=18,
        reason="admission_waiting",
        target_release=64,
        feasible_release=7,
        candidate_breakdown=breakdown,
    )
    scheduler._reflex_int4_frontier_cache.update(summary)

    def fail_plan(**kwargs):
        raise AssertionError("fresh frontier cache should avoid dry-run plan")

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fail_plan

    feasible_release = scheduler._estimate_reflex_int4_feasible_release(
        target_bf16_blocks=64,
        reason="admission_waiting",
    )

    assert feasible_release == 7
    assert scheduler._reflex_int4_last_demote_candidate_capacity == 7
    assert scheduler._reflex_int4_last_candidate_breakdown is breakdown


def test_reflex_int4_admission_ticket_defers_waiting_retry_until_due_step():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler._reflex_int4_scheduler_step = 30
    scheduler._reflex_int4_admission_ticket_retry_delay_steps = 8
    request = SimpleNamespace(request_id="ticketed-waiting")

    ticket = scheduler._record_reflex_int4_admission_ticket(
        request=request,
        required_blocks=512,
        blocked_reason="full_sequence_reserve",
    )

    assert ticket.next_retry_step == 38
    assert (
        scheduler._should_skip_reflex_int4_waiting_request_by_ticket(request)
        is True
    )

    scheduler._reflex_int4_scheduler_step = 38

    assert (
        scheduler._should_skip_reflex_int4_waiting_request_by_ticket(request)
        is False
    )


def test_reflex_int4_ticket_retries_on_bf16_freed_event_next_step():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler._reflex_int4_scheduler_step = 30
    scheduler._reflex_int4_admission_ticket_retry_delay_steps = 8
    request = SimpleNamespace(request_id="ticketed-after-demotion")
    scheduler._record_reflex_int4_admission_ticket(
        request=request,
        required_blocks=512,
        blocked_reason="admission_waiting",
    )

    scheduler._record_reflex_int4_frontier_event("bf16_freed")

    assert (
        scheduler._should_skip_reflex_int4_waiting_request_by_ticket(request)
        is True
    )

    scheduler._reflex_int4_scheduler_step = 31

    assert (
        scheduler._should_skip_reflex_int4_waiting_request_by_ticket(request)
        is False
    )


def test_reflex_int4_full_sequence_ticket_ignores_bf16_freed_before_due_step():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler._reflex_int4_scheduler_step = 30
    scheduler._reflex_int4_admission_ticket_retry_delay_steps = 8
    request = SimpleNamespace(request_id="full-sequence-ticket")
    scheduler._record_reflex_int4_admission_ticket(
        request=request,
        required_blocks=512,
        blocked_reason="full_sequence_reserve",
    )

    scheduler._record_reflex_int4_frontier_event("bf16_freed")
    scheduler._reflex_int4_scheduler_step = 31

    assert (
        scheduler._should_skip_reflex_int4_waiting_request_by_ticket(request)
        is True
    )

    scheduler._record_reflex_int4_frontier_event("request_finished")

    assert (
        scheduler._should_skip_reflex_int4_waiting_request_by_ticket(request)
        is False
    )


def test_reflex_int4_admission_ticket_retry_delay_backs_off():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=0)
    scheduler._reflex_int4_scheduler_step = 30
    scheduler._reflex_int4_admission_ticket_retry_delay_steps = 8
    scheduler._reflex_int4_admission_ticket_max_retry_delay_steps = 64
    request = SimpleNamespace(request_id="backoff-ticket")

    first = scheduler._record_reflex_int4_admission_ticket(
        request=request,
        required_blocks=512,
        blocked_reason="full_sequence_reserve",
    )
    scheduler._reflex_int4_scheduler_step = first.next_retry_step
    second = scheduler._record_reflex_int4_admission_ticket(
        request=request,
        required_blocks=512,
        blocked_reason="full_sequence_reserve",
    )

    assert second.next_retry_step == first.next_retry_step + 16


def test_reflex_int4_demotion_only_step_builds_kv_connector_metadata():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.connector = object()
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    request = SimpleNamespace(num_tokens=32768, num_computed_tokens=0)
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    demotion = ReflexDemotion(
        request_id="req-0",
        page_idx=0,
        bf16_block_id=0,
        int4_block_id=0,
        encoded_block_table_id=-1,
        kv_cache_group_id=0,
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [demotion]
    scheduler._update_after_schedule = lambda scheduler_output: None
    scheduler._try_reflex_int4_demote = (
        lambda *, target_bf16_blocks, force=False, reason="pressure": 1
    )

    metadata = object()
    scheduler._build_kv_connector_meta = lambda connector, output: metadata

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert scheduler_output.kv_connector_metadata is metadata


def test_reflex_int4_waiting_demotion_only_step_skips_active_prefill():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.running = [SimpleNamespace(is_prefill_chunk=True)]
    scheduler.waiting = True
    scheduler.skipped_waiting = False

    def fail_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        raise AssertionError("active prefill should not run demotion-only step")

    scheduler._try_reflex_int4_demote = fail_try_demote

    assert scheduler._try_reflex_int4_demotion_only_step() is None


def test_reflex_int4_waiting_demotion_only_step_skips_inflight_prefill():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.running = [SimpleNamespace(is_prefill_chunk=False)]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler._reflex_int4_prev_step_had_prefill = True

    def fail_select_waiting_queue():
        raise AssertionError("in-flight prefill should skip before queue access")

    scheduler._select_waiting_queue_for_scheduling = fail_select_waiting_queue

    assert scheduler._try_reflex_int4_demotion_only_step() is None


def test_reflex_int4_waiting_demotion_only_step_respects_cooldown():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 9
    request = SimpleNamespace(num_tokens=32768, num_computed_tokens=0)
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )

    def fail_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        raise AssertionError("cooldown should skip admission demotion")

    scheduler._try_reflex_int4_demote = fail_try_demote

    assert scheduler._try_reflex_int4_demotion_only_step() is None


def test_reflex_int4_demote_logs_actual_release_and_skipped_pages(monkeypatch):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.requests = {}
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0

    def fake_plan(**kwargs):
        assert kwargs["target_bf16_blocks"] == 128
        assert kwargs["keep_recent_pages"] == 4
        assert kwargs["keep_initial_pages"] == 4
        assert kwargs["max_int4_fraction_per_request"] == 1.0
        assert kwargs["low_risk_only"] is True
        assert kwargs["sparse_window_pages"] == 32
        assert kwargs["max_demote_per_window"] == 8
        return 64

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan
    log_messages = []

    def fake_log(message, *args):
        log_messages.append(message % args)

    monkeypatch.setattr(scheduler_module.logger, "info", fake_log)

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=128,
        force=True,
        reason="admission_waiting",
    )

    assert released == 64
    assert "actual_release=64" in log_messages[0]
    assert "skipped_pages=64" in log_messages[0]


def test_reflex_int4_demote_logs_candidate_breakdown(monkeypatch):
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.requests = {}
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = lambda **kwargs: 2
    scheduler.kv_cache_manager.get_last_reflex_int4_candidate_capacity = lambda: 3
    scheduler.kv_cache_manager.get_last_reflex_int4_candidate_breakdown = (
        lambda: SimpleNamespace(
            raw_bf16_pages=20,
            eligible_full_unshared_pages=18,
            after_initial_recent_protection=14,
            after_low_risk_filter=7,
            after_request_budget_cap=5,
            after_sparse_window_quota=3,
            after_int4_pool_limit=3,
            selected_actual=2,
        )
    )
    log_messages = []
    monkeypatch.setattr(
        scheduler_module.logger,
        "info",
        lambda message, *args: log_messages.append(message % args),
    )

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=8,
        force=True,
        reason="admission_waiting",
    )

    assert released == 2
    breakdown_logs = [
        message
        for message in log_messages
        if "ReFlexKV trace candidate_breakdown" in message
    ]
    assert breakdown_logs
    assert "reason=admission_waiting" in breakdown_logs[0]
    assert "selection_policy=relevance_sparse" in breakdown_logs[0]
    assert "raw_bf16_pages=20" in breakdown_logs[0]
    assert "eligible_full_unshared_pages=18" in breakdown_logs[0]
    assert "after_low_risk_filter=7" in breakdown_logs[0]
    assert "after_sparse_window_quota=3" in breakdown_logs[0]
    assert "selected_actual=2" in breakdown_logs[0]


def test_reflex_int4_demote_passes_accuracy_protection_knobs():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.requests = {}
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_keep_recent_pages = 16
    scheduler._reflex_int4_keep_initial_pages = 32
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=8,
        force=True,
        reason="admission_waiting",
    )

    assert released == 8
    assert captured["keep_recent_pages"] == 16
    assert captured["keep_initial_pages"] == 32
    assert captured["max_int4_fraction_per_request"] == 0.5
    assert captured["low_risk_only"] is True


def test_reflex_int4_demote_configures_distance_only_ablation():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.requests = {}
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_page_selection_policy = "distance"
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 4

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=4,
        force=True,
        reason="admission_waiting",
    )

    assert released == 4
    assert captured["low_risk_only"] is False
    assert captured["selection_policy"] == "distance"
    assert captured["sparse_window_pages"] == 0


def test_reflex_int4_demote_configures_frontier_dual_optimizer():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=128)
    scheduler.requests = {}
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_page_selection_policy = "frontier_dual"
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 4

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=4,
        force=True,
        reason="admission_waiting",
    )

    assert released == 4
    assert captured["selection_policy"] == "frontier_dual"
    assert captured["low_risk_only"] is False
    assert captured["sparse_window_pages"] == (
        scheduler._reflex_int4_admission_sparse_window_pages
    )
    assert isinstance(captured["dual_price_state"], DualPriceState)
    assert captured["dual_price_state"].admission_price > 0.5
    assert captured["emergency_release"] is False


def test_reflex_int4_frontier_dual_uses_emergency_release_for_large_admission_waiting():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=128)
    scheduler.requests = {}
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_page_selection_policy = "frontier_dual"
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 64

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=scheduler._reflex_int4_admission_reserve_blocks * 2,
        force=True,
        reason="admission_waiting",
    )

    assert released == 64
    assert captured["selection_policy"] == "frontier_dual"
    assert captured["emergency_release"] is True


def test_reflex_int4_frontier_dual_uses_emergency_release_for_allocation_failure():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=128)
    scheduler.requests = {}
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_page_selection_policy = "frontier_dual"
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 4

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=4,
        force=True,
        reason="allocation_failure",
    )

    assert released == 4
    assert captured["selection_policy"] == "frontier_dual"
    assert captured["emergency_release"] is True
    assert captured["dual_price_state"].admission_price > 0.5


def test_reflex_int4_demote_builds_request_aware_precision_budgets():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    short_req = SimpleNamespace(
        request_id="short",
        num_computed_tokens=4096,
        num_prompt_tokens=4096,
        max_tokens=256,
        output_token_ids=[0] * 96,
        is_prefill_chunk=False,
    )
    long_req = SimpleNamespace(
        request_id="long",
        num_computed_tokens=8192,
        num_prompt_tokens=8192,
        max_tokens=512,
        output_token_ids=[0] * 160,
        is_prefill_chunk=False,
    )
    scheduler.requests = {
        "short": short_req,
        "long": long_req,
    }
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    released = scheduler._try_reflex_int4_demote(
        target_bf16_blocks=8,
        force=True,
        reason="admission_waiting",
    )

    assert released == 8
    budgets = captured["request_precision_budgets"]
    assert budgets["short"].max_int4_fraction > 0.0
    assert budgets["long"].max_int4_fraction == 0.5
    assert budgets["long"].release_budget_blocks > (
        budgets["short"].release_budget_blocks
    )
    assert (
        budgets["short"].release_budget_blocks
        + budgets["long"].release_budget_blocks
        == 8
    )
    assert budgets["long"].priority > budgets["short"].priority


def test_reflex_int4_request_budget_applies_pressure_window_multiplier():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=64)
    scheduler._reflex_int4_last_candidate_breakdown = ReflexCandidateBreakdown(
        after_initial_recent_protection=8192,
        after_low_risk_filter=4096,
        after_request_budget_cap=128,
        after_sparse_window_quota=16,
        after_frontier_optimizer=16,
        after_int4_pool_limit=16,
    )
    request = SimpleNamespace(
        request_id="quota-pressure",
        num_computed_tokens=2048 * scheduler.block_size,
        num_prompt_tokens=1024 * scheduler.block_size,
        max_tokens=512,
        output_token_ids=[0] * 256,
        priority=0,
        is_prefill_chunk=False,
        kv_transfer_params={},
    )
    scheduler.requests = {"quota-pressure": request}

    budgets = scheduler._build_reflex_int4_request_precision_budgets(
        reason="admission_waiting",
        target_bf16_blocks=64,
    )

    assert budgets["quota-pressure"].max_demote_per_window > (
        scheduler._reflex_int4_admission_max_demote_per_window
    )


def test_reflex_int4_demote_passes_prefill_page_metadata():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    request = SimpleNamespace(
        request_id="qasper-like",
        num_computed_tokens=4096 + 16,
        num_prompt_tokens=4096,
        max_tokens=128,
        output_token_ids=[0] * 16,
        priority=0,
        is_prefill_chunk=False,
        kv_transfer_params={
            "reflex_page_risks": [0.9, 0.2, 0.8, 0.1],
            "reflex_compressible_pages": [1, 3],
        },
    )
    scheduler.requests = {"qasper-like": request}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 2

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=2,
        force=True,
        reason="admission_waiting",
    )

    assert captured["prefill_page_risks_by_request"] == {
        "qasper-like": [0.9, 0.2, 0.8, 0.1],
    }
    assert captured["compressible_pages_by_request"] == {
        "qasper-like": {1, 3},
    }


def test_reflex_int4_demote_passes_reasoning_prompt_protection():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    request = SimpleNamespace(
        request_id="math-like",
        num_computed_tokens=(8 + 160) * scheduler.block_size,
        num_prompt_tokens=8 * scheduler.block_size,
        max_tokens=4096,
        output_token_ids=[0] * 160,
        priority=1,
        is_prefill_chunk=False,
        kv_transfer_params={
            "reflex_page_risks": [0.1] * 8,
            "reflex_compressible_pages": list(range(8)),
        },
    )
    scheduler.requests = {"math-like": request}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=8,
        force=True,
        reason="admission_waiting",
    )

    assert captured["protected_prompt_pages_by_request"] == {"math-like": 8}


def test_reflex_int4_derives_compressible_pages_from_prefill_scores():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    request = SimpleNamespace(
        request_id="score-only",
        kv_transfer_params={
            "reflex_page_risks": [0.9, 0.2, 0.8, 0.1, 0.7, 0.05, 0.6, 0.4],
        },
    )
    scheduler.requests = {"score-only": request}

    risks, compressible_pages = scheduler._build_reflex_prefill_page_metadata_inputs()

    assert risks == {
        "score-only": [0.9, 0.2, 0.8, 0.1, 0.7, 0.05, 0.6, 0.4],
    }
    assert compressible_pages == {
        "score-only": {3, 5},
    }


def test_reflex_int4_scheduler_has_only_budgeted_background_promotion():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=1024)

    assert not hasattr(scheduler, "_try_reflex_int4_precision_fault_recover")
    assert not hasattr(scheduler, "_apply_reflex_page_attention_mass_output")


def test_reflex_int4_background_promotion_runs_under_low_pressure():
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=3500,
        total_blocks=4096,
    )
    scheduler._reflex_int4_background_promotion_free_ratio = 0.60
    scheduler._reflex_int4_background_promotion_pages_per_step = 2
    scheduler._reflex_int4_promotion_min_remaining_decode_tokens = 16
    scheduler.waiting = []
    scheduler.skipped_waiting = []
    request = SimpleNamespace(
        request_id="long-running",
        max_tokens=128,
        output_token_ids=[0] * 32,
        kv_transfer_params={
            "reflex_page_risks": [0.1, 0.8, 0.6],
        },
    )
    scheduler.requests = {"long-running": request}
    captured = {}

    def fake_promote(**kwargs):
        captured.update(kwargs)
        return 2

    scheduler.kv_cache_manager.promote_reflex_recoverable_pages = fake_promote

    promoted = scheduler._try_reflex_int4_background_promote()

    assert promoted == 2
    assert captured["max_pages"] == 2
    assert captured["prefill_page_risks_by_request"] == {
        "long-running": [0.1, 0.8, 0.6],
    }
    assert captured["remaining_decode_tokens_by_request"] == {
        "long-running": 96,
    }
    assert captured["min_remaining_decode_tokens"] == 16


def test_reflex_int4_background_promotion_skips_when_waiting_pressure_exists():
    scheduler = _make_scheduler_for_reflex_target(
        free_blocks=3500,
        total_blocks=4096,
    )
    scheduler._reflex_int4_background_promotion_free_ratio = 0.60
    scheduler._reflex_int4_background_promotion_pages_per_step = 2
    scheduler.waiting = [object()]
    scheduler.skipped_waiting = []
    scheduler.requests = {}

    def fail_promote(**kwargs):
        raise AssertionError("background promotion should not run under pressure")

    scheduler.kv_cache_manager.promote_reflex_recoverable_pages = fail_promote

    assert scheduler._try_reflex_int4_background_promote() == 0


def test_reflex_int4_request_pressure_increases_with_decode_and_prefill_length():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    cold_short_prompt = SimpleNamespace(
        request_id="cold-short-prompt",
        num_prompt_tokens=1024,
        output_token_ids=[],
        priority=0,
    )
    cold_long_prompt = SimpleNamespace(
        request_id="cold-long-prompt",
        num_prompt_tokens=16384,
        output_token_ids=[],
        priority=0,
    )
    survived_long_prompt = SimpleNamespace(
        request_id="survived-long-prompt",
        num_prompt_tokens=16384,
        output_token_ids=[0] * 256,
        priority=0,
    )
    low_slo_survived_long_prompt = SimpleNamespace(
        request_id="low-slo-survived-long-prompt",
        num_prompt_tokens=16384,
        output_token_ids=[0] * 256,
        priority=1,
    )

    cold_short_pressure = scheduler._reflex_int4_request_demotion_pressure(
        cold_short_prompt
    )
    cold_long_pressure = scheduler._reflex_int4_request_demotion_pressure(
        cold_long_prompt
    )
    survived_long_pressure = scheduler._reflex_int4_request_demotion_pressure(
        survived_long_prompt
    )
    low_slo_survived_long_pressure = (
        scheduler._reflex_int4_request_demotion_pressure(
            low_slo_survived_long_prompt
        )
    )

    assert cold_short_pressure < cold_long_pressure
    assert cold_long_pressure < survived_long_pressure
    assert survived_long_pressure < low_slo_survived_long_pressure


def test_reflex_int4_cold_admission_budget_scales_with_prompt_length():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    short_prompt_req = SimpleNamespace(
        request_id="short-prompt",
        num_computed_tokens=1024 + 16,
        num_prompt_tokens=1024,
        max_tokens=512,
        output_token_ids=[0] * 16,
        priority=0,
        is_prefill_chunk=False,
    )
    long_prompt_req = SimpleNamespace(
        request_id="long-prompt",
        num_computed_tokens=16384 + 16,
        num_prompt_tokens=16384,
        max_tokens=512,
        output_token_ids=[0] * 16,
        priority=0,
        is_prefill_chunk=False,
    )
    scheduler.requests = {
        "short-prompt": short_prompt_req,
        "long-prompt": long_prompt_req,
    }
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 16

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=16,
        force=True,
        reason="admission_waiting",
    )

    budgets = captured["request_precision_budgets"]
    assert budgets["long-prompt"].max_int4_fraction > (
        budgets["short-prompt"].max_int4_fraction
    )
    assert budgets["long-prompt"].max_int4_pages > (
        budgets["short-prompt"].max_int4_pages
    )
    assert budgets["long-prompt"].release_budget_blocks > (
        budgets["short-prompt"].release_budget_blocks
    )


def test_reflex_int4_admission_release_budget_uses_remaining_int4_capacity():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    saturated_req = SimpleNamespace(
        request_id="saturated",
        num_computed_tokens=8192 + 256,
        num_prompt_tokens=8192,
        max_tokens=512,
        output_token_ids=[0] * 256,
        priority=1,
        is_prefill_chunk=False,
    )
    fresh_req = SimpleNamespace(
        request_id="fresh",
        num_computed_tokens=8192 + 256,
        num_prompt_tokens=8192,
        max_tokens=512,
        output_token_ids=[0] * 256,
        priority=-1,
        is_prefill_chunk=False,
    )
    scheduler.requests = {
        "saturated": saturated_req,
        "fresh": fresh_req,
    }
    max_saturated_pages = int(
        (saturated_req.num_computed_tokens // scheduler.block_size)
        * scheduler._reflex_int4_max_int4_fraction_per_request
    )

    def fake_counts(request_id):
        return {
            "BF16_ACTIVE": 0,
            "INT4_ACTIVE": max_saturated_pages if request_id == "saturated" else 0,
            "INT4_RECOVERABLE": 0,
            "BF16_RECOVERED": 0,
            "RELEASE_PENDING": 0,
            "LANDING_RESERVED": 0,
        }

    scheduler.kv_cache_manager.get_reflex_precision_state_counts = fake_counts

    budgets = scheduler._build_reflex_int4_request_precision_budgets(
        reason="admission_waiting",
        target_bf16_blocks=16,
    )

    assert budgets["saturated"].max_int4_pages > 0
    assert budgets["saturated"].release_budget_blocks == 0
    assert budgets["fresh"].release_budget_blocks == 16


def test_reflex_int4_short_decode_admission_waits_when_cold_cap_disabled():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=512)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_short_decode_tokens = 128
    scheduler._reflex_int4_short_decode_max_int4_fraction = 0.0
    scheduler._reflex_int4_short_admission_max_int4_fraction = 0.03
    scheduler._reflex_int4_cold_admission_max_int4_fraction = 0.0
    scheduler._reflex_int4_risk_warmup_tokens = 16
    short_output_req = SimpleNamespace(
        request_id="hotpot-like",
        num_computed_tokens=16384,
        num_prompt_tokens=16384,
        max_tokens=32,
        output_token_ids=[],
        priority=0,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"hotpot-like": short_output_req}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 0

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=128,
        force=True,
        reason="admission_waiting",
    )

    budget = captured["request_precision_budgets"]["hotpot-like"]
    assert budget.max_int4_fraction == 0.0
    assert budget.max_int4_pages == 0
    assert budget.release_budget_blocks == 0


def test_reflex_int4_emergency_admission_uses_deficit_before_risk_warmup():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=1024)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_risk_warmup_tokens = 16
    scheduler._reflex_int4_cold_admission_max_int4_fraction = 0.10
    scheduler._reflex_int4_cold_admission_emergency_free_ratio = 0.05
    cold_req = SimpleNamespace(
        request_id="cold-emergency",
        num_computed_tokens=16384,
        num_prompt_tokens=16384,
        max_tokens=512,
        output_token_ids=[],
        priority=1,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"cold-emergency": cold_req}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=8,
        force=True,
        reason="full_sequence_reserve",
    )

    budget = captured["request_precision_budgets"]["cold-emergency"]
    assert 0.0 < budget.max_int4_fraction <= 0.10
    assert budget.max_int4_pages > 0
    assert budget.release_budget_blocks == 8


def test_reflex_int4_short_decode_admission_gets_small_cap_after_one_page():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_short_decode_tokens = 128
    scheduler._reflex_int4_short_decode_max_int4_fraction = 0.0
    scheduler._reflex_int4_short_admission_max_int4_fraction = 0.03
    scheduler._reflex_int4_risk_warmup_tokens = 16
    short_output_req = SimpleNamespace(
        request_id="qasper-like",
        num_computed_tokens=16384 + 16,
        num_prompt_tokens=16384,
        max_tokens=128,
        output_token_ids=[0] * 16,
        priority=0,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"qasper-like": short_output_req}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=128,
        force=True,
        reason="admission_waiting",
    )

    budget = captured["request_precision_budgets"]["qasper-like"]
    assert budget.max_int4_fraction == 0.03
    assert budget.max_int4_pages > 0
    assert budget.release_budget_blocks > 0


def test_reflex_int4_short_decode_admission_cap_uses_slo_pressure_only():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_short_decode_tokens = 128
    scheduler._reflex_int4_short_decode_max_int4_fraction = 0.0
    scheduler._reflex_int4_short_admission_max_int4_fraction = 0.03
    scheduler._reflex_int4_risk_warmup_tokens = 16
    requests = {}
    for request_id, priority in (
        ("high-slo", -1),
        ("normal-slo", 0),
        ("low-slo", 1),
    ):
        requests[request_id] = SimpleNamespace(
            request_id=request_id,
            num_computed_tokens=16384 + 16,
            num_prompt_tokens=16384,
            max_tokens=128,
            output_token_ids=[0] * 16,
            priority=priority,
            is_prefill_chunk=False,
        )
    scheduler.requests = requests
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=128,
        force=True,
        reason="admission_waiting",
    )

    budgets = captured["request_precision_budgets"]
    assert budgets["high-slo"].max_int4_fraction == 0.0225
    assert budgets["normal-slo"].max_int4_fraction == 0.03
    assert budgets["low-slo"].max_int4_fraction == 0.0375
    assert budgets["high-slo"].max_demote_per_window == 8
    assert budgets["normal-slo"].max_demote_per_window == 8
    assert budgets["low-slo"].max_demote_per_window == 8
    assert budgets["low-slo"].release_budget_blocks > (
        budgets["high-slo"].release_budget_blocks
    )


def test_reflex_int4_admission_pressure_sets_min_fraction_for_survived_long_prompt():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_short_decode_tokens = 128
    scheduler._reflex_int4_short_admission_max_int4_fraction = 0.03
    scheduler._reflex_int4_admission_pressure_min_int4_fraction = 0.10
    scheduler._reflex_int4_risk_warmup_tokens = 16
    request = SimpleNamespace(
        request_id="survived-high-slo",
        num_computed_tokens=16384 + 448,
        num_prompt_tokens=16384,
        max_tokens=512,
        output_token_ids=[0] * 448,
        priority=-1,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"survived-high-slo": request}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 64

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=640,
        force=True,
        reason="admission_waiting",
    )

    budget = captured["request_precision_budgets"]["survived-high-slo"]
    assert abs(budget.max_int4_fraction - 0.075) < 1e-9
    assert budget.max_int4_pages >= 78
    assert budget.release_budget_blocks > 0


def test_reflex_int4_admission_pressure_expands_sparse_window_quota():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_sparse_window_pages = 32
    scheduler._reflex_int4_max_demote_per_window = 2
    scheduler._reflex_int4_admission_sparse_window_pages = 32
    scheduler._reflex_int4_admission_max_demote_per_window = 8
    request = SimpleNamespace(
        request_id="long-running",
        num_computed_tokens=16384 + 512,
        num_prompt_tokens=16384,
        max_tokens=2048,
        output_token_ids=[0] * 512,
        priority=1,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"long-running": request}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 64

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=640,
        force=True,
        reason="admission_waiting",
    )

    assert captured["sparse_window_pages"] == 32
    assert captured["max_demote_per_window"] == 8
    budget = captured["request_precision_budgets"]["long-running"]
    assert budget.max_demote_per_window == 8


def test_reflex_int4_candidate_funnel_pressure_expands_next_release_target():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=543)
    scheduler.running = [object()]
    scheduler.waiting = True
    scheduler.skipped_waiting = False
    scheduler.finished_req_ids = set()
    scheduler.encoder_cache_manager = SimpleNamespace(
        get_freed_mm_hashes=lambda: []
    )
    scheduler._reflex_int4_last_candidate_breakdown = ReflexCandidateBreakdown(
        after_initial_recent_protection=4096,
        after_low_risk_filter=2048,
        after_request_budget_cap=32,
        after_sparse_window_quota=32,
        after_int4_pool_limit=32,
    )
    request = SimpleNamespace(num_tokens=8192, num_computed_tokens=0)
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: request
    )
    demotion = ReflexDemotion(
        request_id="req-0",
        page_idx=0,
        bf16_block_id=0,
        int4_block_id=0,
        encoded_block_table_id=-1,
        kv_cache_group_id=0,
    )
    scheduler.kv_cache_manager.take_reflex_int4_demotions = lambda: [demotion]
    scheduler._update_after_schedule = lambda scheduler_output: None
    requested_targets = []

    def fake_try_demote(*, target_bf16_blocks, force=False, reason="pressure"):
        requested_targets.append((target_bf16_blocks, force, reason))
        return target_bf16_blocks

    scheduler._try_reflex_int4_demote = fake_try_demote

    scheduler_output = scheduler._try_reflex_int4_demotion_only_step()

    assert scheduler_output is not None
    assert requested_targets == [(256, True, "admission_waiting")]


def test_reflex_int4_candidate_funnel_pressure_expands_low_risk_and_sparse_kwargs():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_low_risk_score_fraction = 0.25
    scheduler._reflex_int4_max_demote_per_window = 2
    scheduler._reflex_int4_admission_max_demote_per_window = 8
    scheduler._reflex_int4_last_candidate_breakdown = ReflexCandidateBreakdown(
        after_initial_recent_protection=4096,
        after_low_risk_filter=256,
        after_request_budget_cap=256,
        after_sparse_window_quota=32,
        after_int4_pool_limit=32,
    )
    request = SimpleNamespace(
        request_id="long-context",
        num_computed_tokens=16384 + 512,
        num_prompt_tokens=16384,
        max_tokens=1024,
        output_token_ids=[0] * 512,
        priority=0,
        is_prefill_chunk=False,
        kv_transfer_params={
            "reflex_page_risks": [0.9, 0.8, 0.1, 0.2] * 256,
        },
    )
    scheduler.requests = {"long-context": request}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 64

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=64,
        force=True,
        reason="admission_waiting",
    )

    assert captured["max_demote_per_window"] > 8
    assert len(captured["compressible_pages_by_request"]["long-context"]) > 256


def test_reflex_int4_planning_passes_sealed_chunk_and_copy_on_demote_pages():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    request = SimpleNamespace(
        request_id="chunk-shared",
        num_computed_tokens=0,
        num_prompt_tokens=4096,
        max_tokens=1024,
        output_token_ids=[0] * 256,
        priority=0,
        is_prefill_chunk=False,
        kv_transfer_params={
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_page_end": 64,
            "reflex_copy_on_demote_pages": [1, 2, 3],
        },
    )
    scheduler.requests = {"chunk-shared": request}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 64

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=64,
        force=True,
        reason="admission_waiting",
    )

    assert captured["sealed_pages_by_request"] == {"chunk-shared": 64}
    assert captured["copy_on_demote_pages_by_request"] == {
        "chunk-shared": {1, 2, 3}
    }


def test_reflex_int4_planning_passes_remote_inflight_pages():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    request = SimpleNamespace(
        request_id="chunk-inflight",
        num_computed_tokens=0,
        num_prompt_tokens=4096,
        max_tokens=1024,
        output_token_ids=[],
        priority=0,
        is_prefill_chunk=False,
        kv_transfer_params={
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_page_start": 64,
            "reflex_remote_chunk_page_end": 96,
            "reflex_remote_chunk_committed_page_end": 64,
            "reflex_remote_chunk_inflight": True,
        },
    )
    scheduler.requests = {"chunk-inflight": request}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 0

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=32,
        force=True,
        reason="admission_waiting",
    )

    assert captured["sealed_pages_by_request"] == {"chunk-inflight": 64}
    assert captured["remote_inflight_pages_by_request"] == {
        "chunk-inflight": set(range(64, 96))
    }


def test_reflex_remote_chunk_commit_frontier_ignores_inflight_page_end():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    request = SimpleNamespace(
        request_id="chunked",
        kv_transfer_params={
            "reflex_remote_chunk_enabled": True,
            "reflex_remote_chunk_id": 1,
            "reflex_remote_chunk_token_end": 1024,
            "reflex_remote_chunk_page_end": 64,
            "reflex_remote_chunk_is_last": False,
            "reflex_remote_chunk_inflight": True,
        },
    )

    assert scheduler._reflex_remote_chunk_sealed_pages(request) == 0

    scheduler._commit_reflex_remote_chunk(request)

    assert request.kv_transfer_params["reflex_remote_chunk_inflight"] is False
    assert (
        request.kv_transfer_params["reflex_remote_chunk_committed_page_end"]
        == 64
    )
    assert scheduler._reflex_remote_chunk_sealed_pages(request) == 64

    request.kv_transfer_params.update(
        {
            "reflex_remote_chunk_id": 2,
            "reflex_remote_chunk_token_end": 1536,
            "reflex_remote_chunk_page_end": 96,
            "reflex_remote_chunk_inflight": True,
        }
    )

    assert scheduler._reflex_remote_chunk_sealed_pages(request) == 64


def test_reflex_int4_short_decode_non_admission_stays_protected():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    scheduler._reflex_int4_short_decode_tokens = 128
    scheduler._reflex_int4_short_admission_max_int4_fraction = 0.03
    scheduler._reflex_int4_risk_warmup_tokens = 16
    short_output_req = SimpleNamespace(
        request_id="qasper-like",
        num_computed_tokens=16384 + 32,
        num_prompt_tokens=16384,
        max_tokens=128,
        output_token_ids=[0] * 32,
        priority=0,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"qasper-like": short_output_req}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 0

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=128,
        force=True,
        reason="pressure",
    )

    budget = captured["request_precision_budgets"]["qasper-like"]
    assert budget.max_int4_fraction == 0.0
    assert budget.max_int4_pages == 0
    assert budget.release_budget_blocks == 0


def test_reflex_int4_cold_admission_uses_deficit_not_global_free_ratio():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=1024)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    cold_req = SimpleNamespace(
        request_id="cold",
        num_computed_tokens=8192 + 16,
        num_prompt_tokens=8192,
        max_tokens=512,
        output_token_ids=[0] * 16,
        priority=1,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"cold": cold_req}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 0

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=128,
        force=True,
        reason="admission_waiting",
    )

    budget = captured["request_precision_budgets"]["cold"]
    assert budget.max_int4_fraction > 0.25
    assert budget.max_int4_pages >= 128
    assert budget.release_budget_blocks == 128


def test_reflex_int4_survival_budget_uses_generated_output_tokens():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_scheduler_step = 10
    scheduler._reflex_int4_last_demote_step = 0
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    prompt_heavy_req = SimpleNamespace(
        request_id="prompt-heavy",
        num_computed_tokens=16384,
        num_prompt_tokens=16384,
        max_tokens=512,
        output_token_ids=[],
        is_prefill_chunk=False,
    )
    warm_req = SimpleNamespace(
        request_id="warm",
        num_computed_tokens=8192 + 160,
        num_prompt_tokens=8192,
        max_tokens=512,
        output_token_ids=[0] * 160,
        is_prefill_chunk=False,
    )
    scheduler.requests = {
        "prompt-heavy": prompt_heavy_req,
        "warm": warm_req,
    }
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 8

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=8,
        force=True,
        reason="pressure",
    )

    budgets = captured["request_precision_budgets"]
    assert budgets["prompt-heavy"].max_int4_pages == 0
    assert budgets["prompt-heavy"].release_budget_blocks == 0
    assert budgets["warm"].max_int4_fraction > 0.0
    assert budgets["warm"].release_budget_blocks == 8


def test_reflex_int4_cold_admission_allows_bounded_early_demotions():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 0.5
    cold_req = SimpleNamespace(
        request_id="cold-low-slo",
        num_computed_tokens=1600 + 16,
        num_prompt_tokens=1600,
        max_tokens=512,
        output_token_ids=[0] * 16,
        priority=1,
        is_prefill_chunk=False,
    )
    scheduler.requests = {"cold-low-slo": cold_req}
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 4

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=4,
        force=True,
        reason="admission_waiting",
    )

    budget = captured["request_precision_budgets"]["cold-low-slo"]
    assert 0.1 < budget.max_int4_fraction < 0.5
    assert budget.max_int4_pages > 3
    assert budget.release_budget_blocks == 4


def test_reflex_int4_slo_priority_pushes_budget_to_lower_slo_requests():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler._reflex_int4_max_int4_fraction_per_request = 1.0
    high_slo_req = SimpleNamespace(
        request_id="high-slo",
        num_computed_tokens=4096,
        num_prompt_tokens=4096,
        max_tokens=512,
        output_token_ids=[0] * 160,
        priority=-1,
        is_prefill_chunk=False,
    )
    low_slo_req = SimpleNamespace(
        request_id="low-slo",
        num_computed_tokens=4096,
        num_prompt_tokens=4096,
        max_tokens=512,
        output_token_ids=[0] * 160,
        priority=1,
        is_prefill_chunk=False,
    )
    scheduler.requests = {
        "high-slo": high_slo_req,
        "low-slo": low_slo_req,
    }
    captured = {}

    def fake_plan(**kwargs):
        captured.update(kwargs)
        return 4

    scheduler.kv_cache_manager.plan_reflex_int4_demotions = fake_plan

    scheduler._try_reflex_int4_demote(
        target_bf16_blocks=4,
        force=True,
        reason="admission_waiting",
    )

    budgets = captured["request_precision_budgets"]
    assert budgets["high-slo"].max_int4_fraction > 0.0
    assert budgets["low-slo"].max_int4_fraction < 1.0
    assert budgets["low-slo"].max_int4_fraction > (
        budgets["high-slo"].max_int4_fraction
    )
    assert budgets["low-slo"].priority > budgets["high-slo"].priority
    assert budgets["low-slo"].release_budget_blocks > budgets["high-slo"].release_budget_blocks


def test_reflex_int4_defers_int4_decode_when_waiting_prefill_exists():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    running_req = SimpleNamespace(
        request_id="decode-req",
        num_computed_tokens=100,
        num_prompt_tokens=64,
    )
    waiting_req = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4096)
    scheduler.running = [running_req]
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: waiting_req
    )
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "decode-req"
    )

    assert scheduler._reflex_int4_should_defer_decode_for_prefill()
    assert scheduler._reflex_int4_should_defer_running_request(running_req)


def test_reflex_int4_does_not_defer_bf16_decode_for_waiting_prefill():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    running_req = SimpleNamespace(
        request_id="decode-req",
        num_computed_tokens=100,
        num_prompt_tokens=64,
    )
    waiting_req = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4096)
    scheduler.running = [running_req]
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: waiting_req
    )
    scheduler.kv_cache_manager.has_reflex_int4_blocks = lambda request_id: False

    assert not scheduler._reflex_int4_should_defer_decode_for_prefill()
    assert not scheduler._reflex_int4_should_defer_running_request(running_req)


def test_reflex_int4_does_not_defer_for_kv_transfer_waiting_request():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.connector = object()
    running_req = SimpleNamespace(
        request_id="decode-req",
        num_computed_tokens=100,
        num_prompt_tokens=64,
    )
    waiting_req = SimpleNamespace(num_computed_tokens=0, num_prompt_tokens=4096)
    scheduler.running = [running_req]
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: waiting_req
    )
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "decode-req"
    )

    assert not scheduler._reflex_int4_should_defer_decode_for_prefill()
    assert not scheduler._reflex_int4_should_defer_running_request(running_req)


def test_reflex_int4_does_not_defer_for_partial_remote_waiting_prefill():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.connector = object()
    running_req = SimpleNamespace(
        request_id="decode-req",
        num_computed_tokens=100,
        num_prompt_tokens=64,
    )
    waiting_req = SimpleNamespace(num_computed_tokens=8192, num_prompt_tokens=16384)
    scheduler.running = [running_req]
    scheduler._select_waiting_queue_for_scheduling = lambda: SimpleNamespace(
        peek_request=lambda: waiting_req
    )
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "decode-req"
    )

    assert not scheduler._reflex_int4_should_defer_decode_for_prefill()
    assert not scheduler._reflex_int4_should_defer_running_request(running_req)


def test_reflex_int4_defers_int4_decode_when_running_prefill_exists():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    decode_req = SimpleNamespace(
        request_id="decode-req",
        num_computed_tokens=100,
        num_prompt_tokens=64,
    )
    prefill_req = SimpleNamespace(
        request_id="prefill-req",
        num_computed_tokens=8192,
        num_prompt_tokens=16000,
    )
    scheduler.running = [decode_req, prefill_req]
    scheduler._select_waiting_queue_for_scheduling = lambda: None
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "decode-req"
    )

    assert scheduler._reflex_int4_should_defer_decode_for_prefill()
    assert scheduler._reflex_int4_should_defer_running_request(decode_req)
    assert not scheduler._reflex_int4_should_defer_running_request(prefill_req)


def test_reflex_int4_does_not_defer_for_running_remote_prefill():
    scheduler = _make_scheduler_for_reflex_target(free_blocks=192)
    scheduler.connector = object()
    decode_req = SimpleNamespace(
        request_id="decode-req",
        num_computed_tokens=100,
        num_prompt_tokens=64,
    )
    remote_prefill_req = SimpleNamespace(
        request_id="remote-prefill-req",
        num_computed_tokens=8192,
        num_prompt_tokens=16000,
    )
    scheduler.running = [decode_req, remote_prefill_req]
    scheduler._select_waiting_queue_for_scheduling = lambda: None
    scheduler.kv_cache_manager.has_reflex_int4_blocks = (
        lambda request_id: request_id == "decode-req"
    )

    assert not scheduler._reflex_int4_should_defer_decode_for_prefill()
    assert not scheduler._reflex_int4_should_defer_running_request(decode_req)
