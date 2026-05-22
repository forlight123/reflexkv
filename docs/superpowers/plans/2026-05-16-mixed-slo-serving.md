# Mixed SLO Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Subagent execution is intentionally not used in this session because the active Codex instructions only allow subagents when the user explicitly asks for them.

**Goal:** Build mixed LongBench + Math500 PD serving accuracy runs with random SLO priorities, priority scheduling, and SLO-aware ReFlexKV request demotion budgets.

**Architecture:** Add a focused mixed accuracy runner that reuses existing PD serving setup, dataset loading, request sending, and scoring helpers. Requests from multiple datasets are shuffled into one serving workload, then predictions are regrouped by dataset for scoring. ReFlexKV uses vLLM request priority as the first SLO signal for request-level demotion pressure.

**Tech Stack:** Python argparse/httpx asyncio runner, existing eval helpers, vLLM OpenAI completion priority field, vLLM V1 scheduler ReFlexKV controller.

---

### Task 1: Mixed Workload Runner

**Files:**
- Create: `scripts/accuracy/run_pd_serving_mixed_accuracy.py`
- Test: `tests/accuracy/test_pd_serving_mixed_accuracy.py`

- [ ] Write tests for loading LongBench and Math500 datasets into one shuffled request list.
- [ ] Verify tests fail because the mixed runner does not exist.
- [ ] Implement mixed dataset loading with deterministic SLO assignment.
- [ ] Run the focused tests and keep them green.

### Task 2: Priority Payload and Priority Scheduling

**Files:**
- Modify: `scripts/accuracy/run_pd_serving_accuracy.py`
- Modify: `scripts/accuracy/run_pd_serving_mixed_accuracy.py`
- Modify: `scripts/accuracy/run_pd_serving_regression.py`
- Test: `tests/accuracy/test_pd_serving_accuracy.py`
- Test: `tests/accuracy/test_pd_serving_regression.py`
- Test: `tests/accuracy/test_pd_serving_mixed_accuracy.py`

- [ ] Write tests that completion payloads include priority when supplied.
- [ ] Write tests that mixed regression commands use the mixed runner and `--scheduling-policy priority`.
- [ ] Implement priority pass-through and mixed regression command construction.
- [ ] Run the focused accuracy tests.

### Task 3: SLO-Aware ReFlexKV Request Budget

**Files:**
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`
- Modify: `scripts/profiling/run_reflex_pd_1p1d.py`
- Test: `tests/profiling/test_reflex_int4_scheduler.py`
- Test: `tests/profiling/test_reflex_pd_1p1d_runner.py`

- [ ] Write tests showing lower-SLO requests receive higher demotion utility/budget than high-SLO requests.
- [ ] Write tests showing cold admission pressure can demote a bounded fraction before survival warmup.
- [ ] Implement priority-derived demotion pressure and the cold-admission cap.
- [ ] Run focused scheduler/profiling tests.

### Final Verification

- [ ] Run the combined accuracy/profiling unit tests touched by this change.
- [ ] Run `py_compile` on modified Python modules.
