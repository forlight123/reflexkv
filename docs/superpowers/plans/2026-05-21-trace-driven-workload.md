# Trace-Driven Workload Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible BurstGPT-shaped, answerable benchmark manifests and replay them through the existing mixed PD accuracy runner.

**Architecture:** Keep traffic shape and task content separate. `gen_data/build_trace_driven_manifest.py` reads BurstGPT CSV rows, selects a controllable trace slice, binds each row to an existing LongBench/reasoning prompt, and writes JSONL plus summary files. `scripts/accuracy/run_pd_serving_mixed_accuracy.py` gains an optional manifest path and a trace arrival policy while preserving the current generated workload and Poisson behavior.

**Tech Stack:** Python argparse/csv/json, existing `run_pd_serving_mixed_accuracy` dataset loaders, pytest, asyncio runner.

---

### Task 1: Trace-Driven Manifest Builder

**Files:**
- Create: `gen_data/build_trace_driven_manifest.py`
- Create: `tests/gen_data/test_build_trace_driven_manifest.py`

- [ ] Write failing tests for BurstGPT CSV selection, benchmark binding, trace profile JSONL, manifest JSONL, and summary output.
- [ ] Run the focused test and verify it fails because the module does not exist.
- [ ] Implement the minimal builder using existing mixed workload loaders and deterministic sampling.
- [ ] Run the focused tests and keep them green.

### Task 2: Mixed Runner Manifest Loading

**Files:**
- Modify: `scripts/accuracy/run_pd_serving_mixed_accuracy.py`
- Modify: `tests/accuracy/test_pd_serving_mixed_accuracy.py`

- [ ] Write failing tests for loading a manifest into `MixedWorkload` with prompt, answers, trace token metadata, arrival offsets, max tokens, and SLO priority preserved.
- [ ] Run the focused test and verify it fails because the loader does not exist.
- [ ] Implement manifest loading behind `--workload-manifest` without changing the default generated workload path.
- [ ] Run the focused tests and keep them green.

### Task 3: Trace Arrival Replay

**Files:**
- Modify: `scripts/accuracy/run_pd_serving_mixed_accuracy.py`
- Modify: `tests/accuracy/test_pd_serving_mixed_accuracy.py`

- [ ] Write failing tests showing `--arrival-policy trace` uses per-request arrival offsets instead of Poisson sleeps.
- [ ] Implement trace scheduling with `--trace-time-scale`, defaulting to `1.0`.
- [ ] Include arrival and trace token fields in `mixed_requests.jsonl` and `mixed_request_trace.jsonl`.
- [ ] Run focused runner tests and `py_compile` on modified scripts.
