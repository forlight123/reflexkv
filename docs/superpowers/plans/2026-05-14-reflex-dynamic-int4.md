# ReFlexKV Dynamic INT4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a first serving-oriented ReFlexKV path that dynamically demotes sealed BF16 KV pages to packed INT4 under KV scheduling pressure.

**Architecture:** Replace the current `mixed_static` research path with a dynamic BF16/INT4 design. BF16 remains the write path for new tokens; old sealed pages can migrate to a separate packed INT4 pool, and the original BF16 block is returned to vLLM's scheduler-visible BF16 block pool. Attention initially materializes mixed BF16/INT4 pages into a compact BF16 workspace before calling the existing Triton attention.

**Tech Stack:** vLLM V1 scheduler, `BlockPool`, `SingleTypeKVCacheManager`, GPU model runner, Triton attention backend, existing packed INT4 Triton kernels.

---

## Implementation Boundaries

- First version supports only `BF16 -> INT4`.
- No FP8, residual, recovery, SLO-aware quality scoring, or prefix-cache demotion.
- Prefix-cached/shared blocks are skipped. Running with prefix caching disabled is the preferred first test mode.
- Only full sealed blocks can be demoted. The current append block and recent tail window stay BF16.
- Dynamic demotion is driven by serving pressure, but the immediate runtime actuator is the number of BF16 blocks required by scheduler allocation.
- Evaluation must compare full BF16 against dynamic INT4 using serving metrics: max sustainable request rate, waiting time, TTFT, TPOT, goodput, admission/preemption, and accuracy.

## Key Design Decision: Block Table Encoding

The existing vLLM block table stores a BF16 physical block id. That cannot represent a demoted page after its BF16 block is returned to the BF16 pool.

The v0 implementation should encode INT4 pages as negative block ids in the worker-facing block table:

```text
BF16 block:  block_table entry = bf16_block_id >= 0
INT4 block:  block_table entry = -(int4_block_id + 1)
```

This keeps block tables as `int32` tensors and avoids introducing a second GPU block-table tensor in the first version. Attention materialization decodes the sign bit:

```text
if entry >= 0:
    copy BF16 cache[entry] into compact workspace
else:
    int4_id = -entry - 1
    dequantize INT4 cache[int4_id] into compact workspace
```

The scheduler/core side must not pass negative block ids to the original `BlockPool.free_blocks()`. Demoted pages need explicit metadata and an INT4 pool free path.

## File Structure

- Modify: `vllm/vllm/config/cache.py`
  - Replace `mixed_static` with a new experimental `reflex_int4` cache dtype.
- Modify: `vllm/vllm/utils/torch_utils.py`
  - Map `reflex_int4` to the BF16 model dtype for primary KV allocation, not `uint8`.
- Modify: `vllm/vllm/v1/kv_cache_interface.py`
  - Do not use INT4 page size for `reflex_int4` primary KV. Primary vLLM KV pages remain BF16-sized.
- Create: `vllm/vllm/v1/core/reflex_int4.py`
  - CPU-side metadata, INT4 free-list allocator, distance demotion planner, and block-table encoding helpers.
- Modify: `vllm/vllm/v1/core/sched/output.py`
  - Add `reflex_int4_demotions` and `reflex_int4_freed_int4_blocks` fields if needed for worker actions.
- Modify: `vllm/vllm/v1/core/block_pool.py`
  - Add a safe method for releasing selected BF16 blocks after demotion without passing demoted INT4 pages through the normal free path.
- Modify: `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
  - Maintain logical page metadata per request.
  - Plan distance-based demotions when allocation pressure appears.
  - Replace demoted request block-table entries with encoded INT4 entries.
  - Free INT4 blocks on request finish.
- Modify: `vllm/vllm/v1/core/kv_cache_manager.py`
  - Retry allocation after synchronous core-side demotion planning.
  - Expose planned demotions to scheduler output.
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`
  - Place demotion plans in `SchedulerOutput`.
  - Ensure demotion execution is ordered before zeroing/reusing newly allocated BF16 blocks.
- Modify: `vllm/vllm/v1/worker/gpu_model_runner.py`
  - Allocate per-layer INT4 pools for `reflex_int4`.
  - Execute planned BF16-to-INT4 demotions before zeroing reused BF16 blocks.
  - Pass INT4 pool state to attention backend.
- Create: `vllm/vllm/v1/attention/ops/reflex_int4_kv_cache.py`
  - Reuse packed INT4 kernels and add mixed BF16/INT4 compact materialization helpers.
- Modify: `vllm/vllm/v1/attention/backends/triton_attn.py`
  - Support `reflex_int4`.
  - Decode signed block-table entries into compact BF16 workspace.
  - Remove `mixed_static` state/path after `reflex_int4` is working.
- Update tests:
  - Create: `tests/profiling/test_reflex_int4_pool.py`
  - Create: `tests/profiling/test_reflex_int4_block_table.py`
  - Create: `tests/profiling/test_reflex_int4_materialize.py`
  - Update/remove: `tests/profiling/test_mixed_static_kv_cache.py`
  - Update: `tests/accuracy/test_kv_accuracy_scripts.py`

---

### Task 1: CPU Metadata And INT4 Pool

**Files:**
- Create: `vllm/vllm/v1/core/reflex_int4.py`
- Test: `tests/profiling/test_reflex_int4_pool.py`

- [ ] **Step 1: Write failing tests**

Test these behaviors:

```python
pool = Int4BlockPool(num_blocks=3)
assert pool.allocate() == 0
assert pool.allocate() == 1
pool.free(0)
assert pool.allocate() == 0
```

Also test block-table encoding:

```python
assert encode_bf16_block_id(7) == 7
assert encode_int4_block_id(2) == -3
assert decode_block_table_entry(-3) == ("int4", 2)
```

- [ ] **Step 2: Run tests and confirm failure**

Run:

```bash
.venv/bin/python -m pytest tests/profiling/test_reflex_int4_pool.py -q
```

Expected: import or symbol failure.

- [ ] **Step 3: Implement metadata primitives**

Add:

```python
class PrecisionState(str, Enum):
    BF16 = "bf16"
    INT4 = "int4"

@dataclass
class ReflexPageMeta:
    request_id: str
    page_idx: int
    precision: PrecisionState
    bf16_block_id: int | None
    int4_block_id: int | None
    is_full: bool = True
    is_shared: bool = False

class Int4BlockPool:
    ...
```

- [ ] **Step 4: Run tests and confirm pass**

Run:

```bash
.venv/bin/python -m pytest tests/profiling/test_reflex_int4_pool.py -q
```

Expected: pass.

### Task 2: Distance-Based Demotion Planner

**Files:**
- Modify: `vllm/vllm/v1/core/reflex_int4.py`
- Test: `tests/profiling/test_reflex_int4_pool.py`

- [ ] **Step 1: Write failing tests**

Test that the planner demotes earliest full pages first, skips shared pages, and preserves recent tail pages:

```python
planner = DistanceDemotionPlanner(keep_recent_pages=2)
plan = planner.plan(request_pages, target_bf16_blocks=3, int4_pool=pool)
assert [item.page_idx for item in plan.items] == [0, 1, 2]
```

- [ ] **Step 2: Implement planner**

Planner inputs:

```python
request_pages: Mapping[str, Sequence[ReflexPageMeta]]
target_bf16_blocks: int
int4_pool: Int4BlockPool
```

Planner output:

```python
@dataclass
class ReflexDemotion:
    request_id: str
    page_idx: int
    bf16_block_id: int
    int4_block_id: int
    encoded_block_table_id: int
```

- [ ] **Step 3: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/profiling/test_reflex_int4_pool.py -q
```

Expected: pass.

### Task 3: Wire Scheduler-Side Planning Without GPU Execution

**Files:**
- Modify: `vllm/vllm/v1/core/sched/output.py`
- Modify: `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- Modify: `vllm/vllm/v1/core/kv_cache_manager.py`
- Modify: `vllm/vllm/v1/core/sched/scheduler.py`
- Test: `tests/profiling/test_reflex_int4_block_table.py`

- [ ] **Step 1: Add tests around encoded block ids**

Use a small manager-level fixture if possible. Otherwise instantiate `ReflexPageMeta` and assert that demoted pages produce negative worker-facing block-table entries while new BF16 pages remain non-negative.

- [ ] **Step 2: Add demotion plan fields to scheduler output**

Add:

```python
reflex_int4_demotions: list[ReflexDemotion] | None = None
```

- [ ] **Step 3: Add manager hooks**

Add methods:

```python
KVCacheManager.plan_reflex_int4_demotions(target_bf16_blocks: int) -> int
KVCacheManager.take_reflex_int4_demotions() -> list[ReflexDemotion]
```

- [ ] **Step 4: Retry allocation after planning**

When `allocate_slots()` returns `None`, attempt distance-based demotion before preemption. The first version can require `cache_dtype == "reflex_int4"` and no prefix cache.

- [ ] **Step 5: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/profiling/test_reflex_int4_block_table.py -q
```

Expected: pass.

### Task 4: GPU INT4 Pool And Demotion Execution

**Files:**
- Modify: `vllm/vllm/v1/worker/gpu_model_runner.py`
- Create: `vllm/vllm/v1/attention/ops/reflex_int4_kv_cache.py`
- Test: `tests/profiling/test_reflex_int4_materialize.py`

- [ ] **Step 1: Write materialization tests**

Create a small BF16 cache and INT4 cache. Quantize one BF16 block into INT4, materialize a block table with mixed positive and negative ids, and assert approximate equality for the INT4-dequantized block.

- [ ] **Step 2: Allocate per-layer INT4 cache**

For `reflex_int4`, allocate packed INT4 tensor per attention layer:

```text
(num_int4_blocks, 2, block_size, num_kv_heads, packed_head_size)
```

Capacity is fixed for v0. Use a conservative ratio and expose it as a constant first; CLI wiring can come later.

- [ ] **Step 3: Execute demotion before zeroing new BF16 blocks**

In `_update_states()`, run demotion plans before `_zero_block_ids()`. This preserves the old BF16 source content even if the block is reused in the same scheduling step.

- [ ] **Step 4: Run tests**

Run:

```bash
.venv/bin/python -m pytest tests/profiling/test_reflex_int4_materialize.py -q
```

Expected: pass.

### Task 5: Attention Backend Integration

**Files:**
- Modify: `vllm/vllm/config/cache.py`
- Modify: `vllm/vllm/utils/torch_utils.py`
- Modify: `vllm/vllm/v1/kv_cache_interface.py`
- Modify: `vllm/vllm/v1/attention/backends/triton_attn.py`
- Modify/remove: `vllm/vllm/v1/attention/ops/mixed_static_kv_cache.py`
- Update tests under `tests/profiling/`

- [ ] **Step 1: Register `reflex_int4`**

Add the dtype and warning. Do not make `reflex_int4` primary cache `uint8`; primary KV remains BF16.

- [ ] **Step 2: Add Triton backend support**

For `reflex_int4`, materialize compact BF16 KV cache from signed block table entries and pass the remapped positive compact table into existing attention.

- [ ] **Step 3: Remove `mixed_static` option**

Remove `mixed_static` from config, dtype mapping, backend support, and accuracy/profiling tests. Keep shared INT4 kernels.

- [ ] **Step 4: Run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/profiling/test_int4_kv_cache.py tests/profiling/test_reflex_int4_pool.py tests/profiling/test_reflex_int4_block_table.py tests/profiling/test_reflex_int4_materialize.py -q
```

Expected: pass.

### Task 6: First Functional Smoke

**Files:**
- No new files unless a script needs a `reflex_int4` variant.

- [ ] **Step 1: Run a tiny offline generation**

Use a small synthetic prompt and force Triton/eager:

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m vllm.entrypoints.openai.api_server ...
```

or the existing local benchmark wrapper once the CLI path is updated.

- [ ] **Step 2: Verify runtime behavior**

Check logs for:

```text
reflex_int4 demoted_pages > 0
bf16_blocks_released > 0
int4_blocks_used > 0
```

- [ ] **Step 3: Run a tiny accuracy check**

Run a small Qasper/Math subset and compare to BF16. Do not treat this as final quality evidence.

### Task 7: Serving Pressure Evaluation

**Files:**
- Update existing serving benchmark wrapper or add a new one under `scripts/` only if necessary.

- [ ] **Step 1: Compare BF16 vs dynamic INT4**

Sweep:

```text
request rate
concurrency
prompt length
output length
burst size
```

- [ ] **Step 2: Report serving metrics**

Report:

```text
max sustainable request rate
waiting time p50/p95/p99
TTFT p50/p95/p99
TPOT p50/p95/p99
admission/preemption/rejection
goodput under a fixed SLO
accuracy drop
```

This is the evidence for ReFlexKV; BF16 free-block count is only an internal control signal.
