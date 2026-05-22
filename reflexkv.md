# ReFlexKV 技术规划文档：从动态 KV 降精度到弹性精度 KV 内存系统

## 1. 项目定位

ReFlexKV 不应被定义为一个简单的 KV cache INT4 量化系统，而应被定义为：

> **面向 LLM Serving 的弹性精度 KV 内存管理系统。**

核心思想是：在 LLM serving 中，KV cache 不应只是被分配、释放、共享的 GPU 内存块，而应成为具有 **precision state、memory tier、risk level、sharing state、SLO ownership** 的运行时资源。ReFlexKV 的目标是让系统能够根据 serving pressure、请求优先级、上下文风险和运行时负载，动态调整 KV page 的精度和存放位置，从而提升 admission capacity、tail latency、goodput 和系统稳定性，同时控制模型质量损失。

现有 vLLM/PagedAttention 已经将 KV cache 做成分页式管理对象，通过 block-level memory management 降低碎片并支持跨请求共享；DistServe 将 prefill 和 decode 分离，以降低两阶段干扰并分别优化 TTFT/TPOT；Mooncake 进一步将 KVCache 作为 disaggregated serving 的核心资源，围绕 KVCache 进行调度和缓存管理；CacheGen 则面向 KV cache 传输和加载，通过压缩与 streaming 降低上下文加载延迟。ReFlexKV 应在这些系统工作之上推进一步：**不只管理 KV 的空间、位置和复用，还管理 KV 的精度状态。** ([arXiv][1])

---

## 2. 核心技术命题

当前 LLM serving 系统通常把 KV cache 视为一种空间资源：

```text
allocate KV block
free KV block
share KV block
offload KV block
transfer KV block
```

ReFlexKV 需要提出新的系统命题：

```text
KV cache precision is a first-class serving resource.
```

也就是说，一个 KV page 的运行时状态不应只是：

```text
logical page -> physical block id
```

而应扩展为：

```text
logical page
    -> precision state
    -> memory tier
    -> physical location
    -> risk score
    -> sharing state
    -> recovery state
    -> request/SLO ownership
```

最终 ReFlexKV 要解决的问题不是：

```text
一个 KV page 应该量化到几 bit？
```

而是：

```text
在当前 serving 状态下，一个 KV page 应该处于什么 precision、tier 和 lifecycle state？
```

---

## 3. 系统总目标

ReFlexKV 最终应支持以下能力：

1. **在线精度弹性管理**
   在 serving 过程中动态决定 KV page 保持 BF16、降级到 INT4、转移到 CPU shadow，或被重新提升到高精度。

2. **请求级精度预算控制**
   不同请求根据 SLO priority、decode progress、prompt risk、remaining decode、prefix sharing 和 quality debt 承担不同程度的降精度压力。

3. **页面级风险感知迁移**
   对每个 page 维护 risk score，优先迁移 sealed、cold、low-risk、non-shared pages，保护 recent pages、evidence-rich pages、shared prefix pages 和高优先级请求依赖的 pages。

4. **P/D 协同精度规划**
   P 侧不只是生成 BF16 KV，也应生成 lightweight page risk hints；D 侧根据自身 landing capacity 和 serving pressure 决定 KV 的安装精度。

5. **支持恢复和提升**
   降精度不应永远不可逆。系统需要支持 precision fault 检测，并在必要时将低精度 page 恢复或提升到高精度。

6. **prefix cache 精度一致性管理**
   对共享 prefix block，系统需要支持 shared precision contract、copy-on-demote 和 multi-version prefix cache，避免单个请求的降精度行为破坏其他请求。

---

## 4. 核心抽象：Precision-Elastic KV Virtual Memory

### 4.1 Precision-aware KV Page Table

当前实现中使用负数 block id 表示 INT4 block，这是一个可用的原型方法，但最终系统需要抽象成显式 page descriptor。

建议定义：

```cpp
struct KVPageRef {
    RequestId request_id;
    LogicalPageId logical_page_id;
    LayerGroupId layer_group_id;

    Precision precision;      // BF16, FP8, INT8, INT4
    MemoryTier tier;          // GPU, CPU, SSD
    PhysicalBlockId block_id;

    PageState state;          // OPEN, SEALED_HOT, SEALED_COLD, INT4_ACTIVE, RECOVERING, ...
    Version version;

    float risk_score;
    float quality_debt;
    int refcount;
    PrefixHash prefix_hash;

    StepId last_access_step;
    StepId last_migration_step;

    RecoveryHandle recovery_handle;
};
```

block table 在 kernel 侧可以继续压缩成紧凑格式，但系统内部应使用显式 page metadata。

目标是把 ReFlexKV 从：

```text
INT4 block pool patch
```

升级为：

```text
precision-aware KV virtual memory layer
```

---

## 5. KV Page Lifecycle

每个 KV page 需要有完整状态机。

### 5.1 基础状态

```text
OPEN_BF16
    当前正在写入的 page，不能被迁移或降精度。

SEALED_BF16_HOT
    已写满，但 recent / high-risk / evidence-rich，需要保护。

SEALED_BF16_WARM
    已写满，暂时观察，不立即迁移。

SEALED_BF16_COLD
    已写满，低风险，可作为 demotion candidate。

GPU_INT4_ACTIVE
    已降级到 GPU INT4 pool，仍参与 decode attention。

CPU_FP8_SHADOW
    高于 INT4 的恢复副本，存放在 CPU，用于后续 recovery。

CPU_COMPRESSED_COLD
    长期低访问 page，可进一步压缩或 offload。

PINNED_SHARED
    prefix-cache/shared page，被多个请求引用，不能被单个请求直接原地降精度。

RECOVERING
    正在从低精度或 CPU shadow 恢复到高精度。

PROMOTED_BF16
    已被重新提升到 BF16，重新进入高精度 working set。
```

### 5.2 状态转移

```text
OPEN_BF16
    -> SEALED_BF16_HOT
    -> SEALED_BF16_WARM
    -> SEALED_BF16_COLD
    -> GPU_INT4_ACTIVE
    -> CPU_FP8_SHADOW / CPU_COMPRESSED_COLD
```

```text
GPU_INT4_ACTIVE
    -> PRECISION_FAULT
    -> RECOVERING
    -> PROMOTED_BF16
```

```text
SEALED_BF16_HOT / SEALED_BF16_COLD
    -> PINNED_SHARED
    -> COPY_ON_DEMOTE
```

这个状态机是 ReFlexKV 的核心系统机制。它让 KV page 从静态内存块变成可迁移、可降级、可恢复的运行时对象。

---

## 6. Precision Working Set

ReFlexKV 需要引入 **precision working set** 概念。

传统 working set 关注哪些 page 需要留在内存中。ReFlexKV 的 precision working set 关注：

```text
哪些 KV pages 需要保持高精度？
哪些 KV pages 可以低精度参与 decode？
哪些 KV pages 可以转入 CPU recovery tier？
哪些 KV pages 需要在未来被恢复？
```

定义：

```text
Precision Working Set = 当前或未来一段 decode window 内，对输出质量敏感、需要高精度保留的 KV pages。
```

precision working set 的来源包括：

```text
recent decode pages
high-attention prompt pages
question/evidence-related pages
shared prefix pages with high-priority consumers
pages used by requests with low SLO slack
pages with high precision fault probability
```

低精度 backing set 包括：

```text
old decode pages
low-attention prompt pages
low-risk context pages
low-priority request pages
cold pages with CPU recovery copy
```

---

## 7. Precision Fault

ReFlexKV 需要支持 **precision fault**。

precision fault 不是缺页，而是：

```text
某个低精度 page 重新变得重要，当前 precision 不再满足质量需求。
```

触发条件可以包括：

```text
attention probe 显示 INT4 page attention mass 升高
query-page relevance 超过阈值
page 所在 evidence window 被重新访问
请求进入低 SLO slack 阶段
质量风险累积超过 request budget
```

触发后：

```text
GPU_INT4_ACTIVE
    -> RECOVERING
    -> PROMOTED_BF16 / PROMOTED_FP8
```

precision fault 是 ReFlexKV 从“只会压缩”的系统升级为“闭环精度管理系统”的关键。

---

## 8. Active Precision 与 Recovery Precision 分离

ReFlexKV 应区分三种精度：

```text
active precision
    decode 当前实际读取的 GPU copy 精度。

recovery precision
    系统保留的可恢复副本精度。

storage precision
    CPU/SSD 上长期保存的压缩副本精度。
```

例如：

```text
active copy: GPU INT4
recovery copy: CPU FP8
storage copy: CPU/SSD compressed
```

这样系统可以实现：

```text
低延迟路径使用低精度 active copy
质量敏感时从 recovery copy 提升
长期不用时转入 storage copy
```

这个设计避免 ReFlexKV 与纯量化算法正面竞争。ReFlexKV 的核心不是提出最好的 INT4 quantizer，而是管理多个 precision copies 在 serving runtime 中的生命周期。

---

## 9. P/D 协同精度规划

当前版本中，P 侧保持 BF16 prefill 和 BF16 handoff，D 侧在 request admission 后执行 demotion。这个设计适合作为第一阶段原型，但最终系统需要支持 P/D 协同。

### 9.1 P 侧职责

P 侧应负责：

```text
执行 BF16 prefill
生成 page-level risk hints
生成 page summary
识别 prompt structure
标记 question/evidence/tail pages
标记 prefix-shareable pages
根据 D 侧 landing contract 生成 mixed-precision handoff plan
```

P 侧不一定需要主动压缩所有 KV，但至少应生成轻量 metadata：

```text
page_attention_mass
page_key_summary
tail_query_relevance
page_position_type
compressibility_hint
prefix_hash
refcount_hint
```

### 9.2 D 侧职责

D 侧负责：

```text
维护 BF16/INT4/CPU KV pools
根据 serving pressure 决定 landing precision
安装 mixed-precision KV pages
执行 decode-side demotion/promotion
维护 precision-aware page table
处理 precision fault
```

### 9.3 Landing Contract

D 侧可以向 P 侧提供 landing contract：

```text
当前 D 侧 BF16 landing capacity
当前 INT4 pool capacity
CPU recovery tier capacity
请求 SLO class
expected decode pressure
```

P 侧根据 contract 决定：

```text
哪些 pages 必须 BF16 landing
哪些 pages 可以 INT4 landing
哪些 pages 只需要 CPU recovery copy
哪些 pages 暂时不进入 GPU
```

这样可以解决当前版本的一个关键限制：

```text
waiting request admission 前必须完整分配 BF16 external KV。
```

最终目标是：

```text
incoming request 不再必须完整 BF16 landing；
而是可以根据风险和 D 侧容量进行 mixed-precision landing。
```

---

## 10. Prefix Cache 精度一致性

prefix cache 是 serving 场景中的核心机制，ReFlexKV 必须把它纳入一等设计对象。

### 10.1 Shared Precision Contract

对于共享 prefix page：

```text
shared_page_precision = max(required_precision of all active consumers)
```

如果一个高优先级请求仍然需要 BF16，则共享 page 不能被低优先级请求原地 demote。

### 10.2 Copy-on-Demote

当不同请求对同一个 shared page 的精度需求不一致时：

```text
shared BF16 page
    ├── high-priority request keeps BF16 reference
    └── low-priority request gets private INT4 copy
```

这类似操作系统中的 copy-on-write，但触发原因是 precision mismatch，因此可称为：

```text
copy-on-demote
```

### 10.3 Multi-version Prefix Cache

同一个 prefix hash 可以对应多个 precision version：

```text
prefix_hash H:
    BF16 version
    FP8 version
    INT4 version
    CPU compressed version
```

prefix cache lookup 应扩展为：

```text
lookup(prefix_hash, required_precision, SLO_class)
```

这使 ReFlexKV 能够同时支持共享、降精度和高优先级保护。

---

## 11. 控制器设计

ReFlexKV 的 controller 不应只是 heuristic rule collection，而应设计为 closed-loop precision controller。

### 11.1 输入

```text
global BF16 pool usage
global INT4 pool usage
CPU recovery tier usage
migration bandwidth usage
waiting queue length
admission failures
TTFT / TPOT / p95 / p99 latency
per-request SLO slack
per-request priority
per-request generated tokens
per-request remaining decode estimate
per-request quality debt
per-page risk score
per-page sharing refcount
per-page state
```

### 11.2 输出

```text
demote BF16 -> INT4
promote INT4 -> BF16
offload BF16/INT4 -> CPU shadow
prefetch CPU shadow -> GPU
copy-on-demote shared page
pin high-risk page
adjust request-level precision budget
adjust page-level risk threshold
```

### 11.3 优化目标

控制器的目标不是最大化压缩率，而是：

```text
maximize goodput under SLO
minimize p95/p99 latency
minimize admission failure
minimize quality degradation
minimize migration overhead
maintain fairness across requests
```

### 11.4 Quality Debt

每个 request 维护 quality debt：

```text
quality_debt(request) =
    sum risk(page) over demoted pages
    + recovery delay penalty
    + precision fault penalty
```

调度时避免不断压同一个请求：

```text
score(page) =
    memory_saving /
    (quality_risk
     + migration_cost
     + promotion_cost
     + sharing_cost
     + request_quality_debt
     + SLO_risk)
```

---

## 12. Feasibility-aware Planning

当前版本中存在 target release 与 actual release 差距过大的问题。最终系统必须先估计可行释放空间，再做 release target 分配。

### 12.1 Feasible Release Frontier

对每个 request 计算：

```text
feasible_release_pages(request) =
    pages satisfying:
        sealed
        non-open
        non-recent
        non-high-risk
        non-pinned-shared
        within quality budget
        within sparse quota
        migration bandwidth available
        INT4/CPU target pool available
```

控制器只在 feasible frontier 上分配 budget：

```text
global_needed_release = admission_required - free_BF16_blocks

actual_target =
    min(global_needed_release, sum(feasible_release_pages))
```

### 12.2 Rejection Reason Accounting

每次不能 demote 的 page 都记录原因：

```text
recent protected
initial protected
not sealed
shared prefix pinned
high prompt risk
window quota exceeded
request quality debt exceeded
SLO slack too low
INT4 pool full
migration bandwidth saturated
CPU shadow unavailable
```

这个 accounting 应作为系统观测和论文分析的重要部分。

---

## 13. Page-level Risk Estimator

page-level risk estimator 应从简单 mask 升级为多信号融合。

### 13.1 Attention-derived Risk

P 侧 prefill 后，使用最后若干 query positions 对 prompt pages 的 attention mass 聚合：

```text
risk_attn(page) =
    sum attention_mass from tail/query positions to page
```

只保留 page-level scalar，不保存完整 attention matrix。

### 13.2 Semantic Relevance Risk

使用 page key summary 和 query summary：

```text
page_summary = mean(K_page)
query_summary = mean(Q_tail)
risk_semantic = query_summary · page_summary
```

### 13.3 Structural Risk

根据 prompt 结构保护：

```text
instruction pages: high risk
question/tail pages: high risk
retrieval evidence pages: high risk
prefix shared pages: special handling
middle low-attention context pages: lower risk
old decode pages: lower risk after cooling
```

### 13.4 Combined Risk

```text
page_risk =
    α * risk_attn
  + β * risk_semantic
  + γ * risk_structural
  + δ * risk_recency
  + η * risk_sharing
```

该 estimator 必须满足：

```text
low overhead
page-level only
compatible with P/D handoff
robust across QA, summarization, reasoning, code
```

---

## 14. Risk-aware Sparse Window

当前 sparse window quota 应升级为 risk-aware sparse window。

### 14.1 Window Risk

将连续 pages 分成 windows：

```text
window_risk = aggregate(page_risk within window)
```

根据风险调整 quota：

```text
low-risk window:
    high demotion quota

medium-risk window:
    moderate demotion quota

high-risk evidence window:
    low or zero demotion quota
```

### 14.2 Neighbor Protection

如果某 page 是 high-risk：

```text
protect page i-k ... i ... i+k
```

因为 evidence span 很可能跨越多个 16-token pages。

### 14.3 Evidence Continuity Protection

避免连续破坏证据段：

```text
never demote too many adjacent pages
never demote all pages in one evidence window
prefer sparse demotion across multiple low-risk regions
```

---

## 15. Migration Engine

ReFlexKV 需要一个独立 migration engine，而不是把迁移逻辑散在 scheduler、cache manager 和 GPU runner 中。

### 15.1 Task Types

```text
DemoteTask:
    BF16 GPU page -> INT4 GPU page

PromoteTask:
    INT4 GPU page -> BF16/FP8 GPU page

OffloadTask:
    GPU page -> CPU shadow

PrefetchTask:
    CPU shadow -> GPU page

RecoverTask:
    CPU recovery copy -> GPU high-precision page

CopyOnDemoteTask:
    shared BF16 page -> private INT4 page
```

### 15.2 Task States

```text
PENDING
RUNNING
PATCHING
COMMITTED
FAILED
ROLLED_BACK
```

### 15.3 Commit Protocol

为了保证 decode 正确性，迁移必须遵守：

```text
1. allocate target block
2. copy / quantize / recover data
3. validate task completion
4. atomically patch page table
5. release old block if no longer referenced
6. update request/page statistics
```

block table patch 之前，decode 仍读旧 page；patch 之后，decode 读新 page。

---

## 16. Correctness Invariants

系统需要明确不变量。

```text
Invariant 1:
    OPEN pages cannot be demoted or offloaded.

Invariant 2:
    Block table update must be atomic with respect to decode execution.

Invariant 3:
    Shared pages cannot be destructively demoted unless all consumers agree.

Invariant 4:
    Copy-on-demote must preserve the original shared BF16 page.

Invariant 5:
    A page cannot be freed until no active page table entry references it.

Invariant 6:
    Recovery copy version must match the active page version.

Invariant 7:
    Precision fault recovery must not violate request SLO slack unless explicitly allowed in critical mode.

Invariant 8:
    The controller cannot allocate release budget beyond feasible demotion frontier.

Invariant 9:
    Quality debt must be monotonically tracked and bounded per request.

Invariant 10:
    Recent and high-risk pages remain protected unless system enters explicitly defined emergency mode.
```

这些不变量应作为系统设计和代码实现的基础。

---

## 17. 系统架构

最终 ReFlexKV 应分为三层。

### 17.1 Precision-aware KV Virtual Memory Layer

负责：

```text
logical page table
precision state machine
multi-tier page references
versioning
shared prefix metadata
copy-on-demote
quality debt bookkeeping
recovery handle management
```

### 17.2 Migration and Execution Data Plane

负责：

```text
BF16 -> INT4 GPU demotion
INT4 -> BF16/FP8 promotion
GPU <-> CPU shadow movement
mixed precision attention
asynchronous migration queue
migration bandwidth throttling
block table patching
kernel-visible compact page table
```

### 17.3 SLO-aware Precision Controller

负责：

```text
admission-aware planning
pressure prediction
request-level quality budget
page-level risk scoring
precision working set estimation
precision fault handling
fairness / quality debt control
prefix sharing policy
```

---

## 18. 当前代码的演进方向

### 18.1 当前模块

```text
vllm/v1/core/reflex_int4.py
    当前策略、INT4 pool、budget、planner。

vllm/v1/core/sched/scheduler.py
    当前 pressure 判断、admission-triggered demotion。

vllm/v1/core/single_type_kv_cache_manager.py
    当前 block table 和 page metadata。

vllm/v1/worker/gpu_model_runner.py
    当前 BF16 -> INT4 quantization 执行。

vllm/distributed/kv_transfer/kv_connector/v1/reflex_mooncake_connector.py
    当前 Mooncake connector compatibility layer。
```

### 18.2 目标模块重构

建议新增：

```text
vllm/v1/core/precision_kv/page_table.py
    Precision-aware KV page table。

vllm/v1/core/precision_kv/state_machine.py
    Page lifecycle and transition rules。

vllm/v1/core/precision_kv/controller.py
    SLO-aware precision controller。

vllm/v1/core/precision_kv/risk_estimator.py
    Page-level risk scoring。

vllm/v1/core/precision_kv/migration_planner.py
    Feasibility-aware transition planner。

vllm/v1/core/precision_kv/migration_engine.py
    Async migration task engine。

vllm/v1/core/precision_kv/prefix_coherence.py
    Shared prefix precision contract and copy-on-demote。

vllm/v1/core/precision_kv/shadow_store.py
    CPU recovery copy management。

vllm/v1/worker/precision_kv/kernels/
    Mixed precision attention and quantization kernels。

vllm/distributed/kv_transfer/kv_connector/v1/reflex_precision_connector.py
    P/D cooperative precision landing。
```

---

## 19. 实验规划

### 19.1 End-to-end Serving Evaluation

对比对象：

```text
BF16 serving baseline
static INT4 KV
naive old-page-first demotion
D-side ReFlexKV demotion
ReFlexKV with recovery
ReFlexKV with P/D cooperative landing
ReFlexKV full system
```

核心指标：

```text
request throughput
goodput under SLO
p50/p95/p99 TTFT
p50/p95/p99 TPOT
end-to-end latency
queue waiting time
admission failure rate
OOM / rejection rate
BF16 pool occupancy
INT4 pool occupancy
CPU shadow usage
migration overhead
quality degradation
```

### 19.2 Pressure Scaling

测试不同压力：

```text
low load
medium load
high load
overload
bursty arrival
mixed priority workload
mixed prompt length workload
mixed output length workload
high prefix sharing workload
```

目标观察：

```text
低压力下 ReFlexKV 接近 BF16
中压力下 ReFlexKV 降低等待和 tail latency
高压力下 ReFlexKV 避免 admission collapse
过载下 ReFlexKV 优雅降级而非直接 OOM
```

### 19.3 Quality Evaluation

任务类型：

```text
long-document QA
multi-hop QA
summarization
few-shot learning
synthetic retrieval
reasoning
code
chat workload
```

需要分析：

```text
quality vs load
quality vs INT4 ratio
quality vs recovery frequency
quality vs page risk policy
quality vs sparse window quota
quality vs precision fault threshold
```

### 19.4 Component Ablation

必须拆解：

```text
without request-level budget
without feasibility-aware planning
without page risk estimator
without risk-aware sparse window
without precision fault recovery
without CPU recovery shadow
without prefix coherence
without copy-on-demote
without quality debt
without admission-aware landing
```

### 19.5 Trace-driven Evaluation

真实两卡系统用于验证机制、kernel overhead 和质量影响；trace-driven simulator 用于扩展更大规模 serving 场景。

模拟变量：

```text
number of P workers
number of D workers
arrival rate
prompt length distribution
output length distribution
prefix sharing ratio
SLO class distribution
GPU memory capacity
CPU recovery bandwidth
migration bandwidth
```

输出指标：

```text
cluster-level goodput
SLO violation rate
admission success rate
queue buildup
precision migration frequency
BF16/INT4/CPU tier occupancy
quality debt distribution
```

---

## 20. 关键图表规划

### Figure 1：Problem Overview

展示静态 BF16、静态 INT4、普通 KV compression 和 ReFlexKV 的区别：

```text
BF16:
    high quality, high memory pressure, admission blocked

static INT4:
    lower memory, uncontrolled quality loss

offline compression:
    not serving-aware

ReFlexKV:
    precision changes online according to serving pressure and page risk
```

### Figure 2：Precision-elastic KV Virtual Memory

展示：

```text
logical KV page table
    -> precision state
    -> GPU BF16 pool
    -> GPU INT4 pool
    -> CPU recovery tier
    -> mixed precision attention
```

### Figure 3：KV Page Lifecycle

展示完整状态机：

```text
OPEN_BF16
    -> SEALED_HOT
    -> SEALED_COLD
    -> GPU_INT4_ACTIVE
    -> CPU_FP8_SHADOW
    -> RECOVERING
    -> PROMOTED_BF16
```

### Figure 4：P/D Cooperative Landing

展示：

```text
P side prefill
    -> page risk hints
    -> landing contract
    -> mixed precision KV package
    -> D side install
```

### Figure 5：Prefix Cache Precision Coherence

展示：

```text
shared BF16 prefix page
    -> shared precision contract
    -> copy-on-demote
    -> multi-version prefix cache
```

### Figure 6：Precision Fault and Recovery

展示 decode 过程中某 INT4 page 重新变重要，然后触发 recovery。

### Figure 7：Serving Timeline Trace

展示：

```text
BF16 occupancy
INT4 occupancy
waiting queue
demotion events
promotion events
admission events
p99 TPOT
quality debt
```

---

## 21. 研发路线

### Phase 1：稳固当前 D-side ReFlexKV

目标：

```text
BF16/INT4 mixed precision decode 稳定
feasibility-aware controller
admission-aware pre-release
risk-aware sparse demotion
完整 rejection reason accounting
完整 timeline trace
```

产出：

```text
可运行 D-side dynamic demotion system
可解释 target/feasible/actual release
质量下降可控
压力下有 clear serving benefit
```

### Phase 2：重构 Precision-aware Page Table

目标：

```text
显式 KVPageRef
precision state machine
page lifecycle tracking
versioning
shared page metadata
quality debt tracking
```

产出：

```text
从 block id hack 升级到 precision virtual memory abstraction
```

### Phase 3：实现 Migration Engine

目标：

```text
DemoteTask
PromoteTask
OffloadTask
PrefetchTask
RecoverTask
CopyOnDemoteTask
atomic page table patch
async migration queue
```

产出：

```text
安全、异步、可观测的 precision migration data plane
```

### Phase 4：实现 Recovery Path

目标：

```text
GPU INT4 active copy
CPU FP8/BF16 recovery copy
precision fault detection
async promotion
recovery overhead measurement
```

产出：

```text
从 irreversible demotion 升级为 closed-loop precision management
```

### Phase 5：实现 P/D Cooperative Landing

目标：

```text
P side page risk hints
D side landing contract
mixed precision landing
admission-aware installation
```

产出：

```text
解决 waiting request 必须完整 BF16 landing 的限制
```

### Phase 6：实现 Prefix Cache Precision Coherence

目标：

```text
shared precision contract
copy-on-demote
multi-version prefix cache
refcount-aware precision policy
```

产出：

```text
支持真实 serving 中的 prefix sharing 与 mixed precision 共存
```

### Phase 7：完整评测与论文收敛

目标：

```text
真实系统评测
trace-driven 扩展评测
pressure scaling
quality-risk analysis
component ablation
case study
timeline visualization
```

产出：

```text
完整系统论文证据链
```

### 2026-05-18 当前实现状态与下一步

已经完成的下一步机制：

```text
precision_kv/landing.py
    PrecisionLandingPlanner
    PrecisionLandingState
    PrecisionLandingDecision

scheduler admission trace
    running feasible release frontier
    request-local INT4 landing frontier
    mixed landing feasibility
    residual BF16 deficit after running demotion
```

这一步的意义不是已经完成 mixed landing 数据路径，而是把 admission failure 从单一的
`not enough BF16 blocks` 拆成两个可验证命题：

```text
1. running requests 的 demotion frontier 是否足够？
2. waiting request 自身的 low-risk prompt pages 是否足以低精度 landing？
```

最新 1P1D smoke 结果：

```text
run_dir:
outputs/accuracy/pd_reflex_landing_frontier_smoke_2026-05-18/
20260518-190938_pdserv_mixed_longbench+reasoning_kv-reflex_int4_c8_ln2_mn2_rinf

total requests: 8
completed: 8/8
decode max KV usage: 95.75%
decode max waiting: 2

positive admission-pressure events: 3794
admission infeasible under current BF16 landing path: 3794
running feasible release total: 26766 blocks
admission requested release total: 699680 blocks

mixed landing feasible events: 3276 / 3794
required INT4 landing blocks total: 672914
eligible INT4 landing blocks total: 944467
planned INT4 landing blocks total: 507792
```

结论：

```text
继续调 decode-side demotion policy 不是主路径。
当前 admission collapse 的主要原因是 D 侧仍要求 external KV 完整 BF16 landing。
下一步必须实现 P/D cooperative mixed-precision landing 数据路径。
```

下一步实现边界：

```text
1. D side 在 allocate external computed KV blocks 时支持 mixed landing plan。
2. waiting request 的部分 low-risk prompt pages 直接进入 INT4 pool。
3. page table 需要能同时记录 BF16 external pages 与 INT4 external pages。
4. KV connector / worker 需要按 landing plan materialize BF16 或 INT4 copy。
5. admission gate 才能从 hypothetical landing_feasible 变成 real admission_success。
```

### 2026-05-18 进展：Landing Contract 传播

在 landing frontier 之后，已经完成 page-level landing contract 的控制面传播：

```text
PrecisionLandingDecision
    -> planned_int4_landing_pages

Scheduler
    -> request.kv_transfer_params["reflex_int4_landing_pages"]
    -> request.kv_transfer_params["reflex_int4_landing_required_blocks"]
    -> request.kv_transfer_params["reflex_int4_landing_planned_blocks"]
    -> request.kv_transfer_params["reflex_int4_landing_reason"]

Mooncake metadata
    -> PullReqMeta.reflex_int4_landing_pages
    -> MooncakeXferMetadata.reflex_int4_landing_pages
```

这一步仍然是安全的 control-plane 改动：不会放开 admission gate，也不会让 D 侧在没有
INT4 materialization 的情况下跳过 BF16 transfer。

最新 smoke：

```text
run_dir:
outputs/accuracy/pd_reflex_landing_contract_smoke_2026-05-18/
20260518-201233_pdserv_mixed_longbench+reasoning_kv-reflex_int4_c8_ln2_mn2_rinf

total requests: 8
completed: 8/8
duration: 74.47s
decode max KV usage: 98.65%
decode max waiting: 2

admission control events: 942
admission infeasible events: 886
mixed landing feasible events: 686
admission requested release total: 135478 blocks
running feasible release total: 3045 blocks
required INT4 landing total: 132433 blocks
planned INT4 landing total: 49055 blocks
```

下一刀必须进入 worker/data plane：

```text
1. D worker 接收 MooncakeXferMetadata 中的 reflex_int4_landing_pages。
2. 对 landing pages 建立 BF16 staging 或 P-side compressed transfer 路径。
3. 调用现有 int4_quantize_blocks_to_cache，把 landing pages materialize 到 INT4 sidecar。
4. 更新 block table，使这些 external pages 从一开始就是 INT4 entries。
5. 验证 admission gate 可安全从 landing contract 转为 real mixed landing。
```

### 2026-05-18 进展：Landing Data-plane Materialization 第一版

本轮把 mixed landing 从纯 scheduler contract 往 worker data-plane 推进了一步：

```text
Scheduler:
    为 planned landing pages 预留 INT4 physical block ids。
    将 reflex_int4_landing_pages / block_ids / planned_blocks 写入 kv_transfer_params。
    已签订的 landing contract 会保持到 KV transfer 阶段，避免被后续 bf16_fit
    或 insufficient frontier 估计擦掉。

Mooncake connector:
    在 MooncakeXferMetadata / PullReqMeta 中传播 landing pages 和 INT4 block ids。
    D worker 在所有 remote pull task 完成后，调用 int4_quantize_blocks_to_cache，
    把已落到本地 BF16 KV buffer 的指定 pages materialize 到 INT4 sidecar。

Model runner:
    reflex_int4 模式下把 INT4 sidecar cache 注册给 KV transfer group。

Observability:
    新增 ReFlexKV trace landing_materialize 事件，并在 mixed accuracy / pressure
    summary 中统计 event_count、pages、layer_copies、gpu_ms。
```

验证结果：

```text
CPU/单元测试：
    88 passed, 17 warnings
    覆盖 scheduler sticky contract、INT4 landing block reservation/free、
    Mooncake metadata propagation、worker materialization、summary parser。

GPU smoke:
    run_dir =
    outputs/accuracy/pd_reflex_forced_landing_materialize_smoke_2026-05-18/
    20260518-205623_pdserv_mixed_longbench+reasoning_kv-reflex_int4_c8_ln2_mn2_rinf

    total_requests = 8
    failed_predictions = 0
    duration_seconds = 68.5604
    landing_materialize_event_count = 1
    landing_materialized_pages_total = 43
    landing_materialize_layer_copies_total = 1376
    landing_materialize_gpu_ms_total = 1.697
    admission_mixed_landing_feasible_total = 315
```

当前边界也很明确：这一步只是把 landing contract 接到 data-plane sidecar
materialization；尚未把这些 external pages 写回 precision-aware block table，
也尚未释放对应 BF16 blocks。因此它已经能证明“D worker 可按 contract
把远端 KV 页变成 INT4 sidecar state”，但还不能宣称 mixed landing 已经带来
admission capacity gain。

下一步应把 materialized landing pages 提交为真实 runtime precision state：

```text
1. 在 KVCacheManager 中增加 commit_reflex_int4_landing_pages(request_id, pages, int4_ids)。
2. 将对应 block table entries 从 BF16 physical ids 切到 encoded INT4 ids。
3. 延迟或安全释放 landing pages 的 BF16 blocks。
4. 让 admission gate 在 landing contract 可执行时真正允许 request 进入。
5. 用 pressure sweep 证明 max_waiting / tail latency / goodput 的改善，而不只看 trace。
```

### 2026-05-19 进展：Landing Commit 进入真实 Runtime Precision State

本轮已经把 2026-05-18 的 landing materialization 从 sidecar-only trace
推进到真实 block table state transition：

```text
D scheduler landing contract
    -> Mooncake worker INT4 sidecar materialization
    -> worker metadata materialized signal
    -> scheduler gated commit
    -> KVCacheManager block table BF16 entry -> encoded INT4 entry
    -> BF16 page delayed release
```

已经完成的机制：

```text
KVCacheManager:
    commit_reflex_int4_landing_pages(request_id, pages, int4_ids)
    要求 landing INT4 block 已预留、page/id 无重复、不能覆盖已有不同 INT4 page。

Scheduler:
    landing commit 必须等待 worker materialized signal；
    未 materialize 的 contract 回退并释放 landing reservation；
    WAITING_FOR_REMOTE_KVS / remote-bound contract 被 demotion planner 保护；
    remote transfer 已绑定的 landing contract 不允许被后续 bf16_fit 或 mixed replan 覆盖。

Mooncake connector:
    worker 在所有 pull task 完成后 materialize landing pages；
    将 reflex_int4_materialized_landing_req_ids 汇总到 KVConnectorOutput；
    scheduler 只对已报告 materialized 的 request 执行 commit。

Observability:
    新增 ReFlexKV trace landing_commit；
    mixed accuracy / pressure summary 统计 landing_commit_event_count
    和 landing_committed_pages_total。
```

这一步修掉了两个关键一致性问题：

```text
1. 不能在 worker 未真正 materialize INT4 sidecar 时提交 block table。
2. remote KV transfer 进行中，scheduler 不能重新规划或清空已发送的 landing contract。
```

验证结果：

```text
CPU/单元测试：
    99 passed, 17 warnings

覆盖范围：
    scheduler landing commit / materialized gating / remote-wait sticky contract；
    remote-wait mixed replan immutability；
    KVCacheManager landing commit validation；
    Mooncake worker materialization metadata merge；
    KVConnectorOutput worker metadata merge；
    mixed accuracy 和 pressure summary parser。

GPU smoke:
    run_dir =
    outputs/accuracy/pd_reflex_landing_commit_smoke_contract_immutable_2026-05-19/
    20260519-115342_pdserv_mixed_longbench+reasoning_kv-reflex_int4_c8_ln2_mn2_rinf

    total_requests = 8
    failed_predictions = 0
    duration_seconds = 123.2699
    landing_materialize_event_count = 2
    landing_materialized_pages_total = 237
    landing_commit_event_count = 2
    landing_committed_pages_total = 237
    landing_materialize_gpu_ms_total = 5.798

    fatal / overwrite / Traceback / EngineCore encountered = 0
    GPU 6/7 after run: no compute processes
```

当前边界：

```text
1. landing commit 已经是真实 precision state transition，但还缺少压力扫描证明收益。
2. 未 materialize 的 landing contract 目前安全回退为 BF16 path，后续需要解释
   这是容量不足下的保守 fallback，还是应在 admission 阶段避免生成该 contract。
3. block table 已支持 INT4 entry commit，但还没有系统化暴露 page descriptor /
   lifecycle state / quality debt。
4. admission gate 仍需要从 trace-level feasibility 走向 policy-level decision：
   什么时候必须等 landing 可执行，什么时候允许回退，什么时候拒绝。
```

下一步应进入 OSDI 论文所需的系统性评估和控制闭环：

```text
1. 做 pressure sweep：并发、request rate、context length、BF16 budget 多维扫描。
2. 报告 max_waiting、TTFT、TPOT、p95/p99 latency、goodput、failed request、
   INT4 ratio、landing commit ratio、quality score。
3. 将 landing fallback 分类为 explicit policy outcome，而不是日志 warning。
4. 引入 page lifecycle descriptor 的最小可观测版本：
   BF16_ACTIVE / INT4_LANDING / INT4_ACTIVE / RELEASE_PENDING。
5. 加入 commit 后 block table invariant checker，支持长跑稳定性实验。
```

### 2026-05-19 进展：Fallback Policy Outcome 与 Commit Invariant

本轮把上一节的第 3 和第 5 项落到了代码路径中。mixed landing 不再只有
warning 日志，而是有显式 policy outcome；commit 之后也有最小 runtime
state invariant checker。

新增机制：

```text
Scheduler:
    landing contract 在 worker 未上报 materialized signal 时，记录：
    ReFlexKV trace landing_policy
        outcome=fallback_unmaterialized
        planned_pages=<N>
        materialized=False
        reason=no_materialized_signal

KVCacheManager:
    get_reflex_precision_state_counts(request_id)
        BF16_ACTIVE
        INT4_ACTIVE
        RELEASE_PENDING
        LANDING_RESERVED

    check_reflex_int4_invariants(request_id)
        检查 BF16 entry 必须指向 live BF16 block；
        检查 INT4 entry 必须对应 null BF16 slot 和已分配 INT4 block；
        检查 landing reservation 没有重复且仍然 allocated。

Summary:
    landing_policy_event_count
    landing_fallback_event_count
    landing_fallback_pages_total
    landing_fallback_unmaterialized_total
```

验证结果：

```text
CPU/单元测试：
    101 passed, 17 warnings

GPU smoke:
    run_dir =
    outputs/accuracy/pd_reflex_policy_invariant_smoke_2026-05-19/
    20260519-120703_pdserv_mixed_longbench+reasoning_kv-reflex_int4_c8_ln2_mn2_rinf

    total_requests = 8
    failed_predictions = 0
    duration_seconds = 122.1031

    landing_materialize_event_count = 2
    landing_materialized_pages_total = 235
    landing_commit_event_count = 2
    landing_committed_pages_total = 235

    landing_policy_event_count = 2
    landing_fallback_event_count = 2
    landing_fallback_pages_total = 786
    landing_fallback_unmaterialized_total = 2

    invariant violation / overwrite / Traceback / EngineCore encountered = 0
    GPU 6/7 after run: no compute processes
```

这一步的意义是把 mixed landing 从“能跑的机制”推进到“可解释的系统
policy”。后续 pressure sweep 可以直接区分：

```text
1. landing feasible 但未 materialize，所以保守 fallback；
2. materialized 并成功 commit，进入 INT4_ACTIVE；
3. commit 后 block table invariant 是否保持成立；
4. fallback pages 对 admission/tail latency 的影响。
```

下一步应真正开始 pressure sweep，并把 fallback outcome 纳入图表：

```text
commit_ratio = landing_committed_pages_total / landing_materialized_pages_total
fallback_ratio = landing_fallback_pages_total /
                 (landing_fallback_pages_total + landing_committed_pages_total)
quality_vs_fallback_ratio
tail_latency_vs_commit_ratio
```

### 2026-05-19 进展：目标 1-4 的最小 Runtime 闭环

本轮把前四个目标落成一个统一的、可测试的 runtime state/trace 闭环。
这还不是最终系统形态，但已经不再只是分散的 INT4 block patch。

目标 1：在线精度弹性管理

```text
新增 KVPageRuntimeDescriptor：
    request_id
    page_idx
    precision: BF16 / INT4
    tier: GPU
    lifecycle:
        BF16_ACTIVE
        INT4_LANDING
        RELEASE_PENDING
        INT4_ACTIVE
    physical_block_id
    bf16_block_id / int4_block_id
    planned_precision
    bf16_release_pending
    quality_debt
```

当前已经能表达 BF16 -> INT4 landing/demotion 的 online transition：

```text
BF16_ACTIVE
    -> INT4_LANDING
    -> RELEASE_PENDING
    -> INT4_ACTIVE
```

目标 2：请求级精度预算控制

Scheduler 现在显式输出 request-level precision budget trace：

```text
ReFlexKV trace precision_budget
    request
    max_int4_pages
    priority
    max_int4_fraction
    release_budget_blocks
    max_demote_per_window
    request_priority
    generated_decode_tokens
    remaining_decode_tokens
    prompt_pages
```

summary 新增：

```text
precision_budget_event_count
precision_budget_max_int4_pages_total
precision_budget_release_budget_total
precision_budget_priority_total
```

目标 3：页面级风险感知迁移

page descriptor 现在能把 P 侧风险和 D 侧迁移资格放在同一页状态里：

```text
risk_score
is_low_risk
is_full
is_shared
is_initial_protected
is_recent_protected
```

这使得每个 page 不只是 block id，而是有 precision、risk、sharing、
protection 和 lifecycle 的 runtime object。

目标 4：P/D 协同精度规划

landing contract 现在会被 manager 记录为 page-level INT4_LANDING state：

```text
D planner:
    planned_int4_landing_pages
    planned INT4 block ids

KVCacheManager:
    record_reflex_int4_landing_pages
    get_reflex_page_runtime_descriptors

Worker:
    landing_materialize

Scheduler:
    materialized-signal gated landing_commit
    fallback_unmaterialized policy outcome
```

验证结果：

```text
CPU/单元测试：
    123 passed, 17 warnings

GPU smoke:
    run_dir =
    outputs/accuracy/pd_reflex_targets_1_4_runtime_state_smoke_2026-05-19/
    20260519-135452_pdserv_mixed_longbench+reasoning_kv-reflex_int4_c8_ln2_mn2_rinf

    total_requests = 8
    failed_predictions = 0
    duration_seconds = 124.8365

    precision_budget_event_count = 3701
    precision_budget_max_int4_pages_total = 359233
    precision_budget_release_budget_total = 354063

    landing_materialize_event_count = 2
    landing_materialized_pages_total = 243
    landing_commit_event_count = 2
    landing_committed_pages_total = 243

    landing_policy_event_count = 2
    landing_fallback_pages_total = 789

    invariant violation / overwrite / Traceback / EngineCore encountered = 0
    GPU 6/7 after run: no compute processes
```

当前边界仍然清楚：

```text
1. tier 目前只有 GPU；CPU shadow 还没有进入数据路径。
2. quality_debt 目前是 descriptor 字段，不是闭环控制变量。
3. request budget 已可观测，但还没有独立 policy module 输出最终 SLO decision。
4. page risk 仍以 P-side lightweight hint / low-risk mask 为主，还不是完整
   evidence-rich risk model。
5. P/D landing 已有真实 commit/fallback，但 admission gate 还没有把
   fallback cost 反馈到下一轮 admission policy。
```

---

## 21. 2026-05-19 进展：目标 5 的 Phase-1 Recoverable Demotion

这一轮实现的是恢复机制的第一阶段，不做 DecDEC-style online residual
compensation kernel，而是先把低精度 KV page 做成有恢复路径的运行时资源：

```text
BF16_ACTIVE
    -> INT4_ACTIVE_RECOVERABLE + CPU BF16 shadow
    -> BF16_RECOVERED
```

已经进入代码路径的能力：

```text
Core metadata:
    RecoveryClass = NONE / BF16_SHADOW / FP8_SHADOW / RESIDUAL_CAPSULE
    ReflexRecoveryArtifact
    ReflexRecovery

Manager:
    demotion-time 生成 recovery artifact
    recovery_shadow_pages_by_request 显式选择部分 page 存 BF16 shadow
    recovery_shadow_pages_per_request 控制每请求最多保留多少个 shadow
    recover_reflex_int4_pages(request_id, page_indices)
    take_reflex_int4_recoveries()
    INT4_ACTIVE_RECOVERABLE / BF16_RECOVERED lifecycle

Scheduler:
    SEMANTIQ_REFLEX_BF16_SHADOW_PAGES_PER_REQUEST
    默认每个请求为本轮 demotion 的前 1 个 page 保留 BF16 shadow path
    SchedulerOutput 携带 reflex_int4_recoveries

Worker:
    recoverable demotion 时把源 BF16 block copy 到 CPU shadow store
    recovery event 将 CPU BF16 shadow copy 回新分配的 BF16 block
    request state 和 worker block table 支持 INT4 -> BF16 patch
```

这一步的系统语义是：

```text
只有真正被 demote 的 page 才付 CPU shadow 成本；
INT4 active copy 仍在 GPU 上服务；
CPU BF16 shadow 作为按需恢复路径存在；
恢复时不是全局恢复，而是由 recovery event 选择具体 page。
```

仍然没有做的部分：

```text
1. attention backend 已有可选 page-level attention mass telemetry，
   2D/3D Triton attention 已支持可选 INT4 page kernel-side counter；
   后续还需把它从近似 counter 升级为严格全局 softmax mass。
2. precision fault detector 已能消费 runtime attention-mass hint，
   并能基于 recoverable page descriptor + prefill risk 合成 relevance。
3. BF16 shadow 目前是 full-page CPU copy，不是 FP8 shadow 或 residual capsule。
4. inline residual compensation 还没有做；K/V residual kernel 仍是第二阶段。
5. recovery 当前还没有和真实 quality debt 指标闭环绑定。
```

验证结果：

```text
LD_LIBRARY_PATH=.venv/cuda-cu13-overlay/lib:... \
.venv/bin/python -m pytest tests/profiling/test_reflex_int4*.py -q

126 passed, 16 warnings
```

### 21.1 继续进展：自动恢复决策闭环

在 recoverable demotion 基础上，已经补上两类自动恢复策略：

```text
Precision fault recovery:
    从 kv_transfer_params 读取：
        reflex_precision_fault_pages
        reflex_page_relevance
        reflex_page_attention_mass
    manager 维护 page-level consecutive hit count
    relevance 连续命中达到阈值后触发 INT4 -> BF16 recovery

Synthetic recovery relevance:
    如果没有外部 attention/fault hint
    scheduler 会扫描 recoverable INT4 page descriptors
    对 risk 高于阈值且 remaining decode 仍足够长的 page
    合成 page_relevance，进入同一个 precision fault recovery planner

Background promotion:
    当 BF16 free ratio 高于 promotion watermark
    且 waiting/skipped_waiting queue 没有 admission pressure
    且 request remaining decode 足够长
    在 recoverable INT4 pages 中按 risk / remaining decode 选择 top-k 恢复

Attention backend telemetry:
    SEMANTIQ_REFLEX_ATTENTION_MASS_METADATA=1 时
    triton unified attention 在 decode step 写出 INT4 page mass counter
    默认只对包含 INT4 page 的 decode row 产出 telemetry
    如果 kernel-side counter 不可用，仍可回退到 decode-time estimator
    GPU worker 通过 ModelRunnerOutput 返回：
        reflex_page_attention_mass_by_request
        reflex_page_attention_mass_profile
    profile 记录：
        requests
        pages
        nonzero_pages
        drain_cpu_ms
    scheduler 写回 request.kv_transfer_params:
        reflex_page_attention_mass
    scheduler 同时保留最近一次 attention-mass profile，
    用于后续 recovery overhead 和 telemetry cost 分解
    下一轮 precision fault recovery planner 消费该信号后清除
    避免 stale attention mass 反复触发 recovery
```

新增控制参数：

```text
SEMANTIQ_REFLEX_RECOVERY_MAX_PAGES_PER_STEP
SEMANTIQ_REFLEX_FAULT_RELEVANCE_THRESHOLD
SEMANTIQ_REFLEX_FAULT_CONSECUTIVE_HITS
SEMANTIQ_REFLEX_FAULT_RECOVERY_MIN_FREE_RATIO
SEMANTIQ_REFLEX_ATTENTION_MASS_FAULT_THRESHOLD
SEMANTIQ_REFLEX_ATTENTION_MASS_METADATA
SEMANTIQ_REFLEX_SYNTHETIC_RECOVERY_RELEVANCE
SEMANTIQ_REFLEX_SYNTHETIC_RECOVERY_RISK_THRESHOLD
SEMANTIQ_REFLEX_BACKGROUND_PROMOTION_FREE_RATIO
SEMANTIQ_REFLEX_BACKGROUND_PROMOTION_PAGES_PER_STEP
SEMANTIQ_REFLEX_PROMOTION_MIN_REMAINING_DECODE_TOKENS
```

新的运行时链路：

```text
scheduler.new_step
    -> demotion-only/admission logic
    -> normal scheduling
    -> triton unified attention optional INT4 page mass counter
    -> worker optional attention-mass telemetry drain
    -> ModelRunnerOutput.reflex_page_attention_mass_by_request
    -> ModelRunnerOutput.reflex_page_attention_mass_profile
    -> scheduler writes request.kv_transfer_params.reflex_page_attention_mass
    -> scheduler records last attention-mass profile
    -> explicit fault / relevance / attention-mass signal merge
    -> synthetic recovery relevance from runtime page descriptors
    -> precision fault recovery planner
    -> background promotion planner
    -> SchedulerOutput.reflex_int4_recoveries
    -> worker copies CPU BF16 shadow back to GPU BF16 block
    -> block table INT4 id patched to BF16 block id
```

新的验证结果：

```text
CUDA_VISIBLE_DEVICES=6,7 \
LD_LIBRARY_PATH=.venv/cuda-cu13-overlay/lib:... \
.venv/bin/python -m pytest tests/profiling/test_reflex*.py -q

163 passed, 16 warnings
```

### 21.2 真实 1P1D 测试结果：性能、压力、精度

2026-05-19 在 GPU6/GPU7 上完成了一组真实 1P1D serving 测试：

```text
prefill worker: GPU6
decode worker:  GPU7
model:          /home/ytm/models/Llama-3.1-8B-Instruct
transport:      ReFlexMooncakeConnector
serving path:   OpenAI completions via disaggregated proxy
```

输出文件：

```text
Performance summary:
    outputs/profiling/reflexkv_real_metrics_2026-05-19/perf_summary.csv
    outputs/profiling/reflexkv_real_metrics_2026-05-19/perf_trace.jsonl

Accuracy summary:
    outputs/accuracy/reflexkv_real_metrics_2026-05-19/accuracy_summary.csv
    outputs/accuracy/reflexkv_real_metrics_2026-05-19/accuracy_trace_summary.csv
```

性能测试结果：

```text
4k input, 64 output, 4 requests, max concurrency 2:

auto:
    completed=4 failed=0
    req/s=0.2457
    total_token_throughput=1021.98 tok/s
    mean_ttft=4949.69 ms
    mean_tpot=50.48 ms
    mean_e2e=8129.64 ms

reflex_int4:
    completed=4 failed=0
    req/s=0.2994
    total_token_throughput=1245.12 tok/s
    mean_ttft=3564.09 ms
    mean_tpot=49.31 ms
    mean_e2e=6670.37 ms
    max_int4_ratio=0.0
```

这个点没有触发降级，说明低压力下 ReFlexKV 不会主动扰动 BF16
working set；收益主要来自运行波动和 decode trace 路径差异，不能作为
INT4 机制收益结论。

```text
8k input, 64 output, 4 requests, max concurrency 4:

auto:
    completed=4 failed=0
    req/s=0.1660
    total_token_throughput=1370.41 tok/s
    mean_ttft=13650.91 ms
    mean_tpot=74.01 ms
    mean_e2e=18313.70 ms
    max_decode_kv_usage=70.11%

reflex_int4 default policy:
    completed=4 failed=0
    req/s=0.1456
    total_token_throughput=1201.94 tok/s
    mean_ttft=14858.97 ms
    mean_tpot=50.15 ms
    mean_e2e=18018.40 ms
    demoted_pages=8
    max_int4_ratio=0.0
    admission_infeasible=189

reflex_int4 aggressive policy:
    completed=4 failed=0
    req/s=0.1921
    total_token_throughput=1585.40 tok/s
    mean_ttft=11489.53 ms
    mean_tpot=47.35 ms
    mean_e2e=14472.87 ms
    demoted_pages=354
    landing_materialized_pages=642
    max_int4_ratio=0.6854
    attention_mass_profile_events=171
    recovery_exec_pages=1
```

这个压力点暴露出一个关键系统问题：

```text
默认 policy 的 risk/sparse gate 太保守。
在 8k/c4 下出现大量 int4_landing_frontier_insufficient，
实际无法把足够多 BF16 pages 转为 INT4。

aggressive policy 证明底层机制可用：
    BF16 pressure -> INT4 demotion / mixed landing
    max_int4_ratio -> 0.6854
    throughput 相对 auto pressure 点提升约 15.7%
    mean E2E latency 从 18.31s 降到 14.47s

但 aggressive policy 不是最终论文策略，
下一步需要把 default policy 调整到接近 aggressive 的容量收益，
同时保留 risk / SLO / sharing 保护。
```

小规模精度测试结果：

```text
datasets:
    qasper:   2 samples
    hotpotqa: 2 samples
    math500:  2 samples

auto:
    qasper score=0.3350, latency=7.91s
    hotpotqa score=0.6667, latency=15.77s
    math500 score=0.5000, latency=17.33s

reflex_int4 aggressive:
    qasper score=0.3350, latency=9.63s
    hotpotqa score=0.6667, latency=13.03s
    math500 score=1.0000, latency=31.10s
    landing_materialized_pages=128
    max_int4_ratio=0.5689
    attention_mass_profile_events=30
```

由于每个数据集只有 2 条样本，这组结果只能作为 smoke accuracy，
不能作为论文质量结论。它证明的是：

```text
真实 LongBench/Math500 请求可以走通 P/D serving + ReFlexKV。
ReFlexKV aggressive 下没有请求失败。
qasper/hotpotqa 的小样本分数未低于 auto。
math500 小样本分数升高只是采样方差，不能解读为质量提升。
```

当前最重要的实验结论：

```text
1. 机制可用：
   ReFlexKV 可以在真实 P/D serving 中产生 INT4 KV working set，
   并输出 demotion、landing、attention-mass telemetry、recovery trace。

2. 默认策略不足：
   default risk/sparse policy 在压力下过保守，
   admission_infeasible 很多，INT4 ratio 上不去。

3. 下一步不是继续堆功能，
   而是把 policy frontier 从 aggressive 拉回到 risk-aware：
       保留 evidence/recent/shared/SLO 保护
       但允许足够的 low-risk pages 进入 INT4
       让 default policy 在 8k/c4 下也能达到非零且稳定的 INT4 ratio。
```

---

## 22. 最终论文技术贡献

最终 ReFlexKV 应形成四个明确贡献。

### Contribution 1：Precision-elastic KV Memory Abstraction

提出 precision-aware KV page table 和 KV page lifecycle，将 KV cache 从空间资源扩展为 precision/tier/state/risk/sharing 共同管理的运行时对象。

### Contribution 2：Closed-loop Precision Controller

提出 SLO-aware、quality-aware、feasibility-aware 的在线控制器，维护 precision working set，在 memory pressure 和 precision fault 之间进行闭环迁移。

### Contribution 3：Cooperative Landing and Recovery

提出 P/D 协同精度规划机制，P 侧生成轻量 page risk hints，D 侧执行 mixed-precision landing、active/recovery precision 分离和低精度 page recovery。

### Contribution 4：Precision-coherent Shared KV Cache

提出 shared prefix pages 的精度一致性管理，包括 shared precision contract、copy-on-demote 和 multi-version prefix cache，使 mixed precision KV 管理能够适配真实 serving 中的 prefix sharing。

---

## 23. 成功标准

ReFlexKV 最终不应只证明：

```text
INT4 KV 可以省显存。
```

而应证明：

```text
在动态、混合、长上下文、多优先级、prefix-sharing 的 LLM serving 场景中，
KV precision 必须成为运行时可调度资源；
通过 precision-elastic KV memory management，
系统可以在质量可控的前提下显著改善 admission、tail latency、goodput 和过载稳定性。
```

最终系统应达到以下效果：

```text
低压力：
    几乎不触发迁移，质量和 latency 接近 BF16。

中压力：
    通过低风险 demotion 降低 BF16 pressure，提高 goodput。

高压力：
    通过 admission-aware demotion 和 mixed-precision landing 避免 admission collapse。

过载：
    通过 quality debt、SLO-aware policy 和 recovery path 实现优雅降级。

质量敏感场景：
    通过 precision fault 和 recovery 避免不可逆质量损失。

prefix sharing 场景：
    通过 shared precision contract 和 copy-on-demote 保持共享一致性。
```

---

## 24. 一句话总结

ReFlexKV 的最终形态应是：

> **一个将 KV cache 的 precision、tier、risk、sharing 和 recovery 纳入统一运行时管理的弹性 KV 内存系统，使 LLM serving 能够在动态压力下以质量可控的方式调整 KV 精度状态，而不是静态地选择 BF16 或 INT4。**

[1]: https://arxiv.org/abs/2309.06180?utm_source=chatgpt.com "Efficient Memory Management for Large Language Model Serving with PagedAttention"
