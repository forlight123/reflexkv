# ReFlexKV 当前实现总结

更新时间：2026-05-23，Asia/Shanghai。

本文档是 SemantiQ/ReFlexKV 当前代码状态、策略设计、实现细节和实验结果的详细交接记录。它对照 `reflexkv.md` 的论文级目标，说明当前系统已经做到了什么、这轮修复解决了什么、真实效果是什么，以及后续还需要继续推进哪些工作。

## 1. 当前一句话状态

ReFlexKV 现在已经不是单纯的“INT4 KV cache 量化”原型，而是一个围绕 precision-aware KV memory 的闭环在线优化系统雏形。当前系统已经包含：

- precision-aware KV memory/page metadata；
- chunk/frontier admission；
- decode-side dynamic demotion；
- direct/mixed INT4 landing；
- simplified frontier-dual online planner；
- P-side page risk/semantic summary；
- risk-aware INT4 quantizer；
- BF16 shadow + budgeted background promotion plumbing（实时注意力质量故障恢复路径已删除，避免高开销且不可稳定验证的闭环）；
- prefix precision contract 雏形（已有 contract/helper、page protection 和 copy-on-demote hook，但还不是完整 shared-prefix precision 系统）；
- P/D Mooncake connector integration；
- telemetry/accounting/rejection reason breakdown。

它仍然没有达到 `reflexkv.md` 中完整 ATC/OSDI 论文目标。最重要的差距是：decode-side demotion、frontier admission 和 P-side risk metadata 已经能在真实压力下工作，但 risk estimator、quantizer、recovery、prefix contract 和 online optimizer 还需要更强的算法与系统级实验矩阵来证明质量和吞吐收益。

### 1.1 2026-05-22 状态修订

这次需要特别修正两个容易被误读的点：

1. 精度恢复路径已经收敛。
   - 保留：BF16 shadow metadata、cache manager recovery entry、budgeted background promotion、telemetry 和单元测试。
   - 删除：decode attention mass 采集、按实时注意力触发的恢复 planner、synthetic relevance fallback、对应 kernel/worker/scheduler 汇总字段。
   - 结论：当前 recovery 不是“模型发现 INT4 页重要后立即恢复”的闭环，而是低开销的 shadow + 后台 promotion 机制。后续如果要恢复闭环检测，必须先有低成本 relevance signal 和明确的 on/off 消融收益。

2. Prefix 不是完全没考虑，但也没有完整落地。
   - 已有：prompt/page protection、landing contract 的 page-level protection、`PrefixPrecisionContractManager`、copy-on-demote metadata/hook、shared page 单元测试。
   - 未验证：没有真实 shared-prefix workload；最新 smoke 使用的路径里 `candidate_shared_bf16_pages_total=0`、`candidate_copy_on_demote_pages_total=0`。
   - 结论：当前 prefix 是 control-plane skeleton + safety hook，不是完整 multi-version prefix cache / precision ownership / eviction 系统。

### 1.1.1 2026-05-23 论文实验数据准备

为了从 n=100 smoke 走向 ATC/OSDI 级实验，已先补齐 reasoning 和 RULER 测试文件：

Reasoning 数据：

| dataset | source | output | samples | status |
|---|---|---|---:|---|
| GSM8K | `data/reasoning/gsm8k.parquet` | `data/reasoning/gsm8k.jsonl` | 1319 | 已转换 |
| AIME 2024 | `data/reasoning/aime24.parquet` | `data/reasoning/aime24.jsonl` | 30 | 已转换 |
| AIME 2025 | `data/reasoning/aime25.jsonl` | `data/reasoning/aime25.jsonl` | 30 | 已标准化重写 |
| Math500 | `data/reasoning/math500.jsonl` | 原文件 | 500 | 已有 |

新增转换脚本：

```text
gen_data/prepare_reasoning_datasets.py
```

该脚本会：

- 从 GSM8K 的 `####` 后提取最终答案；
- 将 GSM8K/AIME 统一成 `{problem, answer, unique_id}` reasoning schema；
- 更新 `eval/config/reasoning_dataset2{prompt,maxlen,metric,samples}.json`；
- 当前 max_new_tokens：Math500/AIME=4096，GSM8K=1024；
- metric 暂用现有 `boxed_accuracy`，因此 prompt 要求最终答案放在 `\boxed{}`。

RULER 数据：

新增生成脚本：

```text
gen_data/prepare_ruler_datasets.py
```

已用 RULER 官方 synthetic generator 生成 3 个 NIAH/retrieval 任务，每个 4 个长度，每组 100 samples：

| task | lengths | normalized output |
|---|---|---|
| `niah_single_1` | 4k/8k/16k/32k | `data/ruler/ruler_niah_single_1_{4k,8k,16k,32k}.jsonl` |
| `niah_multikey_2` | 4k/8k/16k/32k | `data/ruler/ruler_niah_multikey_2_{4k,8k,16k,32k}.jsonl` |
| `niah_multikey_3` | 4k/8k/16k/32k | `data/ruler/ruler_niah_multikey_3_{4k,8k,16k,32k}.jsonl` |

RULER raw 文件保存在：

```text
data/ruler_raw/
```

RULER normalized 文件保存在：

```text
data/ruler/
```

校验结果：

- 12 个 normalized RULER jsonl 文件；
- 每个 100 samples，总计 1200 samples；
- raw + normalized 共 2400 行；
- 每个文件的 `length <= max_seq_length`，无超长样本；
- normalized `unique_id` 使用文件行号，避免 RULER raw `index` 字段不是稳定样本序号的问题。

注意：RULER 文件已经生成，但还没有接入 ReFlexKV mixed serving/evaluator。下一步需要把 RULER 接成单独 `ruler` task 或转接 LongBench-style retrieval evaluator，不能直接用 reasoning 的 `boxed_accuracy`。

### 1.2 最新 admission 修复结果

最新 smoke 目录：

```text
outputs/accuracy/reflex_next_policy_admission_fix7_2026-05-22/ablation_00_frontier_dual_reflex
```

关键结果：

| metric | value |
|---|---:|
| duration | 62.658s |
| total requests | 4 |
| failed predictions | 0 |
| max decode KV usage | 95.65% |
| max decode running | 2 |
| max decode waiting | 1 |
| avg decode waiting | 0.25 |
| demotion_event_count | 10 |
| demoted_pages_total | 1098 |
| actual_release_blocks_total | 1098 |
| admission_blocked_total | 0 |
| admission_infeasible_total | 0 |
| demotion_gpu_ms_total | 59.021 |
| mean_int4_ratio | 0.2692 |
| max_int4_ratio | 0.6683 |
| page_metadata_real_risk_coverage_ratio | 1.0 |

Dataset 结果：

| dataset | completed | failed | avg latency | avg score |
|---|---:|---:|---:|---:|
| gov_report | 2 | 0 | 22.35s | 0.4000 |
| math500 | 2 | 0 | 13.52s | 0.5 |

和上一轮 `fix6` 相比：

- `duration_seconds` 从约 133.65s 降到 62.66s；
- `admission_blocked_total` 从 19 降到 0；
- `admission_infeasible_total` 从 2 降到 0；
- `demotion_gpu_ms_total` 从约 892.54ms 降到 59.02ms；
- `max_int4_ratio` 从 0.9654 降到 0.6683，说明策略不再为了 admission 把页面压得过狠。

这轮修复说明 admission emergency release 和 allocation failure target slack 有效：系统不再靠大量重复 demotion 和极端 INT4 ratio 硬顶过去。但这只是 2+2 小样本 smoke，还不是最终论文实验。

### 1.3 当前实现和结果快照

当前代码实现边界：

- 保留并继续作为主线：P-side page risk metadata、chunk/frontier admission、frontier-dual online planner、mixed/direct INT4 landing、decode-side dynamic demotion、mixed-precision attention、BF16 shadow metadata、budgeted background promotion、prefix protection/copy-on-demote skeleton。
- 已删除并不再作为当前机制：decode attention mass profile、attention relevance 触发的 precision fault recovery、synthetic recovery relevance fallback、`RecoveryQualityEvaluator`、`precision_kv/recovery.py`、相关 worker/model output/kernel 字段和 profiling/accuracy 汇总列。
- 当前 recovery 只能表述为 “BF16 shadow + budgeted background promotion plumbing”，不能表述为“实时精度故障检测与恢复闭环”。

当前最新 post-optimization n=100 Burst-like mixed workload 结果：

| metric | BF16 stable | ReFlexKV frontier-dual |
|---|---:|---:|
| samples | 100 | 100 |
| completed | 100 | 100 |
| goodput | 0.0337 req/s | 0.0968 req/s |
| weighted avg latency | 113.57s | 39.14s |
| weighted avg score | 0.3402 | 0.3493 |
| admission_blocked_total | N/A | 0 |
| admission_infeasible_total | N/A | 0 |
| admission_control_event_count | N/A | 108 |
| page_metadata_plan_event_count | N/A | 817 |
| demotion_event_count | N/A | 236 |
| demoted_pages_total | N/A | 15111 |
| max_int4_ratio | N/A | 0.8743 |
| background_promoted_pages_total | N/A | 275 |

数据集构成：

| dataset | samples | max_new_tokens |
|---|---:|---:|
| gov_report | 35 | 512 |
| math500 | 40 | 4096 |
| qasper | 25 | 128 |

这个结果可以支持的结论：

- ReFlexKV 在该压力 workload 下完成了所有请求，并且相比 BF16 stable baseline 有约 2.9x goodput、约 2.9x weighted latency 改善。
- 当前分数没有下降，但不能写成“ReFlexKV 提升精度”，因为 score 差异可能来自调度、截断边界、完成时序和生成路径差异。
- 最新 scheduler/planner 优化显著降低了控制面事件和 demotion GPU time，但端到端 duration 基本持平，说明当前 n=100 主要瓶颈已经转向 decode 计算/长输出本身。
- 主要收益应归因于 admission + demotion + mixed precision KV，而不是 recovery。background promotion 真实触发过，但没有 recovery on/off 消融。

本轮清理验证：

| check | result |
|---|---:|
| runtime/scripts `py_compile` | pass |
| 删除旧 recovery 入口 + background promotion targeted tests | 4 passed |
| profiling/accuracy regression subset | 210 passed |
| CUDA direct attention tests | 2 passed |

### 1.4 2026-05-22 post-cleanup INT4 测试与 scheduler/planner 优化

#### 1.4.1 post-cleanup n=100 结果

清理实时 attention/precision-fault recovery 后，重新跑了当前 INT4 主路径：

```text
outputs/accuracy/burstgpt_answerable_n100_reflex_post_cleanup_b736_2026-05-22/ablation_00_frontier_dual_reflex
```

配置：

- dataset：BurstGPT-shaped n=100 manifest；
- samples：gov_report 35、math500 40、qasper 25；
- decode KV dtype：`reflex_int4`；
- BF16 block override：736；
- page selection：`frontier_dual`；
- max concurrency：4；
- proxy prefill max inflight：4；
- prompt truncation：0。

总体结果：

| metric | value |
|---|---:|
| completed | 100 / 100 |
| failures | 0 |
| duration | 1030.20s |
| goodput | 0.0971 req/s |
| weighted avg latency | 38.85s |
| weighted avg score | 0.3744 |
| max decode KV usage | 100.0% |
| avg decode KV usage | 66.27% |
| max decode running | 4 |
| avg decode running | 3.56 |
| max decode waiting | 3 |
| avg decode waiting | 0.21 |

Dataset 结果：

| dataset | samples | avg latency | score |
|---|---:|---:|---:|
| gov_report | 35 | 18.96s | 0.3452 |
| math500 | 40 | 76.57s | 0.4500 |
| qasper | 25 | 6.35s | 0.2944 |

ReFlex trace：

| metric | value |
|---|---:|
| demotion_event_count | 731 |
| demoted_pages_total | 14904 |
| admission_control_event_count | 5556 |
| admission_blocked_total | 0 |
| admission_infeasible_total | 0 |
| admission_wait_reduction_total | 1598 |
| page_metadata_real_risk_coverage_ratio | 1.0 |
| recovery_plan_event_count | 385 |
| background_promoted_pages_total | 385 |
| demotion_gpu_ms_total | 2312.76 |
| mean_int4_ratio | 0.3503 |
| max_int4_ratio | 0.8667 |

对比稳定 BF16 baseline：

| metric | BF16 stable | post-cleanup ReFlexKV | ratio / delta |
|---|---:|---:|---:|
| goodput | 0.0337 req/s | 0.0971 req/s | 2.88x |
| weighted avg latency | 113.57s | 38.85s | 2.92x lower |
| weighted avg score | 0.3402 | 0.3744 | +0.0342 |

结论仍需谨慎：score 更高不能直接写成 ReFlexKV 提升精度，只能写成在该 trace 下没有观察到 accuracy degradation。当前主要收益还是 admission + demotion + mixed precision KV。

#### 1.4.2 本轮发现的问题

post-cleanup run 能完整跑完，但暴露出 scheduler/planner cadence 问题：

1. `admission_control_event_count=5556`，其中大量是同一个 waiting / remote chunk request 在每个 decode step 反复检查；很多记录是 `requested_release=0`、`landing_reason=bf16_fit`，没有必要进入 demotion planner。
2. 后台 demotion 有大量小批量事件。n=100 中 `demotion_event_count=731`，`demotion_gpu_ms_total=2312.76ms`；日志中多次出现 1-2 pages 的 background demotion，单次仍要付 planning 和 kernel launch 成本。
3. 如果 background planner 选不出 page，旧逻辑不会进入 cooldown，可能每个 step 重复做无效 `page_metadata_plan / precision_budget / candidate_breakdown`。

#### 1.4.3 已实现的 scheduler/planner 修复

本轮做了三个低风险修复：

1. zero-deficit waiting admission 直接跳过 planner。
   - 当 waiting request 已经 BF16 fit，`target_blocks=0` 时，`_try_reflex_int4_demotion_only_step()` 不再调用 admission planner、landing planner 和 admission trace logging。
   - 目标是减少无意义 `admission_control` 事件和每步轮询开销。

2. background pressure 使用最小 demotion batch。
   - 新增 `SEMANTIQ_REFLEX_BACKGROUND_MIN_DEMOTIONS_PER_STEP`。
   - 默认最小后台 batch 为 8 pages，最大仍受 `SEMANTIQ_REFLEX_BACKGROUND_DEMOTIONS_PER_STEP` / 内部 limit 限制。
   - 目标是避免 1-page/2-page demotion 的高固定成本。

3. background no-op planning 进入 cooldown。
   - 如果 background pressure 已经尝试 planning，但 `planned_blocks=0`，也会更新 `_reflex_int4_last_demote_step`。
   - 目标是避免没有可选页时每个 step 重复做同一轮空 planner。

新增/更新单测：

- `test_reflex_int4_waiting_demotion_only_step_skips_planner_when_request_fits`
- `test_reflex_int4_background_target_uses_min_batch_under_pressure`
- `test_reflex_int4_background_noop_demote_enters_cooldown`

#### 1.4.4 post-optimization n=32 smoke

优化后跑了一个较短 serving smoke：

```text
outputs/accuracy/burstgpt_answerable_n32_reflex_scheduler_opt_b736_2026-05-22/ablation_00_frontier_dual_reflex
```

结果：

| metric | value |
|---|---:|
| completed | 32 / 32 |
| failures | 0 |
| duration | 336.05s |
| goodput | 0.0952 req/s |
| weighted avg latency | 34.58s |
| weighted avg score | 0.4436 |
| admission_blocked_total | 0 |
| admission_infeasible_total | 0 |
| demotion_event_count | 60 |
| demoted_pages_total | 5622 |
| demotion_gpu_ms_total | 403.38 |
| max_int4_ratio | 0.9083 |

注意：这个 n=32 smoke 覆盖了前两项 scheduler 修改；第三项 no-op cooldown 是 smoke 启动后根据日志继续修的。完整集成后的 n=100 结果见下一节。

#### 1.4.5 post-optimization opt2 n=100 结果

完整集成三项 scheduler/planner 修复后，又跑了一次 n=100：

```text
outputs/accuracy/burstgpt_answerable_n100_reflex_scheduler_opt2_b736_2026-05-22/ablation_00_frontier_dual_reflex
```

总体结果：

| metric | value |
|---|---:|
| completed | 100 / 100 |
| failures | 0 |
| duration | 1033.00s |
| goodput | 0.0968 req/s |
| weighted avg latency | 39.14s |
| weighted avg score | 0.3493 |
| max decode KV usage | 100.0% |
| avg decode KV usage | 65.90% |
| max decode running | 4 |
| avg decode running | 3.60 |
| max decode waiting | 2 |
| avg decode waiting | 0.18 |

Dataset 结果：

| dataset | samples | avg latency | score |
|---|---:|---:|---:|
| gov_report | 35 | 18.19s | 0.3504 |
| math500 | 40 | 78.06s | 0.4000 |
| qasper | 25 | 6.19s | 0.2668 |

ReFlex trace：

| metric | post-cleanup | scheduler opt2 | change |
|---|---:|---:|---:|
| admission_control_event_count | 5556 | 108 | -98.1% |
| page_metadata_plan_event_count | 2511 | 817 | -67.5% |
| precision_budget_event_count | 10040 | 3266 | -67.5% |
| candidate_breakdown_event_count | 2405 | 709 | -70.5% |
| demotion_event_count | 731 | 236 | -67.7% |
| demotion_gpu_ms_total | 2312.76 | 969.19 | -58.1% |
| trace_events | 81841 | 66351 | -18.9% |
| admission_blocked_total | 0 | 0 | same |
| admission_infeasible_total | 0 | 0 | same |
| demoted_pages_total | 14904 | 15111 | +1.4% |
| max_int4_ratio | 0.8667 | 0.8743 | +0.0076 |

对比 BF16 stable baseline：

| metric | BF16 stable | scheduler opt2 ReFlexKV | ratio / delta |
|---|---:|---:|---:|
| goodput | 0.0337 req/s | 0.0968 req/s | 2.87x |
| weighted avg latency | 113.57s | 39.14s | 2.90x lower |
| weighted avg score | 0.3402 | 0.3493 | +0.0091 |

结论：

- scheduler/planner 修复有效，控制面噪声明显下降，demotion event 和 migration GPU time 大幅下降。
- 端到端 duration 没明显下降：`1030.20s -> 1033.00s`，说明这个 n=100 trace 上主要瓶颈已经不是 planner spam，而是 decode 计算、长输出和实际 attention path。
- score 仍不能归因；只能写没有观察到 accuracy degradation。

#### 1.4.6 本轮验证

| check | result |
|---|---:|
| post-cleanup n=100 ReFlexKV serving | 100 / 100 completed |
| post-optimization n=32 ReFlexKV smoke | 32 / 32 completed |
| post-optimization opt2 n=100 ReFlexKV serving | 100 / 100 completed |
| scheduler/policy/frontier/optimizer/tests | 213 passed |
| `py_compile` touched scheduler/tests | pass |

下一步建议：

1. 先不要继续堆 admission planner 优化；当前控制面已经下降明显，n=100 端到端瓶颈更像 decode/mixed attention 和长输出。
2. 下一轮应做 decode path profiling：打开 attention/kernel 级 timing，区分 BF16 attention、INT4 decode attention、dequant/metadata lookup、migration copy 的真实占比。
3. 单独加 background promotion/shadow off ablation，确认 275 pages promotion 对 accuracy 和 latency 的影响。
4. 如果 n=200/更高压力下 `admission_control_event_count` 或 `page_metadata_plan_event_count` 又升高，再做第二层 admission decision cache。

## 2. 这轮根本修复：compiled prefill 下 P-side metadata 不能丢

### 2.1 问题

上一轮 smoke 暴露了一个很关键的问题：

```text
prefill metadata recorder 会在 torch.compile / TorchDynamo tracing 中碰到 threading.Lock，
导致 prefill engine 失败。
```

临时处理是：

```python
if torch.compiler.is_compiling():
    return
```

这个处理虽然避免了 engine crash，但副作用非常大：compiled prefill 路径下 P-side page risk / semantic summary 被直接跳过。也就是说：

- decode scheduler 可以看到 synthetic fallback landing pages；
- 但是真正来自 P-side 的 semantic risk telemetry 没有稳定进入 P/D 数据面；
- long prompt 场景下的风险估计、compressible page、BF16 shadow selection 都会退化；
- ReFlexKV 的 P2 质量控制路径会变成“有接口但没有真实在线输入”。

这个问题不能继续保留，否则后面的 estimator/quantizer/recovery 实验都没有可靠基础。

### 2.2 修复策略

这轮没有继续使用 `is_compiling()` skip，也没有要求实验强制 eager。新的策略是：

```text
把 recorder 的 Python side effect 包成 torch custom op。
TorchDynamo/Inductor 只看到一个 opaque op 和 fake implementation；
真正运行时再由 custom op 调用 Python recorder，把 page-level summary 写入 side-channel。
```

具体设计：

- `maybe_record_reflex_prefill_layer()` 不再在 compile 中 return。
- 新增 `torch.ops.vllm.record_reflex_prefill_metadata(query, key, layer_name)`。
- custom op 注册在 `CompositeExplicitAutograd` dispatch key 上，CPU/CUDA tensor 都可以跑测试。
- fake impl 返回 `None`，用于 Dynamo/FakeTensor tracing。
- op 的真实实现调用 `_RECORDER.record_layer(layer_name, query, key)`。
- custom op 声明 `mutates_args=["query", "key"]`，目的是让编译器保留这个 side-effect op，不把它当成无用的纯函数删掉。

涉及文件：

- `vllm/vllm/v1/core/reflex_prefill_metadata.py`
- `tests/profiling/test_reflex_prefill_metadata.py`

核心效果：

```text
compiled prefill 不再 trace recorder 内部的 Python lock；
但 runtime 仍然会执行 page metadata recording。
```

### 2.3 回归测试

新增/替换测试：

```text
test_reflex_prefill_recorder_records_from_torch_compile
```

测试行为：

1. 打开 `SEMANTIQ_REFLEX_PREFILL_PAGE_METADATA=1`。
2. 用真实 `ReflexPrefillMetadataRecorder`。
3. 在 `torch.compile(backend="eager")` 的函数中调用 `maybe_record_reflex_prefill_layer()`。
4. 退出 batch 后 drain completed request。
5. 验证 `req-compiled` 产生了 page risk scores。

这个测试在旧实现上失败：

```text
KeyError: 'req-compiled'
```

原因是 compile 时直接 return，没有记录任何 metadata。

新实现通过：

```text
1 passed
```

这说明问题不是被隐藏，而是 compiled path 已经能真实产出 page risk。

## 2.4 本轮 1-4 实现：可解释 telemetry、ablation runner、阻塞诊断、实际 workload 生成

本轮按下一步建议完成了四件直接服务实验闭环的工作。目标不是继续堆策略，而是先让后续实验能回答清楚：

```text
到底是谁在挡 admission、真实 P-side risk 有没有到 D-side、direct landing 有没有 materialize、
不同策略的吞吐/准确率/blocked reason 能不能用同一套脚本对比。
```

### 2.4.1 Telemetry / accounting

新增和统一的 trace 字段：

- `page_metadata_produce`：P-side prefill recorder 产出了多少 request/page 级 risk metadata。
- `page_metadata_receive`：Mooncake connector/D-side 收到了多少 request/page 级 metadata。
- `page_metadata_plan`：D-side planner 当前使用了多少 real risk、explicit compressible、BF16 shadow、synthetic fallback pages。
- `landing_metadata_source`：admission 时 direct/mixed landing 的候选来自 `real_risk`、`explicit_compressible`、`synthetic_chunk` 还是 `none`。
- `landing_real_risk_pages`、`landing_explicit_compressible_pages`、`landing_synthetic_pages`：直接量化 landing 候选的来源分解。
- `landing_fallback_unmaterialized_ratio`：planned landing pages 和真正 materialized pages 的差距。
- `page_metadata_real_risk_coverage_ratio`：真实 P-side risk 相对 planner 使用页面的覆盖率。

涉及文件：

- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
- `scripts/profiling/summarize_reflex_pd_pressure.py`
- `scripts/accuracy/run_pd_serving_mixed_accuracy.py`
- `scripts/profiling/run_reflex_pd_1p1d.py`
- `scripts/accuracy/run_pd_serving_accuracy.py`

关键变化：

- profiling summarizer 现在同时解析 `prefill_server.log`、`decode_server.log`、`proxy.log`，不再只看 decode log。
- mixed accuracy 的 `mixed_summary.json` 现在复用同一套 trace parser，所以 accuracy run 和 profiling run 的 ReFlexKV 字段一致。
- 新增 `--disable-reflex-prefill-page-metadata`，用于 P-side risk on/off 消融。

### 2.4.2 Ablation runner

新增脚本：

```text
scripts/accuracy/run_reflex_ablation_matrix.py
```

当前内置 cases：

- `bf16_baseline`
- `heuristic_reflex`
- `frontier_dual_reflex`
- `direct_landing_off`
- `direct_landing_on`
- `p_side_risk_off`

设计意图：

- 默认 dry-run，先输出命令和环境变量，不直接占 GPU。
- 每个 case 统一走 `run_pd_serving_mixed_accuracy.py`，便于后续比较吞吐、准确率和 blocked reason。
- 输出 `commands.jsonl`，后续可以直接追踪每次实验配置。

### 2.4.3 Blocker diagnosis

新增脚本：

```text
scripts/profiling/diagnose_reflex_blockers.py
```

输入可以是 profiling summary，也可以是 `mixed_summary.json`。输出结构化诊断：

- `area`
- `severity`
- `signal`
- `finding`
- `action`

当前诊断重点：

- `direct_landing_materialization`
- `p_side_risk_metadata`
- `chunk_admission`
- `request_precision_budget`
- `sparse_window_quota`
- `frontier_dual_optimizer`
- `page_lifecycle`

这个脚本的作用是把 “admission 卡了” 拆成可操作的策略问题，而不是只看一个总的 infeasible 数字。

### 2.4.4 实际 workload/testset 生成

新增脚本：

```text
scripts/accuracy/generate_reflex_workloads.py
```

能力：

- 离线生成 mixed workload manifest。
- 复用 `run_pd_serving_mixed_accuracy.py` 的 LongBench/Math500 loader、prompt template、max_new_tokens 和 SLO/priority 分配。
- 输出 JSONL，每条包含 request index、task/dataset/source index、max_new_tokens、SLO class / priority、prompt、answers、all_classes 和 meta。
- 输出 summary JSON，记录 dataset counts 和 prompt chars 统计。

这样后面跑 BF16 baseline、heuristic、frontier_dual、direct landing on/off、P-side risk on/off 时，可以固定同一批请求，避免 workload shuffle 影响结论。

## 2.5 本轮继续修复：metadata race 和 direct landing 诊断口径

这次中断恢复后，先跑了一个小 mixed workload 实测，暴露了两个不同层次的问题：

1. P-side 明明产出了 page metadata，但 D-side planner 仍然退回 synthetic fallback。
2. diagnosis 把 admission 阶段的 landing 试规划误认为真实 direct landing contract，导致 `direct_landing_materialization` P0 误报。

这两个问题很关键，因为它们会直接污染后续实验判断：前者会让 P-side risk 消融不可信，后者会让 direct landing on/off 的 blocked reason 归因不可信。

### 2.5.1 Workload generator 修复

`generate_reflex_workloads.py` 现在补了两个实际运行中遇到的问题：

- 新增 `--model` 参数，支持传入真实 tokenizer 路径，例如 `/home/ytm/models/Llama-3.1-8B-Instruct`。
- `answers` / `all_classes` 允许为 `None`，生成 JSONL 时统一落成空 list，避免 LongBench/GovReport 某些字段为空时崩溃。

实际 smoke 生成：

```text
/tmp/reflex_workload_gov4_math4.jsonl
/tmp/reflex_workload_gov4_math4_summary.json
```

结果：

| dataset | requests | avg prompt chars |
|---|---:|---:|
| gov_report | 4 | 59470.25 |
| math500 | 4 | 283.25 |

对应测试：

```text
tests/accuracy/test_generate_reflex_workloads.py
2 passed
```

### 2.5.2 Proxy metadata wait 修复

问题根因在 proxy：

```text
generate_stream 开始 decode 时，只在 prefill_task.done() 时读取 prefill 返回的 kv_transfer_params。
如果 prefill 还没 done，decode 直接开始，后续 prefill 返回的 reflex_page_risks 不会再进入 D-side request。
```

这会导致一种很隐蔽的 race：

- P-side `page_metadata_produce` 非零；
- Mooncake/D-side `page_metadata_receive` 也非零；
- 但 admission planner 当轮没有拿到 request-level `reflex_page_risks`；
- direct/mixed landing 候选退回 synthetic chunk fallback。

修复：

- proxy 新增 `--prefill-metadata-wait-timeout-sec`。
- `run_reflex_pd_1p1d.py` 和 accuracy runners 接入该参数。
- ablation runner 默认对 ReFlexKV + P-side metadata case 设置 `5.0s`，对 BF16 或 `p_side_risk_off` 设置 `0.0s`。
- decode stream 启动前，如果配置了 wait timeout，会短暂等待 prefill task 返回 metadata；timeout 时不取消 prefill task，仍允许请求继续。

这不是把 prefill/decode 强行串行化，而是只在需要 page risk metadata 的 ReFlexKV 路径上给一个有限等待窗口，避免 D-side 永久走 synthetic fallback。

### 2.5.3 小 mixed 实测前后对比

修复前目录：

```text
outputs/accuracy/reflex_next_fix_2026-05-21/ablation_00_frontier_dual_reflex
```

修复后目录：

```text
outputs/accuracy/reflex_next_fix_wait_2026-05-21/ablation_00_frontier_dual_reflex
```

对比：

| metric | before metadata wait | after metadata wait |
|---|---:|---:|
| completed | 8/8 | 8/8 |
| duration | 132.0736s | 130.6103s |
| gov_report score | 0.3540 | 0.3765 |
| Math500 score | 0.75 | 0.75 |
| page_metadata_produced_pages | 2866 | 2866 |
| page_metadata_received_pages | 2866 | 2866 |
| page_metadata_plan_real_risk_pages | 0 | 402368 |
| page_metadata_plan_synthetic_pages | 11336 | 0 |
| page_metadata_real_risk_coverage_ratio | 0 | 1.0 |
| demotion events | not primary | 85 |
| demoted pages | not primary | 2045 |
| max INT4 ratio | not primary | 0.9615 |
| mean INT4 ratio | not primary | 0.1271 |

结论：

- P-side metadata 生产/传输本身不是主要问题，race 在 proxy decode-start 时机。
- metadata wait 后，D-side planner 不再依赖 synthetic fallback。
- 这是后续 P-side risk on/off 消融能够成立的必要条件。

### 2.5.4 Landing contract telemetry / diagnosis 修复

第二个问题是诊断误报。之前 diagnosis 用：

```text
admission_planned_int4_landing_total - landing_materialized_pages_total
```

判断 direct landing materialization 是否失败。

这个口径不对，因为 `admission_planned_int4_landing_total` 是 scheduler 每步 admission trial 的规划量，可能重复出现，也可能只是“理论上 mixed landing feasible”。它不代表真正写入 request `kv_transfer_params` 的 landing contract。

修复：

- scheduler 在 `_persist_reflex_int4_landing_contract()` 真正写入或变更 contract 时，新增 trace：

```text
ReFlexKV trace landing_contract request=... pages=... direct=... required_blocks=... planned_blocks=... reason=...
```

- summarizer 新增：

```text
landing_contract_event_count
landing_contract_persisted_pages_total
landing_contract_direct_pages_total
```

- diagnoser 改成只用：

```text
landing_contract_persisted_pages_total - landing_materialized_pages_total
```

判断真实 materialization gap。

用修复后的 diagnoser 重新诊断上面的 after run，`direct_landing_materialization` P0 不再出现。剩余问题变成真实策略问题：

| area | severity | signal |
|---|---:|---:|
| chunk_admission | P0 | 6 |
| request_precision_budget | P1 | 42786 |
| sparse_window_quota | P1 | 5376 |
| frontier_dual_optimizer | P1 | 2676 |
| page_lifecycle | P1 | 96204 |

这说明下一步不应该继续追 direct landing 假阳性，而应该继续改 admission/chunk-native scheduler、层次化 page budget、sparse quota 和 shared/open page lifecycle。

### 2.5.5 新代码 real smoke

为确认修复没有只停留在单测，又跑了一轮更小的真实 serving：

```text
outputs/accuracy/reflex_next_contract_2026-05-21/ablation_00_frontier_dual_reflex
```

核心配置：

```text
gov_report 2
Math500 2
max_concurrency 2
num_gpu_blocks_override 736
prefill GPU 0
decode GPU 1
frontier_dual_reflex
```

结果：

| metric | value |
|---|---:|
| completed | 4/4 |
| duration | 72.9859s |
| gov_report score | 0.3973 |
| Math500 score | 1.0 |
| demotion events | 27 |
| demoted pages | 1059 |
| admission control events | 216 |
| admission infeasible | 4 |
| admission blocked | 32 |
| admission success after demote | 169 |
| admission planned INT4 landing | 1289 |
| landing contract persisted pages | 0 |
| landing materialized pages | 0 |
| page_metadata_produced_pages | 1704 |
| page_metadata_received_pages | 1704 |
| page_metadata_plan_real_risk_pages | 116196 |
| page_metadata_plan_synthetic_pages | 0 |
| page_metadata_real_risk_coverage_ratio | 1.0 |
| max INT4 ratio | 0.9712 |
| mean INT4 ratio | 0.1473 |

日志检查：

```text
无 ERROR / Traceback / RuntimeError / EngineCore failed / metadata_wait_timeout
```

diagnosis：

| area | severity | signal |
|---|---:|---:|
| request_precision_budget | P1 | 8004 |
| sparse_window_quota | P1 | 2105 |
| frontier_dual_optimizer | P1 | 1112 |
| page_lifecycle | P1 | 32217 |

解释：

- 新代码下真实 P-side metadata 能稳定到 D-side，且 planner 走 real risk，不走 synthetic fallback。
- 这轮 admission trial 里仍有 `admission_planned_int4_landing_total=1289`，但没有真实持久化 landing contract，所以 `landing_contract_persisted_pages_total=0` 是合理的。
- 修复后的 diagnoser 没有把这个 admission trial 误判成 direct landing materialization failure。
- 当前剩余问题仍然是策略层：request budget、sparse quota、frontier optimizer 和 shared/open page lifecycle。

## 2.6 本轮下一步迭代：blocked-only diagnosis 和严重度自适应 pressure policy

上一轮 2+2 real smoke 的 diagnosis 仍然显示 `request_precision_budget=8004`、`sparse_window_quota=2105`。继续追日志后发现这里有一个统计口径问题：

```text
admission_frontier_rejection_reason_totals 统计了所有 admission_control event，
包括 admission_success_after_demote=True 的成功步骤。
```

成功步骤里的 frontier rejection 只是“候选池被裁剪过”，不等价于“admission 被它卡住”。所以这轮先把诊断口径改成 blocked-only：

```text
admission_blocked_frontier_rejection_reason_totals
```

实现：

- `summarize_reflex_pd_pressure.py` 新增 blocked-only frontier rejection 汇总。
- `run_pd_serving_mixed_accuracy.py` 因复用同一套 `_trace_stats()`，后续 mixed summary 会自动带上这个字段。
- `diagnose_reflex_blockers.py` 优先使用 blocked-only 字段；旧 summary 没有该字段时才回退到总量字段。

用上一轮 real smoke 重新解析后，口径变化如下：

| metric | all admission events | blocked admission only |
|---|---:|---:|
| shared_or_open | 32217 | 22153 |
| request_budget | 4002 | 41 |
| sparse_quota | 2105 | 4 |
| frontier_optimizer | 1112 | 13 |

这说明之前 request budget / sparse quota 的信号有相当部分是成功步骤中的正常裁剪，真正 blocked admission 的最大问题反而是 `shared_or_open`，也就是 page lifecycle / chunk sealing / shared prefix contract。

同时，pressure policy 从固定倍数放松改成按 funnel 严重程度自适应：

- request budget cap 严重时，`request_release_budget_multiplier` 不再固定 2x，而是按 `after_low_risk_filter / after_request_budget_cap` 的严重程度放大，最多 8x。
- sparse quota 严重时，`max_demote_per_window_multiplier` 不再固定 2x，而是按 `after_request_budget_cap / after_sparse_window_quota` 放大，最多 8x。
- frontier optimizer 严重时，release multiplier 最多放到 4x。

这不是取消保护，而是只在 admission pressure 且上一次 funnel 明确显示某一级过紧时，下一轮更积极释放。

新增/更新测试：

```text
tests/profiling/test_precision_kv_policy.py
tests/profiling/test_summarize_reflex_pd_pressure.py
tests/profiling/test_diagnose_reflex_blockers.py
tests/profiling/test_reflex_int4_scheduler.py
```

新 policy real smoke：

```text
outputs/accuracy/reflex_next_policy_2026-05-21/ablation_00_frontier_dual_reflex
```

配置：

```text
gov_report 2
Math500 2
max_concurrency 2
num_gpu_blocks_override 736
prefill GPU 1
decode GPU 3
frontier_dual_reflex
```

结果：

| metric | previous 2+2 | new policy 2+2 |
|---|---:|---:|
| completed | 4/4 | 4/4 |
| duration | 72.9859s | 61.1643s |
| gov_report score | 0.3973 | 0.3811 |
| Math500 score | 1.0 | 0.5 |
| demotion events | 27 | 31 |
| demoted pages | 1059 | 1031 |
| admission blocked | 32 | 24 |
| admission infeasible | 4 | 3 |
| admission success after demote | 169 | 217 |
| real risk coverage | 1.0 | 1.0 |
| synthetic pages | 0 | 0 |
| max INT4 ratio | 0.9712 | 0.9712 |
| mean INT4 ratio | 0.1473 | 0.2020 |

日志检查：

```text
无 ERROR / Traceback / RuntimeError / EngineCore failed / metadata_wait_timeout / CUDA OOM
无残留 vLLM/proxy serving 进程
```

blocked-only diagnosis：

| area | severity | signal |
|---|---:|---:|
| page_lifecycle | P1 | 16415 |
| request_precision_budget | P1 | 40 |
| frontier_dual_optimizer | P1 | 14 |
| sparse_window_quota | P1 | 6 |

结论：

- 这轮策略让小 mixed smoke 的 admission blocked / infeasible 降低，成功 demotion admission 增加，耗时下降。
- 2+2 样本太小，Math500 score 波动不能作为质量结论。
- 下一步真正该修的是 `shared_or_open` 的 page lifecycle：chunk sealing、open page 可见性、remote chunk page close/commit，以及 shared prefix copy-on-demote。

## 3. 与 `reflexkv.md` 目标的对照

`reflexkv.md` 的核心目标可以概括为：

```text
Precision-aware KV memory as a first-class serving resource.
```

也就是 ReFlexKV 不是单个 quantizer，而是一个闭环系统：

- 状态基座：precision-aware KV memory manager / page table / lifecycle / prefix state
- 决策平面：SLO-aware controller / online optimizer / risk estimator / quality debt
- 执行平面：migration engine / quantizer / dequantizer / recovery / mixed-precision attention
- serving 集成面：scheduler admission / P-D connector / telemetry / accounting

当前实现与目标的差距：

| 目标能力 | 当前状态 | 主要缺口 |
|---|---|---|
| precision-aware KV page table | 已有 BF16/INT4 page metadata、negative block id、recovery artifacts | 还不是完全独立的 VM-style page table |
| page lifecycle | 已有 BF16 active、INT4 active、landing、release pending、sealed/open/shared 过滤 | 状态机仍分散在 scheduler/cache manager/worker |
| decode dynamic demotion | 已实现并在真实压力下触发 | migration overlap 还不够强 |
| online optimizer | 已有 `RunCandidate`、frontier pruning、`frontier_dual` | 仍是轻量 primal-dual/heuristic 混合，不是成熟 solver |
| chunk/frontier admission | 已有 active chunk frontier、relaxed reserve、direct landing admission | scheduler 还没有完全改成 chunk-native 主循环 |
| risk estimator | 这轮修复后 compiled prefill 可以真实采集 page summary | estimator 本身仍简单，需要更强 semantic/position/task 特征 |
| INT4 quantizer | 已有独立 risk-aware quantizer 和 residual capsule | 还不是最终 CUDA kernel 级高性能实现 |
| recovery/compensation | 保留 BF16 shadow、cache manager recovery entry、budgeted background promotion 和单测 | 还缺 recovery on/off、shadow on/off、promotion 代价评估 |
| prefix precision contract | 已有 contract helper、page protection、copy-on-demote hook 和单测 | 尚未完全接入 prefix cache 生命周期，也缺 shared-prefix workload 验证 |
| P/D connector | Mooncake chunk metadata、direct landing、materialized landing req ids 已接入 | 还需要更强 telemetry 来量化每轮 P-side risk 到 D-side planner 的贡献 |
| paper-grade evaluation | 有 Math500/mixed pressure 初步结果 | 还缺完整 ablation matrix 和准确率/吞吐/blocked reason 系统对比 |

## 4. 当前模块划分

### 4.1 State substrate

主要文件：

- `vllm/vllm/v1/core/precision_kv/types.py`
- `vllm/vllm/v1/core/precision_kv/accounting.py`
- `vllm/vllm/v1/core/precision_kv/contracts.py`
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/vllm/v1/worker/block_table.py`

当前能力：

- 用 precision state 区分 BF16 / INT4 / recovery / landing 状态。
- 通过 negative block id 表达 INT4 block。
- scheduler 和 worker 都能识别 mixed precision block table。
- prefix precision contract 目前是控制平面雏形，初步支持 shared prefix protection/copy-on-demote 语义。

仍需改进：

- page lifecycle 需要变成更显式的状态机。
- prefix ownership 需要和 prefix cache 真正绑定，而不是只在 contract manager 中表达。

### 4.2 Decision plane

主要文件：

- `vllm/vllm/v1/core/precision_kv/controller.py`
- `vllm/vllm/v1/core/precision_kv/policy.py`
- `vllm/vllm/v1/core/precision_kv/frontier.py`
- `vllm/vllm/v1/core/precision_kv/run_optimizer.py`
- `vllm/vllm/v1/core/precision_kv/landing.py`
- `vllm/vllm/v1/core/precision_kv/risk.py`

当前能力：

- admission controller 能计算 requested release / feasible release / planned release。
- landing planner 能区分普通 BF16 fit、mixed landing、relaxed reserve landing。
- frontier 先构造 feasible candidates，再做 run-level selection。
- `frontier_dual` 维护简化 dual pressure，用 memory/admission/quality/backlog/SLO 信号给 candidate run 打分。
- risk helper 能从 P-side page score 推导 compressible pages 和 BF16 shadow pages。

这轮重要修复：

- direct remote chunk landing 在 P-side metadata 尚未到达时，可以用当前 chunk 合成保守候选页。
- direct chunk landing 可以 relax full-sequence reserve，只要求当前 chunk 的 needed deficit 可执行。
- compiled prefill metadata 不再丢失，为真实 P-side semantic risk 输入打通路径。

### 4.3 Execution plane

主要文件：

- `vllm/vllm/v1/attention/ops/int4_kv_cache.py`
- `vllm/vllm/v1/attention/ops/reflex_int4_codec.py`
- `vllm/vllm/v1/attention/ops/reflex_int4_kv_cache.py`
- `vllm/vllm/v1/attention/ops/reflex_quantizer.py`
- `vllm/vllm/v1/attention/backends/triton_attn.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`

当前能力：

- BF16 KV 可以被 materialize/demote 成 INT4 sidecar。
- decode attention 可以消费 mixed BF16/INT4 KV。
- risk-aware INT4 quantizer 已单独拆出，便于后续替换算法。
- recovery 当前只保留 BF16 shadow + budgeted background promotion，不再在热路径采集 decode relevance。
- BF16 shadow selection 现在优先选高风险 page，而不是任意 page。

仍需改进：

- quantizer 仍是研究/CPU-friendly 形态，不是最终 CUDA fused implementation。
- residual compensation 还没完全进入真实 serving hot path。
- promotion/recovery 需要真实质量评估和 recovery on/off 实验。

### 4.4 Serving integration plane

主要文件：

- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/worker/gpu_model_runner.py`
- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
- `vllm/examples/online_serving/disaggregated_serving/mooncake_connector/mooncake_connector_proxy.py`
- `scripts/profiling/run_reflex_pd_1p1d.py`
- `scripts/accuracy/run_pd_serving_mixed_accuracy.py`

当前能力：

- P-side prefill recorder 生成 page-level risk。
- Mooncake worker drain completed requests，把 risk metadata 合并进 connector metadata。
- D-side scheduler 从 `kv_transfer_params` 读取 `reflex_page_risks` 和 `reflex_compressible_pages`。
- direct INT4 landing 可以让部分 incoming remote chunk 不占用 BF16 staging block。
- telemetry 会记录 admission、landing、candidate rejection、INT4 ratio、waiting/running、demotion 等信息。

这轮修复后的关键变化：

```text
P-side recorder 可以在 compiled prefill 下继续工作，
因此 P/D connector 后续拿到的 page risk 不再依赖 eager-only 路径。
```

## 5. P2 细节：risk estimator、quantizer、recovery、prefix contract

### 5.1 P-side risk estimator

文件：

- `vllm/vllm/v1/core/precision_kv/risk.py`
- `vllm/vllm/v1/core/reflex_prefill_metadata.py`

当前实现口径：这里说的是代码路径和单测覆盖，不代表最新真实 workload 已经触发 recovery。

当前实现：

- `PageRiskSummary` 记录：
  - request id
  - page index
  - token range
  - risk score
  - semantic hash
  - compressible flag
- `ReflexPrefillMetadataRecorder` 在选定 attention layer 上记录：
  - query tail anchor
  - full query anchor fallback
  - 每个 page 的 key anchor
- `PrefillRiskEstimator` 用 normalized cosine risk 给 page 打分。
- 低风险页通过 `derive_compressible_pages_from_risks()` 选择。
- 高风险 BF16 shadow 通过 `select_bf16_shadow_pages()` 选择。

策略含义：

- prompt 页的 risk 来自 P-side prefill 的语义摘要；
- 短 prompt / reasoning prompt 可以被保护；
- long prompt 的低风险 page 可以作为 direct landing 或 demotion 候选；
- 高风险 page 可以保留 BF16 shadow，后续 recovery 使用。

当前局限：

- risk score 仍是 anchor-level 近似，不是完整 token-level semantic dependency。
- decode-generated pages 主要还是 age/recent/budget 策略，没有真正 semantic estimator。
- 需要加入更细的特征：position、layer variance、query entropy、task type。

### 5.2 Risk-aware INT4 quantizer

文件：

- `vllm/vllm/v1/attention/ops/reflex_quantizer.py`

当前实现：

- groupwise symmetric INT4 quantization；
- 支持 group size；
- 支持 high-risk tensor 的 residual capsule；
- dequantize 时可以把 residual 补回；
- 单测证明 high-risk residual compensation 降低 reconstruction error。

策略含义：

- 不同风险页不应只用同一个量化策略；
- 高风险 page 可以保留少量 residual；
- 后续可以把 residual budget 纳入 optimizer 的 quality/cost tradeoff。

当前局限：

- 还不是 CUDA fused kernel；
- residual capsule 还没有完全接入真实 serving 的 sidecar format；
- 没有完成 per-head/per-layer/group-size 的 ablation。

### 5.3 Recovery / compensation

文件：

- `vllm/vllm/v1/core/sched/scheduler.py`
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`

当前实现：

- page metadata 可以标记 BF16 shadow / recoverable page；
- cache manager 保留 `recover_reflex_int4_pages()` 和 `promote_reflex_recoverable_pages()`；
- scheduler 只保留 budgeted background promotion；
- 实时 decode attention mass 采集、按注意力触发恢复、synthetic relevance fallback 已删除。

策略含义：

- 不把高开销、不稳定的 attention 监控放进热路径；
- 保留可控的后台 promotion，用 free ratio、waiting queue、request 剩余 decode token 和每步页数预算控制；
- 后续真正要证明 recovery，需要单独做 background promotion on/off 和 shadow selection on/off，而不是把结果归因给已经删除的实时检测路径。

当前局限：

- 还缺真实 workload 下 recovery on/off 的准确率对比；
- 最新 smoke 中 recovery 相关计数全为 0，说明当前实验还没有证明恢复路径的实际收益；
- promotion 的 GPU time 和 queue impact 需要单独统计。

### 5.4 Prefix precision contract

文件：

- `vllm/vllm/v1/core/precision_kv/contracts.py`

当前实现口径：这里说的是 contract/helper 能表达 prefix precision 语义，不代表已经完成 vLLM prefix cache 的多版本精度系统。

当前实现：

- `PrefixPrecisionContractManager`
- shared prefix version tracking
- `requires_copy_on_demote()`
- `copy_on_demote()`
- per-request active prefix version
- precision ownership

策略含义：

- shared prefix 不能被某个 request 原地 demote 破坏；
- 需要 copy-on-demote，给该 request 生成 INT4 version；
- 原 owner / 其他 request 继续看到 BF16 version；
- 这是 prefix precision cache 的基础。

当前局限：

- 还没完全接入 prefix cache allocation/reuse path；
- multi-version 生命周期和 eviction policy 还没完成；
- prefix page 的 accounting 还需要进入 telemetry。
- 最新 smoke 没有 shared-prefix workload，`candidate_shared_bf16_pages_total=0`、`candidate_copy_on_demote_pages_total=0`，所以 prefix 路径尚未被真实验证。

## 6. Admission / landing / scheduler 的当前策略

### 6.1 普通 landing、mixed landing、demotion 的区别

普通 landing：

```text
新请求 incoming KV 全部以 BF16 放入 decode cache。
```

mixed landing：

```text
新请求 incoming KV 的一部分直接进入 BF16，另一部分直接进入 INT4 sidecar。
```

demotion：

```text
已经在 decode cache 里的 BF16 page 后续被量化搬到 INT4，并释放 BF16 block。
```

三者触发时机不同：

- 普通 landing：BF16 block 足够，或者没有启用 direct INT4 landing。
- mixed landing：admission 阶段发现 BF16 不够，但当前 remote chunk 有可直接 INT4 landing 的 page。
- demotion：decode cache 已经有压力，需要从 running requests 中释放 BF16 block。

### 6.2 Chunk/frontier admission

当前已经实现的策略：

- remote chunk pages 可以在 chunk sealed/full 后进入候选，不必等完整 prompt。
- admission target 以 active chunk frontier 为主，而不是无脑 full sequence reserve。
- direct landing 可以 relax reserve：

```text
只要 current chunk 的 needed BF16 deficit 能被 mixed landing 覆盖，
就不要求同时满足 full-sequence reserve。
```

这避免了之前的问题：

```text
requested_release = needed + reserve
direct landing 明明能覆盖当前 chunk，
但因为 cover 不住 reserve，admission 仍被判不可行。
```

### 6.3 当前 rejection reason

现在能看到更细的阻塞原因：

- `shared_or_open`
- `recent_or_initial`
- `high_risk`
- `request_fraction_cap`
- `quality_debt_cap`
- `request_release_budget`
- `short_decode_protection`
- `reasoning_prompt_protection`
- `request_budget`
- `sparse_quota`
- `frontier_optimizer`
- `int4_pool_full`

这个 breakdown 很重要，因为下一步调策略时不能再只看一个总的 `request_budget`。

## 7. 最新验证结果

### 7.1 单元/回归测试

本轮 1-4 定向验证：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_summarize_reflex_pd_pressure.py \
  tests/profiling/test_diagnose_reflex_blockers.py \
  tests/accuracy/test_reflex_ablation_matrix.py \
  tests/accuracy/test_generate_reflex_workloads.py \
  tests/accuracy/test_pd_serving_mixed_accuracy.py -q
```

结果：

```text
18 passed, 1 warning in 0.39s
```

相关 serving/scheduler 回归：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_reflex_int4_scheduler.py \
  tests/profiling/test_reflex_mooncake_connector.py \
  tests/profiling/test_reflex_pd_1p1d_runner.py \
  tests/accuracy/test_pd_serving_mixed_accuracy.py \
  tests/accuracy/test_pd_serving_accuracy.py -q
```

结果：

```text
136 passed, 17 warnings in 6.76s
```

静态编译检查：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ python -m py_compile \
  scripts/profiling/summarize_reflex_pd_pressure.py \
  scripts/profiling/diagnose_reflex_blockers.py \
  scripts/accuracy/run_reflex_ablation_matrix.py \
  scripts/accuracy/generate_reflex_workloads.py \
  scripts/profiling/run_reflex_pd_1p1d.py \
  scripts/accuracy/run_pd_serving_accuracy.py \
  scripts/accuracy/run_pd_serving_mixed_accuracy.py
```

结果：

```text
passed
```

命令：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ytm/code/quant/SemantiQ/vllm:/home/ytm/code/quant/SemantiQ pytest tests/profiling -q
```

结果：

```text
295 passed, 16 warnings in 8.03s
```

命令：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ/vllm:/home/ytm/code/quant/SemantiQ pytest tests/accuracy -q
```

结果：

```text
24 passed, 1 warning in 0.32s
```

针对本轮修复的定向测试：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ/vllm:/home/ytm/code/quant/SemantiQ pytest tests/profiling/test_reflex_prefill_metadata.py tests/profiling/test_precision_kv_p2.py tests/profiling/test_reflex_int4_scheduler.py::test_reflex_int4_direct_remote_chunk_landing_synthesizes_current_chunk_pages tests/profiling/test_reflex_mooncake_connector.py::test_mooncake_worker_metadata_merges_materialized_landing_reqs -q
```

结果：

```text
13 passed
```

本次中断恢复后的定向回归：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_diagnose_reflex_blockers.py \
  tests/profiling/test_summarize_reflex_pd_pressure.py -q
```

结果：

```text
7 passed
```

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_mooncake_proxy_state.py \
  tests/profiling/test_reflex_pd_1p1d_runner.py::test_1p1d_env_and_proxy_benchmark_target_the_expected_processes \
  tests/profiling/test_reflex_pd_1p1d_runner.py::test_1p1d_proxy_metadata_wait_defaults_to_zero_without_reflex_metadata \
  tests/accuracy/test_reflex_ablation_matrix.py::test_ablation_command_contains_mixed_accuracy_args_and_case_flags \
  tests/accuracy/test_generate_reflex_workloads.py -q
```

结果：

```text
11 passed, 1 warning
```

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_reflex_int4_scheduler.py -q
```

结果：

```text
99 passed, 16 warnings
```

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/accuracy/test_pd_serving_mixed_accuracy.py -q
```

结果：

```text
7 passed, 1 warning
```

静态检查也通过：

```bash
python -m py_compile \
  scripts/profiling/summarize_reflex_pd_pressure.py \
  scripts/profiling/diagnose_reflex_blockers.py \
  vllm/vllm/v1/core/sched/scheduler.py
```

### 7.2 本轮工具 smoke

Ablation matrix dry-run：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ python \
  scripts/accuracy/run_reflex_ablation_matrix.py \
  --limit 2 \
  --dry-run \
  --commands-out /tmp/reflex_ablation_commands.jsonl \
  --output-root outputs/accuracy/reflex_ablation_test \
  --longbench-max-samples 2 \
  --reasoning-max-samples 2
```

结果：

```text
输出 bf16_baseline 和 heuristic_reflex 两条完整 serving 命令；
/tmp/reflex_ablation_commands.jsonl 写入成功。
```

旧 mixed run 的 blocker diagnosis：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ python \
  scripts/profiling/diagnose_reflex_blockers.py \
  outputs/accuracy/reflexkv_next_2026-05-21/mixed_gov50_math50_c16_b736_chunk_target_gpu01/mixed_summary.json
```

输出主因：

| area | severity | signal |
|---|---:|---:|
| chunk_admission | P0 | 1400 |
| request_precision_budget | P1 | 247357 |
| sparse_window_quota | P1 | 23038 |
| frontier_dual_optimizer | P1 | 2151 |
| page_lifecycle | P1 | 434242 |

解释：

- 旧 run 没有本轮新增的 metadata produce/receive/plan trace，所以 `p_side_risk_metadata` 诊断不会在旧 summary 中触发。
- 新 run 的 `mixed_summary.json` 会包含 real risk coverage、synthetic fallback、landing materialization ratio 等字段。

### 7.3 本轮 1P1D smoke

运行目录：

```text
outputs/profiling/reflex_p2_rootfix/20260521-160905_p2_compile_metadata_smoke_i4096_o32_n8_c4
```

命令核心参数：

```text
prefill GPU 0
decode GPU 1
input len 4096
output len 32
num prompts 8
max concurrency 4
remote chunk tokens 512
decode cache dtype reflex_int4
page selection frontier_dual
force triton attention
```

结果：

| metric | value |
|---|---:|
| completed | 8 |
| failed | 0 |
| duration | 17.111s |
| request throughput | 0.468 req/s |
| output throughput | 14.961 tok/s |
| total token throughput | 1929.491 tok/s |
| mean TTFT | 7211.33 ms |
| median TTFT | 6027.04 ms |
| mean TPOT | 19.32 ms |
| p95 TPOT | 21.26 ms |

日志检查：

```text
没有 ERROR / Traceback / RuntimeError / EngineCore failed
没有 Unsupported context manager / threading.Lock 编译错误
```

decode trace 中仍能看到：

```text
landing_eligible_int4_blocks=32
landing_reason=mixed_landing_feasible
landing_reason=mixed_landing_relaxed_reserve_feasible
```

解释：

- 本轮修复确认 compiled prefill metadata path 不再导致 engine 崩溃。
- 小 smoke 主要验证系统可运行，不作为吞吐收益结论。
- direct/mixed landing 的真实收益仍要靠 n=100 和 mixed workload 压力实验确认。

## 8. 之前真实压力实验结果

### 8.1 Math500-only n=100

目录：

```text
outputs/accuracy/reflexkv_next_2026-05-21/math500_n100_c16_b736_chunk_target_gpu01
```

结果：

| metric | value |
|---|---:|
| requests | 100/100 completed |
| duration | 207.5006s |
| throughput | 0.482 req/s |
| Math500 score | 0.53 |
| avg latency | 19.90s |
| p95 latency | 101.04s |
| decode max KV | 100.0% |
| decode avg running | 10.59 |
| decode avg waiting | 0.079 |
| demotion events | 395 |
| demoted pages | 1535 |
| admission infeasible | 3 |
| admission success after demote | 392 |
| max INT4 ratio | 0.6562 |
| mean INT4 ratio | 0.2523 |

结论：

- Math500 prompt 短，主要压力来自 decode 阶段。
- ReFlexKV 的 decode-side dynamic demotion 已经能触发。
- 短 prompt 保护是合理的，不能为了压缩率破坏 reasoning prompt。

### 8.2 mixed 50 gov_report + 50 Math500

目录：

```text
outputs/accuracy/reflexkv_next_2026-05-21/mixed_gov50_math50_c16_b736_chunk_target_gpu01
```

结果：

| metric | value |
|---|---:|
| requests | 100/100 completed |
| duration | 315.2172s |
| throughput | 0.317 req/s |
| Math500 score | 0.48 |
| gov_report score | 0.3308 |
| decode max KV | 100.0% |
| decode avg KV | 83.95% |
| decode max running | 16 |
| decode avg running | 9.14 |
| decode max waiting | 15 |
| decode avg waiting | 3.49 |
| demotion events | 1183 |
| demoted pages | 22714 |
| admission infeasible | 921 |
| admission success after demote | 696 |
| max INT4 ratio | 0.9103 |
| mean INT4 ratio | 0.4890 |

candidate rejection totals：

| reason | count |
|---|---:|
| shared_or_open | 333837 |
| recent_or_initial | 299488 |
| request_budget | 378060 |
| sparse_quota | 8049 |
| frontier_optimizer | 62719 |
| int4_pool_full | 0 |

结论：

- ReFlexKV 在 mixed pressure 下能完成 100/100。
- INT4 pool 不再是主要瓶颈。
- 主要问题变成 admission policy、request budget、sparse quota、direct landing materialization 效果。

### 8.3 mixed 10 + 10 remaining-capacity budget fix

before：

```text
outputs/accuracy/reflexkv_next_2026-05-21/mixed_gov10_math10_c16_b736_chunk_rebudget_gpu01
```

after：

```text
outputs/accuracy/reflexkv_next_2026-05-21/mixed_gov10_math10_c16_b736_remaining_capacity_gpu01
```

对比：

| metric | before | after |
|---|---:|---:|
| requests completed | 20/20 | 20/20 |
| duration | 131.0506s | 75.1186s |
| Math500 score | 0.40 | 0.50 |
| gov_report score | 0.2621 | 0.3228 |
| decode avg waiting | 3.27 | 6.29 |
| candidate request_budget rejection | 118256 | 28155 |
| admission blocked | 419 | 346 |
| admission infeasible | 419 | 345 |
| demoted pages | 3906 | 3973 |
| mean INT4 ratio | 0.2441 | 0.4925 |

结论：

- remaining-capacity release budget 是有效修复。
- 它减少了已经达到 INT4 cap 的 request 继续拿 release budget 的错误。
- duration 明显下降。
- waiting 均值上升但总时长下降，说明单看 waiting 不够，需要 queue time / phase time breakdown。

## 9. 当前可以确认的效果

已经确认：

- decode-side dynamic demotion 在真实压力下工作。
- mixed workload 可以跑完，不再卡死在 admission 零进展。
- INT4 pool rebudget 修复后，`int4_pool_full` 不再是主要瓶颈。
- remote chunk sealed pages 可以提前进入候选。
- direct chunk landing 可以 relax reserve，缓解 full-sequence reserve 过保守。
- P2 模块已经从接口级原型推进到独立文件和可测试实现。
- 这轮修复后，compiled prefill path 不再丢掉 page risk metadata。
- 1P1D smoke 没有再出现 TorchDynamo tracing recorder lock 的严重 bug。

不能过度声明：

- 还没有证明 risk estimator 明显提升准确率。
- 还没有证明 residual compensation 在真实 serving hot path 下提升质量。
- 还没有证明 recovery/promotion 的质量收益大于迁移成本。
- 还没有证明 direct landing 在 n=100 mixed pressure 下稳定降低 TTFT/queue time。
- 还没有完成 paper-grade ablation。

## 10. 当前主要问题

### 10.1 optimizer 仍不够强

现在的 `frontier_dual` 已经比纯 heuristic 更合理，但仍然偏局部：

- target release 多来自当前 admission pressure；
- quality debt 还不够真实；
- migration backlog price 还没有和 GPU time 强绑定；
- SLO risk 仍是粗粒度；
- 缺少对未来 waiting queue 和 chunk arrival 的预测。

下一步应该把 optimizer 的输入从 page/run 静态属性扩展到：

- active chunk frontier；
- future chunk arrival；
- request remaining decode budget；
- per-request quality debt；
- migration queue latency；
- observed promotion cost / recovery benefit。

### 10.2 scheduler 仍不是完全 chunk-native

现在是把 chunk frontier 和 relaxed reserve 接进了原 scheduler path，但理想状态应该是：

```text
prefill/decode 都以 chunk 为 admission 单元，
每个 chunk 明确选择 BF16 landing / INT4 landing / defer / partial admission。
```

仍需重构：

- waiting request 的 chunk progress state；
- partial admission；
- chunk-level reserve；
- chunk-level direct landing materialization accounting；
- request-level full sequence reserve 的降级路径。

### 10.3 P-side risk telemetry 已有基础，但还需要真实 run 验证

本轮已经补齐基础可观测性：

- 每个 request 产生了多少 page risk；
- P-side risk metadata 何时到 D-side；
- D-side 使用了真实 risk 还是 synthetic fallback；
- direct/mixed landing 的 real risk、explicit compressible、synthetic pages 分解；
- planned landing 与 materialized landing 的差距。

仍需通过真实 n=100/mixed run 确认：

- compiled prefill 下 `page_metadata_produce/receive/plan` 是否稳定非零；
- real risk coverage 是否足够高；
- synthetic fallback 是否只是 metadata race fallback，而不是长期主路径；
- BF16 shadow pages 是否实际被 recovery 使用。

### 10.4 quantizer 还不是最终算法

当前 quantizer 适合研究验证，但论文级还需要：

- per-layer/per-head group size；
- activation-aware scale；
- outlier channel handling；
- residual budget optimizer；
- CUDA kernel 级实现；
- 和 attention kernel 的 packed layout 一致化。

### 10.5 prefix precision contract 还没完全落地

当前 contract manager 能表达 copy-on-demote，但还需要：

- prefix cache 多版本索引；
- prefix page eviction；
- per-version accounting；
- shared prefix direct landing policy；
- contract violation telemetry。

## 11. 下一步实验矩阵

必须系统跑以下 ablation：

| 实验 | 目的 |
|---|---|
| BF16 baseline | 确定无压缩吞吐/准确率基线 |
| naive oldest demotion | 和最简单策略对比 |
| heuristic ReFlexKV | 与当前非 dual 策略对比 |
| frontier_dual ReFlexKV | 验证 optimizer 收益 |
| direct landing on/off | 验证 mixed landing 是否减少 BF16 admission pressure |
| P-side risk on/off | 验证 semantic risk 是否真的影响候选选择 |
| synthetic fallback on/off | 验证 metadata race fallback 的必要性 |
| recovery on/off | 验证 promotion 是否改善准确率 |
| residual compensation on/off | 验证 quantizer 质量收益 |
| quantizer group-size ablation | 找到吞吐/质量平衡点 |
| prefix contract on/off | 验证 shared prefix 场景安全性 |

每个实验至少报告：

- completed / failed；
- throughput；
- TTFT / TPOT / p50 / p95 / p99 latency；
- per-dataset accuracy；
- decode/prefill running；
- decode/prefill waiting；
- KV usage；
- INT4 ratio；
- demotion event count；
- demoted pages；
- direct landing planned/materialized pages；
- recovery/promoted pages；
- blocked reason breakdown；
- migration GPU time；
- quantization GPU time；
- page risk coverage；
- fallback-vs-real-risk ratio。

## 12. 下一轮建议

本轮 1-4 已经实现，下一步优先级应该从“继续补工具”切到“跑固定矩阵并按诊断修策略”。

1. 跑固定 workload 的小矩阵。
   - 先用 `generate_reflex_workloads.py` 固定 gov_report + Math500 请求集。
   - 用 `run_reflex_ablation_matrix.py --dry-run` 检查命令。
   - 再在空闲卡上跑 `bf16_baseline`、`heuristic_reflex`、`frontier_dual_reflex`、`direct_landing_on/off`、`p_side_risk_on/off`。

2. 对每个 run 立刻跑 diagnosis。
   - 如果 `direct_landing_materialization` 是 P0，先修 materialize/commit path。
   - 如果 `p_side_risk_metadata` 是 P0，先修 P/D metadata 传输或 planner 使用路径。
   - 如果 `chunk_admission` 仍是 P0，再继续推进 partial/chunk-native admission。
   - 如果主要是 `request_precision_budget` / `sparse_window_quota`，再调层次化 page 等级和 per-window quota。

3. 根据矩阵结果改 scheduler / optimizer。
   - 当前 blocker diagnosis 已经能把 full-sequence reserve、request budget、sparse quota、frontier optimizer、shared/open pages 拆开。
   - 下一步不要盲目改所有策略，而是按最大 P0/P1 signal 排序修。

4. 继续推进 quantizer/recovery。
   - 把 residual compensation 接入真实 INT4 sidecar。
   - 加 recovery on/off accuracy eval。
   - 加 promotion cost telemetry。

## 13. 常用命令

完整 profiling 回归：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/ytm/code/quant/SemantiQ/vllm:/home/ytm/code/quant/SemantiQ pytest tests/profiling -q
```

accuracy 回归：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ/vllm:/home/ytm/code/quant/SemantiQ pytest tests/accuracy -q
```

生成固定 mixed workload：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ python \
  scripts/accuracy/generate_reflex_workloads.py \
  --output outputs/accuracy/reflex_workloads/gov50_math50.jsonl \
  --summary-out outputs/accuracy/reflex_workloads/gov50_math50_summary.json \
  --tasks longbench,reasoning \
  --longbench-datasets gov_report \
  --reasoning-datasets math500 \
  --longbench-max-samples 50 \
  --reasoning-max-samples 50 \
  --workload-mix-policy balanced
```

Ablation matrix dry-run：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ python \
  scripts/accuracy/run_reflex_ablation_matrix.py \
  --dry-run \
  --commands-out outputs/accuracy/reflex_ablation_commands.jsonl \
  --output-root outputs/accuracy/reflex_ablation_matrix \
  --longbench-max-samples 50 \
  --reasoning-max-samples 50
```

诊断 mixed/profiling summary：

```bash
PYTHONPATH=/home/ytm/code/quant/SemantiQ python \
  scripts/profiling/diagnose_reflex_blockers.py \
  outputs/accuracy/reflexkv_next_2026-05-21/mixed_gov50_math50_c16_b736_chunk_target_gpu01/mixed_summary.json
```

小 1P1D smoke：

```bash
python scripts/profiling/run_reflex_pd_1p1d.py \
  --prefill-gpu 0 \
  --decode-gpu 1 \
  --prefill-port 8650 \
  --decode-port 8750 \
  --proxy-port 8850 \
  --prefill-bootstrap-port 8950 \
  --run-name p2_compile_metadata_smoke_i4096_o32_n8_c4 \
  --output-root outputs/profiling/reflex_p2_rootfix \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 4096 \
  --num-gpu-blocks-override 736 \
  --input-len 4096 \
  --output-len 32 \
  --num-prompts 8 \
  --max-concurrency 4 \
  --proxy-prefill-max-inflight 4 \
  --reflex-remote-chunk-tokens 512 \
  --reflex-page-selection-policy frontier_dual \
  --enable-reflex-trace \
  --force-triton-attn
```

检查 smoke 错误：

```bash
rg -n "ERROR|Traceback|RuntimeError|EngineCore failed|Unsupported context manager|threading\\.Lock" \
  outputs/profiling/reflex_p2_rootfix/20260521-160905_p2_compile_metadata_smoke_i4096_o32_n8_c4
```

检查 GPU：

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
```

## 14. Bottom line

这轮最关键的进展是：

```text
P2 的 P-side semantic risk path 不再只是 eager-only 或接口级功能。
它现在可以在 compiled prefill 下通过 custom op side-channel 真实记录 page metadata。
同时，实验闭环也补齐了 telemetry 汇总、ablation matrix、blocked reason diagnosis 和固定 workload 生成。
```

这解决了后续 estimator、direct landing、BF16 shadow、recovery/promotion 的基础数据来源问题。

但是 ReFlexKV 仍然不能说已经完成论文目标。下一步必须用 n=100 和 mixed workload 做系统矩阵，明确证明：

- direct landing 是否减少 BF16 admission pressure；
- real P-side risk 是否优于 synthetic fallback；
- frontier-dual 是否优于 heuristic；
- residual/recovery 是否能在高 INT4 ratio 下保护准确率；
- prefix precision contract 是否能在 shared prefix 场景下保证安全。

## 15. Page lifecycle iteration: chunk sealing / open visibility / remote commit / copy-on-demote

本轮集中修了 page lifecycle 的四个薄弱点。目标不是继续调阈值，而是把 planner 看到的 page 状态从一个模糊的 `shared_or_open` 拆成可解释的生命周期信号：

- chunk sealing：remote chunk 已经 closed/committed 的页可以作为 sealed frontier 进入 demotion planner；
- open page visibility：candidate breakdown 里单独记录 open BF16 页、shared BF16 页、copy-on-demote 页；
- remote chunk close/commit：D 端收到 remote chunk 后显式写 committed frontier，in-flight chunk 的 `page_end` 不再被误当成 sealed；
- shared prefix copy-on-demote：允许有明确 contract 的 shared prefix 页走 copy-on-demote， demote 当前 request 的 block-table entry，不原地破坏其它 request 的 shared BF16 prefix。

### 15.1 实现细节

改动位置：

- `vllm/vllm/v1/core/precision_kv/types.py`
  - `ReflexPageMeta` 增加 `copy_on_demote`。
  - `ReflexDemotion` 增加 `copy_on_demote`。

- `vllm/vllm/v1/core/precision_kv/demotion_planner.py`
  - `ReflexCandidateBreakdown` 增加：
    - `open_bf16_pages`
    - `shared_bf16_pages`
    - `copy_on_demote_pages`
  - planner 的候选条件从：
    - sealed/full
    - unshared
    - not prompt protected
    改成：
    - sealed/full
    - unshared 或者 explicit `copy_on_demote`
    - not prompt protected
  - `ReflexDemotion` 现在保留 `copy_on_demote`，用于日志和后续 execution/recovery 语义。

- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
  - `plan_reflex_int4_demotions()` 新增：
    - `sealed_pages_by_request`
    - `copy_on_demote_pages_by_request`
  - `_build_reflex_page_metadata()` 现在把 `computed_tokens // block_size` 和 explicit sealed frontier 合并：
    - ordinary decode 用 computed tokens sealing；
    - remote chunk 用 committed sealed pages override；
    - prompt 未完成但 explicit chunk committed 时，不再把整段都标成 open。
  - shared BF16 页如果有 copy-on-demote contract，可以进入候选；执行时只替换当前 request 的 encoded block table，并把 BF16 block 放入 pending release。`BlockPool.free_blocks()` 会递减 ref_cnt，因此不会破坏其它 request 的 shared prefix。

- `vllm/vllm/v1/core/sched/scheduler.py`
  - `_reflex_remote_chunk_sealed_pages()` 新增 committed frontier 语义。
  - `_commit_reflex_remote_chunk()` 在 D 端 remote KV recv 完成后写：
    - `reflex_remote_chunk_committed_token_end`
    - `reflex_remote_chunk_committed_page_end`
    - `reflex_remote_chunk_inflight=False`
  - in-flight chunk 的 `reflex_remote_chunk_page_end` 不再直接暴露给 planner；只有 committed frontier 可见。
  - `_build_reflex_int4_demotion_planning_kwargs()` 现在把 `sealed_pages_by_request` 和 `copy_on_demote_pages_by_request` 传到 KV manager。
  - candidate breakdown trace 新增 open/shared/copy 三个字段。

- `vllm/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py`
  - D-side remote prefill chunk allocation 后写 `reflex_remote_chunk_inflight=True`。
  - scheduler 收到 worker `finished_recving` 后才 commit chunk frontier。

- `scripts/profiling/summarize_reflex_pd_pressure.py`
  - summary 新增：
    - `candidate_open_bf16_pages_total`
    - `candidate_shared_bf16_pages_total`
    - `candidate_copy_on_demote_pages_total`

### 15.2 新增回归测试

新增/扩展测试覆盖：

- `test_manager_uses_remote_chunk_sealed_page_frontier`
  - 验证 computed tokens 仍为 0 时，explicit sealed pages 可以让 remote chunk pages 进入 demotion planner。

- `test_manager_copy_on_demote_shared_page_releases_only_request_ref`
  - 验证 shared BF16 页在 explicit copy-on-demote contract 下可以 demote；
  - demotion 后当前 request 切到 INT4；
  - `new_step_starts()` 只把 shared block 的 ref_cnt 从 2 降到 1，不释放其它 request 的 prefix。

- `test_reflex_int4_planning_passes_sealed_chunk_and_copy_on_demote_pages`
  - 验证 scheduler 把 remote chunk sealed frontier 和 copy-on-demote page set 传到 planner。

- `test_reflex_remote_chunk_commit_frontier_ignores_inflight_page_end`
  - 验证 inflight chunk 的 `page_end` 不算 sealed；
  - `_commit_reflex_remote_chunk()` 后才暴露 committed page frontier；
  - 下一块 chunk inflight 时，sealed frontier 仍停在上一个 committed page end。

### 15.3 验证结果

单元/编译：

```text
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_reflex_int4_block_table.py \
  tests/profiling/test_reflex_int4_scheduler.py \
  tests/profiling/test_reflex_int4_pool.py \
  tests/profiling/test_precision_kv_run_optimizer.py \
  tests/profiling/test_precision_kv_frontier.py \
  tests/profiling/test_summarize_reflex_pd_pressure.py -q

176 passed, 16 warnings
```

`py_compile` 覆盖了修改过的 scheduler、KV manager、planner、types、Mooncake connector、summary parser，全部通过。

Mooncake connector async 单测尝试运行过，但当前环境没有 `pytest-asyncio`，失败原因是 pytest 无法原生执行 async test，不是代码断言失败。

### 15.4 真实 smoke 结果

最终 smoke：

```text
outputs/accuracy/reflex_next_lifecycle_commit_2026-05-21/ablation_00_frontier_dual_reflex
```

配置：

- prefill GPU 1
- decode GPU 3
- `frontier_dual_reflex`
- gov_report 2 samples
- Math500 2 samples
- max concurrency 2
- `num_gpu_blocks_override=736`
- remote chunk tokens 512

结果：

| dataset | completed | failed | avg latency | avg score |
|---|---:|---:|---:|---:|
| gov_report | 2 | 0 | 19.80s | 0.3901 |
| math500 | 2 | 0 | 15.43s | 0.5 |

关键 ReFlexKV trace：

| metric | value |
|---|---:|
| demotion_event_count | 34 |
| demoted_pages_total | 1033 |
| actual_release_blocks_total | 1033 |
| remote_chunk_send events | 59 |
| remote_chunk_commit events | 55 |
| max committed page end | 1012 |
| max committed token end | 16183 |
| candidate_breakdown_event_count | 104 |
| candidate_raw_bf16_pages_total | 74521 |
| candidate_open_bf16_pages_total | 8119 |
| candidate_shared_bf16_pages_total | 0 |
| candidate_copy_on_demote_pages_total | 0 |
| candidate_eligible_full_unshared_pages_total | 65442 |
| candidate_selected_actual_total | 1033 |
| max decode KV usage | 100% |
| max decode running | 2 |
| max decode waiting | 1 |

解释：

- `remote_chunk_commit=55` 说明 D 端 chunk close/commit path 已经实际触发，不只是单元测试接口。
- `max committed page end=1012` 对应第二个 gov_report 长 prompt，说明长 prompt remote chunks 的 committed frontier 能推进到完整 prompt page 范围。
- `candidate_open_bf16_pages_total=8119` 仍然存在，但现在它不是“已传完却没 sealed”的同一种问题；它主要包含 in-flight chunk、decode open tail 和受保护状态下的页。
- `candidate_shared_bf16_pages_total=0` 和 `candidate_copy_on_demote_pages_total=0` 是预期的，因为本轮 smoke 命令使用 `--no-enable-prefix-caching`，没有真实 shared prefix workload。copy-on-demote 目前由单元测试覆盖，后续需要打开 prefix caching 或构造 shared-prefix workload 做真实验证。

诊断输出：

```json
[
  {"area": "request_precision_budget", "severity": "P1", "signal": 40},
  {"area": "sparse_window_quota", "severity": "P1", "signal": 6},
  {"area": "frontier_dual_optimizer", "severity": "P1", "signal": 17},
  {"area": "page_lifecycle", "severity": "P1", "signal": 13658}
]
```

这里的 page lifecycle signal 仍然大，但 breakdown 已经能继续下钻：

- open BF16：8119；
- shared BF16：0；
- copy-on-demote：0；
- remaining `shared_or_open - open - shared` 主要来自 prompt protection / protected request 状态，需要下一轮继续拆出 `prompt_protected_bf16_pages` 和 `remote_inflight_pages`，避免诊断字段继续混淆。

### 15.5 当前结论

这轮已经把 page lifecycle 的关键安全边界补上了：

- in-flight remote chunk 不会被提前当作 sealed；
- committed chunk 才能暴露给 demotion planner；
- open/shared/copy page 有了独立 telemetry；
- shared prefix copy-on-demote 不再需要破坏 shared BF16 block。

但这不是最终策略优化。真实 smoke 显示下一轮重点应该是：

1. 把 `shared_or_open` 继续拆成 `open_tail`、`remote_inflight`、`prompt_protected`、`shared_blocked`。
2. 构造 prefix-cache shared workload，真实验证 copy-on-demote 是否能安全释放最后一个 BF16 ref。
3. 继续修 request budget / quality debt / sparse quota；现在它们才是 admission frontier 的主要策略瓶颈。
4. 对 background pressure 做更低频或 event-driven planning；当前 104 次 candidate breakdown 中大量是 background_pressure 重复 dry-run，会增加调度损耗。

## 16. Lifecycle blocker breakdown iteration

这一轮目标不是继续盲调 demotion 阈值，而是把上一轮仍然很大的 `shared_or_open` 诊断信号拆开。之前 `shared_or_open` 同时混了几类完全不同的问题：

- remote chunk 还在传输中的 open page；
- decode 当前 chunk 的 open tail；
- whole-request protected 状态；
- prompt protected page；
- shared prefix page；
- copy-on-demote 可处理的 shared page。

这些状态如果继续聚合在一个字段里，后续策略会误判：例如把 request protection 当成 shared prefix 问题，或者把 in-flight remote chunk 当成 sparse quota 问题。

### 16.1 实现内容

新增 page lifecycle 元数据：

- `ReflexPageMeta.is_remote_inflight`
- `ReflexPageMeta.is_request_protected`

新增候选池 breakdown 字段：

- `remote_inflight_bf16_pages`
- `open_tail_bf16_pages`
- `request_protected_bf16_pages`
- `prompt_protected_bf16_pages`
- 继续保留 `open_bf16_pages`、`shared_bf16_pages`、`copy_on_demote_pages`、`eligible_full_unshared_pages`

修改后的含义：

- `open_bf16_pages`：所有非 full 的 raw BF16 page 总量；
- `remote_inflight_bf16_pages`：还在 remote chunk inflight 区间，不能 demote；
- `open_tail_bf16_pages`：本地 decode/prefill 尾部还没 closed/sealed 的 page；
- `request_protected_bf16_pages`：整个 request 暂时被 protection contract 挡住；
- `prompt_protected_bf16_pages`：full page，但因为 prompt protection / risk contract 被挡住；
- `shared_bf16_pages`：shared prefix 且没有 copy-on-demote contract；
- `copy_on_demote_pages`：shared prefix 但 contract 允许 copy-on-demote 的 page。

修改路径：

- `vllm/vllm/v1/core/precision_kv/types.py`
- `vllm/vllm/v1/core/precision_kv/demotion_planner.py`
- `vllm/vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/vllm/v1/core/kv_cache_coordinator.py`
- `vllm/vllm/v1/core/kv_cache_manager.py`
- `vllm/vllm/v1/core/sched/scheduler.py`
- `scripts/profiling/summarize_reflex_pd_pressure.py`
- `scripts/profiling/diagnose_reflex_blockers.py`

### 16.2 Scheduler / metadata path

Scheduler 现在会显式传递 `remote_inflight_pages_by_request`：

- 只有 `reflex_remote_chunk_inflight=True` 且 remote chunking 启用时才标记；
- 范围是 `max(committed_page_end, page_start)` 到当前 chunk `page_end`；
- 已 committed 的 page 不再被当作 inflight；
- planner 侧将这批 page 标为 `is_remote_inflight=True`。

同时，whole-request protected 状态会落到每个 page 的 `is_request_protected` 上，而不是继续被折叠进通用的 `shared_or_open` 里。

### 16.3 单元测试

新增/更新测试：

- `test_planner_reports_lifecycle_blocker_breakdown`
  - 验证 planner 能区分 remote inflight、open tail、request protected、prompt protected、shared 和 copy-on-demote。

- `test_reflex_int4_planning_passes_remote_inflight_pages`
  - 验证 scheduler 正确传递 remote inflight pages；
  - 同时验证 sealed frontier 仍停在 committed page end，不把 inflight chunk 提前暴露给 demotion。

- `test_summarize_run_computes_trace_summary_and_timeline_metrics`
  - 验证 summary CSV 聚合新增 lifecycle breakdown 字段。

- `test_diagnose_summary_reports_page_lifecycle_subsignals`
  - 验证 diagnosis action 会把 page lifecycle 的主要子信号打印出来。

验证结果：

```text
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_reflex_int4_block_table.py \
  tests/profiling/test_reflex_int4_scheduler.py \
  tests/profiling/test_reflex_int4_pool.py \
  tests/profiling/test_precision_kv_run_optimizer.py \
  tests/profiling/test_precision_kv_frontier.py \
  tests/profiling/test_summarize_reflex_pd_pressure.py \
  tests/profiling/test_diagnose_reflex_blockers.py \
  tests/accuracy/test_pd_serving_mixed_accuracy.py -q

192 passed, 17 warnings
```

`py_compile` 覆盖了本轮修改的 core / scheduler / scripts 文件，全部通过。

### 16.4 真实 smoke 结果

运行目录：

```text
outputs/accuracy/reflex_next_lifecycle_breakdown_2026-05-21/ablation_00_frontier_dual_reflex
```

配置：

- prefill GPU 1
- decode GPU 3
- `frontier_dual_reflex`
- gov_report 2 samples
- Math500 2 samples
- max concurrency 2
- `num_gpu_blocks_override=736`
- `max_num_seqs=4`
- `max_num_batched_tokens=4096`
- `proxy_prefill_max_inflight=2`
- `force_triton_attn`

结果：

| dataset | completed | failed | avg latency | avg score |
|---|---:|---:|---:|---:|
| gov_report | 2 | 0 | 20.15s | 0.3415 |
| math500 | 2 | 0 | 15.51s | 0.5 |

关键 ReFlexKV trace：

| metric | value |
|---|---:|
| max decode KV usage | 98.50% |
| max decode running | 2 |
| max decode waiting | 1 |
| avg decode waiting | 0.2703 |
| demotion_event_count | 44 |
| demoted_pages_total | 1034 |
| actual_release_blocks_total | 1034 |
| candidate_breakdown_event_count | 110 |
| candidate_raw_bf16_pages_total | 78779 |
| candidate_open_bf16_pages_total | 9487 |
| candidate_remote_inflight_bf16_pages_total | 0 |
| candidate_open_tail_bf16_pages_total | 185 |
| candidate_request_protected_bf16_pages_total | 9302 |
| candidate_shared_bf16_pages_total | 0 |
| candidate_prompt_protected_bf16_pages_total | 1008 |
| candidate_copy_on_demote_pages_total | 0 |
| candidate_eligible_full_unshared_pages_total | 68284 |
| candidate_after_request_budget_cap_total | 4189 |
| candidate_after_sparse_window_quota_total | 3232 |
| candidate_after_frontier_optimizer_total | 1034 |

Admission blocked reason：

| reason | count |
|---|---:|
| partial_release | 1 |
| shared_or_open | 19 |
| allocation_failure | 8 |

Frontier rejection reason totals：

| reason | count |
|---|---:|
| shared_or_open | 13827 |
| recent_or_initial | 381 |
| request_fraction_cap | 4 |
| request_release_budget | 34 |
| request_budget | 38 |
| sparse_quota | 6 |
| frontier_optimizer | 29 |

Diagnosis 输出现在能给出更具体的 lifecycle action：

```json
[
  {"area": "request_precision_budget", "severity": "P1", "signal": 76},
  {"area": "sparse_window_quota", "severity": "P1", "signal": 6},
  {"area": "frontier_dual_optimizer", "severity": "P1", "signal": 29},
  {
    "area": "page_lifecycle",
    "severity": "P1",
    "signal": 13827,
    "action": "Audit page lifecycle subsignals before changing policy: open_tail=185, request_protected=9302, prompt_protected=1008."
  }
]
```

### 16.5 结果解释

这轮 smoke 里，原先看起来很大的 `shared_or_open` 其实不是 shared prefix，也不是 remote inflight：

- `remote_inflight_bf16_pages_total=0`
- `shared_bf16_pages_total=0`
- `copy_on_demote_pages_total=0`

真正主要的 lifecycle 子信号是：

- `request_protected_bf16_pages_total=9302`
- `prompt_protected_bf16_pages_total=1008`
- `open_tail_bf16_pages_total=185`

所以本轮结果改变了下一步优化方向：

1. 当前 workload 下不应该优先继续修 remote chunk commit 或 shared copy-on-demote；它们没有成为这次 smoke 的瓶颈。
2. `request_protected` 占比最大，说明 waiting/remote-KV/request-level protection 的边界过粗，可能把大量已经可安全 demote 的 closed page 一起挡住了。
3. `prompt_protected` 仍有 1008 页，需要改成分层保护：reasoning/short prompt 强保护，长 prompt 的低风险窗口允许进入 optimizer。
4. `request_budget`、`sparse_quota`、`frontier_optimizer` 仍然在收缩候选池，但这次能看到它们是在 `eligible_full_unshared_pages=68284` 之后发生的策略收缩，不是 page lifecycle 基础状态错误。

### 16.6 下一步建议

下一轮应该进入策略层优化：

1. 把 request-level protection 拆成 page/window-level protection。
   - remote inflight chunk 继续保护；
   - already committed closed pages 不应该因为 request 仍在等待或传输其它 chunk 而整体禁压；
   - waiting admission 只保护必要的 active frontier，不保护全 request。

2. 改 prompt protection 为分层 contract。
   - short reasoning prompt 强保护；
   - long prompt 的 early/global summary pages 强保护；
   - long prompt 的普通正文窗口允许低比例 demotion；
   - risk estimator 输出 window-level protection grade。

3. 调整 request budget / sparse quota 为层次化预算。
   - 不再只有一个全局 request fraction cap；
   - 每个 request 维护 protected / compressible / recovery-shadow 三类 page；
   - sparse window quota 根据窗口风险和 waiting pressure 动态放宽。

4. 降低 background pressure dry-run 频率。
   - 当前 110 次 candidate breakdown 中仍有大量重复 planning；
   - 应改成 event-driven：KV usage crossing threshold、waiting queue 变化、chunk commit、decode step interval 到达时才重新规划。

## 17. 2026-05-22 policy/admission 修复与真实状态校准

上一轮 `lifecycle_breakdown` 已经证明 page lifecycle 字段能解释 blocked reason，但仍然有两个实际问题：

- admission 在强压力下仍可能因为 reserve target 太小或重试路径太保守而反复卡住；
- 文档里对 recovery 和 prefix 的描述容易过度声称，必须和真实 trace 对齐。

### 17.1 本轮策略修复

主要修复：

1. Allocation failure demotion target 增加 admission slack floor。
   - 新增 `_reflex_int4_allocation_failure_demote_target()`。
   - 当 `_estimate_reflex_admission_demote_target()` 给出的释放量太小，但 admission reserve 本身需要更多余量时，用 `_reflex_int4_admission_reserve_blocks` 作为 floor。
   - 同时仍受 `_reflex_int4_fast_demotions_per_step` 限制，避免一次释放过猛。

2. `frontier_dual` 对大 admission waiting 启用 emergency release。
   - `full_sequence_reserve` / `allocation_failure` 继续走 emergency path。
   - `admission_waiting` 只有在 target 明显超过 reserve floor 时才启用 emergency release。
   - 目的不是绕过所有保护，而是在确实卡 admission 时允许 planner 放宽 request budget / sparse quota 的局部限制。

3. 保留 page/window-level protection。
   - prompt protected pages、request protected pages、open tail、remote inflight、shared/copy-on-demote 仍然分开统计。
   - 这次修复没有把 protected page 粗暴放开，而是只调整 admission pressure 下的释放目标和 optimizer emergency path。

涉及文件：

- `vllm/vllm/v1/core/sched/scheduler.py`
- `tests/profiling/test_reflex_int4_scheduler.py`

新增/更新测试：

- `test_reflex_int4_allocation_failure_target_keeps_admission_slack`
- `test_reflex_int4_frontier_dual_uses_emergency_release_for_large_admission_waiting`

完整单测命令：

```text
PYTHONPATH=/home/ytm/code/quant/SemantiQ pytest \
  tests/profiling/test_reflex_int4_block_table.py \
  tests/profiling/test_reflex_int4_scheduler.py \
  tests/profiling/test_reflex_int4_pool.py \
  tests/profiling/test_precision_kv_run_optimizer.py \
  tests/profiling/test_precision_kv_frontier.py \
  tests/profiling/test_precision_kv_policy.py \
  tests/profiling/test_summarize_reflex_pd_pressure.py \
  tests/profiling/test_diagnose_reflex_blockers.py \
  tests/accuracy/test_pd_serving_mixed_accuracy.py -q
```

结果：

```text
209 passed, 17 warnings in 6.58s
```

### 17.2 最新真实 smoke

运行目录：

```text
outputs/accuracy/reflex_next_policy_admission_fix7_2026-05-22/ablation_00_frontier_dual_reflex
```

配置仍是小规模 mixed pressure：

- gov_report 2 samples；
- Math500 2 samples；
- max concurrency 2；
- decode BF16 blocks override 736；
- `frontier_dual_reflex`；
- P/D disaggregated serving。

Dataset 结果：

| dataset | completed | failed | avg latency | avg score |
|---|---:|---:|---:|---:|
| gov_report | 2 | 0 | 22.35s | 0.4000 |
| math500 | 2 | 0 | 13.52s | 0.5 |

Serving / ReFlexKV 关键指标：

| metric | value |
|---|---:|
| duration_seconds | 62.658 |
| decode max KV usage | 95.65% |
| decode avg KV usage | 53.46% |
| decode max running | 2 |
| decode max waiting | 1 |
| decode avg waiting | 0.25 |
| demotion_event_count | 10 |
| demoted_pages_total | 1098 |
| actual_release_blocks_total | 1098 |
| admission_control_event_count | 246 |
| admission_blocked_total | 0 |
| admission_infeasible_total | 0 |
| candidate_breakdown_event_count | 10 |
| candidate_open_bf16_pages_total | 39 |
| candidate_request_protected_bf16_pages_total | 32 |
| candidate_prompt_protected_bf16_pages_total | 112 |
| candidate_after_frontier_optimizer_total | 1098 |
| demotion_gpu_ms_total | 59.021 |
| mean_int4_ratio | 0.2692 |
| max_int4_ratio | 0.6683 |

Candidate rejection totals：

| reason | count |
|---|---:|
| shared_or_open | 151 |
| recent_or_initial | 278 |
| request_budget / request_release_budget | 556 |
| sparse_quota | 38 |
| frontier_optimizer | 4765 |
| int4_pool_full | 0 |

Admission blocked frontier rejection totals 全部为 0，说明这些 rejection 这次没有导致 admission 失败，而是正常候选裁剪。

### 17.3 与上一轮 fix6 对比

| metric | fix6 | fix7 |
|---|---:|---:|
| duration_seconds | 133.65 | 62.66 |
| admission_blocked_total | 19 | 0 |
| admission_infeasible_total | 2 | 0 |
| demotion_event_count | 285 | 10 |
| demoted_pages_total | 1189 | 1098 |
| demotion_gpu_ms_total | 892.54 | 59.02 |
| max_int4_ratio | 0.9654 | 0.6683 |

这个结果说明上一轮存在明显过度 planning / 过度 demotion。fix7 后，系统仍然释放了足够 BF16 blocks，但用更少 demotion event、更低 GPU migration cost 和更低 INT4 ratio 完成 admission。

### 17.4 精度恢复现在到底有没有

结论：有低开销恢复 plumbing，但实时注意力触发的闭环恢复已经删除。

已有部分：

- BF16 shadow selection / recovery artifact metadata
- background promotion hook
- cache manager 的 recovery/promote entry
- scheduler recovery telemetry
- recovery 相关单测

最新真实 smoke 中没有触发：

| recovery metric | value |
|---|---:|
| recovery_plan_event_count | 0 |
| background_promoted_pages_total | 0 |
| recovery_exec_event_count | 0 |

所以目前不能在论文或总结里写成“ReFlexKV 已经具备有效精度恢复”。更准确的说法是：

```text
ReFlexKV keeps BF16-shadow recovery plumbing and budgeted background promotion, but recovery has not yet been validated as an accuracy-improving mechanism in real serving workloads.
```

下一步如果要把它变成真实功能，需要：

- 跑 recovery on/off 的准确率、吞吐、GPU time 对比；
- 证明 BF16 shadow/residual/promotion 的收益大于内存和迁移成本。

### 17.5 Prefix 现在到底有没有考虑

结论：prefix 已经被考虑，但目前只是局部安全机制和 control-plane skeleton，不是完整 prefix precision 系统。

已有部分：

- `PrefixPrecisionContractManager`
- shared prefix version / active version helper
- `requires_copy_on_demote()`
- `copy_on_demote()`
- prompt/page-level protection
- scheduler 传递 `copy_on_demote_pages_by_request`
- planner 允许 explicit copy-on-demote shared page 进入候选
- shared page copy-on-demote 的单元测试

最新真实 smoke 中没有覆盖 shared-prefix path：

| prefix metric | value |
|---|---:|
| candidate_shared_bf16_pages_total | 0 |
| candidate_copy_on_demote_pages_total | 0 |
| landing_contract_event_count | 0 |
| landing_commit_event_count | 0 |

所以目前不能写成“prefix precision contract 已经完整实现”。更准确的说法是：

```text
ReFlexKV has prefix safety hooks and a contract manager prototype, but it does not yet implement a full multi-version shared-prefix precision cache.
```

还缺：

- prefix cache allocation/reuse path 里的 precision ownership；
- shared prefix 的 copy-on-demote 真实 serving 验证；
- multi-version prefix index；
- prefix page eviction/accounting；
- shared-prefix workload 下的 correctness 和 throughput 实验。

### 17.6 当前真实结论

截至 2026-05-22，最可靠的 claim 是：

```text
ReFlexKV 的 decode-side dynamic demotion、frontier-dual admission、P-side risk metadata 传递和 page/window-level protection 已经能在小规模真实压力测试中跑通，并且 fix7 明显降低了 admission block、过度 demotion 和 migration cost。
```

不能过度 claim 的部分是：

- recovery 还没有真实触发和证明；
- prefix precision contract 还没有完整接入 prefix cache；
- quantizer/residual 还没有 CUDA hot-path 和系统消融；
- 当前 smoke 是 2+2 小样本，不足以支撑论文级结论；
- 下一步仍然需要 n=100、mixed workload、prefix-cache workload、recovery on/off、quantizer ablation 和 BF16 baseline 对比。

## 18. BurstGPT-shaped n=32 BF16 vs ReFlexKV smoke 对比

本轮使用 `gen_data` 里已经生成的 BurstGPT time-length 分布替换精度 workload：

```text
gen_data/burstgpt_answerable_mix_n32_cap20/burstgpt_n32_li40_lo30_cap20_seed0_manifest.jsonl
```

这个 manifest 有 32 条请求：

- gov_report：11；
- math500：13；
- qasper：8；
- long input requests：13；
- long output requests：10；
- cap 后 trace duration：605s；
- scaled trace duration：30.25s。

### 18.1 实验配置

BF16 baseline：

- output：`outputs/accuracy/burstgpt_answerable_n32_bf16_current_fullmem_inflight1_2026-05-22/ablation_00_bf16_baseline`
- decode KV dtype：`auto`
- decode KV capacity：full BF16 capacity，由 vLLM 根据 `gpu_memory_utilization=0.85` 计算；
- `proxy_prefill_max_inflight=1`
- P-side ReFlex metadata disabled；
- max concurrency：4。

ReFlexKV：

- output：`outputs/accuracy/burstgpt_answerable_n32_reflex_current_b736_2026-05-22/ablation_00_frontier_dual_reflex`
- decode KV dtype：`reflex_int4`
- decode BF16 block override：736；
- `proxy_prefill_max_inflight=4`
- page selection policy：`frontier_dual`
- P-side page risk metadata enabled；
- max concurrency：4。

这个对比的口径是：

```text
BF16 使用自身能稳定完成的 full-memory + conservative inflight 配置；
ReFlexKV 使用 decode BF16 受限的压力配置，依赖 demotion/INT4/recovery plumbing 释放容量。
```

### 18.2 总体结果

| metric | BF16 stable | ReFlexKV | ratio / delta |
|---|---:|---:|---:|
| completed | 32/32 | 32/32 | same |
| failures | 0 | 0 | same |
| duration | 921.72s | 359.93s | 2.56x faster |
| goodput | 0.0347 req/s | 0.0889 req/s | 2.56x |
| weighted avg latency | 99.07s | 37.54s | 2.64x lower |
| weighted avg score | 0.3443 | 0.4440 | +0.0998 |

Dataset-level score：

| dataset | BF16 score | ReFlexKV score | delta |
|---|---:|---:|---:|
| gov_report | 0.3613 | 0.3558 | -0.0055 |
| math500 | 0.3846 | 0.5385 | +0.1538 |
| qasper | 0.2553 | 0.4119 | +0.1566 |

Dataset-level latency：

| dataset | BF16 avg latency | ReFlexKV avg latency | improvement |
|---|---:|---:|---:|
| gov_report | 75.64s | 22.64s | 3.34x |
| math500 | 134.26s | 69.44s | 1.93x |
| qasper | 74.13s | 6.20s | 11.95x |

当前最稳妥的解释是：

```text
在这个 BurstGPT-shaped n=32 trace 上，ReFlexKV 没有观察到 accuracy degradation，并且在 admission 压力下显著提升 goodput/latency。
```

不能写成“ReFlexKV 提升数学/问答精度”。math500 和 qasper 分数高于 BF16，可能来自调度顺序、完成长度、截断、生成路径或评测随机细节；需要多 seed、多样本、同 token budget 的严格复现实验才能归因。

### 18.3 Serving / ReFlexKV trace

BF16 decode：

| metric | value |
|---|---:|
| max decode KV usage | 57.67% |
| avg decode KV usage | 10.33% |
| max running | 1 |
| avg running | 0.927 |
| max waiting | 1 |
| avg waiting | 0.070 |

ReFlexKV decode：

| metric | value |
|---|---:|
| max decode KV usage | 99.59% |
| avg decode KV usage | 65.21% |
| max running | 4 |
| avg running | 3.300 |
| max waiting | 2 |
| avg waiting | 0.199 |
| demotion_event_count | 130 |
| demoted_pages_total | 5862 |
| actual_release_blocks_total | 5862 |
| admission_blocked_total | 0 |
| admission_infeasible_total | 0 |
| admission_success_after_demote_total | 1486 |
| candidate_breakdown_event_count | 421 |
| demotion_gpu_ms_total | 546.907 |
| mean_int4_ratio | 0.2607 |
| max_int4_ratio | 0.8416 |
| page_metadata_real_risk_coverage_ratio | 1.0 |

Candidate rejection totals：

| reason | count |
|---|---:|
| shared_or_open | 15950 |
| recent_or_initial | 23959 |
| request_fraction_cap | 222471 |
| request_release_budget | 9389 |
| request_budget | 231860 |
| sparse_quota | 334 |
| frontier_optimizer | 21609 |
| int4_pool_full | 0 |

这些 rejection 在本轮没有造成 admission failure，`admission_blocked_total=0`、`admission_infeasible_total=0`。它们更像正常的 policy pruning，而不是卡死原因。

### 18.4 Recovery 和 prefix 在 BurstGPT n=32 下的状态

Recovery 这次比 2+2 smoke 更进一步：background promotion 路径实际触发了。

| recovery metric | value |
|---|---:|
| recovery_plan_event_count | 84 |
| recovery_exec_event_count | 84 |
| recovery_exec_pages_total | 84 |
| recovery_exec_layer_copies_total | 2688 |
| recovery_exec_cpu_ms_total | 74.034 |
| background_promoted_pages_total | 84 |
| page_metadata_plan_shadow_pages_total | 1868 |

结论需要谨慎：

- background promotion plumbing 已经能在真实 run 中执行；
- 实时注意力触发恢复路径已经删除，后续结果不再统计这类指标；
- 当前 score 不能归因到 recovery，因为没有 recovery on/off ablation；
- 下一步需要补 recovery on/off、background promotion on/off、shadow selection on/off 的矩阵。

Prefix 仍然没有真实覆盖：

| prefix metric | value |
|---|---:|
| candidate_shared_bf16_pages_total | 0 |
| candidate_copy_on_demote_pages_total | 0 |

这和当前 run 使用 `--no-enable-prefix-caching` 一致。prefix precision contract 仍然只是 helper/hook + 单测覆盖，尚未在 shared-prefix serving workload 中验证。

### 18.5 和旧 n=32 ReFlexKV 结果的差异

旧 ReFlexKV run：

```text
outputs/accuracy/burstgpt_answerable_n32_reflex_only_b736_2026-05-21/ablation_00_frontier_dual_reflex
```

旧结果约为：

- duration：308.00s；
- goodput：0.1039 req/s；
- weighted avg latency：约 33.15s；
- admission_blocked_total：248；
- admission_infeasible_total：64；
- demotion_event_count：571；
- demotion_gpu_ms_total：1624.614；
- max_int4_ratio：0.9467。

当前 run：

- duration：359.93s；
- goodput：0.0889 req/s；
- weighted avg latency：37.54s；
- admission_blocked_total：0；
- admission_infeasible_total：0；
- demotion_event_count：130；
- demotion_gpu_ms_total：546.907；
- max_int4_ratio：0.8416。

因此最新策略牺牲了一部分吞吐峰值，但换来了更稳定的 admission frontier、更少过度 demotion、更低 migration cost、更低 max INT4 ratio。这更接近可解释、可写论文的系统行为，但后续还需要把吞吐回收回来。

### 18.6 本轮验证

日志扫描没有发现：

```text
ERROR
Traceback
OOM
OutOfMemory
Aborting requests
Cannot overwrite
Cannot commit stale
RuntimeError
```

BF16 和 ReFlexKV run 都完整生成 `mixed_summary.json`，GPU0/1/2/3 已释放。

## 19. BurstGPT-shaped n=100 BF16 vs ReFlexKV 对比

用户指出 n=32 太少，这个判断是对的。n=32 只能作为 smoke，不应该作为主要结论。后续当前结果以 n=100 为准。

### 19.1 n=100 manifest

新 manifest：

```text
gen_data/burstgpt_answerable_mix_n100_cap20/burstgpt_n100_li40_lo30_cap20_seed0_manifest.jsonl
```

生成策略：

- BurstGPT trace：`data/burstgpt/data/BurstGPT_1.csv`
- selected rows：100
- failed response rows：0
- time scale：0.05
- max inter-arrival cap：20s
- scaled trace duration：96.9s
- long input threshold：2048 tokens
- long output threshold：512 tokens
- long input requests：40
- long output requests：30
- benchmark mix：gov_report 35%、math500 40%、qasper 25%
- prompt fit policy：`none`

Dataset 分布：

| dataset | task | samples | source_index | max_new_tokens |
|---|---|---:|---|---:|
| gov_report | LongBench summarization | 35 | 0-34 | 512 |
| math500 | reasoning | 40 | 0-39 | 4096 |
| qasper | LongBench QA | 25 | 0-24 | 128 |
| total | mixed | 100 | | |

Trace bucket 分布：

| bucket | count |
|---|---:|
| short input | 30 |
| medium input | 30 |
| long input | 40 |
| short output | 8 |
| medium output | 62 |
| long output | 30 |

### 19.2 截断检查

这轮明确禁止截断。生成前检查过前 120 个候选样本，gov_report 的 over-limit 样本从 source index 56 以后才出现；本轮只用了 gov_report 0-34。

生成后重新用 Llama-3.1-8B-Instruct tokenizer 对 manifest 逐条检查：

| dataset | samples | prompt tokens min/max | prompt + max_new + margin max | over 32768 |
|---|---:|---:|---:|---:|
| gov_report | 35 | 2663 / 20515 | 21035 | 0 |
| math500 | 40 | 40 / 359 | 4463 | 0 |
| qasper | 25 | 2988 / 17637 | 17773 | 0 |
| total | 100 | | | 0 |

运行生成的 `prompt_fit_summary.json`：

```text
BF16 []
ReFlexKV []
```

结论：

```text
n=100 manifest 没有 prompt truncation，也没有 runtime prompt fitting。
```

### 19.3 实验配置

BF16 baseline：

- output：`outputs/accuracy/burstgpt_answerable_n100_bf16_current_fullmem_inflight1_2026-05-22/ablation_00_bf16_baseline`
- decode KV dtype：`auto`
- decode KV capacity：vLLM full BF16 capacity，`gpu_memory_utilization=0.85`
- `proxy_prefill_max_inflight=1`
- P-side ReFlex metadata disabled
- max concurrency：4

ReFlexKV：

- output：`outputs/accuracy/burstgpt_answerable_n100_reflex_current_b736_2026-05-22/ablation_00_frontier_dual_reflex`
- decode KV dtype：`reflex_int4`
- decode BF16 block override：736
- `proxy_prefill_max_inflight=4`
- page selection policy：`frontier_dual`
- P-side page risk metadata enabled
- max concurrency：4

### 19.4 总体结果

| metric | BF16 stable | ReFlexKV | ratio / delta |
|---|---:|---:|---:|
| completed | 100/100 | 100/100 | same |
| failures | 0 | 0 | same |
| duration | 2970.21s | 1145.34s | 2.59x faster |
| goodput | 0.0337 req/s | 0.0873 req/s | 2.59x |
| weighted avg latency | 113.57s | 43.44s | 2.61x lower |
| weighted avg score | 0.3402 | 0.3581 | +0.0179 |

Dataset-level score：

| dataset | BF16 score | ReFlexKV score | delta |
|---|---:|---:|---:|
| gov_report | 0.3527 | 0.3486 | -0.0041 |
| math500 | 0.4000 | 0.4250 | +0.0250 |
| qasper | 0.2271 | 0.2644 | +0.0373 |

Dataset-level latency：

| dataset | BF16 avg latency | ReFlexKV avg latency | improvement |
|---|---:|---:|---:|
| gov_report | 102.73s | 20.31s | 5.06x |
| math500 | 142.69s | 86.68s | 1.65x |
| qasper | 82.13s | 6.65s | 12.36x |

更稳妥的结论：

```text
在无截断 BurstGPT-shaped n=100 workload 上，ReFlexKV 相比稳定 BF16 baseline 有约 2.6x goodput/latency 改善，并且没有观察到 accuracy degradation。
```

仍然不能 claim “ReFlexKV 提升精度”。ReFlexKV 的平均分略高，但差异需要多 seed、更大样本、同输出长度分布和 recovery/quantizer ablation 才能归因。

### 19.5 Serving / ReFlexKV trace

BF16 decode：

| metric | value |
|---|---:|
| max decode KV usage | 63.36% |
| avg decode KV usage | 10.00% |
| max running | 1 |
| avg running | 0.927 |
| max waiting | 1 |
| avg waiting | 0.071 |

ReFlexKV decode：

| metric | value |
|---|---:|
| max decode KV usage | 99.86% |
| avg decode KV usage | 66.86% |
| max running | 4 |
| avg running | 3.606 |
| max waiting | 3 |
| avg waiting | 0.187 |
| demotion_event_count | 920 |
| demoted_pages_total | 14922 |
| actual_release_blocks_total | 14922 |
| admission_blocked_total | 0 |
| admission_infeasible_total | 0 |
| admission_success_after_demote_total | 5264 |
| candidate_breakdown_event_count | 3110 |
| demotion_gpu_ms_total | 2402.798 |
| mean_int4_ratio | 0.3440 |
| max_int4_ratio | 0.8698 |
| page_metadata_real_risk_coverage_ratio | 1.0 |

Candidate rejection totals：

| reason | count |
|---|---:|
| shared_or_open | 104718 |
| recent_or_initial | 184220 |
| request_fraction_cap | 1663219 |
| request_release_budget | 194204 |
| request_budget | 1857423 |
| sparse_quota | 1563 |
| frontier_optimizer | 54363 |
| int4_pool_full | 0 |

这些 rejection 没有导致 admission failure：

```text
admission_blocked_total = 0
admission_infeasible_total = 0
```

### 19.6 Recovery 和 prefix 状态

Recovery：

| metric | value |
|---|---:|
| recovery_plan_event_count | 335 |
| recovery_exec_pages_total | 335 |
| background_promoted_pages_total | 335 |

解释：

- background promotion 在 n=100 中真实执行；
- 实时注意力触发恢复已经从系统里删除；
- 当前 accuracy 不能归因到 recovery，因为还没有 recovery on/off 消融。

Prefix：

| metric | value |
|---|---:|
| candidate_shared_bf16_pages_total | 0 |
| candidate_copy_on_demote_pages_total | 0 |

这轮仍然没有打开 prefix caching，也不是 shared-prefix workload，所以 prefix precision contract 仍未被真实 serving 验证。

### 19.7 验证

日志扫描没有发现：

```text
ERROR
Traceback
OOM
OutOfMemory
Aborting requests
Cannot overwrite
Cannot commit stale
RuntimeError
```

BF16 和 ReFlexKV 都完整生成 `mixed_summary.json`，GPU0/1/4/5 已释放。

## 20. 2026-05-22 精度恢复路径清理

这轮根据最新讨论，把高开销、难验证、没有真实触发价值的实时注意力恢复路径从系统中删除。当前恢复口径收敛为：

- 保留 BF16 shadow metadata；
- 保留 cache manager 的 explicit recovery/promote entry；
- 保留 scheduler 的 budgeted background promotion；
- 保留 recovery/promotion telemetry；
- 删除 decode attention mass profile；
- 删除按 attention relevance 触发的 precision fault planner；
- 删除 synthetic recovery relevance fallback；
- 删除 `RecoveryQualityEvaluator` 和 `precision_kv/recovery.py`；
- 删除 worker/model output/kernel 中为 attention mass profile 增加的字段和参数；
- 删除 profiling/accuracy summary 里的旧 attention/fault 指标列。

这不是削弱 ReFlexKV 的核心机制。ReFlexKV 的核心仍然是：

- P-side risk metadata；
- chunk/frontier admission；
- online planner；
- BF16/INT4 mixed landing；
- decode-side dynamic demotion；
- mixed-precision attention；
- prefix protection / copy-on-demote skeleton；
- telemetry / blocked reason breakdown。

清理后的结论：

- 当前 n=100 的吞吐/延迟收益主要来自 admission + demotion + mixed precision KV，而不是实时恢复；
- background promotion 在历史 n=100 run 中触发过 335 pages，但还没有 recovery on/off 消融，不能把 accuracy 变化归因给 recovery；
- 后续论文实验应把 recovery 独立成可开关模块：background promotion on/off、BF16 shadow on/off、residual compensation on/off；
- 如果未来重新做实时恢复，必须先证明低成本 relevance signal 的收益，否则不应放回热路径。

本轮验证：

| check | result |
|---|---:|
| `py_compile` touched runtime/scripts | pass |
| targeted removal/background promotion tests | 4 passed |
| profiling/accuracy regression subset | 210 passed |
| CUDA direct attention tests | 2 passed |

## 21. 2026-05-23 n=500 公平 backpressure baseline

这轮先修正 BF16 baseline 的公平性问题：不再用
`proxy_prefill_max_inflight=1` 作为主对比，而是在 proxy 中加入通用
decode-capacity-aware backpressure。BF16 和 ReFlexKV 都使用：

- `proxy_prefill_max_inflight=4`
- `proxy_decode_backpressure_policy=metrics`
- `proxy_decode_backpressure_max_kv_usage=0.90`
- `proxy_decode_backpressure_max_waiting=0`
- 同一个 `paper_accuracy_n500_seed0_manifest.jsonl`
- 同一个 `max_concurrency=4`
- 同一个 P/D serving path

新增实现：

- proxy 在启动 remote prefill 前轮询 decode `/metrics`；
- 当 decode KV usage 超过阈值，或 decode waiting requests 大于阈值时，暂停新的 prefill admission；
- telemetry 不可用时 fail-open，避免 proxy 变成单点阻塞；
- ablation matrix 默认 BF16 和 ReFlexKV 都走同一套 decode backpressure；
- BF16 baseline 不再被 case preset 强制改成 `inflight=1`。

相关验证：

| check | result |
|---|---:|
| proxy state tests | 8 passed |
| runner / matrix command tests | 19 passed |
| selected accuracy runner tests | 2 passed |
| combined targeted tests | 21 passed |
| n=20 BF16 smoke | 20/20, no failure |
| n=20 ReFlexKV smoke | 20/20, no failure |

### 21.1 n=500 workload

| dataset | samples |
|---|---:|
| gov_report | 80 |
| qasper | 80 |
| passage_retrieval_en | 80 |
| math500 | 90 |
| gsm8k | 110 |
| aime24 | 30 |
| aime25 | 30 |
| total | 500 |

这批 manifest 没有 prompt truncation。最大 `prompt + max_new_tokens`
小于 `max_model_len=32768`。

### 21.2 n=500 result: BF16-backpressure vs ReFlexKV-backpressure

Run dirs：

- BF16:
  `outputs/accuracy/paper_accuracy_n500_bf16_backpressure_inflight4_2026-05-23/bf16_backpressure_n500`
- ReFlexKV:
  `outputs/accuracy/paper_accuracy_n500_reflex_backpressure_b736_2026-05-23/reflex_backpressure_n500`

Overall：

| metric | BF16-backpressure | ReFlexKV-backpressure |
|---|---:|---:|
| completed | 500/500 | 500/500 |
| failed | 0 | 0 |
| duration | 3941.43 s | 4448.47 s |
| goodput | 0.1269 req/s | 0.1124 req/s |
| weighted avg latency | 31.29 s | 35.34 s |
| weighted avg score | 0.4975 | 0.5155 |
| proxy backpressure holds | 91 | 34 |
| decode max KV usage | 91.80% | 100.00% |
| decode avg KV usage | 31.77% | 56.28% |
| decode max waiting | 2 | 3 |
| decode avg waiting | 0.280 | 0.191 |

Per-dataset score：

| dataset | BF16 | ReFlexKV | delta |
|---|---:|---:|---:|
| gov_report | 0.3551 | 0.3454 | -0.0097 |
| qasper | 0.2011 | 0.2130 | +0.0118 |
| passage_retrieval_en | 0.9281 | 0.9134 | -0.0147 |
| math500 | 0.4778 | 0.5333 | +0.0556 |
| gsm8k | 0.7818 | 0.8364 | +0.0545 |
| aime24 | 0.0000 | 0.0000 | 0.0000 |
| aime25 | 0.0333 | 0.0000 | -0.0333 |

Per-dataset average latency：

| dataset | BF16 | ReFlexKV |
|---|---:|---:|
| gov_report | 15.34 s | 18.23 s |
| qasper | 5.48 s | 6.64 s |
| passage_retrieval_en | 6.62 s | 7.33 s |
| math500 | 71.37 s | 79.04 s |
| gsm8k | 20.71 s | 22.60 s |
| aime24 | 77.91 s | 91.51 s |
| aime25 | 80.39 s | 91.69 s |

ReFlexKV telemetry：

| metric | value |
|---|---:|
| demotion_event_count | 976 |
| demoted_pages_total | 64063 |
| released_bf16_blocks_total | 64063 |
| admission_success_after_demote_total | 481 |
| admission_blocked_total | 0 |
| admission_infeasible_total | 0 |
| mean_int4_ratio | 0.3661 |
| max_int4_ratio | 0.8891 |
| demotion_gpu_ms_total | 3899.112 |
| background_promoted_pages_total | 1565 |
| recovery_exec_pages_total | 1565 |
| attention_trace_event_count | 0 |

Log scan：

| log issue | BF16 | ReFlexKV |
|---|---:|---:|
| ERROR | 0 | 0 |
| Traceback | 0 | 0 |
| Prefill task failed | 0 | 0 |
| P-side ready timeout | 0 | 0 |
| backpressure timeout | 0 | 0 |

### 21.3 Interpretation

这轮结果不能写成 ReFlexKV 吞吐优于 BF16。相反，在 full-memory BF16
加通用 backpressure 后，BF16 稳定跑完，并且 goodput 更高。当前结果说明：

- `inflight=1` BF16 baseline 不够公平，应该降级为 conservative baseline；
- 通用 decode backpressure 是必须的 serving integration 组件；
- ReFlexKV 在 n=500 下机制稳定：demotion 触发 64063 pages，admission
  没有 blocked/infeasible，P/D 没有 timeout；
- 但当前 ReFlexKV 的 planner/demotion/INT4 path 有可见开销；
- 当前 ReFlexKV 使用 `num_gpu_blocks_override=736` 制造 decode BF16
  pressure，而 BF16-backpressure 是 full-memory baseline，因此这不是同容量
  throughput 对比。

下一步论文实验应拆成三组：

1. Full-memory BF16-backpressure vs ReFlexKV-backpressure：证明 ReFlexKV
   在不降精度的情况下可稳定运行，但不主张吞吐优势。
2. Same BF16 block budget baseline：BF16 也限制到同等 BF16 block frontier，
   测它是否卡住、降吞吐或需要更强 backpressure。
3. Pressure sweep：扫 `num_gpu_blocks_override` / request rate /
   max_concurrency，找到 BF16 刚开始不稳定而 ReFlexKV 仍稳定的区间。

这轮最重要的结论是：系统稳定性已经明显改善，但 ReFlexKV 的论文主结果还需要
在同容量压力矩阵里证明，而不是用 full-memory BF16 做吞吐加速结论。
