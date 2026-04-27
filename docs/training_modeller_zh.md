# 训练建模器 —— 架构审查与差距分析

_2026-04-23。参考文档：`docs/ai_infra_modeller_design.md`。_

---

## 整体系统结构

```
┌─────────────────────────────────────────────────────────────────────┐
│  输入                                                                │
│  HuggingFace 模型（任意 —— Llama / DeepSeek / Qwen / ...）           │
│  + 硬件 YAML  + 训练策略（CLI 或 API）                                │
└───────────────────────────┬─────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  图捕获  (graph/)                                                    │
│                                                                     │
│  run_trace_phases("train_forward", "train_backward")                │
│    FakeTensorMode + TorchDispatchMode + ModuleTracker               │
│    → 原始记录包含：op_type、张量形状、层索引、                          │
│      模块作用域、组件标签                                              │
│                                                                     │
│  fusion.py → 将叶子算子融合为模块级分组                                │
│  graph_builder.py → OpGraph[fwd]  +  OpGraph[bwd]（各自独立）        │
│                                                                     │
│  ⚠  阶段 0：stitch_fwd_bwd() → 统一 OpGraph                         │
│     通过张量 ID 交叉引用，添加 fwd→bwd 依赖边                          │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  统一 OpGraph
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  变换流水线  (transform/)                                             │
│                                                                     │
│  ── SPLIT 阶段 ───────────────────────────────────────────────────   │
│  TensorParallelPass      分片列/行并行线性层；插入 AG/RS               │
│  ContextParallelPass  ⬡  Ulysses A2A | Ring 每 KV 块 P2P             │
│  ExpertParallelPass      分片专家 FFN；插入 dispatch A2A              │
│  PipelineParallelPass ⬡  基于 node.layer 标注 stage_id；             │
│                          在阶段边界插入 P2P send/recv                │
│  DataParallelPass     ⬡  在 bwd 节点后插入梯度 AR/RS                  │
│  CommInserterPass        将 CP + DP 通信集合接入图中                  │
│                                                                     │
│  ── FUSE 阶段 ────────────────────────────────────────────────────   │
│  FusionPass              将细粒度算子合并为模块级分组                   │
│                                                                     │
│  ── OPTIM 阶段 ───────────────────────────────────────────────────   │
│  RecomputePass           标记重计算算子（层级：full /                   │
│                          attn / ffn_swiglu / ln）                   │
│  ZeroFSDPPass            标注 weight/grad/optstate 分片因子；         │
│                          为 ZeRO-3 插入 FSDP AG                     │
│  OffloadPass             插入 D2H / H2D 传输节点                     │
│  OptimizerPass           附加 Adam/Muon 更新步节点；                   │
│                          标注 state_bytes、step_flops                │
│                                                                     │
│  ── ANALYZE 阶段 ─────────────────────────────────────────────────   │
│  FlopsPass（训练模式）   每节点：flops_fwd、flops_dx、flops_dw        │
│  RooflinePass            每节点：compute_us、memory_us、bound         │
│  CommLatencyPass         每通信节点：latency_us（α-β 模型，分层）       │
│  StreamAssignPass        分配 stream_id；应用 CoC/MC2 重叠             │
│                       ⬡  规则；DP-in-bubble 标注                    │
│  TrainMemoryPass      ⬡  遍历拼接后的图：                             │
│                          参数 = is_param 节点 / ZeRO 分片             │
│                          激活 = fwd 存活至 bwd 的张量                 │
│                          梯度 + 优化器状态 = OptimizerPass 标注        │
│                          在途深度 = pp - stage_rank                  │
│                                                                     │
│  ⬡ = 全新或需要大幅修复                                               │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  完整标注的统一 OpGraph
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  按阶段调度  (executor/)                                              │
│                                                                     │
│  for s in range(pp):                                                │
│      stage_nodes = [n for n in graph.nodes                          │
│                     if n.annotations["stage_id"] == s]             │
│      timeline[s] = DAGScheduler(hw).schedule(stage_nodes)          │
│      t_fwd[s] = timeline[s].phase_latency("fwd")                   │
│      t_bwd[s] = timeline[s].phase_latency("bwd")                   │
│                                                                     │
│  DAGScheduler：拓扑排序 + 贪心流分配                                   │
│  → 通信与计算在不同流上并发执行                                          │
│  → overlap_us 由流并行窗口计算得出                                     │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  list[Timeline]，每个 PP 阶段一个
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  流水线调度合成器  (transform/schedule/composer.py) ⬡                 │
│                                                                     │
│  输入：stage_timelines[(t_fwd, t_bwd)]、strategy、dp_ar_time         │
│                                                                     │
│  1F1B：                                                             │
│    t_step = (pp-1)·t_fwd[0]  +  M·t_bottleneck  +  (pp-1)·t_bwd[-1]
│    bubble = 2·(pp-1)·t_bottleneck / t_step                         │
│                                                                     │
│  VPP（交错）：                                                        │
│    bubble = (pp-1) / (vpp_chunks · M)                               │
│                                                                     │
│  DualPipe / DualPipeV：                                             │
│    fwd+bwd 配对重叠；EP A2A 由对端 μbatch 计算隐藏                    │
│                                                                     │
│  DP-in-bubble：                                                     │
│    t_exposed_dp = max(0, dp_ar_time - bubble_window)                │
│    t_step += t_exposed_dp                                           │
└───────────────────────────┬─────────────────────────────────────────┘
                            │  StepResult
                            ▼
┌─────────────────────────────────────────────────────────────────────┐
│  TrainingReport（训练报告）                                           │
│    step_time_ms      MFU        bubble_fraction                     │
│    memory_breakdown  （权重 / 梯度 / 优化器状态 / 激活 / 通信缓冲）     │
│    per_stage_ms      forward_flops  backward_flops                  │
│    chrome_trace  （可选 —— chrome://tracing 可视化）                  │
└─────────────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴────────────┐
              ▼                          ▼
     单点 estimate()              sweep() → Pareto 帕累托前沿
     CLI / Python API             按 (step_time, peak_HBM) 优化
```

---

## 核心设计原则

**一张捕获图。各 Pass 对其标注与变换。最终分析读取标注结果。**

捕获的算子图（通过 `FakeTensorMode` + `TorchDispatchMode`）携带真实张量形状、真实层结构（`node.layer`、`node.scope`）以及任意 HuggingFace 模型的算子级细节。所有建模 Pass —— 并行化、内存、时延、通信 —— 都对这张单一图进行变换或标注。最终的流水线合成器读取来自最终标注图的逐阶段时延。

`python/zrt/training/`（基于规格说明的 Stack A）**不是主要路径**。其解析公式可作为参考验证；当捕获图可用时，无需其独立的 IR 和 `ModelSpec`。

---

## 结构性障碍：前向图与反向图未连通

`run_trace_phases`（graph/main.py）输出**两个独立的 `OpGraph` 对象**，以 `{"train_forward": (raw, fused), "train_backward": (raw, fused)}` 形式存储。反向图中没有指向前向图的依赖边，这意味着：

- 激活值的生命周期（前向产生、反向消费）无法计算 —— 缺少跨图边
- 基于图的内存建模在拼接之前无法实现
- `modeller.py` 中的 `estimate_training_from_graphs()` 将两者作为独立图处理

此问题必须首先解决，后续所有工作均依赖统一的前向+反向图。

---

## 阶段 0 —— 统一前向图与反向图

**目标：** 生成一张代表单 Rank、单 PP 阶段完整训练步骤的 `OpGraph`。

1. 在 `graph/graph_builder.py` 或 `ir/adapter.py` 中实现 `stitch_fwd_bwd(fwd_graph, bwd_graph) -> OpGraph`：
   - 合并两图节点（为反向节点 ID 加前缀以避免冲突）
   - 对于每个反向节点，若其消费的张量 ID 由前向节点产生，则插入依赖边
   - 标注：`node.annotations["phase"] = "bwd"` / `"fwd"`
   - 标注参数节点：`node.annotations["is_param"] = True`（通过 scope 路径模式判断）
2. 更新 `estimate_training_from_graphs()` 以在传入流水线前先调用 `stitch_fwd_bwd`。
3. 验证：拼接后的图中反向节点无悬空输入；张量 ID 查找为 `O(1)`。

**解锁能力：** 激活值生命周期分析、正确的逐步内存模型、统一的 DAGScheduler 运行。

---

## 阶段 1 —— 修复正确性 Bug（当前输出有误）

三个 Bug 影响所有当前输出：

**1.1 —— 步骤时间公式（training.py:229）**
`per_stage_us = stage_time_us / pp` 将对未分割图的一次 DAGScheduler 运行结果除以 PP 度。这不是真正的逐阶段时间 —— 忽略了阶段异构性和 warmup/cooldown 结构。
- **修复**：对每个阶段视图分别运行 `DAGScheduler`（阶段 2 提供相关机制）；至少立即修正 1F1B 公式为 `M·t_stage + (pp-1)·t_fwd_stage + (pp-1)·t_bwd_stage`。

**1.2 —— 激活内存（training.py:151）**
固定系数 `10 × hidden × seq × layers / tp` 忽略：`RecomputePass` 层级标记（可减少 50–80%）、CP 分片（`/cp`）、逐阶段在途 μbatch 深度。同时未读取 `ZeroFSDPPass` 标注 —— ZeRO 因子被重复独立推导。
- **修复**：读取 `ZeroFSDPPass` 写入的 `g.metadata["zero"]`；应用重计算零化；在可用时用实际节点输出大小替换系数。

**1.3 —— 总 FLOPs（training.py:56）**
`TrainingFlopsPass` 计算 `6 * count_params(g) * tokens` 并覆盖 `FlopsPass`（passes.py，训练模式）已正确标注的逐节点 `flops_fwd/dx/dw`。
- **修复**：对所有节点求和 `node.annotations.get("flops_fwd", 0) + flops_dx + flops_dw`；删除覆盖逻辑；仅在 `FlopsPass` 未运行时将 `6P` 作为回退。

---

## 阶段 2 —— 流水线并行阶段分配与合成

**目标：** 真实的逐阶段时延 → 正确的步骤时间。

**2.1 —— `PipelineParallelPass`**（`transform/parallel/pipeline_parallel.py`）
- 每个 `OpNode` 携带 `node.layer`（如 `"3"`）。解析得到排序后的层索引集合。
- 将连续层组分配给 `pp` 个阶段。默认：均分。更优：按每层 `node.annotations["compute_us"]` 之和贪心装箱（需先运行 `RooflinePass`）。
- 标注：`node.annotations["stage_id"] = i`。
- 在阶段边界插入 `comm.send_recv` P2P 节点（跨阶段边界的激活张量，大小来自 `TensorMeta.mem_bytes`）。

**2.2 —— `estimate_training_from_graphs()` 中的逐阶段调度**
- 变换流水线运行完毕后，提取 `pp` 个阶段视图：`[n for n in g.nodes if n.annotations["stage_id"] == s]`。
- 对每个阶段视图运行 `DAGScheduler` → 每阶段一个 `Timeline`。
- 从 Timeline 中提取每阶段的 `(t_fwd_us, t_bwd_us)`（按 `node.annotations["phase"]` 拆分）。

**2.3 —— 1F1B 合成器**（修复 `transform/analysis/training.py::TrainingPipelinePass`）
```
t_stage = max(t_fwd[s] + t_bwd[s] for s in stages)   # 瓶颈阶段
step_us = (pp-1)*t_fwd[0] + M*t_stage + (pp-1)*t_bwd[pp-1]
bubble = (2*(pp-1)*t_stage_avg) / step_us
```
使用真实阶段时延，而非基于计数的公式。

---

## 阶段 3 —— 并行度完整性与精确内存建模

**3.1 —— `ContextParallelPass`**（`transform/parallel/context_parallel.py`）
- **Ulysses CP**：在注意力前插入 `comm.all_to_all`（分散序列、聚合头），在注意力后插入逆变换。组大小 = `cp`，数据量 = `b × s/cp × h`。
- **Ring CP**：在注意力内插入 `cp` 轮 `comm.send_recv`，每轮对应一个 KV 块。标记为可与 FA tile 计算重叠（`annotations["overlap_target"] = paired_fa_tile_id`）。
- 扩展 `CommInserterPass` 以支持 CP 通信。

**3.2 —— `DataParallelPass`**（`transform/parallel/data_parallel.py`）
- 在反向节点后，对每个参数节点的梯度张量插入 `comm.all_reduce`（ZeRO-0）或 `comm.reduce_scatter`（ZeRO-2/3）。
- 标注 `annotations["dp_comm"] = True`；当设置 `dp_overlap_in_bubble` 时标注 `annotations["overlap_in_bubble"] = True`。
- 流水线合成器从 bubble 窗口中减去 `t_dp_ar`：`t_exposed_dp = max(0, t_dp - bubble_duration)`。

**3.3 —— `StreamAssignPass` 中的 CoC/MC2 重叠规则**
- **CoC**：当 `tp_overlap == COC` 时，对配对的 AG/RS + matmul 节点标注 `annotations["coc_tile_k"] = k`。暴露通信时间 = `max(0, t_comm - t_matmul * (k-1)/k)`。
- **MC2**：将 AG+matmul 标注为单个融合节点，暴露通信时间为零。
- **Ring-CP**：相邻 FA tile 节点的 P2P 节点获得重叠标注；暴露时间 = `max(0, t_p2p - t_fa_tile)`。

**3.4 —— 基于图的内存建模**
前向+反向图拼接后（阶段 0）：
- **参数**：`annotations["is_param"] == True` 的节点 → 对 `output.mem_bytes` 求和 / ZeRO 权重分片因子（来自 `ZeroFSDPPass`）。
- **激活值**：前向节点的输出张量出现在反向节点输入中 → 这些张量为"已保存"，从前向存活至反向。对 `mem_bytes` 求和；对 `RecomputePass` 标记的层级应用零化。
- **梯度**与**优化器状态**：来自 `OptimizerPass` 标注（`state_bytes`），除以 ZeRO 分片因子。
- **在途 μbatch 深度**：1F1B 稳态下，阶段 `s` 同时持有 `pp - s` 个 μbatch 的激活值。相应地乘以该倍数。

---

## 阶段 4 —— 高级调度、搜索与验证

**4.1 —— VPP / 交错 1F1B**
正确的 bubble 公式：`(pp-1) / (vpp × M)`。需要 `TrainingConfig` 中的 `vpp_chunks`。每个阶段以交错顺序运行 `vpp` 个块。

**4.2 —— DualPipe / DualPipeV**
每个阶段并发运行 μbatch_i 的前向与 μbatch_{i-1} 的反向。Bubble 约为 I1F1B 的一半。当 `dualbatch=True` 时，EP A2A 由对端 μbatch 计算隐藏。

**4.3 —— EP 负载不均衡**
`ModelSpec.expert_imbalance`（或 CLI 参数）对瓶颈专家计算进行缩放：`t_expert_bottleneck = t_avg * (1 + imbalance)`。作为乘数应用于专家矩阵乘节点的时延。

**4.4 —— 搜索与帕累托前沿**（`search/sweep.py`）
对 `(tp, cp, pp, ep, dp, zero_stage)` 进行网格搜索，附带剪枝：不允许跨节点 TP、CP 仅在 `seq ≥ 32k` 时启用、EP 仅在 `num_experts > 1` 时启用。输出按 `(step_time, peak_hbm)` 的帕累托前沿。

**4.5 —— 锚点验证**（`tests/training/anchors/`）
包含已发表训练配置及其实测 MFU：GPT-3 175B（Megatron）、Llama-3 70B（Meta）、DeepSeek-V3（技术报告第 4 节）。CI 断言估算 MFU 误差在 15% 以内。这是所有上述工作的正确性门控。

**4.6 —— Chrome Trace 导出**
在 `TrainingReport` 中添加 `build_chrome_trace(g, stage_timelines)` —— 输出可在 `chrome://tracing` 中查看的 `trace.json`。

---

## 关键文件

| 文件 | 作用 | 阶段 |
|---|---|---|
| `graph/graph_builder.py` | `stitch_fwd_bwd()` 实现 | 0 |
| `transform/analysis/training.py` | 修复三处正确性 Bug | 1 |
| `transform/parallel/pipeline_parallel.py` | PP 阶段分配 | 2 |
| `transform/analysis/modeller.py` | 接入逐阶段 DAGScheduler + 合成器 | 2 |
| `transform/parallel/context_parallel.py` | CP（Ulysses + Ring） | 3 |
| `transform/parallel/data_parallel.py` | DP 梯度插入 | 3 |
| `transform/analysis/passes.py` | `StreamAssignPass` CoC/MC2 规则 | 3 |
| `training/compose/pipeline.py` | VPP / DualPipe 公式 | 4 |
| `tests/training/anchors/` | 已发表 MFU 基准 | 4 |
