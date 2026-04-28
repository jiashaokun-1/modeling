# 训练建模器 —— 双路径架构与统一方案

_2026-04-28。修订自 `docs/training_modeller_zh.md`（原文保留）。_

---

## 双路径现状

系统当前存在两条并行的训练性能估算路径，均收敛于同一组 `PipelineComposer` 类：

```
┌───────────────────────────────────────────────────────────────────────────────┐
│  Stack A：规格驱动路径（快速分析估算）                                          │
│                                                                               │
│  ModelSpec + Strategy                                                         │
│      │                                                                        │
│      ▼                                                                        │
│  build_graph()  →  training.ir.Graph（层级算子列表 + 集合通信列表）             │
│      │                                                                        │
│      ├── total_training_flops()   ← models/flops.py（解析公式）               │
│      ├── collective_time()        ← models/comm.py（α-β 模型）                │
│      ├── memory_breakdown()       ← models/memory.py（Korthikanti 公式）      │
│      └── stage_time()            ← compose/stage.py                          │
│               │                                                               │
│               ▼                                                               │
│         PipelineComposer（共享）→ StepResult → Report                        │
│                                                                               │
│  入口：zrt.training.search.estimator.estimate()                               │
│  特点：无需真实模型权重；速度快；用于搜索/扫描                                    │
└───────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────────┐
│  Stack B：图捕获路径（主路径 ✅）                                               │
│                                                                               │
│  HuggingFace 模型 + 硬件 YAML + 训练策略                                      │
│      │                                                                        │
│      ▼                                                                        │
│  run_trace_phases("train_forward", "train_backward")                          │
│    FakeTensorMode + TorchDispatchMode + ModuleTracker                         │
│    → OpGraph[fwd]  +  OpGraph[bwd]                                            │
│      │                                                                        │
│      ▼                                                                        │
│  stitch_fwd_bwd()  ✅ 已实现（ir/adapter.py）                                 │
│    → 统一 OpGraph（fwd+bwd 节点，跨图依赖边，激活生命周期可计算）                │
│      │                                                                        │
│      ▼                                                                        │
│  变换流水线（TransformPipeline）                                               │
│    split → fuse → optim → analyze                                             │
│    FlopsPass / RooflinePass / CommLatencyPass                                 │
│    TrainingFlopsPass / TrainingMemoryPass                                     │
│      │                                                                        │
│      ▼                                                                        │
│  TrainingPipelinePass                                                         │
│    DAGScheduler（每 PP 阶段）→ Timeline → StageTime                           │
│               │                                                               │
│               ▼                                                               │
│         PipelineComposer（共享）→ StepResult → TrainingReport                 │
│                                                                               │
│  入口：estimate_training_from_graphs()（transform/analysis/modeller.py）      │
│  特点：真实算子序列；精确张量形状；精确内存生命周期                               │
└───────────────────────────────────────────────────────────────────────────────┘

                    ▲ 两条路径共享的组件 ▲
                    PipelineComposer 及五个具体实现
                    （OneF1B / Interleaved / DualPipe / DualPipeV / ZeroBubble）
                    位于：python/zrt/training/compose/schedules.py
```

---

## 核心设计原则（更新）

**Stack B 是主路径。Stack A 是快速估算回退。**

- Stack B（图捕获）携带真实张量形状、真实算子序列、真实内存生命周期，是所有并行化建模的正确基础。
- Stack A（规格驱动）用于无需完整追踪时的快速分析：搜索空间扫描、初步可行性判断、CI 快速锚点校验。
- 两条路径**不应合并 IR**。Stack A 的 `Graph`（层级列表）和 Stack B 的 `OpGraph`（有向数据流图）服务于不同的抽象层次，强行合并会增加复杂度而无收益。
- **收敛点**：两条路径都通过 `PipelineComposer` 类生成 `StepResult`，并最终输出统一的报告类型。

---

## 统一目标

不统一 IR，而统一**接口契约**：

| 统一项 | 当前状态 | 目标状态 |
|--------|---------|---------|
| 输出类型 | Stack A 返回 `Report`；Stack B 返回 `TrainingReport` | 两者均返回 `TrainingReport` |
| 合成 OpGraph | 不存在 | 新增 `OpGraph.from_model_spec()` 作为快速追踪回退 |
| 跨路径类型泄漏 | `TrainingPipelinePass` 用下划线别名导入 Stack A 类型 | 清理别名，保持导入语义清晰 |

---

## 已完成工作（不需重新实现）

| 组件 | 文件 | 状态 |
|------|------|------|
| `stitch_fwd_bwd()` | `python/zrt/ir/adapter.py` | ✅ 已实现 |
| `PipelineComposer` + 五个具体实现 | `python/zrt/training/compose/schedules.py` | ✅ 两条路径共享 |
| `TrainingPipelinePass`（Stack B 调度桥接） | `python/zrt/transform/analysis/training.py` | ✅ 已实现 |
| `estimate_training_from_graphs()` | `python/zrt/transform/analysis/modeller.py` | ✅ Stack B 主入口 |

---

## 结构性障碍（已解决）

~~`run_trace_phases` 输出两个独立的 `OpGraph` 对象，反向图中没有指向前向图的依赖边。~~

**已解决**：`stitch_fwd_bwd(fwd_graph, bwd_graph) → OpGraph` 已在 `ir/adapter.py:613–749` 实现：
- 合并两图节点（反向节点 ID 加 `bwd_` 前缀）
- 通过张量 ID 匹配插入 fwd→bwd 跨图依赖边
- 标注 `node.annotations["phase"] = "fwd" / "bwd"`
- 结果：`metadata["fwd_bwd_stitched"] = True`

---

## 阶段 0 —— 输出类型统一（当前优先级最高）

**目标**：两条路径的调用者无需区分入口，均可使用 `TrainingReport`。

**方案**：
1. 将 `TrainingReport` 移至共享位置 `python/zrt/training/spec/report.py`
2. Stack A 的 `estimator.estimate()` 改为返回 `TrainingReport`（填充可计算的字段子集）
3. Stack B 的 `modeller.py` 更新导入路径

**影响文件**：
- `python/zrt/training/search/estimator.py`（~10 行）
- `python/zrt/training/spec/report.py`（新建或扩展，~30 行）
- `python/zrt/transform/analysis/modeller.py`（~2 行）

---

## 阶段 1 —— 新增 `OpGraph.from_model_spec()` 工厂方法

**目标**：为 Stack A 提供一个"合成 OpGraph"，使其可在有限情况下接入 Stack B 的变换流水线。

**设计**：

```python
# python/zrt/ir/graph.py
@classmethod
def from_model_spec(cls, model: ModelSpec, strategy: Strategy, phase: str = "training") -> "OpGraph":
    """从 ModelSpec 构建合成 OpGraph，节点对应 training.ir.Graph 中的算子。
    
    用途：无真实追踪时的快速估算回退。节点携带层级元数据但无真实张量数据流。
    """
    from zrt.training.ir.builders import build_graph
    training_g = build_graph(model, strategy)
    
    nodes = {}
    for op in training_g.ops:          # op 字段：name, kind, inputs, outputs, meta, layer_id, layer_kind
        nodes[op.name] = OpNode(
            id=op.name,
            op_type=op.kind,
            annotations={"layer_id": op.layer_id, "layer_kind": op.layer_kind},
            meta={**op.meta},
        )
    
    edges = []
    op_names = list(nodes.keys())
    for i in range(len(op_names) - 1):
        curr, nxt = op_names[i], op_names[i + 1]
        if nodes[curr].annotations.get("layer_id") == nodes[nxt].annotations.get("layer_id"):
            edges.append(Edge(src=curr, dst=nxt))
    
    return cls(
        name=f"{model.name}_{phase}",
        nodes=nodes,
        edges=edges,
        metadata={
            "source": "model_spec",
            "model": model.name,
            "strategy": strategy,
            "collectives": {c.name: c for c in training_g.collectives},
        }
    )
```

**影响文件**：`python/zrt/ir/graph.py`（+80 行）

**测试**：`tests/training/test_opgraph_from_spec.py`
- `len(opgraph.nodes) == len(training_g.ops)`
- op 类型逐一匹配
- `metadata["source"] == "model_spec"`

---

## 阶段 2 —— 清理跨路径类型泄漏

**当前状态**：`TrainingPipelinePass` 用下划线别名导入 Stack A 类型：

```python
from python.zrt.training.compose.stage import StageTime as _StageTime
from python.zrt.training.compose.schedules import PP_SCHED_BY_NAME, COMPOSER_BY_SCHED
from python.zrt.training.spec.strategy import Strategy as _Strategy, OptKind
```

**处理方式**：
- `COMPOSER_BY_SCHED` 导入合理，无需修改（共享组件）
- `_StageTime` 和 `_Strategy` 的下划线别名无实际意义，直接去掉别名前缀
- 不需要抽象为 Protocol —— 导入本身是合理的，仅是命名风格问题

**影响文件**：`python/zrt/transform/analysis/training.py`（~5 行）

---

## 明确不做的事项

| 原计划项 | 放弃原因 |
|---------|---------|
| 将 `TrainingGraph` 改为包装 `OpGraph` | 反转依赖方向；为层级公式强加节点级遍历，复杂度净增 |
| 将 `flops.py` / `comm.py` / `memory.py` 改为接受 `OpGraph` | 这些模型服务于 Stack A 的解析公式；切换到节点级遍历会丢失清晰的分析结构 |
| 修改 `PipelineComposer` 签名使其接受 `graph: OpGraph` | Composer 已被两条路径共享且工作正常；强加图依赖会破坏 Stack A 的独立性 |
| 删除 `training.ir.training_graph.Graph` | Stack A 的 `Graph` 对规格驱动估算而言是正确的抽象，保留它 |

---

## 关键文件（更新后）

| 文件 | 作用 | 路径所属 |
|------|------|---------|
| `python/zrt/training/ir/training_graph.py` | Stack A 的 `Graph` + `Op` + `Collective` | Stack A |
| `python/zrt/training/ir/builders.py` | `build_graph(ModelSpec, Strategy) → Graph` | Stack A |
| `python/zrt/training/models/flops.py` | 层级 FLOPs 公式 | Stack A |
| `python/zrt/training/models/comm.py` | α-β 集合通信模型 | Stack A |
| `python/zrt/training/models/memory.py` | Korthikanti 内存公式 | Stack A |
| `python/zrt/training/compose/schedules.py` | `PipelineComposer` + 五个实现（**两路共享**） | 共享 |
| `python/zrt/training/compose/stage.py` | `stage_time()` + `StageTime`（**两路共享**） | 共享 |
| `python/zrt/training/search/estimator.py` | Stack A 入口 → 目标返回 `TrainingReport` | Stack A |
| `python/zrt/ir/graph.py` | `OpGraph` + 新增 `from_model_spec()` | Stack B / 共享 |
| `python/zrt/ir/adapter.py` | `stitch_fwd_bwd()`（已实现） | Stack B |
| `python/zrt/transform/analysis/training.py` | `TrainingPipelinePass`（调度桥接） | Stack B |
| `python/zrt/transform/analysis/modeller.py` | Stack B 入口 `estimate_training_from_graphs()` | Stack B |

---

## 验证策略

```bash
# 阶段 0：输出类型统一
PYTHONPATH=python pytest tests/training/ -v -k "estimator or report" 2>&1 | tail -n 20

# 阶段 1：合成 OpGraph 工厂
PYTHONPATH=python pytest tests/training/test_opgraph_from_spec.py -v

# 全量回归：所有训练测试通过
PYTHONPATH=python pytest tests/training/ -v 2>&1 | tail -n 30

# 锚点回归：MFU 不漂移
PYTHONPATH=python pytest tests/training/anchors/test_anchors.py -v
```

---

## 成功标准

1. `estimator.estimate()` 和 `estimate_training_from_graphs()` 均返回 `TrainingReport`
2. `OpGraph.from_model_spec(model, strategy)` 产生节点数和类型与 `build_graph()` 一致的 OpGraph
3. 所有现有训练测试通过（无回归）
4. Stack A 和 Stack B 保持独立执行路径 —— 互不强依赖对方的运行时
