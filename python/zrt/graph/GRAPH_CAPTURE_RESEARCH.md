# Graph Capture Approach Research

本文调研了除当前 `TorchDispatchMode` eager tracing 之外的替代算子图捕获策略，目标是获取更简洁、更具 hardware-representative 的 operator graphs，用于 GPU/NPU 性能建模。

**状态**: 研究完成。代码尚未迁移。参见 [Next Steps](#next-steps)。

---

## Background

目前的流水线通过 `RecordingDispatch`（`TorchDispatchMode` 的子类）捕获 **eager execution** 期间发出的每个 **aten op**。生成的扁平化算子列表随后在 `graph_builder.py` 中组装成 `OpGraph`。

**当前方案的已知痛点：**

- **Layout noise**: `view / _unsafe_view / unsqueeze / transpose / slice` 占所有记录算子的 ~50%，且不携带计算语义。
- **GEMM fragmentation**: `mm` 和 `addmm` 显示为不同的算子类型，即使它们都映射到同一个 linear projection。
- **No explicit data-flow edges**: 边是在 `graph_builder.py` 中后验推断（inferred）的，而不是直接从 **graph IR** 中读取。
- **Fusion 需要在嘈杂的序列上进行模式匹配**: `fusion.py` 必须在冗长的列表中识别 **attention**、**norm** 和 **activation** 模式，而这些 node 之间夹杂着数十个 **reshape ops**。

---

## Probe Scripts

所有脚本均位于 `scripts/` 目录下。它们是 **read-only research tools**，不会影响生产流水线。

| Script | Purpose |
| :--- | :--- |
| `scripts/probe_inductor_fusion.py` | 通过对 `Scheduler._init` 进行 **monkey-patch** 捕获 **CPU inductor Scheduler** 节点 |
| `scripts/probe_predispatch_vs_eager.py` | 三方对比：**eager** vs `pre_dispatch=True` vs `aten_graph=True` |

输出的 JSON 文件保存在 `scripts/output/`。

---

## Approach A — Current: TorchDispatchMode Eager

```
model(fake_input)
  └─ RecordingDispatch (TorchDispatchMode)
       └─ 记录每一个 aten op，每次 dispatch 调用对应一个条目
```

**结果 (DeepSeek-V3, 2 layers, seq=16):**

| Metric | Value |
| :--- | :--- |
| Total ops | 334 |
| Unique op types | 35 |
| Layout ops (view/reshape/…) | 120 (36%) |
| GEMM representation | `mm×17`, 无 `addmm` (V3 中没有 bias) |
| Attention | `bmm×4` (源于手动 matmul 展开) |
| Norm | `mean×9 + rsqrt×9` (源于自定义 RMSNorm module 展开) |

---

## Approach B — CPU Inductor Scheduler Capture

```
torch.compile(model, backend="inductor")
  └─ dynamo → AOT autograd → inductor GraphLowering
       └─ Scheduler._init monkey-patch 在 fuse_nodes() 之后捕获 self.nodes
```

`Scheduler._init` 在调用 `codegen()` 之前完成（包括 `fuse_nodes()`）。在没有 MSVC 的 Windows 上，虽然 `codegen()` 会失败，但此时 **schedule** 已经捕获完成。

**结果 (Qwen2.5-7B, 2 layers, seq=16 — 214 eager ops → 52 kernel nodes):**

| Kernel type | Count | Description |
| :--- | :--- | :--- |
| `ExternKernelSchedulerNode` | 20 | BLAS 调用: `mm×9, addmm×6, bmm×5` |
| `SchedulerNode` | 19 | 单一 compute node |
| `OuterLoopFusedSchedulerNode` | 9 | CPU **outer-loop fusion** 组 |
| `Nop` | 4 | 已消除的节点 |

**压缩率: 214 eager ops → 52 kernel nodes (4.1×)**

**Layout ops** (`view, expand, …`) 被 **completely eliminated** —— **inductor** 将它们视为零成本的 **storage reshapes**。

### CPU vs GPU/NPU Divergence

| Pattern | CPU inductor | GPU (Triton/cuBLAS) | NPU (CANN) |
| :--- | :--- | :--- | :--- |
| **GEMM** | `ExternKernel(mm/addmm/bmm)` | `ExternKernel(mm)` + epilogue fusion | Atlas CUBE matmul |
| **Pointwise fusion** | `OuterLoopFusedSchedulerNode` (row-parallel) | `FusedSchedulerNode` (warp-level) | AscendCL elementwise |
| **Attention** | 3 个独立节点 (QK^T + softmax + V) | **FlashAttention** kernel | `FlashAttentionScore` |

**关键差距 (Critical gap)**: **CPU inductor** 使用 `OuterLoopFusedSchedulerNode` (outer-loop parallelism) 而不是 GPU 的 `FusedSchedulerNode` (warp-level pointwise fusion)。这是不同的 **fusion strategies** —— CPU schedule 不能直接用于推断 GPU kernel 的边界。

---

## Approach D — dynamo.export(pre_dispatch=True)  ← Recommended Direction

```
torch._dynamo.export(model, aten_graph=True, pre_dispatch=True)(inputs)
  └─ 在进行 per-backend dispatch 决策之前捕获图
       → 带有 explicit data-flow edges 的 FX GraphModule
```

### 相比 Eager 的关键改进

**1. GEMM 统一为 `aten.linear.default`**
`mm` 和 `addmm`（带/不带 bias 的 linear）都统一为 `aten.linear.default`。在 `graph_builder.py` / `fusion.py` 中，这消除了区分两者并单独处理 **bias-add node** 的需求。

**2. Layout noise 减少 ~45–51%**
`view / _unsafe_view` 的数量显著下降。剩余的 **layout ops** (`slice`, `unsqueeze`, `transpose`) 带有真正的语义（如 RoPE head-splitting 等），应当保留。

**3. 带有显式边的 FX Graph**
返回的 `GraphModule` 具有规范的 `torch.fx.Graph`。**Explicit producer–consumer edges** 可以直接从 `node.args` 读取，取代现有的边推断逻辑。

**4. 保留了 Model-specific 高层算子**
`aten.softmax.int` 和 `aten.silu.default` 被保留而没有被过度分解（over-decomposed），这有利于识别 **Attention** 和 **activation functions**。

### pre_dispatch 无法解决的问题

- **Attention**: Qwen 和 DeepSeek-V3 使用显式的 `matmul` 调用，而不是 `F.scaled_dot_product_attention`。`pre_dispatch=True` 仅对原生调用 **SDPA** 的模型有效。
- **RMSNorm**: 实现为包含 `mean / pow / rsqrt` 的自定义 `nn.Module`，在 **dynamo** 看到它之前就已经展开。**Pattern matching** 仍然是必要的。

---

## Summary Comparison

| Criterion | A. Eager (current) | B. CPU Inductor | D. pre_dispatch FX |
| :--- | :--- | :--- | :--- |
| **Layout noise** | High (36–51%) | Eliminated | **Medium (21–31%)** |
| **GEMM representation** | mm + addmm split | ExternKernel | **linear (unified)** |
| **Explicit edges** | No (inferred) | No | **Yes (FX Graph)** |
| **Hardware portability** | Neutral | **CPU-specific** | Neutral / Portable |
| **Fusion Pass work** | High (noise-heavy) | 仍需要处理 Attention | **Medium (less noise)** |
| **Requires real weights** | No (FakeTensor) | **Yes** | Yes (Zero-init ok) |

---

## Next Steps

### Track 1: 迁移捕获流程至 pre_dispatch FX Graph (中等工作量)
将 `RecordingDispatch` 替换为 `dynamo.export(pre_dispatch=True)`。
- 迭代 `node.args` 以获取 **explicit edges**。
- 将 **FX node** 映射到 `OpNode`（保持字段兼容）。
- 收益: 简化 `graph_builder.py`，降低 **Fusion Pass** 在 layout ops 上的误匹配率。

### Track 2: 在 pre_dispatch 图上进行 Target-aware Fusion Pass (中等工作量)
在更干净的 `pre_dispatch` 图上编写显式的 **fusion rules**:
- `matmul + scale + softmax + matmul → FlashAttention`
- `mean + pow + rsqrt + mul → RMSNorm`
由于噪声减少，模式匹配将更加健壮。

### Track 3: Model-side 对齐 (低工作量)
对于新模型：使用 `F.scaled_dot_product_attention` 和 `torch.nn.functional.rms_norm`。通过 `pre_dispatch` 捕获，它们将显示为单一算子，从而消除 **pattern matching** 的需求。

---

## Appendix: Technical Notes

### CPU Inductor Scheduler 捕获机制

```python
from torch._inductor.scheduler import Scheduler

# 保存原始 init
original_init = Scheduler._init

def patched_init(self, nodes):
    # original_init 会完成 fuse_nodes()
    original_init(self, nodes)
    # 在 codegen() 之前在此处捕获
    for snode in self.nodes:
        # 记录 snode 信息
        pass

# 应用 monkey-patch
Scheduler._init = patched_init
try:
    torch.compile(model, backend="inductor")(inputs)
except Exception:
    # 即使 Windows 上的 C++ codegen 失败，捕获也已经完成
    pass
finally:
    # 还原环境
    Scheduler._init = original_init
```